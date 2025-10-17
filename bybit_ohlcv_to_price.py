#!/usr/bin/env python3
import os
import sys
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple

import requests
import psycopg2
from psycopg2.extras import Json

BYBIT_BASE_URL = "https://api.bybit.com"  # public v5
INTERVAL_MAP = {
    "30m": "30",
    "1h":  "60",
    "4h":  "240",
    "1d":  "D"
}

def interval_ms(tf: str) -> int:
    if tf == "30m":
        return 30 * 60 * 1000
    if tf == "1h":
        return 60 * 60 * 1000
    if tf == "4h":
        return 4 * 60 * 60 * 1000
    if tf == "1d":
        return 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported timeframe: {tf}")


@dataclass
class Bar:
    ts: int          # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float

def month_iter(start_dt: datetime, end_dt: datetime):
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
    while cur < end_dt:
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)
        yield cur, min(nxt, end_dt)
        cur = nxt

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

def fetch_bybit_klines(symbol: str, category: str, interval: str,
                       start_ms: int, end_ms: int,
                       timeout: float = 15.0, retries: int = 4, verbose: bool = False) -> List[Bar]:
    """
    GET /v5/market/kline?category=...&symbol=...&interval=...&start=...&end=...&limit=1000
    Returns oldest->newest list of Bar.
    """
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": category,      # "linear" for USDT perpetual; "spot" for spot
        "symbol": symbol,
        "interval": interval,      # "240" for 4h
        "start": str(start_ms),
        "end": str(end_ms),
        "limit": "1000",
    }
    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if verbose:
                print(f"[http] {symbol} {interval} {start_ms}->{end_ms} -> {r.status_code}", file=sys.stderr)
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit retCode={data.get('retCode')} retMsg={data.get('retMsg')}")
            rows = (data.get("result", {}) or {}).get("list", []) or []
            rows.reverse()  # Bybit returns newest-first
            out: List[Bar] = []
            for row in rows:
                # [startTime(ms), open, high, low, close, volume, turnover]
                ts = int(row[0])
                o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4]); v = float(row[5])
                out.append(Bar(ts, o, h, l, c, v))
            return out
        except Exception as e:
            if attempt >= retries:
                raise
            sleep = min(2.0 * attempt, 6.0)
            if verbose:
                print(f"[warn] attempt {attempt} failed: {e}; retrying in {sleep:.1f}s", file=sys.stderr)
            time.sleep(sleep)

def dedupe_merge(acc: Dict[int, Bar], chunk: List[Bar]) -> int:
    """Merge new bars into dict by ts; overwrite on duplicate ts. Returns number of new inserts (not overwrites)."""
    added = 0
    for b in chunk:
        if b.ts not in acc:
            added += 1
        acc[b.ts] = b
    return added

def bars_to_json(bars: List[Bar]) -> List[Dict[str, Any]]:
    return [{"ts": b.ts, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume} for b in bars]

def ensure_columns(conn, verbose=False):
    with conn, conn.cursor() as cur:
        cur.execute("""
        DO $$
        BEGIN
          IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='price' AND column_name='ticker') THEN
            ALTER TABLE price ADD COLUMN ticker TEXT;
          END IF;
          IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='price' AND column_name='timeframe') THEN
            ALTER TABLE price ADD COLUMN timeframe TEXT;
          END IF;
          IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='price' AND column_name='price_json') THEN
            ALTER TABLE price ADD COLUMN price_json JSONB;
          END IF;
        END $$;
        """)
        if verbose:
            print("[db] ensured columns ticker, timeframe, price_json", file=sys.stderr)

def save_price(conn, ticker: str, timeframe: str, payload: List[Dict[str, Any]]):
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM price WHERE ticker=%s AND timeframe=%s", (ticker, timeframe))
        cur.execute(
            "INSERT INTO price (ticker, timeframe, price_json) VALUES (%s, %s, %s)",
            (ticker, timeframe, Json(payload))
        )

def load_price(conn, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT price_json FROM price WHERE ticker=%s AND timeframe=%s", (ticker, timeframe))
        row = cur.fetchone()
        return row[0] if row else []

def normalize_and_report_dupes(bars_raw: List[Dict[str, Any]]) -> Tuple[List[Bar], int]:
    """Convert to Bar, remove dups, return sorted bars + dup_count_removed."""
    seen: Dict[int, Bar] = {}
    dupes = 0
    for r in bars_raw:
        ts = int(r["ts"])
        b = Bar(ts, float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]))
        if ts in seen:
            dupes += 1
        seen[ts] = b
    ordered = [seen[k] for k in sorted(seen.keys())]
    return ordered, dupes

def find_gaps(ts_sorted: List[int], step_ms: int) -> List[Tuple[int, int]]:
    """
    Given a sorted list of timestamps, return a list of (missing_start, missing_end) inclusive ranges
    that should exist at 4h spacing but are absent.
    """
    gaps: List[Tuple[int, int]] = []
    if not ts_sorted:
        return gaps
    for i in range(1, len(ts_sorted)):
        prev_ts = ts_sorted[i-1]
        cur_ts  = ts_sorted[i]
        expected_next = prev_ts + step_ms
        if cur_ts == expected_next:
            continue
        # We have a gap between prev_ts and cur_ts. Build the missing range:
        # missing bars are: expected_next, expected_next+step, ..., cur_ts-step
        miss_start = expected_next
        miss_end   = cur_ts - step_ms
        if miss_start <= miss_end:
            gaps.append((miss_start, miss_end))
    return gaps

def coalesce_missing_ranges(missing_list: List[Tuple[int, int]], step_ms: int) -> List[Tuple[int, int]]:
    """
    Merge adjacent missing ranges into bigger fetch windows to minimize API calls.
    """
    if not missing_list:
        return []
    missing_list.sort()
    merged = []
    cur_start, cur_end = missing_list[0]
    for s, e in missing_list[1:]:
        # If the next range starts exactly where the current ends + step, merge it
        if s == cur_end + step_ms:
            cur_end = e
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged

def repair_missing_ranges(symbol: str, category: str, interval: str,
                          gaps: List[Tuple[int, int]], timeout: float, retries: int, verbose: bool) -> Dict[int, Bar]:
    fixed: Dict[int, Bar] = {}
    for (start_ms, end_ms) in gaps:
        # Expand window by one step on each side to make sure Bybit returns boundary bars
        pad = step_ms
        s = start_ms - pad
        e = end_ms + pad
        if verbose:
            sdt, edt = ms_to_dt(s), ms_to_dt(e)
            print(f"[gap] refetch {symbol} {sdt} → {edt}", file=sys.stderr)
        chunk = fetch_bybit_klines(symbol, category, interval, s, e, timeout=timeout, retries=retries, verbose=verbose)
        for b in chunk:
            fixed[b.ts] = b
        # polite pause
        time.sleep(0.15)
    return fixed

def main():
    ap = argparse.ArgumentParser(description="Fetch 4h OHLCV from Bybit month-by-month; store in price; then validate & repair duplicates/gaps.")
    ap.add_argument("--dsn", default=os.getenv("DATABASE_URL"), help="PostgreSQL DSN, e.g. postgresql://user:pass@localhost:5432/db")
    ap.add_argument("--symbol", default=os.getenv("SYMBOL", "BTCUSDT"), help="Bybit symbol, e.g. BTCUSDT")
    ap.add_argument("--category", default=os.getenv("BYBIT_CATEGORY", "linear"), help="Bybit category: linear | inverse | spot")
    ap.add_argument("--start", default=os.getenv("START_DATE", "2023-01-01"), help="Start date YYYY-MM-DD (default 2023-01-01)")
    ap.add_argument("--timeframe", default=None, choices=["30m","1h","4h","1d"], help="Timeframe to store in DB (defaults to --interval)")
    ap.add_argument("--interval", default=os.getenv("INTERVAL", "4h"), choices=["30m","1h","4h","1d"], help="Bybit kline interval / human timeframe")
    ap.add_argument("--sleep", type=float, default=0.20, help="Sleep seconds between month requests")
    ap.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout")
    ap.add_argument("--retries", type=int, default=4, help="HTTP retries")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-repair", action="store_true", help="Only fetch & store; skip duplicate/gap repair step")
    args = ap.parse_args()

    if not args.dsn:
        print("ERROR: --dsn or env DATABASE_URL required", file=sys.stderr)
        sys.exit(2)

    tf = ((args.timeframe or args.interval) or args.interval)
    interval = INTERVAL_MAP[tf]
    step_ms = interval_ms(tf)
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print("ERROR: --start must be YYYY-MM-DD", file=sys.stderr)
        sys.exit(2)
    end_dt = datetime.now(timezone.utc)

    conn = psycopg2.connect(args.dsn)
    ensure_columns(conn, verbose=args.verbose)

    if args.verbose:
        print(f"[run] {args.symbol} {(args.timeframe or args.interval)} from {start_dt.date()} to {end_dt.date()} (monthly)", file=sys.stderr)

    # -------- Fetch month-by-month ----------
    acc: Dict[int, Bar] = {}
    months = list(month_iter(start_dt, end_dt))
    for i, (m_start, m_end) in enumerate(months, 1):
        if args.verbose:
            print(f"[month] {i}/{len(months)} {m_start.date()} → {m_end.date()}", file=sys.stderr)
        chunk = fetch_bybit_klines(
            symbol=args.symbol,
            category=args.category,
            interval=interval,
            start_ms=to_ms(m_start),
            end_ms=to_ms(m_end) - 1,   # inclusive
            timeout=args.timeout,
            retries=args.retries,
            verbose=args.verbose,
        )
        dedupe_merge(acc, chunk)
        if args.verbose:
            print(f"[month] got {len(chunk)} bars; total unique {len(acc)}", file=sys.stderr)
        time.sleep(args.sleep)

    bars = [acc[k] for k in sorted(acc.keys())]
    payload = bars_to_json(bars)

    # Save fetched payload under desired timeframe
    save_price(conn, ticker=args.symbol.upper(), timeframe=((args.timeframe or args.interval) or args.interval), payload=payload)
    
    if args.verbose and bars:
        first = ms_to_dt(bars[0].ts)
        last  = ms_to_dt(bars[-1].ts)
        print(f"[done/fetch] bars={len(bars)} {first} → {last}", file=sys.stderr)

    # -------- Save initial snapshot ----------
    save_price(conn, ticker=args.symbol.upper(), timeframe=(args.timeframe or args.interval), payload=payload)
    if args.verbose:
        print("[db] initial save done", file=sys.stderr)

    if args.skip_repair:
        if args.verbose:
            print("[skip] repair phase skipped", file=sys.stderr)
        return

    # -------- Validate & Repair ----------
    stored_raw = load_price(conn, ticker=args.symbol.upper(), timeframe=(args.timeframe or args.interval))
    stored_bars, dupes_removed = normalize_and_report_dupes(stored_raw)
    ts_sorted = [b.ts for b in stored_bars]

    # Detect gaps
    missing_ranges = find_gaps(ts_sorted, step_ms)
    coalesced = coalesce_missing_ranges(missing_ranges, step_ms)

    if args.verbose:
        print(f"[check] duplicates: {dupes_removed}, gaps: {len(missing_ranges)} (coalesced {len(coalesced)})", file=sys.stderr)

    changed = False

    # If duplicates existed, we already removed via normalize; mark changed
    if dupes_removed > 0:
        changed = True

    # Try to fill gaps from Bybit
    if coalesced:
        fetched_fix = repair_missing_ranges(
            symbol=args.symbol.upper(),
            category=args.category,
            interval=interval,
            gaps=coalesced,
            timeout=args.timeout,
            retries=args.retries,
            verbose=args.verbose,
        )
        if fetched_fix:
            # merge fixes
            merged: Dict[int, Bar] = {b.ts: b for b in stored_bars}
            newly_added = 0
            for ts, bar in fetched_fix.items():
                # Only add bars that align to 4h grid and are inside any missing gap
                if ts % step_ms == 0 and any(start <= ts <= end for (start, end) in missing_ranges):
                    if ts not in merged:
                        newly_added += 1
                    merged[ts] = bar
            if args.verbose:
                print(f"[repair] added {newly_added} bars from Bybit for gaps", file=sys.stderr)
            if newly_added > 0:
                changed = True
            # rebuild ordered list
            stored_bars = [merged[k] for k in sorted(merged.keys())]

    # If anything changed, write back
    if changed:
        new_payload = bars_to_json(stored_bars)
        save_price(conn, ticker=args.symbol.upper(), timeframe=((args.timeframe or args.interval) or args.interval), payload=new_payload)
        if args.verbose:
            first = ms_to_dt(stored_bars[0].ts)
            last  = ms_to_dt(stored_bars[-1].ts)
            total = len(stored_bars)
            print(f"[db] repaired save done | bars={total} {first} → {last}", file=sys.stderr)
    else:
        if args.verbose:
            print("[ok] no duplicates or gaps to fix", file=sys.stderr)

if __name__ == "__main__":
    main()
