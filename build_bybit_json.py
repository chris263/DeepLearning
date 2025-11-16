#!/usr/bin/env python3
"""
Fetch OHLCV candles from Bybit via ccxt and save them to a JSON file.

Usage example (2 years of 30m ETH/USDC perpetual):

  python3 build_bybit_json.py \
    --symbol ETH/USDC:USDC \
    --timeframe 30m \
    --years 2 \
    --outfile /home/production/tmp/ETHUSDC_30m_2y.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import ccxt


def log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{now}] {msg}", flush=True)


def fetch_ohlcv_series(
    exchange: ccxt.bybit,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    *,
    limit: int = 1000,
    category: str = "linear",
) -> list:
    """
    Fetch OHLCV from Bybit in a loop from since_ms up to until_ms (both in ms).

    Returns a list of [ts, open, high, low, close, volume].
    Deduplication / sorting is done later.
    """
    all_rows = []
    ms_per_bar = exchange.parse_timeframe(timeframe) * 1000

    cursor = int(since_ms)
    params = {"category": category} if category else {}

    log(
        f"Starting OHLCV fetch: symbol={symbol}, timeframe={timeframe}, "
        f"since={since_ms}, until={until_ms}, category={category}"
    )

    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=cursor,
                limit=limit,
                params=params,
            )
        except Exception as e:
            log(f"[ERROR] fetch_ohlcv failed at since={cursor}: {e}")
            # Basic retry logic
            time.sleep(5)
            continue

        if not batch:
            log("[INFO] No more candles returned by exchange; stopping.")
            break

        all_rows.extend(batch)

        last_ts = batch[-1][0]
        log(
            f"Fetched {len(batch)} candles; "
            f"ts range: {batch[0][0]} .. {last_ts} "
            f"({datetime.fromtimestamp(batch[0][0]/1000, tz=timezone.utc)} .. "
            f"{datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)})"
        )

        # Move cursor forward by one bar to avoid re-fetching same candle
        cursor = last_ts + ms_per_bar

        # Safety stop
        if last_ts >= until_ms - ms_per_bar:
            break

        # Respect rate limit
        time.sleep(exchange.rateLimit / 1000.0)

    log(f"Total raw candles fetched (with possible duplicates): {len(all_rows)}")
    return all_rows


def dedup_and_sort_ohlcv(rows: list) -> list:
    """
    Deduplicate rows by timestamp and sort by ts ascending.

    Each row is [ts, open, high, low, close, volume, ...].
    """
    by_ts = {}
    for row in rows:
        if not row:
            continue
        ts = int(row[0])
        # later occurrences overwrite earlier ones, but all come from same source
        by_ts[ts] = row

    deduped = list(by_ts.values())
    deduped.sort(key=lambda r: r[0])
    log(
        f"After deduplication: {len(deduped)} candles "
        f"(removed {len(rows) - len(deduped)} duplicates)."
    )
    return deduped


def ohlcv_to_json_struct(rows: list) -> list:
    """
    Convert OHLCV rows into your JSON format:
      {ts, open, high, low, close, volume}
    """
    out = []
    for r in rows:
        # ccxt standard: [timestamp, open, high, low, close, volume]
        ts, o, h, l, c, v = r[:6]
        out.append(
            {
                "ts": int(ts),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        )
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fetch OHLCV from Bybit and save to a JSON file."
    )
    p.add_argument(
        "--symbol",
        required=True,
        help="ccxt symbol, e.g. 'ETH/USDC:USDC' or 'ETH/USDT:USDT'",
    )
    p.add_argument(
        "--timeframe",
        default="30m",
        help="Timeframe string (ccxt style), e.g. 30m, 1h, 4h, 1d (default: 30m)",
    )
    p.add_argument(
        "--years",
        type=float,
        default=None,
        help="How many years back to fetch from now (e.g., 2 for ~2 years).",
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Alternative to --years: how many days back from now.",
    )
    p.add_argument(
        "--since-ms",
        type=int,
        default=None,
        help="Alternative explicit start time in milliseconds since epoch (UTC).",
    )
    p.add_argument(
        "--outfile",
        required=True,
        help="Path to output JSON file, e.g. /home/production/tmp/ETHUSDC_30m_2y.json",
    )
    p.add_argument(
        "--category",
        default="linear",
        help=(
            "Bybit v5 category: 'spot', 'linear', 'inverse', 'option'. "
            "Default: linear (USDT/USDC perpetual)."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max candles per API call (default 1000).",
    )
    return p.parse_args(argv)


def compute_since_ms(args, now_ms: int) -> int:
    """
    Decide the starting timestamp based on args.

    Priority:
      1) --since-ms
      2) --days
      3) --years
      4) default 2 years if nothing provided
    """
    if args.since_ms is not None:
        return int(args.since_ms)

    if args.days is not None:
        span_ms = args.days * 24 * 60 * 60 * 1000
        return now_ms - span_ms

    years = args.years if args.years is not None else 2.0
    span_ms = int(years * 365 * 24 * 60 * 60 * 1000)
    return now_ms - span_ms


def main(argv=None):
    args = parse_args(argv)

    # Prepare exchange (public data only; no API keys required)
    exchange = ccxt.bybit(
        {
            "enableRateLimit": True,
        }
    )

    now_ms = exchange.milliseconds()
    since_ms = compute_since_ms(args, now_ms)
    until_ms = now_ms

    log(
        f"Will fetch {args.symbol} {args.timeframe} from "
        f"{datetime.fromtimestamp(since_ms/1000, tz=timezone.utc)} "
        f"to {datetime.fromtimestamp(until_ms/1000, tz=timezone.utc)} "
        f"(UTC)."
    )

    rows_raw = fetch_ohlcv_series(
        exchange=exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=args.limit,
        category=args.category,
    )

    rows = dedup_and_sort_ohlcv(rows_raw)
    json_data = ohlcv_to_json_struct(rows)

    outpath = args.outfile
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    log(f"Saved {len(json_data)} candles to {outpath}")


if __name__ == "__main__":
    main(sys.argv[1:])
