#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import ccxt

# 6 months in milliseconds
TWO_YEARS_MS = int(0.6 * 365 * 24 * 60 * 60 * 1000)


def dedupe_by_ts(candles):
    """
    Remove duplicated candles based on 'ts'.
    Keeps the last occurrence for each ts.
    Returns a new list sorted by ts.
    """
    by_ts = {}
    for c in candles:
        ts = c.get("ts")
        if ts is None:
            continue
        by_ts[ts] = c  # last one wins

    deduped = list(by_ts.values())
    deduped.sort(key=lambda x: x["ts"])
    return deduped


def load_existing_json(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of candles")

    # Ensure each item has a ts
    data = [c for c in data if isinstance(c, dict) and "ts" in c]

    # Sort by ts (rough sanity)
    data.sort(key=lambda x: x["ts"])

    return data


def filter_last_two_years(candles):
    if not candles:
        return candles

    # Base cutoff on latest candle in file
    last_ts = candles[-1]["ts"]
    cutoff_ts = last_ts - TWO_YEARS_MS
    filtered = [c for c in candles if c["ts"] >= cutoff_ts]
    return filtered


def fetch_new_candles_bybit(symbol, timeframe, since_ts):
    """
    Fetch OHLCV candles from Bybit using ccxt.bybit.
    Returns list of dicts with the same format as your JSON.
    """
    ex = ccxt.bybit({"enableRateLimit": True})
    ex.load_markets()

    ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)

    result = []
    for ts, o, h, l, c, v in ohlcv:
        result.append(
            {
                "ts": ts,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        )
    return result


def merge_candles(existing, new):
    """
    Merge existing + new candles, dedup by ts, and return sorted list.
    """
    by_ts = {c["ts"]: c for c in existing}
    for c in new:
        by_ts[c["ts"]] = c  # overwrite if duplicate ts

    merged = list(by_ts.values())
    merged.sort(key=lambda x: x["ts"])
    return merged


def update_json(json_file, symbol, timeframe):
    path = Path(json_file)

    # 1) Read existing JSON
    candles = load_existing_json(path)
    print(f"[INFO] Loaded {len(candles)} candles from {path}")

    # 🔹 Remove duplicates already in the file
    before_dedup = len(candles)
    candles = dedupe_by_ts(candles)
    removed = before_dedup - len(candles)
    if removed > 0:
        print(f"[INFO] Removed {removed} duplicated candles from existing file.")
    else:
        print("[INFO] No duplicated candles found in existing file.")

    # 3) Filter for 2-year data (based on latest ts in file)
    candles = filter_last_two_years(candles)
    print(f"[INFO] After 2-year filter: {len(candles)} candles remain")

    # Get 'since' timestamp for Bybit request
    since_ts = candles[-1]["ts"] if candles else None

    print(f"[INFO] Fetching candles from Bybit for {symbol} {timeframe}, since={since_ts}")
    raw_new_candles = fetch_new_candles_bybit(symbol, timeframe, since_ts)

    # Only keep candles whose ts is NOT already in the file.
    existing_ts = {c["ts"] for c in candles}
    new_unique = [c for c in raw_new_candles if c["ts"] not in existing_ts]

    print(
        f"[INFO] Fetched {len(raw_new_candles)} candles from API, "
        f"{len(new_unique)} are actually new."
    )

    if not new_unique:
        print("[INFO] No new candles to append.")
        updated = candles
    else:
        updated = merge_candles(candles, new_unique)
        print(f"[INFO] After merge: {len(updated)} total candles")

    # 🔹 Final safety: dedupe again in case of any weirdness
    before_final_dedup = len(updated)
    updated = dedupe_by_ts(updated)
    final_removed = before_final_dedup - len(updated)
    if final_removed > 0:
        print(f"[INFO] Removed {final_removed} duplicated candles after merge.")

    # Re-apply 2-year window to final data
    if updated:
        last_ts = updated[-1]["ts"]
        cutoff_ts = last_ts - TWO_YEARS_MS
        updated = [c for c in updated if c["ts"] >= cutoff_ts]
        print(f"[INFO] After final 2-year trim: {len(updated)} candles")

    # Save back in same format
    path.write_text(json.dumps(updated, indent=2))
    print(f"[INFO] Wrote updated candles to {path}")


def main():
    parser = argparse.ArgumentParser(description="Update local OHLCV JSON with Bybit data.")
    parser.add_argument(
        "--json-file",
        required=True,
        help="Path to existing JSON file (with ts/open/high/low/close/volume).",
    )
    parser.add_argument(
        "--symbol",
        default="ETH/USDT:USDT",  # Bybit USDT perpetual futures
        help='Bybit symbol (e.g. "ETH/USDT:USDT" for perp futures, "ETH/USDT" for spot).',
    )
    parser.add_argument(
        "--timeframe",
        default="30m",
        help='Timeframe string (e.g. "30m", "1h", "4h").',
    )
    args = parser.parse_args()

    update_json(args.json_file, args.symbol, args.timeframe)


if __name__ == "__main__":
    main()
