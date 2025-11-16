#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import ccxt

# 2 years in milliseconds
TWO_YEARS_MS = int(2 * 365 * 24 * 60 * 60 * 1000)


def load_existing_json(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of candles")

    # Ensure each item has a ts
    data = [c for c in data if isinstance(c, dict) and "ts" in c]

    # 2) Assure the right order
    data.sort(key=lambda x: x["ts"])

    return data


def filter_last_two_years(candles):
    if not candles:
        return candles

    # Base cutoff on **latest** candle in file (more robust than wall-clock)
    last_ts = candles[-1]["ts"]
    cutoff_ts = last_ts - TWO_YEARS_MS
    filtered = [c for c in candles if c["ts"] >= cutoff_ts]
    return filtered


def fetch_new_candles_bybit(symbol: str, timeframe: str, since_ts: int | None):
    """
    Fetch new OHLCV candles from Bybit using ccxt.bybit.
    Returns list of dicts with the same format as your JSON.
    """
    ex = ccxt.bybit({"enableRateLimit": True})
    ex.load_markets()

    # symbol examples:
    #   spot:    "ETH/USDT"
    #   futures: "ETH/USDT:USDT"
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


def update_json(json_file: str, symbol: str, timeframe: str):
    path = Path(json_file)

    # 1) Read existing JSON
    candles = load_existing_json(path)
    print(f"[INFO] Loaded {len(candles)} candles from {path}")

    # 3) Filter for 2-year data (based on latest ts in file)
    candles = filter_last_two_years(candles)
    print(f"[INFO] After 2-year filter: {len(candles)} candles remain")

    # Get 'since' timestamp for Bybit request
    since_ts = candles[-1]["ts"] if candles else None

    # 4) Append the new candle(s) from Bybit
    print(f"[INFO] Fetching new candles from Bybit for {symbol} {timeframe}, since={since_ts}")
    new_candles = fetch_new_candles_bybit(symbol, timeframe, since_ts)

    if new_candles:
        print(f"[INFO] Fetched {len(new_candles)} new candles from Bybit")
    else:
        print("[INFO] No new candles from Bybit")

    updated = merge_candles(candles, new_candles)
    print(f"[INFO] After merge: {len(updated)} total candles")

    # (Optional) Re-apply 2-year filter again to ensure final file stays in window
    if updated:
        last_ts = updated[-1]["ts"]
        cutoff_ts = last_ts - TWO_YEARS_MS
        updated = [c for c in updated if c["ts"] >= cutoff_ts]
        print(f"[INFO] After final 2-year trim: {len(updated)} candles")

    # Save in the exact same JSON shape you showed
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
        default="BTC/USDT:USDT",  # Bybit USDT perpetual futures
        help='Bybit symbol (e.g. "BTC/USDT:USDT" for perp futures, "BTC/USDT" for spot).',
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help='Timeframe string (e.g. "30m", "1h", "4h").',
    )
    args = parser.parse_args()

    update_json(args.json_file, args.symbol, args.timeframe)


if __name__ == "__main__":
    main()
