#!/usr/bin/env python3
import os
import sys
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timezone

FOUR_HOURS_MS = 4 * 60 * 60 * 1000

# ---- Ichimoku & MACD defaults from your Pine ----
TENKAN_LEN = 7
KIJUN_LEN = 211
SENKOU_LEN = 120
DISPLACEMENT = 41  # used only for plotting in Pine; conditions use unshifted spans
EMA_LEN = 172
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    reason_exit: Optional[str] = None


# -----------------------------
# DB loader
# -----------------------------
def load_price_json(dsn: str, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
    con = psycopg2.connect(dsn)
    try:
        with con.cursor() as cur:
            cur.execute(
                "SELECT price_json FROM price WHERE ticker=%s AND timeframe=%s",
                (ticker, timeframe),
            )
            row = cur.fetchone()
            if not row or row[0] is None:
                raise RuntimeError(f"No data found in price for ({ticker}, {timeframe}).")
            return row[0]
    finally:
        con.close()

def json_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise RuntimeError("Empty price_json.")
    df = pd.DataFrame(rows)
    # expected keys: ts (ms), open, high, low, close, volume
    need = {"ts","open","high","low","close","volume"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise RuntimeError(f"price_json missing keys: {missing}")
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    # de-dup + sort
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df = df.set_index("ts")
    return df


# -----------------------------
# Indicators (Pine-equivalent)
# -----------------------------
def rolling_midpoint(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    hh = high.rolling(length, min_periods=length).max()
    ll = low.rolling(length, min_periods=length).min()
    return (hh + ll) / 2.0

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    prev = (a.shift(1) <= b.shift(1))
    now  = (a > b)
    return (prev & now).fillna(False)

def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    prev = (a.shift(1) >= b.shift(1))
    now  = (a < b)
    return (prev & now).fillna(False)


# -----------------------------
# Backtest Engine
# -----------------------------
def run_backtest(df: pd.DataFrame,
                 start_date: str = "2022-01-01",
                 end_date: str = "2069-01-01",
                 commission_pct: float = 0.1,    # percent per side
                 slippage_bps: float = 3.0       # basis points per side
                 ) -> Dict[str, Any]:
    # Indicators (match Pine)
    tenkan = rolling_midpoint(df["high"], df["low"], TENKAN_LEN)
    kijun  = rolling_midpoint(df["high"], df["low"], KIJUN_LEN)
    senkouA = (tenkan + kijun) / 2.0
    senkouB = rolling_midpoint(df["high"], df["low"], SENKOU_LEN)
    ema_p = ema(df["close"], EMA_LEN)

    macd_line, signal_line, _ = macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    # Range
    isInRange = df.index.to_series().between(
        pd.Timestamp(start_date, tz="UTC"),
        pd.Timestamp(end_date, tz="UTC"),
        inclusive="both",
    )


    # Conditions (unshifted spans in Pine conditions)
    priceAboveCloud = (df["close"] > senkouA) & (df["close"] > senkouB)
    priceBelowCloud = (df["close"] < senkouA) & (df["close"] < senkouB)
    tenkanAboveKijun = (tenkan > kijun)
    tenkanBelowKijun = (tenkan < kijun)

    # Your Pine buy clause uses the lastExitMacd flag logic; keep it.
    lastExitMacd = False

    commission = commission_pct / 100.0
    slip = slippage_bps / 10000.0

    equity = 10_000.0
    position = 0
    entry_price = None
    entry_time = None
    trades: List[Trade] = []
    equity_curve = []

    for i in range(len(df)):
        ts = df.index[i]
        o, h, l, c = df.iloc[i][["open","high","low","close"]]

        # mark-to-market at close
        if position == 1 and entry_price is not None:
            eq = equity * (1 + (c - entry_price) / entry_price)
            equity_curve.append((ts, eq))
        else:
            equity_curve.append((ts, equity))

        if i == len(df) - 1:
            break  # no next bar to trade

        in_range = bool(isInRange.iloc[i])

        # matches Pine precedence:
        # (crossover(macd, signal) and macd >= 50) or (macd <= -50)
        last_exit_reset_ok = (not lastExitMacd) or (
            (bool(crossover(macd_line, signal_line).iloc[i]) and (macd_line.iloc[i] >= 50))
            or (macd_line.iloc[i] <= -50)
        )

        buyCondition = (
            in_range
            and bool(tenkanAboveKijun.iloc[i])
            and bool(priceAboveCloud.iloc[i])
            and (c > float(ema_p.iloc[i]) if not math.isnan(ema_p.iloc[i]) else False)
            and (position == 0)
            and last_exit_reset_ok
        )


        ichimokuExit = (
            (bool(tenkanBelowKijun.iloc[i]) or bool(priceBelowCloud.iloc[i]))
            and (c < float(ema_p.iloc[i]) if not math.isnan(ema_p.iloc[i]) else False)
        )
        macdExit = (macd_line.iloc[i] > 100) and bool(crossunder(macd_line, signal_line).iloc[i])
        sellCondition = in_range and (ichimokuExit or macdExit)

        next_o = float(df.iloc[i+1]["open"])

        if buyCondition and position == 0:
            fill = next_o * (1 + slip)
            cost = equity
            fee = cost * commission
            equity -= fee
            entry_price = fill
            entry_time = df.index[i+1]
            position = 1
            trades.append(Trade(entry_time=entry_time, entry_price=entry_price))
        elif sellCondition and position == 1:
            fill = next_o * (1 - slip)
            ret = (fill - entry_price) / entry_price
            gross = equity
            pnl = gross * ret
            fee = (gross + pnl) * commission
            equity = equity + pnl - fee
            t = trades[-1]
            t.exit_time = df.index[i+1]
            t.exit_price = fill
            t.pnl_pct = ret * 100.0
            t.reason_exit = "macdExit" if macdExit else "ichimokuExit"
            lastExitMacd = macdExit
            position = 0
            entry_price = None
            entry_time = None

    # close open position at last close for reporting
    if position == 1 and entry_price is not None:
        last_close = float(df["close"].iloc[-1])
        ret = (last_close - entry_price) / entry_price
        gross = equity
        pnl = gross * ret
        fee = (gross + pnl) * commission
        equity = equity + pnl - fee
        t = trades[-1]
        t.exit_time = df.index[-1]
        t.exit_price = last_close
        t.pnl_pct = ret * 100.0
        t.reason_exit = "eod"

    # stats
    buys = sum(1 for t in trades if t.entry_time is not None)
    sells = sum(1 for t in trades if t.exit_time is not None)
    closed = [t for t in trades if t.exit_time is not None]
    wins = sum(1 for t in closed if (t.exit_price - t.entry_price) > 0)
    losses = sells - wins
    net_profit_pct = (equity / 10_000.0 - 1.0) * 100.0

    # equity curve + max DD
    eq_series = pd.Series([e for _, e in equity_curve], index=[t for t, _ in equity_curve], dtype=float)
    roll_max = eq_series.cummax()
    dd = (eq_series / roll_max - 1.0).fillna(0.0)
    max_dd_pct = dd.min() * 100.0 if len(dd) else 0.0

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "stats": {
            "buys": buys,
            "sells": sells,
            "wins": wins,
            "losses": losses,
            "net_profit_pct": net_profit_pct,
            "max_drawdown_pct": max_dd_pct,
            "final_equity": equity,
        },
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Backtest Ichimoku+MACD strategy using OHLCV from price table.")
    ap.add_argument("--dsn", default=os.getenv("DATABASE_URL"), help="PostgreSQL DSN, e.g. postgresql://user:pass@localhost:5432/db")
    ap.add_argument("--ticker", default="ETHUSDT", help="Ticker symbol in price.ticker (default ETHUSDT)")
    ap.add_argument("--timeframe", default="4h", help="Timeframe in price.timeframe (default 4h)")
    ap.add_argument("--start", default="2022-01-01", help="Backtest start date YYYY-MM-DD (default 2022-01-01)")
    ap.add_argument("--end",   default="2069-01-01", help="Backtest end date YYYY-MM-DD (default 2069-01-01)")
    ap.add_argument("--commission", type=float, default=0.1, help="Commission percent per side (default 0.1)")
    ap.add_argument("--slippage_bps", type=float, default=3.0, help="Slippage in basis points each side (default 3)")
    ap.add_argument("--save_equity_csv", default="", help="Optional path to save equity curve CSV")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.dsn:
        print("ERROR: --dsn or env DATABASE_URL required", file=sys.stderr)
        sys.exit(2)

    # Load from DB
    raw = load_price_json(args.dsn, args.ticker.upper(), args.timeframe)
    df = json_to_df(raw)

    # Optional check: 4h cadence (warn only)
    if len(df) > 1:
        gaps = (df.index.to_series().diff().dropna() != pd.Timedelta(hours=4)).sum()
        if args.verbose and gaps:
            print(f"[warn] detected {int(gaps)} non-4h step(s) in series (will still backtest).", file=sys.stderr)

    if args.verbose:
        print(f"[data] {args.ticker}/{args.timeframe} bars={len(df)} {df.index.min()} → {df.index.max()}", file=sys.stderr)

    res = run_backtest(
        df,
        start_date=args.start,
        end_date=args.end,
        commission_pct=args.commission,
        slippage_bps=args.slippage_bps,
    )

    s = res["stats"]
    print("\n== Backtest Summary ==")
    print(f"Symbol/TF     : {args.ticker} / {args.timeframe}")
    print(f"Range         : {args.start} → {args.end}")
    print(f"Bars          : {len(df)}")
    print(f"Buys / Sells  : {s['buys']} / {s['sells']}")
    print(f"Wins / Losses : {s['wins']} / {s['losses']}")
    print(f"Net Profit %  : {s['net_profit_pct']:.2f}%")
    print(f"Max Drawdown %: {s['max_drawdown_pct']:.2f}%")
    print(f"Final Equity  : ${s['final_equity']:.2f}")

    if res["trades"]:
        print("\n== Trades (first 10) ==")
        for t in res["trades"][:10]:
            pnl_str = f"{t.pnl_pct:.2f}%" if t.pnl_pct is not None else "NA"
            print(f"{t.entry_time} @ {t.entry_price:.2f}  ->  {t.exit_time} @ {t.exit_price if t.exit_price else None}  "
                  f"pnl={pnl_str}  reason={t.reason_exit}")

    if args.save_equity_csv:
        eq_df = pd.DataFrame(res["equity_curve"], columns=["timestamp","equity"])
        eq_df.to_csv(args.save_equity_csv, index=False)
        if args.verbose:
            print(f"[save] equity curve -> {args.save_equity_csv}", file=sys.stderr)

if __name__ == "__main__":
    main()
