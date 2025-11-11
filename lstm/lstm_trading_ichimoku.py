#!/usr/bin/env python3
"""
LSTM Ichimoku Trader (Classification)

Artifacts written to --out-dir:
 - model.pt               (PyTorch state_dict)
 - meta.json              (provenance & wiring)
 - preprocess.json        (scaler, features, lookback/horizon)
 - postprocess.json       (thresholds & trade params)
 - trades.json            (executed trades over TEST window)
 - summary.json           (win/loss breakdown, net profit, MDD, etc.)

If --bundle-dir is provided (and different), artifacts are also copied there.

Trading simulation:
 - Signals from model probability p = sigmoid(logit)
   * Long  if p > threshold
   * Short if p < 1 - threshold
 - Entry at next bar OPEN with slippage modeled as a time offset:
     fill_price = open + (close - open) * (slippage_sec / bar_seconds)
 - Exit rules:
   * Intrabar SL/TP checked every bar after entry (conservative priority: SL before TP if both hit)
   * Opposite signal => exit at next bar OPEN (+slippage)
   * Final bar => force close at last CLOSE
 - Fees charged per side (entry and exit): fee_pct (default 0.1%) or fee_bps/10000
 - No overlapping positions; optional same-bar SL/TP after entry is allowed.

Example
-------
python lstm_trading_ichimoku.py \
  --db-url "postgresql://postgres:postgres@localhost:5432/sat" \
  --ticker BTCUSDT --timeframe 1h \
  --train-start 2023-01-01 --train-end 2024-12-31 --test-end 2025-11-03 \
  --lookback 64 --horizon 4 --epochs 15 --batch-size 256 --pos-thr 0.55 \
  --sl-pct 0.02 --tp-pct 0.06 --device cpu \
  --out-dir /home/christiano/Documents/DeepLearning/lstm/btc_lstm_long_1h/ \
  --bundle-dir /home/christiano/Documents/DeepLearning/lstm/btc_lstm_long_1h/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import psycopg
except Exception:
    psycopg = None

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ==========================
# Helpers: timeframes, MDD, JSON I/O
# ==========================

def timeframe_to_seconds(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("ms"):
        return max(1, int(tf[:-2])) // 1000
    if tf.endswith("s"):
        return int(tf[:-1])
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    if tf.endswith("w"):
        return int(tf[:-1]) * 7 * 86400
    raise ValueError(f"Unrecognized timeframe: {tf}")

def calc_max_drawdown(equity: np.ndarray) -> Tuple[float, float]:
    """
    Returns (max_drawdown_pct, max_drawdown_abs) where pct is negative (e.g., -0.23 for -23%).
    """
    peak = -np.inf
    mdd = 0.0
    mdd_abs = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v / peak) - 1.0 if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
            mdd_abs = v - peak
    return float(mdd), float(mdd_abs)

def _save_json(path: Path, payload: dict):
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


# ==========================
# Data Loading & Utilities
# ==========================

def _coerce_timestamp_series(ts: pd.Series) -> pd.Series:
    """Coerce mixed timestamps to tz-naive pandas datetime."""
    def _coerce_one(x):
        if pd.isna(x):
            return pd.NaT
        try:
            xv = float(x)
            if xv >= 1e12:
                return pd.to_datetime(int(xv), unit="ms", utc=False)
            elif xv >= 1e9:
                return pd.to_datetime(int(xv), unit="s", utc=False)
        except Exception:
            pass
        return pd.to_datetime(x, utc=False, errors="coerce")

    out = ts.map(_coerce_one)
    if out.isna().any():
        out2 = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(None)
        out = out.fillna(out2)
    return out.dt.tz_localize(None) if getattr(out.dt, "tz", None) is not None else out


def load_db_ohlcv(db_url: str, ticker: str, timeframe: str,
                   start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV from Postgres `price.data_json` (JSON array of bars)."""
    if psycopg is None:
        raise RuntimeError("psycopg (v3) is required for DB mode. Install `psycopg[binary]`.")

    def sql_tpl(time_key: str, col: str) -> str:
        return f"""
        SELECT
            bar->>'{time_key}' AS ts,
            (bar->>'open')::double precision  AS open,
            (bar->>'high')::double precision  AS high,
            (bar->>'low')::double precision   AS low,
            (bar->>'close')::double precision AS close,
            COALESCE((bar->>'volume')::double precision, 0) AS volume
        FROM price p, jsonb_array_elements(p.data_json) AS bar
        WHERE p.{col} = %s AND p.timeframe = %s
        """

    with psycopg.connect(db_url) as conn:
        conn.execute("SET TIME ZONE 'UTC'")
        try:
            cur = conn.execute(sql_tpl("ts", "ticker"), (ticker, timeframe))
        except Exception:
            cur = conn.execute(sql_tpl("ts", "symbol"), (ticker, timeframe))
        rows = cur.fetchall()
        if not rows:
            try:
                cur = conn.execute(sql_tpl("timestamp", "ticker"), (ticker, timeframe))
            except Exception:
                cur = conn.execute(sql_tpl("timestamp", "symbol"), (ticker, timeframe))
            rows = cur.fetchall()

    if not rows:
        raise ValueError("No bars returned — check timeframe/ticker or DB contents.")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = _coerce_timestamp_series(df["timestamp"]).astype("datetime64[ns]")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    if start:
        df = df[df.timestamp >= pd.to_datetime(start)]
    if end:
        df = df[df.timestamp <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError("Bars exist but are outside the requested date window.")

    return df


# ==============
# Ichimoku & Features
# ==============

def rolling_mid(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    n = int(n)
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return (hh + ll) / 2.0


def ichimoku(df: pd.DataFrame, tenkan: int, kijun: int, senkou: int) -> pd.DataFrame:
    d = df.copy()
    d["tenkan"] = rolling_mid(d.high, d.low, tenkan)
    d["kijun"] = rolling_mid(d.high, d.low, kijun)
    d["span_a"] = (d["tenkan"] + d["kijun"]) / 2.0
    d["span_b"] = rolling_mid(d.high, d.low, senkou)
    d["cloud_top"] = d[["span_a", "span_b"]].max(axis=1)
    d["cloud_bot"] = d[["span_a", "span_b"]].min(axis=1)
    return d


def build_features(df: pd.DataFrame, tenkan: int, kijun: int, senkou: int,
                   displacement: int, slope_window: int = 8) -> Tuple[pd.DataFrame, List[str]]:
    d = ichimoku(df, tenkan, kijun, senkou)
    d["px"] = d["close"]
    d["ret1"] = d["close"].pct_change().fillna(0)
    d["oc_diff"] = (d["close"] - d["open"]) / d["open"]
    d["hl_range"] = (d["high"] - d["low"]) / d["px"]
    d["logv"] = np.log1p(d["volume"])
    d["logv_chg"] = d["logv"].diff().fillna(0)

    d["dist_px_cloud_top"] = (d["px"] - d["cloud_top"]) / d["px"]
    d["dist_px_cloud_bot"] = (d["px"] - d["cloud_bot"]) / d["px"]
    d["dist_tk_kj"] = (d["tenkan"] - d["kijun"]) / d["px"]
    d["span_order"] = (d["span_a"] > d["span_b"]).astype(float)

    sw = int(max(1, slope_window))
    d["tk_slope"] = (d["tenkan"] - d["tenkan"].shift(sw)) / (d["px"] + 1e-9)
    d["kj_slope"] = (d["kijun"] - d["kijun"].shift(sw)) / (d["px"] + 1e-9)
    d["span_a_slope"] = (d["span_a"] - d["span_a"].shift(sw)) / (d["px"] + 1e-9)
    d["span_b_slope"] = (d["span_b"] - d["span_b"].shift(sw)) / (d["px"] + 1e-9)

    D = int(displacement)
    d["chikou_above"] = (d["close"] > d["close"].shift(D)).astype(float)

    d["vol20"] = d["ret1"].rolling(20, min_periods=20).std().fillna(0)

    feature_cols = [
        "ret1", "oc_diff", "hl_range", "logv_chg",
        "dist_px_cloud_top", "dist_px_cloud_bot", "dist_tk_kj", "span_order",
        "tk_slope", "kj_slope", "span_a_slope", "span_b_slope",
        "chikou_above", "vol20",
    ]

    feat = d[feature_cols].copy().dropna().reset_index(drop=True)
    ts = d.loc[feat.index, "timestamp"].reset_index(drop=True)
    px = d.loc[feat.index, "close"].reset_index(drop=True)
    out = feat.copy()
    out.insert(0, "timestamp", ts)
    out.insert(1, "close", px)
    return out, feature_cols


# ==============
# Dataset helpers
# ==============

def fit_scaler(train_feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_feat.mean(axis=(0, 1), keepdims=True)
    std = train_feat.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)

def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)

def make_sequences_for_range(df_feat: pd.DataFrame, feature_cols: List[str], lookback: int,
                             start_idx: int, end_idx: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Build sequences for indices [start_idx..end_idx] inclusive.
    Returns X [N, lookback, F], and sequence end timestamps.
    """
    X, t_end = [], []
    for j in range(start_idx, end_idx + 1):
        if j - lookback + 1 < 0:
            continue
        seq = df_feat.loc[j - lookback + 1 : j, feature_cols].values.astype(np.float32)
        X.append(seq)
        t_end.append(df_feat.loc[j, "timestamp"])
    if not X:
        return np.zeros((0, lookback, len(feature_cols)), np.float32), []
    return np.stack(X), t_end

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X; self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============
# Model
# ==============

class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=layers,
                            dropout=(dropout if layers > 1 else 0.0), batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
    def forward(self, x):  # x: [B, T, F]
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        logit = self.head(h_last).squeeze(-1)
        return logit


# ==============
# Training
# ==============

def class_weights(y: np.ndarray) -> float:
    p = y.mean() if len(y) else 0.5
    if p <= 0: 
        return 1.0
    return float((1 - p) / max(1e-6, p))

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader],
                epochs: int, lr: float, device: str, pos_weight: float) -> List[tuple]:
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    hist = []
    for ep in range(epochs):
        model.train(); loss_sum = 0.0; n = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            logit = model(xb)
            loss = crit(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb); n += len(xb)
        train_loss = loss_sum / max(1, n)
        if val_loader is not None:
            model.eval(); vloss = 0.0; vn=0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    vloss += crit(model(xb), yb).item() * len(xb); vn += len(xb)
            val_loss = vloss / max(1, vn)
            hist.append((train_loss, val_loss))
            print(f"Epoch {ep+1}/{epochs} - train {train_loss:.4f} - val {val_loss:.4f}")
        else:
            hist.append((train_loss, None))
            print(f"Epoch {ep+1}/{epochs} - train {train_loss:.4f}")
    return hist


# ==============
# CLI & Orchestration
# ==============

def parse_args():
    ap = argparse.ArgumentParser(description="LSTM Ichimoku trading classifier")
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL"))
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--timeframe", required=True)

    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--test-end", required=False)

    # Ichimoku params
    ap.add_argument("--tenkan", type=int, default=9)
    ap.add_argument("--kijun", type=int, default=26)
    ap.add_argument("--senkou", type=int, default=52)
    ap.add_argument("--displacement", type=int, default=26)

    # Model/Training
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=4)  # only for label gen (training)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")

    # Trading thresholds & risk
    ap.add_argument("--pos-thr", type=float, default=0.55, help="Long threshold on p.")
    ap.add_argument("--sl-pct", type=float, default=0.02)
    ap.add_argument("--tp-pct", type=float, default=0.06)

    # Fees & slippage
    ap.add_argument("--fee-pct", type=float, default=0.001, help="Per-side commission (0.001 = 0.1%).")
    ap.add_argument("--fee-bps", type=float, default=None, help="Alternative fee in basis points; if set overrides fee-pct.")
    ap.add_argument("--slippage-sec", type=float, default=3.0, help="Fill delay in seconds within the bar.")

    # Capital (for equity calc; fully invested per trade)
    ap.add_argument("--initial-capital", type=float, default=10000.0)

    # Output locations
    ap.add_argument("--out-dir", default=".", help="Directory to write artifacts.")
    ap.add_argument("--bundle-dir", default=None, help="Optional second directory to also copy artifacts.")
    return ap.parse_args()


# ==============
# Backtest Engine
# ==============

@dataclass
class Position:
    side: str           # "long" or "short"
    entry_idx: int      # bar index where entry happened
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    take_price: float

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def bar_fill_with_slippage(open_price: float, close_price: float, frac: float, side: str) -> float:
    """
    Fill at next bar open adjusted by 'frac' progress towards close.
    side is not used for direction (we already bias via prices); kept for clarity/extension.
    """
    frac = max(0.0, min(1.0, float(frac)))
    return float(open_price + (close_price - open_price) * frac)

def simulate_trades(df: pd.DataFrame,
                    preds_p: pd.Series,
                    test_start_ts: pd.Timestamp,
                    test_end_ts: pd.Timestamp,
                    pos_thr: float,
                    sl_pct: float,
                    tp_pct: float,
                    fee_rate: float,
                    slip_frac: float,
                    initial_capital: float) -> Tuple[List[dict], dict]:
    """
    df: full OHLCV aligned to preds (same indexing as features).
    preds_p: Series of probabilities (index aligned to df rows where a prediction exists).
    We execute at next bar open, so last usable signal index is len(df)-2.
    """
    # Map timestamp -> row index
    ts = df["timestamp"].values
    # Identify usable signal indices (we require i+1 for execution)
    sig_idx = preds_p.index.astype(int)

    # Test window: signals strictly after train_end up to test_end-1 (because we need i+1 to exist)
    start_mask = df.loc[sig_idx, "timestamp"] > test_start_ts
    end_mask   = df.loc[sig_idx, "timestamp"] <= test_end_ts
    usable_idx = sig_idx[np.where(start_mask & end_mask)[0]]
    usable_idx = usable_idx[usable_idx < (len(df) - 1)]
    if len(usable_idx) == 0:
        return [], {
            "total_trades": 0, "long_trades": 0, "short_trades": 0,
            "long_wins": 0, "short_wins": 0, "losses": 0,
            "net_profit": 0.0, "pct_net_profit": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": None, "win_rate_pct": None, "max_drawdown_pct": 0.0,
            "buy_hold_pct": 0.0, "initial_capital": float(initial_capital)
        }

    trades: List[dict] = []
    pos: Optional[Position] = None
    equity = [initial_capital]
    capital = initial_capital

    # For buy&hold benchmark: from first EXECUTION bar open to last exit/close time
    first_exec_open = df.loc[usable_idx[0] + 1, "open"]
    last_close = df.loc[min(len(df)-1, usable_idx[-1] + 1), "close"]

    for i in usable_idx:
        next_i = i + 1  # execution bar
        # Ensure next_i exists
        if next_i >= len(df):
            break

        open_n = df.loc[next_i, "open"]
        high_n = df.loc[next_i, "high"]
        low_n  = df.loc[next_i, "low"]
        close_n= df.loc[next_i, "close"]

        # probability at signal time i
        p = float(preds_p.loc[i])

        # 1) Manage existing position: check SL/TP on execution bar (and subsequent bars as loop continues)
        if pos is not None:
            # Intrabar hit priority: SL first (conservative), then TP
            if pos.side == "long":
                # After entry we check every bar's range
                if low_n <= pos.stop_price:
                    exit_price = pos.stop_price * (1.0 - 0.0)  # assume stop hits at price; slippage ignored on stops
                    gross = (exit_price / pos.entry_price) - 1.0
                    net = gross - 2.0 * fee_rate
                    capital *= (1.0 + net)
                    trades.append({
                        "side": "long", "entry_index": pos.entry_idx, "exit_index": next_i,
                        "entry_time": str(pos.entry_time), "exit_time": str(df.loc[next_i, "timestamp"]),
                        "entry_price": pos.entry_price, "exit_price": float(exit_price),
                        "gross_return": float(gross), "net_return": float(net),
                        "reason": "SL"
                    })
                    pos = None
                    equity.append(capital)
                    continue
                elif high_n >= pos.take_price:
                    exit_price = pos.take_price
                    gross = (exit_price / pos.entry_price) - 1.0
                    net = gross - 2.0 * fee_rate
                    capital *= (1.0 + net)
                    trades.append({
                        "side": "long", "entry_index": pos.entry_idx, "exit_index": next_i,
                        "entry_time": str(pos.entry_time), "exit_time": str(df.loc[next_i, "timestamp"]),
                        "entry_price": pos.entry_price, "exit_price": float(exit_price),
                        "gross_return": float(gross), "net_return": float(net),
                        "reason": "TP"
                    })
                    pos = None
                    equity.append(capital)
                    continue
                # Signal to reverse/exit?
                if p < (1.0 - pos_thr):
                    # exit at next open (i+1 open we already have; apply slippage model)
                    fill = bar_fill_with_slippage(open_n, close_n, slip_frac, "exit")
                    exit_price = float(fill)
                    gross = (exit_price / pos.entry_price) - 1.0
                    net = gross - 2.0 * fee_rate
                    capital *= (1.0 + net)
                    trades.append({
                        "side": "long", "entry_index": pos.entry_idx, "exit_index": next_i,
                        "entry_time": str(pos.entry_time), "exit_time": str(df.loc[next_i, "timestamp"]),
                        "entry_price": pos.entry_price, "exit_price": exit_price,
                        "gross_return": float(gross), "net_return": float(net),
                        "reason": "SIGNAL"
                    })
                    pos = None
                    equity.append(capital)
                    # fall-through to possibly open new short below
                else:
                    # keep holding
                    pass

            else:  # short
                if high_n >= pos.stop_price:
                    exit_price = pos.stop_price
                    gross = (pos.entry_price / exit_price) - 1.0
                    net = gross - 2.0 * fee_rate
                    capital *= (1.0 + net)
                    trades.append({
                        "side": "short", "entry_index": pos.entry_idx, "exit_index": next_i,
                        "entry_time": str(pos.entry_time), "exit_time": str(df.loc[next_i, "timestamp"]),
                        "entry_price": pos.entry_price, "exit_price": float(exit_price),
                        "gross_return": float(gross), "net_return": float(net),
                        "reason": "SL"
                    })
                    pos = None
                    equity.append(capital)
                    continue
                elif low_n <= pos.take_price:
                    exit_price = pos.take_price
                    gross = (pos.entry_price / exit_price) - 1.0
                    net = gross - 2.0 * fee_rate
                    capital *= (1.0 + net)
                    trades.append({
                        "side": "short", "entry_index": pos.entry_idx, "exit_index": next_i,
                        "entry_time": str(pos.entry_time), "exit_time": str(df.loc[next_i, "timestamp"]),
                        "entry_price": pos.entry_price, "exit_price": float(exit_price),
                        "gross_return": float(gross), "net_return": float(net),
                        "reason": "TP"
                    })
                    pos = None
                    equity.append(capital)
                    continue
                if p > pos_thr:
                    fill = bar_fill_with_slippage(open_n, close_n, slip_frac, "exit")
                    exit_price = float(fill)
                    gross = (pos.entry_price / exit_price) - 1.0
                    net = gross - 2.0 * fee_rate
                    capital *= (1.0 + net)
                    trades.append({
                        "side": "short", "entry_index": pos.entry_idx, "exit_index": next_i,
                        "entry_time": str(pos.entry_time), "exit_time": str(df.loc[next_i, "timestamp"]),
                        "entry_price": pos.entry_price, "exit_price": exit_price,
                        "gross_return": float(gross), "net_return": float(net),
                        "reason": "SIGNAL"
                    })
                    pos = None
                    equity.append(capital)
                    # allow opening long below

        # 2) If flat, consider new entries
        if pos is None:
            if p > pos_thr:
                # Open long at next open + slippage
                fill = bar_fill_with_slippage(open_n, close_n, slip_frac, "long")
                entry_price = float(fill)
                stop_price = entry_price * (1.0 - sl_pct)
                take_price = entry_price * (1.0 + tp_pct)
                pos = Position(
                    side="long",
                    entry_idx=next_i,
                    entry_time=df.loc[next_i, "timestamp"],
                    entry_price=entry_price,
                    stop_price=stop_price,
                    take_price=take_price
                )
            elif p < (1.0 - pos_thr):
                fill = bar_fill_with_slippage(open_n, close_n, slip_frac, "short")
                entry_price = float(fill)
                stop_price = entry_price * (1.0 + sl_pct)
                take_price = entry_price * (1.0 - tp_pct)
                pos = Position(
                    side="short",
                    entry_idx=next_i,
                    entry_time=df.loc[next_i, "timestamp"],
                    entry_price=entry_price,
                    stop_price=stop_price,
                    take_price=take_price
                )

    # 3) Force close if a position remains at the very end of usable data
    if pos is not None:
        last_idx = min(len(df)-1, usable_idx[-1] + 1)
        exit_price = float(df.loc[last_idx, "close"])
        if pos.side == "long":
            gross = (exit_price / pos.entry_price) - 1.0
        else:
            gross = (pos.entry_price / exit_price) - 1.0
        net = gross - 2.0 * fee_rate
        capital *= (1.0 + net)
        trades.append({
            "side": pos.side, "entry_index": pos.entry_idx, "exit_index": last_idx,
            "entry_time": str(pos.entry_time), "exit_time": str(df.loc[last_idx, "timestamp"]),
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "gross_return": float(gross), "net_return": float(net),
            "reason": "END"
        })
        equity.append(capital)

    # -------- Summary -------
    total = len(trades)
    long_trades = sum(1 for t in trades if t["side"] == "long")
    short_trades = total - long_trades
    long_wins = sum(1 for t in trades if t["side"] == "long" and t["net_return"] > 0)
    short_wins = sum(1 for t in trades if t["side"] == "short" and t["net_return"] > 0)
    losses = sum(1 for t in trades if t["net_return"] <= 0)

    gross_profit = sum(t["net_return"] for t in trades if t["net_return"] > 0) * initial_capital
    gross_loss = sum(t["net_return"] for t in trades if t["net_return"] <= 0) * initial_capital
    net_profit = capital - initial_capital
    pct_net_profit = (capital / initial_capital) - 1.0
    profit_factor = (abs(gross_profit) / abs(gross_loss)) if gross_loss != 0 else None
    win_rate = (long_wins + short_wins) / total * 100.0 if total > 0 else None

    equity_arr = np.array(equity, dtype=float)
    mdd_pct, _mdd_abs = calc_max_drawdown(equity_arr)
    buy_hold_pct = (last_close / first_exec_open) - 1.0 if total > 0 and first_exec_open > 0 else 0.0

    summary = {
        "total_trades": total,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_wins": long_wins,
        "short_wins": short_wins,
        "losses": losses,
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "net_profit": float(net_profit),
        "pct_net_profit": float(pct_net_profit),
        "profit_factor": (None if profit_factor is None else float(profit_factor)),
        "win_rate_pct": (None if win_rate is None else float(win_rate)),
        "max_drawdown_pct": float(mdd_pct),  # negative number e.g. -0.23
        "buy_hold_pct": float(buy_hold_pct),
        "initial_capital": float(initial_capital)
    }
    return trades, summary


# ==============
# Main
# ==============

def main():
    args = parse_args()
    if not args.db_url:
        raise SystemExit("Provide --db-url or set DATABASE_URL env var.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = None
    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir).expanduser().resolve()
        bundle_dir.mkdir(parents=True, exist_ok=True)

    # Load data over [train_start .. test_end]
    df_all = load_db_ohlcv(args.db_url, ticker=args.ticker, timeframe=args.timeframe,
                           start=args.train_start, end=args.test_end or args.train_end)

    # Build features
    feat_all, feature_cols = build_features(
        df_all, args.tenkan, args.kijun, args.senkou, args.displacement
    )

    # Build training sequences strictly inside the training window (for labels)
    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)

    # For labels, need horizon into the future; find training end index in features
    # But for simplicity we build sequences with labels on [..train_end] and discard incomplete
    # Generate sequences across the *feature* index range covering training dates.
    # First, map timestamps to feature indices
    feat_ts = feat_all["timestamp"]
    tr_mask = (feat_ts >= train_start) & (feat_ts <= train_end)
    tr_idx = np.where(tr_mask.values)[0]
    if len(tr_idx) == 0:
        raise SystemExit("No features in training window.")
    tr_start_idx, tr_end_idx = int(tr_idx[0]), int(tr_idx[-1])

    # Build sequences for training labels using horizon
    # We'll compute labels by future close comparison like earlier function did.
    # Re-implement a tight version here:
    X_raw, Y, _t_end = [], [], []
    for j in range(tr_start_idx, tr_end_idx + 1):
        if j - args.lookback + 1 < 0: 
            continue
        if j + args.horizon >= len(feat_all):
            break
        seq = feat_all.loc[j - args.lookback + 1 : j, feature_cols].values.astype(np.float32)
        c0 = float(feat_all.loc[j, "close"])
        cH = float(feat_all.loc[j + args.horizon, "close"])
        label = 1.0 if (cH / c0 - 1.0) > 0.0 else 0.0
        X_raw.append(seq)
        Y.append(label)
        _t_end.append(feat_all.loc[j, "timestamp"])
    if not X_raw:
        raise SystemExit("Not enough data to build training sequences. Reduce lookback/horizon or extend window.")

    X_raw = np.stack(X_raw)
    Y = np.array(Y, dtype=np.float32)

    # Time-ordered split
    n = len(X_raw)
    val_n = int(round(max(0.0, min(1.0, args.val_frac)) * n))
    if val_n > 0 and val_n < n:
        idx_tr = np.arange(0, n - val_n)
        idx_va = np.arange(n - val_n, n)
    else:
        idx_tr = np.arange(0, n)
        idx_va = np.array([], dtype=int)

    # Fit scaler on train only
    mean, std = fit_scaler(X_raw[idx_tr])
    X_tr = apply_scaler(X_raw[idx_tr], mean, std)
    y_tr = Y[idx_tr]
    X_va = apply_scaler(X_raw[idx_va], mean, std) if len(idx_va) else None
    y_va = Y[idx_va] if len(idx_va) else None

    # Train
    ds_tr = SeqDataset(X_tr, y_tr)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False, drop_last=False)
    dl_va = None
    if X_va is not None and len(idx_va) > 0:
        ds_va = SeqDataset(X_va, y_va)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = LSTMClassifier(n_features=len(feature_cols), hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    pw = class_weights(y_tr)
    print(f"Class balance: pos_rate={y_tr.mean():.3f}, pos_weight={pw:.3f}")
    train_model(model, dl_tr, dl_va, epochs=args.epochs, lr=args.lr, device=args.device, pos_weight=pw)

    # =========================
    # SAVE CORE ARTIFACTS
    # =========================
    fp_model = out_dir / "model.pt"
    fp_pre = out_dir / "preprocess.json"
    fp_post = out_dir / "postprocess.json"
    fp_meta = out_dir / "meta.json"
    fp_trades = out_dir / "trades.json"
    fp_summary = out_dir / "summary.json"

    torch.save(model.state_dict(), str(fp_model))

    preprocess = {
        "version": 1,
        "normalization": "standardize",
        "feature_cols": feature_cols,
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "mean": np.asarray(mean, dtype=np.float32).reshape(-1).tolist(),
        "std": np.asarray(std, dtype=np.float32).reshape(-1).tolist()
    }
    _save_json(fp_pre, preprocess)

    fee_rate = (args.fee_bps / 10000.0) if (args.fee_bps is not None) else float(args.fee_pct)
    postprocess = {
        "problem": "binary_up_probability",
        "decision_rule": "threshold_long_and_short",
        "threshold_long": float(args.pos_thr),
        "threshold_short": float(1.0 - args.pos_thr),
        "sl_pct": float(args.sl_pct),
        "tp_pct": float(args.tp_pct),
        "fee_rate_per_side": float(fee_rate),
        "slippage_seconds": float(args.slippage_sec)
    }
    _save_json(fp_post, postprocess)

    meta = {
        "model_name": "lstm_trading_ichimoku",
        "framework": "pytorch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "test_end": args.test_end,
        "artifact_dir": str(out_dir),
        "ichimoku": {
            "tenkan": int(args.tenkan),
            "kijun": int(args.kijun),
            "senkou": int(args.senkou),
            "displacement": int(args.displacement)
        },
        "architecture": {
            "type": "LSTMClassifier",
            "n_features": len(feature_cols),
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "dropout": float(args.dropout),
            "input_shape": [int(args.lookback), len(feature_cols)],
            "output": "logit(1)"
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "val_frac": float(args.val_frac),
            "seed": int(args.seed),
            "pos_weight": float(pw)
        },
        "files": {
            "model": "model.pt",
            "preprocess": "preprocess.json",
            "postprocess": "postprocess.json",
            "trades": "trades.json",
            "summary": "summary.json"
        }
    }
    _save_json(fp_meta, meta)

    # =========================
    # INFERENCE OVER TEST WINDOW & BACKTEST
    # =========================
    # Build *all* sequences for inference (no labels needed)
    # We need predictions for indices covering (train_end .. test_end)
    feat_ts = feat_all["timestamp"].reset_index(drop=True)
    all_idx = np.arange(len(feat_all), dtype=int)
    X_all, t_end_all = make_sequences_for_range(
        feat_all, feature_cols, args.lookback,
        start_idx=0, end_idx=len(feat_all)-1
    )
    if len(X_all) == 0:
        print("WARNING: No sequences built for inference; skipping trades/summary.")
        trades, summary = [], {}
    else:
        X_all = apply_scaler(X_all, mean, std)
        model.eval()
        with torch.no_grad():
            logits = []
            B = 1024
            for s in range(0, len(X_all), B):
                xb = torch.tensor(X_all[s:s+B]).to(args.device)
                lb = model(xb).cpu().numpy()
                logits.append(lb)
            logits = np.concatenate(logits, axis=0)
        preds_p = sigmoid(logits)

        # Align predictions to feature indices (each prediction corresponds to sequence ending at index j)
        # Create a Series mapping index j -> p
        pred_index_map = {}
        # t_end_all is aligned to sequence end indices; retrieve their integer index in feat_ts
        ts_to_idx = {ts: i for i, ts in enumerate(feat_ts)}
        for j_ts, p in zip(t_end_all, preds_p):
            j = ts_to_idx.get(j_ts)
            if j is not None:
                pred_index_map[j] = float(p)
        preds_series = pd.Series(pred_index_map)

        # Test window bounds
        test_start_ts = pd.to_datetime(args.train_end)  # strictly > this for signals
        test_end_ts = pd.to_datetime(args.test_end) if args.test_end else feat_ts.iloc[-1]

        # Slippage fraction within one bar
        bar_sec = timeframe_to_seconds(args.timeframe)
        slip_frac = float(args.slippage_sec) / float(bar_sec) if bar_sec > 0 else 0.0
        slip_frac = max(0.0, min(1.0, slip_frac))

        trades, summary = simulate_trades(
            df_all, preds_series, test_start_ts, test_end_ts,
            pos_thr=float(args.pos_thr), sl_pct=float(args.sl_pct), tp_pct=float(args.tp_pct),
            fee_rate=float(fee_rate), slip_frac=slip_frac, initial_capital=float(args.initial_capital)
        )

    _save_json(fp_trades, {"ticker": args.ticker, "timeframe": args.timeframe, "trades": trades})
    _save_json(fp_summary, summary)

    # Optionally copy to bundle-dir
    if bundle_dir and bundle_dir != out_dir:
        for p in (fp_model, fp_pre, fp_post, fp_meta, fp_trades, fp_summary):
            shutil.copy2(str(p), str(bundle_dir / p.name))

    print("\nSaved artifacts:")
    for p in (fp_model, fp_meta, fp_pre, fp_post, fp_trades, fp_summary):
        print(f"  - {p}")
    if bundle_dir and bundle_dir != out_dir:
        print(f"Copied artifacts to bundle-dir: {bundle_dir}")


if __name__ == "__main__":
    main()
