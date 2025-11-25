#!/usr/bin/env python3
"""
LSTM Ichimoku Trader — Long & Short (No Overlap) + RSI + Train/Val/Test
=======================================================================

Key features:

- Ichimoku + RSI(14) with 2-std bands:
  - RSI (Wilder)
  - RSI z-score relative to ±2 std band (rsi_z2)

- Three levels of splitting:

  1) Training window (by date):
       [--train-start, --train-end]

     → Sequences built here are further split *by time* into:
         - TRAIN (early part)
         - INTERNAL VALIDATION (later part) via --val-frac
       This is purely for model selection / early stopping.

  2) Validation *backtest* window (optional):
       [--val-start, --val-end]

     → Model (best epoch) is frozen and applied in a trading backtest.
       Outputs:
         - lstm_ichimoku_equity_val_<TICKER>.csv
         - lstm_ichimoku_trades_val_<TICKER>.json
         - summary["validation"]

  3) Test *backtest* window (optional):
       [--test-start, --test-end]
     If --test-start is omitted but --test-end is given, we default to:
       test_start = train_end + 1 second

- SAT bundle output:
  - model.pt (TorchScript)
  - preprocess.json (mean/std + feature names)
  - meta.json (Ichimoku+RSI params, thresholds, risk, etc.)

Example:

  python lstm_trading_ichimoku_longshort.py \
    --ticker BTCUSDT --timeframe 1h \
    --train-start 2023-01-01 --train-end 2024-12-31 \
    --val-start 2025-01-01 --val-end 2025-11-01 \
    --lookback 64 --horizon 4 --epochs 15 --batch-size 256 \
    --pos-thr 0.55 --sl-pct 0.02 --tp-pct 0.06 \
    --out-dir ./results/BTCUSDT_1h
"""
from __future__ import annotations

import argparse
import json
import os
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

import shutil
import time


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
            # Heuristic: treat large numbers as ms/s since epoch
            if xv >= 1e12:  # ms
                return pd.to_datetime(int(xv), unit="ms", utc=False)
            elif xv >= 1e9:  # s
                return pd.to_datetime(int(xv), unit="s", utc=False)
        except Exception:
            pass
        # Try general parse
        return pd.to_datetime(x, utc=False, errors="coerce")

    out = ts.map(_coerce_one)
    if out.isna().any():
        out2 = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(None)
        out = out.fillna(out2)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


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
# Ichimoku, RSI & Features
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


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder-style RSI.
    """
    length = int(length)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_features(
    df: pd.DataFrame,
    tenkan: int,
    kijun: int,
    senkou: int,
    displacement: int,
    slope_window: int = 8,
    rsi_len: int = 14,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build Ichimoku + RSI feature set.
    """
    d = ichimoku(df, tenkan, kijun, senkou)

    # Basic price / volume transforms
    d["px"] = d["close"]
    d["ret1"] = d["close"].pct_change().fillna(0)
    d["oc_diff"] = (d["close"] - d["open"]) / d["open"]
    d["hl_range"] = (d["high"] - d["low"]) / d["px"]
    d["logv"] = np.log1p(d["volume"])
    d["logv_chg"] = d["logv"].diff().fillna(0)

    # Ichimoku distances
    d["dist_px_cloud_top"] = (d["px"] - d["cloud_top"]) / d["px"]
    d["dist_px_cloud_bot"] = (d["px"] - d["cloud_bot"]) / d["px"]
    d["dist_tk_kj"] = (d["tenkan"] - d["kijun"]) / d["px"]
    d["span_order"] = (d["span_a"] > d["span_b"]).astype(float)

    # Slopes
    sw = int(max(1, slope_window))
    d["tk_slope"] = (d["tenkan"] - d["tenkan"].shift(sw)) / (d["px"] + 1e-9)
    d["kj_slope"] = (d["kijun"] - d["kijun"].shift(sw)) / (d["px"] + 1e-9)
    d["span_a_slope"] = (d["span_a"] - d["span_a"].shift(sw)) / (d["px"] + 1e-9)
    d["span_b_slope"] = (d["span_b"] - d["span_b"].shift(sw)) / (d["px"] + 1e-9)

    D = int(displacement)
    d["chikou_above"] = (d["close"] > d["close"].shift(D)).astype(float)
    d["vol20"] = d["ret1"].rolling(20, min_periods=20).std().fillna(0)

    # ---- RSI features (length=rsi_len, using ±2 std bands) ----
    d["rsi"] = compute_rsi(d["close"], length=rsi_len)
    d["rsi_ma"] = d["rsi"].rolling(rsi_len, min_periods=rsi_len).mean()
    d["rsi_std"] = d["rsi"].rolling(rsi_len, min_periods=rsi_len).std()

    # Bollinger-style bands on RSI (± 2 * std)
    d["rsi_upper"] = d["rsi_ma"] + 2.0 * d["rsi_std"]
    d["rsi_lower"] = d["rsi_ma"] - 2.0 * d["rsi_std"]

    # Normalized deviation from the band width; uses "2 std" explicitly
    denom = (2.0 * d["rsi_std"]).replace(0, np.nan)
    d["rsi_z2"] = (d["rsi"] - d["rsi_ma"]) / denom

    feature_cols = [
        "ret1", "oc_diff", "hl_range", "logv_chg",
        "dist_px_cloud_top", "dist_px_cloud_bot", "dist_tk_kj", "span_order",
        "tk_slope", "kj_slope", "span_a_slope", "span_b_slope",
        "chikou_above", "vol20",
        "rsi", "rsi_z2",
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


def make_sequences(df_feat: pd.DataFrame, feature_cols: List[str], lookback: int, horizon: int,
                   start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Build sequences from df_feat in [start_ts, end_ts], using global index
    so the label horizon can use bars after end_ts if needed.
    """
    X, y, t_end = [], [], []

    # We'll iterate on a mask in the *full* df_feat (global index),
    # but it will only include timestamps inside [start_ts, end_ts].
    mask = (df_feat["timestamp"] >= start_ts) & (df_feat["timestamp"] <= end_ts)
    idxs = df_feat.index[mask].tolist()

    for j in idxs:
        if j - lookback + 1 < 0:
            continue
        if j + horizon >= len(df_feat):
            break

        seq = df_feat.loc[j - lookback + 1 : j, feature_cols].values.astype(np.float32)
        c0 = float(df_feat.loc[j, "close"])
        cH = float(df_feat.loc[j + horizon, "close"])
        ret = (cH / c0) - 1.0
        label = 1.0 if ret > 0.0 else 0.0

        X.append(seq)
        y.append(label)
        t_end.append(df_feat.loc[j, "timestamp"])

    if not X:
        return np.zeros((0, lookback, len(feature_cols)), np.float32), np.zeros((0,), np.float32), []
    return np.stack(X), np.array(y, dtype=np.float32), t_end


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ==============
# Model
# ==============

class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=(dropout if layers > 1 else 0.0),
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.head(h_last).squeeze(-1)


# ==============
# Training / Eval
# ==============

def class_weights(y: np.ndarray) -> float:
    p = y.mean() if len(y) else 0.5
    if p <= 0:
        return 1.0
    return float((1.0 - p) / max(1e-6, p))


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader],
                epochs: int, lr: float, device: str, pos_weight: float, seed: int = 0) -> List[Tuple[float, Optional[float]]]:
    torch.manual_seed(seed)
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    hist: List[Tuple[float, Optional[float]]] = []

    import copy
    best_val = float("inf")
    best_state = None
    best_epoch = None

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            n += len(xb)

        train_loss = loss_sum / max(1, n)

        if val_loader is not None:
            model.eval()
            vloss = 0.0
            vn = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    vloss += crit(model(xb), yb).item() * len(xb)
                    vn += len(xb)
            val_loss = vloss / max(1, vn)
            hist.append((train_loss, val_loss))
            print(f"Epoch {ep+1}/{epochs} - train {train_loss:.4f} - val {val_loss:.4f}")

            # track best validation epoch
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = ep + 1  # 1-based
                best_state = copy.deepcopy(model.state_dict())
        else:
            hist.append((train_loss, None))
            print(f"Epoch {ep+1}/{epochs} - train {train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model from epoch {best_epoch} (val={best_val:.4f})")
    else:
        print("WARNING: No best_state found, exporting last-epoch weights.")

    return hist


def predict_proba(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            xb = torch.tensor(X[i:i+2048], dtype=torch.float32, device=device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0) if out else np.zeros((0,), np.float32)


# ==============
# Backtest (long/short, no overlap)
# ==============

def backtest_threshold_ls(df_raw: pd.DataFrame, df_feat: pd.DataFrame, feature_cols: List[str],
                          model: nn.Module, mean: np.ndarray, std: np.ndarray,
                          lookback: int, pos_thr: float, neg_thr: float,
                          start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                          sl_pct: float, tp_pct: float, fee_bps: float,
                          device: str = "cpu") -> Tuple[pd.Series, List[Dict]]:
    """
    Long/Short backtester with single position at a time (no overlap).
    pos==0: flat, pos==+1: long, pos==-1: short.

    Uses timestamp-based alignment between features and raw OHLCV to
    avoid index-offset issues.
    """
    equity = 1_000.0
    fee_mult = 1.0 - (fee_bps / 1_000.0)
    pos = 0
    entry_px = 0.0
    eq_curve: List[Tuple[pd.Timestamp, float]] = []
    trades: List[Dict] = []

    # Map raw timestamps to their indices for safe alignment
    raw_ts_to_idx = {ts: i for i, ts in enumerate(df_raw["timestamp"])}

    # Iterate only inside trading window
    mask = (df_feat["timestamp"] >= start_ts) & (df_feat["timestamp"] <= end_ts)
    idxs = df_feat.index[mask].tolist()

    for j in idxs:
        # Need lookback and also ensure we can build p_prev
        if j - lookback + 1 < 0 or j - lookback < 0:
            continue

        ts_now = df_feat.loc[j, "timestamp"]
        k = raw_ts_to_idx.get(ts_now)
        if k is None:
            # No matching raw bar for this feature timestamp
            continue

        k1 = k + 1
        if k1 >= len(df_raw):
            # No next bar to execute on
            continue

        # --- Build sequences for p_last and p_prev ---
        seq_last = df_feat.loc[j - lookback + 1 : j, feature_cols].values[np.newaxis, :, :].astype(np.float32)
        seq_last = (seq_last - mean) / std
        p_last = float(predict_proba(model, seq_last, device=device)[0])

        seq_prev = df_feat.loc[j - lookback : j - 1, feature_cols].values[np.newaxis, :, :].astype(np.float32)
        seq_prev = (seq_prev - mean) / std
        p_prev = float(predict_proba(model, seq_prev, device=device)[0])

        # Execution bar (next raw bar)
        o1 = float(df_raw.loc[k1, "open"])
        h1 = float(df_raw.loc[k1, "high"])
        l1 = float(df_raw.loc[k1, "low"])
        c1 = float(df_raw.loc[k1, "close"])
        ts1 = df_raw.loc[k1, "timestamp"]

        # ------ Flat: consider entries ------
        if pos == 0:
            take_long = (p_last >= pos_thr) and (p_prev < p_last)
            take_short = (p_last <= neg_thr) and (p_prev > p_last)

            if take_long:
                pos = +1
                entry_px = o1
                equity *= fee_mult
                trades.append({
                    "side": "LONG",
                    "entry_ts": str(ts1),
                    "entry_px": entry_px,
                    "p_prev": p_prev,
                    "p_last": p_last,
                })
                eq_curve.append((ts1, equity))
                continue

            if take_short:
                pos = -1
                entry_px = o1
                equity *= fee_mult
                trades.append({
                    "side": "SHORT",
                    "entry_ts": str(ts1),
                    "entry_px": entry_px,
                    "p_prev": p_prev,
                    "p_last": p_last,
                })
                eq_curve.append((ts1, equity))
                continue

            eq_curve.append((ts1, equity))
            continue

        # ------ Manage open LONG ------
        if pos == +1:
            sl = entry_px * (1.0 - sl_pct)
            tp = entry_px * (1.0 + tp_pct)
            exit_now = False
            exit_px = None
            reason = None

            hit_sl = (l1 <= sl)
            hit_tp = (h1 >= tp)
            # prefer TP
            if hit_tp:
                exit_now = True
                exit_px = tp
                reason = "TP"
            elif hit_sl:
                exit_now = True
                exit_px = sl
                reason = "SL"

            # Opposite fresh-cross to close (reversal)
            if not exit_now and ((p_last < neg_thr) and (p_prev > p_last)):
                exit_now = True
                exit_px = o1
                reason = "REV"

            if exit_now:
                qty = equity / entry_px
                pnl = (exit_px - entry_px) * qty
                equity += pnl
                equity *= fee_mult
                pos = 0
                trades[-1].update({
                    "exit_ts": str(ts1),
                    "exit_px": exit_px,
                    "reason": reason,
                    "pnl": pnl,
                })
                eq_curve.append((ts1, equity))
                continue

            eq_curve.append((ts1, equity))
            continue

        # ------ Manage open SHORT ------
        if pos == -1:
            sl = entry_px * (1.0 + sl_pct)
            tp = entry_px * (1.0 - tp_pct)
            exit_now = False
            exit_px = None
            reason = None

            hit_tp = (l1 <= tp)
            hit_sl = (h1 >= sl)
            if hit_tp:
                exit_now = True
                exit_px = tp
                reason = "TP"
            elif hit_sl:
                exit_now = True
                exit_px = sl
                reason = "SL"

            if not exit_now and ((p_last > pos_thr) and (p_prev < p_last)):
                exit_now = True
                exit_px = o1
                reason = "REV"

            if exit_now:
                qty = equity / entry_px
                pnl = (entry_px - exit_px) * qty
                equity += pnl
                equity *= fee_mult
                pos = 0
                trades[-1].update({
                    "exit_ts": str(ts1),
                    "exit_px": exit_px,
                    "reason": reason,
                    "pnl": pnl,
                })
                eq_curve.append((ts1, equity))
                continue

            eq_curve.append((ts1, equity))
            continue

    # Convert curve to Series and return
    if eq_curve:
        ts_list, eq_list = zip(*eq_curve)
        eq_series = pd.Series(eq_list, index=pd.to_datetime(ts_list), name="equity")
    else:
        eq_series = pd.Series(dtype=float)

    return eq_series, trades


# ==============
# CLI & Orchestration
# ==============

def parse_args():
    ap = argparse.ArgumentParser(description="LSTM Ichimoku+RSI trading classifier + backtests (train/val/test)")

    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="Postgres DSN; falls back to env DATABASE_URL")
    ap.add_argument("--ticker", required=True, help="Ticker (e.g., ETHUSDT)")
    ap.add_argument("--timeframe", required=True, help="Timeframe label (e.g., 4h, 1h)")

    # Primary training window
    ap.add_argument("--train-start", required=True, help="YYYY-MM-DD (training window start)")
    ap.add_argument("--train-end", required=True, help="YYYY-MM-DD (training window end)")

    # Validation backtest window (optional)
    ap.add_argument("--val-start", help="YYYY-MM-DD (validation backtest start)")
    ap.add_argument("--val-end", help="YYYY-MM-DD (validation backtest end)")

    # Test backtest window (optional)
    ap.add_argument("--test-start", help="YYYY-MM-DD (test backtest start; defaults to train-end+1s if --test-end is given)")
    ap.add_argument("--test-end", help="YYYY-MM-DD (test backtest end)")

    # Ichimoku params
    ap.add_argument("--tenkan", type=int, default=9)
    ap.add_argument("--kijun", type=int, default=26)
    ap.add_argument("--senkou", type=int, default=52)
    ap.add_argument("--displacement", type=int, default=26)

    # RSI params
    ap.add_argument("--rsi-len", type=int, default=14, help="RSI length (used with 2 std bands)")

    # Model/Training
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fraction of TRAIN window reserved for internal validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")

    # Trading thresholds / risk
    ap.add_argument("--pos-thr", type=float, default=0.55, help="Go LONG if P(up) >= pos_thr")
    ap.add_argument("--neg-thr", type=float, default=None, help="Go SHORT if P(up) <= neg_thr (default=1-pos_thr)")
    ap.add_argument("--sl-pct", type=float, default=0.02)
    ap.add_argument("--tp-pct", type=float, default=0.06)
    ap.add_argument("--fee-bps", type=float, default=1.0)

    # Output directory
    ap.add_argument("--out-dir", type=str, default=".", help="Directory to save artifacts (created if missing)")
    ap.add_argument("--bundle-dir", type=str, help="If set, also write SAT bundle (model.pt, preprocess.json, meta.json)")

    return ap.parse_args()


def main():
    args = parse_args()
    if not args.db_url:
        raise SystemExit("Provide --db-url or set DATABASE_URL env var.")

    # Compute neg_thr if not given
    neg_thr = (1.0 - args.pos_thr) if args.neg_thr is None else float(args.neg_thr)
    if neg_thr >= args.pos_thr:
        print(f"[warn] neg_thr ({neg_thr}) >= pos_thr ({args.pos_thr}); tightening to avoid overlap.")
        neg_thr = max(1e-6, args.pos_thr - 1e-6)

    # Prepare output directory
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    def OP(name: str) -> str:
        return os.path.join(out_dir, name)

    # -------------------
    # Date handling
    # -------------------
    train_start_dt = pd.to_datetime(args.train_start)
    train_end_dt = pd.to_datetime(args.train_end)

    val_start_dt = pd.to_datetime(args.val_start) if args.val_start else None
    val_end_dt = pd.to_datetime(args.val_end) if args.val_end else None

    test_start_dt = pd.to_datetime(args.test_start) if args.test_start else None
    test_end_dt = pd.to_datetime(args.test_end) if args.test_end else None

    # Global load range: min start → max end among all provided windows
    date_starts = [train_start_dt]
    date_ends = [train_end_dt]

    if val_start_dt is not None:
        date_starts.append(val_start_dt)
    if val_end_dt is not None:
        date_ends.append(val_end_dt)
    if test_start_dt is not None:
        date_starts.append(test_start_dt)
    if test_end_dt is not None:
        date_ends.append(test_end_dt)

    global_start = min(date_starts).strftime("%Y-%m-%d")
    global_end = max(date_ends).strftime("%Y-%m-%d")

    print(f"[INFO] Global load window: {global_start} → {global_end}")

    # 1) Load full window (train + optional val/test)
    df_all = load_db_ohlcv(
        args.db_url,
        ticker=args.ticker,
        timeframe=args.timeframe,
        start=global_start,
        end=global_end,
    )

    # 2) Build features (no look-ahead)
    feat_all, feature_cols = build_features(
        df_all,
        args.tenkan,
        args.kijun,
        args.senkou,
        args.displacement,
        rsi_len=args.rsi_len,
    )

    # 3) Train/Val sequences (fit scaler on TRAIN subset only)
    X_train_raw, y_train_all, t_end_train = make_sequences(
        feat_all,
        feature_cols,
        args.lookback,
        args.horizon,
        start_ts=train_start_dt,
        end_ts=train_end_dt,
    )
    if len(X_train_raw) == 0:
        raise SystemExit("Not enough data to build training sequences. Reduce lookback/horizon or extend window.")

    n = len(X_train_raw)
    if n < 2:
        raise SystemExit("Need at least 2 sequences to form train/val splits.")

    # ---- Time-based split: TRAIN / INTERNAL VALIDATION ----
    val_n = int(max(1, round(args.val_frac * n)))
    if val_n >= n:
        val_n = n - 1  # keep at least one train sample

    train_n = n - val_n
    tr_idx = np.arange(0, train_n)
    val_idx = np.arange(train_n, n)

    mean, std = fit_scaler(X_train_raw[tr_idx])
    X_train = apply_scaler(X_train_raw[tr_idx], mean, std)
    y_train = y_train_all[tr_idx]
    X_val = apply_scaler(X_train_raw[val_idx], mean, std)
    y_val = y_train_all[val_idx]

    print(f"Total train-window sequences: {n}")
    print(f"Train sequences (internal): {len(tr_idx)} "
          f"({t_end_train[tr_idx[0]]} → {t_end_train[tr_idx[-1]]})")
    print(f"Val sequences   (internal): {len(val_idx)} "
          f"({t_end_train[val_idx[0]]} → {t_end_train[val_idx[-1]]})")

    # 4) Train
    ds_tr = SeqDataset(X_train, y_train)
    ds_va = SeqDataset(X_val, y_val) if len(X_val) else None
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False) if ds_va else None

    model = LSTMClassifier(
        n_features=len(feature_cols),
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    )
    pw = class_weights(y_train)
    print(f"Class balance: pos_rate={y_train.mean():.3f}, pos_weight={pw:.3f}")
    train_model(
        model,
        dl_tr,
        dl_va,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        pos_weight=pw,
        seed=args.seed,
    )

    # ================================
    # Save artifacts (TorchScript model + scaler)
    # ================================
    model_cpu = model.to("cpu").eval()
    example = torch.zeros(1, args.lookback, len(feature_cols), dtype=torch.float32)
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception:
        scripted = torch.jit.trace(model_cpu, example)

    mdl_path = OP(f"lstm_ichimoku_{args.ticker}.pt")
    scripted.save(mdl_path)

    ckpt_path = OP(f"lstm_ichimoku_{args.ticker}.ckpt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "mean": mean,
            "std": std,
            "tenkan": args.tenkan,
            "kijun": args.kijun,
            "senkou": args.senkou,
            "displacement": args.displacement,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
        },
        ckpt_path,
    )

    # 5) Training window predictions (for inspection)
    X_tr_all = apply_scaler(X_train_raw, mean, std)
    p_tr = predict_proba(model_cpu, X_tr_all, device="cpu")
    preds_train = pd.DataFrame(
        {"timestamp": t_end_train, "prob": p_tr, "label": y_train_all}
    )
    preds_train_path = OP(f"lstm_ichimoku_preds_train_{args.ticker}.csv")
    preds_train.to_csv(preds_train_path, index=False)

    # 6) Summary structure
    summary: Dict[str, object] = {
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "model_path": mdl_path,
        "pos_thr": args.pos_thr,
        "neg_thr": float(neg_thr),
        "out_dir": out_dir,
    }

    equity_val_path = None
    trades_val_path = None
    equity_test_path = None
    trades_test_path = None

    summary_path = OP(f"lstm_ichimoku_summary_{args.ticker}.json")

    # ================================
    # Validation backtest (if val period provided)
    # ================================
    if val_start_dt is not None and val_end_dt is not None:
        print(f"Validation backtest window: {val_start_dt} → {val_end_dt}")
        eq_val, trades_val = backtest_threshold_ls(
            df_all,
            feat_all,
            feature_cols,
            model_cpu,
            mean,
            std,
            lookback=args.lookback,
            pos_thr=args.pos_thr,
            neg_thr=neg_thr,
            start_ts=val_start_dt,
            end_ts=val_end_dt,
            sl_pct=args.sl_pct,
            tp_pct=args.tp_pct,
            fee_bps=args.fee_bps,
            device="cpu",
        )

        if len(eq_val) > 0:
            net_ret_pct_val = (eq_val.iloc[-1] / eq_val.iloc[0] - 1.0) * 100.0
            mdd_pct_val = float(((eq_val / eq_val.cummax()) - 1.0).min()) * 100.0
        else:
            net_ret_pct_val = 0.0
            mdd_pct_val = 0.0

        long_val = sum(1 for t in trades_val if t.get("side") == "LONG" and "exit_px" in t)
        short_val = sum(1 for t in trades_val if t.get("side") == "SHORT" and "exit_px" in t)

        equity_val_path = OP(f"lstm_ichimoku_equity_val_{args.ticker}.csv")
        eq_val.to_csv(equity_val_path)

        trades_val_path = OP(f"lstm_ichimoku_trades_val_{args.ticker}.json")
        with open(trades_val_path, "w") as f:
            json.dump(trades_val, f, indent=2)

        summary["validation"] = {
            "net_profit_pct": float(net_ret_pct_val),
            "max_drawdown_pct": float(-mdd_pct_val),
            "trades": len(trades_val),
            "long_trades": long_val,
            "short_trades": short_val,
            "equity_path": equity_val_path,
            "trades_path": trades_val_path,
        }

    # ================================
    # Test backtest (if test_end provided)
    # ================================
    if test_end_dt is not None:
        if test_start_dt is None:
            test_start_dt = train_end_dt + pd.Timedelta(seconds=1)

        print(f"Test backtest window: {test_start_dt} → {test_end_dt}")
        eq_test, trades_test = backtest_threshold_ls(
            df_all,
            feat_all,
            feature_cols,
            model_cpu,
            mean,
            std,
            lookback=args.lookback,
            pos_thr=args.pos_thr,
            neg_thr=neg_thr,
            start_ts=test_start_dt,
            end_ts=test_end_dt,
            sl_pct=args.sl_pct,
            tp_pct=args.tp_pct,
            fee_bps=args.fee_bps,
            device="cpu",
        )

        if len(eq_test) > 0:
            net_ret_pct_test = (eq_test.iloc[-1] / eq_test.iloc[0] - 1.0) * 100.0
            mdd_pct_test = float(((eq_test / eq_test.cummax()) - 1.0).min()) * 100.0
        else:
            net_ret_pct_test = 0.0
            mdd_pct_test = 0.0

        long_test = sum(1 for t in trades_test if t.get("side") == "LONG" and "exit_px" in t)
        short_test = sum(1 for t in trades_test if t.get("side") == "SHORT" and "exit_px" in t)

        equity_test_path = OP(f"lstm_ichimoku_equity_test_{args.ticker}.csv")
        eq_test.to_csv(equity_test_path)

        trades_test_path = OP(f"lstm_ichimoku_trades_test_{args.ticker}.json")
        with open(trades_test_path, "w") as f:
            json.dump(trades_test, f, indent=2)

        summary["test"] = {
            "net_profit_pct": float(net_ret_pct_test),
            "max_drawdown_pct": float(-mdd_pct_test),
            "trades": len(trades_test),
            "long_trades": long_test,
            "short_trades": short_test,
            "equity_path": equity_test_path,
            "trades_path": trades_test_path,
        }

    # Write summary JSON
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ================================
    # Write SAT bundle
    # ================================
    default_bundle_dir = os.path.join(
        out_dir,
        f"lstm_longshort_{str(args.ticker).lower()}_{str(args.timeframe).lower()}",
    )
    bdir = os.path.abspath(os.path.expanduser(args.bundle_dir)) if args.bundle_dir else default_bundle_dir
    os.makedirs(bdir, exist_ok=True)

    # 1) TorchScript model -> bundle/model.pt
    target_model = os.path.join(bdir, "model.pt")
    shutil.copy2(mdl_path, target_model)

    # 2) Standardization stats as JSON -> bundle/preprocess.json
    pre_json_path = os.path.join(bdir, "preprocess.json")
    pre_dict = {
        "kind": "standard",
        "mean": np.asarray(mean).reshape(-1).astype(float).tolist(),
        "std": np.asarray(std).reshape(-1).astype(float).tolist(),
        "feature_names": list(feature_cols),
    }
    with open(pre_json_path, "w") as f:
        json.dump(pre_dict, f, indent=2)

    # 3) SAT-friendly meta.json -> bundle/meta.json
    meta = {
        "name": f"lstm_longshort_{args.ticker}_{args.timeframe}",
        "framework": "torch",
        "torchscript": True,
        "model_kind": "lstm_ichimoku_rsi",
        "input_layout": "NLC",  # (batch, length, channels)
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "ichimoku": {
            "tenkan": int(args.tenkan),
            "kijun": int(args.kijun),
            "senkou": int(args.senkou),
            "displacement": int(args.displacement),
        },
        "rsi": {
            "length": int(args.rsi_len),
            "bands_std": 2.0,
        },
        "n_features": len(feature_cols),
        "in_channels": len(feature_cols),
        "features": list(feature_cols),
        "preprocess_file": "preprocess.json",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pos_thr": args.pos_thr,
        "neg_thr": float(neg_thr),
        "prob_threshold": args.pos_thr,
        "short_prob_threshold": float(neg_thr),
        "risk": {
            "sl_pct": args.sl_pct,
            "tp_pct": args.tp_pct,
            "fee_bps": args.fee_bps,
        },
    }
    with open(os.path.join(bdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Final prints
    print("Saved artifacts:")
    print("  out_dir   :", out_dir)
    print("  model     :", mdl_path, "(TorchScript)")
    print("  ckpt      :", ckpt_path)
    print("  trainpred :", preds_train_path)
    if equity_val_path:
        print("  val_equity:", equity_val_path)
    if trades_val_path:
        print("  val_trades:", trades_val_path)
    if equity_test_path:
        print("  tst_equity:", equity_test_path)
    if trades_test_path:
        print("  tst_trades:", trades_test_path)
    print("  summary   :", summary_path)
    print("  bundle    :", bdir)
    print("    - model.pt")
    print("    - preprocess.json")
    print("    - meta.json")


if __name__ == "__main__":
    main()
