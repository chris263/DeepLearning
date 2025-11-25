#!/usr/bin/env python3
"""
LSTM Ichimoku Trader — Long & Short (No Overlap)
================================================

This revision keeps your original pipeline/results **unchanged** but adjusts how
artifacts are saved so SAT can load the bundle directly:

- **model.pt** is now **TorchScript** (jit script → fallback to trace), so
  `torch.jit.load()` in SAT works without shipping Python class code.
- **preprocess.npz** includes `mean`, `std`, **and** `features` so SAT can
  standardize inputs exactly like training.
- **meta.json** uses SAT-friendly keys (`features`, `torchscript`, `input_layout`=NLC,
  `prob_threshold`, `short_prob_threshold`, plus the original `pos_thr`/`neg_thr`).
- Safe prints even when `--bundle-dir` isn't provided.

Everything else (training, backtest, thresholds, SL/TP, etc.) is intact.

Example (also writes an SAT-ready bundle):
  python lstm_trading_ichimoku_longshort.py \
    --ticker BTCUSDT --timeframe 1h \
    --train-start 2023-01-01 --train-end 2024-12-31 --test-end 2025-11-03 \
    --lookback 64 --horizon 4 --epochs 15 --batch-size 256 --pos-thr 0.55 \
    --sl-pct 0.02 --tp-pct 0.06 --fee-bps 1.0 --seed 7 --device cpu \
    --out-dir ./results/BTCUSDT_1h \
    --bundle-dir /app/app/analysis/models/lstm_longshort_btc_1h
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

import shutil, time

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
    """
    Load OHLCV from a JSON file (replaces the old Postgres-based version).

    The JSON file is expected to be a list of bars like:
        [
          {"ts": 1669487400000, "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...},
          ...
        ]

    `db_url` is now interpreted as the path to this JSON file.
    `ticker` and `timeframe` are kept only for API compatibility with existing callers.
    """
    import json
    import os

    json_path = os.path.abspath(os.path.expanduser(db_url))
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of bar objects.")

    df = pd.DataFrame(data)

    # Required columns based on your example
    required_cols = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"JSON missing required columns: {missing}")

    # Convert ts -> timestamp (datetime64[ns]) to match the original DB version
    df["timestamp"] = _coerce_timestamp_series(df["ts"]).astype("datetime64[ns]")

    # Coerce numeric columns
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep exactly the same final structure as the original version
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    # Date filters (on timestamp) – same semantics as before
    if start:
        df = df[df.timestamp >= pd.to_datetime(start)]
    if end:
        df = df[df.timestamp <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError("Bars exist but are outside the requested date window or all NaN.")

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

def make_sequences(df_feat: pd.DataFrame, feature_cols: List[str], lookback: int, horizon: int,
                   start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    X, y, t_end = [], [], []
    mask = (df_feat["timestamp"] >= start_ts) & (df_feat["timestamp"] <= end_ts)
    d = df_feat.loc[mask].reset_index(drop=True)
    ts_to_idx = {ts: i for i, ts in enumerate(df_feat["timestamp"]) }

    for i in range(len(d)):
        ts_i = d.loc[i, "timestamp"]
        j = ts_to_idx.get(ts_i)
        if j is None: continue
        if j - lookback + 1 < 0: continue
        if j + horizon >= len(df_feat): break
        seq = df_feat.loc[j - lookback + 1 : j, feature_cols].values.astype(np.float32)
        c0 = float(df_feat.loc[j, "close"])
        cH = float(df_feat.loc[j + horizon, "close"])
        ret = (cH / c0) - 1.0
        label = 1.0 if ret > 0.0 else 0.0
        X.append(seq); y.append(label); t_end.append(df_feat.loc[j, "timestamp"])

    if not X:
        return np.zeros((0, lookback, len(feature_cols)), np.float32), np.zeros((0,), np.float32), []
    return np.stack(X), np.array(y, dtype=np.float32), t_end

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==============
# Model
# ==============

class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=layers,
                            dropout=(dropout if layers > 1 else 0.0), batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.head(h_last).squeeze(-1)

# ==============
# Training / Eval
# ==============

def class_weights(y: np.ndarray) -> float:
    p = y.mean() if len(y) else 0.5
    if p <= 0: return 1.0
    return float((1 - p) / max(1e-6, p))

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader],
                epochs: int, lr: float, device: str, pos_weight: float, seed: int = 0) -> List[float]:
    torch.manual_seed(seed)
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    hist = []

    import copy
    best_val = float("inf")
    best_state = None
    best_epoch = None

    for ep in range(epochs):
        model.train(); loss_sum = 0.0; n = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb); n += len(xb)
        train_loss = loss_sum / max(1, n)

        if val_loader is not None:
            model.eval(); vloss = 0.0; vn = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    vloss += crit(model(xb), yb).item() * len(xb); vn += len(xb)
            val_loss = vloss / max(1, vn)
            hist.append((train_loss, val_loss))
            print(f"Epoch {ep+1}/{epochs} - train {train_loss:.4f} - val {val_loss:.4f}")

            # ---- NEW: track best validation epoch ----
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = ep + 1  # human-friendly (1-based)
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
    """
    equity = 1_000.0
    fee_mult = 1.0 - (fee_bps / 1_000.0)
    pos = 0
    entry_px = 0.0
    eq_curve = []
    trades: List[Dict] = []

    # Iterate only inside trading window
    mask = (df_feat["timestamp"] >= start_ts) & (df_feat["timestamp"] <= end_ts)
    idxs = df_feat.index[mask].tolist()

    for j in idxs:
        # Need lookback and the next bar (execution)
        if j - lookback + 1 < 0 or j + 1 >= len(df_feat):
            continue

        # Build one sequence ending at j
        seq = df_feat.loc[j - lookback + 1 : j, feature_cols].values[np.newaxis, :, :].astype(np.float32)
        seq = (seq - mean) / std
        prob = float(predict_proba(model, seq, device=device)[0])  # P(up)  (kept for debug if needed)

        ts_now = df_feat.loc[j, "timestamp"]
        j1 = j + 1
        o1 = float(df_raw.loc[j1, "open"])
        h1 = float(df_raw.loc[j1, "high"])
        l1 = float(df_raw.loc[j1, "low"])
        c1 = float(df_raw.loc[j1, "close"])
        ts1 = df_raw.loc[j1, "timestamp"]

        # ------ Build p_last / p_prev for fresh-cross logic ------
        # p_last: sequence ending at j
        seq_last = df_feat.loc[j - lookback + 1 : j, feature_cols].values[np.newaxis, :, :].astype(np.float32)
        seq_last = (seq_last - mean) / std
        p_last = float(predict_proba(model, seq_last, device=device)[0])

        # p_prev: sequence ending at j-1
        if j - lookback < 0:
            continue
        seq_prev = df_feat.loc[j - lookback : j - 1, feature_cols].values[np.newaxis, :, :].astype(np.float32)
        seq_prev = (seq_prev - mean) / std
        p_prev = float(predict_proba(model, seq_prev, device=device)[0])

        # ------ Flat: consider entries (fresh-cross, no overlap) ------
        if pos == 0:
            take_long  = (p_last >= pos_thr) and (p_prev <  p_last)
            take_short = (p_last <= neg_thr) and (p_prev >  p_last)
            if take_long:
                pos = +1
                entry_px = o1
                equity *= fee_mult
                trades.append({
                    "side": "LONG",
                    "entry_ts": str(ts1),
                    "entry_px": entry_px,
                    "p_prev": p_prev,
                    "p_last": p_last
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
                    "p_last": p_last
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
            # prefer TP like the live runner
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
                    "pnl": pnl
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
                    "pnl": pnl
                })
                eq_curve.append((ts1, equity))
                continue

            eq_curve.append((ts1, equity))
            continue

    # ===== NEW: convert curve to Series and return =====
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
    ap = argparse.ArgumentParser(description="LSTM Ichimoku trading classifier + backtest (long & short, no overlap)")
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="Postgres DSN; falls back to env DATABASE_URL")
    ap.add_argument("--ticker", required=True, help="Ticker (e.g., ETHUSDT)")
    ap.add_argument("--timeframe", required=True, help="Timeframe label (e.g., 4h, 1h)")

    ap.add_argument("--train-start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--train-end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--test-end", required=False, help="YYYY-MM-DD")

    # Ichimoku params
    ap.add_argument("--tenkan", type=int, default=9)
    ap.add_argument("--kijun", type=int, default=26)
    ap.add_argument("--senkou", type=int, default=52)
    ap.add_argument("--displacement", type=int, default=26)

    # Model/Training
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
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
    ap.add_argument("--bundle-dir", type=str, help="If set, also write SAT bundle (model.pt, preprocess.npz, meta.json)")

    return ap.parse_args()

def main():
    args = parse_args()
    if not args.db_url:
        raise SystemExit("Provide --db-url or set DATABASE_URL env var.")
    neg_thr = (1.0 - args.pos_thr) if args.neg_thr is None else float(args.neg_thr)
    if neg_thr >= args.pos_thr:
        print(f"[warn] neg_thr ({neg_thr}) >= pos_thr ({args.pos_thr}); tightening to avoid overlap.")
        neg_thr = max(1e-6, min(neg_thr, args.pos_thr - 1e-6))

    # Prepare output directory
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    def OP(name: str) -> str:
        return os.path.join(out_dir, name)

    # 1) Load full window (train + (optional) test)
    df_all = load_db_ohlcv(args.db_url, ticker=args.ticker, timeframe=args.timeframe,
                           start=args.train_start, end=args.test_end or args.train_end)

    # 2) Build features (no look-ahead)
    feat_all, feature_cols = build_features(df_all, args.tenkan, args.kijun, args.senkou, args.displacement)

    # 3) Train sequences (fit scaler on train window only)
    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    X_train_raw, y_train_all, t_end_train = make_sequences(
        feat_all, feature_cols, args.lookback, args.horizon, start_ts=train_start, end_ts=train_end
    )
    if len(X_train_raw) == 0:
        raise SystemExit("Not enough data to build training sequences. Reduce lookback/horizon or extend window.")

    # Split (random holdout on sequences, not shuffling time)
    n = len(X_train_raw)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    val_n = int(max(1, round(args.val_frac * n)))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    mean, std = fit_scaler(X_train_raw[tr_idx])
    X_train = apply_scaler(X_train_raw[tr_idx], mean, std)
    y_train = y_train_all[tr_idx]
    X_val = apply_scaler(X_train_raw[val_idx], mean, std)
    y_val = y_train_all[val_idx]

    # 4) Train
    ds_tr = SeqDataset(X_train, y_train)
    ds_va = SeqDataset(X_val, y_val) if len(X_val) else None
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False) if ds_va else None

    model = LSTMClassifier(n_features=len(feature_cols), hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    pw = class_weights(y_train)
    print(f"Class balance: pos_rate={y_train.mean():.3f}, pos_weight={pw:.3f}")
    train_model(model, dl_tr, dl_va, epochs=args.epochs, lr=args.lr, device=args.device, pos_weight=pw, seed=args.seed)

    # ================================
    # Save artifacts (TorchScript model + scaler)
    # ================================
    # Always export a CPU TorchScript for maximum portability in SAT.
    model_cpu = model.to("cpu").eval()
    example = torch.zeros(1, args.lookback, len(feature_cols), dtype=torch.float32)
    try:
        scripted = torch.jit.script(model_cpu)
    except Exception:
        scripted = torch.jit.trace(model_cpu, example)
    mdl_path = OP(f"lstm_ichimoku_{args.ticker}.pt")
    scripted.save(mdl_path)

    # (Optional) also save a training checkpoint for your own experiments
    ckpt_path = OP(f"lstm_ichimoku_{args.ticker}.ckpt")
    torch.save({
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "mean": mean, "std": std,
        "tenkan": args.tenkan, "kijun": args.kijun, "senkou": args.senkou, "displacement": args.displacement,
        "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
    }, ckpt_path)

    # 5) Train preds (optional artifact)
    X_tr = apply_scaler(X_train_raw, mean, std)
    p_tr = predict_proba(model, X_tr, device="cpu")  # model is on CPU now
    preds_train = pd.DataFrame({"timestamp": t_end_train, "prob": p_tr, "label": y_train_all})
    preds_train_path = OP(f"lstm_ichimoku_preds_train_{args.ticker}.csv")
    preds_train.to_csv(preds_train_path, index=False)

    # 6) Test backtest (if test_end provided)
    summary = {
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "model_path": mdl_path,
        "pos_thr": args.pos_thr,
        "neg_thr": float((1.0 - args.pos_thr) if args.neg_thr is None else args.neg_thr),
        "out_dir": out_dir,
    }

    equity_path = None
    trades_path = None
    summary_path = OP(f"lstm_ichimoku_summary_{args.ticker}.json")

    if args.test_end:
        test_start = pd.to_datetime(args.train_end) + pd.Timedelta(seconds=1)
        test_end = pd.to_datetime(args.test_end)

        eq, trades = backtest_threshold_ls(
            df_all, feat_all, feature_cols, model_cpu, mean, std,
            lookback=args.lookback, pos_thr=args.pos_thr, neg_thr=summary["neg_thr"],
            start_ts=test_start, end_ts=test_end,
            sl_pct=args.sl_pct, tp_pct=args.tp_pct, fee_bps=args.fee_bps,
            device="cpu"
        )

        if len(eq) > 0:
            net_ret_pct = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
            mdd_pct = float(((eq / eq.cummax()) - 1.0).min()) * 100.0
        else:
            net_ret_pct = 0.0; mdd_pct = 0.0

        # Side stats
        long_trades = sum(1 for t in trades if t.get("side") == "LONG" and "exit_px" in t)
        short_trades = sum(1 for t in trades if t.get("side") == "SHORT" and "exit_px" in t)

        equity_path = OP(f"lstm_ichimoku_equity_test_{args.ticker}.csv")
        eq.to_csv(equity_path)

        trades_path = OP(f"lstm_ichimoku_trades_{args.ticker}.json")
        with open(trades_path, "w") as f:
            json.dump(trades, f, indent=2)

        summary["test"] = {
            "net_profit_pct": float(net_ret_pct),
            "max_drawdown_pct": float(-mdd_pct),
            "trades": len(trades),
            "long_trades": long_trades,
            "short_trades": short_trades,
            "equity_path": equity_path,
            "trades_path": trades_path,
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ================================
    # Write SAT bundle (always) — to --bundle-dir or default under out_dir
    # ================================
    # If bundle_dir not provided, default to out_dir/lstm_longshort_<ticker>_<timeframe>/
    default_bundle_dir = os.path.join(
        out_dir, f"lstm_longshort_{str(args.ticker).lower()}_{str(args.timeframe).lower()}"
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
        "std":  np.asarray(std).reshape(-1).astype(float).tolist(),
        "feature_names": list(feature_cols),
    }
    with open(pre_json_path, "w") as f:
        json.dump(pre_dict, f, indent=2)

    # 3) SAT-friendly meta.json -> bundle/meta.json
    meta = {
        "name": f"lstm_longshort_{args.ticker}_{args.timeframe}",
        "framework": "torch",
        "torchscript": True,
        "model_kind": "lstm_ichimoku",
        "input_layout": "NLC",  # sequences as (batch, length, channels)
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "ichimoku": {
            "tenkan": int(args.tenkan),
            "kijun": int(args.kijun),
            "senkou": int(args.senkou),
            "displacement": int(args.displacement)
        },
        "n_features": len(feature_cols),
        "in_channels": len(feature_cols),
        "features": list(feature_cols),
        "preprocess_file": "preprocess.json",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pos_thr": args.pos_thr,
        "neg_thr": summary["neg_thr"],
        "prob_threshold": args.pos_thr,
        "short_prob_threshold": summary["neg_thr"],
        "risk": {"sl_pct": args.sl_pct, "tp_pct": args.tp_pct, "fee_bps": args.fee_bps},
    }
    with open(os.path.join(bdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


    print("Saved artifacts:")
    print("  out_dir   :", out_dir)
    print("  model     :", mdl_path, "(TorchScript)")
    print("  ckpt      :", ckpt_path)
    print("  trainpred :", preds_train_path)
    if equity_path: print("  equity    :", equity_path)
    if trades_path: print("  trades    :", trades_path)
    print("  summary   :", summary_path)
    print("  bundle    :", bdir)
    print("    - model.pt")
    print("    - preprocess.json")
    print("    - meta.json")


if __name__ == "__main__":
    main()
