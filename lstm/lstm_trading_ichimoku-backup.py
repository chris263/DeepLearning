#!/usr/bin/env python3
"""
LSTM Ichimoku Trader (Classification + Backtest)
================================================

- Data source: Postgres table `price`, column `data_json` is a JSONB array of
  bars with keys: ts, open, high, low, close, volume (fallback to `timestamp`).
- Features: OHLCV-derived + Ichimoku lines/distances/slopes (no look-ahead).
- Model: LSTM classifier predicting prob(up) over a future horizon.
- Execution: decisions at bar t execute at next bar open (t+1),
  with intrabar SL/TP simulation using conservative stop-first priority.
- Split: train on [train_start, train_end]; evaluate on (train_end, test_end].
- Artifacts: model.pt (+ meta/scaler), predictions.csv, equity.csv, trades.json, summary.json.

Example
-------
export DATABASE_URL="postgresql://user:pass@host:5432/sat"
python lstm_trading_ichimoku.py \
  --ticker ETHUSDT --timeframe 4h \
  --train-start 2023-01-01 --train-end 2024-12-31 --test-end 2025-11-03 \
  --lookback 64 --horizon 4 --epochs 15 --batch-size 256 --pos-thr 0.55 \
  --sl-pct 0.02 --tp-pct 0.06 --fee-bps 1.0 --seed 7 --device cpu

Dependencies
------------
- Python 3.10+
- psycopg[binary]
- pandas, numpy, torch
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
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
# Data Loading & Utilities
# ==========================

def _coerce_timestamp_series(ts: pd.Series) -> pd.Series:
    """Coerce mixed timestamps to tz-naive pandas datetime.
    Accepts ISO strings or epoch seconds/milliseconds.
    """
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
    """Load OHLCV from Postgres `price.data_json` (JSON array of bars).

    Expected keys in each bar: ts, open, high, low, close, volume.
    Fallback to `timestamp` if `ts` key is not present in stored JSON.
    """
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
    # No forward shift to avoid look-ahead
    d["span_a"] = (d["tenkan"] + d["kijun"]) / 2.0
    d["span_b"] = rolling_mid(d.high, d.low, senkou)
    d["cloud_top"] = d[["span_a", "span_b"]].max(axis=1)
    d["cloud_bot"] = d[["span_a", "span_b"]].min(axis=1)
    return d


def build_features(df: pd.DataFrame, tenkan: int, kijun: int, senkou: int,
                   displacement: int, slope_window: int = 8) -> Tuple[pd.DataFrame, List[str]]:
    d = ichimoku(df, tenkan, kijun, senkou)
    # Basic
    d["px"] = d["close"]
    d["ret1"] = d["close"].pct_change().fillna(0)
    d["oc_diff"] = (d["close"] - d["open"]) / d["open"]
    d["hl_range"] = (d["high"] - d["low"]) / d["px"]
    d["logv"] = np.log1p(d["volume"])  # stabilize
    d["logv_chg"] = d["logv"].diff().fillna(0)

    # Ichimoku distances/slopes
    d["dist_px_cloud_top"] = (d["px"] - d["cloud_top"]) / d["px"]
    d["dist_px_cloud_bot"] = (d["px"] - d["cloud_bot"]) / d["px"]
    d["dist_tk_kj"] = (d["tenkan"] - d["kijun"]) / d["px"]
    d["span_order"] = (d["span_a"] > d["span_b"]).astype(float)

    sw = int(max(1, slope_window))
    d["tk_slope"] = (d["tenkan"] - d["tenkan"].shift(sw)) / (d["px"] + 1e-9)
    d["kj_slope"] = (d["kijun"] - d["kijun"].shift(sw)) / (d["px"] + 1e-9)
    d["span_a_slope"] = (d["span_a"] - d["span_a"].shift(sw)) / (d["px"] + 1e-9)
    d["span_b_slope"] = (d["span_b"] - d["span_b"].shift(sw)) / (d["px"] + 1e-9)

    # Chikou-like
    D = int(displacement)
    d["chikou_above"] = (d["close"] > d["close"].shift(D)).astype(float)

    # Realized vol (20 lookback)
    d["vol20"] = d["ret1"].rolling(20, min_periods=20).std().fillna(0)

    feature_cols = [
        # OHLCV
        "ret1", "oc_diff", "hl_range", "logv_chg",
        # Ichimoku distances/order
        "dist_px_cloud_top", "dist_px_cloud_bot", "dist_tk_kj", "span_order",
        # Slopes
        "tk_slope", "kj_slope", "span_a_slope", "span_b_slope",
        # Chikou, vol
        "chikou_above", "vol20",
    ]

    feat = d[feature_cols].copy().dropna().reset_index(drop=True)
    # Align timestamps to features
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
    """Build sequences ending at t with label from future horizon return sign.
       Returns X [N, lookback, F], y [N], and end timestamps for each sample.
    """
    X, y, t_end = [], [], []
    # Restrict to date window first
    mask = (df_feat["timestamp"] >= start_ts) & (df_feat["timestamp"] <= end_ts)
    d = df_feat.loc[mask].reset_index(drop=True)
    # We need indices relative to the full df_feat to compute label vs future close
    # Build a mapping from timestamp to row in the *full* df_feat
    ts_to_idx = {ts: i for i, ts in enumerate(df_feat["timestamp"]) }

    for i in range(len(d)):
        # i corresponds to some absolute idx j in full df_feat
        ts_i = d.loc[i, "timestamp"]
        j = ts_to_idx.get(ts_i)
        if j is None: continue
        if j - lookback + 1 < 0: continue
        if j + horizon >= len(df_feat): break
        # sequence window [j-lookback+1 .. j]
        seq = df_feat.loc[j - lookback + 1 : j, feature_cols].values.astype(np.float32)
        # label: future return from close_j to close_{j+horizon}
        c0 = float(df_feat.loc[j, "close"]) ; cH = float(df_feat.loc[j + horizon, "close"]) ; ret = (cH / c0) - 1.0
        label = 1.0 if ret > 0.0 else 0.0
        X.append(seq)
        y.append(label)
        t_end.append(df_feat.loc[j, "timestamp"])  # end time of the sequence

    if not X:
        return np.zeros((0, lookback, len(feature_cols)), np.float32), np.zeros((0,), np.float32), []

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    return X, y, t_end

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
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: [B, T, F]
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]  # [B, H]
        logit = self.head(h_last).squeeze(-1)  # [B]
        return logit

# ==============
# Training / Eval / Inference
# ==============

def class_weights(y: np.ndarray) -> float:
    # Return pos_weight for BCEWithLogits to counter class imbalance
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


def predict_proba(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            xb = torch.tensor(X[i:i+2048], dtype=torch.float32, device=device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0) if out else np.zeros((0,), np.float32)

# ==============
# Backtest (prob-threshold long/flat)
# ==============

def backtest_threshold(df_raw: pd.DataFrame, df_feat: pd.DataFrame, feature_cols: List[str],
                       model: nn.Module, mean: np.ndarray, std: np.ndarray,
                       lookback: int, pos_thr: float,
                       start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                       sl_pct: float, tp_pct: float, fee_bps: float,
                       device: str = "cpu") -> Tuple[pd.Series, List[Dict]]:
    # Build rolling sequences across the full feature df, but only trade inside window
    ts_to_idx = {ts: i for i, ts in enumerate(df_feat["timestamp"]) }

    equity = 10_000.0
    fee_mult = 1.0 - (fee_bps / 10_000.0)
    pos = 0; entry_px = 0.0

    eq_curve = []
    trades = []

    # Iterate over timestamps that are within [start_ts, end_ts]
    mask = (df_feat["timestamp"] >= start_ts) & (df_feat["timestamp"] <= end_ts)
    idxs = df_feat.index[mask].tolist()

    for k, j in enumerate(idxs):
        # Need lookback history
        if j - lookback + 1 < 0 or j + 1 >= len(df_feat):
            continue
        # Sequence ending at j uses features up to j
        seq = df_feat.loc[j - lookback + 1 : j, feature_cols].values[np.newaxis, :, :].astype(np.float32)
        seq = apply_scaler(seq, mean, std)

        # Decision at time j -> execute at open of j+1
        prob = float(predict_proba(model, seq, device=device)[0])
        ts_now = df_feat.loc[j, "timestamp"]
        j1 = j + 1
        o1 = float(df_raw.loc[j1, "open"]) ; h1 = float(df_raw.loc[j1, "high"]) ; l1 = float(df_raw.loc[j1, "low"]) ; c1 = float(df_raw.loc[j1, "close"]) ; ts1 = df_raw.loc[j1, "timestamp"]

        if pos == 0:
            if prob >= pos_thr:
                pos = 1
                entry_px = o1
                equity *= fee_mult
                trades.append({"entry_ts": str(ts1), "entry_px": entry_px, "prob": prob})
                eq_curve.append((ts1, equity))
                continue
            else:
                eq_curve.append((ts1, equity))
                continue

        # pos == 1 -> manage risk first
        sl = entry_px * (1.0 - sl_pct)
        tp = entry_px * (1.0 + tp_pct)
        hit_sl = (l1 <= sl)
        hit_tp = (h1 >= tp)

        exit_now = False; exit_px = None; reason = None
        if hit_sl and hit_tp:
            exit_now = True; exit_px = sl; reason = "SL"  # conservative stop-first
        elif hit_sl:
            exit_now = True; exit_px = sl; reason = "SL"
        elif hit_tp:
            exit_now = True; exit_px = tp; reason = "TP"

        if not exit_now:
            # Strategy exit on prob drop
            if prob < pos_thr:
                exit_now = True; exit_px = o1; reason = "SIG"

        if exit_now:
            qty = equity / entry_px
            pnl = (exit_px - entry_px) * qty
            equity += pnl
            equity *= fee_mult
            pos = 0
            trades[-1].update({"exit_ts": str(ts1), "exit_px": exit_px, "reason": reason, "pnl": pnl})
            eq_curve.append((ts1, equity))
            continue

        # still in position -> mark equity at end of bar
        eq_curve.append((ts1, equity))

    eq = pd.Series([v for _, v in eq_curve], index=[t for t, _ in eq_curve], name="equity")
    return eq, trades

# ==============
# CLI & Orchestration
# ==============

def parse_args():
    ap = argparse.ArgumentParser(description="LSTM Ichimoku trading classifier + backtest")
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

    # Trading
    ap.add_argument("--pos-thr", type=float, default=0.55)
    ap.add_argument("--sl-pct", type=float, default=0.02)
    ap.add_argument("--tp-pct", type=float, default=0.06)
    ap.add_argument("--fee-bps", type=float, default=1.0)

    return ap.parse_args()


def main():
    args = parse_args()
    if not args.db_url:
        raise SystemExit("Provide --db-url or set DATABASE_URL env var.")

    # 1) Load full window (train + test) for feature continuity
    df_all = load_db_ohlcv(args.db_url, ticker=args.ticker, timeframe=args.timeframe,
                           start=args.train_start, end=args.test_end or args.train_end)

    # 2) Build features (no look-ahead)
    feat_all, feature_cols = build_features(df_all, args.tenkan, args.kijun, args.senkou, args.displacement)

    # 3) Build train sequences (fit scaler ONLY on train window)
    train_start = pd.to_datetime(args.train_start)
    train_end = pd.to_datetime(args.train_end)
    X_train_raw, y_train, t_end_train = make_sequences(feat_all, feature_cols, args.lookback, args.horizon,
                                                       start_ts=train_start, end_ts=train_end)
    if len(X_train_raw) == 0:
        raise SystemExit("Not enough data to build training sequences. Reduce lookback/horizon or extend window.")

    # split train/val
    n = len(X_train_raw)
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    val_n = int(max(1, round(args.val_frac * n)))
    val_idx = idx[:val_n]; tr_idx = idx[val_n:]

    # Fit scaler on *train* only
    mean, std = fit_scaler(X_train_raw[tr_idx])
    X_train = apply_scaler(X_train_raw[tr_idx], mean, std)
    y_train = y_train[tr_idx]
    X_val = apply_scaler(X_train_raw[val_idx], mean, std)
    y_val = y_train = y_train  # keep a copy of y_train before overwrite? We'll recompute
    y_val = make_sequences(feat_all, feature_cols, args.lookback, args.horizon, start_ts=train_start, end_ts=train_end)[1][val_idx]

    # 4) Train model
    ds_tr = SeqDataset(X_train, y_train)
    ds_va = SeqDataset(X_val, y_val) if len(X_val) else None
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False) if ds_va else None

    model = LSTMClassifier(n_features=len(feature_cols), hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    pw = class_weights(y_train)
    print(f"Class balance: pos_rate={y_train.mean():.3f}, pos_weight={pw:.3f}")
    train_model(model, dl_tr, dl_va, epochs=args.epochs, lr=args.lr, device=args.device, pos_weight=pw, seed=args.seed)

    # Save model & scaler
    mdl_path = f"lstm_ichimoku_{args.ticker}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "mean": mean, "std": std,
        "tenkan": args.tenkan, "kijun": args.kijun, "senkou": args.senkou, "displacement": args.displacement,
        "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
    }, mdl_path)

    # 5) Evaluate on TRAIN window (predictions & summary)
    X_tr_raw, y_tr_all, t_tr = X_train_raw, make_sequences(feat_all, feature_cols, args.lookback, args.horizon, train_start, train_end)[1], make_sequences(feat_all, feature_cols, args.lookback, args.horizon, train_start, train_end)[2]
    X_tr = apply_scaler(X_tr_raw, mean, std)
    p_tr = predict_proba(model, X_tr, device=args.device)

    preds_train = pd.DataFrame({
        "timestamp": t_tr,
        "prob_up": p_tr,
        "label": y_tr_all,
    })
    preds_train.to_csv(f"lstm_ichimoku_preds_train_{args.ticker}.csv", index=False)

    # 6) Test window (if provided): predictions + backtest
    summary = {
        "ticker": args.ticker,
        "timeframe": args.timeframe,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "model_path": mdl_path,
    }

    if args.test_end:
        test_start = pd.to_datetime(args.train_end) + pd.Timedelta(seconds=1)
        test_end = pd.to_datetime(args.test_end)
        # Backtest with rolling threshold decisions
        eq, trades = backtest_threshold(df_all, feat_all, feature_cols, model, mean, std,
                                        lookback=args.lookback, pos_thr=args.pos_thr,
                                        start_ts=test_start, end_ts=test_end,
                                        sl_pct=args.sl_pct, tp_pct=args.tp_pct, fee_bps=args.fee_bps,
                                        device=args.device)
        if len(eq) > 0:
            ret = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
            mdd = float(((eq / eq.cummax()) - 1.0).min()) * 100.0
        else:
            ret = 0.0; mdd = 0.0
        eq.to_csv(f"lstm_ichimoku_equity_test_{args.ticker}.csv")
        with open(f"lstm_ichimoku_trades_{args.ticker}.json", "w") as f:
            json.dump(trades, f, indent=2)
        summary["test"] = {"net_profit_pct": ret, "max_drawdown_pct": -mdd, "trades": len(trades)}

    with open(f"lstm_ichimoku_summary_{args.ticker}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved artifacts:")
    for k, v in summary.items():
        if k != "test": print(" ", k, ":", v)
    if "test" in summary: print("  test:", summary["test"])


if __name__ == "__main__":
    main()
