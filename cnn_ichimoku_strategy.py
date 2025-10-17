#!/usr/bin/env python3
import os
import sys
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# PyTorch (CPU is fine)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# DB IO
# -----------------------------
def load_price_json(dsn: str, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
    con = psycopg2.connect(dsn)
    try:
        with con.cursor() as cur:
            cur.execute("SELECT price_json FROM price WHERE ticker=%s AND timeframe=%s", (ticker, timeframe))
            row = cur.fetchone()
            if not row or row[0] is None:
                raise RuntimeError(f"No data in price for ({ticker}, {timeframe}).")
            return row[0]
    finally:
        con.close()

def json_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    need = {"ts","open","high","low","close","volume"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"price_json missing keys: {need - set(df.columns)}")
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df.set_index("ts")

# -----------------------------
# Indicators (Ichimoku)
# -----------------------------
def rolling_mid(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    hh = high.rolling(length, min_periods=length).max()
    ll = low.rolling(length, min_periods=length).min()
    return (hh + ll) / 2.0

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def add_ichimoku_features(df: pd.DataFrame,
                          tenkan_len=7, kijun_len=211, senkou_len=120, ema_len=172) -> pd.DataFrame:
    tenkan  = rolling_mid(df["high"], df["low"], tenkan_len)
    kijun   = rolling_mid(df["high"], df["low"], kijun_len)
    senkouA = (tenkan + kijun) / 2.0
    senkouB = rolling_mid(df["high"], df["low"], senkou_len)
    ema_p   = ema(df["close"], ema_len)

    out = df.copy()
    out["tenkan"]  = tenkan
    out["kijun"]   = kijun
    out["senkouA"] = senkouA
    out["senkouB"] = senkouB
    out["ema_p"]   = ema_p

    # Relational / normalized features (scale-robust)
    out["close_over_ema"]    = (out["close"] - ema_p) / out["close"]
    out["close_over_kijun"]  = (out["close"] - kijun) / out["close"]
    out["close_over_tenkan"] = (out["close"] - tenkan) / out["close"]
    out["cloud_thickness"]   = (senkouA - senkouB) / out["close"]
    out["above_cloud"]       = ((out["close"] > senkouA) & (out["close"] > senkouB)).astype(float)
    out["below_cloud"]       = ((out["close"] < senkouA) & (out["close"] < senkouB)).astype(float)
    out["tenkan_gt_kijun"]   = (tenkan > kijun).astype(float)

    # Drop rows until indicators are valid
    out = out.dropna()
    return out

# -----------------------------
# Labels & windows
# -----------------------------
def make_labels(df: pd.DataFrame, horizon:int=6, pos_thr:float=0.004) -> pd.Series:
    """
    Label 1 if future return over 'horizon' bars > pos_thr (e.g., 0.4%)
    else 0. Long-only framing.
    """
    fut = df["close"].shift(-horizon)
    ret = (fut - df["close"]) / df["close"]
    y = (ret > pos_thr).astype(int)
    return y.dropna()

def build_windows(df_feat: pd.DataFrame, y: pd.Series, lookback:int=64) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Create (N, C, L) tensors from rolling windows, aligned so that
    each sample uses the past 'lookback' bars to predict y at t (already future-shifted).
    """
    common_index = df_feat.index.intersection(y.index)
    df_feat = df_feat.loc[common_index]
    y = y.loc[common_index]

    X_list = []
    y_list = []
    t_list = []
    feats = df_feat.columns.tolist()

    for i in range(lookback, len(df_feat)):
        X_window = df_feat.iloc[i-lookback:i].values.T  # shape (C, L)
        X_list.append(X_window)
        y_list.append(int(y.iloc[i]))   # label at current bar
        t_list.append(df_feat.index[i])

    X = np.array(X_list, dtype=np.float32)  # (N, C, L)
    y = np.array(y_list, dtype=np.int64)    # (N,)
    return X, y, t_list

# -----------------------------
# Normalization
# -----------------------------
def fit_norm(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Channel-wise z-score using training set only.
    X shape: (N, C, L)
    """
    # flatten over samples & time for each channel
    C = X.shape[1]
    means = []
    stds = []
    for c in range(C):
        v = X[:, c, :].reshape(-1)
        m = float(np.nanmean(v))
        s = float(np.nanstd(v) + 1e-8)
        means.append(m); stds.append(s)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)

def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Y = X.copy()
    for c in range(X.shape[1]):
        Y[:, c, :] = (Y[:, c, :] - mean[c]) / std[c]
    return Y

# -----------------------------
# Torch dataset/model
# -----------------------------
class SeqDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)      # (N, C, L)
        self.y = torch.from_numpy(y)      # (N,)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class CNN1D(nn.Module):
    def __init__(self, in_channels:int, seq_len:int):
        super().__init__()
        # small but expressive 1D CNN
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # logits for classes 0/1
        )
    def forward(self, x):  # x: (B, C, L)
        z = self.net(x)
        return self.head(z)

def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_state = None
    best_val = 1e9
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                vl_loss += float(loss.item()) * xb.size(0)
        vl_loss /= len(val_loader.dataset)

        print(f"[epoch {ep:02d}] train_loss={tr_loss:.4f} val_loss={vl_loss:.4f}")
        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -----------------------------
# Backtest (long-only)
# -----------------------------
@dataclass
class Trade:
    entry_t: pd.Timestamp
    entry_p: float
    exit_t: Optional[pd.Timestamp] = None
    exit_p: Optional[float] = None
    pnl_pct: Optional[float] = None

def backtest_long(df: pd.DataFrame,
                  pred_proba: pd.Series,
                  prob_threshold: float = 0.55,
                  commission_pct: float = 0.1,
                  slippage_bps: float = 3.0) -> Dict[str, Any]:
    """
    - Enter next bar open when pred_proba > threshold and flat
    - Exit when pred_proba <= 1 - threshold or when next opposite (simple symmetry), at next open
    - Commission each side, slippage each side
    """
    commission = commission_pct / 100.0
    slip = slippage_bps / 10000.0

    equity = 10_000.0
    position = 0
    entry_price = None
    trades: List[Trade] = []
    eq_curve = []

    idx = pred_proba.index
    for i in range(len(idx) - 1):
        t = idx[i]
        next_t = idx[i+1]
        close = float(df.loc[t, "close"])
        # mark-to-market
        if position == 1 and entry_price is not None:
            eq_curve.append((t, equity * (1 + (close - entry_price) / entry_price)))
        else:
            eq_curve.append((t, equity))

        p = float(pred_proba.iloc[i])
        next_open = float(df.loc[next_t, "open"])
        if position == 0 and p > prob_threshold:
            # enter long next open
            fill = next_open * (1 + slip)
            fee = equity * commission
            equity -= fee
            entry_price = fill
            position = 1
            trades.append(Trade(entry_t=next_t, entry_p=fill))
        elif position == 1 and p <= (1.0 - prob_threshold):
            # exit next open
            fill = next_open * (1 - slip)
            ret = (fill - entry_price) / entry_price
            pnl = equity * ret
            fee = (equity + pnl) * commission
            equity = equity + pnl - fee
            tr = trades[-1]
            tr.exit_t = next_t
            tr.exit_p = fill
            tr.pnl_pct = ret * 100.0
            position = 0
            entry_price = None

    # close at last close for reporting
    last_t = idx[-1]
    last_close = float(df.loc[last_t, "close"])
    if position == 1 and entry_price is not None:
        ret = (last_close - entry_price) / entry_price
        pnl = equity * ret
        fee = (equity + pnl) * commission
        equity = equity + pnl - fee
        tr = trades[-1]
        tr.exit_t = last_t
        tr.exit_p = last_close
        tr.pnl_pct = ret * 100.0

    eq_curve.append((last_t, equity))
    # stats
    sells = sum(1 for t in trades if t.exit_t is not None)
    wins = sum(1 for t in trades if (t.exit_p and t.exit_p > t.entry_p))
    losses = sells - wins
    net_profit_pct = (equity / 10_000.0 - 1.0) * 100.0

    eq_series = pd.Series([e for _, e in eq_curve], index=[t for t, _ in eq_curve], dtype=float)
    roll_max = eq_series.cummax()
    dd = (eq_series / roll_max - 1.0).fillna(0.0)
    max_dd_pct = dd.min() * 100.0 if len(dd) else 0.0

    return {
        "trades": trades,
        "equity_curve": eq_curve,
        "stats": {
            "trades": sells,
            "wins": wins,
            "losses": losses,
            "net_profit_pct": net_profit_pct,
            "max_drawdown_pct": max_dd_pct,
            "final_equity": equity,
        },
    }

# -----------------------------
# Pipeline
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="CNN Ichimoku strategy: train 2023-2024, test 2025 YTD from price table.")
    ap.add_argument("--dsn", default=os.getenv("DATABASE_URL"), help="postgres://user:pass@host:5432/db")
    ap.add_argument("--ticker", default="ETHUSDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--lookback", type=int, default=64, help="bars per training sample")
    ap.add_argument("--horizon", type=int, default=6, help="future bars for label (e.g., 6 = 1 day on 4h)")
    ap.add_argument("--pos_thr", type=float, default=0.004, help="future return threshold for label (e.g., 0.004=0.4%)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prob_threshold", type=float, default=0.55)
    ap.add_argument("--commission", type=float, default=0.1)
    ap.add_argument("--slippage_bps", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.dsn:
        print("ERROR: provide --dsn or set DATABASE_URL", file=sys.stderr)
        sys.exit(2)

    # 1) Load from DB
    raw = load_price_json(args.dsn, args.ticker.upper(), args.timeframe)
    df = json_to_df(raw)

    # 2) Build features
    feat = add_ichimoku_features(df)

    # 3) Train/test split by date
    train_mask = (feat.index >= pd.Timestamp("2023-01-01", tz="UTC")) & (feat.index <= pd.Timestamp("2024-12-31", tz="UTC"))
    test_mask  = (feat.index >= pd.Timestamp("2025-01-01", tz="UTC"))

    feat_train = feat.loc[train_mask].copy()
    feat_test  = feat.loc[test_mask].copy()

    # 4) Labels
    y_all = make_labels(feat, horizon=args.horizon, pos_thr=args.pos_thr)
    y_train = y_all.loc[feat_train.index.intersection(y_all.index)]
    y_test  = y_all.loc[feat_test.index.intersection(y_all.index)]

    # 5) Windows
    Xtr, ytr, ttr = build_windows(feat_train, y_train, lookback=args.lookback)
    Xte, yte, tte = build_windows(feat_test,  y_test,  lookback=args.lookback)

    if Xtr.size == 0 or Xte.size == 0:
        raise RuntimeError("Not enough data to make windows. Reduce lookback or horizon.")

    # 6) Normalize with train stats only
    mean, std = fit_norm(Xtr)
    Xtr = apply_norm(Xtr, mean, std)
    Xte = apply_norm(Xte, mean, std)

    # 7) Torch datasets
    tr_ds = SeqDS(Xtr, ytr)
    te_ds = SeqDS(Xte, yte)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    # small val split from train (time-split would be better; we’ll just take tail 10% as val)
    n_val = max(1, len(tr_ds)//10)
    val_ds = SeqDS(Xtr[-n_val:], ytr[-n_val:])
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # 8) Model
    in_channels = Xtr.shape[1]
    seq_len = Xtr.shape[2]
    model = CNN1D(in_channels, seq_len)

    # 9) Train
    model = train_model(model, tr_loader, val_loader, epochs=args.epochs, lr=args.lr, device=args.device)

    # 10) Predict on test windows
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(te_ds), args.batch):
            xb = te_ds.X[i:i+args.batch].to(args.device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:,1]  # prob of class 1 (long)
            all_probs.append(probs.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    proba_series = pd.Series(probs, index=tte, name="p_long")

    # 11) Backtest on 2025 YTD using proba series
    res = backtest_long(
        df=feat,  # same index, includes open/close
        pred_proba=proba_series,
        prob_threshold=args.prob_threshold,
        commission_pct=args.commission,
        slippage_bps=args.slippage_bps,
    )

    s = res["stats"]
    print("\n=== CNN Ichimoku Strategy — 2025 YTD Backtest ===")
    print(f"Symbol/TF       : {args.ticker}/{args.timeframe}")
    print(f"Train period    : 2023-01-01 → 2024-12-31")
    print(f"Test period     : 2025-01-01 → {feat.index.max().date()}")
    print(f"Train samples   : {len(tr_ds)}  | Test samples: {len(te_ds)}")
    print(f"Trades (closed) : {s['trades']}")
    print(f"Wins / Losses   : {s['wins']} / {s['losses']}")
    print(f"Net Profit %    : {s['net_profit_pct']:.2f}%")
    print(f"Max Drawdown %  : {s['max_drawdown_pct']:.2f}%")
    print(f"Final Equity    : ${s['final_equity']:.2f}")

    # show first few trades
    if res["trades"]:
        print("\nFirst trades:")
        for tr in res["trades"][:5]:
            print(f"{tr.entry_t} @ {tr.entry_p:.2f}  ->  {tr.exit_t} @ {tr.exit_p:.2f}  pnl={tr.pnl_pct:.2f}%")

if __name__ == "__main__":
    main()
