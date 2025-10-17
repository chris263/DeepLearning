#!/usr/bin/env python3
import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timezone

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Determinism helpers
# =========================
def set_global_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (even if you’re on CPU, it’s harmless)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# DB IO
# =========================
def load_price_json(dsn: str, ticker: str, timeframe: str) -> List[Dict[str, Any]]:
    con = psycopg2.connect(dsn)
    try:
        with con.cursor() as cur:
            cur.execute("SELECT price_json FROM price WHERE ticker=%s AND timeframe=%s", (ticker, timeframe))
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
    need = {"ts","open","high","low","close","volume"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"price_json missing keys: {need - set(df.columns)}")
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df.set_index("ts")

# =========================
# Indicators / Features
# =========================
def rolling_mid(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    hh = high.rolling(length, min_periods=length).max()
    ll = low.rolling(length, min_periods=length).min()
    return (hh + ll) / 2.0

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def add_ichimoku_features(df: pd.DataFrame,
                          tenkan_len=7, kijun_len=211, senkou_len=120, ema_len=172) -> pd.DataFrame:
    out = df.copy()

    tenkan  = rolling_mid(out["high"], out["low"], tenkan_len)
    kijun   = rolling_mid(out["high"], out["low"], kijun_len)
    senkouA = (tenkan + kijun) / 2.0
    senkouB = rolling_mid(out["high"], out["low"], senkou_len)
    ema_p   = ema(out["close"], ema_len)

    out["tenkan"]  = tenkan
    out["kijun"]   = kijun
    out["senkouA"] = senkouA
    out["senkouB"] = senkouB
    out["ema_p"]   = ema_p

    # Relative / regime features
    out["close_over_ema"]    = (out["close"] - ema_p) / out["close"]
    out["close_over_kijun"]  = (out["close"] - kijun) / out["close"]
    out["close_over_tenkan"] = (out["close"] - tenkan) / out["close"]
    out["cloud_thickness"]   = (senkouA - senkouB) / out["close"]
    out["above_cloud"]       = ((out["close"] > senkouA) & (out["close"] > senkouB)).astype(float)
    out["below_cloud"]       = ((out["close"] < senkouA) & (out["close"] < senkouB)).astype(float)
    out["tenkan_gt_kijun"]   = (tenkan > kijun).astype(float)

    # Volatility & slopes
    out["atr14"]       = atr(out, 14)
    out["atr_pct"]     = out["atr14"] / out["close"]
    out["ema_slope"]   = out["ema_p"].pct_change().fillna(0)
    out["kijun_slope"] = out["kijun"].pct_change().fillna(0)

    # Normalized cloud position
    denom = (out["senkouA"] - out["senkouB"]).replace(0, np.nan)
    out["cloud_pos"] = ((out["close"] - out["senkouB"]) / denom).clip(-2, 2).fillna(0)

    return out.dropna()

# =========================
# Labels (Triple-Barrier)
# =========================
def make_labels_triple_barrier(df: pd.DataFrame, horizon: int = 24, tp: float = 0.03, sl: float = 0.02) -> pd.Series:
    """
    Label = 1 if TP (= +tp) is hit before SL (= -sl) within 'horizon' bars; else 0.
    """
    close = df["close"].values
    idx = df.index
    y = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df) - horizon):
        entry = close[i]
        up = entry * (1 + tp)
        dn = entry * (1 - sl)
        path = close[i+1:i+1+horizon]
        hit_tp = np.where(path >= up)[0]
        hit_sl = np.where(path <= dn)[0]
        if hit_tp.size and (not hit_sl.size or hit_tp[0] < hit_sl[0]):
            y[i] = 1.0
        elif hit_sl.size and (not hit_tp.size or hit_sl[0] < hit_tp[0]):
            y[i] = 0.0
        else:
            y[i] = 0.0  # conservative
    return pd.Series(y, index=idx).dropna()

# =========================
# Windows & Normalization
# =========================
def build_windows(df_feat: pd.DataFrame, y: pd.Series, lookback: int = 64):
    common_index = df_feat.index.intersection(y.index)
    df_feat = df_feat.loc[common_index]
    y = y.loc[common_index]

    X_list, y_list, t_list = [], [], []
    for i in range(lookback, len(df_feat)):
        X_window = df_feat.iloc[i - lookback:i].values.T  # (C, L)
        X_list.append(X_window)
        y_list.append(int(y.iloc[i]))
        t_list.append(df_feat.index[i])

    X = np.array(X_list, dtype=np.float32)  # (N, C, L)
    y = np.array(y_list, dtype=np.int64)
    return X, y, t_list, df_feat.columns.tolist()

def fit_norm(X: np.ndarray):
    C = X.shape[1]
    means = np.zeros(C, dtype=np.float32)
    stds  = np.ones(C,  dtype=np.float32)
    for c in range(C):
        v = X[:, c, :].reshape(-1)
        means[c] = float(np.nanmean(v))
        s = float(np.nanstd(v))
        stds[c] = s if s > 1e-8 else 1.0
    return means, stds

def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    Y = X.copy()
    for c in range(X.shape[1]):
        Y[:, c, :] = (Y[:, c, :] - mean[c]) / std[c]
    return Y

# =========================
# Datasets / Model
# =========================
class SeqDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class CNN1D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.head(self.net(x))

def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3, wd=1e-4, patience=5, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    best = 1e9
    bad = 0
    best_state = None
    for ep in range(1, epochs + 1):
        # train
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

        # val
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

        if vl_loss < best:
            best = vl_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("[early-stop] patience exceeded.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# =========================
# Backtest (ATR SL/TP, time-stop, cooldown)
# =========================
@dataclass
class Trade:
    entry_t: pd.Timestamp
    entry_p: float
    exit_t: Optional[pd.Timestamp] = None
    exit_p: Optional[float] = None
    pnl_pct: Optional[float] = None

def backtest_long(
    df: pd.DataFrame,
    pred_proba: pd.Series,
    prob_threshold: float = 0.55,
    commission_pct: float = 0.1,
    slippage_bps: float = 3.0,
    atr_mult_sl: float = 2.0,
    atr_mult_tp: float = 4.0,
    time_stop: int = 48,
    cooldown: int = 6
) -> Dict[str, Any]:
    commission = commission_pct / 100.0
    slip = slippage_bps / 10000.0
    equity = 10_000.0
    position = 0
    entry_px = None
    entry_i = None
    trades: List[Trade] = []
    eq: List[Tuple[pd.Timestamp, float]] = []

    atr14 = atr(df, 14).reindex(df.index).fillna(method="bfill")
    idx = pred_proba.index
    cool_until: Optional[int] = None

    for k in range(len(idx) - 1):
        t = idx[k]
        t_next = idx[k + 1]
        i = df.index.get_loc(t)
        close = float(df.loc[t, "close"])
        # MTM
        if position == 1 and entry_px is not None:
            eq.append((t, equity * (1 + (close - entry_px) / entry_px)))
        else:
            eq.append((t, equity))

        p = float(pred_proba.iloc[k])
        next_open = float(df.loc[t_next, "open"])

        if position == 0:
            if (cool_until is None or i >= cool_until) and p > prob_threshold:
                fill = next_open * (1 + slip)
                fee = equity * commission
                equity -= fee
                entry_px = fill
                entry_i = i + 1
                position = 1
                trades.append(Trade(entry_t=t_next, entry_p=fill))
        else:
            a = float(atr14.iloc[i])
            atr_frac = a / max(entry_px, 1e-9)
            sl_px = entry_px * (1 - atr_mult_sl * atr_frac)
            tp_px = entry_px * (1 + atr_mult_tp * atr_frac)

            timed_out = (i - entry_i) >= time_stop
            flip = p <= (1.0 - prob_threshold)
            hit_sl = close <= sl_px
            hit_tp = close >= tp_px

            if timed_out or flip or hit_sl or hit_tp:
                fill = next_open * (1 - slip)
                ret = (fill - entry_px) / entry_px
                pnl = equity * ret
                fee = (equity + pnl) * commission
                equity = equity + pnl - fee
                tr = trades[-1]
                tr.exit_t = t_next
                tr.exit_p = fill
                tr.pnl_pct = ret * 100.0
                position = 0
                entry_px = None
                entry_i = None
                cool_until = i + cooldown

    last_t = idx[-1]
    last_close = float(df.loc[last_t, "close"])
    if position == 1 and entry_px is not None:
        ret = (last_close - entry_px) / entry_px
        pnl = equity * ret
        fee = (equity + pnl) * commission
        equity = equity + pnl - fee
        tr = trades[-1]
        tr.exit_t = last_t
        tr.exit_p = last_close
        tr.pnl_pct = ret * 100.0
    eq.append((last_t, equity))

    sells = sum(1 for t in trades if t.exit_t is not None)
    wins = sum(1 for t in trades if (t.exit_p is not None and t.exit_p > t.entry_p))
    losses = sells - wins
    net_profit_pct = (equity / 10_000.0 - 1.0) * 100.0
    eq_series = pd.Series([e for _, e in eq], index=[t for t, _ in eq], dtype=float)
    mdd = (eq_series / eq_series.cummax() - 1.0).min() * 100.0 if len(eq_series) else 0.0

    return {
        "trades": trades,
        "equity_curve": eq,
        "stats": {
            "trades": sells,
            "wins": wins,
            "losses": losses,
            "net_profit_pct": net_profit_pct,
            "max_drawdown_pct": mdd,
            "final_equity": equity,
        },
    }

# =========================
# Threshold picker (on internal validation within 2023–2024)
# =========================
def pick_threshold(df_feat: pd.DataFrame, proba_series: pd.Series, dd_cap: float = 0.25):
    best_th, best_eq = 0.55, -1e30
    for th in np.linspace(0.50, 0.80, 13):
        res = backtest_long(df_feat, proba_series, prob_threshold=th)
        dd = abs(res["stats"]["max_drawdown_pct"]) / 100.0
        if dd <= dd_cap and res["stats"]["final_equity"] > best_eq:
            best_eq = res["stats"]["final_equity"]
            best_th = th
    return best_th

# =========================
# Pipeline (Deterministic)
# =========================
def main():
    ap = argparse.ArgumentParser(description="Deterministic CNN Ichimoku: Train+Test 2023–2024, Validate on 2025 YTD.")
    ap.add_argument("--dsn", default=os.getenv("DATABASE_URL"))
    ap.add_argument("--ticker", default="ETHUSDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--tp", type=float, default=0.03)
    ap.add_argument("--sl", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dd_cap", type=float, default=0.25)
    ap.add_argument("--commission", type=float, default=0.1)
    ap.add_argument("--slippage_bps", type=float, default=3.0)
    ap.add_argument("--atr_sl", type=float, default=2.0)
    ap.add_argument("--atr_tp", type=float, default=4.0)
    ap.add_argument("--time_stop", type=int, default=48)
    ap.add_argument("--cooldown", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.dsn:
        print("ERROR: provide --dsn or set DATABASE_URL", file=sys.stderr)
        sys.exit(2)

    # Make runs reproducible
    set_global_seed(args.seed)

    # 1) Load & features
    raw = load_price_json(args.dsn, args.ticker.upper(), args.timeframe)
    df = json_to_df(raw)
    feat = add_ichimoku_features(df)

    # 2) Splits — per your request
    #    Train+Test period: 2023-01-01 → 2024-12-31
    #    Internal validation (for early stop & threshold): last 20% of that period
    #    OOS validation: 2025-01-01 → latest
    in_start = pd.Timestamp("2023-01-01", tz="UTC")
    in_end   = pd.Timestamp("2024-12-31", tz="UTC")
    oos_start= pd.Timestamp("2025-01-01", tz="UTC")

    in_mask  = (feat.index >= in_start) & (feat.index <= in_end)
    oos_mask = (feat.index >= oos_start)

    feat_in  = feat.loc[in_mask].copy()
    feat_oos = feat.loc[oos_mask].copy()

    # carve internal validation (time-based): last 20% of in-sample index
    n_in = len(feat_in)
    if n_in < 500:
        raise RuntimeError("Not enough in-sample data; ensure 2023–2024 exists in table.")
    split_idx = int(n_in * 0.8)
    feat_train = feat_in.iloc[:split_idx].copy()
    feat_val   = feat_in.iloc[split_idx:].copy()

    # 3) Labels on full feature space (triple barrier)
    y_all = make_labels_triple_barrier(feat, horizon=args.horizon, tp=args.tp, sl=args.sl)
    y_train = y_all.loc[feat_train.index.intersection(y_all.index)]
    y_val   = y_all.loc[feat_val.index.intersection(y_all.index)]
    y_test  = y_all.loc[feat_in.index.intersection(y_all.index)]   # in-sample backtest set
    y_oos   = y_all.loc[feat_oos.index.intersection(y_all.index)]  # 2025 YTD (may be shorter due to horizon)

    # 4) Windows
    Xtr, ytr, ttr, _ = build_windows(feat_train, y_train, lookback=args.lookback)
    Xva, yva, tva, _ = build_windows(feat_val,   y_val,   lookback=args.lookback)
    Xte, yte, tte, _ = build_windows(feat_in,    y_test,  lookback=args.lookback)  # full in-sample
    Xoo, yoo, too, _ = build_windows(feat_oos,   y_oos,   lookback=args.lookback)  # OOS

    if Xtr.size == 0 or Xva.size == 0 or Xte.size == 0 or Xoo.size == 0:
        raise RuntimeError("Windowing failed; try smaller --lookback or check data coverage.")

    # 5) Normalize with TRAIN stats only
    mean, std = fit_norm(Xtr)
    Xtr = apply_norm(Xtr, mean, std)
    Xva = apply_norm(Xva, mean, std)
    Xte = apply_norm(Xte, mean, std)
    Xoo = apply_norm(Xoo, mean, std)

    # 6) Datasets & deterministic DataLoaders
    tr_ds = SeqDS(Xtr, ytr)
    va_ds = SeqDS(Xva, yva)
    te_ds = SeqDS(Xte, yte)
    oo_ds = SeqDS(Xoo, yoo)

    # Use a generator with a fixed seed for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, generator=g)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

    # 7) Model & training
    in_channels = Xtr.shape[1]
    model = CNN1D(in_channels)
    model = train_model(model, tr_loader, va_loader,
                        epochs=args.epochs, lr=args.lr, wd=args.wd,
                        patience=args.patience, device=args.device)

    # 8) Predict on internal validation to choose threshold
    model.eval()
    with torch.no_grad():
        def predict_probs(ds: SeqDS, batch: int) -> np.ndarray:
            probs = []
            for i in range(0, len(ds), batch):
                xb = ds.X[i:i+batch].to(args.device)
                logits = model(xb)
                pr = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.append(pr)
            return np.concatenate(probs, axis=0)

        val_probs = predict_probs(va_ds, args.batch)
        te_probs  = predict_probs(te_ds, args.batch)
        oo_probs  = predict_probs(oo_ds, args.batch)

    val_proba_series = pd.Series(val_probs, index=tva, name="p_long")
    best_th = pick_threshold(feat, val_proba_series, dd_cap=args.dd_cap)
    if args.verbose:
        print(f"[deterministic] threshold tuned on internal val: {best_th:.3f}")

    # 9) In-sample backtest (2023–2024) with frozen model+threshold
    te_proba_series = pd.Series(te_probs, index=tte, name="p_long")
    res_in = backtest_long(
        df=feat, pred_proba=te_proba_series, prob_threshold=best_th,
        commission_pct=args.commission, slippage_bps=args.slippage_bps,
        atr_mult_sl=args.atr_sl, atr_mult_tp=args.atr_tp,
        time_stop=args.time_stop, cooldown=args.cooldown,
    )

    # 10) OOS validation (2025 YTD) with same frozen model+threshold
    oo_proba_series = pd.Series(oo_probs, index=too, name="p_long")
    res_oos = backtest_long(
        df=feat, pred_proba=oo_proba_series, prob_threshold=best_th,
        commission_pct=args.commission, slippage_bps=args.slippage_bps,
        atr_mult_sl=args.atr_sl, atr_mult_tp=args.atr_tp,
        time_stop=args.time_stop, cooldown=args.cooldown,
    )

    # 11) Report
    si = res_in["stats"]; so = res_oos["stats"]
    print("\n=== In-Sample (Train+Test) — 2023-01-01 → 2024-12-31 ===")
    print(f"Trades={si['trades']}  Wins={si['wins']}  Losses={si['losses']}")
    print(f"Net Profit %={si['net_profit_pct']:.2f}%  Max DD %={si['max_drawdown_pct']:.2f}%  Final Equity=${si['final_equity']:.2f}")

    print("\n=== Out-of-Sample Validation — 2025-01-01 → YTD ===")
    print(f"Trades={so['trades']}  Wins={so['wins']}  Losses={so['losses']}")
    print(f"Net Profit %={so['net_profit_pct']:.2f}%  Max DD %={so['max_drawdown_pct']:.2f}%  Final Equity=${so['final_equity']:.2f}")

    # Optional: show first few trades for OOS
    if res_oos["trades"]:
        print("\nOOS first trades:")
        for tr in res_oos["trades"][:5]:
            print(f"{tr.entry_t} @ {tr.entry_p:.2f}  ->  {tr.exit_t} @ {tr.exit_p:.2f}  pnl={tr.pnl_pct:.2f}%")

if __name__ == "__main__":
    main()
