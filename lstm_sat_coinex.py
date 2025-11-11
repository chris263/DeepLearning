#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Ichimoku runner (CoinEx, USDT linear swap) that:
- Loads bars from a JSON file (no DB calls).
- Uses pub/secret *variable names* to pick credentials from ~/.ssh/coinex_keys.env
- Handles SL/TP exits using values from meta.json (sl_pct, tp_pct), including meta["risk"] fallback.
- Avoids pyramiding; flips positions when signal reverses.
- Idempotent per broker/symbol/timeframe/category to avoid cross-broker blocking.
"""

import os, sys, json, time, argparse, pathlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
except Exception as e:
    print("Please install torch:  pip install torch", file=sys.stderr)
    raise

try:
    import ccxt
except Exception as e:
    print("Please install ccxt:  pip install ccxt", file=sys.stderr)
    raise

# =========================
# Config / constants
# =========================
LOOKBACK_MIN = 8
SIX_MIN_MS = 6 * 60 * 1000
TF_TO_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}
def tf_ms(tf: str) -> int:
    if tf not in TF_TO_MS: raise ValueError(f"Unsupported timeframe '{tf}'")
    return TF_TO_MS[tf]

# =========================
# JSON loader — robust to common formats
# =========================
def _normalize_bar(d: Dict) -> Optional[Dict]:
    if not isinstance(d, dict):
        return None
    # Timestamp
    ts = (
        d.get("ts") or d.get("t") or d.get("time") or d.get("open_time") or d.get("openTime") or d.get("timestamp")
    )
    if ts is None:
        return None
    # Convert ISO -> ms if needed
    if isinstance(ts, str):
        try:
            ts_dt = pd.to_datetime(ts, utc=True)
            ts_ms = int(ts_dt.value // 1_000_000)  # ns -> ms
        except Exception:
            return None
    else:
        try:
            ts_f = float(ts)
            ts_ms = int(ts_f if ts_f > 10_000_000_000 else ts_f * 1000)  # sec->ms heuristic
        except Exception:
            return None

    # Prices/volume
    def get_num(*keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                try: return float(d[k])
                except Exception: pass
        return default

    o = get_num("open", "o")
    h = get_num("high", "h")
    l = get_num("low", "l")
    c = get_num("close", "c")
    v = get_num("volume", "v", default=0.0)

    if None in (o, h, l, c):
        return None
    return {"ts": ts_ms, "open": o, "high": h, "low": l, "close": c, "volume": v}

def load_bars_from_json(path: str) -> pd.DataFrame:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"Bars JSON not found: {p}")

    content = p.read_text().strip()
    bars: List[Dict] = []

    # Try array or dict first
    try:
        obj = json.loads(content)
        if isinstance(obj, list):
            bars = obj
        elif isinstance(obj, dict):
            # common wrappers
            for key in ("data", "bars", "result", "items"):
                if key in obj and isinstance(obj[key], list):
                    bars = obj[key]
                    break
            if not bars and "price" in obj and isinstance(obj["price"], list):
                bars = obj["price"]
    except Exception:
        # Fallback NDJSON
        bars = []
        for line in content.splitlines():
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                bars.append(rec)
            except Exception:
                pass

    norm = []
    for b in bars:
        nb = _normalize_bar(b)
        if nb: norm.append(nb)

    if not norm:
        raise SystemExit(f"No valid bars decoded from {p}")

    df = (pd.DataFrame(norm)
            .dropna()
            .sort_values("ts")
            .reset_index(drop=True))
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    return df[["timestamp", "ts", "open", "high", "low", "close", "volume"]]

# =========================
# Ichimoku features (align with trainer)
# =========================
def rolling_mid(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    n = int(n)
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return (hh + ll) / 2.0

def ichimoku(df: pd.DataFrame, tenkan: int, kijun: int, senkou: int) -> pd.DataFrame:
    d = df.copy()
    d["tenkan"] = rolling_mid(d.high, d.low, tenkan)
    d["kijun"]  = rolling_mid(d.high, d.low, kijun)
    d["span_a"] = (d["tenkan"] + d["kijun"]) / 2.0
    d["span_b"] = rolling_mid(d.high, d.low, senkou)
    d["cloud_top"] = d[["span_a", "span_b"]].max(axis=1)
    d["cloud_bot"] = d[["span_a", "span_b"]].min(axis=1)
    return d

# --- Feature synonyms so runner matches meta["features"] exactly ---
_SYNONYMS = {
    "ret1": ["ret_1", "r1", "return1"],
    "oc_diff": ["ocdiff", "oc_change"],
    "hl_range": ["hlrange", "high_low_range"],
    "logv_chg": ["logv_change", "dlogv", "logv_diff"],
    "dist_px_cloud_top": ["dist_px_cloudtop", "dist_px_cloudTop"],
    "dist_px_cloud_bot": ["dist_px_cloudbot", "dist_px_cloudBottom"],
    "dist_tk_kj": ["dist_tk_kijun", "tk_kj_dist"],
    "span_order": ["spanOrder", "span_order_flag"],
    "tk_slope": ["tenkan_slope", "tkSlope"],
    "kj_slope": ["kijun_slope", "kjSlope"],
    "span_a_slope": ["spana_slope", "spanA_slope"],
    "span_b_slope": ["spanb_slope", "spanB_slope"],
    "chikou_above": ["chikouAbove", "chikou_flag"],
    "vol20": ["vol_20", "volatility20"],
}
def align_features_to_meta(feat_df: pd.DataFrame, meta_features: List[str]) -> pd.DataFrame:
    cols = set(feat_df.columns)
    for name in meta_features:
        if name in cols: 
            continue
        for cand in _SYNONYMS.get(name, []):
            if cand in cols:
                feat_df[name] = feat_df[cand]
                cols.add(name)
                break
    return feat_df

def build_features(df: pd.DataFrame, tenkan: int, kijun: int, senkou: int,
                   displacement: int, slope_window: int = 8) -> pd.DataFrame:
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
    d["tk_slope"]     = (d["tenkan"] - d["tenkan"].shift(sw)) / (d["px"] + 1e-9)
    d["kj_slope"]     = (d["kijun"]  - d["kijun"].shift(sw))  / (d["px"] + 1e-9)
    d["span_a_slope"] = (d["span_a"] - d["span_a"].shift(sw)) / (d["px"] + 1e-9)
    d["span_b_slope"] = (d["span_b"] - d["span_b"].shift(sw)) / (d["px"] + 1e-9)

    D = int(displacement)
    d["chikou_above"] = (d["close"] > d["close"].shift(D)).astype(float)
    d["vol20"] = d["ret1"].rolling(20, min_periods=20).std().fillna(0)

    d = d.reset_index(drop=True)
    d["timestamp"] = df["timestamp"].values
    return d

# =========================
# Bundle I/O
# =========================
def load_bundle(model_dir: str):
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    meta_path = os.path.join(model_dir, "meta.json")
    pre_path  = os.path.join(model_dir, "preprocess.json")
    mdl_path  = os.path.join(model_dir, "model.pt")
    for p in (meta_path, pre_path, mdl_path):
        if not os.path.exists(p):
            raise SystemExit(f"Missing bundle file: {p}")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    with open(pre_path, "r") as f:
        pre = json.load(f)

    feature_names = meta.get("features")
    if not feature_names:
        raise SystemExit("feature_names missing in preprocess/meta.")
    lookback = int(meta.get("lookback", 64))
    if lookback < LOOKBACK_MIN:
        raise SystemExit(f"lookback too small: {lookback}")

    pos_thr = float(meta.get("pos_thr", meta.get("prob_threshold", 0.55)))
    neg_thr = float(meta.get("neg_thr", meta.get("short_prob_threshold", 1.0 - pos_thr)))

    # SL/TP (fractions), also allow nested under risk{}
    sl_pct = meta.get("sl_pct", meta.get("stop_loss_pct", None))
    tp_pct = meta.get("tp_pct", meta.get("take_profit_pct", None))
    risk = meta.get("risk") or {}
    if sl_pct is None:
        sl_pct = risk.get("sl_pct", risk.get("stop_loss_pct", None))
    if tp_pct is None:
        tp_pct = risk.get("tp_pct", risk.get("take_profit_pct", None))
    sl_pct = None if sl_pct is None else float(sl_pct)
    tp_pct = None if tp_pct is None else float(tp_pct)

    mean = np.array(pre["mean"], dtype=np.float32).reshape(1, 1, -1)
    std  = np.array(pre["std"],  dtype=np.float32).reshape(1, 1, -1)
    std = np.where(std < 1e-8, 1.0, std)

    model = torch.jit.load(mdl_path, map_location="cpu").eval()

    ichimoku_params = {
        "tenkan":   int(meta.get("ichimoku", {}).get("tenkan", 7)),
        "kijun":    int(meta.get("ichimoku", {}).get("kijun", 211)),
        "senkou":   int(meta.get("ichimoku", {}).get("senkou", 120)),
        "displacement": int(meta.get("ichimoku", {}).get("displacement", 41)),
    }

    return {
        "model": model,
        "feature_names": feature_names,
        "lookback": lookback,
        "pos_thr": pos_thr,
        "neg_thr": neg_thr,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "mean": mean,
        "std": std,
        "ichimoku": ichimoku_params,
        "meta": meta,
        "paths": {"dir": model_dir},
    }

# =========================
# Sequences & prediction
# =========================
def to_sequences_latest(feat_df: pd.DataFrame, cols: List[str], lookback: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    if len(feat_df) < lookback + 1:
        return np.zeros((0, lookback, len(cols)), np.float32), []
    j_last = len(feat_df) - 1
    j_prev = j_last - 1
    X, ts = [], []
    for j in (j_prev, j_last):
        start = j - lookback + 1
        if start < 0: continue
        seq = feat_df.loc[start:j, cols].values.astype(np.float32)
        X.append(seq); ts.append(feat_df.loc[j, "timestamp"])
    if not X: return np.zeros((0, lookback, len(cols)), np.float32), []
    X = np.stack(X, axis=0)
    return X, ts

def predict_proba(model, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if len(X) == 0: 
        return np.zeros((0,), np.float32)
    with torch.no_grad():
        Xn = (X - mean) / std
        out = torch.sigmoid(model(torch.tensor(Xn))).cpu().numpy().reshape(-1)
    return out.astype(np.float32)

# =========================
# CoinEx: keys, exchange, helpers
# =========================
def read_coinex_env(pub_key_name: Optional[str] = None,
                    sec_key_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Read API credentials from ~/.ssh/coinex_keys.env by variable *names*.

    Example file lines:
      API_KEY_ETH=pk_123
      API_SECRET_ETH=sk_abc

    CLI:
      --pub_key API_KEY_ETH --sec_key API_SECRET_ETH
    """
    pub_key_name = pub_key_name or "API_KEY_ETH"
    sec_key_name = sec_key_name or "API_SECRET_ETH"

    path = pathlib.Path("~/.ssh/coinex_keys.env").expanduser()
    if not path.exists():
        raise SystemExit(f"Keys file not found: {path}")

    envmap: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        envmap[k.strip()] = v.strip().strip('"').strip("'")

    api_key = envmap.get(pub_key_name)
    api_secret = envmap.get(sec_key_name)
    if not api_key:
        raise SystemExit(f"Public key name '{pub_key_name}' not found in {path}")
    if not api_secret:
        raise SystemExit(f"Secret key name '{sec_key_name}' not found in {path}")

    return api_key, api_secret

def coinex_exchange(api_key: str, api_secret: str):
    # Linear USDT swaps
    ex = ccxt.coinex({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    ex.load_markets()
    return ex

def resolve_symbol(ex, ticker: str) -> str:
    base = (ticker.replace("USDT","")
                  .replace("/USDT","")
                  .replace(":USDT","")
                  .replace("/","")
                  .upper())
    candidates = [
        f"{base}/USDT:USDT",
        f"{base}USDT:USDT",
        f"{base}/USDT",
        f"{base}USDT",
    ]
    symbols = set(ex.symbols or [])
    markets = ex.markets or {}; ids = set(markets.keys())
    for c in candidates:
        if c in symbols: return c
        if c in ids: return markets[c].get("symbol") or c
    for s in symbols:
        if s.upper().startswith(base + "/") and s.upper().endswith(("USDT", "USDT:USDT")):
            return s
    raise SystemExit(f"Cannot resolve market for {ticker} on {ex.id}")

def fetch_usdt_balance(ex, acct_type: str) -> float:
    params = {"type": acct_type} if acct_type else {}
    bal = ex.fetch_balance(params)
    usdt = bal.get("USDT") or {}
    free = float(usdt.get("free") or 0)
    total = float(usdt.get("total") or 0)
    return free if free > 0 else total

def ensure_swap_funds(ex, needed_usdt: float, auto_transfer: bool, transfer_buffer: float, debug: bool) -> float:
    swap_before = fetch_usdt_balance(ex, "swap")
    if debug:
        spot_before = fetch_usdt_balance(ex, "spot")
        print(f"[DEBUG] balances before — swap={swap_before:.8f} USDT, spot={spot_before:.8f} USDT")

    if swap_before >= needed_usdt:
        return swap_before

    if not auto_transfer:
        return swap_before

    try:
        spot_before = fetch_usdt_balance(ex, "spot")
        delta = max(0.0, (needed_usdt * (1.0 + transfer_buffer)) - swap_before)
        amt = min(delta, spot_before)
        if amt > 0:
            if debug:
                print(f"[DEBUG] transferring {amt:.8f} USDT spot → swap")
            ex.transfer(code="USDT", amount=float(f"{amt:.6f}"), fromAccount="spot", toAccount="swap")
            time.sleep(0.5)
    except Exception as e:
        print(f"[WARN] transfer failed: {e}")

    return fetch_usdt_balance(ex, "swap")

def amount_to_precision(ex, symbol: str, amount: float) -> float:
    try:
        return float(ex.amount_to_precision(symbol, amount))
    except Exception:
        return float(f"{amount:.6f}")

def price_to_precision(ex, symbol: str, price: float) -> float:
    try:
        return float(ex.price_to_precision(symbol, price))
    except Exception:
        return float(f"{price:.6f}")

# =========================
# Idempotency helpers (per-broker/symbol/timeframe/category)
# =========================
def last_executed_guard(model_dir: str, suffix: Optional[str] = None) -> Tuple[Optional[int], str]:
    fname = ".last_order_ts.txt" if not suffix else f".last_order_ts_{suffix}.txt"
    path = os.path.join(model_dir, fname)
    if os.path.exists(path):
        try:
            v = int(open(path, "r").read().strip() or "0")
        except Exception:
            v = None
    else:
        v = None
    return v, path

def write_last_executed(path: str, ts_ms: int):
    with open(path, "w") as f:
        f.write(str(int(ts_ms)))

# =========================
# Time helpers
# =========================
def resolve_last_closed(now_ms: int, last_ts_ms: int, timeframe: str) -> Tuple[Optional[int], str, Optional[int]]:
    step = tf_ms(timeframe)
    candidates = [(last_ts_ms, "close_stamp"), (last_ts_ms + step, "open_stamp")]
    valid = [(c, tag, now_ms - c) for (c, tag) in candidates if now_ms >= c]
    if not valid:
        return None, "future", None
    c, tag, age = min(valid, key=lambda x: x[2])
    return c, tag, age

# =========================
# Core flow
# =========================
def decide_and_maybe_trade(args):
    # 1) Load bundle & bars
    bundle = load_bundle(args.model_dir)
    meta = bundle["meta"]
    model = bundle["model"]
    feats = bundle["feature_names"]
    lookback = bundle["lookback"]
    pos_thr, neg_thr = bundle["pos_thr"], bundle["neg_thr"]
    sl_pct, tp_pct   = bundle["sl_pct"], bundle["tp_pct"]
    mean, std = bundle["mean"], bundle["std"]
    ik = bundle["ichimoku"]
    model_dir = bundle["paths"]["dir"]

    # Allow other bundles: resolve ticker/timeframe from meta or CLI overrides
    ticker = args.ticker or meta.get("ticker") or "BTCUSDT"
    timeframe = args.timeframe or meta.get("timeframe") or "1h"

    # Broker category: spot for longs, futures for shorts (routing is handled elsewhere)
    category = (args.category or "spot").lower().strip()

    # Load bars from DB
    df = load_bars(args.db_url, ticker=ticker, timeframe=timeframe)
    if df is None or len(df) < (lookback + 3):
        print("Not enough bars to build features.")
        return

    # 2) Build features frame and standardize
    base = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    feat_df_full = build_features(
        base.rename(columns={"timestamp":"ts"}),
        ik["tenkan"], ik["kijun"], ik["senkou"], ik["displacement"]
    )

    # Guard: ensure all required features exist
    for c in feats:
        if c not in feat_df_full.columns:
            raise SystemExit(f"Feature '{c}' missing in computed frame.")

    feat_df = feat_df_full.copy()
    # Standardize using saved mean/std
    X = feat_df[feats].astype("float32").to_numpy()
    X = (X - mean) / std
    X = X[-(lookback+2):]  # last lookback + prev bar
    if X.shape[0] < (lookback + 2):
        print("Not enough bars to form sequences yet.")
        return

    # Build overlapping sequences for last two bars (prev, last)
    seq_prev = X[-(lookback+2):-(1)]          # bars [.. lookback + prev]
    seq_last = X[-(lookback+1):]              # bars [.. lookback + last]
    X_prev = seq_prev.reshape(1, lookback, -1)
    X_last = seq_last.reshape(1, lookback, -1)

    # Inference
    with torch.no_grad():
        p_prev = float(model(torch.from_numpy(X_prev)).sigmoid().item())
        p_last = float(model(torch.from_numpy(X_last)).sigmoid().item())
    probs = (p_prev, p_last)

    # 3) Determine last closed time from series
    now_ms = int(time.time() * 1000)
    ts_last = int(df["ts"].iloc[-1])
    last_close_ms, stamp_tag, age_ms = resolve_last_closed(now_ms, ts_last, timeframe)

    if args.debug:
        print(f"[DEBUG] last bar — ts_last={ts_last} tag={stamp_tag} age_min={(age_ms/60000.0) if age_ms is not None else None}")

    if last_close_ms is None or not (0 <= age_ms <= SIX_MIN_MS):
        print("Last closed bar is not within the 6-minute window — not acting.")
        return

    # 4) New signal on last bar (avoid overlap via previous bar)
    take_long  = (p_last >= pos_thr) and not (p_prev >= pos_thr)
    take_short = (p_last <= neg_thr) and not (p_prev <= neg_thr)

    # 5) Connect to exchange
    ex = make_exchange(args, category=category)

# --- Idempotency: one order per bar per broker/symbol/timeframe/category ---
symbol = resolve_symbol(ex, ticker)
sym_safe = symbol.replace("/", "-").replace(":", "-")
broker = "coinex"
suffix = f"{broker}_{sym_safe}_{timeframe.lower()}_{category}"
last_ts_seen, guard_path = last_executed_guard(model_dir, suffix=suffix)
if last_ts_seen and last_ts_seen == last_close_ms:
    print(f"Signal already executed for this bar. Skipping. [guard={guard_path}]")
    return

    # Precision helpers
    def amount_to_precision(ex, symbol, amt):
        prec = ex.markets[symbol]["precision"].get("amount", 8) if symbol in ex.markets else 8
        return float(ex.amount_to_precision(symbol, amt)) if hasattr(ex, "amount_to_precision") else round(float(amt), prec)

    def price_to_precision(ex, symbol, px):
        prec = ex.markets[symbol]["precision"].get("price", 8) if symbol in ex.markets else 8
        return float(ex.price_to_precision(symbol, px)) if hasattr(ex, "price_to_precision") else round(float(px), prec)

    # 6) Position checks and exits (SL/TP closures if already in position)
    pos = get_open_position(ex, symbol, timeframe, category)
    last_close = float(df["close"].iloc[-1])
    last_high  = float(df["high"].iloc[-1])
    last_low   = float(df["low"].iloc[-1])

    if pos is not None:
        # Check TP/SL on open position by comparing to last high/low
        entry = float(pos.get("entry", last_close))
        sz    = float(pos.get("size", 0))
        side_open = pos.get("side", "long")

        tp_hit = False
        sl_hit = False
        if side_open == "long":
            if tp_pct is not None and last_high >= entry * (1.0 + tp_pct):
                tp_hit = True
            if sl_pct is not None and last_low <= entry * (1.0 - sl_pct):
                sl_hit = True
        else:  # short
            if tp_pct is not None and last_low <= entry * (1.0 - tp_pct):
                tp_hit = True
            if sl_pct is not None and last_high >= entry * (1.0 + sl_pct):
                sl_hit = True

        if tp_hit or sl_hit:
            close_side = "buy" if pos["side"] == "short" else "sell"
            close_qty = amount_to_precision(ex, symbol, sz)
            reason = "TP hit" if tp_hit else "SL hit"
            print(f"{reason} — reduce-only MARKET {close_side.upper()} {ticker} qty={close_qty} (entry≈{entry:.6f}, H/L={last_high:.6f}/{last_low:.6f})")
            try:
                ex.create_order(symbol=symbol, type="market", side=close_side, amount=close_qty,
                                params={"reduceOnly": True})
            except Exception as e:
                print(f"[ERROR] close failed: {e}")
            return  # do not flip/open new after a stop/take exit

    # 7) No overlapping: open only if there is no active position
    if pos is not None:
        print("Avoiding opening another position — pyramiding disabled.")
        return

    # 8) Decide side
    if not take_long and not take_short:
        print(f"No order triggered: {ticker} {timeframe} — no active signal")
        return

    side = "buy" if take_long else "sell"

    # 9) Compute qty from capital (simplified; keep your original logic intact)
    bal = get_trade_balance(ex, category=category)  # in quote
    if bal is None or bal <= 0:
        print("[WARN] No available balance to trade.")
        return

    # Use a conservative sizing (example: 95% of quote for spot longs, or x USD for futures)
    qty = compute_order_size(ex, ticker, category, bal, last_close, side)
    qty = amount_to_precision(ex, symbol, qty)
    if qty <= 0:
        print("[WARN] Computed qty <= 0 — aborting.")
        return

    # 10) Place MARKET order — keep exactly the same server call, no attached brackets
    px = price_to_precision(ex, symbol, last_close)

    # Compute reference SL/TP prices from last close and configured percentages.
    sl_price = None
    tp_price = None
    try:
        if side == "buy":
            if sl_pct is not None:
                sl_price = price_to_precision(ex, symbol, last_close * (1.0 - sl_pct))
            if tp_pct is not None:
                tp_price = price_to_precision(ex, symbol, last_close * (1.0 + tp_pct))
        else:  # opening a SHORT
            if sl_pct is not None:
                sl_price = price_to_precision(ex, symbol, last_close * (1.0 + sl_pct))
            if tp_pct is not None:
                tp_price = price_to_precision(ex, symbol, last_close * (1.0 - tp_pct))
    except Exception:
        # keep going even if precision helpers fail
        pass

    _sl_str = f"{sl_price:.6f}" if isinstance(sl_price, (int, float)) and sl_price is not None else "-"
    _tp_str = f"{tp_price:.6f}" if isinstance(tp_price, (int, float)) and tp_price is not None else "-"
    print(f"Placing MARKET {side.upper()} {ticker} qty={qty} (px≈{px}) — prob={p_last:.3f} | SL≈{_sl_str} TP≈{_tp_str}")
    try:
        order = ex.create_order(symbol=symbol, type="market", side=side, amount=qty)
        # attach reference SL/TP to the local order dict for logging/tracking (no server-side brackets)
        if isinstance(order, dict):
            order["sat_sl_price"] = sl_price
            order["sat_tp_price"] = tp_price
        oid = order.get("id") or order.get("orderId") or order
        print(f"Order placed: {oid} | SL≈{_sl_str} TP≈{_tp_str}")
        write_last_executed(guard_path, last_close_ms)
    except Exception as e:
        print(f"[ERROR] order failed: {e}")

# --- position helpers (after we define exchange) ---
def _extract_entry_price(p) -> Optional[float]:
    for k in ("entryPrice", "avgPrice", "averagePrice", "price", "markPrice"):
        v = p.get(k)
        if v is not None:
            try:
                fv = float(v)
                if fv > 0:
                    return fv
            except Exception:
                pass
    info = p.get("info") or {}
    for k in ("avg_entry_price", "entry_price", "avg_cost_price", "average_entry_price"):
        v = info.get(k)
        if v is not None:
            try:
                fv = float(v)
                if fv > 0:
                    return fv
            except Exception:
                pass
    return None

def get_open_position(ex, symbol: str):
    """
    Return {"side": "long"|"short", "size": float, "entry_price": float|None}
    or None if flat.
    """
    def _scan(positions):
        for p in positions or []:
            if p.get("symbol") and p["symbol"] != symbol:
                continue
            size = float(p.get("contracts") or p.get("size") or p.get("positionAmt") or 0)
            if abs(size) > 0:
                side = (p.get("side") or ("long" if size > 0 else "short")).lower()
                entry_price = _extract_entry_price(p)
                return {"side": side, "size": abs(size), "entry_price": entry_price}
        return None

    try:
        pos = _scan(ccxt.Exchange.fetch_positions.__get__(ex, ex)([symbol]))
        if pos: return pos
    except Exception:
        pass
    try:
        return _scan(ex.fetch_positions())
    except Exception:
        return None

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Run LSTM bundle; CoinEx order on fresh signal within 6 minutes — bars read from a JSON file."
    )
    ap.add_argument("--model-dir", required=True,
                    help="Folder with model.pt, preprocess.json, meta.json")
    ap.add_argument("--bars-json", required=True,
                    help="Path to JSON file containing OHLCV bars")
    ap.add_argument("--ticker", default=None,
                    help="Override ticker (otherwise taken from meta.json or defaults to BTCUSDT)")
    ap.add_argument("--timeframe", default=None,
                    help="Override timeframe (otherwise taken from meta.json or defaults to 1h)")
    ap.add_argument("--auto-transfer", action="store_true",
                    help="If needed, auto-transfer USDT from SPOT to SWAP before ordering")
    ap.add_argument("--transfer-buffer", type=float, default=0.01,
                    help="Extra fraction to transfer (e.g., 0.01=+1%) to cover fees/rounding")
    ap.add_argument("--debug", action="store_true", help="Print balances and transfer details")

    # Select account by *variable names* in ~/.ssh/coinex_keys.env
    ap.add_argument("--pub_key", default=None,
                    help="Name of the API key variable (e.g., API_KEY_ETH)")
    ap.add_argument("--sec_key", default=None,
                    help="Name of the API secret variable (e.g., API_SECRET_ETH)")
    return ap.parse_args()

def main():
    args = parse_args()
    decide_and_maybe_trade(args)

if __name__ == "__main__":
    main()
