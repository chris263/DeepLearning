#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Ichimoku runner (Bybit) that:
- Loads bars from a JSON file (no database calls).
- Uses pub/secret *variable names* to pick credentials from ~/.ssh/coinex_keys.env
  (same location/mechanism as your CoinEx runner).
- Handles SL/TP exits using values from meta.json (sl_pct, tp_pct).
- Avoids pyramiding; flips positions when signal reverses.
- Idempotent: one execution per closed bar via .last_order_ts.txt in model dir.
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
    if tf not in TF_TO_MS: raise ValueError(f"Unsupported timeframe: {tf}")
    return TF_TO_MS[tf]

# =========================
# Load bars from JSON (SAT format or simple OHLCV arrays)
# =========================
def _normalize_bar(b: Dict) -> Optional[Dict]:
    # Accept several shapes:
    # - {"ts": 1700000000000, "o":..,"h":..,"l":..,"c":..,"v":..}
    # - {"timestamp":..., "open":...,"high":...,"low":...,"close":...,"volume":...}
    # - {"t":..., "o":...,"h":...,"l":...,"c":...,"v":...}
    # - {"time": ..., "open":...,"high":...,"low":...,"close":...,"volume":...}
    def get_num(*keys, default=None):
        for k in keys:
            if k in b and b[k] is not None:
                try:
                    return float(b[k])
                except Exception:
                    pass
        return default

    ts_ms = None
    for tk in ("ts", "timestamp", "t", "time"):
        if tk in b:
            try:
                v = int(b[tk])
                ts_ms = v
                break
            except Exception:
                pass
    if ts_ms is None:
        return None

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

    if content.startswith("["):
        # JSON array of bars
        try:
            arr = json.loads(content)
        except Exception as e:
            raise SystemExit(f"Invalid JSON in {p}: {e}")
        for b in arr:
            nb = _normalize_bar(b)
            if nb: bars.append(nb)
    else:
        # line-delimited JSON
        for line in content.splitlines():
            line = line.strip()
            if not line: continue
            try:
                b = json.loads(line)
            except Exception:
                continue
            nb = _normalize_bar(b)
            if nb: bars.append(nb)

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

# --- Map/align computed feature columns to the exact names expected by meta.json ---
# This lets us tolerate minor naming differences between trainer & runner.
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
    # For each desired feature name, if missing, search synonyms and duplicate the column
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
# Bundle loader (TorchScript)
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

    # SL/TP from meta (fractions, e.g., 0.02 = 2%). Accept top-level OR nested risk{}.
    sl_pct = meta.get("sl_pct", meta.get("stop_loss_pct", None))
    tp_pct = meta.get("tp_pct", meta.get("take_profit_pct", None))
    # also allow nested under risk{}
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
        "tenkan":   int(meta.get("tenkan", 9)),
        "kijun":    int(meta.get("kijun", 26)),
        "senkou":   int(meta.get("senkou", 52)),
        "displacement": int(meta.get("displacement", 26)),
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
        return np.zeros((0,), dtype=np.float32)
    with torch.no_grad():
        Xn = (X - mean) / std
        t = torch.from_numpy(Xn).float()
        logits = model(t)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return probs.astype(np.float32)

# =========================
# Bybit helpers
# =========================
def bybit_exchange(api_key: str, api_secret: str):
    # Linear USDT perps (unified contract)
    ex = ccxt.bybit({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",       # derivatives
            "defaultSubType": "linear",  # USDT-margined
        },
    })
    ex.load_markets()
    return ex

def resolve_symbol(ex, ticker: str) -> str:
    base = (ticker.replace("USDT","")
                    .replace("/USDT","")
                    .replace(":USDT","")
                    .upper())
    candidates = [
        f"{base}/USDT:USDT",
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
    # Bybit uses 'contract' for derivatives wallet in ccxt
    params = {"type": acct_type} if acct_type else {}
    bal = ex.fetch_balance(params)
    usdt = bal.get("USDT") or {}
    free = float(usdt.get("free") or 0)
    total = float(usdt.get("total") or 0)
    return free if free > 0 else total

def ensure_contract_funds(ex, needed_usdt: float, auto_transfer: bool, transfer_buffer: float, debug: bool) -> float:
    contract_before = fetch_usdt_balance(ex, "contract")
    if debug:
        spot_before = fetch_usdt_balance(ex, "spot")
        print(f"[DEBUG] balances before — contract={contract_before:.8f} USDT, spot={spot_before:.8f} USDT")

    if contract_before >= needed_usdt:
        return contract_before

    if not auto_transfer:
        return contract_before

    try:
        # move USDT from spot to contract (unified account)
        spot_before = fetch_usdt_balance(ex, "spot")
        delta = max(0.0, (needed_usdt * (1.0 + transfer_buffer)) - contract_before)
        amt = min(delta, spot_before)
        if amt > 0:
            ex.transfer("USDT", amt, "spot", "contract")
            if debug:
                print(f"[DEBUG] transferred {amt:.8f} USDT spot -> contract")
    except Exception as e:
        print(f"[WARN] auto-transfer failed: {e}")

    return fetch_usdt_balance(ex, "contract")

def amount_to_precision(ex, symbol: str, qty: float) -> float:
    market = ex.markets[symbol]
    step = market.get("precision", {}).get("amount")
    if step is None:
        step = market.get("limits", {}).get("amount", {}).get("min", None)
    if step is None:
        return float(qty)
    prec = market["precision"]["amount"]
    return float(ex.amount_to_precision(symbol, qty))

def price_to_precision(ex, symbol: str, price: float) -> float:
    return float(ex.price_to_precision(symbol, price))

def read_keys_from_env(pub_var: Optional[str], sec_var: Optional[str]) -> Tuple[str, str]:
    env_path = os.path.expanduser("~/.ssh/coinex_keys.env")
    if not os.path.exists(env_path):
        raise SystemExit(f"Env file not found: {env_path}")
    d = {}
    for line in open(env_path):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "=" not in line: continue
        k, v = line.split("=", 1)
        d[k.strip()] = v.strip()
    kname = pub_var or os.environ.get("BYBIT_KEY_VAR") or "BYBIT_KEY_MAIN"
    sname = sec_var or os.environ.get("BYBIT_SECRET_VAR") or "BYBIT_SECRET_MAIN"
    if kname not in d or sname not in d:
        raise SystemExit(f"Missing {kname}/{sname} in {env_path}")
    return d[kname], d[sname]

def get_open_position(ex, symbol: str) -> Optional[Dict]:
    def _scan(pos_list):
        if not pos_list: return None
        for p in pos_list:
            sym = p.get("symbol") or p.get("info", {}).get("symbol")
            if sym == symbol or (ex.markets.get(sym, {}).get("symbol") == symbol):
                sz = float(p.get("contracts") or p.get("size") or p.get("positionAmt") or 0.0)
                if abs(sz) <= 0: continue
                side = "long" if sz > 0 else "short"
                entry = _extract_entry_price(p) or _extract_entry_price(p.get("info", {}))
                return {"side": side, "size": abs(sz), "entry_price": entry}
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

# --- helpers to extract entry price from position payload ---
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
    return None

# =========================
# Resolve last closed bar from series
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
# Idempotency guard (per broker/symbol/timeframe/category)
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

    # ---- JSON bars (replaces any DB query) ----
    df = load_bars_from_json(args.bars_json)

    # Build features
    feat_df_full = build_features(
        df[["timestamp","open","high","low","close","volume"]].copy(),
        ik["tenkan"], ik["kijun"], ik["senkou"], ik["displacement"]
    )
    # Align column names to exactly match meta.features
    feat_df_full = align_features_to_meta(feat_df_full, feats)
    for c in feats:
        if c not in feat_df_full.columns:
            raise SystemExit(f"Feature '{c}' missing in computed frame.")
    feat_df = feat_df_full.copy()

    # 2) Get last two bar probs
    X, ts_seq = to_sequences_latest(feat_df[feats + ["timestamp"]], feats, lookback)
    probs = predict_proba(model, X, mean, std)
    if len(ts_seq) < 2 or len(probs) < 2:
        print("Not enough bars to form sequences yet.")
        return

    p_prev, p_last = float(probs[0]), float(probs[1])

    # 3) Determine last closed time from series
    now_ms = int(time.time() * 1000)
    ts_last = int(df["ts"].iloc[-1])
    last_close_ms, stamp_tag, age_ms = resolve_last_closed(now_ms, ts_last, timeframe)

    if args.debug:
        print(f"[DEBUG] last bar — ts_last={ts_last} tag={stamp_tag} age_min={(age_ms/60000.0) if age_ms is not None else None}")

    if last_close_ms is None or not (0 <= age_ms <= SIX_MIN_MS):
        print("Last closed bar is not within the 6-minute window — not acting.")
        return

    # 3.5) SL/TP EXIT CHECK on the last closed bar, if a position exists
    api_key, api_secret = read_keys_from_env(args.pub_key, args.sec_key)
    ex = bybit_exchange(api_key, api_secret)
    symbol = resolve_symbol(ex, ticker)

    pos = get_open_position(ex, symbol)
    if pos and (sl_pct is not None or tp_pct is not None):
        entry = pos.get("entry_price")
        if entry and entry > 0:
            last_high = float(df["high"].iloc[-1])
            last_low  = float(df["low"].iloc[-1])
            sl_hit = tp_hit = False

            if pos["side"] == "long":
                if tp_pct is not None:
                    tp_hit = last_high >= entry * (1.0 + tp_pct)
                if sl_pct is not None:
                    sl_hit = last_low  <= entry * (1.0 - sl_pct)
            else:  # short
                if tp_pct is not None:
                    tp_hit = last_low  <= entry * (1.0 - tp_pct)
                if sl_pct is not None:
                    sl_hit = last_high >= entry * (1.0 + sl_pct)

            if tp_hit or sl_hit:
                close_side = "buy" if pos["side"] == "short" else "sell"
                close_qty = amount_to_precision(ex, symbol, pos["size"])
                reason = "TP hit" if tp_hit else "SL hit"
                print(f"{reason} — reduce-only MARKET {close_side.upper()} {symbol} qty={close_qty} (entry≈{entry:.6f}, H/L={last_high:.6f}/{last_low:.6f})")
                try:
                    ex.create_order(symbol=symbol, type="market", side=close_side, amount=close_qty,
                                    params={"reduceOnly": True})
                except Exception as e:
                    print(f"[ERROR] close failed: {e}")
                return  # do not flip on the same run after SL/TP exit

    # 4) New signal on last bar (avoid overlap via previous bar)
    take_long  = (p_last >= pos_thr) and (p_prev < pos_thr)
    take_short = (p_last <= neg_thr) and (p_prev > neg_thr)
    if take_long and take_short:
        if (p_last - pos_thr) >= (neg_thr - p_last):
            take_short = False
        else:
            take_long = False
    if not (take_long or take_short):
        print(f"No new signal. prev={p_prev:.3f}, last={p_last:.3f}, pos_thr={pos_thr:.3f}, neg_thr={neg_thr:.3f}")
        return

    # 5) Idempotency: one order per bar (per-broker/symbol/timeframe/category)
    sym_safe = symbol.replace("/", "-").replace(":", "-")
    broker = "bybit"; category = "futures"
    suffix = f"{broker}_{sym_safe}_{timeframe.lower()}_{category}"
    last_ts_seen, guard_path = last_executed_guard(model_dir, suffix=suffix)
    if last_ts_seen and last_ts_seen == last_close_ms:
        print(f"Signal already executed for this bar. Skipping. [guard={guard_path}]")
        return

    # --- handle existing position wrt signal (no pyramiding; flip allowed) ---
    if pos:
        if (take_long and pos["side"] == "long") or (take_short and pos["side"] == "short"):
            print("Avoiding opening another position - pyramiding.")
            return
        # flip
        close_side = "buy" if pos["side"] == "short" else "sell"
        close_qty = amount_to_precision(ex, symbol, pos["size"])
        print(f"Close position sent — reduce-only MARKET {close_side.upper()} {symbol} qty={close_qty}")
        try:
            ex.create_order(symbol=symbol, type="market", side=close_side, amount=close_qty,
                            params={"reduceOnly": True})
        except Exception as e:
            print(f"[ERROR] close failed: {e}")
            return

    # 6) Place new order (MARKET, unified linear swap)
    side = "buy" if take_long else "sell"
    # position sizing: 100% of contract balance (simple)
    contract_usdt = ensure_contract_funds(ex, needed_usdt=10.0, auto_transfer=args.auto_transfer,
                                          transfer_buffer=args.transfer_buffer, debug=args.debug)
    # Use 95% of what we have to be safe with fees/precision
    usdt_to_use = max(0.0, contract_usdt * 0.95)
    # approximate qty from last close price
    last_close = float(df["close"].iloc[-1])
    qty_approx = usdt_to_use / max(1e-12, last_close)
    qty = amount_to_precision(ex, symbol, qty_approx)

    # place order
    try:
        px = price_to_precision(ex, symbol, last_close)
        print(f"Placing MARKET {side.upper()} {symbol} qty={qty} (px≈{px}) — prob={p_last:.3f}")
        order = ex.create_order(symbol=symbol, type="market", side=side, amount=qty)
        oid = order.get("id") or order.get("orderId") or order.get("info", {}).get("orderId")
        print(f"Order placed: {oid}")
        write_last_executed(guard_path, last_close_ms)
    except Exception as e:
        print(f"[ERROR] order failed: {e}")

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Run LSTM bundle; Bybit order on fresh signal within 6 minutes — bars read from a JSON file."
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
                    help="If needed, auto-transfer USDT from SPOT to CONTRACT before ordering")
    ap.add_argument("--transfer-buffer", type=float, default=0.01,
                    help="Extra fraction to transfer (e.g., 0.01=+1%) to cover fees/rounding")
    ap.add_argument("--debug", action="store_true", help="Print balances and transfer details")

    # Select account by *variable names* from the SAME env file (~/.ssh/coinex_keys.env)
    ap.add_argument("--pub_key", default=None,
                    help="Name of the API key variable (e.g., BYBIT_KEY_MAIN)")
    ap.add_argument("--sec_key", default=None,
                    help="Name of the API secret variable (e.g., BYBIT_SECRET_MAIN)")
    return ap.parse_args()

def main():
    args = parse_args()
    decide_and_maybe_trade(args)

if __name__ == "__main__":
    main()
