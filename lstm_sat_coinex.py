#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Ichimoku runner for CoinEx (USDT linear swap, i.e., futures only).

Key points:
- Always uses futures (swap) for both LONG and SHORT.
- No args.category; removed entirely.
- Bars loaded from a JSON file with closed bars.
- TorchScript model bundle: model.pt, preprocess.json, meta.json.
- SL/TP exits from meta.json (sl_pct / tp_pct).
- Pyramiding protection & flip on reversal.
- 6-minute window after last bar close.
- Writes/reads SHARED last-bar ID at ~/.sat_state/lastbars.json (override via SAT_STATE_DIR)
  so other broker scripts (Bybit/Binance/Coinbase) can use the same last-bar identity.
- Keeps a per-broker idempotency stamp file in the model dir to avoid double action for CoinEx.

CLI keys:
--pub_key/--sec_key are names of variables inside ~/.ssh/coinex_keys.env
(e.g. API_KEY_COINEX=..., API_SECRET_COINEX=...).
"""

from __future__ import annotations
import os, sys, json, time, argparse, pathlib, tempfile
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    print("Please install torch:  pip install torch", file=sys.stderr)
    raise

try:
    import ccxt
except Exception:
    print("Please install ccxt:  pip install ccxt", file=sys.stderr)
    raise


# =========================
# Config / constants
# =========================
SIX_MIN_MS = 6 * 60 * 1000

TF_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

def tf_ms(tf: str) -> int:
    v = TF_TO_MS.get(str(tf).lower())
    if not v:
        raise SystemExit(f"Unsupported timeframe '{tf}' — add it to TF_TO_MS.")
    return v


# =========================
# JSON bars I/O
# =========================
def _normalize_bar(b: Dict) -> Optional[Dict]:
    """
    Normalize possible bar shapes to:
      {ts(ms), open, high, low, close, volume}
    """
    if not isinstance(b, dict):
        return None
    keys = {k.lower(): k for k in b.keys()}

    def g(*cands, default=None):
        for c in cands:
            k = keys.get(c)
            if k in b:
                return b[k]
        return default

    ts = g("ts", "timestamp", "time", "t", default=None)
    if ts is None:
        return None
    ts = int(ts)
    if ts < 10_000_000_000:  # seconds → ms
        ts *= 1000

    try:
        o = float(g("open", "o"))
        h = float(g("high", "h"))
        l = float(g("low", "l"))
        c = float(g("close", "c"))
        v = float(g("volume", "v", "vol"))
    except Exception:
        return None

    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}

def load_bars_from_json(path: str) -> pd.DataFrame:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"Bars JSON not found: {p}")

    content = p.read_text().strip()
    bars: List[Dict] = []

    # Try array or wrapped object
    try:
        obj = json.loads(content)
        if isinstance(obj, list):
            bars = obj
        elif isinstance(obj, dict):
            for key in ("data", "bars", "result", "items"):
                if key in obj and isinstance(obj[key], list):
                    bars = obj[key]
                    break
    except Exception:
        # LDJSON fallback
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                bars.append(json.loads(line))
            except Exception:
                pass

    if not bars:
        raise SystemExit(f"No bars decoded from {p}")

    norm = []
    for b in bars:
        nb = _normalize_bar(b)
        if nb:
            norm.append(nb)
    if not norm:
        raise SystemExit(f"No valid bars decoded from {p}")

    df = pd.DataFrame(norm).dropna().sort_values("ts").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    return df[["timestamp","ts","open","high","low","close","volume"]]


# =========================
# Ichimoku + features
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
    d["chikou"] = d["close"].shift(-kijun)  # trainer parity assumption
    return d

def slope(series: pd.Series, w: int = 8) -> pd.Series:
    return series.diff(w)

def build_features(df: pd.DataFrame, tenkan: int, kijun: int, senkou: int,
                   displacement: int, slope_window: int = 8) -> pd.DataFrame:
    d = ichimoku(df, tenkan, kijun, senkou)
    d["ret1"] = d["close"].pct_change()
    d["oc_diff"] = d["close"] - d["open"]
    d["hl_range"] = d["high"] - d["low"]
    d["logv_chg"] = np.log1p(d["volume"]).diff()
    d["dist_px_cloud_top"] = d["close"] - d[["span_a","span_b"]].max(axis=1)
    d["dist_px_cloud_bot"] = d["close"] - d[["span_a","span_b"]].min(axis=1)
    d["dist_tk_kj"]        = d["tenkan"] - d["kijun"]
    d["span_order"]        = np.where(d["span_a"] >= d["span_b"], 1.0, -1.0)
    d["tk_slope"]     = slope(d["tenkan"], slope_window)
    d["kj_slope"]     = slope(d["kijun"], slope_window)
    d["span_a_slope"] = slope(d["span_a"], slope_window)
    d["span_b_slope"] = slope(d["span_b"], slope_window)
    d["chikou_above"] = np.where(d["chikou"] > d["close"], 1.0, -1.0)
    d["vol20"] = d["volume"].rolling(20, min_periods=1).mean()
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

    feature_names = meta.get("features") or meta.get("feature_names")
    if not feature_names:
        raise SystemExit("meta.json missing 'features' list")

    lookback = int(meta.get("lookback") or meta.get("window") or 64)
    pos_thr  = float(meta.get("pos_thr", 0.55))
    neg_thr  = float(meta.get("neg_thr", 0.45))
    sl_pct   = meta.get("sl_pct", None)
    tp_pct   = meta.get("tp_pct", None)
    sl_pct   = float(sl_pct) if sl_pct is not None else None
    tp_pct   = float(tp_pct) if tp_pct is not None else None

    ichimoku_params = {
        "tenkan": int(meta.get("tenkan", 7)),
        "kijun": int(meta.get("kijun", 211)),
        "senkou": int(meta.get("senkou", 120)),
        "displacement": int(meta.get("displacement", 41)),
    }

    mean = np.array(pre.get("mean"), dtype=np.float32)
    std  = np.array(pre.get("std"), dtype=np.float32)
    if mean.shape[0] != len(feature_names) or std.shape[0] != len(feature_names):
        raise SystemExit("preprocess.json shapes don't match features")

    model = torch.jit.load(mdl_path, map_location="cpu")
    model.eval()

    return {
        "meta": meta,
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
        "paths": {"dir": model_dir},
    }


# =========================
# Time helpers
# =========================
def resolve_last_closed(now_ms: int, last_bar_open_ms: int, timeframe: str) -> Tuple[Optional[int], str, Optional[int]]:
    """
    Returns (last_close_ms, stamp_tag, age_ms).
    Prefers close_stamp = last_bar_open + step; fallback open_stamp.
    """
    step = tf_ms(timeframe)
    candidates = [
        (last_bar_open_ms + step, "close_stamp"),
        (last_bar_open_ms, "open_stamp"),
    ]
    valid = [(c, tag, now_ms - c) for (c, tag) in candidates if now_ms >= c]
    if not valid:
        return None, "future", None
    c, tag, age = min(valid, key=lambda x: x[2])
    return c, tag, age


# =========================
# Shared last-bar ID store (cross-broker)
# =========================
def state_dir() -> str:
    base = os.getenv("SAT_STATE_DIR", os.path.expanduser("~/.sat_state"))
    pathlib.Path(base).mkdir(parents=True, exist_ok=True)
    return base

def lastbars_json_path() -> str:
    return os.path.join(state_dir(), "lastbars.json")

def read_lastbars_store() -> Dict[str, Dict]:
    p = pathlib.Path(lastbars_json_path())
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def write_lastbars_store(data: Dict[str, Dict]) -> None:
    p = pathlib.Path(lastbars_json_path())
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="lastbars_", suffix=".json", dir=str(p.parent))
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, separators=(",", ":"), sort_keys=True)
        os.replace(tmp_path, p)  # atomic
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def bar_id(ticker: str, timeframe: str, last_ts_ms: int) -> str:
    return f"{ticker}|{timeframe}|{int(last_ts_ms)}"

def update_shared_lastbar(ticker: str, timeframe: str, last_open_ts_ms: int, last_close_ms: int) -> None:
    key = f"{ticker}:{timeframe}"
    store = read_lastbars_store()
    store[key] = {
        "bar_id": bar_id(ticker, timeframe, last_open_ts_ms),
        "ticker": ticker,
        "timeframe": timeframe,
        "last_open_ts": int(last_open_ts_ms),
        "last_close_ts": int(last_close_ms),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_lastbars_store(store)


# =========================
# Per-bundle idempotency (per broker)
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
# Exchange helpers (CoinEx - swap only)
# =========================
def make_exchange(pub_key_name: Optional[str], sec_key_name: Optional[str]):
    """
    Always defaultType='swap' (futures).
    Keys are looked up in ~/.ssh/coinex_keys.env by variable names.
    """
    api_key, api_secret = None, None
    if pub_key_name or sec_key_name:
        keyfile = os.path.expanduser("~/.ssh/coinex_keys.env")
        kv = {}
        if os.path.exists(keyfile):
            for line in open(keyfile, "r"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
        api_key = kv.get(pub_key_name) if pub_key_name else None
        api_secret = kv.get(sec_key_name) if sec_key_name else None

    ex = ccxt.coinex({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},  # FUTURES ONLY
    })
    ex.load_markets()
    return ex

def resolve_symbol(ex, ticker: str) -> str:
    # e.g., ETHUSDT -> ETH/USDT:USDT for linear USDT perp
    ticker = ticker.upper().replace("/", "")
    base = ticker[:-4] if ticker.endswith("USDT") else ticker
    candidates = [s for s in ex.symbols if s.startswith(f"{base}/USDT") and (":USDT" in s)]
    if candidates:
        return sorted(candidates)[0]
    for s in ex.symbols:
        if f"{base}/" in s and ":USDT" in s:
            return s
    raise SystemExit(f"No swap symbol resolved for {ticker} on CoinEx")

def fetch_usdt_balance_swap(ex) -> float:
    try:
        bal = ex.fetch_balance(params={"type": "swap"})
        total = bal.get("total", {}).get("USDT")
        free  = bal.get("free", {}).get("USDT")
        return float(free if free is not None else (total or 0.0))
    except Exception:
        return 0.0

def transfer_spot_to_swap_if_needed(ex, min_usdt: float, buffer_frac: float = 0.01, debug: bool = False) -> float:
    """Best-effort top up for swap account."""
    try:
        swap_free = fetch_usdt_balance_swap(ex)
        if swap_free >= min_usdt:
            return swap_free
        if debug:
            print(f"[DEBUG] swap USDT={swap_free:.2f} < {min_usdt:.2f}; topping up from spot...")
        bal_spot = ex.fetch_balance(params={"type": "spot"})
        spot_free = float(bal_spot.get("free", {}).get("USDT") or 0.0)
        if spot_free <= 0:
            return swap_free
        amt = min(spot_free, min_usdt * (1.0 + max(0.0, buffer_frac)))
        if debug:
            print(f"[DEBUG] transferring {amt:.6f} USDT spot→swap")
        ex.transfer(code="USDT", amount=float(f"{amt:.6f}"), fromAccount="spot", toAccount="swap")
        time.sleep(0.5)
        return fetch_usdt_balance_swap(ex)
    except Exception as e:
        print(f"[WARN] auto-transfer failed: {e}")
        return fetch_usdt_balance_swap(ex)

def get_swap_position(ex, symbol: str) -> Optional[Dict]:
    def _scan(ps):
        if not ps:
            return None
        for p in ps:
            if (p or {}).get("symbol") == symbol:
                side = (p.get("side") or "").lower()
                sz   = float(p.get("contracts") or p.get("contractSize") or p.get("size") or 0.0)
                entry= float(p.get("entryPrice") or 0.0)
                if sz and side:
                    return {"side": side, "size": abs(sz), "entry": entry}
        return None
    try:
        pos = _scan(ex.fetch_positions([symbol]))
        if pos: return pos
    except Exception:
        pass
    try:
        return _scan(ex.fetch_positions())
    except Exception:
        return None


# =========================
# Inference helpers
# =========================
def to_sequences_latest(feat_df: pd.DataFrame, features: List[str], lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns X (2, lookback, n_features) for prev and last bars + ts seq.
    """
    if len(feat_df) < (lookback + 1):
        raise SystemExit("Not enough rows to build lookback sequences")
    sub = feat_df.iloc[-(lookback+1):].copy().reset_index(drop=True)
    prev = sub.iloc[:-1][features].to_numpy(dtype=np.float32)
    last = sub.iloc[1:][features].to_numpy(dtype=np.float32)
    X = np.stack([prev, last], axis=0)
    ts_seq = sub["ts"].to_numpy()
    return X, ts_seq

def run_model(model, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> Tuple[float, float]:
    """
    Returns (p_prev, p_last).
    """
    Xn = (X - mean) / (std + 1e-12)
    with torch.no_grad():
        t = torch.from_numpy(Xn).float()  # (2, L, F)
        out = model(t)
        if isinstance(out, (list, tuple)):
            out = out[0]
        x = out.squeeze()
        # if binary logit -> sigmoid; if 2-class -> softmax prob of class 1
        x = torch.sigmoid(x) if x.ndim == 0 or x.shape[-1] != 2 else torch.softmax(x, dim=-1)[..., 1]
        arr = x.detach().cpu().numpy().astype(np.float64)
    if np.ndim(arr) == 0:
        return float(arr), float(arr)
    if arr.shape[0] == 2:
        return float(arr[0]), float(arr[1])
    return float(arr[-2]), float(arr[-1])


# =========================
# Core flow (futures only)
# =========================
def decide_and_maybe_trade(args):
    # 1) Load bundle
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

    # 2) Resolve ticker/timeframe
    ticker = args.ticker or meta.get("ticker") or "BTCUSDT"
    timeframe = args.timeframe or meta.get("timeframe") or "1h"

    # 3) Load bars (JSON)
    df = load_bars_from_json(args.bars_json)
    if df is None or len(df) < (lookback + 3):
        print("Not enough bars to build features.")
        return

    # Build features & check feature names
    feat_df_full = build_features(
        df[["timestamp","ts","open","high","low","close","volume"]].copy(),
        ik["tenkan"], ik["kijun"], ik["senkou"], ik["displacement"]
    )
    for c in feats:
        if c not in feat_df_full.columns:
            raise SystemExit(f"Feature '{c}' missing in computed frame.")
    feat_df = feat_df_full.copy()

    # 4) Inference
    X, ts_seq = to_sequences_latest(feat_df[feats + ["ts"]], feats, lookback)
    p_prev, p_last = run_model(model, X, mean, std)
    print(f"LSTM inference | p_prev={p_prev:.3f} | p_last={p_last:.3f} | pos_thr={pos_thr:.3f} | neg_thr={neg_thr:.3f}")

    # 5) Time gating (6-minute window after last bar close)
    now_ms = int(time.time() * 1000)
    ts_last_open = int(df["ts"].iloc[-1])
    last_close_ms, stamp_tag, age_ms = resolve_last_closed(now_ms, ts_last_open, timeframe)
    if args.debug:
        print(f"[DEBUG] last bar — ts_last={ts_last_open} tag={stamp_tag} age_min={(age_ms/60000.0) if age_ms is not None else None}")

    if last_close_ms is None or not (0 <= age_ms <= SIX_MIN_MS):
        print("Last closed bar is not within the 6-minute window — not acting.")
        return

    # 6) Shared last-bar ID (cross-broker coordination)
    update_shared_lastbar(ticker, timeframe, ts_last_open, last_close_ms)
    shared = read_lastbars_store().get(f"{ticker}:{timeframe}", {})
    if args.debug:
        print(f"[DEBUG] shared lastbar: {shared.get('bar_id')} last_close_ts={shared.get('last_close_ts')}")

    # 7) Per-broker idempotency (CoinEx-only guard)
    guard_suffix = f"coinex_swap_{ticker}_{timeframe}"
    last_seen_ts, guard_path = last_executed_guard(model_dir, guard_suffix)
    if last_seen_ts is not None and last_seen_ts == last_close_ms:
        print("Already acted on this bar for CoinEx — not acting again.")
        return

    # 8) New signal detection
    take_long  = (p_last >= pos_thr) and not (p_prev >= pos_thr)
    take_short = (p_last <= neg_thr) and not (p_prev <= neg_thr)
    if not take_long and not take_short:
        print("No fresh signal on the last bar — not acting.")
        return

    # 9) Connect exchange (swap only)
    ex = make_exchange(args.pub_key, args.sec_key)
    symbol = resolve_symbol(ex, ticker)

    # 10) Position and SL/TP on swap
    pos = get_swap_position(ex, symbol)

    last_close = float(df["close"].iloc[-1])
    last_high  = float(df["high"].iloc[-1])
    last_low   = float(df["low"].iloc[-1])

    if pos is not None and pos.get("side"):
        entry = float(pos.get("entry") or last_close)
        side_open = pos["side"]  # 'long' or 'short'
        sz = float(pos.get("size") or 0.0)

        tp_hit = False
        sl_hit = False
        if side_open == "long":
            if tp_pct is not None and last_high >= entry * (1.0 + tp_pct): tp_hit = True
            if sl_pct is not None and last_low  <= entry * (1.0 - sl_pct): sl_hit = True
        else:  # short
            if tp_pct is not None and last_low  <= entry * (1.0 - tp_pct): tp_hit = True
            if sl_pct is not None and last_high >= entry * (1.0 + sl_pct): sl_hit = True

        if tp_hit or sl_hit:
            reason = "TP" if tp_hit else "SL"
            try:
                side = "buy" if side_open == "short" else "sell"  # reduce to close
                params = {"reduceOnly": True}
                print(f"{reason} hit — closing existing {side_open} position.")
                ex.create_order(symbol, "market", side, sz or 1, None, params)
                write_last_executed(guard_path, last_close_ms)
            except Exception as e:
                print(f"[ERROR] close on {reason} failed: {e}")
            return

    # 11) Avoid pyramiding (same direction)
    if pos is not None and pos.get("side") and (
        (take_long  and pos["side"] == "long") or
        (take_short and pos["side"] == "short")
    ):
        print("Avoiding opening another position - pyramiding.")
        return

    # 12) Flip if necessary (close opposite before open)
    if pos is not None and pos.get("side"):
        try:
            side_open = pos["side"]
            sz = float(pos.get("size") or 0.0)
            if (take_long and side_open == "short") or (take_short and side_open == "long"):
                side = "buy" if side_open == "short" else "sell"
                print("Signal reversal — closing existing position.")
                ex.create_order(symbol, "market", side, sz or 1, None, {"reduceOnly": True})
                time.sleep(0.2)
        except Exception as e:
            print(f"[WARN] failed to close before flip: {e}")

    # 13) Optional top-up for swap
    if args.auto_transfer:
        transfer_spot_to_swap_if_needed(ex, min_usdt=50.0, buffer_frac=args.transfer_buffer, debug=args.debug)

    # 14) Open order on swap
    try:
        if take_long:
            side = "buy"   # open LONG
            print("Opening LONG (futures/swap).")
        else:
            side = "sell"  # open SHORT
            print("Opening SHORT (futures/swap).")

        qty = 1  # adjust sizing as needed in your environment
        ex.create_order(symbol, "market", side, qty, None, {"reduceOnly": False})
        write_last_executed(guard_path, last_close_ms)
    except Exception as e:
        print(f"[ERROR] order failed: {e}")


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Run LSTM bundle; CoinEx FUTURES (swap) orders on fresh signal within 6 minutes — bars read from a JSON file."
    )
    ap.add_argument("--model-dir", required=True,
                    help="Folder with model.pt, preprocess.json, meta.json")
    ap.add_argument("--bars-json", required=True,
                    help="Path to JSON file containing OHLCV bars (closed bars)")
    ap.add_argument("--ticker", default=None,
                    help="Override ticker (otherwise meta.json or BTCUSDT)")
    ap.add_argument("--timeframe", default=None,
                    help="Override timeframe (otherwise meta.json or 1h)")
    ap.add_argument("--auto-transfer", action="store_true",
                    help="Auto-transfer USDT from spot→swap before opening futures")
    ap.add_argument("--transfer-buffer", type=float, default=0.01,
                    help="Extra fraction to transfer (e.g., 0.01=+1%)")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs")

    # Variable names inside ~/.ssh/coinex_keys.env
    ap.add_argument("--pub_key", default=None,
                    help="Name of the API key variable (e.g., API_KEY_COINEX)")
    ap.add_argument("--sec_key", default=None,
                    help="Name of the API secret variable (e.g., API_SECRET_COINEX)")
    return ap.parse_args()


def main():
    args = parse_args()
    decide_and_maybe_trade(args)

if __name__ == "__main__":
    main()
