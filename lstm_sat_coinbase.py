#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Ichimoku runner for coinbase (USDT linear swap, i.e., futures only).

- Fresh-cross trigger logic (from old).
- Close-only on reversal by default (from new). Optional: --reversal flip|cooldown
- Robust JSON loader; 6-minute window after real close time.
- SL/TP exits from meta.json (sl_pct / tp_pct).
- Balance-based sizing: 95% long, 80% short.
- Feature-name alignment to meta["features"].
- Shared last-bar ID at ~/.sat_state/lastbars.json (override via SAT_STATE_DIR).
- Per-broker idempotency file to avoid double actions per bar.

Requires: torch, ccxt, numpy, pandas
"""

from __future__ import annotations
import os, sys, json, time, argparse, pathlib, tempfile, math
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
# Constants / timeframe helpers
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
    if not isinstance(b, dict):
        return None
    keys = {k.lower(): k for k in b.keys()}

    def g(*cands, default=None):
        for c in cands:
            k = keys.get(c)
            if k in b:
                return b[k]
        return default

    ts = g("ts", "timestamp", "time", "t", "open_time", "opentime", default=None)
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts_dt = pd.to_datetime(ts, utc=True)
            ts = int(ts_dt.value // 1_000_000)
        except Exception:
            return None
    ts = int(ts)
    if ts < 10_000_000_000:  # seconds → ms
        ts *= 1000

    try:
        o = float(g("open", "o"))
        h = float(g("high", "h"))
        l = float(g("low", "l"))
        c = float(g("close", "c"))
        v = float(g("volume", "v", "vol", default=0.0))
    except Exception:
        return None

    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def load_bars_from_json(path: str) -> pd.DataFrame:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"Bars JSON not found: {p}")

    content = p.read_text().strip()
    bars: List[Dict] = []

    # Try single JSON object/array
    try:
        obj = json.loads(content)
        if isinstance(obj, list):
            bars = obj
        elif isinstance(obj, dict):
            for key in ("data", "bars", "result", "items", "price"):
                if key in obj and isinstance(obj[key], list):
                    bars = obj[key]
                    break
    except Exception:
        # NDJSON fallback
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
    return df[["timestamp", "ts", "open", "high", "low", "close", "volume"]]


# =========================
# Ichimoku + feature engineering
# =========================

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


def slope(series: pd.Series, w: int = 8) -> pd.Series:
    return series.diff(w)


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


def build_features(
    df: pd.DataFrame,
    tenkan: int,
    kijun: int,
    senkou: int,
    displacement: int,
    slope_window: int = 8,
) -> pd.DataFrame:
    d = ichimoku(df, tenkan, kijun, senkou)
    d["px"] = d["close"]

    d["ret1"] = d["close"].pct_change().fillna(0.0)
    d["oc_diff"] = (d["close"] - d["open"]) / d["open"]
    d["hl_range"] = (d["high"] - d["low"]) / d["px"]

    d["logv"] = np.log1p(d["volume"])
    d["logv_chg"] = d["logv"].diff().fillna(0.0)

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
    d["vol20"] = d["ret1"].rolling(20, min_periods=20).std().fillna(0.0)

    d["ts"] = df["ts"].values
    d["timestamp"] = df["timestamp"].values
    return d


# =========================
# Bundle I/O
# =========================

def load_bundle(model_dir: str):
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    meta_path = os.path.join(model_dir, "meta.json")
    pre_path = os.path.join(model_dir, "preprocess.json")
    mdl_path = os.path.join(model_dir, "model.pt")
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
    pos_thr = float(meta.get("pos_thr", 0.55))
    neg_thr = float(meta.get("neg_thr", 0.45))

    # risk
    risk = meta.get("risk") or {}
    sl_pct = risk.get("sl_pct", meta.get("sl_pct"))
    tp_pct = risk.get("tp_pct", meta.get("tp_pct"))
    fee_bps = risk.get("fee_bps", meta.get("fee_bps"))  # not used here but kept

    sl_pct = float(sl_pct) if sl_pct is not None else None
    tp_pct = float(tp_pct) if tp_pct is not None else None

    # ichimoku
    ik = meta.get("ichimoku") or {}
    ichimoku_params = {
        "tenkan": int(ik.get("tenkan", meta.get("tenkan", 9))),
        "kijun": int(ik.get("kijun", meta.get("kijun", 26))),
        "senkou": int(ik.get("senkou", meta.get("senkou", 52))),
        "displacement": int(ik.get("displacement", meta.get("displacement", 26))),
    }

    mean = np.array(pre.get("mean"), dtype=np.float32)
    std = np.array(pre.get("std"), dtype=np.float32)
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
    step = tf_ms(timeframe)
    candidates = [(last_bar_open_ms + step, "close_stamp"), (last_bar_open_ms, "open_stamp")]
    valid = [(c, tag, now_ms - c) for (c, tag) in candidates if now_ms >= c]
    if not valid:
        return None, "future", None
    c, tag, age = min(valid, key=lambda x: x[2])
    return c, tag, age


# =========================
# Shared last-bar ID
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
        os.replace(tmp_path, p)
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
# Idempotency & reversal state
# =========================

def last_executed_guard(model_dir: str, suffix: str) -> Tuple[Optional[int], str]:
    fname = f".last_order_ts_{suffix}.txt"
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


def reversal_state_paths(model_dir: str, suffix: str) -> str:
    return os.path.join(model_dir, f".reversal_state_{suffix}.json")


def read_reversal_state(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    try:
        return json.loads(open(path, "r").read())
    except Exception:
        return {}


def write_reversal_state(path: str, data: Dict[str, int]):
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

# =========================
# Exchange helpers (Coinbase)
# =========================

import os
from typing import Optional
import ccxt

def make_exchange(pub_key_name: Optional[str], sec_key_name: Optional[str]):
    keyfile = os.path.expanduser("~/.ssh/coinex_keys.env")
    if not os.path.exists(keyfile):
        raise SystemExit(f"[FATAL] Keyfile not found: {keyfile}")

    with open(keyfile, "r") as fh:
        lines = fh.readlines()

    kv = {}
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n").rstrip("\r")
        if not raw or raw.lstrip().startswith("#"):
            i += 1
            continue

        if "=" not in raw:
            # plain line, only possible as part of multi-line PEM handled below
            i += 1
            continue

        key, v0 = raw.split("=", 1)
        key = key.strip()
        v0 = v0.rstrip("\r")

        # Special handling for the secret key: it might be multi-line PEM.
        if sec_key_name and key == sec_key_name:
            value_lines = [v0]
            j = i + 1
            # Collect subsequent lines until we hit an 'END ...PRIVATE KEY-----'
            # or another "KEY=VALUE" line.
            while j < len(lines):
                nxt = lines[j].rstrip("\n").rstrip("\r")
                # Another KEY=VALUE line → stop, do not consume it.
                if "=" in nxt and not nxt.lstrip().startswith("#") and not nxt.startswith(" "):
                    break
                value_lines.append(nxt)
                if "END EC PRIVATE KEY-----" in nxt or "END PRIVATE KEY-----" in nxt:
                    j += 1
                    break
                j += 1

            value = "\n".join(value_lines)
            kv[key] = value
            i = j
            continue

        # Normal single-line KEY=VALUE
        kv[key] = v0.strip()
        i += 1

    api_key = kv.get(pub_key_name) if pub_key_name else None
    api_secret = kv.get(sec_key_name) if sec_key_name else None

    if not api_key or not api_secret:
        raise SystemExit(
            f"[FATAL] Missing Coinbase credentials for pub={pub_key_name!r} "
            f"sec={sec_key_name!r} in {keyfile}"
        )

    # Case 1: stored as single line with literal '\n'
    if "\\n" in api_secret:
        api_secret = api_secret.replace("\\n", "\n")

    # Sanity checks: must look like a PEM EC key
    if "BEGIN" not in api_secret or "PRIVATE KEY" not in api_secret:
        raise SystemExit(
            "[FATAL] Coinbase secret does not look like a PEM EC private key. "
            "Check COINBASE_SECRET formatting in ~/.ssh/coinex_keys.env"
        )

    # Optional debug; keep commented to avoid leaking anything
    # print(f"[DEBUG] api_key={api_key}, secret_len={len(api_secret)}")

    ex = ccxt.coinbase({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    # This will now either succeed or give you a clear 401 if the key/scopes are wrong,
    # but the PEM parsing error should be gone.
    ex.load_markets()
    return ex



def resolve_symbol(ex, ticker: str) -> str:
    ticker = ticker.upper().replace("/", "")
    base = ticker[:-4] if ticker.endswith("USDC") else ticker
    candidates = [s for s in ex.symbols if s.startswith(f"{base}/USDC") and (":USDC" in s)]
    if candidates:
        return sorted(candidates)[0]
    for s in ex.symbols:
        if f"{base}/" in s and ":USDC" in s:
            return s
    raise SystemExit(f"No swap symbol resolved for {ticker} on Coinbase")


def fetch_usdc_balance_swap(ex) -> float:
    try:
        bal = ex.fetch_balance(params={"type": "swap"})
        total = bal.get("total", {}).get("USDC")
        free = bal.get("free", {}).get("USDC")
        return float(free if free is not None else (total or 0.0))
    except Exception:
        return 0.0

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

def get_min_amount(ex, symbol: str) -> float:
    try:
        m = ex.markets.get(symbol) or ex.market(symbol)
        limits = m.get("limits") or {}
        amt = limits.get("amount") or {}
        mn = amt.get("min")
        if mn is not None:
            return float(mn)
    except Exception:
        pass
    return 0.0        

def safe_close_amount(ex, symbol: str, position_size: float) -> float:
    sz = float(abs(position_size))
    if sz <= 0:
        return 0.0

    # Try to derive a step from precision
    step = None
    try:
        m = ex.markets.get(symbol) or ex.market(symbol)
        prec_obj = m.get("precision") or {}
        prec = prec_obj.get("amount", None)
    except Exception:
        prec = None

    if prec is not None:
        try:
            # If prec is an integer like 3 or 4, treat it as decimal places
            if isinstance(prec, int) or (isinstance(prec, float) and prec >= 1 and float(prec).is_integer()):
                decimals = int(prec)
                if 0 <= decimals <= 18:
                    step = 10.0 ** (-decimals)
            # If prec is a small float < 1, treat it as a step size directly (common on Coinbase)
            elif isinstance(prec, float) and 0 < prec < 1:
                step = float(prec)
        except Exception:
            step = None

    if step is not None and step > 0:
        steps = math.floor(sz / step)
        qty = steps * step
        qty = float(f"{qty:.12f}")
        if qty <= 0:
            return 0.0
        return qty

    # Fallback: use amount_to_precision but clamp to <= position size
    try:
        q = float(ex.amount_to_precision(symbol, sz))
        if q > sz:
            eps = max(sz * 1e-6, 1e-12)
            q = float(ex.amount_to_precision(symbol, max(0.0, sz - eps)))
        q = float(f"{q:.12f}")
        return q if q > 0 else 0.0
    except Exception:
        return sz

def get_swap_position(ex, symbol: str) -> Optional[Dict]:
    def _scan(ps):
        if not ps:
            return None
        for p in ps:
            if (p or {}).get("symbol") == symbol:
                side = (p.get("side") or "").lower()
                sz = float(p.get("contracts") or p.get("contractSize") or p.get("size") or 0.0)
                entry = float(p.get("entryPrice") or 0.0)
                if sz and side:
                    return {"side": side, "size": abs(sz), "entry": entry}
        return None

    try:
        pos = _scan(ex.fetch_positions([symbol]))
        if pos:
            return pos
    except Exception:
        pass
    try:
        return _scan(ex.fetch_positions())
    except Exception:
        return None

# =========================
# Inference helpers
# =========================

def run_model(model, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> Tuple[float, float]:
    """
    X shape: (2, lookback, n_features) — first is previous window, second is last window.
    """
    Xn = (X - mean) / (std + 1e-12)

    with torch.no_grad():
        t_prev = torch.from_numpy(Xn[0:1]).float()
        out_prev = model(t_prev)
        if isinstance(out_prev, (list, tuple)):
            out_prev = out_prev[0]
        p_prev = float(torch.sigmoid(out_prev).reshape(-1)[-1].item())

        t_last = torch.from_numpy(Xn[1:2]).float()
        out_last = model(t_last)
        if isinstance(out_last, (list, tuple)):
            out_last = out_last[0]
        p_last = float(torch.sigmoid(out_last).reshape(-1)[-1].item())

    return p_prev, p_last


def _explain_no_open(p_prev: float, p_last: float, pos_thr: float, neg_thr: float) -> str:
    fp = lambda x: f"{x:.3f}"

    in_neutral_now = (neg_thr < p_last < pos_thr)
    in_neutral_prev = (neg_thr < p_prev < pos_thr)

    # 1) Neutral band => close any open position and wait
    if in_neutral_now:
        if p_prev > pos_thr:
            # Came from LONG zone into neutral
            return (
                f"Neutral band: p_last={fp(p_last)} moved down from LONG zone "
                f"(p_prev={fp(p_prev)} ≥ pos_thr={fp(pos_thr)}). "
                "Strategy closes any existing LONG here, but no fresh position is opened."
            )
        if p_prev < neg_thr:
            # Came from SHORT zone into neutral
            return (
                f"Neutral band: p_last={fp(p_last)} moved up from SHORT zone "
                f"(p_prev={fp(p_prev)} ≤ neg_thr={fp(neg_thr)}). "
                "Strategy closes any existing SHORT here, but no fresh position is opened."
            )
        # Stayed inside neutral
        return (
            f"No trade: p_prev={fp(p_prev)} → p_last={fp(p_last)} both inside "
            f"neutral band ({fp(neg_thr)} < p < {fp(pos_thr)}). "
            "We require a cross out of neutral to open a position."
        )

    # 2) Already in LONG or SHORT zone and stayed there => no fresh cross
    if p_last >= pos_thr and p_prev > p_last:
        return (
            f"No new LONG: probability stayed in LONG zone "
            f"(p_prev={fp(p_prev)}  →  p_last={fp(p_last)} ≥ pos_thr={fp(pos_thr)}). "
            f"We only open a LONG when p_last > p_prev."
        )
    if p_last <= neg_thr and p_prev < p_last:
        return (
            f"No new SHORT: probability stayed in SHORT zone "
            f"(p_prev={fp(p_prev)} → p_last={fp(p_last)} ≤ neg_thr={fp(neg_thr)}). "
            f"We only open a LONG when p_last < p_prev."
        )

    # 3) Crossed but not in a valid fresh-cross configuration
    return (
        f"No trade: p_prev={fp(p_prev)}, p_last={fp(p_last)} — does not satisfy "
        f"fresh-cross rules for LONG (p_prev<pos_thr≤p_last) or SHORT (p_prev>neg_thr≥p_last)."
    )


# =========================
# Core flow
# =========================

def decide_and_maybe_trade(args):
    # 1) Load bundle
    bundle = load_bundle(args.model_dir)
    meta = bundle["meta"]
    model = bundle["model"]
    feats = bundle["feature_names"]
    lookback = bundle["lookback"]
    pos_thr, neg_thr = bundle["pos_thr"], bundle["neg_thr"]
    sl_pct, tp_pct = bundle["sl_pct"], bundle["tp_pct"]
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

    # Enforce strict time order & no duplicates
    ts_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else None)
    if ts_col:
        df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)

    # 4) Build + align features
    cols = [c for c in ["timestamp", "ts", "open", "high", "low", "close", "volume"] if c in df.columns]
    feat_df_full = build_features(
        df[cols].copy(),
        ik["tenkan"],
        ik["kijun"],
        ik["senkou"],
        ik["displacement"],
    )
    feat_df_full = align_features_to_meta(feat_df_full, feats)
    for c in feats:
        if c not in feat_df_full.columns:
            raise SystemExit(f"Feature '{c}' missing in computed frame.")
    feat_df = feat_df_full.copy()

    # 5) Inference windows
    feat_mat = feat_df[feats].to_numpy(dtype=np.float32)
    X_prev = feat_mat[-lookback - 1: -1]  # ends at t-1
    X_last = feat_mat[-lookback:]         # ends at t
    X = np.stack([X_prev, X_last], axis=0)

    if args.debug:
        closes = df["close"].to_numpy()
        feat_l1 = float(np.abs(X_last - X_prev).sum())
        print(f"[DEBUG] prev_close→last_close: {closes[-2]:.4f}→{closes[-1]:.4f} | feat_L1_diff={feat_l1:.4f}")

    p_prev, p_last = run_model(model, X, mean, std)
    if args.debug:
        print(f"[DEBUG] proba — prev={p_prev:.8f} last={p_last:.8f} Δ={p_last - p_prev:+.8f}")

    # 6) Time gating (6-minute window after close)
    now_ms = int(time.time() * 1000)
    ts_last_open = int(df["ts"].iloc[-1])
    last_close_ms, stamp_tag, age_ms = resolve_last_closed(now_ms, ts_last_open, timeframe)
    if args.debug:
        print(
            f"[DEBUG] last bar — ts_last_open={ts_last_open} tag={stamp_tag} "
            f"age_min={(age_ms / 60000.0) if age_ms is not None else None}"
        )

    if last_close_ms is None or age_ms is None or not (0 <= age_ms <= SIX_MIN_MS):
        print("Last closed bar is not within the 6-minute window — not acting.")
        return

    # 7) Shared last-bar ID
    update_shared_lastbar(ticker, timeframe, ts_last_open, last_close_ms)
    if args.debug:
        shared = read_lastbars_store().get(f"{ticker}:{timeframe}", {})
        print(f"[DEBUG] shared lastbar: {shared.get('bar_id')} last_close_ts={shared.get('last_close_ts')}")

    # 8) Idempotency
    suffix = f"coinbase_swap_{ticker}_{timeframe}"
    last_seen_ts, guard_path = last_executed_guard(model_dir, suffix)
    rev_state_path = reversal_state_paths(model_dir, suffix)
    rev_state = read_reversal_state(rev_state_path)  # reserved for future use

    if last_seen_ts is not None and last_seen_ts == last_close_ms:
        print("Already acted on this bar for Coinbase — not acting again.")
        return

    # 9) Fresh-cross trigger logic
    take_long = (p_last >= pos_thr) and (p_prev < p_last)
    take_short = (p_last <= neg_thr) and (p_prev > p_last)

    # 10) Exchange (swap only)
    ex = make_exchange(args.pub_key, args.sec_key)
    symbol = resolve_symbol(ex, ticker)

    # 11) Position & SL/TP (+ signal exit on neutral)
    pos = get_swap_position(ex, symbol)
    last_close = float(df["close"].iloc[-1])
    last_high = float(df["high"].iloc[-1])
    last_low = float(df["low"].iloc[-1])

    in_neutral = (neg_thr < p_last < pos_thr)

    if pos is not None and pos.get("side"):
        # WE HAVE AN OPEN POSITION ON EXCHANGE ----
        side_open_raw = pos.get("side")
        side_open = str(side_open_raw).lower()  # 'long' or 'short' ideally
        entry = float(pos.get("entry") or last_close)

        # IMPORTANT: get a safe, positive, precision-checked size that never exceeds the current position
        raw_size = float(pos.get("size") or 0.0)
        sz_abs = abs(raw_size)
        close_qty = safe_close_amount(ex, symbol, sz_abs)

        print(f"[DEBUG] Existing position detected: side={side_open_raw!r}, "
              f"raw_size={raw_size}, close_qty={close_qty}, entry={entry}")

        if close_qty <= 0:
            print("[WARN] Position size is below minimum tradable size after precision; cannot safely close — skipping new trades.")
            return
        else:
            if side_open == "long":
                # LONG SL/TP prices
                sl_px = entry * (1.0 - (sl_pct or 0.0)) if sl_pct is not None else None
                tp_px = entry * (1.0 + (tp_pct or 0.0)) if tp_pct is not None else None
                hit_sl = (sl_px is not None) and (last_low <= sl_px)
                hit_tp = (tp_px is not None) and (last_high >= tp_px)

                # SL/TP first
                if hit_sl or hit_tp:
                    reason = "SL" if hit_sl else "TP"
                    try:
                        ex.create_order(symbol, "market", "sell", close_qty, None, {"reduceOnly": True})
                        print(f"{reason} hit — closing existing LONG at ~{(sl_px if hit_sl else tp_px):.8g}")
                        write_last_executed(guard_path, last_close_ms)
                    except Exception as e:
                        print(f"[ERROR] close LONG on {reason} failed: {e}")
                    return

                # Signal exit: leave LONG zone (neutral or short)
                if p_last < pos_thr:
                    try:
                        ex.create_order(symbol, "market", "sell", close_qty, None, {"reduceOnly": True})
                        zone = "neutral" if in_neutral else "short"
                        print(
                            f"Signal exit — LONG → {zone} zone: "
                            f"p_last={p_last:.3f} < pos_thr={pos_thr:.3f}; closing LONG at ~{last_close}"
                        )
                        write_last_executed(guard_path, last_close_ms)
                    except Exception as e:
                        print(f"[ERROR] close LONG (SIG) failed: {e}")
                    return

            elif side_open == "short":
                # SHORT SL/TP prices
                sl_px = entry * (1.0 + (sl_pct or 0.0)) if sl_pct is not None else None
                tp_px = entry * (1.0 - (tp_pct or 0.0)) if tp_pct is not None else None
                hit_sl = (sl_px is not None) and (last_high >= sl_px)
                hit_tp = (tp_px is not None) and (last_low <= tp_px)

                # SL/TP first
                if hit_sl or hit_tp:
                    reason = "SL" if hit_sl else "TP"
                    try:
                        ex.create_order(symbol, "market", "buy", close_qty, None, {"reduceOnly": True})
                        print(f"{reason} hit — closing existing SHORT at ~{(sl_px if hit_sl else tp_px):.8g}")
                        write_last_executed(guard_path, last_close_ms)
                    except Exception as e:
                        print(f"[ERROR] close SHORT on {reason} failed: {e}")
                    return

                # Signal exit: leave SHORT zone (neutral or long)
                if p_last > neg_thr:
                    try:
                        ex.create_order(symbol, "market", "buy", close_qty, None, {"reduceOnly": True})
                        zone = "neutral" if in_neutral else "long"
                        print(
                            f"Signal exit — SHORT → {zone} zone: "
                            f"p_last={p_last:.3f} > neg_thr={neg_thr:.3f}; closing SHORT at ~{last_close}"
                        )
                        write_last_executed(guard_path, last_close_ms)
                    except Exception as e:
                        print(f"[ERROR] close SHORT (SIG) failed: {e}")
                    return

            else:
                print(f"[WARN] Unknown open side {side_open_raw!r}; not opening new trades.")
                return

            # If we reach here, we decided to keep the position
            print(
                f"Keeping existing {side_open.upper()} open — "
                f"p_last={p_last:.3f}, pos_thr={pos_thr:.3f}, neg_thr={neg_thr:.3f}"
            )
            return

    # 12) Flat & no fresh signal → explanation only
    if not (take_long or take_short):
        msg = _explain_no_open(p_prev, p_last, pos_thr, neg_thr)
        print(msg)
        return

    # 13) OPEN order — balance-based sizing, now with attached TP/SL on exchange
    try:
        quote_bal_swap = fetch_usdt_balance_swap(ex)
        side = "buy" if take_long else "sell"
        usd_to_use = (quote_bal_swap * (0.95 if take_long else 0.80)) if quote_bal_swap > 0 else 0.0
        if usd_to_use <= 0:
            print("No USDT balance available in SWAP.")
            return
        qty_approx = usd_to_use / max(1e-12, last_close)
        qty = amount_to_precision(ex, symbol, qty_approx)

        min_amt = get_min_amount(ex, symbol)
        if min_amt > 0 and qty < min_amt:
            print(f"Calculated order size {qty} is below exchange minimum {min_amt} for {symbol}; skipping trade.")
            return

        if qty <= 0:
            print("Calculated order size is zero after precision rounding.")
            return
        try:
            ex.set_leverage(1, symbol)
        except Exception:
            pass
        px = price_to_precision(ex, symbol, last_close)
        print(f"Opening {('LONG' if side=='buy' else 'SHORT')} (futures/swap) — "
              f"MARKET {side.upper()} {symbol} qty={qty} (px≈{px})")

        # Entry order
        order = ex.create_order(symbol, "market", side, qty, None, {"reduceOnly": False})
        oid = order.get("id") or order.get("orderId") or order
        print(f"Order placed: {oid}")

        # Attach TP/SL as reduce-only orders
        try:
            entry_price = float(order.get("average") or order.get("price") or last_close)

            tp_order_id = None
            sl_order_id = None

            if side == "buy":
                # LONG: TP above, SL below
                if tp_pct is not None and tp_pct > 0:
                    tp_price = price_to_precision(ex, symbol, entry_price * (1.0 + tp_pct))
                    tp = ex.create_order(
                        symbol,
                        "limit",
                        "sell",
                        qty,
                        tp_price,
                        {
                            "reduceOnly": True,
                            "takeProfitPrice": float(tp_price),
                        },
                    )
                    tp_order_id = tp.get("id") or tp.get("orderId") or tp

                if sl_pct is not None and sl_pct > 0:
                    sl_price = price_to_precision(ex, symbol, entry_price * (1.0 - sl_pct))
                    sl = ex.create_order(
                        symbol,
                        "market",
                        "sell",
                        qty,
                        None,
                        {
                            "reduceOnly": True,
                            "stopLossPrice": float(sl_price),
                        },
                    )
                    sl_order_id = sl.get("id") or sl.get("orderId") or sl

            else:
                # SHORT: TP below, SL above
                if tp_pct is not None and tp_pct > 0:
                    tp_price = price_to_precision(ex, symbol, entry_price * (1.0 - tp_pct))
                    tp = ex.create_order(
                        symbol,
                        "limit",
                        "buy",
                        qty,
                        tp_price,
                        {
                            "reduceOnly": True,
                            "takeProfitPrice": float(tp_price),
                        },
                    )
                    tp_order_id = tp.get("id") or tp.get("orderId") or tp

                if sl_pct is not None and sl_pct > 0:
                    sl_price = price_to_precision(ex, symbol, entry_price * (1.0 + sl_pct))
                    sl = ex.create_order(
                        symbol,
                        "market",
                        "buy",
                        qty,
                        None,
                        {
                            "reduceOnly": True,
                            "stopLossPrice": float(sl_price),
                        },
                    )
                    sl_order_id = sl.get("id") or sl.get("orderId") or sl

            if tp_order_id or sl_order_id:
                parts = []
                if tp_order_id:
                    parts.append(f"TP[id={tp_order_id!r}, px={tp_price}]")
                if sl_order_id:
                    parts.append(f"SL[id={sl_order_id!r}, px={sl_price}]")
                print("Attached TP/SL orders — " + ", ".join(parts))

        except Exception as e:
            print(f"[WARN] failed to attach TP/SL orders: {e}")

        # Guard file: one action per bar across brokers
        write_last_executed(guard_path, last_close_ms)

    except Exception as e:
        print(f"[ERROR] order failed: {e}")

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run LSTM bundle; Coinbase FUTURES (swap) orders on fresh signal within "
            "6 minutes — bars read from a JSON file."
        )
    )
    ap.add_argument("--model-dir", required=True, help="Folder with model.pt, preprocess.json, meta.json")
    ap.add_argument("--bars-json", required=True, help="Path to JSON file containing OHLCV bars (closed bars)")
    ap.add_argument("--ticker", default=None, help="Override ticker (otherwise meta.json or BTCUSDT)")
    ap.add_argument("--timeframe", default=None, help="Override timeframe (otherwise meta.json or 1h)")
    ap.add_argument("--auto-transfer", action="store_true", help="Auto-transfer USDT spot→swap before opening futures")
    ap.add_argument("--transfer-buffer", type=float,  default=0.01,   help="Extra fraction to transfer (e.g., 0.01=+1%)")
    ap.add_argument("--reversal",  choices=["close", "flip", "cooldown"],     default="close",     help="How to react on opposite signal: close (default), flip, cooldown" )
    ap.add_argument("--cooldown-seconds",    type=int, default=0,   help="Cooldown duration when --reversal=cooldown")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs")
    ap.add_argument("--pub_key", default=None, help="Name of the API key variable in ~/.ssh/coinex_keys.env" )
    ap.add_argument("--sec_key", default=None,  help="Name of the API secret variable in ~/.ssh/coinex_keys.env")
    return ap.parse_args()


def main():
    args = parse_args()
    decide_and_maybe_trade(args)


if __name__ == "__main__":
    main()