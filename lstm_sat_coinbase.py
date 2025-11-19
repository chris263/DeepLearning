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
from coinbase.rest import RESTClient

import uuid
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
# Exchange helpers (coinbase)
# =========================

from requests import HTTPError

def place_coinbase_perp_order(
    client,
    product_id: str,
    side: str,              # "BUY" or "SELL"
    base_size: float,       # BTC size
    last_close: float
):
    """
    Place a MARKET IOC order on Coinbase Advanced (spot / futures).
    For futures, Coinbase will use your buying power in USD and convert USDC if needed.
    """
    side = side.upper()
    base_str = f"{base_size:.8f}"  # Coinbase likes strings

    payload = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": product_id,  # e.g. "BTC-USDC" (or "BTC-USD" if you change it)
        "side": side,              # "BUY" or "SELL"
        "order_configuration": {
            "market_market_ioc": {
                "base_size": base_str
                # If you want to experiment with quote_size instead:
                # "quote_size": f"{usd_to_use:.2f}"
            }
        },
    }

    print(
        f"Submitting Coinbase MARKET {side}:\n"
        f"  product_id = {product_id}\n"
        f"  base_size  = {base_str} BTC\n"
        f"  est notional ≈ {base_size * last_close:.2f}"
    )
    print(f"[DEBUG] Coinbase order payload: {payload}")

    try:
        resp = client.post(
            "/api/v3/brokerage/orders",
            data=payload,
        )
        # If using the SDK's convenience method instead, you can do:
        # resp = client.market_order_buy(**payload) or market_order_sell(**payload)
    except HTTPError as e:
        # Show full Coinbase error body
        try:
            body = e.response.text
        except Exception:
            body = "<no body>"
        print(f"[ERROR] Coinbase HTTPError: {e} | body={body}")
        raise
    except Exception as e:
        print(f"[ERROR] Generic error placing Coinbase order: {e}")
        raise

    print(f"[OK] Coinbase order response: {resp}")
    return resp


import os
from typing import Optional, Tuple
from coinbase.rest import RESTClient

def make_exchange(pub_key_name: Optional[str],
                  sec_key_name: Optional[str]) -> Tuple[str, str]:
    """
    Load Coinbase Advanced API key + secret from ~/.ssh/coinex_keys.env
    (same file you are already using). Returns (api_key, api_secret).
    """
    api_key = None
    api_secret = None

    if pub_key_name or sec_key_name:
        keyfile = os.path.expanduser("~/.ssh/coinex_keys.env")  # this is the right place!
        kv = {}
        if os.path.exists(keyfile):
            with open(keyfile, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip()
        else:
            raise SystemExit(f"[FATAL] Keyfile not found: {keyfile}")

        api_key = kv.get(pub_key_name) if pub_key_name else None
        api_secret = kv.get(sec_key_name) if sec_key_name else None

    if not api_key or not api_secret:
        raise SystemExit(
            "[FATAL] Missing Coinbase API key and/or secret in ~/.ssh/coinex_keys.env\n"
            f"  pub_key_name={pub_key_name!r}, sec_key_name={sec_key_name!r}"
        )

    return api_key, api_secret

def make_coinbase_client(pub_key_name: str, sec_key_name: str) -> RESTClient:
    api_key, api_secret = make_exchange(pub_key_name, sec_key_name)
    client = RESTClient(api_key=api_key, api_secret=api_secret)

    # --- NEW: verify connection using CFM balance_summary (futures) ---
    try:
        raw = client.get("/api/v3/brokerage/cfm/balance_summary")

        # normalize to dict
        if hasattr(raw, "to_dict"):
            data = raw.to_dict()
        elif isinstance(raw, dict):
            data = raw
        else:
            data = {}

        bs = data.get("balance_summary") or {}
        fb = bs.get("futures_buying_power") or {}
        am = bs.get("available_margin") or {}

        val = fb.get("value") or am.get("value") or "0"
        cur = fb.get("currency") or am.get("currency") or "USD"

        print(
            "[OK] Coinbase connection verified. "
            f"Futures buying power (CFM) = {val} {cur}"
        )
    except Exception as e:
        print(f"[ERROR] Coinbase CFM balance_summary sanity check failed: {e}")
        raise

    return client




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
    """
    Return the exchange-reported minimum tradable amount for this symbol,
    or 0.0 if not available.
    """
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

def resolve_symbol(ex, ticker: str) -> str:
    """
    Normalize any of these:
      - 'BTCUSDC'
      - 'BTC/USDC'
      - 'BTC/USDC:USDC'
      - 'BTCUSDT' or 'BTC/USDT'  (we treat USDT as USDC on Coinbase)
      - 'BTC-USDC' (already good)

    into a Coinbase product_id like 'BTC-USDC'.

    `ex` is kept only for signature compatibility, it is not used.
    """
    if not ticker:
        raise SystemExit("resolve_symbol() got empty ticker")

    t = ticker.upper().strip()

    # If user already passed a Coinbase-style ID like 'BTC-USDC', just keep it
    if "-" in t and len(t.split("-")) == 2:
        return t

    # Strip ccxt style junk: '/', ':' etc.  e.g. 'BTC/USDC:USDC' -> 'BTCUSDCUSDC'
    t_clean = t.replace("/", "").replace(":", "").replace(" ", "")

    # Prefer USDC; also map 'USDT' → 'USDC' for convenience
    if t_clean.endswith("USDC"):
        base = t_clean[:-4]
        return f"{base}-USDC"
    if t_clean.endswith("USDT"):
        base = t_clean[:-4]
        return f"{base}-USDC"   # on Coinbase we use USDC
    if t_clean.endswith("USD"):
        base = t_clean[:-3]
        return f"{base}-USD"

    # Fallback: assume USDC quote
    return f"{t_clean}-USDC"

def fetch_usdc_balance_swap(ex) -> float:
    """
    For Coinbase US Derivatives:
      - read the CFM futures balance summary
      - use futures_buying_power (in USD) as our 'swap balance'.

    This corresponds to the 'Available to trade derivatives' number you see
    in the UI.
    """
    bs = _get_cfm_balance_summary(ex)
    if not bs:
        print("[WARN] No CFM balance_summary returned; using 0 balance.")
        return 0.0

    fb = bs.get("futures_buying_power") or {}
    am = bs.get("available_margin") or {}

    # Prefer futures_buying_power; fall back to available_margin
    val_str = fb.get("value") or am.get("value")
    cur = fb.get("currency") or am.get("currency") or "USD"

    try:
        bal = float(val_str or 0.0)
    except Exception:
        bal = 0.0

    print(f"[DEBUG] CFM balance: futures_buying_power={fb.get('value')} {fb.get('currency')}, "
          f"available_margin={am.get('value')} {am.get('currency')}")
    print(f"[INFO] Using {bal} {cur} as 'swap' balance for sizing Coinbase futures positions.")
    return bal


import math
from typing import Dict, Any, Optional, Tuple, List

# =========================
# Coinbase-specific helpers
# =========================

def safe_close_amount(ex, symbol: str, position_size: float) -> float:
    """
    Coinbase version.

    Compute a close quantity that:
      - respects the product's base_increment; and
      - never exceeds the current position size (floored to step).

    This prevents 'amount exceed limit' / 'insufficient size' errors when the
    true position is slightly smaller than a naive rounded value.
    """
    sz = float(abs(position_size))
    if sz <= 0:
        return 0.0

    base_increment = None

    # Try to get base_increment from Coinbase product metadata
    try:
        prod = ex.get_product(symbol)  # e.g. "BTC-USDC"
        # prod can be a dict or a dataclass-like object
        if isinstance(prod, dict):
            base_increment = prod.get("base_increment")
        elif hasattr(prod, "base_increment"):
            base_increment = getattr(prod, "base_increment")
        elif hasattr(prod, "to_dict"):
            d = prod.to_dict()
            base_increment = d.get("base_increment")
    except Exception as e:
        print(f"[WARN] safe_close_amount: failed to load product info for {symbol}: {e}")

    # If we have base_increment, use it as the step size
    if base_increment is not None:
        try:
            step = float(base_increment)
            if step <= 0:
                raise ValueError("base_increment <= 0")

            steps = math.floor(sz / step)
            qty = steps * step

            # Clamp to sz just in case
            if qty > sz:
                qty = sz

            # Trim floating noise
            qty = float(f"{qty:.12f}")
            if qty <= 0:
                print(
                    f"[WARN] safe_close_amount: rounded quantity is zero "
                    f"(pos={sz}, base_increment={base_increment})"
                )
                return 0.0
            return qty
        except Exception as e:
            print(f"[WARN] safe_close_amount: could not use base_increment={base_increment!r}: {e}")

    # Fallback: send "almost full" size, slightly under, without any precision info
    eps = sz * 1e-6 or 1e-8
    qty = max(0.0, sz - eps)
    qty = float(f"{qty:.12f}")
    if qty <= 0:
        return 0.0
    return qty

def _get_intx_portfolio_uuid(ex) -> Optional[str]:
    """
    Resolve the INTX (perpetuals) portfolio UUID for this key.

    Priority:
      1) COINBASE_INTX_PORTFOLIO_UUID env var (if you set it manually)
      2) Scan get_portfolios() for type == 'INTX'
    """
    portfolio_uuid = os.environ.get("COINBASE_INTX_PORTFOLIO_UUID")
    if portfolio_uuid:
        return portfolio_uuid

    try:
        resp = ex.get_portfolios()
        portfolios = None
        if hasattr(resp, "portfolios"):
            portfolios = resp.portfolios
        elif isinstance(resp, dict):
            portfolios = resp.get("portfolios", [])
        else:
            portfolios = []

        if not portfolios:
            return None

        for pf in portfolios:
            if isinstance(pf, dict):
                pf_type = pf.get("type") or ""
                pf_uuid = pf.get("uuid")
            else:
                pf_type = getattr(pf, "type", "") or ""
                pf_uuid = getattr(pf, "uuid", None)

            if pf_uuid and pf_type == "INTX":
                return pf_uuid

    except Exception as e:
        print(f"[WARN] _get_intx_portfolio_uuid: failed to list portfolios: {e}")

    return None

def _get_cfm_balance_summary(ex) -> dict:
    """
    Coinbase US Derivatives (CFM) balance summary.
    Uses /api/v3/brokerage/cfm/balance_summary.
    """
    try:
        raw = ex.get("/api/v3/brokerage/cfm/balance_summary")
    except Exception as e:
        print(f"[WARN] CFM balance_summary request failed: {e}")
        return {}

    # RESTClient objects usually have .to_dict()
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()

    if not isinstance(raw, dict):
        print(f"[WARN] Unexpected CFM balance_summary payload type: {type(raw)}")
        return {}

    bs = raw.get("balance_summary") or {}
    return bs


def get_swap_position(ex, product_id: str) -> Optional[Dict]:
    raw = ex.get("/api/v3/brokerage/cfm/positions")
    data = raw.to_dict() if hasattr(raw, "to_dict") else raw
    positions = data.get("positions") or []

    for p in positions:
        if p.get("product_id") != product_id:
            continue
        side_raw = (p.get("side") or "").lower()
        side = "long" if "long" in side_raw else ("short" if "short" in side_raw else "")
        contracts = float(p.get("number_of_contracts") or 0.0)
        entry = float(p.get("avg_entry_price") or 0.0)
        if contracts > 0 and side:
            return {"side": side, "size": contracts, "entry": entry}
    return None


def resolve_coinbase_perp_product_id(client, base: str = "BTC") -> str:
    """
    Resolve the Coinbase perpetual FUTURE product_id for a given base, e.g. 'BTC'.

    Uses:
      - product_type = FUTURE
      - future_product_details.contract_expiry_type = PERPETUAL

    Prefers product_venue == 'INTX' when multiple matches exist.
    """
    base_u = base.upper()

    resp = client.get(
        "/api/v3/brokerage/products",
        params={
            "product_type": "FUTURE",
            "contract_expiry_type": "PERPETUAL",
            "limit": 250,
        },
    )
    products = resp.get("products", []) or []

    candidates = []
    for p in products:
        fpd = p.get("future_product_details") or {}
        contract_code = (fpd.get("contract_code") or "").upper()
        root_unit     = (fpd.get("contract_root_unit") or "").upper()
        base_id       = (p.get("base_currency_id") or "").upper()
        disp_name     = (p.get("display_name") or "").upper()

        # Match BTC by contract_code, root_unit, base_id or display_name prefix
        if base_u not in {contract_code, root_unit, base_id} and not disp_name.startswith(base_u + " "):
            continue

        candidates.append(p)

    if not candidates:
        raise SystemExit(f"No perpetual FUTURE product found for base={base_u}")

    # Prefer INTX venue if present
    intx = [p for p in candidates if p.get("product_venue") == "INTX"]
    chosen = intx[0] if intx else candidates[0]

    pid = chosen["product_id"]
    print(
        f"[DEBUG] Resolved Coinbase perp product for base={base_u} -> "
        f"{pid} (venue={chosen.get('product_venue')}, display_name={chosen.get('display_name')})"
    )
    return pid


# =========================
# Inference helpers (unchanged)
# =========================

def run_model(model, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> Tuple[float, float]:
    # X shape: (2, lookback, n_features) — run two independent forwards
    Xn = (X - mean) / (std + 1e-12)

    with torch.no_grad():
        # prev window
        t_prev = torch.from_numpy(Xn[0:1]).float()   # (1, L, C)
        out_prev = model(t_prev)
        if isinstance(out_prev, (list, tuple)):
            out_prev = out_prev[0]
        p_prev = float(torch.sigmoid(out_prev).reshape(-1)[-1].item())

        # last window
        t_last = torch.from_numpy(Xn[1:2]).float()   # (1, L, C)
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
            f"We only open a SHORT when p_last < p_prev."
        )

    # 3) Crossed but not in a valid fresh-cross configuration
    return (
        f"No trade: p_prev={fp(p_prev)}, p_last={fp(p_last)} — does not satisfy "
        f"fresh-cross rules for LONG (p_prev<pos_thr≤p_last) or SHORT (p_prev>neg_thr≥p_last)."
    )

# =========================
# Core flow (Coinbase perps, close-only reversal)
# =========================
def decide_and_maybe_trade(args):
    # 1) Load bundle
    bundle = load_bundle(args.model_dir)
    meta  = bundle["meta"]
    model = bundle["model"]
    feats = bundle["feature_names"]
    lookback        = bundle["lookback"]
    pos_thr, neg_thr= bundle["pos_thr"], bundle["neg_thr"]
    sl_pct, tp_pct  = bundle["sl_pct"],  bundle["tp_pct"]
    mean, std       = bundle["mean"],    bundle["std"]
    ik              = bundle["ichimoku"]
    model_dir       = bundle["paths"]["dir"]

    # 2) Resolve ticker/timeframe (model defaults)
    ticker    = args.ticker    or meta.get("ticker")    or "BTCUSDC"
    timeframe = args.timeframe or meta.get("timeframe") or "1h"

    # 3) Load bars (JSON)
    df = load_bars_from_json(args.bars_json)
    if df is None or len(df) < (lookback + 3):
        print("Not enough bars to build features.")
        return

    # Build + align features & check names
    feat_df_full = build_features(
        df[["timestamp", "ts", "open", "high", "low", "close", "volume"]].copy(),
        ik["tenkan"], ik["kijun"], ik["senkou"], ik["displacement"]
    )
    feat_df_full = align_features_to_meta(feat_df_full, feats)
    for c in feats:
        if c not in feat_df_full.columns:
            raise SystemExit(f"Feature '{c}' missing in computed frame.")
    feat_df = feat_df_full.copy()

    # 4) Inference (prev vs last bar) — explicit windows
    feat_mat = feat_df[feats].to_numpy(dtype=np.float32)
    X_prev   = feat_mat[-lookback-1 : -1]
    X_last   = feat_mat[-lookback   :   ]
    X        = np.stack([X_prev, X_last], axis=0)

    if getattr(args, "debug", False):
        closes  = df["close"].to_numpy()
        feat_l1 = float(np.abs(X_last - X_prev).sum())
        print(f"[DEBUG] prev_close→last_close: {closes[-2]:.4f}→{closes[-1]:.4f} | feat_L1_diff={feat_l1:.6g}")

    p_prev, p_last = run_model(model, X, mean, std)
    if getattr(args, "debug", False):
        print(f"[DEBUG] proba — prev={p_prev:.8f} last={p_last:.8f} Δ={p_last-p_prev:+.8f}")

    # 5) Time gating (6-minute window after close)
    now_ms       = int(time.time() * 1000)
    ts_last_open = int(df["ts"].iloc[-1])
    last_close_ms, stamp_tag, age_ms = resolve_last_closed(now_ms, ts_last_open, timeframe)
    if args.debug:
        age_min = (age_ms / 60000.0) if age_ms is not None else None
        print(f"[DEBUG] last bar — ts_last_open={ts_last_open} tag={stamp_tag} age_min={age_min}")

    if last_close_ms is None or age_ms is None or not (0 <= age_ms <= SIX_MIN_MS):
        print("Last closed bar is not within the 6-minute window — not acting.")
        return

    # 6) Shared last-bar ID (cross-broker guard)
    update_shared_lastbar(ticker, timeframe, ts_last_open, last_close_ms)
    if args.debug:
        shared = read_lastbars_store().get(f"{ticker}:{timeframe}", {})
        print(f"[DEBUG] shared lastbar: {shared.get('bar_id')} last_close_ts={shared.get('last_close_ts')}")

    # 7) Guard & reversal state
    suffix         = f"coinbase_perp_{ticker}_{timeframe}"
    last_seen_ts, guard_path = last_executed_guard(model_dir, suffix)
    rev_state_path = reversal_state_paths(model_dir, suffix)
    rev_state      = read_reversal_state(rev_state_path)

    if last_seen_ts is not None and last_seen_ts == last_close_ms:
        print("Already acted on this bar for Coinbase — not acting again.")
        return

    # 8) Fresh-cross trigger logic (same rules)
    take_long  = (p_last >= pos_thr) and (p_prev <  p_last)
    take_short = (p_last <= neg_thr) and (p_prev >  p_last)

    # 9) Coinbase client + product_id
    client = make_coinbase_client(args.pub_key, args.sec_key)

    # Derive base symbol from model ticker (BTCUSDC, BTC/USDC, etc.)
    raw_ticker = (ticker or "").upper()
    base = raw_ticker.replace("/", "").replace(":", "")
    for suf in ("USDC", "USDT", "USD"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if not base:
        base = "BTC"  # safe fallback

    product_id = resolve_coinbase_perp_product_id(client, base=base)


    # 10) Position & SL/TP (+ signal exit on neutral)
    pos       = get_swap_position(client, product_id)  # your Coinbase version
    last_close= float(df["close"].iloc[-1])
    last_high = float(df["high"].iloc[-1])
    last_low  = float(df["low"].iloc[-1])

    in_neutral = (neg_thr < p_last < pos_thr)

    # --- if we have an open position, we only manage/close it; no new opens this bar ---
    if pos is not None and pos.get("side"):
        side_open_raw = pos.get("side")
        side_open     = str(side_open_raw).lower()  # 'long' or 'short'
        entry         = float(pos.get("entry") or last_close)
        raw_size      = float(pos.get("size") or 0.0)

        if raw_size <= 0:
            print(
                f"[WARN] Coinbase position reported with non-positive size "
                f"({raw_size}); treating as flat."
            )
            pos = None
        else:
            print(
                f"[DEBUG] Existing position detected on Coinbase: side={side_open_raw!r}, "
                f"size={raw_size}, entry={entry}"
            )

            if side_open == "long":
                # LONG SL/TP prices
                sl_px = entry * (1.0 - (sl_pct or 0.0)) if sl_pct is not None else None
                tp_px = entry * (1.0 + (tp_pct or 0.0)) if tp_pct is not None else None
                hit_sl = (sl_px is not None) and (last_low  <= sl_px)
                hit_tp = (tp_px is not None) and (last_high >= tp_px)

                # 10a) SL/TP first (backtest parity)
                if hit_sl or hit_tp:
                    reason = "SL" if hit_sl else "TP"
                    try:
                        client.close_position(product_id=product_id)
                        print(
                            f"{reason} hit — closing existing LONG on Coinbase at "
                            f"~{(sl_px if hit_sl else tp_px):.8g}"
                        )
                        write_last_executed(guard_path, last_close_ms)
                    except Exception as e:
                        print(f"[ERROR] close LONG on {reason} (Coinbase) failed: {e}")
                    return

                # 10b) SIGNAL EXIT: leave LONG zone → neutral or short
                if p_last < pos_thr:
                    try:
                        client.close_position(product_id=product_id)
                        zone = "neutral" if in_neutral else "short"
                        print(
                            f"Signal exit — LONG → {zone} zone on Coinbase: "
                            f"p_last={p_last:.3f} < pos_thr={pos_thr:.3f}; closing LONG at ~{last_close}"
                        )
                        write_last_executed(guard_path, last_close_ms)
                    except Exception as e:
                        print(f"[ERROR] close LONG (SIG, Coinbase) failed: {e}")
                    return

            elif side_open == "short":
                # SHORT SL/TP prices
                sl_px = entry * (1.0 + (sl_pct or 0.0)) if sl_pct is not None else None
                tp_px = entry * (1.0 - (tp_pct or 0.0)) if tp_pct is not None else None
                hit_sl = (sl_px is not None) and (last_high >= sl_px)
                hit_tp = (tp_px is not None) and (last_low  <= tp_px)

                # 10c) SL/TP first
                if hit_sl or hit_tp:
                    reason = "SL" if hit_sl else "TP"
                    try:
                        client.close_position(product_id=product_id)
                        print(
                            f"{reason} hit — closing existing SHORT on Coinbase at "
                            f"~{(sl_px if hit_sl else tp_px):.8g}"
                        )
                        write_last_executed(guard_path, last_close_ms)
                    except Exception as e:
                        print(f"[ERROR] close SHORT on {reason} (Coinbase) failed: {e}")
                    return

                # 10d) SIGNAL EXIT: leave SHORT zone → neutral or long
                if p_last > neg_thr:
                    try:
                        client.close_position(product_id=product_id)
                        zone = "neutral" if in_neutral else "long"
                        print(
                            f"Signal exit — SHORT → {zone} zone on Coinbase: "
                            f"p_last={p_last:.3f} > neg_thr={neg_thr:.3f}; closing SHORT at ~{last_close}"
                        )
                        write_last_executed(guard_path, last_close_ms)
                    except Exception as e:
                        print(f"[ERROR] close SHORT (SIG, Coinbase) failed: {e}")
                    return

            else:
                print(f"[WARN] Unknown open side {side_open!r} on Coinbase; not opening new trades.")
                return

            # If we reach here, we keep the existing position and do not flip this bar
            print(
                f"Keeping existing {side_open.upper()} open on Coinbase — "
                f"p_last={p_last:.3f}, pos_thr={pos_thr:.3f}, neg_thr={neg_thr:.3f}"
            )
            return

    # 11) If flat and no fresh signal, do nothing
    if not (take_long or take_short):
        msg = _explain_no_open(p_prev, p_last, pos_thr, neg_thr)
        print(msg)
        return

    # 12) No auto-transfer on Coinbase (user manages collateral manually)
    # (nothing here by design)

    # 13) OPEN order — Coinbase futures/perps (CFM), balance-based sizing
    try:
        # 13a) Get futures buying power (CFM)
        cb_usdc_bal = fetch_usdc_balance_swap(client)  # uses /cfm/balance_summary
        if cb_usdc_bal <= 0:
            print("No USDC balance available on Coinbase futures/perps.")
            return

        # Risk fraction of buying power
        risk_long  = 0.80
        risk_short = 0.80
        usd_to_use = cb_usdc_bal * (risk_long if take_long else risk_short)
        if usd_to_use <= 0:
            print("Calculated usd_to_use is zero or negative; skipping trade.")
            return

        # 13b) Price of last bar
        last_close = float(df["close"].iloc[-1])

        # 13c) Get product increments (base_increment, base_min_size)
        try:
            prod_raw = client.get_product(product_id)
            prod = prod_raw.to_dict() if hasattr(prod_raw, "to_dict") else prod_raw
            base_inc = float(prod.get("base_increment") or "0.0001")
            base_min = float(prod.get("base_min_size") or "0.0001")
        except Exception as e:
            print(f"[WARN] Failed to load product increments, using defaults: {e}")
            base_inc = 0.0001
            base_min = 0.0001

        # 13d) Convert USD notional -> base size and floor to base_increment
        base_size_raw = usd_to_use / max(1e-12, last_close)
        steps = math.floor(base_size_raw / base_inc)
        base_size = steps * base_inc

        if base_size < base_min:
            print(
                f"Calculated base_size={base_size:.8f} below base_min_size={base_min:.8f}; "
                "skipping Coinbase trade."
            )
            return

        print(
            f"[DEBUG] CB_USDC_bal={cb_usdc_bal:.4f} "
            f"usd_to_use={usd_to_use:.4f} "
            f"base_size_raw={base_size_raw:.8f} "
            f"base_size={base_size:.8f} "
            f"notional≈{base_size * last_close:.4f}"
        )

        # 13e) Place the perp order
        side = "BUY" if take_long else "SELL"
        print(
            f"Opening {'LONG' if side=='BUY' else 'SHORT'} (Coinbase perps) — "
            f"MARKET {side} {product_id} base_size={base_size:.8f} (px≈{last_close:.2f})"
        )

        place_coinbase_perp_order(
            client=client,
            product_id=product_id,
            side=side,
            base_size=base_size,
            last_close=last_close,
        )

        # Guard: one action per bar
        write_last_executed(guard_path, last_close_ms)

    except Exception as e:
        print(f"[ERROR] Coinbase order failed: {e}")


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Run LSTM bundle; Coinbase FUTURES (swap, unified) orders on fresh signal within 6 minutes — bars read from a JSON file."
    )
    ap.add_argument("--model-dir", required=True, help="Folder with model.pt, preprocess.json, meta.json")
    ap.add_argument("--bars-json", required=True, help="Path to JSON file containing OHLCV bars (closed bars)")
    ap.add_argument("--ticker", default=None, help="Override ticker (otherwise meta.json or BTCUSDT)")
    ap.add_argument("--timeframe", default=None, help="Override timeframe (otherwise meta.json or 1h)")
    ap.add_argument("--auto-transfer", action="store_true", help="Try to top up USDT margin for swap if balance is low")
    ap.add_argument("--transfer-buffer", type=float, default=0.01, help="Extra fraction to transfer (e.g., 0.01=+1%)")
    ap.add_argument("--reversal", choices=["close","flip","cooldown"], default="close",
                    help="How to react on opposite signal: close (default), flip, cooldown")
    ap.add_argument("--cooldown-seconds", type=int, default=0, help="Cooldown duration when --reversal=cooldown")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs")

    ap.add_argument("--pub_key", default=None, help="Name of API key var (e.g., API_KEY_BYBIT)")
    ap.add_argument("--sec_key", default=None, help="Name of API secret var (e.g., API_SECRET_BYBIT)")
    ap.add_argument("--keys-file", default=None, help="Env file path with KEY=VALUE pairs (default ~/.ssh/coinex_keys.env)")
    return ap.parse_args()

def main():
    args = parse_args()
    decide_and_maybe_trade(args)

if __name__ == "__main__":
    main()
