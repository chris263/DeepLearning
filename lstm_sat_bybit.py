#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Ichimoku runner for Bybit (USDT linear swap, unified account).

- Always futures (swap). No args.category.
- Fresh-cross trigger (old logic).
- Close-only on reversal by default (new logic). Optional: --reversal flip|cooldown
- SL/TP exits from meta.json (sl_pct / tp_pct).
- Balance-based sizing: 100% long, 80% short (USDT in unified).
- Feature-name alignment to meta["features"].
- 6-minute window after last bar close (open+step).
- Shared last-bar ID at ~/.sat_state/lastbars.json (override via SAT_STATE_DIR).
- Per-broker idempotency file in model dir.

Requires: torch, ccxt, numpy, pandas
"""

from __future__ import annotations
import os, re, sys, json, time, argparse, pathlib, tempfile, math
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import datetime

import torch
import ccxt




# =========================
# Constants / timeframe helpers
# =========================
SIX_MIN_MS = 6 * 60 * 1000
TF_TO_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
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
    if ts < 10_000_000_000:
        ts *= 1000
    try:
        o = float(g("open", "o")); h = float(g("high", "h"))
        l = float(g("low", "l"));  c = float(g("close", "c"))
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
    # JSON array or wrapped object
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
    d["kijun"] = rolling_mid(d.high, d.low, kijun)
    d["span_a"] = (d["tenkan"] + d["kijun"]) / 2.0
    d["span_b"] = rolling_mid(d.high, d.low, senkou)
    d["cloud_top"] = d[["span_a", "span_b"]].max(axis=1)
    d["cloud_bot"] = d[["span_a", "span_b"]].min(axis=1)
    return d

def slope(series: pd.Series, w: int = 8) -> pd.Series:
    return series.diff(w)

# --- Feature synonyms to align to meta["features"] ---
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
    d["tk_slope"] = (d["tenkan"] - d["tenkan"].shift(sw)) / (d["px"] + 1e-9)
    d["kj_slope"] = (d["kijun"] - d["kijun"].shift(sw)) / (d["px"] + 1e-9)
    d["span_a_slope"] = (d["span_a"] - d["span_a"].shift(sw)) / (d["px"] + 1e-9)
    d["span_b_slope"] = (d["span_b"] - d["span_b"].shift(sw)) / (d["px"] + 1e-9)

    D = int(displacement)
    d["chikou_above"] = (d["close"] > d["close"].shift(D)).astype(float)
    d["vol20"] = d["ret1"].rolling(20, min_periods=20).std().fillna(0)
    d["ts"] = df["ts"].values
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

    feature_names = meta.get("features") or meta.get("feature_names")
    if not feature_names:
        raise SystemExit("meta.json missing 'features' list")

    lookback = int(meta.get("lookback") or meta.get("window") or 64)
    pos_thr  = float(meta.get("pos_thr", 0.55))
    neg_thr  = float(meta.get("neg_thr", 0.45))

    # --- risk (nested first, then legacy top-level) ---
    risk = meta.get("risk") or {}
    sl_pct = risk.get("sl_pct", meta.get("sl_pct"))
    tp_pct = risk.get("tp_pct", meta.get("tp_pct"))
    fee_bps = risk.get("fee_bps", meta.get("fee_bps"))  # not required by runners, but available

    sl_pct = float(sl_pct) if sl_pct is not None else None
    tp_pct = float(tp_pct) if tp_pct is not None else None

    # --- ichimoku (nested first, then legacy top-level, then defaults) ---
    ik = meta.get("ichimoku") or {}
    ichimoku_params = {
        "tenkan":       int(ik.get("tenkan",       meta.get("tenkan", 9))),
        "kijun":        int(ik.get("kijun",        meta.get("kijun", 26))),
        "senkou":       int(ik.get("senkou",       meta.get("senkou", 52))),
        "displacement": int(ik.get("displacement", meta.get("displacement", 26))),
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
        # "fee_bps": float(fee_bps) if fee_bps is not None else None,  # expose if you plan to use it
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
# Daily profit goal 2%
# =========================


DAILY_PROFIT_TARGET_PCT = float(os.getenv("DAILY_PROFIT_TARGET_PCT", "0.02"))  # 2% default

def get_equity_for_daily_guard(ex) -> float:
    """
    Try to get a realistic account equity in USDT/USDC to use as the
    baseline for daily % PnL. Prefer total USDT balance; fall back
    to swap-only free balance if needed.

    IMPORTANT:
    - This value is only used when creating a NEW day state.
      On the same day, equity_start will NOT be changed.
    """
    # 1) Prefer total account USDT/USDC (ccxt-style)
    try:
        bal = ex.fetch_balance()
        for code in ("USDT", "USDC"):
            acc = bal.get(code)
            if isinstance(acc, dict):
                v = acc.get("total") or acc.get("free") or acc.get("used")
                if v is not None:
                    eq = float(v)
                    print(f"[DAILY PROFIT] Using {code} total={eq:.2f} as total (including unrealized).")
                    return eq
    except Exception as e:
        print(f"[WARN] fetch_balance() failed for daily guard, falling back to swap-only: {e}")

    # 2) Fallback: swap-only free balance (what we used before)
    try:
        eq = float(fetch_usdt_balance_swap(ex))
        print(f"[DAILY PROFIT] Using swap-only balance={eq:.2f} as equity baseline (fallback).")
        return eq
    except Exception as e:
        print(f"[WARN] fetch_usdt_balance_swap() failed for daily guard, equity_now=0: {e}")
        return 0.0


def _save_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_daily_profit_state(path: str, today: str, equity_now: float) -> Dict[str, Any]:
    """
    Load or (if new day) initialize the daily profit state.

    equity_now: current account equity in quote currency (e.g. USDT/USDC).
                Used ONLY when creating a NEW day state.
    """
    st: Dict[str, Any] = {}

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                st = json.load(f)
        except Exception as e:
            print(f"[DAILY PROFIT] failed to read {path}: {e}; resetting state.")
            st = {}

    # 1) New or different day → reset with new equity_start
    if not st or st.get("date") != today:
        eq0 = float(equity_now or 0.0)
        st = {
            "date": today,
            "equity_start": eq0,          # FIXED for the whole day
            "current_balance": eq0,       # realized balance after last closed trade
            "realized_pnl": 0.0,          # derived from current_balance - equity_start
            "daily_pct": 0.0,
            "hit_target": False,
        }
        _save_json_atomic(path, st)
        print(
            f"[DAILY PROFIT] Initialized daily state (Bybit): "
            f"date={today}, equity_start={eq0:.2f}"
        )
        return st

    # 2) Same day: DO NOT change equity_start
    eq0 = float(st.get("equity_start", 0.0) or 1.0)

    # Ensure current_balance exists
    if "current_balance" not in st:
        # If missing (old file), infer from equity_now if provided, else from equity_start
        if equity_now is not None:
            st["current_balance"] = float(equity_now)
        else:
            st["current_balance"] = eq0

    cur_bal = float(st.get("current_balance", eq0))

    # Realized PnL derived from current_balance - equity_start
    realized = cur_bal - eq0
    st["realized_pnl"] = realized
    st["daily_pct"] = realized / eq0 if eq0 > 0 else 0.0

    st.setdefault("hit_target", False)
    st.setdefault("date", today)

    _save_json_atomic(path, st)
    return st


def log_daily_status(state: Dict[str, Any], target_pct: float) -> None:
    """
    Log daily status based purely on realized data:
      realized_pnl = current_balance - equity_start
    """
    eq0 = float(state.get("equity_start", 0.0) or 1.0)
    cur_bal = float(state.get("current_balance", eq0))
    realized = cur_bal - eq0
    pct = realized / eq0 if eq0 > 0 else 0.0
    date = state.get("date", "?")
    hit = bool(state.get("hit_target"))

    print(
        f"[DAILY PROFIT STATUS] date={date} | "
        f"equity_start={eq0:.2f} | "
        f"current_balance={cur_bal:.2f} | "
        f"realized={realized:.2f} ({pct*100:.2f}%) | "
        f"target={target_pct*100:.2f}% | hit_target={hit} | "
        f"[GUARD uses REALIZED PnL derived from (current_balance - equity_start)]"
    )


def record_realized_pnl(path: str, state: Dict[str, Any], pnl_quote: float, target_pct: float) -> Dict[str, Any]:
    """
    Update the daily realized PnL after a trade is closed.

    pnl_quote: realized profit/loss in quote currency (e.g. USDT).
               >0 = profit, <0 = loss.

    All PnL is derived from:
        current_balance = previous_current_balance + pnl_quote
        realized_pnl    = current_balance - equity_start
    """
    eq0 = float(state.get("equity_start", 0.0) or 0.0)
    if eq0 <= 0:
        # Very defensive: if equity_start is zero or missing, warn and avoid division issues
        print("[DAILY PROFIT] WARNING: equity_start is 0 or missing in state; "
              "initializing to 1.0 just for percentage math.")
        eq0 = 1.0

    # Ensure current_balance exists
    if "current_balance" not in state:
        # This usually means old JSON from before we added the field
        print("[DAILY PROFIT] WARNING: 'current_balance' missing in state; "
              "initializing it from equity_start.")
        state["current_balance"] = float(state.get("equity_start", 0.0) or 0.0)

    # Previous derived values
    cur_bal_prev = float(state.get("current_balance", eq0))
    prev_realized = cur_bal_prev - eq0
    prev_pct = prev_realized / eq0 if eq0 > 0 else 0.0

    # New balance and derived PnL
    pnl_quote = float(pnl_quote)
    cur_bal_new = cur_bal_prev + pnl_quote
    new_realized = cur_bal_new - eq0
    daily_pct = new_realized / eq0 if eq0 > 0 else 0.0

    state["current_balance"] = cur_bal_new
    state["realized_pnl"] = new_realized
    state["daily_pct"] = daily_pct

    hit_before = bool(state.get("hit_target"))
    hit_after = daily_pct >= target_pct
    state["hit_target"] = hit_after

    _save_json_atomic(path, state)

    # Main status line
    print(
        "[DAILY PROFIT] "
        f"P&LΔ={pnl_quote:+.2f} | "
        f"realized={prev_realized:.2f}→{new_realized:.2f} "
        f"({prev_pct*100:.2f}%→{daily_pct*100:.2f}%) | "
        f"current_balance={cur_bal_prev:.2f}→{cur_bal_new:.2f} | "
        f"target={target_pct*100:.2f}% | hit_target={hit_after}"
    )

    # One-time banner when target is crossed
    if (not hit_before) and hit_after:
        print("=" * 72)
        print(
            f"[DAILY PROFIT GUARD] 🎯 Daily target reached! +{daily_pct*100:.2f}% "
            f"(target={target_pct*100:.2f}%)."
        )
        print("[DAILY PROFIT GUARD] No NEW positions will be opened for the rest of this day.")
        print("=" * 72)

    return state


def daily_guard_blocks_new_trades(state: Dict[str, Any], target_pct: float) -> bool:
    """
    Returns True if we must NOT open new positions today.

    HARD RULE:
      - If daily_pct >= target_pct OR hit_target is True → BLOCK new trades.
    """
    eq0 = float(state.get("equity_start", 0.0) or 0.0)
    if eq0 <= 0:
        # If somehow equity_start is missing/zero, don't blow up; just treat pct as 0
        print("[DAILY PROFIT] WARNING: equity_start is 0 or missing when checking guard.")
        eq0 = 1.0

    if "current_balance" not in state:
        # This really shouldn't happen anymore after at least one record_realized_pnl,
        # but let's be safe and noisy.
        print("[DAILY PROFIT] WARNING: 'current_balance' missing when checking guard; "
              "initializing from equity_start.")
        state["current_balance"] = float(state.get("equity_start", 0.0) or 0.0)

    cur_bal = float(state.get("current_balance", eq0))

    daily_pct = (cur_bal - eq0) / eq0 if eq0 > 0 else 0.0
    state["daily_pct"] = daily_pct
    state["realized_pnl"] = cur_bal - eq0  # keep realized_pnl consistent too

    if daily_pct >= target_pct:
        return True

    if state.get("hit_target"):
        return True

    if daily_pct <= -2.0:
        print(
            "[DAILY SL] [BLOCK] "
            f"equity_now={equity_now:.2f} | "
            f"equity_start={eq0:.2f} | "
            f"realized={realized:.2f} ({daily_pct*100:.2f}%) | "
            f"No more trades today!"
        )
        return True

    # (Optional: could also block if daily_pct <= -max_loss_pct)
    return False

def refresh_daily_state_with_equity(
    path: str,
    state: Dict[str, Any],
    equity_now: float,
    target_pct: float,
) -> Dict[str, Any]:
    """
    Hard-sync the daily state with the *actual* account equity from the exchange.

    Use this when you want:
      current_balance = equity_now_from_exchange
      realized_pnl    = current_balance - equity_start

    NOTE:
      - Best used when FLAT (no open position), otherwise you are mixing
        floating PnL into the "realized" PnL measure.
    """
    eq0_raw = state.get("equity_start", 0.0)
    eq0 = float(eq0_raw if eq0_raw is not None else 0.0)
    if eq0 <= 0:
        print("[DAILY PROFIT] WARNING: equity_start is 0 or missing in state when refreshing from equity.")
        # Avoid division by zero; we keep eq0 as 1.0 only for percentage math.
        eq0 = 1.0

    equity_now = float(equity_now or 0.0)

    state["current_balance"] = equity_now
    realized = equity_now - eq0
    state["realized_pnl"] = realized
    daily_pct = realized / eq0 if eq0 > 0 else 0.0
    state["daily_pct"] = daily_pct

    hit_before = bool(state.get("hit_target"))
    hit_after = daily_pct >= target_pct
    state["hit_target"] = hit_after

    _save_json_atomic(path, state)

    print(
        "[DAILY PROFIT] [REFRESH] "
        f"equity_now={equity_now:.2f} | "
        f"equity_start={eq0:.2f} | "
        f"realized={realized:.2f} ({daily_pct*100:.2f}%) | "
        f"target={target_pct*100:.2f}% | hit_target={hit_after}"
    )

    if (not hit_before) and hit_after:
        print("=" * 72)
        print(
            f"[DAILY PROFIT GUARD] 🎯 Daily target reached (via refresh)! "
            f"+{daily_pct*100:.2f}% (target={target_pct*100:.2f}%)."
        )
        print("[DAILY PROFIT GUARD] No NEW positions will be opened for the rest of this day.")
        print("=" * 72)

    return state


# =========================
# Shared last-bar ID (cross-broker)
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
        "ticker": ticker, "timeframe": timeframe,
        "last_open_ts": int(last_open_ts_ms), "last_close_ts": int(last_close_ms),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_lastbars_store(store)


# =========================
# Per-broker idempotency & reversal state
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
# Exchange helpers (Bybit unified swap)
# =========================
def _read_kv_file(path: str) -> Dict[str, str]:
    kv = {}
    if os.path.exists(path):
        for line in open(path, "r"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv

def make_exchange(pub_key_name: Optional[str], sec_key_name: Optional[str], keys_file: Optional[str] = None):
    keyfile = os.path.expanduser(keys_file or os.getenv("BYBIT_KEYS_FILE", "~/.ssh/coinex_keys.env"))
    keyfile = os.path.expanduser(keyfile)
    kv = _read_kv_file(keyfile)
    api_key = kv.get(pub_key_name) if pub_key_name else None
    api_secret = kv.get(sec_key_name) if sec_key_name else None
    ex = ccxt.bybit({
        "apiKey": api_key, "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "hedgeMode": False, "accountType": "UNIFIED"}
    })
    ex.load_markets()
    return ex

def resolve_symbol(ex, ticker: str) -> str:
    ticker = ticker.upper().replace("/", "")
    base = ticker[:-4] if ticker.endswith("USDT") else ticker
    candidates = [s for s in ex.symbols if s.startswith(f"{base}/USDT") and (":USDT" in s)]
    if candidates:
        return sorted(candidates)[0]
    for s in ex.symbols:
        if f"{base}/" in s and ":USDT" in s:
            return s
    raise SystemExit(f"No swap symbol resolved for {ticker} on Bybit")

def fetch_usdt_balance_swap(ex) -> float:
    try:
        bal = ex.fetch_balance(params={"type": "swap", "accountType": "UNIFIED"})
        total = bal.get("total", {}).get("USDT"); free  = bal.get("free", {}).get("USDT")
        return float(free if free is not None else (total or 0.0))
    except Exception:
        try:
            bal = ex.fetch_balance()
            total = bal.get("total", {}).get("USDT"); free  = bal.get("free", {}).get("USDT")
            return float(free if free is not None else (total or 0.0))
        except Exception:
            return 0.0

def transfer_spot_to_swap_if_needed(ex, min_usdt: float, buffer_frac: float = 0.01, debug: bool = False) -> float:
    try:
        swap_free = fetch_usdt_balance_swap(ex)
        if swap_free >= min_usdt:
            return swap_free
        if debug:
            print(f"[DEBUG] swap USDT={swap_free:.2f} < {min_usdt:.2f}; attempting top-up...")
        bal = ex.fetch_balance()
        spot_free = float(bal.get("free", {}).get("USDT") or 0.0)
        if spot_free <= 0:
            return swap_free
        amt = min(spot_free, min_usdt * (1.0 + max(0.0, buffer_frac)))
        if debug:
            print(f"[DEBUG] attempting internal transfer of {amt:.6f} USDT to derivatives margin (if supported)")
        try:
            ex.transfer(code="USDT", amount=float(f"{amt:.6f}"), fromAccount="spot", toAccount="contract")
            time.sleep(0.5)
        except Exception:
            pass
        return fetch_usdt_balance_swap(ex)
    except Exception as e:
        print(f"[WARN] auto-transfer failed: {e}")
        return fetch_usdt_balance_swap(ex)

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


def safe_close_amount(ex, symbol: str, position_size: float) -> float:
    """
    Compute a close quantity that:
      - respects the symbol's amount precision; and
      - never exceeds the current position size (floor to step).

    This prevents 'amount exceed limit' / 'insufficient size' type errors
    when the true position is slightly smaller than the rounded value.
    """
    sz = float(abs(position_size))
    if sz <= 0:
        return 0.0

    # Try to floor using the amount precision (number of decimals)
    try:
        m = ex.markets.get(symbol) or {}
        prec_obj = m.get("precision") or {}
        prec = prec_obj.get("amount", None)
    except Exception:
        prec = None

    if prec is not None:
        try:
            decimals = int(prec)
            step = 10.0 ** (-decimals)
            steps = math.floor(sz / step)
            qty = steps * step
            qty = float(f"{qty:.12f}")
            return qty if qty > 0 else 0.0
        except Exception:
            pass

    # Fallback: clamp amount_to_precision result to <= position size
    try:
        q = float(ex.amount_to_precision(symbol, sz))
        if q > sz:
            eps = sz * 1e-6 or 1e-8
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
                sz   = float(p.get("contracts") or p.get("size") or p.get("contractSize") or 0.0)
                entry= float(p.get("entryPrice") or 0.0)
                if sz and side:
                    return {"side": side, "size": abs(sz), "entry": entry}
        return None
    try:
        pos = _scan(ex.fetch_positions([symbol], params={"accountType": "UNIFIED"}))
        if pos: return pos
    except Exception:
        pass
    try:
        return _scan(ex.fetch_positions(params={"accountType": "UNIFIED"}))
    except Exception:
        return None


# =========================
# Inference helpers
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

## SL Protection
def _sl_guard_file(guard_path: str) -> str:
    # store per-script SL info right next to the normal guard file
    return guard_path + ".sl_guard.json"

def load_sl_guard(guard_path: str) -> dict:
    path = _sl_guard_file(guard_path)
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"active": False, "neutral_seen": False}
            return {
                "active": bool(data.get("active", False)),
                "neutral_seen": bool(data.get("neutral_seen", False)),
                "sl_ts": data.get("sl_ts"),
                "sl_side": data.get("sl_side"),
            }
    except Exception:
        return {"active": False, "neutral_seen": False}

def save_sl_guard(guard_path: str, data: dict) -> None:
    path = _sl_guard_file(guard_path)
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[WARN] Could not write SL guard file {path}: {e}")

def activate_sl_guard(guard_path: str, sl_ts: int, sl_side: str) -> None:
    data = {
        "active": True,
        "neutral_seen": False,  # will be set True once we see neutral zone
        "sl_ts": int(sl_ts),
        "sl_side": str(sl_side),
    }
    save_sl_guard(guard_path, data)
    print(f"[SL-GUARD] Activated after SL: side={sl_side}, sl_ts={sl_ts}")

        
# =========================
# Core flow (futures only, close-only reversal by default)
# =========================
def decide_and_maybe_trade(args):
    # 1) Load bundle
    bundle = load_bundle(args.model_dir)
    meta = bundle["meta"]; model = bundle["model"]
    feats = bundle["feature_names"]; lookback = bundle["lookback"]
    pos_thr, neg_thr = bundle["pos_thr"], bundle["neg_thr"]
    sl_pct, tp_pct   = bundle["sl_pct"], bundle["tp_pct"]
    mean, std = bundle["mean"], bundle["std"]
    ik = bundle["ichimoku"]; model_dir = bundle["paths"]["dir"]

    # Derive a per-strategy identifier from the model directory
    strategy_id = os.path.basename(os.path.normpath(model_dir))
    safe_strategy_id = re.sub(r'[^A-Za-z0-9_.-]', '_', strategy_id)

    # 2) Resolve ticker/timeframe
    ticker = args.ticker or meta.get("ticker") or "BTCUSDT"
    timeframe = args.timeframe or meta.get("timeframe") or "1h"

    # 3) Load bars (JSON)
    df = load_bars_from_json(args.bars_json)
    if df is None or len(df) < (lookback + 3):
        print("Not enough bars to build features.")
        return

    # Build + align features & check names
    feat_df_full = build_features(
        df[["timestamp","ts","open","high","low","close","volume"]].copy(),
        ik["tenkan"], ik["kijun"], ik["senkou"], ik["displacement"]
    )
    feat_df_full = align_features_to_meta(feat_df_full, feats)
    for c in feats:
        if c not in feat_df_full.columns:
            raise SystemExit(f"Feature '{c}' missing in computed frame.")
    feat_df = feat_df_full.copy()

    # 4) Inference (prev vs last bar)
    feat_mat = feat_df[feats].to_numpy(dtype=np.float32)
    X_prev = feat_mat[-lookback-1 : -1]
    X_last = feat_mat[-lookback   :   ]
    X = np.stack([X_prev, X_last], axis=0)

    if getattr(args, "debug", False):
        closes = df["close"].to_numpy()
        feat_l1 = float(np.abs(X_last - X_prev).sum())
        print(f"[DEBUG] prev_close→last_close: {closes[-2]:.4f}→{closes[-1]:.4f} | feat_L1_diff={feat_l1:.6g}")

    p_prev, p_last = run_model(model, X, mean, std)
    if getattr(args, "debug", False):
        print(f"[DEBUG] proba — prev={p_prev:.8f} last={p_last:.8f} Δ={p_last-p_prev:+.8f}")

    # 5) Time gating (6-minute window after close)
    now_ms = int(time.time() * 1000)
    ts_last_open = int(df["ts"].iloc[-1])
    last_close_ms, stamp_tag, age_ms = resolve_last_closed(now_ms, ts_last_open, timeframe)
    if args.debug:
        print(
            f"[DEBUG] last bar — ts_last_open={ts_last_open} tag={stamp_tag} "
            f"age_min={(age_ms/60000.0) if age_ms is not None else None}"
        )
    if last_close_ms is None or not (0 <= age_ms <= SIX_MIN_MS):
        print("Last closed bar is not within the 6-minute window — not acting.")
        return

    # 6) Shared last-bar ID
    update_shared_lastbar(ticker, timeframe, ts_last_open, last_close_ms)
    if args.debug:
        shared = read_lastbars_store().get(f"{ticker}:{timeframe}", {})
        print(f"[DEBUG] shared lastbar: {shared.get('bar_id')} last_close_ts={shared.get('last_close_ts')}")

    # 7) Guard & reversal state (per-bar)
    suffix = f"bybit_swap_{ticker}_{timeframe}"
    last_seen_ts, guard_path = last_executed_guard(model_dir, suffix)
    rev_state_path = reversal_state_paths(model_dir, suffix)
    rev_state = read_reversal_state(rev_state_path)
    if last_seen_ts is not None and last_seen_ts == last_close_ms:
        print("Already acted on this bar for Bybit — not acting again.")
        return

    # 8) Fresh-cross trigger logic
    take_long = (p_last >= pos_thr) and (p_prev < p_last)
    take_short = (p_last <= neg_thr) and (p_prev > p_last)

    # 9) Exchange (unified swap)
    ex = make_exchange(args.pub_key, args.sec_key, keys_file=args.keys_file)
    symbol = resolve_symbol(ex, ticker)

    # --- DAILY PROFIT GUARD INIT (after we can read equity from Bybit) ---
    equity_now = get_equity_for_daily_guard(ex)

    today_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    guard_dir = "/home/production/guards"
    os.makedirs(guard_dir, exist_ok=True)

    daily_guard_filename = f"bybit_daily_profit_{safe_strategy_id}.json"
    daily_guard_path = os.path.join(guard_dir, daily_guard_filename)
    
    daily_state = load_daily_profit_state(daily_guard_path, today_str, equity_now)
    log_daily_status(daily_state, DAILY_PROFIT_TARGET_PCT)

    ## Update status
    daily_state = refresh_daily_state_with_equity(
        daily_guard_path,
        daily_state,
        equity_now=equity_now,
        target_pct=DAILY_PROFIT_TARGET_PCT,
    )

    # === DAILY PROFIT GUARD: block NEW trades once target hit ===
    if daily_guard_blocks_new_trades(daily_state, DAILY_PROFIT_TARGET_PCT):
        print(
            f"[DAILY PROFIT GUARD] Not opening because the daily target 🎯 "
            f"{DAILY_PROFIT_TARGET_PCT*100:.2f}% is already reached "
            f"({daily_state.get('daily_pct', 0.0)*100:.2f}%). "
            "No NEW positions will be opened today."
        )
        return

    
    # 10) Position & SL/TP (+ signal exit on neutral)
    pos = get_swap_position(ex, symbol)

    last_close = float(df["close"].iloc[-1])
    last_high = float(df["high"].iloc[-1])
    last_low = float(df["low"].iloc[-1])

    # 1) Still useful to have the simple "in neutral now"
    in_neutral_now = (neg_thr < p_last < pos_thr)

    # 2) NEW: detect a jump that *crosses* the neutral band
    crossed_neutral_down = (p_prev >= pos_thr and p_last <= neg_thr)
    crossed_neutral_up   = (p_prev <= neg_thr and p_last >= pos_thr)

    crossed_neutral = in_neutral_now or crossed_neutral_down or crossed_neutral_up

    # --- SL GUARD: block immediate re-entry after a Stop Loss ---
    sl_guard = load_sl_guard(guard_path)
    sl_active = bool(sl_guard.get("active"))
    sl_neutral_seen = bool(sl_guard.get("neutral_seen"))

    # If we already took an SL and now the probabilities are in the neutral band,
    # mark that as "neutral crossed" and allow future entries.
    if sl_active and not crossed_neutral and not sl_neutral_seen:
        sl_guard["neutral_seen"] = True
        sl_neutral_seen = True
        save_sl_guard(guard_path, sl_guard)
        print("[SL-GUARD] Neutral zone touched after last SL — "
              "new trades will be allowed on the next fresh-cross.")
        return

    # ===== OPEN POSITION HANDLING =====
    if pos is not None and pos.get("side"):
        # There is an open position on the exchange
        side_open_raw = pos.get("side")
        side_open = str(side_open_raw).lower()  # expect 'long' or 'short'
        entry = float(pos.get("entry") or last_close)

        raw_size = float(pos.get("size") or 0.0)
        sz_abs = abs(raw_size)
        min_amt = get_min_amount(ex, symbol)
        close_qty = amount_to_precision(ex, symbol, sz_abs)

        if close_qty <= 0 or (min_amt and close_qty < min_amt):
            print(
                f"[WARN] Position size {close_qty} is zero or below minimum "
                f"trade size {min_amt} after precision; cannot safely close — skipping new trades."
            )
            return

        print(
            f"[DEBUG] Existing position detected: side={side_open_raw!r}, "
            f"raw_size={raw_size}, close_qty={close_qty}, entry={entry}"
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
                    ex.create_order(symbol, "market", "sell", close_qty, None, {"reduceOnly": True})
                    print(f"{reason} hit — closing existing LONG at ~{(sl_px if hit_sl else tp_px):.8g}")
                    write_last_executed(guard_path, last_close_ms)

                    # NEW: SL guard on Stop Loss
                    if reason == "SL":
                        activate_sl_guard(guard_path, last_close_ms, sl_side="long")

                    # NEW: record realized PnL for LONG close
                    exit_px = last_close
                    pnl_quote = (exit_px - entry) * close_qty
                    daily_state = record_realized_pnl(
                        daily_guard_path,
                        daily_state,
                        pnl_quote=pnl_quote,
                        target_pct=DAILY_PROFIT_TARGET_PCT,
                    )

                except Exception as e:
                    print(f"[ERROR] close LONG on {reason} failed: {e}")
                return

            # 10b) SIGNAL EXIT: close LONG when leaving LONG zone
            if p_last < pos_thr:
                try:
                    ex.create_order(symbol, "market", "sell", close_qty, None, {"reduceOnly": True})
                    zone = "neutral" if in_neutral_now else "short"
                    print(
                        f"Signal exit — LONG → {zone} zone: "
                        f"p_last={p_last:.3f} < pos_thr={pos_thr:.3f}; closing LONG at ~{last_close}"
                    )
                    write_last_executed(guard_path, last_close_ms)

                    # NEW: record realized PnL for LONG signal exit
                    exit_px = last_close
                    pnl_quote = (exit_px - entry) * close_qty
                    daily_state = record_realized_pnl(
                        daily_guard_path,
                        daily_state,
                        pnl_quote=pnl_quote,
                        target_pct=DAILY_PROFIT_TARGET_PCT,
                    )

                except Exception as e:
                    print(f"[ERROR] close LONG (SIG) failed: {e}")
                return

        elif side_open == "short":
            # SHORT SL/TP prices
            sl_px = entry * (1.0 + (sl_pct or 0.0)) if sl_pct is not None else None
            tp_px = entry * (1.0 - (tp_pct or 0.0)) if tp_pct is not None else None
            hit_sl = (sl_px is not None) and (last_high >= sl_px)
            hit_tp = (tp_px is not None) and (last_low  <= tp_px)

            # 10c) SL/TP first (backtest parity)
            if hit_sl or hit_tp:
                reason = "SL" if hit_sl else "TP"
                try:
                    ex.create_order(symbol, "market", "buy", close_qty, None, {"reduceOnly": True})
                    print(f"{reason} hit — closing existing SHORT at ~{(sl_px if hit_sl else tp_px):.8g}")
                    write_last_executed(guard_path, last_close_ms)

                    # NEW: SL guard on Stop Loss
                    if reason == "SL":
                        activate_sl_guard(guard_path, last_close_ms, sl_side="short")

                    # NEW: record realized PnL for SHORT close
                    exit_px = last_close
                    pnl_quote = (entry - exit_px) * close_qty
                    daily_state = record_realized_pnl(
                        daily_guard_path,
                        daily_state,
                        pnl_quote=pnl_quote,
                        target_pct=DAILY_PROFIT_TARGET_PCT,
                    )

                except Exception as e:
                    print(f"[ERROR] close SHORT on {reason} failed: {e}")
                return

            # 10d) SIGNAL EXIT: close SHORT when leaving SHORT zone
            if p_last > neg_thr:
                try:
                    ex.create_order(symbol, "market", "buy", close_qty, None, {"reduceOnly": True})
                    zone = "neutral" if in_neutral_now else "long"
                    print(
                        f"Signal exit — SHORT → {zone} zone: "
                        f"p_last={p_last:.3f} > neg_thr={neg_thr:.3f}; closing SHORT at ~{last_close}"
                    )
                    write_last_executed(guard_path, last_close_ms)

                    # NEW: record realized PnL for SHORT signal exit
                    exit_px = last_close
                    pnl_quote = (entry - exit_px) * close_qty
                    daily_state = record_realized_pnl(
                        daily_guard_path,
                        daily_state,
                        pnl_quote=pnl_quote,
                        target_pct=DAILY_PROFIT_TARGET_PCT,
                    )

                except Exception as e:
                    print(f"[ERROR] close SHORT (SIG) failed: {e}")
                return
        else:
            print(f"[WARN] Unknown open side {side_open!r}; keeping position and not opening new trades.")
            return

        # If we still have a position and no SL/TP or signal-exit, we keep it
        print(
            f"Keeping existing {side_open.upper()} open — "
            f"p_last={p_last:.3f}, pos_thr={pos_thr:.3f}, neg_thr={neg_thr:.3f}"
        )
        return

    # ===== FLAT LOGIC =====

    # SL-GUARD: if an SL just happened and we have NOT yet seen neutral, block fresh-cross entries.
    if sl_active and not sl_neutral_seen:
        if take_long or take_short:
            print(
                "[SL-GUARD] Fresh signal blocked after SL — waiting for probabilities "
                "to pass through the neutral band (neg_thr < p < pos_thr). "
                f"p_prev={p_prev:.3f}, p_last={p_last:.3f}, "
                f"pos_thr={pos_thr:.3f}, neg_thr={neg_thr:.3f}"
            )
        else:
            msg = _explain_no_open(p_prev, p_last, pos_thr, neg_thr)
            print(msg)
        return

    # If flat and no fresh signal, do nothing
    if not (take_long or take_short):
        msg = _explain_no_open(p_prev, p_last, pos_thr, neg_thr)
        print(msg)
        return

    # === DAILY PROFIT GUARD: block NEW trades once target hit ===
    if daily_guard_blocks_new_trades(daily_state, DAILY_PROFIT_TARGET_PCT):
        print(
            f"[DAILY PROFIT GUARD] Blocking NEW trade: "
            f"🎯 target {DAILY_PROFIT_TARGET_PCT*100:.2f}% already reached "
            f"({daily_state.get('daily_pct', 0.0)*100:.2f}%)."
        )
        return

    # 12) Optional top-up for swap
    if args.auto_transfer:
        transfer_spot_to_swap_if_needed(ex, min_usdt=50.0, buffer_frac=args.transfer_buffer, debug=args.debug)

    # 13) OPEN order — balance-based sizing, with attached TP/SL on exchange
    try:
        quote_bal_swap = fetch_usdt_balance_swap(ex)
        side = "buy" if take_long else "sell"

        # Leave a small safety buffer
        risk_long = 0.97
        risk_short = 0.77

        usd_to_use = (
            quote_bal_swap * (risk_long if take_long else risk_short)
            if quote_bal_swap > 0
            else 0.0
        )
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

        print(
            f"[DEBUG] swap_bal={quote_bal_swap:.4f} usd_to_use={usd_to_use:.4f} "
            f"qty≈{qty_approx:.6f} notional≈{qty*last_close:.4f}"
        )

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

        # Reset SL guard because we are taking a new trade
        if sl_active:
            save_sl_guard(guard_path, {"active": False, "neutral_seen": False})
            print("[SL-GUARD] Reset after new position opening.")

        # --- Attach TP & SL as exchange-side reduce-only orders (existing logic) ---
        try:
            entry_price = float(order.get("average") or order.get("price") or last_close)

            tp_order_id = None
            sl_order_id = None
            tp_price_logged = None
            sl_price_logged = None

            def _safe_order_id(o):
                if isinstance(o, dict):
                    info = o.get("info") or {}
                    return (
                        o.get("id")
                        or o.get("orderId")
                        or info.get("orderId")
                        or info.get("order_id")
                    )
                return o

            if ex.id == "bybit":
                params_common = {
                    "reduceOnly": True,
                    "tpslMode": "Full",
                    "positionIdx": 0,
                }

                if side == "buy":
                    # LONG: TP above, SL below
                    if tp_pct is not None and tp_pct > 0:
                        tp_price = price_to_precision(ex, symbol, entry_price * (1.0 + tp_pct))
                        tp = ex.create_order(
                            symbol,
                            "market",
                            "sell",
                            qty,
                            None,
                            {
                                **params_common,
                                "takeProfitPrice": float(tp_price),
                                "tpTriggerBy": "LastPrice",
                            },
                        )
                        oid = _safe_order_id(tp)
                        if oid:
                            tp_order_id = oid
                            tp_price_logged = tp_price
                        else:
                            print(f"[WARN] TP order returned no id on {ex.id}: {tp!r}")

                    if sl_pct is not None and sl_pct > 0:
                        sl_price = price_to_precision(ex, symbol, entry_price * (1.0 - sl_pct))
                        sl = ex.create_order(
                            symbol,
                            "market",
                            "sell",
                            qty,
                            None,
                            {
                                **params_common,
                                "stopLossPrice": float(sl_price),
                                "slTriggerBy": "LastPrice",
                            },
                        )
                        oid = _safe_order_id(sl)
                        if oid:
                            sl_order_id = oid
                            sl_price_logged = sl_price
                        else:
                            print(f"[WARN] SL order returned no id on {ex.id}: {sl!r}")

                else:
                    # SHORT: TP below, SL above
                    if tp_pct is not None and tp_pct > 0:
                        tp_price = price_to_precision(ex, symbol, entry_price * (1.0 - tp_pct))
                        tp = ex.create_order(
                            symbol,
                            "market",
                            "buy",
                            qty,
                            None,
                            {
                                **params_common,
                                "takeProfitPrice": float(tp_price),
                                "tpTriggerBy": "LastPrice",
                            },
                        )
                        oid = _safe_order_id(tp)
                        if oid:
                            tp_order_id = oid
                            tp_price_logged = tp_price
                        else:
                            print(f"[WARN] TP order returned no id on {ex.id}: {tp!r}")

                    if sl_pct is not None and sl_pct > 0:
                        sl_price = price_to_precision(ex, symbol, entry_price * (1.0 + sl_pct))
                        sl = ex.create_order(
                            symbol,
                            "market",
                            "buy",
                            qty,
                            None,
                            {
                                **params_common,
                                "stopLossPrice": float(sl_price),
                                "slTriggerBy": "LastPrice",
                            },
                        )
                        oid = _safe_order_id(sl)
                        if oid:
                            sl_order_id = oid
                            sl_price_logged = sl_price
                        else:
                            print(f"[WARN] SL order returned no id on {ex.id}: {sl!r}")

            else:
                # Generic / other exchanges (kept as-is)
                if side == "buy":
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
                        oid = _safe_order_id(tp)
                        if oid:
                            tp_order_id = oid
                            tp_price_logged = tp_price
                        else:
                            print(f"[WARN] TP order returned no id on {ex.id}: {tp!r}")

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
                        oid = _safe_order_id(sl)
                        if oid:
                            sl_order_id = oid
                            sl_price_logged = sl_price
                        else:
                            print(f"[WARN] SL order returned no id on {ex.id}: {sl!r}")

                else:
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
                        oid = _safe_order_id(tp)
                        if oid:
                            tp_order_id = oid
                            tp_price_logged = tp_price
                        else:
                            print(f"[WARN] TP order returned no id on {ex.id}: {tp!r}")

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
                        oid = _safe_order_id(sl)
                        if oid:
                            sl_order_id = oid
                            sl_price_logged = sl_price
                        else:
                            print(f"[WARN] SL order returned no id on {ex.id}: {sl!r}")

            if tp_order_id or sl_order_id:
                parts = []
                # if tp_order_id:
                #     parts.append(f"TP[id={tp_order_id!r}, px={tp_price_logged}]")
                # if sl_order_id:
                #     parts.append(f"SL[id={sl_order_id!r}, px={sl_price_logged}]")
                print("Attached TP/SL orders — Success!!")

        except Exception as e:
            print(f"[WARN] failed to attach TP/SL orders: {e}")

        # Guard file: one action per bar across brokers
        write_last_executed(guard_path, last_close_ms)

    except Exception as e:
        print(f"[ERROR] order failed: {e}")


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Run LSTM bundle; Bybit FUTURES (swap, unified) orders on fresh signal within 6 minutes — bars read from a JSON file."
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
