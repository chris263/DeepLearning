#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Ichimoku runner for Binance (USDT-M futures).

Parity with your CoinEx/Bybit scripts:
- Fresh-cross trigger, close-only reversal (flip/cooldown optional)
- SL/TP exits from meta.json
- Balance-based sizing: 100% long, 80% short (USDT on futures)
- 6-min window after real close, shared last-bar, idempotency guard
- Feature-name alignment; bars read from JSON

Requires: torch, ccxt, numpy, pandas
"""

from __future__ import annotations
import os, sys, json, time, argparse, pathlib, tempfile
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

try:
    import torch
    import ccxt
except Exception as e:
    print("Please install dependencies: pip install torch ccxt pandas numpy", file=sys.stderr); raise

SIX_MIN_MS = 6 * 60 * 1000
TF_TO_MS = {"1m":60000,"3m":180000,"5m":300000,"15m":900000,"30m":1800000,"1h":3600000,"2h":7200000,"4h":14400000,"6h":21600000,"8h":28800000,"12h":43200000,"1d":86400000}
def tf_ms(tf: str) -> int:
    v = TF_TO_MS.get(str(tf).lower())
    if not v: raise SystemExit(f"Unsupported timeframe '{tf}'")
    return v

# ---------- JSON bars ----------
def _normalize_bar(b: Dict) -> Optional[Dict]:
    if not isinstance(b, dict): return None
    kl = {k.lower(): k for k in b}
    def g(*names, default=None):
        for n in names:
            k = kl.get(n); 
            if k in b: return b[k]
        return default
    ts = g("ts","timestamp","time","t","open_time","opentime")
    if ts is None: return None
    if isinstance(ts,str):
        try: ts = int(pd.to_datetime(ts, utc=True).value // 1_000_000)
        except: return None
    ts = int(ts); 
    if ts < 10_000_000_000: ts *= 1000
    try:
        o=float(g("open","o")); h=float(g("high","h")); l=float(g("low","l")); c=float(g("close","c")); v=float(g("volume","v","vol",default=0.0))
    except: return None
    return {"ts":ts,"open":o,"high":h,"low":l,"close":c,"volume":v}

def load_bars_from_json(path: str) -> pd.DataFrame:
    p = pathlib.Path(path).expanduser()
    if not p.exists(): raise SystemExit(f"Bars JSON not found: {p}")
    content = p.read_text().strip(); bars: List[Dict] = []
    try:
        obj = json.loads(content)
        if isinstance(obj, list): bars = obj
        elif isinstance(obj, dict):
            for k in ("data","bars","result","items","price"):
                if k in obj and isinstance(obj[k], list): bars = obj[k]; break
    except:
        for line in content.splitlines():
            line=line.strip(); 
            if line:
                try: bars.append(json.loads(line))
                except: pass
    if not bars: raise SystemExit(f"No bars decoded from {p}")
    norm=[_normalize_bar(b) for b in bars if _normalize_bar(b)]
    if not norm: raise SystemExit(f"No valid bars decoded from {p}")
    df=pd.DataFrame(norm).sort_values("ts").reset_index(drop=True)
    df["timestamp"]=pd.to_datetime(df["ts"],unit="ms",utc=True).dt.tz_convert(None)
    return df[["timestamp","ts","open","high","low","close","volume"]]

# ---------- Features ----------
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

_SYNONYMS={"ret1":["ret_1","r1","return1"],"oc_diff":["ocdiff","oc_change"],"hl_range":["hlrange","high_low_range"],"logv_chg":["logv_change","dlogv","logv_diff"],"dist_px_cloud_top":["dist_px_cloudtop"],"dist_px_cloud_bot":["dist_px_cloudbot"],"dist_tk_kj":["dist_tk_kijun","tk_kj_dist"],"span_order":["spanOrder","span_order_flag"],"tk_slope":["tenkan_slope","tkSlope"],"kj_slope":["kijun_slope","kjSlope"],"span_a_slope":["spana_slope","spanA_slope"],"span_b_slope":["spanb_slope","spanB_slope"],"chikou_above":["chikou_flag"],"vol20":["vol_20","volatility20"]}
def align_features_to_meta(feat_df, meta_features):
    cols=set(feat_df.columns)
    for n in meta_features:
        if n in cols: continue
        for cand in _SYNONYMS.get(n, []):
            if cand in cols: feat_df[n]=feat_df[cand]; cols.add(n); break
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

# ---------- Bundle ----------
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

# ---------- Time helpers ----------
def resolve_last_closed(now_ms:int,last_bar_open_ms:int,timeframe:str)->Tuple[Optional[int],str,Optional[int]]:
    step=tf_ms(timeframe); candidates=[(last_bar_open_ms+step,"close_stamp"),(last_bar_open_ms,"open_stamp")]
    valid=[(c,tag,now_ms-c) for (c,tag) in candidates if now_ms>=c]
    if not valid: return None,"future",None
    c,tag,age=min(valid,key=lambda x:x[2]); return c,tag,age

# ---------- Shared state ----------
def state_dir()->str:
    base=os.getenv("SAT_STATE_DIR",os.path.expanduser("~/.sat_state")); pathlib.Path(base).mkdir(parents=True,exist_ok=True); return base
def lastbars_json_path()->str: return os.path.join(state_dir(),"lastbars.json")
def read_lastbars_store()->Dict[str,Dict]:
    p=pathlib.Path(lastbars_json_path()); 
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except: return {}
def write_lastbars_store(data:Dict[str,Dict])->None:
    p=pathlib.Path(lastbars_json_path())
    fd,tmp=tempfile.mkstemp(prefix="lastbars_",suffix=".json",dir=str(p.parent))
    with os.fdopen(fd,"w") as f: json.dump(data,f,separators=(",",":"),sort_keys=True)
    os.replace(tmp,p)
def bar_id(tkr,tf,last_ts): return f"{tkr}|{tf}|{int(last_ts)}"
def update_shared_lastbar(tkr,tf,last_open,last_close):
    key=f"{tkr}:{tf}"; store=read_lastbars_store()
    store[key]={"bar_id":bar_id(tkr,tf,last_open),"ticker":tkr,"timeframe":tf,"last_open_ts":int(last_open),"last_close_ts":int(last_close),"updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    write_lastbars_store(store)

def last_executed_guard(model_dir:str,suffix:str)->Tuple[Optional[int],str]:
    path=os.path.join(model_dir,f".last_order_ts_{suffix}.txt")
    val=None
    if os.path.exists(path):
        try: val=int(open(path).read().strip() or "0")
        except: val=None
    return val,path
def write_last_executed(path:str,ts:int):
    with open(path,"w") as f: f.write(str(int(ts)))
def reversal_state_path(model_dir:str,suffix:str)->str: return os.path.join(model_dir,f".reversal_state_{suffix}.json")
def read_reversal_state(path:str)->Dict[str,int]:
    if not os.path.exists(path): return {}
    try: return json.loads(open(path).read())
    except: return {}
def write_reversal_state(path:str,data:Dict[str,int]):
    with open(path,"w") as f: json.dump(data,f,separators=(",",":"))

# ---------- Binance helpers (USDT-M futures) ----------
def _read_kv_file(path:str)->Dict[str,str]:
    kv={}; p=pathlib.Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k,v=line.split("=",1); kv[k.strip()]=v.strip()
    return kv

def make_exchange(pub_key_name:Optional[str],sec_key_name:Optional[str],keys_file:Optional[str]=None):
    keyfile=os.path.expanduser(keys_file or os.getenv("BINANCE_KEYS_FILE","~/.ssh/binance_keys.env"))
    kv=_read_kv_file(keyfile)
    ex=ccxt.binance({
        "apiKey": kv.get(pub_key_name) if pub_key_name else None,
        "secret": kv.get(sec_key_name) if sec_key_name else None,
        "enableRateLimit": True,
        "options": {"defaultType":"future"}  # USDT-M futures
    })
    ex.load_markets(); return ex

def resolve_symbol(ex,ticker:str)->str:
    t=ticker.upper().replace("/","")
    base=t[:-4] if t.endswith("USDT") else t
    # prefer contract notation with settlement suffix
    opts=[s for s in ex.symbols if s.startswith(f"{base}/USDT") and (":USDT" in s)]
    if opts: return sorted(opts)[0]
    # fallback: any USDT contract
    for s in ex.symbols:
        if f"{base}/USDT" in s: return s
    raise SystemExit(f"No futures symbol for {ticker} on Binance")

def get_quote_code_from_symbol(symbol:str)->str:
    return symbol.split(":")[-1] if ":" in symbol else symbol.split("/")[1]

def fetch_quote_balance_future(ex, quote_code:str)->float:
    # with defaultType=future this should read futures wallet
    try:
        bal=ex.fetch_balance(params={"type":"future"})
        return float((bal.get("free",{}) or {}).get(quote_code, 0.0) or 0.0)
    except Exception:
        try:
            bal=ex.fetch_balance()
            return float((bal.get("free",{}) or {}).get(quote_code, 0.0) or 0.0)
        except Exception:
            return 0.0

def transfer_spot_to_future_if_needed(ex, quote_code:str, min_amt:float, buffer_frac:float=0.01, debug:bool=False)->float:
    try:
        fut_free=fetch_quote_balance_future(ex, quote_code)
        if fut_free>=min_amt: return fut_free
        if debug: print(f"[DEBUG] futures {quote_code}={fut_free:.2f} < {min_amt:.2f}; topping up…")
        spot=ex.fetch_balance(params={"type":"spot"})
        spot_free=float((spot.get("free",{}) or {}).get(quote_code,0.0) or 0.0)
        if spot_free<=0: return fut_free
        amt=min(spot_free, min_amt*(1.0+max(0.0,buffer_frac)))
        try:
            ex.transfer(code=quote_code, amount=float(f"{amt:.6f}"), fromAccount="spot", toAccount="future")
            time.sleep(0.5)
        except Exception:
            pass
        return fetch_quote_balance_future(ex, quote_code)
    except Exception as e:
        print(f"[WARN] auto-transfer failed: {e}")
        return fetch_quote_balance_future(ex, quote_code)

def amount_to_precision(ex,symbol,amount): 
    try: return float(ex.amount_to_precision(symbol,amount))
    except: return float(f"{amount:.6f}")
def price_to_precision(ex,symbol,price):
    try: return float(ex.price_to_precision(symbol,price))
    except: return float(f"{price:.6f}")

def get_future_position(ex, symbol:str)->Optional[Dict]:
    def _scan(ps):
        if not ps: return None
        for p in ps:
            if (p or {}).get("symbol")==symbol:
                side=(p.get("side") or "").lower()
                sz=float(p.get("contracts") or p.get("size") or 0.0)
                entry=float(p.get("entryPrice") or 0.0)
                if sz and side: return {"side":side,"size":abs(sz),"entry":entry}
        return None
    try:
        pos=_scan(ex.fetch_positions([symbol]))
        if pos: return pos
    except Exception: pass
    try: return _scan(ex.fetch_positions())
    except Exception: return None

# ---------- Inference ----------
def to_sequences_latest(feat_df, features, lookback):
    if len(feat_df)<(lookback+1): raise SystemExit("Not enough rows to build lookback sequences")
    sub=feat_df.iloc[-(lookback+1):].copy().reset_index(drop=True)
    prev=sub.iloc[:-1][features].to_numpy(np.float32)
    last=sub.iloc[1:][features].to_numpy(np.float32)
    X=np.stack([prev,last],axis=0); ts_seq=sub["ts"].to_numpy()
    return X, ts_seq

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
    # Sanity: neutral band
    if neg_thr < p_last < pos_thr:
        return (f"No fresh signal: p_last={fp(p_last)} is inside the neutral band "
                f"({fp(neg_thr)} < p_last < {fp(pos_thr)}). "
                f"Fresh-cross requires p_prev<{fp(pos_thr)}≤p_last for LONG or "
                f"p_prev>{fp(neg_thr)}≥p_last for SHORT.")

    # Already in zones (no re-open without a fresh cross)
    if p_last >= pos_thr and p_prev >= pos_thr:
        return (f"No LONG open: previous bar already in LONG zone "
                f"(p_prev={fp(p_prev)} ≥ pos_thr={fp(pos_thr)}), so no fresh cross. "
                f"Current p_last={fp(p_last)}.")
    if p_last <= neg_thr and p_prev <= neg_thr:
        return (f"No SHORT open: previous bar already in SHORT zone "
                f"(p_prev={fp(p_prev)} ≤ neg_thr={fp(neg_thr)}), so no fresh cross. "
                f"Current p_last={fp(p_last)}.")

    # Approaching but didn’t cross
    if p_prev < pos_thr and p_last < pos_thr:
        gap = pos_thr - p_last
        return (f"No LONG: probability stayed below pos_thr "
                f"(p_prev={fp(p_prev)} → p_last={fp(p_last)}; needs +{fp(gap)} to reach {fp(pos_thr)}).")
    if p_prev > neg_thr and p_last > neg_thr:
        gap = p_last - neg_thr
        return (f"No SHORT: probability stayed above neg_thr "
                f"(p_prev={fp(p_prev)} → p_last={fp(p_last)}; needs -{fp(gap)} to reach {fp(neg_thr)}).")

    # Edge/equality cases (e.g., sitting exactly on a threshold without fresh-cross)
    return (f"No fresh signal: p_prev={fp(p_prev)} → p_last={fp(p_last)}; "
            f"thresholds pos_thr={fp(pos_thr)}, neg_thr={fp(neg_thr)}. "
            f"Fresh-cross requires p_prev<{fp(pos_thr)}≤p_last (LONG) or p_prev>{fp(neg_thr)}≥p_last (SHORT).")

# ---------- Core ----------
def decide_and_maybe_trade(args):
    B=load_bundle(args.model_dir)
    meta=B["meta"]
    model=B["model"]
    feats=B["feature_names"]
    lookback=B["lookback"]
    pos_thr,neg_thr=B["pos_thr"],B["neg_thr"]
    sl,tp=B["sl_pct"],B["tp_pct"]
    mean, std=B["mean"],B["std"]; ik=B["ichimoku"]
    model_dir=B["paths"]["dir"]
    ticker=args.ticker or meta.get("ticker") or "BTCUSDT"
    timeframe=args.timeframe or meta.get("timeframe") or "1h"
    df=load_bars_from_json(args.bars_json)
    
    if len(df)<(lookback+3):
        print("Not enough bars to build features.")
        return
    
    feat_full=build_features(df[["timestamp","ts","open","high","low","close","volume"]].copy(), ik["tenkan"],ik["kijun"],ik["senkou"],ik["displacement"])
    feat_full=align_features_to_meta(feat_full,feats)
    
    for c in feats:
        if c not in feat_full.columns: raise SystemExit(f"Feature '{c}' missing.")
    feat_df=feat_full.copy()
    X,ts_seq=to_sequences_latest(feat_df[feats+["ts"]],feats,lookback); p_prev,p_last=run_model(model,X,mean,std)
    print(f"LSTM inference | p_prev={p_prev:.3f} | p_last={p_last:.3f} | pos_thr={pos_thr:.3f} | neg_thr={neg_thr:.3f}")
    now_ms=int(time.time()*1000); ts_last_open=int(df["ts"].iloc[-1]); last_close_ms, tag, age=resolve_last_closed(now_ms, ts_last_open, timeframe)
    if args.debug: print(f"[DEBUG] last bar — ts_last_open={ts_last_open} tag={tag} age_min={(age/60000.0) if age is not None else None}")
    if last_close_ms is None or not (0<=age<=SIX_MIN_MS): print("Last closed bar is not within the 6-minute window — not acting."); return
    update_shared_lastbar(ticker,timeframe,ts_last_open,last_close_ms)
    suffix=f"binance_fut_{ticker}_{timeframe}"; last_seen, guard_path=last_executed_guard(model_dir,suffix)
    rev_path=reversal_state_path(model_dir,suffix); rev_state=read_reversal_state(rev_path)
    if last_seen is not None and last_seen==last_close_ms: print("Already acted on this bar for Binance — not acting again."); return
    take_long  = (p_last >= pos_thr) and (p_prev <  pos_thr)
    take_short = (p_last <= neg_thr) and (p_prev >  neg_thr)
    if not take_long and not take_short:
        print(_explain_no_open(p_prev, p_last, pos_thr, neg_thr))
        return

    ex=make_exchange(args.pub_key,args.sec_key,keys_file=args.keys_file); symbol=resolve_symbol(ex,ticker)
    pos = get_future_position(ex, symbol)
    last_close = float(df["close"].iloc[-1])
    last_high  = float(df["high"].iloc[-1])
    last_low   = float(df["low"].iloc[-1])

    if pos is not None and pos.get("side"):
        entry = float(pos.get("entry") or last_close)
        side_open = (pos.get("side") or "").lower()  # 'long' or 'short'
        sz = float(pos.get("size") or pos.get("contracts") or 0.0)

        if side_open == "long":
            sl_px = entry * (1.0 - (sl_pct or 0.0)) if sl_pct is not None else None
            tp_px = entry * (1.0 + (tp_pct or 0.0)) if tp_pct is not None else None
            hit_sl = (sl_px is not None) and (last_low  <= sl_px)
            hit_tp = (tp_px is not None) and (last_high >= tp_px)

            # SL-first (backtest parity)
            if hit_sl or hit_tp:
                reason = "SL" if hit_sl else "TP"
                try:
                    ex.create_order(symbol, "market", "sell", sz or 1, None, {"reduceOnly": True})
                    print(f"{reason} hit — closing existing LONG at ~{(sl_px if hit_sl else tp_px):.8g}")
                    write_last_executed(guard_path, last_close_ms)
                except Exception as e:
                    print(f"[ERROR] close LONG on {reason} failed: {e}")
                return

            # SIGNAL EXIT: close LONG when p_last drops below pos_thr (no cross required)
            if p_last < pos_thr:
                try:
                    ex.create_order(symbol, "market", "sell", sz or 1, None, {"reduceOnly": True})
                    print(f"Signal exit — p_last={p_last:.3f} < pos_thr={pos_thr:.3f}: closing LONG at ~{last_close}")
                    write_last_executed(guard_path, last_close_ms)
                except Exception as e:
                    print(f"[ERROR] close LONG (SIG) failed: {e}")
                return

        elif side_open == "short":
            sl_px = entry * (1.0 + (sl_pct or 0.0)) if sl_pct is not None else None
            tp_px = entry * (1.0 - (tp_pct or 0.0)) if tp_pct is not None else None
            hit_sl = (sl_px is not None) and (last_high >= sl_px)
            hit_tp = (tp_px is not None) and (last_low  <= tp_px)

            # SL-first (backtest parity)
            if hit_sl or hit_tp:
                reason = "SL" if hit_sl else "TP"
                try:
                    ex.create_order(symbol, "market", "buy", sz or 1, None, {"reduceOnly": True})
                    print(f"{reason} hit — closing existing SHORT at ~{(sl_px if hit_sl else tp_px):.8g}")
                    write_last_executed(guard_path, last_close_ms)
                except Exception as e:
                    print(f"[ERROR] close SHORT on {reason} failed: {e}")
                return

            # SIGNAL EXIT: close SHORT when p_last rises above neg_thr (no cross required)
            if p_last > neg_thr:
                try:
                    ex.create_order(symbol, "market", "buy", sz or 1, None, {"reduceOnly": True})
                    print(f"Signal exit — p_last={p_last:.3f} > neg_thr={neg_thr:.3f}: closing SHORT at ~{last_close}")
                    write_last_executed(guard_path, last_close_ms)
                except Exception as e:
                    print(f"[ERROR] close SHORT (SIG) failed: {e}")
                return

    if pos and pos.get("side") and ((take_long and pos["side"]=="long") or (take_short and pos["side"]=="short")):
        print("Avoiding opening another position - pyramiding."); return
    opposite = pos and pos.get("side") and ((pos["side"]=="long" and take_short) or (pos["side"]=="short" and take_long))
    
    if opposite:
        try:
            side_open=pos["side"]; sz=float(pos.get("size") or 0.0); side="buy" if side_open=="short" else "sell"
            policy=args.reversal
            if policy=="flip": print("Signal reversal — flip mode: closing existing, then opening opposite.")
            elif policy=="cooldown": print("Signal reversal — cooldown mode: closing existing; no new open until cooldown expires.")
            else: print("Signal reversal — close-only mode: closing existing; not opening a new one this bar.")
            ex.create_order(symbol,"market",side,sz or 1,None,{"reduceOnly":True})
            if args.reversal in ("close","cooldown"):
                write_reversal_state(rev_path,{"last_close_time_ms":now_ms,"last_bar_ts":int(ts_last_open)})
                write_last_executed(guard_path,last_close_ms); return
        except Exception as e: print(f"[WARN] failed to close before reversal handling: {e}")
        if args.reversal=="flip": time.sleep(0.2)
    
    cooldown_active = False
    remain = None
    if args.reversal == "cooldown" and os.path.exists(rev_state_path):
        try:
            with open(rev_state_path, "r") as f:
                st = json.load(f)
            last_ms = int(st.get("last_close_time_ms") or 0)
            cd_ms = int(getattr(args, "cooldown_seconds", 0) or 0) * 1000
            if last_ms and cd_ms:
                elapsed = now_ms - last_ms
                if elapsed < cd_ms:
                    cooldown_active = True
                    remain = int((cd_ms - elapsed) / 1000)
        except Exception as e:
            print(f"[WARN] cooldown state read failed: {e}")

    if cooldown_active:
        print(f"Cooldown active — {remain}s remaining; not opening new positions.")
        return
    
    quote=get_quote_code_from_symbol(symbol)
    if args.auto_transfer: transfer_spot_to_future_if_needed(ex, quote, min_amt=50.0, buffer_frac=args.transfer_buffer, debug=args.debug)
    try:
        quote_free=fetch_quote_balance_future(ex, quote)
        side="buy" if take_long else "sell"
        usd_to_use=(quote_free*(1.00 if take_long else 0.80)) if quote_free>0 else 0.0
        if usd_to_use<=0: print(f"No {quote} balance available on futures."); return
        qty_approx=usd_to_use/max(1e-12,last_close); qty=amount_to_precision(ex,symbol,qty_approx)
        if qty<=0: print("Calculated order size is zero after precision rounding."); return
        try: ex.set_leverage(1,symbol)
        except Exception: pass
        px=price_to_precision(ex,symbol,last_close)
        print(f"Opening {('LONG' if side=='buy' else 'SHORT')} (futures) — MARKET {side.upper()} {symbol} qty={qty} (px≈{px})")
        order=ex.create_order(symbol,"market",side,qty,None,{"reduceOnly":False})
        oid=order.get("id") or order.get("orderId") or order; print(f"Order placed: {oid}")
        write_last_executed(guard_path,last_close_ms)
    except Exception as e:
        print(f"[ERROR] order failed: {e}")

def parse_args():
    ap=argparse.ArgumentParser(description="Run LSTM bundle; Binance USDT-M FUTURES on fresh signal within 6 minutes — bars from JSON.")
    ap.add_argument("--model-dir",required=True); ap.add_argument("--bars-json",required=True)
    ap.add_argument("--ticker",default=None); ap.add_argument("--timeframe",default=None)
    ap.add_argument("--auto-transfer",action="store_true"); ap.add_argument("--transfer-buffer",type=float,default=0.01)
    ap.add_argument("--reversal",choices=["close","flip","cooldown"],default="close")
    ap.add_argument("--cooldown-seconds",type=int,default=0); ap.add_argument("--debug",action="store_true")
    ap.add_argument("--pub_key",default=None); ap.add_argument("--sec_key",default=None)
    ap.add_argument("--keys-file",default=None, help="Env file with KEY=VALUE (default ~/.ssh/binance_keys.env)")
    return ap.parse_args()

if __name__=="__main__":
    decide_and_maybe_trade(parse_args())
