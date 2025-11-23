#!/usr/bin/env python3
"""
cross_prob_bybit.py

Cross-probability script:

- Reads BTC and ETH probability JSONs (prev_prob, last_prob/last_prov).
- If both BTC & ETH are in the same zone:
    LONG zone  -> open LONG on --trade-ticker
    SHORT zone -> open SHORT on --trade-ticker
- Exit rules:
    1) SL/TP (8% TP, 3% SL) – close position and stop trading for the day.
    2) Either BTC or ETH goes NEUTRAL or opposite zone – close position.
       If both move to opposite zone AND daily guard not hit, open reverse.
- Daily guard: 3% of equity_start (absolute) – no more trades for the day.

This version uses Bybit via ccxt and a local JSON "state file" to track position
and daily PnL.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import ccxt  # make sure ccxt is installed


POS_THR_DEFAULT = 0.55   # p > 0.55 => LONG zone
NEG_THR_DEFAULT = 0.45   # p < 0.45 => SHORT zone

TP_PCT_DEFAULT = 0.08    # 8% take profit
SL_PCT_DEFAULT = 0.03    # 3% stop loss
DAILY_TARGET_PCT_DEFAULT = 0.03  # 3% daily target (absolute)

STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-probability Bybit trader (BTC+ETH → trade SOL/XRP, etc.)"
    )
    p.add_argument(
        "--btc-json", required=True,
        help="JSON file with BTC probabilities (prev_prob, last_prob/last_prov)"
    )
    p.add_argument(
        "--eth-json", required=True,
        help="JSON file with ETH probabilities (prev_prob, last_prob/last_prov)"
    )
    p.add_argument(
        "--trade-ticker", required=True,
        help="Ticker to trade as base (e.g., SOLUSDT, XRPUSDT, or ccxt symbol like SOL/USDT:USDT)"
    )
    p.add_argument(
        "--key-name", required=True,
        help="Key name in keyfile/env for the Bybit API key"
    )
    p.add_argument(
        "--key-secret", required=True,
        help="Key name in keyfile/env for the Bybit API secret"
    )
    p.add_argument(
        "--keyfile", default=os.path.expanduser("~/.ssh/coinex_keys.env"),
        help="Path to KEY=VALUE env-style file with Bybit credentials (default: ~/.ssh/coinex_keys.env)"
    )
    p.add_argument(
        "--state-json", default=None,
        help="Path to state JSON file (optional; default: cross_state_<TRADE_TICKER>.json)"
    )
    p.add_argument(
        "--pos-thr", type=float, default=POS_THR_DEFAULT,
        help="Positive threshold for LONG zone (default: 0.55)"
    )
    p.add_argument(
        "--neg-thr", type=float, default=NEG_THR_DEFAULT,
        help="Negative threshold for SHORT zone (default: 0.45)"
    )
    p.add_argument(
        "--tp-pct", type=float, default=TP_PCT_DEFAULT,
        help="Take profit as fraction of entry (default: 0.08 = 8%)"
    )
    p.add_argument(
        "--sl-pct", type=float, default=SL_PCT_DEFAULT,
        help="Stop loss as fraction of entry (default: 0.03 = 3%)"
    )
    p.add_argument(
        "--daily-target-pct", type=float, default=DAILY_TARGET_PCT_DEFAULT,
        help="Absolute daily target on realized PnL / equity_start (default: 0.03 = 3%)"
    )
    p.add_argument(
        "--risk", type=float, default=0.05,
        help="Fraction of equity to allocate per trade (default: 0.05 = 5%)"
    )
    p.add_argument(
        "--testnet", action="store_true",
        help="Use Bybit testnet (ccxt sandbox mode)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Dry-run mode: no real orders, just logs and simulated state"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Verbose debug prints"
    )
    return p.parse_args()


def today_str_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def normalize_trade_symbol(trade_ticker: str) -> str:
    """
    Normalize user-provided trade ticker into ccxt Bybit symbol.

    - If user passes a ccxt-style symbol already (contains '/'), return it.
    - If user passes 'SOLUSDT' or 'XRPUSDT', map to 'SOL/USDT:USDT', etc.
    """
    t = trade_ticker.strip().upper()
    if "/" in t:
        # Assume user is passing a ccxt symbol like "SOL/USDT:USDT"
        return t

    if t.endswith("USDT"):
        base = t[:-4]
        return f"{base}/USDT:USDT"
    elif t.endswith("USDC"):
        base = t[:-4]
        return f"{base}/USDC:USDT"
    else:
        raise ValueError(f"Unsupported trade-ticker format: {trade_ticker}")


def default_state_path(trade_ticker: str) -> str:
    safe = trade_ticker.upper().replace("/", "_").replace(":", "_")
    return f"cross_state_{safe}.json"


def load_api_keys(
    keyfile: str,
    key_name: str,
    key_secret_name: str,
) -> Tuple[str, str]:
    """
    Load API key/secret from env OR from keyfile with KEY=VALUE lines.
    """
    api_key = os.environ.get(key_name)
    api_secret = os.environ.get(key_secret_name)

    if os.path.exists(keyfile):
        kv: Dict[str, str] = {}
        with open(keyfile, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
        api_key = api_key or kv.get(key_name)
        api_secret = api_secret or kv.get(key_secret_name)

    if not api_key or not api_secret:
        raise RuntimeError(
            f"Could not load Bybit API credentials for {key_name}/{key_secret_name} "
            f"from env or {keyfile}"
        )
    return api_key, api_secret


def init_bybit(api_key: str, api_secret: str, testnet: bool) -> ccxt.bybit:
    cfg: Dict[str, Any] = {
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # use USDT perpetual / swap
        },
    }
    ex = ccxt.bybit(cfg)
    if testnet:
        ex.set_sandbox_mode(True)
        print("[INFO] Using Bybit testnet (ccxt sandbox mode).")
    return ex


def load_probs(path: str) -> Tuple[float, float]:
    """
    Load prev_prob and last_prob from a JSON file.

    Supports keys:
      - prev_prob or p_prev
      - last_prob, last_prov, or p_last
    """
    with open(path, "r") as fh:
        data = json.load(fh)

    def _get_float(d: Dict[str, Any], *keys: str) -> float:
        for k in keys:
            if k in d and d[k] is not None:
                return float(d[k])
        raise KeyError(f"None of keys {keys} found in {path}")

    p_prev = _get_float(data, "prev_prob", "p_prev")
    p_last = _get_float(data, "last_prob", "last_prov", "p_last")

    return p_prev, p_last


def classify_zone(p_last: float, p_prev: float, pos_thr: float, neg_thr: float) -> str:
    """
    Return 'LONG', 'SHORT', or 'NEUTRAL' based on p_last.

    LONG zone   : p_last >= pos_thr
    SHORT zone  : p_last <= neg_thr
    NEUTRAL     : otherwise
    """
    if (p_last >= pos_thr) and (p_prev < p_last):
        return "LONG"
    elif (p_last <= neg_thr) and (p_prev > p_last):
        return "SHORT"
    else:
        return "NEUTRAL"


def load_state(path: str, symbol_label: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "version": STATE_VERSION,
            "symbol": symbol_label,
            "date": None,             # YYYY-MM-DD
            "equity_start": None,     # USDT
            "realized_today_quote": 0.0,
            "stopped_for_today": False,
            "has_position": False,
            "side": None,             # 'long' or 'short'
            "size": 0.0,              # base size
            "entry_price": 0.0,
            "tp_price": 0.0,
            "sl_price": 0.0,
        }
    with open(path, "r") as fh:
        state = json.load(fh)

    state.setdefault("version", STATE_VERSION)
    state.setdefault("symbol", symbol_label)
    state.setdefault("realized_today_quote", 0.0)
    state.setdefault("stopped_for_today", False)
    state.setdefault("has_position", bool(state.get("has_position")))
    state.setdefault("side", state.get("side"))
    state.setdefault("size", float(state.get("size", 0.0)))
    state.setdefault("entry_price", float(state.get("entry_price", 0.0)))
    state.setdefault("tp_price", float(state.get("tp_price", 0.0)))
    state.setdefault("sl_price", float(state.get("sl_price", 0.0)))
    return state


def save_state(path: str, state: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def fetch_usdt_equity(ex: ccxt.Exchange) -> float:
    """
    Fetch total USDT equity for the swap/futures account.
    This is a simple approach; adapt to your exact Bybit account structure if needed.
    """
    bal = ex.fetch_balance()
    if "USDT" not in bal:
        raise RuntimeError("USDT balance not found in Bybit account.")
    entry = bal["USDT"]
    total = entry.get("total")
    if total is None:
        total = (entry.get("free", 0.0) or 0.0) + (entry.get("used", 0.0) or 0.0)
    return float(total)


def fetch_last_price(ex: ccxt.Exchange, symbol: str) -> float:
    ticker = ex.fetch_ticker(symbol)
    price = ticker.get("last") or ticker.get("close")
    if price is None:
        raise RuntimeError(f"Could not get last price for {symbol}")
    return float(price)


def check_sl_tp(
    last_price: float,
    side: str,
    tp_price: float,
    sl_price: float,
) -> Tuple[bool, bool]:
    """
    Return (hit_tp, hit_sl) given last_price, side, tp/sl prices.
    """
    if side == "long":
        hit_tp = last_price >= tp_price
        hit_sl = last_price <= sl_price
    else:  # short
        hit_tp = last_price <= tp_price
        hit_sl = last_price >= sl_price
    return hit_tp, hit_sl


def compute_pnl_quote(
    entry_price: float,
    close_price: float,
    side: str,
    size: float,
) -> float:
    """
    Approximate realized PnL in quote currency (USDT) for this trade.
    """
    if side == "long":
        return (close_price - entry_price) * size
    else:  # short
        return (entry_price - close_price) * size


def open_position(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,    # 'long' or 'short'
    size: float,
    last_price: float,
    tp_pct: float,
    sl_pct: float,
    dry_run: bool,
) -> Tuple[float, float]:
    """
    Open a LONG/SHORT position at market and return (tp_price, sl_price).
    """
    side_ccxt = "buy" if side == "long" else "sell"
    print(
        f"Opening {side.upper()} — MARKET {side_ccxt.upper()} {symbol} "
        f"qty={size:.6f} (px≈{last_price:.4f})"
    )
    if not dry_run:
        try:
            ex.create_order(symbol, "market", side_ccxt, size, None, {})
        except Exception as e:
            print(f"[ERROR] open {side.upper()} failed: {e}")
            raise

    if side == "long":
        tp_price = last_price * (1.0 + tp_pct)
        sl_price = last_price * (1.0 - sl_pct)
    else:
        tp_price = last_price * (1.0 - tp_pct)
        sl_price = last_price * (1.0 + sl_pct)

    return tp_price, sl_price


def close_position(
    ex: ccxt.Exchange,
    symbol: str,
    side: str,  # 'long' or 'short'
    size: float,
    last_price: float,
    reason: str,
    dry_run: bool,
) -> None:
    """
    Close an existing position at market (reduce-only).
    """
    side_ccxt = "sell" if side == "long" else "buy"
    params = {"reduceOnly": True}
    print(
        f"{reason} — closing existing {side.upper()} at ~{last_price:.4f}, "
        f"qty={size:.6f}"
    )
    if not dry_run:
        try:
            ex.create_order(symbol, "market", side_ccxt, size, None, params)
        except Exception as e:
            print(f"[ERROR] close {side.upper()} on {reason} failed: {e}")
            raise


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    ccxt_symbol = normalize_trade_symbol(args.trade_ticker)
    state_path = args.state_json or default_state_path(args.trade_ticker)
    state = load_state(state_path, ccxt_symbol)

    api_key, api_secret = load_api_keys(
        args.keyfile,
        args.key_name,
        args.key_secret,
    )
    ex = init_bybit(api_key, api_secret, args.testnet)

    today = today_str_utc()

    # --- daily reset (equity_start for daily target) ---
    if state.get("date") != today:
        eq = 0.0
        try:
            eq = fetch_usdt_equity(ex)
        except Exception as e:
            print(f"[WARN] Could not fetch equity_start, using previous/1.0; error: {e}")
            eq = float(state.get("equity_start") or 1.0)

        print(
            f"[DAILY RESET] New day {today}. "
            f"equity_start set to {eq:.4f} USDT"
        )
        state["date"] = today
        state["equity_start"] = eq
        state["realized_today_quote"] = 0.0
        state["stopped_for_today"] = False

    eq0 = float(state.get("equity_start") or 1.0)
    realized_today = float(state.get("realized_today_quote") or 0.0)
    realized_pct = realized_today / eq0 if eq0 > 0 else 0.0

    print(
        f"[DAILY STATUS] date={state['date']} | equity_start={eq0:.4f} "
        f"| realized_today={realized_today:.4f} USDT ({realized_pct*100:.2f}%) "
        f"| stopped_for_today={state.get('stopped_for_today')}"
    )

    # If daily guard already hit, do nothing more (but we KEEP any existing position).
    if state.get("stopped_for_today"):
        print("[GUARD] Daily limit reached previously. No new trades today.")
        save_state(state_path, state)
        return

    # --- load probabilities ---
    btc_prev, btc_last = load_probs(args.btc_json)
    eth_prev, eth_last = load_probs(args.eth_json)

    pos_thr = float(args.pos_thr)
    neg_thr = float(args.neg_thr)

    zone_btc = classify_zone(btc_last, btc_prev, pos_thr, neg_thr)
    zone_eth = classify_zone(eth_last, eth_prev, pos_thr, neg_thr)

    print(
        f"[PROBS] BTC prev={btc_prev:.6f} last={btc_last:.6f} zone={zone_btc} | "
        f"ETH prev={eth_prev:.6f} last={eth_last:.6f} zone={zone_eth}"
    )

    # --- current price of trade symbol ---
    last_price = fetch_last_price(ex, ccxt_symbol)
    print(f"[PRICE] {ccxt_symbol} last_price={last_price:.4f}")

    # --- existing position handling ---
    if state.get("has_position"):
        side = state.get("side")
        size = float(state.get("size") or 0.0)
        entry_price = float(state.get("entry_price") or 0.0)
        tp_price = float(state.get("tp_price") or 0.0)
        sl_price = float(state.get("sl_price") or 0.0)

        if size <= 0.0 or not side:
            print("[WARN] State says has_position but size<=0 or side missing; resetting to flat.")
            state["has_position"] = False
            state["side"] = None
            state["size"] = 0.0
            save_state(state_path, state)
            return

        print(
            f"[POSITION] Existing {side.upper()} | size={size:.6f} | "
            f"entry={entry_price:.4f} | TP={tp_price:.4f} | SL={sl_price:.4f}"
        )

        hit_tp, hit_sl = check_sl_tp(last_price, side, tp_price, sl_price)

        # 1) SL/TP guard (per-trade)
        if hit_tp or hit_sl:
            reason = "TP hit" if hit_tp else "SL hit"
            close_position(ex, ccxt_symbol, side, size, last_price, reason, args.dry_run)

            # approx realized PnL
            pnl_quote = compute_pnl_quote(entry_price, last_price, side, size)
            realized_today += pnl_quote
            state["realized_today_quote"] = realized_today
            realized_pct = realized_today / eq0 if eq0 > 0 else 0.0

            print(
                f"[REALIZED] {reason} | pnl={pnl_quote:.4f} USDT | "
                f"realized_today={realized_today:.4f} ({realized_pct*100:.2f}%)"
            )

            # Once TP or SL is hit, stop trading for the rest of the day.
            state["has_position"] = False
            state["side"] = None
            state["size"] = 0.0
            state["entry_price"] = 0.0
            state["tp_price"] = 0.0
            state["sl_price"] = 0.0
            state["stopped_for_today"] = True

            print("[GUARD] TP/SL reached. No more trades for today.")
            save_state(state_path, state)
            return

        # 2) Exit by probability (neutral or reverse)
        any_neutral = (zone_btc == "NEUTRAL" or zone_eth == "NEUTRAL")
        opposite_signal = (
            (side == "long" and (zone_btc == "SHORT" or zone_eth == "SHORT")) or
            (side == "short" and (zone_btc == "LONG" or zone_eth == "LONG"))
        )

        if any_neutral or opposite_signal:
            reason = "NEUTRAL zone" if any_neutral else "REVERSE signal"
            close_position(ex, ccxt_symbol, side, size, last_price, reason, args.dry_run)

            pnl_quote = compute_pnl_quote(entry_price, last_price, side, size)
            realized_today += pnl_quote
            state["realized_today_quote"] = realized_today
            realized_pct = realized_today / eq0 if eq0 > 0 else 0.0

            print(
                f"[REALIZED] {reason} close | pnl={pnl_quote:.4f} USDT | "
                f"realized_today={realized_today:.4f} ({realized_pct*100:.2f}%)"
            )

            # Daily target guard: if |realized_pct| >= daily_target_pct => stop.
            if abs(realized_pct) >= float(args.daily_target_pct):
                print(
                    f"[GUARD] Daily target reached "
                    f"({realized_pct*100:.2f}% vs target {args.daily_target_pct*100:.2f}%)."
                )
                state["stopped_for_today"] = True

            # Clear position in state
            state["has_position"] = False
            state["side"] = None
            state["size"] = 0.0
            state["entry_price"] = 0.0
            state["tp_price"] = 0.0
            state["sl_price"] = 0.0

            # If this was a reverse signal and we are NOT stopped for the day
            # AND BTC+ETH are aligned in the opposite zone, we can open new reverse trade.
            if opposite_signal and not state["stopped_for_today"]:
                all_bull = (zone_btc == "LONG" and zone_eth == "LONG")
                all_bear = (zone_btc == "SHORT" and zone_eth == "SHORT")

                if all_bull or all_bear:
                    desired_side = "long" if all_bull else "short"
                    print(
                        f"[REVERSE] BTC & ETH aligned as {zone_btc}/{zone_eth}. "
                        f"Opening reverse {desired_side.upper()} position."
                    )

                    # sizing based on current equity
                    try:
                        equity = fetch_usdt_equity(ex)
                    except Exception as e:
                        print(f"[WARN] fetch_usdt_equity failed, using equity_start: {e}")
                        equity = eq0

                    usd_to_use = equity * float(args.risk)
                    size_raw = usd_to_use / max(1e-12, last_price)
                    size_new = float(ex.amount_to_precision(ccxt_symbol, size_raw))
                    if size_new <= 0.0:
                        print(
                            "[WARN] Computed trade size <= 0 after precision; "
                            "skipping reverse open."
                        )
                    else:
                        tp_price, sl_price = open_position(
                            ex, ccxt_symbol, desired_side, size_new,
                            last_price, float(args.tp_pct), float(args.sl_pct),
                            args.dry_run
                        )
                        state["has_position"] = True
                        state["side"] = desired_side
                        state["size"] = size_new
                        state["entry_price"] = last_price
                        state["tp_price"] = tp_price
                        state["sl_price"] = sl_price
                else:
                    print(
                        "[INFO] Reverse signal exit, but BTC & ETH are not aligned in "
                        "a clear opposite zone; staying flat."
                    )

            save_state(state_path, state)
            return

        # 3) No exit condition met -> keep position open
        print(
            f"[HOLD] Keeping existing {side.upper()} open — "
            f"zones: BTC={zone_btc}, ETH={zone_eth}"
        )
        save_state(state_path, state)
        return

    # -----------------------------------------------------------------------
    # No existing position: decide whether to open a new one
    # -----------------------------------------------------------------------
    all_bull = (zone_btc == "LONG" and zone_eth == "LONG")
    all_bear = (zone_btc == "SHORT" and zone_eth == "SHORT")

    if not (all_bull or all_bear):
        print(
            "[FLAT] BTC & ETH not aligned in same non-neutral zone; "
            "no new trade."
        )
        save_state(state_path, state)
        return

    if state.get("stopped_for_today"):
        print(
            "[GUARD] Daily limit reached earlier in the day; "
            "skipping new trade despite aligned signal."
        )
        save_state(state_path, state)
        return

    desired_side = "long" if all_bull else "short"
    print(
        f"[SIGNAL] BTC & ETH both in {zone_btc} zone. "
        f"Opening {desired_side.upper()} on {ccxt_symbol}."
    )

    # Sizing based on equity
    try:
        equity = fetch_usdt_equity(ex)
    except Exception as e:
        print(f"[WARN] fetch_usdt_equity failed, using equity_start: {e}")
        equity = eq0

    usd_to_use = equity * float(args.risk)
    size_raw = usd_to_use / max(1e-12, last_price)
    size = float(ex.amount_to_precision(ccxt_symbol, size_raw))

    print(
        f"[SIZING] equity≈{equity:.4f} USDT | risk={args.risk:.4f} "
        f"| usd_to_use≈{usd_to_use:.4f} | size_raw={size_raw:.6f} | "
        f"size={size:.6f}"
    )

    if size <= 0.0:
        print(
            "[WARN] Computed trade size <= 0 after precision; "
            "skipping new trade."
        )
        save_state(state_path, state)
        return

    tp_price, sl_price = open_position(
        ex, ccxt_symbol, desired_side, size,
        last_price, float(args.tp_pct), float(args.sl_pct),
        args.dry_run
    )

    state["has_position"] = True
    state["side"] = desired_side
    state["size"] = size
    state["entry_price"] = last_price
    state["tp_price"] = tp_price
    state["sl_price"] = sl_price

    save_state(state_path, state)
    print("[DONE] New position opened and state saved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting.")
        sys.exit(1)
