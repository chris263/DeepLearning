#!/usr/bin/env python3
import os
import math
import uuid
import json
import argparse

from coinbase.rest import RESTClient


# ---------- key loader (same style as your main script) ----------
def load_api_keys(pub_key_name: str, sec_key_name: str, keyfile: str) -> tuple[str, str]:
    keyfile = os.path.expanduser(keyfile)
    kv = {}
    if os.path.exists(keyfile):
        with open(keyfile) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()

    api_key = kv.get(pub_key_name)
    api_secret = kv.get(sec_key_name)

    print(f"[DEBUG] Using keyfile: {keyfile}")
    print(f"[DEBUG] Loaded api_key? {'yes' if api_key else 'NO'}")
    print(f"[DEBUG] Loaded api_secret? {'yes' if api_secret else 'NO'}")

    if not api_key or not api_secret:
        raise SystemExit("Missing Coinbase API keys in keyfile.")

    return api_key, api_secret


# ---------- CFM futures balance (same as in SAT) ----------
def fetch_usdc_balance_swap(client: RESTClient) -> float:
    """
    Coinbase US Derivatives (CFM) balance summary:
      - /api/v3/brokerage/cfm/balance_summary
      - use futures_buying_power.value as 'swap balance' in USD
    """
    raw = client.get("/api/v3/brokerage/cfm/balance_summary")
    data = raw.to_dict() if hasattr(raw, "to_dict") else raw

    bs = data.get("balance_summary") or {}
    fb = bs.get("futures_buying_power") or {}
    val_str = fb.get("value", "0") or "0"
    cur = fb.get("currency", "USD")

    try:
        bal = float(val_str)
    except Exception:
        bal = 0.0

    print("[FUTURES BALANCE SUMMARY]")
    print(json.dumps(data, indent=2))
    print(f"[INFO] Futures buying power: {bal} {cur}")
    return bal


# ---------- resolve BTC perp: BTC-PERP-INTX ----------
def resolve_coinbase_perp_product_id(client: RESTClient, base: str = "BTC") -> str:
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

        if base_u not in {contract_code, root_unit, base_id} and not disp_name.startswith(base_u + " "):
            continue

        candidates.append(p)

    if not candidates:
        raise SystemExit(f"No perpetual FUTURE product found for base={base_u}")

    intx = [p for p in candidates if p.get("product_venue") == "INTX"]
    chosen = intx[0] if intx else candidates[0]

    pid = chosen["product_id"]
    print(
        f"[DEBUG] Resolved Coinbase perp product for base={base_u} -> "
        f"{pid} (venue={chosen.get('product_venue')}, display_name={chosen.get('display_name')})"
    )
    print("[DEBUG] Full chosen product JSON:")
    print(json.dumps(chosen, indent=2))
    return pid


# ---------- same order helper as in SAT ----------
def place_coinbase_perp_order(
    client: RESTClient,
    product_id: str,
    side: str,
    base_size: float,
    last_close: float,
):
    """
    Place a MARKET IOC order on Coinbase for a perp future using base_size.
    """
    side = side.upper()
    base_str = f"{base_size:.8f}"

    payload = {
        "client_order_id": str(uuid.uuid4()),
        "product_id": product_id,
        "side": side,  # "BUY" or "SELL"
        "order_configuration": {
            "market_market_ioc": {
                "base_size": base_str
            }
        },
    }

    print(
        f"Submitting Coinbase MARKET {side}:\n"
        f"  product_id = {product_id}\n"
        f"  base_size  = {base_str} BTC\n"
        f"  est notional ≈ {base_size * last_close:.2f}"
    )
    print(f"[DEBUG] Coinbase order payload: {json.dumps(payload, indent=2)}")

    resp = client.post("/api/v3/brokerage/orders", data=payload)
    print("[OK] Coinbase order response:")
    print(resp)
    return resp


# ---------- main test flow ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub-key", default="API_COINBASE_SAT")
    ap.add_argument("--sec-key", default="API_COINBASE_SECRET_SAT")
    ap.add_argument("--keys-file", default="~/.ssh/coinex_keys.env")
    ap.add_argument("--base", default="BTC")
    ap.add_argument("--side", choices=["BUY", "SELL"], default="SELL",
                    help="Direction of test order (BUY=long, SELL=short)")
    ap.add_argument("--risk", type=float, default=0.05,
                    help="Fraction of futures_buying_power to use (e.g. 0.05=5%)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print everything but do NOT place the order.")
    args = ap.parse_args()

    # 1) Client
    api_key, api_secret = load_api_keys(args.pub_key, args.sec_key, args.keys_file)
    client = RESTClient(api_key=api_key, api_secret=api_secret)

    # 2) Futures buying power
    bp = fetch_usdc_balance_swap(client)
    if bp <= 0:
        print("[FATAL] No futures buying power; aborting test.")
        return

    # 3) Resolve BTC perp product
    product_id = resolve_coinbase_perp_product_id(client, base=args.base)

    # 4) Load product increments & price
    prod_raw = client.get_product(product_id)
    prod = prod_raw.to_dict() if hasattr(prod_raw, "to_dict") else prod_raw

    last_price = float(prod.get("price") or "0")
    base_inc   = float(prod.get("base_increment") or "0.0001")
    base_min   = float(prod.get("base_min_size") or "0.0001")

    print(f"[DEBUG] last_price={last_price}, base_increment={base_inc}, base_min_size={base_min}")

    # 5) Compute notional and base_size (same idea as in SAT)
    usd_to_use = bp * args.risk
    if usd_to_use <= 0:
        print("[FATAL] usd_to_use <= 0 after risk factor; aborting.")
        return

    base_size_raw = usd_to_use / max(1e-12, last_price)
    steps = math.floor(base_size_raw / base_inc)
    base_size = steps * base_inc
    notional = base_size * last_price

    print(
        f"[DEBUG] futures_bp={bp:.4f} usd_to_use={usd_to_use:.4f} "
        f"base_size_raw={base_size_raw:.8f} base_size={base_size:.8f} "
        f"notional≈{notional:.4f}"
    )

    if base_size < base_min:
        print(
            f"[WARN] base_size={base_size:.8f} < base_min_size={base_min:.8f}; "
            "increase risk or deposit more funds to test."
        )
        return

    # 6) Dry-run or real order
    if args.dry_run:
        print(
            f"[DRY-RUN] Would place MARKET {args.side} on {product_id} "
            f"base_size={base_size:.8f} (notional≈{notional:.2f})"
        )
        return

    try:
        place_coinbase_perp_order(
            client=client,
            product_id=product_id,
            side=args.side,
            base_size=base_size,
            last_close=last_price,
        )
    except Exception as e:
        print(f"[ERROR] Coinbase test order failed: {e}")


if __name__ == "__main__":
    main()
