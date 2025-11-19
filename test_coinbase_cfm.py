#!/usr/bin/env python3
import os, uuid, math
from coinbase.rest import RESTClient

# import your helpers from lstm_sat_coinbase.py
from lstm_sat_coinbase import (
    make_coinbase_client,
    resolve_symbol,
    fetch_usdc_balance_swap,
    get_swap_position,
)

PUB = "API_COINBASE_SAT"
SEC = "API_COINBASE_SECRET_SAT"
TICKER = "BTCUSDC"   # or whatever the model uses

def main():
    print("=== Coinbase CFM probe ===")

    client = make_coinbase_client(PUB, SEC)
    product_id = resolve_symbol(client, TICKER)
    print(f"[INFO] Resolved ticker {TICKER} -> product_id={product_id}")

    bal = fetch_usdc_balance_swap(client)
    print(f"[INFO] Perps/futures balance used for sizing: {bal}")

    pos = get_swap_position(client, product_id)
    print(f"[INFO] Current position for {product_id}: {pos}")

    print("No orders were sent. If balance > 0 and this didn’t error, "
          "your Coinbase/CFM connection is working.")

if __name__ == "__main__":
    main()
