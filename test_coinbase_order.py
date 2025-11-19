import os, math, uuid
from json import dumps
from coinbase.rest import RESTClient
from requests import HTTPError

pub_key_name = "API_COINBASE_SAT"
sec_key_name = "API_COINBASE_SECRET_SAT"
keyfile = os.path.expanduser("~/.ssh/coinex_keys.env")

kv = {}
if os.path.exists(keyfile):
    for line in open(keyfile, "r"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()

api_key    = kv.get(pub_key_name)
api_secret = kv.get(sec_key_name)

client = RESTClient(api_key=api_key, api_secret=api_secret)

# 1) Show futures buying power (same as SAT)
fs_raw = client.get("/api/v3/brokerage/cfm/balance_summary")
fs = fs_raw.to_dict() if hasattr(fs_raw, "to_dict") else fs_raw
print("[FUTURES BALANCE SUMMARY]")
print(dumps(fs, indent=2))

bp = float(fs.get("balance_summary", {})
             .get("futures_buying_power", {})
             .get("value", 0.0) or 0.0)
print(f"Futures buying power: {bp} USD")


# 2) Fetch product info for what we *think* we’re trading
product_id = "BTC-USDC"   # try also "BTC-USD" to see the difference
prod = client.get_product(product_id).to_dict()
print("[PRODUCT INFO]", dumps(prod, indent=2))

# 3) Try a tiny market order (should be well below your buying power)
notional = min(10.0, bp * 0.01)  # $10 or 1% of buying power
if notional <= 0:
    print("No buying power; aborting test.")
    raise SystemExit

last_px = float(prod["price"])
base_size = notional / last_px
print(f"Test order: product_id={product_id}, notional≈{notional}, base_size≈{base_size}")

payload = {
    "client_order_id": str(uuid.uuid4()),
    "product_id": product_id,
    "side": "SELL",  # or "BUY"
    "order_configuration": {
        "market_market_ioc": {
            "base_size": f"{base_size:.8f}"
        }
    },
}

print("[DEBUG] Payload:", dumps(payload, indent=2))

try:
    resp = client.post("/api/v3/brokerage/orders", data=payload)
    print("[OK] Order placed:", resp)
except HTTPError as e:
    print("[HTTP ERROR]", e)
    try:
        print("[BODY]", e.response.text)
    except Exception:
        pass
except Exception as e:
    print("[GENERIC ERROR]", e)
