#!/usr/bin/env bash
LOG_DIR="/home/production/logs"
LOG_FILE="$LOG_DIR/load_trade.log"
WORKDIR="/home/production/DeepLearning"

exec >> "$LOG_FILE" 2>&1
echo
echo "================= local_trade ================="
echo "started at: $(date '+%F %T %Z')"
echo "log file  : $LOG_FILE"
echo "==============================================="

cd "$WORKDIR"

# Always update 30m files
python3 update_bybit_json.py --json-file /home/production/tmp/ETHUSDT_30m_6m.json --symbol ETH/USDT:USDT --timeframe 30m
# python3 update_bybit_json.py --json-file /home/production/tmp/ETHUSDC_30m_6m.json --symbol ETH/USDC:USDC --timeframe 30m
python3 update_bybit_json.py --json-file /home/production/tmp/BTCUSDT_30m_6m.json --symbol BTC/USDT:USDT --timeframe 30m

# Run 1h updates only near full hour
MINUTE=$(date +%M)
echo "Current minute: $MINUTE"

if [ "$MINUTE" -le 2 ]; then
    echo "Top of hour detected — running 1h updates."
    python3 update_bybit_json.py --json-file /home/production/tmp/ETHUSDT_1h_6m.json --symbol ETH/USDT:USDT --timeframe 1h
    python3 update_bybit_json.py --json-file /home/production/tmp/BTCUSDT_1h_6m.json --symbol BTC/USDT:USDT --timeframe 1h
    # python3 update_bybit_json.py --json-file /home/production/tmp/BTCUSDC_1h_6m.json --symbol BTC/USDC:USDC --timeframe 1h
else
    echo "Not top of hour — skipping 1h updates."
fi

echo "Load finished at: $(date '+%F %T %Z')"
echo "================================================"

echo
echo "================= local_trade ================="
echo "started at: $(date '+%F %T %Z')"
echo "log file  : $LOG_FILE"
echo "==============================================="

echo "Coinex ETH 30m"
python3 $WORKDIR/lstm_sat_coinex.py  --model-dir "$WORKDIR/lstm/eth_lstm_30m_2025/"   --bars-json "/home/production/tmp/ETHUSDT_30m_6m.json" --ticker ETHUSDT --timeframe 30m --pub_key API_KEY_ETH --sec_key API_SECRET_ETH --debug

echo
echo "Bybit BTC 30M"
python3 $WORKDIR/lstm_sat_bybit.py  --model-dir "$WORKDIR/lstm/btc_lstm_30m_2025/"  --bars-json "/home/production/tmp/BTCUSDT_30m_6m.json" --ticker BTCUSDT --timeframe 30m  --pub_key API_BYBIT_SAT --sec_key API_BYBIT_SECRET_SAT --debug



if [ "$MINUTE" -le 2 ]; then
	echo 
	echo "Coinex BTC 1H"
	python3 $WORKDIR/lstm_sat_coinex.py  --model-dir "$WORKDIR/lstm/btc_lstm_1h_2025/"   --bars-json "/home/production/tmp/BTCUSDT_1h_6m.json" --ticker BTCUSDT --timeframe 1h  --pub_key API_KEY --sec_key API_SECRET --debug

	echo
	echo "Bybit ETH 1H"
	python3 $WORKDIR/lstm_sat_bybit.py  --model-dir "$WORKDIR/lstm/eth_lstm_1h_2025/"  --bars-json "/home/production/tmp/ETHUSDT_1h_6m.json" --ticker ETHUSDT --timeframe 1h  --pub_key API_BYBIT_BTC --sec_key API_BYBIT_SECRET_BTC --debug
    
   echo
   echo "CROSS TRADE SOLUSDT"
   python3 cross_prob_bybit.py --btc-json ~/guards/coinex_daily_profit_btc_lstm_1h_2025.json --eth-json ~/guards/bybit_daily_profit_eth_lstm_1h_2025.json --trade-ticker SOLUSDT --key-name API_CROSSTRADE --key-secret API_CROSSTRADE_SECRET --verbose
    
    # echo
    # echo "COINBASE SAT BTCUSDC 1h"
    # python3 $WORKDIR/lstm_sat_coinbase.py  --model-dir "$WORKDIR/lstm/btc_lstm_1h_2025/"  --bars-json "/home/production/tmp/BTCUSDT_1h_6m.json" --ticker BTCUSDC --timeframe 1h --pub_key API_COINBASE_SAT --sec_key API_COINBASE_SECRET_SAT --debug

else
    echo "Not running 1 hour scripts."
fi


echo
echo "Running script finished at: $(date '+%F %T %Z')"
echo "================================================"
