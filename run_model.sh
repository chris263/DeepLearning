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

cd $WORKDIR

python3 update_bybit_json.py --json-file /home/production/tmp/ETHUSDT_30m_2y.json --symbol ETH/USDT:USDT --timeframe 30m
python3 update_bybit_json.py --json-file /home/production/tmp/ETHUSDT_1h_2y.json  --symbol ETH/USDT:USDT --timeframe 1h
python3 update_bybit_json.py --json-file /home/production/tmp/BTCUSDT_30m_2y.json --symbol BTC/USDT:USDT --timeframe 30m
python3 update_bybit_json.py --json-file /home/production/tmp/BTCUSDT_1h_2y.json  --symbol BTC/USDT:USDT --timeframe 1h

echo "Load finished at: $(date '+%F %T %Z')"
echo "================================================"

echo
echo "================= local_trade ================="
echo "started at: $(date '+%F %T %Z')"
echo "log file  : $LOG_FILE"
echo "==============================================="

echo "Coinex ETH 30m script"
python3 $WORKDIR/lstm_sat_coinex.py  --model-dir "$WORKDIR/lstm/eth_lstm_30m_2025/"   --bars-json "/home/production/tmp/ETHUSDT_30m_2y.json" --ticker ETHUSDT --timeframe 30m --pub_key API_KEY_ETH --sec_key API_SECRET_ETH --debug

echo
echo "Bybit SAT BTC 30M"
python3 $WORKDIR/lstm_sat_bybit.py  --model-dir "$WORKDIR/lstm/btc_lstm_30m_2025/"  --bars-json "/home/production/tmp/BTCUSDT_30m_2y.json" --ticker BTCUSDT --timeframe 30m  --pub_key API_BYBIT_SAT --sec_key API_BYBIT_SECRET_SAT --debug

echo 
echo "Coinex BTC 1h script"
python3 $WORKDIR/lstm_sat_coinex.py  --model-dir "$WORKDIR/lstm/btc_lstm_1h_2025/"   --bars-json "/home/production/tmp/BTCUSDT_1h_2y.json" --ticker BTCUSDT --timeframe 1h  --pub_key API_KEY --sec_key API_SECRET --debug

echo
echo "Bybit ETH 1h - Conta: BTC 1h"
python3 $WORKDIR/lstm_sat_bybit.py  --model-dir "$WORKDIR/lstm/eth_lstm_1h_2025/"  --bars-json "/home/production/tmp/ETHUSDT_1h_2y.json" --ticker ETHUSDT --timeframe 1h  --pub_key API_BYBIT_BTC --sec_key API_BYBIT_SECRET_BTC --debug



echo
echo "Running script finished at: $(date '+%F %T %Z')"
echo "================================================"
