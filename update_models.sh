#!/usr/bin/env bash
LOG_DIR="/home/production/logs"
LOG_FILE="$LOG_DIR/update_models.log"
WORKDIR="/home/production/DeepLearning"

exec >> "$LOG_FILE" 2>&1
echo
echo "================= update models ================="
echo "started at: $(date '+%F %T %Z')"
echo "log file  : $LOG_FILE"
echo "================================================="

cd "$WORKDIR"

python3 update_training_json.py --json-file /home/production/tmp/ETHUSDT_30m_3y.json --symbol ETH/USDT:USDT --timeframe 30m
python3 update_training_json.py --json-file /home/production/tmp/BTCUSDT_30m_3y.json --symbol BTC/USDT:USDT --timeframe 30m

python3 update_training_json.py --json-file /home/production/tmp/ETHUSDT_1h_3y.json --symbol ETH/USDT:USDT --timeframe 1h
python3 update_training_json.py --json-file /home/production/tmp/BTCUSDT_1h_3y.json --symbol BTC/USDT:USDT --timeframe 1h

echo
echo "Training Model ETHUSDT 30M"
python3 training/lstm_trading_ichimoku_longshort.py  --db-url /home/production/tmp/ETHUSDT_30m_3y.json --ticker ETHUSDT --timeframe 30m --train-start 2023-01-01 --train-end 2024-12-31 --test-end 2025-11-24 --lookback 64 --horizon 4 --epochs 50 --batch-size 256 --pos-thr 0.55 --sl-pct 0.02 --tp-pct 0.06 --out-dir /home/production/training/ETHUSDT_30m --bundle-dir /home/production/training/ETHUSDT_30m

echo
echo "Training Model BTCUSDT 30M"
python3 training/lstm_trading_ichimoku_longshort.py  --db-url /home/production/tmp/BTCUSDT_30m_3y.json --ticker BTCUSDT --timeframe 30m --train-start 2023-01-01 --train-end 2024-12-31 --test-end 2025-11-24 --lookback 64 --horizon 4 --epochs 50 --batch-size 256 --pos-thr 0.55 --sl-pct 0.02 --tp-pct 0.06 --out-dir /home/production/training/BTCUSDT_30m --bundle-dir /home/production/training/BTCUSDT_30m

echo
echo "Training Model ETHUSDT 1H"
python3 training/lstm_trading_ichimoku_longshort.py  --db-url /home/production/tmp/ETHUSDT_1h_3y.json --ticker ETHUSDT --timeframe 1h --train-start 2023-01-01 --train-end 2024-12-31 --test-end 2025-11-24 --lookback 64 --horizon 4 --epochs 50 --batch-size 256 --pos-thr 0.55 --sl-pct 0.02 --tp-pct 0.06 --out-dir /home/production/training/ETHUSDT_1h --bundle-dir /home/production/training/ETHUSDT_1h

echo
echo "Training Model BTCUSDT 1H"
python3 training/lstm_trading_ichimoku_longshort.py  --db-url /home/production/tmp/BTCUSDT_1h_3y.json --ticker BTCUSDT --timeframe 1h --train-start 2023-01-01 --train-end 2024-12-31 --test-end 2025-11-24 --lookback 64 --horizon 4 --epochs 50 --batch-size 256 --pos-thr 0.55 --sl-pct 0.02 --tp-pct 0.06 --out-dir /home/production/training/BTCUSDT_1h --bundle-dir /home/production/training/BTCUSDT_1h



echo
echo
echo "================= moving files ================="
cp /home/production/training/ETHUSDT_30m/model.pt $WORKDIR/lstm/ethusdt_lstm_30m_2025/model.pt
cp /home/production/training/ETHUSDT_30m/meta.json $WORKDIR/lstm/ethusdt_lstm_30m_2025/meta.json
cp /home/production/training/ETHUSDT_30m/preprocess.json $WORKDIR/lstm/ethusdt_lstm_30m_2025/preprocess.json
echo 
cp /home/production/training/BTCUSDT_30m/model.pt $WORKDIR/lstm/btc_lstm_30m_2025/model.pt
cp /home/production/training/BTCUSDT_30m/meta.json $WORKDIR/lstm/btc_lstm_30m_2025/meta.json
cp /home/production/training/BTCUSDT_30m/preprocess.json $WORKDIR/lstm/btc_lstm_30m_2025/preprocess.json
echo 
cp /home/production/training/ETHUSDT_1h/model.pt $WORKDIR/lstm/eth_lstm_1h_2025/model.pt
cp /home/production/training/ETHUSDT_1h/meta.json $WORKDIR/lstm/eth_lstm_1h_2025/meta.json
cp /home/production/training/ETHUSDT_1h/preprocess.json $WORKDIR/lstm/eth_lstm_1h_2025/preprocess.json
echo 
cp /home/production/training/BTCUSDT_1h/model.pt $WORKDIR/lstm/btc_lstm_1h_2025/model.pt
cp /home/production/training/BTCUSDT_1h/meta.json $WORKDIR/lstm/btc_lstm_1h_2025/meta.json
cp /home/production/training/BTCUSDT_1h/preprocess.json $WORKDIR/lstm/btc_lstm_1h_2025/preprocess.json
echo
echo "Process finished at: $(date '+%F %T %Z')"
echo "================================================"
