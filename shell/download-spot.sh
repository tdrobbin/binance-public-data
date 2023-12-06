source ./venv/bin/activate
set -euxo pipefail
export STORE_DIRECTORY=/mnt/pool1/binance-public-data/
python3 download-kline.py -t spot -i 1m -skip-daily 1
python3 download-trade.py -t spot -skip-daily 1
python3 download-aggTrade.py -t spot -skip-daily 1
