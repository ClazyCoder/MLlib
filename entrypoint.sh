#!/bin/bash

MODE=${1:-train}
CONFIG_PATH=${2:-/app/configs/default.yaml}

echo "Using config: $CONFIG_PATH"
echo "Using mode: $MODE"

if [ $MODE == "train" ]; then
    echo "[INFO] Training mode."
    python main.py --mode train --config $CONFIG_PATH
elif [ $MODE == "test" ]; then
    echo "[INFO] Testing mode."
    python main.py --mode test --config $CONFIG_PATH
fi

echo "[INFO] Done."