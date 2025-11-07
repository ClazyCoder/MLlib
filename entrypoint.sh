#!/bin/bash

MODE=${1:-train}
CONFIG_NAME=${2:-default.yaml}

echo "Using config: $CONFIG_NAME"
echo "Using mode: $MODE"

if [ $MODE == "train" ]; then
    echo "[INFO] Training mode."
    python main.py --mode train --config $CONFIG_NAME
elif [ $MODE == "test" ]; then
    echo "[INFO] Testing mode."
    python main.py --mode test --config $CONFIG_NAME
fi

echo "[INFO] Done."