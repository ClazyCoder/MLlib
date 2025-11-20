#!/bin/bash

MODE=${1:-train}
CONFIG_NAME=${2:-default.yaml}

echo "Using config: $CONFIG_NAME"
echo "Using mode: $MODE"

if [ "$MODE" = "train" ]; then
    echo "[INFO] Training mode."
    uv run src/main.py --mode train --config $CONFIG_NAME
elif [ "$MODE" = "test" ]; then
    echo "[INFO] Testing mode."
    uv run src/main.py --mode test --config $CONFIG_NAME
else
    echo "[ERROR] Invalid mode: $MODE"
    exit 1
fi

echo "[INFO] Done."