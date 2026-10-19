#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
for split in train_annotated dev test train_distant; do
    ../.venv/bin/python -m infrastructure.data_format \
        --source "data/raw/docred/$split.json" \
        --output "data/docred/$split.json"
done
