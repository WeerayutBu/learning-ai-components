#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$(cd .. && pwd)/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
mkdir -p checkpoints artifacts
python -m bootstrap.train --data_dir ./data/docred \
--transformer_type bert \
--model_name_or_path checkpoints/bert-base-cased \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed 66 \
--num_class 97 \
--save_path checkpoints/local-bert.pt 2>&1 | tee artifacts/train.log
python src/interfaces/compare.py artifacts/train.log --completed > artifacts/comparison.tmp
mv artifacts/comparison.tmp results.json
cat results.json
