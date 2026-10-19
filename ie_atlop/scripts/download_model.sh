#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints/bert-base-cased
revision=cd5ef92a9fb2f889e972770a36d4ed042daf221e
for filename in config.json vocab.txt tokenizer_config.json pytorch_model.bin; do
    curl --fail --location --retry 3 "https://huggingface.co/google-bert/bert-base-cased/resolve/$revision/$filename" -o "checkpoints/bert-base-cased/$filename"
done
