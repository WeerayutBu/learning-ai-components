#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p checkpoints
curl --fail --location --retry 3 https://github.com/wzhouad/ATLOP/releases/download/1.0/atlop-bert-base -o checkpoints/atlop-bert-base.partial
printf '%s  %s\n' '16f5d36738731cb00912b03586d51e647e4a03af573777e766be394f9c03d3e5' 'checkpoints/atlop-bert-base.partial' | sha256sum --check
mv checkpoints/atlop-bert-base.partial checkpoints/atlop-bert-base
