#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/docred
# IDs from the Google Drive folder linked by the official DocRED data README.
uv tool run --from gdown gdown 1NN33RzyETbanw4Dg2sRrhckhWpzuBQS9 -O data/docred/train_annotated.json
uv tool run --from gdown gdown 1Qr4Jct2IJ9BVI86_mCk_Pz0J32ww9dYw -O data/docred/train_distant.json
uv tool run --from gdown gdown 1AHUm1-_V9GCtGuDcc8XrMUCJE8B-HHoL -O data/docred/dev.json
uv tool run --from gdown gdown 1lAVDcD94Sigx7gR3jTfStI66o86cflum -O data/docred/test.json
uv tool run --from gdown gdown 11liQ1Mfb8hpy5rOobtDDSF_rxFrld1Os -O data/rel2id.json
