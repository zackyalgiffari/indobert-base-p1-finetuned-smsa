#!/usr/bin/env bash
# Runs ON the Colab VM, from /content/app. Arguments are forwarded to src.train.
set -euo pipefail
echo "[setup] $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Only installs what Colab lacks; already-satisfied packages are left untouched,
# so the preinstalled torch/pyarrow stack is not upgraded underneath other libs.
python -m pip install -q -r requirements.txt
echo "[setup] done"
python -m src.train "$@"
