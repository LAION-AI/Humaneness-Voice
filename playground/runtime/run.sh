#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-/mnt/nvme/hf_cache}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec env -u LD_LIBRARY_PATH /home/c4r33u19/moss15v2/.venv/bin/python -m uvicorn small_app:app --host 127.0.0.1 --port "${SMALL_PORT:-8792}"
