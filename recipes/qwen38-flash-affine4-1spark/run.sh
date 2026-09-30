#!/usr/bin/env bash
set -euo pipefail
: "${QWEN_DATA:?Set QWEN_DATA to the absolute directory populated by stage.py}"
case "$QWEN_DATA" in /*) ;; *) echo 'QWEN_DATA must be absolute' >&2; exit 1;; esac
python3 "$(dirname "$0")/stage.py" --directory "$QWEN_DATA" --verify-only
mkdir -p "$QWEN_DATA/cache"
exec docker run --rm --name "${QWEN_CONTAINER:-spark-qwen-affine4}" --gpus all --network host --ipc host --cap-add IPC_LOCK --ulimit memlock=-1 \
 -v "$QWEN_DATA/target:/target:ro" -v "$QWEN_DATA/cache:/cache" \
 -e OMP_NUM_THREADS=4 -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=2 \
 -e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/hf -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch -e PYTHONUNBUFFERED=1 \
 "${QWEN_IMAGE:-spark-qwen-affine4:20260930}" serve /target --backend cuda --name qwen3.8-flash-next \
 --host 127.0.0.1 --port 8898 --context 262144 --parallel 4 --max-tokens 8192 --mtp-drafts 6 --mtp-confidence 0.6 \
 --kv-dtype int8 --ple-on-ssd --prompt-cache-gib 2 --snapshot-dir none --no-thinking --no-update-check
