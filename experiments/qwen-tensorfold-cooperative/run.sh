#!/usr/bin/env bash
set -euo pipefail
# Run on the Spark. Mount only the verified model and runtime cache.
: "${QWEN_DATA:?Set QWEN_DATA to the absolute directory populated by stage.py}"
case "$QWEN_DATA" in /*) ;; *) echo 'QWEN_DATA must be absolute' >&2; exit 1;; esac
IMAGE=${QWEN_IMAGE:-spark-qwen-cooperative:0.3.6.1}
NAME=${QWEN_CONTAINER:-spark-qwen-cooperative}
python3 "$(dirname "$0")/stage.py" --directory "$QWEN_DATA" --verify-only
python3 "$(dirname "$0")/overlay.py" --directory "$QWEN_DATA"
mkdir -p "$QWEN_DATA/cache"
docker run --rm --name "$NAME" --gpus all --network host --ipc host \
  --cap-add IPC_LOCK --ulimit memlock=-1 \
  -v "$QWEN_DATA/target:/target:ro" \
  -v "$QWEN_DATA/config-overlay.json:/target/config.json:ro" \
  -v "$QWEN_DATA/cache:/cache" \
  -e OMP_NUM_THREADS=4 -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=2 \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/hf -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCH_EXTENSIONS_DIR=/cache/torch -e PYTHONUNBUFFERED=1 \
  -e SPARK_COOPERATIVE="${SPARK_COOPERATIVE:-1}" \
  -e SPARK_PREFILL_ROWS="${SPARK_PREFILL_ROWS:-2048}" \
  -e SPARK_DECODE_BURST="${SPARK_DECODE_BURST:-4}" \
  -e SPARK_FINAL_HEAD="${SPARK_FINAL_HEAD:-1}" \
  "$IMAGE" serve /target --backend cuda --name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8898 --context 262144 --parallel 4 \
  --max-tokens 8192 --mtp-drafts 6 --prompt-cache-gib 2 \
  --snapshot-dir none --no-thinking --no-update-check
