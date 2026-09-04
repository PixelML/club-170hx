#!/usr/bin/env bash
# PP4 + MTP x5, partition 14/12/12/7, enforce-eager. This decides whether the
# PP4 breakage seen on our 4x box (3.35 tok/s, acceptance 1.34, repetitive
# greedy output) is our box or the build.
set -euo pipefail
PORT="${PORT:-18099}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/glm53}"
NAME="${NAME:-glm53-pp4}"
DEVICES="${DEVICES:-0,1,2,3}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=${DEVICES}\"" \
  --shm-size 16g --ipc=host -p "127.0.0.1:${PORT}:8000" \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  -e NCCL_VERSION=2.28.3-1 -e TORCH_CUDA_ARCH_LIST=8.0 -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_PP_LAYER_PARTITION="14,12,12,7" \
  --mount type=bind,src="$MODEL_DIR",dst=/model,readonly \
  ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903 \
  vllm serve /model --served-model-name glm-5.3-flash \
  --pipeline-parallel-size 4 --enforce-eager \
  --max-model-len 524288 --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45

echo "[launch_pp4] container $NAME up on port $PORT, PP4 partition 14/12/12/7, k=5, enforce-eager"
