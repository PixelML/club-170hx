#!/usr/bin/env bash
# TP8 across all 8 cards, k=3.
set -euo pipefail
PORT="${PORT:-18098}"
K="${K:-3}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/glm53}"
NAME="${NAME:-glm53-tp8}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=${DEVICES}\"" \
  --shm-size 16g --ipc=host -p "127.0.0.1:${PORT}:8000" \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  -e NCCL_VERSION=2.28.3-1 -e TORCH_CUDA_ARCH_LIST=8.0 -e VLLM_TARGET_DEVICE=cuda \
  --mount type=bind,src="$MODEL_DIR",dst=/model,readonly \
  ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903 \
  vllm serve /model --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --max-model-len 524288 \
  --gpu-memory-utilization 0.92 --max-num-seqs 16 \
  --max-num-batched-tokens 8192 --enable-prefix-caching \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}' \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45

echo "[launch_tp8] container $NAME up on port $PORT, TP8, k=$K"
