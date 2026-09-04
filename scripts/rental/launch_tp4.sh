#!/usr/bin/env bash
# TP4 recipe of record, cards 0-3. Env overrides: K (spec depth, default 3),
# NO_CUSTOM_AR=0 to re-enable custom all-reduce (P2P path test), PORT (default 18098).
set -euo pipefail
K="${K:-3}"
PORT="${PORT:-18098}"
NO_CUSTOM_AR="${NO_CUSTOM_AR:-1}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/glm53}"
NAME="${NAME:-glm53-tp4}"
DEVICES="${DEVICES:-0,1,2,3}"
EXTRA_MM_FLAG="${EXTRA_MM_FLAG:-}"   # set to '--limit-mm-per-prompt {"image": 0, "video": 0}' if mm-profiling crashes

ARGS=(--tensor-parallel-size 4 --max-model-len 524288 \
  --gpu-memory-utilization 0.92 --max-num-seqs 16 \
  --max-num-batched-tokens 8192 --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}' \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45)

if [[ "$NO_CUSTOM_AR" == "1" ]]; then
  ARGS+=(--disable-custom-all-reduce)
fi
if [[ -n "$EXTRA_MM_FLAG" ]]; then
  ARGS+=($EXTRA_MM_FLAG)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=${DEVICES}\"" \
  --shm-size 16g --ipc=host -p "127.0.0.1:${PORT}:8000" \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  -e NCCL_VERSION=2.28.3-1 -e TORCH_CUDA_ARCH_LIST=8.0 -e VLLM_TARGET_DEVICE=cuda \
  --mount type=bind,src="$MODEL_DIR",dst=/model,readonly \
  ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903 \
  vllm serve /model --served-model-name glm-5.3-flash "${ARGS[@]}"

echo "[launch_tp4] container $NAME up on port $PORT, k=$K, custom-all-reduce-disabled=$NO_CUSTOM_AR, devices=$DEVICES"
