#!/usr/bin/env bash
# PP4 + MTP x5, partition 14/12/12/7, enforce-eager. Direct-exec, no docker.
set -euo pipefail
PORT="${PORT:-18099}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/glm53}"
NAME="${NAME:-glm53-pp4}"
DEVICES="${DEVICES:-0,1,2,3}"
LOGDIR="${LOGDIR:-/workspace/logs}"
PIDDIR="${PIDDIR:-/workspace/pids}"
mkdir -p "$LOGDIR" "$PIDDIR"

if [[ -f "$PIDDIR/$NAME.pid" ]]; then
  kill "$(cat "$PIDDIR/$NAME.pid")" 2>/dev/null || true
  sleep 2
fi

CUDA_VISIBLE_DEVICES="$DEVICES" \
HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ENGINE_READY_TIMEOUT_S=1800 VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
NCCL_VERSION=2.28.3-1 TORCH_CUDA_ARCH_LIST=8.0 VLLM_TARGET_DEVICE=cuda \
VLLM_PP_LAYER_PARTITION="14,12,12,7" \
nohup vllm serve "$MODEL_DIR" --served-model-name glm-5.3-flash \
  --pipeline-parallel-size 4 --enforce-eager \
  --max-model-len 524288 --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --port "$PORT" > "$LOGDIR/$NAME.log" 2>&1 &
echo $! > "$PIDDIR/$NAME.pid"

echo "[launch_pp4] pid $(cat "$PIDDIR/$NAME.pid") on port $PORT, PP4 partition 14/12/12/7, k=5, enforce-eager"
