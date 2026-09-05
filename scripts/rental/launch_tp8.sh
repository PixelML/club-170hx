#!/usr/bin/env bash
# TP8 across all 8 cards, k=3. Direct-exec, no docker.
set -euo pipefail
PORT="${PORT:-18098}"
K="${K:-3}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/glm53}"
NAME="${NAME:-glm53-tp8}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
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
nohup vllm serve "$MODEL_DIR" --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --max-model-len 524288 \
  --gpu-memory-utilization 0.92 --max-num-seqs 16 \
  --max-num-batched-tokens 8192 --enable-prefix-caching \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}' \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --port "$PORT" > "$LOGDIR/$NAME.log" 2>&1 &
echo $! > "$PIDDIR/$NAME.pid"

echo "[launch_tp8] pid $(cat "$PIDDIR/$NAME.pid") on port $PORT, TP8, k=$K"
