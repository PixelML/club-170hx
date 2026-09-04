#!/usr/bin/env bash
# TP4 recipe of record, cards 0-3, direct-exec (no nested docker — the
# rental IS the club-170hx image container already). Env overrides:
# K (spec depth, default 3), NO_CUSTOM_AR=0 to re-enable custom all-reduce
# (P2P path test), PORT (default 18098), DEVICES (default 0,1,2,3).
set -euo pipefail
K="${K:-3}"
PORT="${PORT:-18098}"
NO_CUSTOM_AR="${NO_CUSTOM_AR:-1}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/glm53}"
NAME="${NAME:-glm53-tp4}"
DEVICES="${DEVICES:-0,1,2,3}"
EXTRA_MM_FLAG="${EXTRA_MM_FLAG:-}"   # e.g. '--limit-mm-per-prompt {"image": 0, "video": 0}'
LOGDIR="${LOGDIR:-/workspace/logs}"
PIDDIR="${PIDDIR:-/workspace/pids}"
mkdir -p "$LOGDIR" "$PIDDIR"

if [[ -f "$PIDDIR/$NAME.pid" ]]; then
  kill "$(cat "$PIDDIR/$NAME.pid")" 2>/dev/null || true
  sleep 2
fi

ARGS=(--served-model-name glm-5.3-flash --tensor-parallel-size 4 --max-model-len 524288 \
  --gpu-memory-utilization 0.92 --max-num-seqs 16 \
  --max-num-batched-tokens 8192 --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}' \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${K}}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --port "$PORT")

if [[ "$NO_CUSTOM_AR" == "1" ]]; then
  ARGS+=(--disable-custom-all-reduce)
fi
if [[ -n "$EXTRA_MM_FLAG" ]]; then
  ARGS+=($EXTRA_MM_FLAG)
fi

CUDA_VISIBLE_DEVICES="$DEVICES" \
HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ENGINE_READY_TIMEOUT_S=1800 VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
NCCL_VERSION=2.28.3-1 TORCH_CUDA_ARCH_LIST=8.0 VLLM_TARGET_DEVICE=cuda \
nohup vllm serve "$MODEL_DIR" "${ARGS[@]}" > "$LOGDIR/$NAME.log" 2>&1 &
echo $! > "$PIDDIR/$NAME.pid"

echo "[launch_tp4] pid $(cat "$PIDDIR/$NAME.pid") on port $PORT, k=$K, custom-all-reduce-disabled=$NO_CUSTOM_AR, devices=$DEVICES"
