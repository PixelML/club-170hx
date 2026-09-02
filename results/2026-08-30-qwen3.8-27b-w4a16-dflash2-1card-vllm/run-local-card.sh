#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=${1:?usage: run-local-card.sh GPU_INDEX OUTPUT_DIR}
OUTPUT_DIR=${2:?usage: run-local-card.sh GPU_INDEX OUTPUT_DIR}
PORT=$((18020 + GPU_INDEX))
API_KEY=<bench-api-key>
QWEN_REPO=<workdir>/repos/qwen-serving
MODEL=<model-storage>/models/qwen38/bench-2026-08-29/Qwen3.8-27B-W4A16-AutoRound-fast
DRAFT=<model-storage>/models/qwen38/bench-2026-08-29/Qwen3.8-27B-DFlash2-W4A16

if pgrep -af 'docker build.*dsv4|vllm serve|EngineCore' >/dev/null; then
  echo "Refusing to start while another build or inference process is active." >&2
  pgrep -af 'docker build.*dsv4|vllm serve|EngineCore' >&2 || true
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
STARTED_AT=$(date -Is)

{
  echo "started_at=$STARTED_AT"
  echo "gpu_index=$GPU_INDEX"
  echo "port=$PORT"
  echo "model=$MODEL"
  echo "draft=$DRAFT"
  git -C "$QWEN_REPO" log -1 --format='qwen_recipe_commit=%H'
  <fast-cache>/venvs/qwen38/bin/pip show vllm torch flashinfer-python flashinfer-cubin \
    | grep -E '^(Name|Version):'
  nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,uuid,pci.bus_id,name,driver_version,power.limit \
    --format=csv,noheader
} >"$OUTPUT_DIR/metadata.txt"

nvidia-smi -i "$GPU_INDEX" -q -d POWER,TEMPERATURE,PERFORMANCE \
  >"$OUTPUT_DIR/nvidia-before.txt"

export PATH=<fast-cache>/venvs/qwen38/bin:$PATH
export HF_HOME=<model-storage>/models/hf-cache
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_API_KEY=$API_KEY
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

cd "$QWEN_REPO"
SPEC=dflash2 CTX=fast MAX_SEQS=1 DFLASH_TOKENS=7 PORT=$PORT \
  GPU_UTIL=0.90 KV_MEM= VLLM_V2_CUDAGRAPH_MEM_MIB=1400 \
  MODEL=$MODEL DRAFT=$DRAFT \
  setsid bash single-user/start_qwen.sh >"$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill -- -"$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

HEALTHY=0
for _ in $(seq 1 150); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null; then
    HEALTHY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 10
done

if [[ $HEALTHY -ne 1 ]]; then
  echo "Server failed its health gate." >&2
  tail -n 100 "$OUTPUT_DIR/server.log" >&2
  exit 3
fi

sleep 20
"$(dirname "$0")/bench-usage.py" \
  --base-url "http://127.0.0.1:$PORT" \
  --api-key "$API_KEY" \
  --gpu-index "$GPU_INDEX" \
  --telemetry-output "$OUTPUT_DIR/telemetry.jsonl" \
  >"$OUTPUT_DIR/bench.jsonl"

nvidia-smi -i "$GPU_INDEX" -q -d POWER,TEMPERATURE,PERFORMANCE \
  >"$OUTPUT_DIR/nvidia-after.txt"
grep -iE 'SpecDecoding metrics|acceptance|DFlash2|lookup-augmented' \
  "$OUTPUT_DIR/server.log" >"$OUTPUT_DIR/specdec.log" || true
sudo -n journalctl -k --since "$STARTED_AT" --no-pager \
  | grep -Ei 'nvrm|xid|gpu has fallen off' \
  >"$OUTPUT_DIR/kernel-errors.log" || true

echo "completed_at=$(date -Is)" >>"$OUTPUT_DIR/metadata.txt"
