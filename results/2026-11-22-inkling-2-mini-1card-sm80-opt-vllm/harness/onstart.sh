#!/usr/bin/env bash
# onstart.sh — One-command Vast.ai / cloud GPU rental reproduction for Inkling-2-Mini SM80 Round-4 Best
#
# Works in TWO modes automatically:
#   Mode A (Vast.ai / RunPod container running vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e):
#     Installs the SM80 Round-4 Best Python overlay + compiles libsm80_fast_v5.so in-place with nvcc,
#     downloads thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7,
#     persists JIT caches under ${WORK_DIR:-/workspace}/cache, starts vllm serve, and optionally
#     runs the verification gates + benchmark when RUN_BENCH=1 (set KV_CACHE_DTYPE=fp8 for 3.07x 262K KV mode).
#   Mode B (Bare-metal / VM Docker host):
#     Pulls the pinned vllm/vllm-openai digest, downloads the checkpoint, and launches harness/run-vllm-r3.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_REPO="thinkingmachines/Inkling-2-Mini"
MODEL_REV="ed40e7da625f594fc9c4990f5f66be24873c22d7"
IMG="vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e"

WORK_DIR="${WORK_DIR:-/workspace}"
if [ ! -d "$WORK_DIR" ]; then
  WORK_DIR="/work"
fi
mkdir -p "$WORK_DIR/models" "$WORK_DIR/cache"/{hf,vllm,triton,inductor,torch_compile,humming} "$WORK_DIR/out"

MODEL_DIR="${MODEL_DIR:-$WORK_DIR/models/Inkling-2-Mini}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
GPU_IDX="${GPU_IDX:-0}"
POWER_LIMIT_W="${POWER_LIMIT_W:-140}"

# 1. Optional power cap (succeeds on bare metal / privileged VM; skipped cleanly in unprivileged rental containers)
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -i "$GPU_IDX" -pl "$POWER_LIMIT_W" >/dev/null 2>&1 || true
  nvidia-smi -i "$GPU_IDX" --query-gpu=name,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv
fi

# 2. Download pinned checkpoint if not already present
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[onstart] Downloading $MODEL_REPO @ $MODEL_REV -> $MODEL_DIR ..."
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
  else
    python3 -m pip install -q "huggingface_hub>=0.25"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', revision='$MODEL_REV', local_dir='$MODEL_DIR')"
  fi
fi

CONTAINER_PKG="/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2"
CUDAGRAPH_CFG='{"custom_ops":["none"],"cudagraph_capture_sizes":[1,2,3,4,5,6,8,9,10,12,15,16,18,20,21,24,27,30,32,36,42,48,54,60,64,72,84,96,108,120,132,144,168,192,216,240,264,288,312,336,384,432,480,512]}'
SPEC_CFG='{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","num_speculative_tokens_per_batch_size":[[1,2,4],[3,6,3],[7,256,2]]}'

KV_EXTRA=()
if [ -n "${KV_CACHE_DTYPE:-}" ]; then
  KV_EXTRA=(--kv-cache-dtype="${KV_CACHE_DTYPE}")
fi

if [ -d "$CONTAINER_PKG" ]; then
  # Mode A: Inside the vllm/vllm-openai container (e.g. Vast.ai / RunPod)
  echo "[onstart] Detected container environment; installing SM80 Round-4 Best stack + compiling libsm80_fast_v5.so..."
  bash "$EXP_DIR/patches/install-sm80-r3.sh" "$CONTAINER_PKG"

  if [ ! -L /root/.humming ]; then
    cp -rn /root/.humming/* "$WORK_DIR/cache/humming/" 2>/dev/null || true
    rm -rf /root/.humming
    ln -s "$WORK_DIR/cache/humming" /root/.humming
  fi

  export CUDA_VISIBLE_DEVICES="$GPU_IDX"
  export HF_HOME="$WORK_DIR/cache/hf"
  export VLLM_CACHE_ROOT="$WORK_DIR/cache/vllm"
  export TRITON_CACHE_DIR="$WORK_DIR/cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/cache/inductor"
  export TORCH_COMPILE_CACHE_DIR="$WORK_DIR/cache/torch_compile"
  export HUMMING_CACHE_DIR="$WORK_DIR/cache/humming"
  export TORCHINDUCTOR_COMPILE_THREADS=1

  vllm serve "$MODEL_DIR" \
    --served-model-name=inkling \
    --host="$HOST" \
    --port="$PORT" \
    --max-model-len=262144 \
    --reasoning-parser=inkling \
    --enable-auto-tool-choice \
    --tool-call-parser=inkling \
    --limit-mm-per-prompt='{"image":0}' \
    --max-num-batched-tokens=4096 \
    --compilation-config="$CUDAGRAPH_CFG" \
    --speculative-config="$SPEC_CFG" \
    "${KV_EXTRA[@]}" \
    > "$WORK_DIR/out/vllm-r4.log" 2>&1 &
  SRV_PID=$!
else
  # Mode B: Host with Docker
  echo "[onstart] Host mode detected; pulling $IMG and launching via run-vllm-r3.sh..."
  docker pull "$IMG"
  WORK_DIR="$WORK_DIR" MODEL_DIR="$MODEL_DIR" bash "$SCRIPT_DIR/run-vllm-r3.sh" "$GPU_IDX" "$PORT" r4 "${KV_EXTRA[@]}"
  SRV_PID=""
fi

echo "[onstart] Waiting for http://127.0.0.1:$PORT/health ..."
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[onstart] Server healthy at http://127.0.0.1:$PORT"
    break
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null

if [ "${RUN_BENCH:-0}" = "1" ]; then
  echo "[onstart] Running 5 capability gates + 2x167.8K soak + PR #90 benchmark..."
  python3 "$SCRIPT_DIR/run_gates.py" --url "http://127.0.0.1:$PORT" --out "$WORK_DIR/out/gates-summary.json"
  python3 "$SCRIPT_DIR/bench.py" --url "http://127.0.0.1:$PORT" --engine r3best-dynk-block \
    --out "$WORK_DIR/out/pr90-r3best.jsonl" --suites c1,conc,prefill --reps 3 --n 768 --temps 0,1 --effort none \
    --conc 16 --prefill 4000,32000
fi

if [ -n "$SRV_PID" ] && [ "${FOREGROUND:-1}" = "1" ]; then
  wait "$SRV_PID"
fi
