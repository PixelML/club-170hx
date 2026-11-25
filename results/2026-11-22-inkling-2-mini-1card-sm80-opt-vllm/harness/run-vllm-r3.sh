#!/usr/bin/env bash
# run-vllm-r3.sh [gpu=0] [port=18000] [tag=r3] [extra vllm serve args...]  — run on the GPU host
#
# Pinned image: vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
# Pinned model: thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7
set -euo pipefail

G="${1:-0}"
P="${2:-18000}"
TAG="${3:-r3}"
if [ "$#" -ge 3 ]; then shift 3; else shift "$#"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMG="${VLLM_IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}"
WORK_DIR="${WORK_DIR:-/work}"
PATCH_DST="${PATCH_DST:-$WORK_DIR/patches/inkling2}"

if [ -n "${MODEL_DIR:-}" ]; then
  MDIR="$MODEL_DIR"
elif [ -d "/models/inkling2-mini/Inkling-2-Mini" ]; then
  MDIR="/models/inkling2-mini/Inkling-2-Mini"
else
  MDIR="$WORK_DIR/models/Inkling-2-Mini"
fi

if [ ! -f "$MDIR/config.json" ]; then
  echo "ERROR: Model checkpoint not found at $MDIR" >&2
  echo "Download first: hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir $MDIR" >&2
  exit 1
fi

# Build SM80 Round-8 Best package + libsm80_fast_v6.so + libsm80_r6_ext.so inside the pinned container if not yet staged
if [ ! -f "$PATCH_DST/libsm80_fast_v6.so" ] || [ ! -f "$PATCH_DST/libsm80_r6_ext.so" ] || [ ! -f "$PATCH_DST/sm80_rejection.py" ] || [ ! -f "$PATCH_DST/humming_include/humming/scheduler.cuh" ]; then
  bash "$EXP_DIR/patches/install-sm80-r8.sh" "$PATCH_DST"
fi

mkdir -p "$WORK_DIR/cache"/{hf,vllm,triton,inductor,torch_compile,humming,flashinfer}

CONTAINER_PKG="/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2"
CUDAGRAPH_CFG='{"custom_ops":["none"],"cudagraph_capture_sizes":[1,2,3,4,5,6,8,9,10,12,15,16,18,20,21,24,27,30,32,36,42,48,54,60,64,72,84,96,108,120,132,144,168,192,216,240,264,288,312,336,384,432,480,512]}'
SPEC_CFG='{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic","rejection_sample_method":"block","num_speculative_tokens_per_batch_size":[[1,2,4],[3,6,3],[7,256,2]]}'

docker rm -f "inkling-vllm-$TAG" >/dev/null 2>&1 || true
docker run -d --name "inkling-vllm-$TAG" \
  --gpus "device=$G" --network host --ipc host \
  -e HF_HOME=/work/cache/hf \
  -e VLLM_CACHE_ROOT=/work/cache/vllm \
  -e TRITON_CACHE_DIR=/work/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor \
  -e TORCH_COMPILE_CACHE_DIR=/work/cache/torch_compile \
  -e HUMMING_CACHE_DIR=/work/cache/humming \
  -e FLASHINFER_WORKSPACE_BASE=/work/cache/flashinfer \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  -v "$MDIR:/models/Inkling-2-Mini:ro" \
  -v "$PATCH_DST:$CONTAINER_PKG:ro" \
  -v "$WORK_DIR/cache:/work/cache" \
  -v "$WORK_DIR/cache/humming:/root/.humming" \
  "$IMG" \
  /models/Inkling-2-Mini \
  --served-model-name=inkling \
  --host=127.0.0.1 \
  --port="$P" \
  --max-model-len=262144 \
  --reasoning-parser=inkling \
  --enable-auto-tool-choice \
  --tool-call-parser=inkling \
  --limit-mm-per-prompt='{"image":0}' \
  --max-num-batched-tokens=4096 \
  --compilation-config="$CUDAGRAPH_CFG" \
  --speculative-config="$SPEC_CFG" \
  "$@"

echo "[run-vllm-r3] Started container inkling-vllm-$TAG on 127.0.0.1:$P (follow logs: docker logs -f inkling-vllm-$TAG)"
