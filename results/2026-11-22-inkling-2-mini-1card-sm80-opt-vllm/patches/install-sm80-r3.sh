#!/usr/bin/env bash
# install-sm80-r3.sh [DEST_DIR] [BASE_INKLING2_DIR]
#
# Builds the Round-5 Best CMP 170HX (sm_80) Inkling-2-Mini package from source (zero .so blobs in git):
#   1. Starts from vllm/models/inkling2 inside vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
#      (or from an existing PixelML/sm80vllm / vllm checkout >= PR #69028).
#   2. Applies the baseline SM80 capability + Triton PDL edits across nvidia/ops/*.py.
#   3. Overlays the Round-1..5 Python modules + sm80_fast_v6.cu (universal 0-ULP 140W Humming v5 buckets,
#      ported PR #69168 low-M Tensor Core GEMMs, 49,152-vocab DSpark pruning, zero-scatter
#      DraftLogitsCache, monolithic M<=21 FULL CUDA graphs, 18-split SM80 decode attention,
#      slots-safe Sconv metadata deduplication, FA2 vision encoder, and 13-group coalesced
#      unit-scale SM80 E4M3 FP8 KV cache across Target + DSpark).
#   4. Compiles libsm80_fast_v6.so inside the container (or with local nvcc) via:
#        nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC sm80_fast_v6.cu -o libsm80_fast_v6.so
#      Note: always replaces DEST_DIR via rm -rf + mv so no running process sees a truncated mmap.
set -euo pipefail

IMG="${VLLM_IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}"
CONTAINER_PKG="/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEST_RAW="${1:-}"
BASE_SRC="${2:-}"

if [ -z "$DEST_RAW" ]; then
  if [ -d "$CONTAINER_PKG" ]; then
    DEST_RAW="$CONTAINER_PKG"
  else
    DEST_RAW="/work/patches/inkling2"
  fi
fi

# Accept a vllm repo root or site-packages root as DEST_RAW as well
if [ -d "$DEST_RAW/vllm/models/inkling2" ]; then
  DEST_RAW="$DEST_RAW/vllm/models/inkling2"
elif [ -d "$DEST_RAW/models/inkling2" ]; then
  DEST_RAW="$DEST_RAW/models/inkling2"
fi

mkdir -p "$(dirname "$DEST_RAW")"
DEST_PARENT="$(cd "$(dirname "$DEST_RAW")" && pwd)"
DEST_NAME="$(basename "$DEST_RAW")"
DEST="$DEST_PARENT/$DEST_NAME"

# Host-mode delegation: if we are outside the container (no base package locally) and docker is available,
# run this installer inside the pinned vllm/vllm-openai container so nvcc + upstream files match 100%.
if [ -z "$BASE_SRC" ] && [ ! -d "$CONTAINER_PKG" ] && [ ! -f "$DEST/__init__.py" ]; then
  if command -v docker >/dev/null 2>&1; then
    echo "[install-sm80-r3] Building $DEST inside container $IMG ..."
    docker run --rm --entrypoint bash \
      -v "$SCRIPT_DIR:/src:ro" \
      -v "$DEST_PARENT:/out" \
      "$IMG" -c "bash /src/install-sm80-r3.sh /out/$DEST_NAME"
    exit 0
  else
    echo "ERROR: Neither $CONTAINER_PKG nor docker found; pass BASE_INKLING2_DIR explicitly." >&2
    exit 1
  fi
fi

if [ -z "$BASE_SRC" ]; then
  if [ -d "$CONTAINER_PKG" ]; then
    BASE_SRC="$CONTAINER_PKG"
  else
    BASE_SRC="$DEST"
  fi
fi

STAGE="$(mktemp -d "${DEST_PARENT}/.inkling2-r3-stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

cp -a "$BASE_SRC/." "$STAGE/"
find "$STAGE" \( -name "__pycache__" -o -name "*.pyc" -o -name "*.so" \) -exec rm -rf {} +

# 1. Baseline SM80 compatibility gates (capability 90 -> 80, strip SM90+ Triton PDL, sm80_moe_align hook)
sed -i 's/^\(\s*\)return 90$/\1return 80  # PixelML sm80 patch/' "$STAGE/quant_config.py"
while IFS= read -r -d '' pyf; do
  sed -i -E \
    -e 's/tl\.extra\.cuda\.gdc_wait\(\)/pass  # sm80: no PDL/g' \
    -e 's/tl\.extra\.cuda\.gdc_launch_dependents\(\)/pass  # sm80: no PDL/g' \
    -e 's/launch_pdl=True/launch_pdl=False/g' \
    "$pyf"
done < <(grep -rlZ -E 'gdc_wait|gdc_launch_dependents|launch_pdl=True' "$STAGE" --include='*.py' || true)

sed -i 's/from vllm.model_executor.layers.fused_moe.moe_align_block_size import (/from vllm.models.inkling2.sm80_moe_align import (  # sm80 patch/' \
  "$STAGE/nvidia/ops/moe_humming.py"

# 2. Overlay Round-6 Python modules + sm80_fast_v6.cu + sm80_r6_ext.cu
cp -a "$SCRIPT_DIR/overlay/." "$STAGE/"
cp -f "$SCRIPT_DIR/sm80_fast_v6.cu" "$STAGE/sm80_fast_v6.cu"
if [ -f "$SCRIPT_DIR/sm80_r6_ext.cu" ]; then
  cp -f "$SCRIPT_DIR/sm80_r6_ext.cu" "$STAGE/sm80_r6_ext.cu"
fi
find "$STAGE" \( -name "__pycache__" -o -name "*.pyc" \) -exec rm -rf {} +

# 3. Compile standalone libsm80_fast_v6.so + libsm80_r6_ext.so for sm_80 (container lacks <cusparse.h>, so compile directly via nvcc + ctypes)
NVCC_BIN="${NVCC:-}"
if [ -z "$NVCC_BIN" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC_BIN="/usr/local/cuda/bin/nvcc"
  fi
fi

if [ "${SKIP_NVCC:-0}" != "1" ]; then
  if [ -n "$NVCC_BIN" ]; then
    echo "[install-sm80-r6] Compiling libsm80_fast_v6.so + libsm80_r6_ext.so with $NVCC_BIN (-O3 -arch=sm_80)..."
    "$NVCC_BIN" -O3 -arch=sm_80 --shared -Xcompiler -fPIC \
      "$STAGE/sm80_fast_v6.cu" -o "$STAGE/libsm80_fast_v6.so"
    if [ -f "$STAGE/sm80_r6_ext.cu" ]; then
      "$NVCC_BIN" -O3 -arch=sm_80 --shared -Xcompiler -fPIC \
        "$STAGE/sm80_r6_ext.cu" -o "$STAGE/libsm80_r6_ext.so"
    fi
  elif command -v docker >/dev/null 2>&1; then
    echo "[install-sm80-r6] Compiling libsm80_fast_v6.so + libsm80_r6_ext.so inside $IMG ..."
    docker run --rm --entrypoint bash \
      -v "$STAGE:/stage" \
      "$IMG" -c "/usr/local/cuda/bin/nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC /stage/sm80_fast_v6.cu -o /stage/libsm80_fast_v6.so && if [ -f /stage/sm80_r6_ext.cu ]; then /usr/local/cuda/bin/nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC /stage/sm80_r6_ext.cu -o /stage/libsm80_r6_ext.so; fi"
  else
    echo "WARNING: nvcc not found; sm80_fast_v6.cu / sm80_r6_ext.cu staged without .so (compile inside container before serving)." >&2
  fi
fi

# 4. Atomic directory replacement (never overwrite mmap'd .so in place)
rm -rf "$DEST"
mv "$STAGE" "$DEST"
trap - EXIT
echo "[install-sm80-r3] Ready at $DEST"
