#!/usr/bin/env bash
# Vast.ai onstart script: 8x (or NUM_GPUS) CMP 170HX box for GLM-5.3-Flash vLLM sm80.
# Assumes the instance is created with --image ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903
# so this script runs INSIDE that container as PID 1's onstart hook (no nested docker).
# Installs nothing the image doesn't already have. Idempotent: safe to re-run.
set -euo pipefail

IMAGE="ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903"
CKPT_REPO="wtdcode/GLM-5.3-Flash-AWQ-W4A16"
CKPT_REVISION="abd7b07719111f137e1de8a0c1b7e01c11b74d1a"
CKPT_BYTES_EXPECTED=190843146533
MODEL_DIR="/workspace/models/glm53"
RECEIPTS="/workspace/receipts/preflight"

mkdir -p "$RECEIPTS" "$MODEL_DIR"

echo "[onstart] $(date -u +%FT%TZ) image=$IMAGE"

# --- Preflight receipts (no GPU state changes beyond power cap below) ---
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,driver_version --format=csv \
  | tee "$RECEIPTS/nvidia-smi-query.csv"
nvidia-smi topo -m | tee "$RECEIPTS/topo-m.txt"
nvidia-smi topo -p2p r | tee "$RECEIPTS/topo-p2p-r.txt" || echo "[onstart] topo -p2p r not supported" | tee -a "$RECEIPTS/topo-p2p-r.txt"
nvcc --version 2>/dev/null | tee "$RECEIPTS/nvcc-version.txt" || true

# --- Power cap: 180 W per card, best-effort (some Vast hosts block this) ---
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "[onstart] detected $NUM_GPUS_DETECTED GPUs"
for i in $(seq 0 $((NUM_GPUS_DETECTED - 1))); do
  nvidia-smi -i "$i" -pl 180 2>&1 | tee -a "$RECEIPTS/power-cap.log" || \
    echo "[onstart] WARNING: could not set power cap on GPU $i (host may not permit it)" | tee -a "$RECEIPTS/power-cap.log"
done

# --- Image is already running as the container; verify tag/digest in-place ---
{
  echo "image_ref=$IMAGE"
  python3 -c "import json;print(json.dumps({'python':__import__('sys').version}))" 2>/dev/null || true
  vllm --version 2>&1 | tail -3
  hf --version 2>&1
} | tee "$RECEIPTS/image-versions.txt"

# --- Checkpoint download with shard-size gate ---
echo "[onstart] downloading $CKPT_REPO@$CKPT_REVISION to $MODEL_DIR"
t0=$(date +%s)
hf download "$CKPT_REPO" --revision "$CKPT_REVISION" --local-dir "$MODEL_DIR" --max-workers 16 \
  | tee "$RECEIPTS/hf-download.log"
t1=$(date +%s)

actual_bytes=$(find "$MODEL_DIR" -type f -exec stat -c '%s' {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
echo "[onstart] checkpoint download took $((t1 - t0))s, ${actual_bytes} bytes (expected ${CKPT_BYTES_EXPECTED})"
if [[ "$actual_bytes" -lt "$((CKPT_BYTES_EXPECTED - 1048576))" ]]; then
  echo "[onstart] ABORT: checkpoint incomplete ($actual_bytes < $CKPT_BYTES_EXPECTED - 1MiB slack)" | tee -a "$RECEIPTS/hf-download.log"
  exit 1
fi
echo "{\"expected_bytes\": $CKPT_BYTES_EXPECTED, \"actual_bytes\": $actual_bytes, \"download_s\": $((t1 - t0))}" \
  > "$RECEIPTS/checkpoint-verify.json"

echo "[onstart] done. Next: run_queue.sh"
