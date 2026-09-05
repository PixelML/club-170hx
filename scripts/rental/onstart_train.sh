#!/usr/bin/env bash
# Vast.ai onstart script: CMP 170HX box rented for drafter TRAINING (+ optional PP4
# slice-C extraction), not inference. Companion to onstart.sh (the inference variant).
# Assumes the instance is created with --image ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905
# so this script runs INSIDE that container directly (no nested docker, no docker binary
# in the image — confirmed on the sibling inference image; per-card work below uses
# nohup + PID files, not `docker run`, translating the drafter lane's bundle 1:1).
set -euo pipefail

DATA=/data
TOOLS=/data/tools
OUT=/out
RECEIPTS=/workspace/receipts/preflight
mkdir -p "$DATA" "$OUT" "$RECEIPTS"

echo "[onstart] $(date -u +%FT%TZ) training/extraction rental"

# --- Preflight receipts ---
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current,driver_version --format=csv \
  | tee "$RECEIPTS/nvidia-smi-query.csv"
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "[onstart] detected $NUM_GPUS_DETECTED GPUs"

# --- Torch import sanity (this image has no pip install step; must already work) ---
python3 -c "import torch, numpy, safetensors; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  | tee "$RECEIPTS/torch-import.txt"

echo "[onstart] done. Data transfer happens from the orchestrating machine (transfer_specdec.sh)."
echo "[onstart] After transfer lands: verify_manifest.py /data/sliceB 400000 MUST print"
echo "[onstart]   'tap hc_post-materialized+stream-mean tokens 455367 shards 9 files 9' then 'OK'"
echo "[onstart] before any training run starts. Do not proceed if it differs."
