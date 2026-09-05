#!/usr/bin/env bash
# Direct-exec translation of the drafter lane's "vast bundle" (seanphan/pixelml#108) six-run
# sweep. The bundle's commands use `docker run --gpus device=N ... --entrypoint python3`
# because it assumes a host with a docker daemon; this rental's --image IS the container
# (no nested docker, no docker binary inside it — same finding as the inference lane), so
# every run below is CUDA_VISIBLE_DEVICES + nohup + PID file instead.
#
# Must run AFTER transfer_specdec.sh has verified:
#   tap hc_post-materialized+stream-mean tokens 455367 shards 9 files 9
#   OK
# Do not run this script if that gate did not pass verbatim.
set -euo pipefail

DATA=/data
TOOLS=/data/tools
OUT=/out
LOGDIR=/workspace/logs/training
PIDDIR=/workspace/pids/training
IMAGE_PY=python3   # running inside the rented image directly; no docker run

mkdir -p "$LOGDIR" "$PIDDIR" "$OUT"

NUM_GPUS_TRAIN="${1:-6}"   # cards reserved for training (rest go to extraction if any)

run() {   # run(name, gpu, block_size, lr, [init_from])
  local name="$1" gpu="$2" bs="$3" lr="$4" init="${5:-}"
  mkdir -p "$OUT/$name"
  if [[ -f "$PIDDIR/$name.pid" ]] && kill -0 "$(cat "$PIDDIR/$name.pid")" 2>/dev/null; then
    echo "[run] $name already running (pid $(cat "$PIDDIR/$name.pid")), skipping"
    return 0
  fi
  local extra=()
  [[ -n "$init" ]] && extra=(--init-from "$init")
  echo "[run] starting $name on gpu $gpu: block_size=$bs lr=$lr init=${init:-none}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$TOOLS" \
    nohup $IMAGE_PY "$TOOLS/train_drafter.py" \
      --data "$DATA/sliceB" --shared "$DATA/target-shared.safetensors" \
      --out "$OUT/$name" --block-size "$bs" --lr "$lr" \
      --steps 8000 --eval-every 500 --eval-blocks 400 "${extra[@]}" \
      > "$LOGDIR/$name.log" 2>&1 &
  echo $! > "$PIDDIR/$name.pid"
}

echo "[run_training] reference band check first (must land IN BAND ~36%):"
PYTHONPATH="$TOOLS" $IMAGE_PY "$TOOLS/ref_eval2.py" \
  --weights "$DATA/ref-drafter/model.safetensors" \
  --shared "$DATA/target-shared.safetensors" --data "$DATA/sliceB" \
  --blocks 800 --stride 16 --label reference-check \
  | tee "$LOGDIR/reference-check.log"
grep -q "IN BAND" "$LOGDIR/reference-check.log" || {
  echo "[run_training] ABORT: reference band check did not land IN BAND (~36%). Data/transfer is wrong."
  exit 1
}

# Per the drafter lane (2026-09-05): against a 6h cap, run all six FROM SCRATCH in
# parallel (one ~2.5h wave) rather than chaining bs13/bs17 off bs8 checkpoints via
# --init-from (which would force two sequential ~2.4h waves and risk the cap). If
# extraction finishes early and hours remain, re-run 13/17 with --init-from then.
run bs8-lr15   0  8  1.5e-4
run bs8-lr30   1  8  3e-4

if [[ "$NUM_GPUS_TRAIN" -ge 6 ]]; then
  run bs13-lr15  2 13 1.5e-4
  run bs13-lr30  3 13 3e-4
  run bs17-lr15  4 17 1.5e-4
  run bs17-lr30  5 17 3e-4
else
  echo "[run_training] NUM_GPUS_TRAIN=$NUM_GPUS_TRAIN < 6: bs8 pair only, skipping bs13/bs17"
fi

echo "[run_training] all training runs launched. Logs: $LOGDIR/*.log"
echo "[run_training] Expected first lines per run (abort that run if missing):"
echo "  [data] 8 train shards, 757 samples, 1 held out"
echo "  [shared] froze embed_tokens + lm_head from .../target-shared.safetensors"
echo "  [model] block_size=N D=... trainable=1.1711B (81 tensors) == exported set"
echo "  [frozen] embed_tokens + lm_head bit-identical after 20 steps"
echo "Poll: tail -f $LOGDIR/<run>.log ; watch nvidia-smi"
