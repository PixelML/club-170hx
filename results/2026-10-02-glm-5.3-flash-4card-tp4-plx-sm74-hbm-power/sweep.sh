#!/usr/bin/env bash
# Power-cap sweep on a running server: one decode matrix per cap, cap set live, highest first.
# Run on the GPU host (needs nvidia-smi with permission to set the power limit) from a folder that
# holds bench_decode.py (MiaAI-Lab @ 943912cd, BASE pointed at the server) and run_decode.sh.
# Restores the cap given in RESTORE_W at the end.
set -u
CAPS=${CAPS:-"150 140 130 120 110 100"}
RESTORE_W=${RESTORE_W:-140}
for w in $CAPS; do
  nvidia-smi -pl "$w" >/dev/null
  D=runs/cap-${w}w; mkdir -p "$D/receipts"; cp bench_decode.py run_decode.sh "$D/"
  ( cd "$D"
    nvidia-smi --query-gpu=timestamp,index,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,utilization.gpu \
      --format=csv -l 1 > receipts/telemetry.csv 2>&1 & T=$!
    bash run_decode.sh > receipts/run_decode.out 2>&1
    kill $T )
done
nvidia-smi -pl "$RESTORE_W" >/dev/null
