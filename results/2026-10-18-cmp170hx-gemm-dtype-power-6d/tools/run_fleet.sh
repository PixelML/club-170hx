#!/bin/bash
# Fleet check after the 6D flash: one card at a time at its normal 170 W cap, ~20 s of tests (mm11.py) each,
# 30 s cool-down, 83 C core / 90 C HBM kill.
# Usage: CARDS="<idx>:cmp-a <idx>:cmp-b ..." PY=python OUT=<dir> bash run_fleet.sh
set -u
V=${PY:-python}; P=${OUT:-.}; D=$(cd "$(dirname "$0")" && pwd)
for spec in ${CARDS:?}; do
  IFS=: read -r I N <<< "$spec"; U=$(nvidia-smi -i $I --query-gpu=uuid --format=csv,noheader)
  [ "$(nvidia-smi -i $I --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "$N busy, skipped"; continue; }
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,pcie.link.width.current --format=csv,noheader,nounits -i $I -lms 250 > $P/smi_$N.csv & SM=$!
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U timeout 120 $V -I $D/mm11.py $N > $P/res_$N.jsonl & PY=$!
  while kill -0 $PY 2>/dev/null; do
    T=$(nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader,nounits -i $I | tr -d ' '); c=${T%,*}; m=${T#*,}; [[ $m =~ ^[0-9]+$ ]] || m=0
    if [ "$c" -ge 83 ] || [ "$m" -ge 90 ]; then echo "ABORT $N core=$c hbm=$m"; kill $PY; break; fi; sleep 0.5
  done; wait $PY; echo "[$(date +%T)] $N rc=$?"; kill $SM; sleep 30
done
