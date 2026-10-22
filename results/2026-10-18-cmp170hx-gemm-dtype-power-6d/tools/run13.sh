#!/bin/bash
# 5 W cap sweep, continuous load (mm13.py sets each cap itself), 100 ms sampler, external 85 C core / 92 C HBM watchdog.
#   cap_sweep_5w: CMP-B (still on 67) 100-250 W, CMP-F (6D) 100-300 W; post6d: CMP-B (after the flash) 100-300 W
# Usage: CARDS="<idx>:<GPU-uuid>:cmp-b:250 <idx>:<GPU-uuid>:cmp-f:300" RESTORE=140 PY=python OUT=<dir> bash run13.sh
set -u
V=${PY:-python}; S=${OUT:-.}; D=$(cd "$(dirname "$0")" && pwd)
for spec in ${CARDS:?}; do
  IFS=: read -r I U N HI <<< "$spec"
  [ "$(nvidia-smi -i $U --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "$N busy, skip"; continue; }
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,temperature.memory --format=csv,noheader,nounits -i $I -lms 100 > $S/smi13_$N.csv & SM=$!
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U timeout 900 $V -I $D/mm13.py $I 100 $HI $N > $S/res13_$N.jsonl & PY=$!
  while kill -0 $PY 2>/dev/null; do   # external watchdog in case the in-process check lags
    T=$(nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader,nounits -i $I | tr -d ' '); c=${T%,*}; m=${T#*,}; [[ $m =~ ^[0-9]+$ ]] || m=0
    if [ "$c" -ge 85 ] || [ "$m" -ge 92 ]; then echo "WATCHDOG $N core=$c hbm=$m"; kill $PY; break; fi; sleep 1
  done; wait $PY; echo "[$(date +%T)] $N rc=$?"; kill $SM
  sudo -n nvidia-smi -i $I -pl ${RESTORE:-140} >/dev/null; nvidia-smi -i $I --query-gpu=power.limit --format=csv,noheader
  sleep 30
done
