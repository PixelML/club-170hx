#!/bin/bash
# Subset table on one factory-6D card (CMP-F) at 140 W then 300 W, 83 C core / 90 C HBM kill -> res10_<cap>.jsonl + smi10_<cap>.csv.
# Usage: IDX=<nvidia-smi index> GPU_UUID=<GPU-...> PY=python OUT=<dir> bash run10.sh
# The original run set the cap through a local wrapper around `nvidia-smi -pl`; the equivalent command is used here.
set -u
I=${IDX:?nvidia-smi index}; U=${GPU_UUID:?GPU UUID}; V=${PY:-python}; S=${OUT:-.}; D=$(cd "$(dirname "$0")" && pwd)
restore() { sudo -n nvidia-smi -i $I -pl 140 >/dev/null; }
trap restore EXIT
for cap in 140 300; do
  sudo -n nvidia-smi -i $I -pl $cap >/dev/null
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,temperature.memory --format=csv,noheader,nounits -i $I -lms 250 > $S/smi10_$cap.csv & SM=$!
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U timeout 300 $V -I $D/mm10.py cap$cap > $S/res10_$cap.jsonl & PY=$!
  while kill -0 $PY 2>/dev/null; do
    T=$(nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader,nounits -i $I | tr -d ' ')
    c=${T%,*}; m=${T#*,}; [[ $m =~ ^[0-9]+$ ]] || m=0
    if [ "$c" -ge 83 ] || [ "$m" -ge 90 ]; then echo "ABORT cap$cap core=$c hbm=$m"; kill $PY; break; fi
    sleep 0.5
  done; wait $PY; echo "cap$cap rc=$?"; kill $SM
done
