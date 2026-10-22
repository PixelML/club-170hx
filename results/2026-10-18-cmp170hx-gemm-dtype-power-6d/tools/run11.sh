#!/bin/bash
# Cap sweep used before and after the 6D flash (receipts/pre6d, receipts/post6d).
# Per card: wait for 0 compute apps, then each cap in CAPS, ~15-20 s of GEMM+copy (mm11.py) per cap,
# 83 C core / 90 C HBM kill, 20 s cool-down, original cap restored.
#   before the flash: CMP-A, CMP-B with CAPS="140 175 225 250"; CMP-F (factory 6D) with CAPS="140 175 225 250 300"
#   after the flash:  CMP-A, CMP-B with CAPS="140 175 225 250 300"
# Usage: CARDS="<idx>:<GPU-uuid>:cmp-a <idx>:<GPU-uuid>:cmp-b" CAPS="140 175 225 250 300" PY=python OUT=<dir> bash run11.sh
# The original run set the cap through a local wrapper around `nvidia-smi -pl`; the equivalent command is used here.
set -u
V=${PY:-python}; S=${OUT:-.}; D=$(cd "$(dirname "$0")" && pwd)
for spec in ${CARDS:?}; do
  IFS=: read -r I U N <<< "$spec"
  until [ "$(nvidia-smi -i $U --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; do sleep 30; done
  ORIG=$(nvidia-smi -i $U --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
  echo "[$(date '+%T')] $N free, orig cap $ORIG W"
  for cap in ${CAPS:?}; do
    sudo -n nvidia-smi -i $I -pl $cap >/dev/null
    nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,temperature.memory --format=csv,noheader,nounits -i $I -lms 250 > $S/smi11_${N}_$cap.csv & SM=$!
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U timeout 120 $V -I $D/mm11.py ${N}_cap$cap > $S/res11_${N}_$cap.jsonl & PY=$!
    while kill -0 $PY 2>/dev/null; do
      T=$(nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader,nounits -i $I | tr -d ' ')
      c=${T%,*}; m=${T#*,}; [[ $m =~ ^[0-9]+$ ]] || m=0
      if [ "$c" -ge 83 ] || [ "$m" -ge 90 ]; then echo "ABORT $N cap$cap core=$c hbm=$m"; kill $PY; break; fi
      sleep 0.5
    done; wait $PY; echo "[$(date '+%T')] $N cap$cap rc=$?"; kill $SM
    sleep 20   # let it cool between steps
  done
  sudo -n nvidia-smi -i $I -pl $ORIG >/dev/null
done
echo "[$(date '+%T')] sweep done"
