#!/bin/bash
# Full dtype table on one card (CMP-C, vBIOS 67) at 225 W then 140 W -> res9_<cap>.jsonl + smi9_<cap>.csv.
# Usage: IDX=<nvidia-smi index> GPU_UUID=<GPU-...> PY=python OUT=<dir> bash run9.sh
# The original run set the cap through a local wrapper around `nvidia-smi -pl`; the equivalent command is used here.
set -u
I=${IDX:?nvidia-smi index}; U=${GPU_UUID:?GPU UUID}; V=${PY:-python}; S=${OUT:-.}; D=$(cd "$(dirname "$0")" && pwd)
trap 'sudo -n nvidia-smi -i $I -pl 140 >/dev/null' EXIT
for cap in 225 140; do
  sudo -n nvidia-smi -i $I -pl $cap >/dev/null
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,memory.used --format=csv,noheader,nounits -i $I -lms 250 > $S/smi9_$cap.csv & SM=$!
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$U timeout 900 $V -I $D/mm9.py cap$cap > $S/res9_$cap.jsonl; echo "cap$cap rc=$?"
  kill $SM
done
