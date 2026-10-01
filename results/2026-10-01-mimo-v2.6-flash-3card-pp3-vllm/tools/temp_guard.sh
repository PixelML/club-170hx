#!/bin/bash
# Stop the named container if any GPU core reaches 80 C or a new Xid appears; log telemetry every 5 s.
# usage: temp_guard.sh <container> <telemetry.csv>
C=$1; LOG=$2
base=$(dmesg | grep -c Xid)
while docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q true; do
  line=$(nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits | tr '\n' ';')
  echo "$(date +%s);$line" >> "$LOG"
  max=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "$max" -ge 80 ] || [ "$(dmesg | grep -c Xid)" != "$base" ]; then
    echo "$(date +%s) SAFETY STOP max_core=$max xid=$(dmesg | grep -c Xid)" >> "$LOG"
    docker stop "$C"; exit 1
  fi
  sleep 5
done
