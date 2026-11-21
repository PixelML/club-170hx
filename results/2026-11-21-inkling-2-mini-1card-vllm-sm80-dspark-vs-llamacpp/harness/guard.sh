#!/bin/bash
# guard.sh <gpu-index-list> <container> <logfile>
# Logs GPU telemetry every 2 s; stops the container
# at core >= 80 C, HBM >= 85 C, or a new NVIDIA Xid in the kernel log.
GPUS=$1; C=$2; L=$3
T0=$(date "+%F %T")
echo "ts,idx,core_c,mem_c,power_w,sm_mhz,mem_mhz,util,mem_used_mib" >> "$L"
while true; do
  while IFS=, read -r idx core mem pw sm mm ut mu; do
    echo "$(date +%s),$idx,$core,$mem,$pw,$sm,$mm,$ut,$mu" >> "$L"
    core=${core// /}; mem=${mem// /}
    if [ "${core%.*}" -ge 80 ] 2>/dev/null || [ "${mem%.*}" -ge 85 ] 2>/dev/null; then
      echo "STOP thermal gpu=$idx core=$core mem=$mem" >> "$L"; docker stop -t 5 "$C"; exit 2
    fi
  done < <(nvidia-smi -i "$GPUS" --query-gpu=index,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader,nounits)
  x=$(journalctl -k --since "$T0" | grep -c "Xid" || true)
  if [ "$x" != 0 ]; then echo "STOP xid count=$x" >> "$L"; journalctl -k --since "$T0" | grep Xid | tail -3 >> "$L"; docker stop -t 5 "$C"; exit 3; fi
  docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q true || { echo "END container not running" >> "$L"; exit 0; }
  sleep 2
done
