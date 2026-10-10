#!/usr/bin/env bash
# 1 Hz GPU telemetry to $1 (CSV). Kills every PID listed in /work/engine.pids when core >= 80 C,
# memory >= 85 C, or the GPU stops answering. Stop it with: kill $(cat /work/guard.pid)
out=${1:-/work/telemetry.csv}
echo $$ > /work/guard.pid
echo "ts,temp_gpu_c,temp_mem_c,power_w,power_limit_w,sm_mhz,mem_mhz,util_pct,mem_used_mib" > "$out"
trip() { echo "GUARD TRIP: $1" | tee -a /work/guard.log; for p in $(cat /work/engine.pids 2>/dev/null); do kill "$p" 2>/dev/null; done; }
while true; do
  l=$(nvidia-smi --query-gpu=temperature.gpu,temperature.memory,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
  if [ -z "$l" ]; then trip "nvidia-smi failed"; sleep 5; continue; fi
  l=$(echo "$l" | tr -d ' ')
  echo "$(date -u +%FT%TZ),$l" >> "$out"
  IFS=, read -r tg tm pw _ <<< "$l"
  [ "${tg%%.*}" -ge 80 ] 2>/dev/null && trip "core ${tg} C"
  [ "${tm%%.*}" -ge 85 ] 2>/dev/null && trip "memory ${tm} C"
  sleep 1
done
