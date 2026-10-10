#!/bin/bash
# stop the engine if the core reaches 80 C or the memory 85 C (repository safety rule); log the event
while true; do
  read -r c m < <(nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader | tr -d ' ' | tr ',' ' ')
  if [ "${c:-0}" -ge 80 ] || [ "${m:-0}" -ge 85 ]; then
    P=$(pidof fnx_spec); [ -n "$P" ] && kill $P && echo "$(date +%T) thermal stop core=$c mem=$m" >> "${1:-/tmp/thermal.log}"
  fi
  sleep 1
done
