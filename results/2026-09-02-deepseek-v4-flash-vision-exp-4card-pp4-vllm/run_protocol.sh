#!/usr/bin/env bash
OUT="$1"; cd "$OUT"; export DSV4_OUT="$OUT" DSV4_URL=http://127.0.0.1:18098
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/logs/protocol.log"; }
start=$(date +%s)
until curl -sf -m 5 http://127.0.0.1:18098/v1/models >/dev/null; do
  if ! docker ps -q -f name=<container> | grep -q .; then log "FAIL container exited"; docker logs <container> 2>&1 | tail -60 > "$OUT/logs/container-tail-at-failure.log"; exit 2; fi
  if [ $(( $(date +%s) - start )) -gt 4500 ]; then log "FAIL readiness timeout 75 min"; exit 3; fi
  sleep 15
done
date -u +%FT%TZ > receipts/ready-utc.txt; log "READY after $(( $(date +%s) - start )) s"
nvidia-smi --query-gpu=index,power.draw,temperature.gpu,memory.used --format=csv > receipts/nvidia-smi-loaded.csv
python3 bench_harness.py gate 2>&1 | tee -a logs/protocol.log || { log "FAIL gate"; exit 4; }
python3 bench_harness.py prefill 2>&1 | tee -a logs/protocol.log
python3 bench_harness.py ttft 2>&1 | tee -a logs/protocol.log
python3 bench_harness.py ladder 1,2,4,8,16 2>&1 | tee -a logs/protocol.log
nvidia-smi --query-gpu=index,power.draw,temperature.gpu,memory.used --format=csv > receipts/nvidia-smi-after-bench.csv
sudo dmesg | grep -iE 'xid|ecc' | grep -v systemd > receipts/dmesg-xid-ecc.txt; log "xid/ecc lines: $(wc -l < receipts/dmesg-xid-ecc.txt)"
docker logs <container> > logs/container.log 2>&1
docker stop -t 60 <container> >/dev/null; docker rm <container> >/dev/null; sleep 5
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv > receipts/nvidia-smi-final.csv; cat receipts/nvidia-smi-final.csv | tee -a logs/protocol.log
log "DONE"
