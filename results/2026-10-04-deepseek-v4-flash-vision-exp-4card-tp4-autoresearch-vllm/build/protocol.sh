#!/bin/bash
# usage: protocol.sh RUN_NAME  -- waits for the server on :18098, then runs the full bench into runs/RUN_NAME
H=<bench>; OUT=$H/runs/$1; mkdir -p $OUT/logs; cd $H
export DSV4_OUT=$OUT DSV4_URL=http://127.0.0.1:18098 DSV4_MODEL_NAME=dsv4v
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/logs/protocol.log; }
start=$(date +%s)
until curl -sf -m 5 $DSV4_URL/v1/models >/dev/null; do
  docker ps -q -f name=dsv4v | grep -q . || { log "FAIL container exited"; docker logs dsv4v > $OUT/logs/container.log 2>&1; exit 2; }
  [ $(( $(date +%s) - start )) -gt 4500 ] && { log "FAIL readiness timeout"; exit 3; }
  sleep 15
done
log "READY after $(( $(date +%s) - start )) s"
docker inspect dsv4v --format "{{json .Config.Env}} {{json .Config.Cmd}}" > $OUT/serve-config.json
nvidia-smi --query-gpu=index,power.draw,temperature.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current --format=csv > $OUT/nvidia-smi-loaded.csv
nohup nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,utilization.gpu --format=csv -l 2 > $OUT/telemetry.csv 2>&1 & TEL=$!
python3 bench_harness.py gate 2>&1 | tee -a $OUT/logs/protocol.log || log "gate FAILED"
python3 correct.py $OUT/correctness.json 2>&1 | tee -a $OUT/logs/protocol.log
python3 probe.py $OUT/probe.json 2>&1 | tee -a $OUT/logs/protocol.log
python3 vision_check.py $OUT/vision-check.json 2>&1 | tee -a $OUT/logs/protocol.log
python3 bench_harness.py prefill 2>&1 | tee -a $OUT/logs/protocol.log
python3 bench_harness.py ttft 2>&1 | tee -a $OUT/logs/protocol.log
python3 bench_harness.py ladder 1,2,4,8 2>&1 | tee -a $OUT/logs/protocol.log
python3 bench_harness.py ladder_image 1,2,4 2>&1 | tee -a $OUT/logs/protocol.log
kill $TEL
dmesg 2>/dev/null | grep -iE "xid|nvrm.*error" | tail -20 > $OUT/dmesg-xid.txt
docker logs dsv4v > $OUT/logs/container.log 2>&1
log "DONE"
