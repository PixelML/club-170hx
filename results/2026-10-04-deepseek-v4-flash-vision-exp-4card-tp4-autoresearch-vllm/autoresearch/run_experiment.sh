#!/bin/bash
# FIXED harness (agent must not edit). usage: run_experiment.sh  (from <repo>)
# Boots the server with ./vllm mounted over the image's /vllm/vllm, ./serve.env and ./serve.args,
# runs quality gates and speed bench, prints "key: value" lines, stops the server.
set -u
REPO=<repo>; B=<bench>
IMG=ghcr.io/pixelml/club-170hx@sha256:b26232f8f041c988d3285e2278c9f5001cc49f96131bb0b22a9d38b5e5e061cd
OUT=$REPO/runs/$(date -u +%Y%m%dT%H%M%S)-$(cd $REPO && git rev-parse --short HEAD); mkdir -p $OUT
URL=http://127.0.0.1:18098
docker rm -f dsv4v >/dev/null 2>&1
ENVS=(); while IFS= read -r l; do [[ -z "$l" || "$l" == \#* ]] && continue; ENVS+=(-e "$l"); done < $REPO/serve.env
read -r -a ARGS < <(tr '\n' ' ' < $REPO/serve.args)
DM0=$(dmesg | grep -ci xid)
T0=$(date +%s)
docker run -d --name dsv4v --device nvidia.com/gpu=all --ipc=host --ulimit memlock=-1 --shm-size 16g -p 127.0.0.1:18098:8000 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=600 \
  "${ENVS[@]}" -v /path/to/DeepSeek-V4-Flash-Vision-Exp:/model:ro -v <cache>:/root/.cache \
  -v $REPO/vllm:/vllm/vllm \
  $IMG vllm serve /model --served-model-name dsv4v "${ARGS[@]}" > /dev/null
ready=0
while [ $(( $(date +%s) - T0 )) -lt 1500 ]; do
  curl -sf -m 5 $URL/v1/models >/dev/null && { ready=1; break; }
  docker ps -q -f name=dsv4v | grep -q . || break
  sleep 10
done
echo "boot_s: $(( $(date +%s) - T0 ))"
if [ $ready = 0 ]; then
  docker logs dsv4v > $OUT/container.log 2>&1; docker rm -f dsv4v >/dev/null 2>&1
  echo "status: crash_boot"; echo "gates_passed: 0/7"; echo "c1_tok_s: 0"; echo "out: $OUT"; exit 0
fi
cd $B; export DSV4_OUT=$OUT DSV4_URL=$URL DSV4_MODEL_NAME=dsv4v
g=0
python3 bench_harness.py gate > $OUT/gate.out 2>&1 && g=$((g+1))
python3 correct.py $OUT/correct.json > $OUT/correct.out 2>&1; grep -q "^5/5 correct" $OUT/correct.out && g=$((g+1))
python3 probe.py $OUT/probe.json > $OUT/probe.out 2>&1; [ "$(grep -c ' ON ' $OUT/probe.out)" = 5 ] && g=$((g+1))
python3 vision_check.py $OUT/vision.json > $OUT/vision.out 2>&1; grep -q '"keyword_match": true' $OUT/vision.out && g=$((g+1))
timeout 300 python3 bench_harness.py prefill > $OUT/prefill.out 2>&1
timeout 400 python3 bench_harness.py ladder 1,4 > $OUT/ladder.out 2>&1
python3 - $OUT > $OUT/metrics.txt <<'PY'
import json, sys, statistics
o = sys.argv[1]
def load(n):
    try: return json.load(open(f"{o}/{n}"))
    except Exception: return None
pf = load("prefill.json"); lad = load("ladder.json")
pts = [r.get("prefill_tok_s") for r in (pf or {}).get("reps", []) if isinstance(r, dict) and r.get("prefill_tok_s")]
print("prefill_tok_s:", round(statistics.median(pts), 1) if pts else 0)
c1 = c4 = 0; c4ok = 0
for lv in (lad or {}).get("levels", []):
    s = lv.get("summary") or {}
    if lv.get("concurrency") == 1: c1 = s.get("aggregate_tok_s_median", 0)
    if lv.get("concurrency") == 4:
        c4 = s.get("aggregate_tok_s_median", 0); c4ok = 1 if lv.get("status") == "PASS" else 0
print("c1_tok_s:", c1); print("c4_agg_tok_s:", c4); print("c4_pass:", c4ok)
PY
cat $OUT/metrics.txt
grep -q "^c4_pass: 1" $OUT/metrics.txt && g=$((g+1))
docker ps -q -f name=dsv4v | grep -q . && g=$((g+1))
[ "$(dmesg | grep -ci xid)" = "$DM0" ] && g=$((g+1))
echo "gates_passed: $g/7"
echo "status: $([ $g = 7 ] && echo ok || echo gate_fail)"
docker logs dsv4v > $OUT/container.log 2>&1
docker stop -t 30 dsv4v >/dev/null 2>&1; docker rm dsv4v >/dev/null 2>&1
echo "out: $OUT"
