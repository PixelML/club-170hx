#!/bin/bash
# bench-1017.sh ARM [extra vllm serve args] -- boot one arm, smoke it, bench it, write receipts.
# Runs INSIDE the bench container. One boot per arm. Receipts: /root/bench/runs/dsv41-1017-<ARM>/
# Layout matches club-170hx results/<exp>/receipts/<run>/ used by the Oct 4 notebooks
# (make_notebook_v3.py, 2026-10-04-glm-5.3-flash-...-p2p-vllm.ipynb):
#   decode-{structured,coding,prose}-c{1,8}.json/.stdout  (bench_decode.py, 400 tok, T=0, thinking off;
#        c1 median of 5, c8 median aggregate of 3; DSpark accept from /metrics)
#   prefill.json/.out (bench_long_1017.py prefill, 2k/8k/32k x2)   long.json (LONG=1: decode at 1k/8k/32k/54k ctx)
#   correctness.json (5 known answers)   qref.json (12 greedy x 64 tok, top-5 logprobs, for qcmp.py across arms)
#   nvidia-smi-start.csv, nvidia-smi-q.txt, topo.txt, telemetry.csv (2 s), launch.txt (docker cmd + env),
#   serve.log.gz, nccl-transport.txt (which NCCL transport each channel used), boot.txt (boot seconds, VRAM),
#   pins.json (image id + repo digests, model revisions, driver, power cap, arm settings)
# Options: SKIP_C8=1, LONG=1, RUNS1=5, RUNS8=3, MAXTOK=400, READY_MIN=90, KEEP=1 (leave server up after bench).
set -euo pipefail
ARM=${1:?usage: bench-1017.sh ARM [serve args]}; shift
D=/root/bench/runs/dsv41-1017-$ARM
B=/root/dsv41-tp4
IMG=${IMG:-dsv41-1017-exl3:sm80-v3p1}
mkdir -p "$D"; cd "$B"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$D/bench.log"; }

[ -e "$D/DONE" ] && [ "${FORCE:-}" != 1 ] && { log "$ARM already done ($D/DONE); FORCE=1 to redo"; exit 0; }
log "arm $ARM start"
nvidia-smi --query-gpu=index,pci.bus_id,serial,memory.total,power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,clocks.max.sm,clocks.mem --format=csv > "$D/nvidia-smi-start.csv"
nvidia-smi -q > "$D/nvidia-smi-q.txt"
{ nvidia-smi topo -m; for t in r w; do echo "-- topo -p2p $t"; nvidia-smi topo -p2p $t; done; } > "$D/topo.txt" 2>&1

T0=$(date +%s)
{ echo "# $(date '+%F %T %Z')  ARM=$ARM  args: $*"; ./run-1017.sh "$ARM" "$@" 2>&1; } > "$D/launch.txt"
docker inspect dsv41-1017 --format '{{json .Config.Env}}' >> "$D/launch.txt"

ready=0
for i in $(seq 1 $(( ${READY_MIN:-90} * 6 ))); do
  sleep 10
  if ! docker ps --format '{{.Names}}' | grep -qx dsv41-1017; then log "container exited during boot"; break; fi
  if curl -sf -m 5 http://127.0.0.1:8040/v1/models | grep -q deepseek-v4.1-flash; then ready=1; break; fi
done
BOOT=$(( $(date +%s) - T0 ))
docker logs dsv41-1017 > "$D/serve.log" 2>&1 || true
grep -E 'NCCL INFO (Channel|Connected|comm .* rank|P2P|SHM|NVLS|Using network)|via (P2P|SHM|NET)|all-reduce|custom.?all.?reduce|CustomAllreduce' "$D/serve.log" | head -400 > "$D/nccl-transport.txt" || true
{ echo "boot_seconds=$BOOT ready=$ready"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv; free -g; } > "$D/boot.txt"
if [ "$ready" != 1 ]; then
  gzip -f "$D/serve.log"; log "FAIL: not ready after ${BOOT}s; see serve.log.gz (tail below)"; zcat "$D/serve.log.gz" | tail -40; exit 1
fi
log "ready after ${BOOT}s"

# pins
python3 - "$D" "$ARM" "$IMG" <<'EOF'
import json, os, subprocess, sys, glob
d, arm, img = sys.argv[1:4]
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
revs = {}
for m in glob.glob("/models/dsv41-1017/**/.cache/huggingface/download/*.metadata", recursive=True)[:400]:
    try: revs.setdefault(m.split("/models/dsv41-1017/")[1].split("/.cache")[0], set()).add(open(m).readline().strip())
    except Exception: pass
pins = {
  "arm": arm,
  "image": img,
  "image_id": sh(f"docker image inspect {img} --format '{{{{.Id}}}}'"),
  "image_repo_digests": sh(f"docker image inspect {img} --format '{{{{json .RepoDigests}}}}'"),
  "container_env": json.loads(sh("docker inspect dsv41-1017 --format '{{json .Config.Env}}'") or "[]"),
  "launch_sh_sha256": sh("sha256sum /root/dsv41-tp4/launch-1017.sh | cut -c1-64"),
  "model_revisions_from_hf_metadata": {k: sorted(v) for k, v in revs.items()},
  "model_REVISIONS_file": sh("cat /models/dsv41-1017/REVISIONS 2>/dev/null"),
  "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
  "kernel": sh("uname -r"),
  "power_limit_w": sh("nvidia-smi --query-gpu=power.limit --format=csv,noheader | xargs"),
  "registry_dwords": sh("grep -i RegistryDwords /proc/driver/nvidia/params 2>/dev/null"),
  "host": "Proxmox host (2x Xeon E5-2686 v4, 251 GiB), privileged LXC bench container, host driver 615.71.09 cmpunlocker + P2P patches",
}
json.dump(pins, open(f"{d}/pins.json", "w"), indent=1); print("pins ok")
EOF

nohup nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,utilization.gpu,clocks.sm,clocks.mem --format=csv -l 2 > "$D/telemetry.csv" 2>&1 & TEL=$!
trap 'kill $TEL 2>/dev/null || true' EXIT

log "smoke: correct.py"
python3 correct.py "$D/correctness.json" > "$D/correctness.out" 2>&1 || true
tail -1 "$D/correctness.out" | tee -a "$D/bench.log"
grep -q '^5/5' "$D/correctness.out" || { log "SMOKE FAIL (not 5/5): stopping this arm; server left up for a look"; exit 1; }
grep -E 'ifiable|nan' -i "$D/correctness.json" >/dev/null && log "WARNING: suspicious tokens in correctness.json" || true

for c in 1 8; do
  [ "$c" = 8 ] && [ "${SKIP_C8:-}" = 1 ] && continue
  for p in structured coding prose; do
    flag=""; [ $p = structured ] && flag=--structured; [ $p = coding ] && flag=--coding
    runs=$([ $c = 1 ] && echo ${RUNS1:-5} || echo ${RUNS8:-3})
    log "decode $p c$c (runs $runs)"
    python3 bench_decode.py --phase $p-c$c $flag --concurrency $c --max-tokens ${MAXTOK:-400} --runs $runs \
      --out "$D/decode-$p-c$c.json" > "$D/decode-$p-c$c.stdout" 2>&1 || log "decode $p c$c FAILED"
    tail -2 "$D/decode-$p-c$c.stdout" | tee -a "$D/bench.log"
  done
done
log "prefill ${PREFILL_TARGETS:-2000,8000,32000}"
python3 bench_long_1017.py prefill "$D/prefill.json" > "$D/prefill.out" 2>&1 || log "prefill FAILED"
cat "$D/prefill.out" | tee -a "$D/bench.log"
if [ "${LONG:-}" = 1 ]; then log "long decode"; python3 bench_long_1017.py decode "$D/long.json" > "$D/long.out" 2>&1 || true; fi
log "qref"
python3 qref.py "$D/qref.json" > "$D/qref.out" 2>&1 || log "qref FAILED"

docker logs dsv41-1017 > "$D/serve.log" 2>&1 || true
gzip -f "$D/serve.log"
date '+%F %T' > "$D/DONE"
log "arm $ARM done -> $D"
[ "${KEEP:-}" = 1 ] || { docker rm -f dsv41-1017 >/dev/null; sleep 10; }
