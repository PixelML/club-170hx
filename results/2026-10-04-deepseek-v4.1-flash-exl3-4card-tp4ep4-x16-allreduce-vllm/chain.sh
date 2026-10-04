#!/bin/bash
# DeepSeek-V4.1-Flash, three all-reduce arms on the published image (int4 Engram in pinned RAM, DSpark k=5).
# Run from this folder. Set MODELS (the folder with the EXL3 checkpoint, the Engram shards and the pack) and INT4 (the int4 tables).
# Bench scripts: results/2026-10-03-deepseek-v4.1-flash-exl3-4card-tp4ep4-dspark-vllm/{bench_long,bench_decode,correct}.py
# The recorded run used two launches of this loop (the NCCL arm first failed to start: the previous container was still being removed).
set -u
B=$(cd "$(dirname "$0")" && pwd); OLD=$B/../2026-10-03-deepseek-v4.1-flash-exl3-4card-tp4ep4-dspark-vllm
MODELS=${MODELS:?set MODELS}; INT4=${INT4:?set INT4}; CACHE=${CACHE:-$HOME/.cache/dsv41}
IMG=ghcr.io/pixelml/club-170hx@sha256:b8b7a1699bf272855a9b8c8b316f796b3ef917696302baf2a62c856707d69932
E=/models/DeepSeek-V4.1-Flash-engram
log() { echo "$(date -u +%T) $*" | tee -a $B/chain.log; }
watchdog() { while sleep 5; do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1); [ "$t" -ge 80 ] && { log "WATCHDOG ${t}C stop"; docker stop dsv41; return; }; done; }
arm() {  # arm NAME ENV-FLAGS...
  local n=$1; shift; mkdir -p $B/receipts/$n
  docker rm -f dsv41 >/dev/null 2>&1; sleep 10
  docker run -d --name dsv41 --gpus all --ulimit memlock=-1 --shm-size 16g -p 8040:8040 \
    -e DSV41_ENGRAM_DISK=0 -e DSV41_ENGRAM_INT4=1 "$@" \
    -v $MODELS:/models:ro -v $CACHE:/root/.cache \
    -v $INT4/model-00047-of-00048.safetensors:$E/model-00047-of-00048.safetensors:ro \
    -v $INT4/model-00048-of-00048.safetensors:$E/model-00048-of-00048.safetensors:ro \
    $IMG --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
  until curl -sf -m 5 localhost:8040/health >/dev/null; do sleep 15; done
  cd $B/receipts/$n
  nvidia-smi --query-gpu=index,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv > nvidia-smi-start.csv
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,utilization.gpu --format=csv -l 2 > telemetry.csv 2>&1 & TEL=$!
  watchdog & WD=$!
  python3 $OLD/bench_long.py prefill prefill.json > prefill.out 2>&1
  for c in 1 4; do
    python3 $OLD/bench_decode.py --phase structured-c$c --structured --concurrency $c --max-tokens 400 --out decode-structured-c$c.json > decode-structured-c$c.stdout 2>&1
    python3 $OLD/bench_decode.py --phase coding-c$c --coding --concurrency $c --max-tokens 400 --out decode-coding-c$c.json > decode-coding-c$c.stdout 2>&1
    python3 $OLD/bench_decode.py --phase prose-c$c --concurrency $c --max-tokens 400 --out decode-prose-c$c.json > decode-prose-c$c.stdout 2>&1
  done
  python3 $OLD/correct.py correctness.json > correctness.out 2>&1
  kill $TEL $WD 2>/dev/null; cd $B; log "$n done: $(tail -1 receipts/$n/correctness.out)"
}
arm int4-hybrid-x16                                                  # image defaults: custom all-reduce below 128 KiB, NCCL above
arm int4-nccl-x16   -e CUSTOM_AR=0 -e VLLM_FORCE_CUSTOM_AR_PCIE=0    # NCCL only
arm int4-custom-x16 -e VLLM_CUSTOM_AR_MAX_BYTES=0                    # custom all-reduce for every size
docker rm -f dsv41 >/dev/null 2>&1
