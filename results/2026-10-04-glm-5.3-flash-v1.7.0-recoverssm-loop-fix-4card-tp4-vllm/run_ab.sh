#!/bin/bash
# The A/B arms of this notebook, in one script. Run from the recipe checkout (Morrowmake/glm53-flash-cmp170hx-recipe @ v1.7.0)
# with .env = LAYOUT=tp4 P2P=auto GLM5_NCCL_P2P_SYS=0. Each arm restarts the server, waits for /health, then runs glm_repro.py twice.
# The recorded run used three launches of an earlier version of this script (receipts/repro/glm_ab.log).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$HERE/receipts/repro}; mkdir -p "$OUT"
FIXED=ghcr.io/pixelml/club-170hx@sha256:f9fc947bb4f1fdacd5e146948cb64cdca01da0942809989d32a6241886defaf0
up() { until curl -sf -m 5 localhost:8030/health >/dev/null; do sleep 10; done; }
arm() {  # arm NAME ENV...
  local name=$1; shift
  env "$@" ./start.sh restart > "$OUT/$name-start.log" 2>&1; up
  for r in 1 2; do python3 "$HERE/glm_repro.py" > "$OUT/ab-$name-r$r.out" 2>&1; echo "$(date -u +%T) $name r$r $(grep BROKEN "$OUT/ab-$name-r$r.out")"; done
}
arm flags0-dflash   VLLM_CUSTOM_ALLREDUCE_FLAGS=0
arm default-dflash  VLLM_CUSTOM_ALLREDUCE_FLAGS=1
arm recover0-dflash VLLM_CUSTOM_ALLREDUCE_FLAGS=1 VLLM_GLM5_KDA_RECOVER=0
arm default-none    VLLM_CUSTOM_ALLREDUCE_FLAGS=1 SPEC_MODE=none BOOT_CHECK=0   # boot check needs spec-decode counters
arm fixed-dflash    VLLM_CUSTOM_ALLREDUCE_FLAGS=1 IMAGE=$FIXED
