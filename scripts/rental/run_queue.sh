#!/usr/bin/env bash
# Full measurement queue for the GLM-5.3-Flash rental. Idempotent/resumable:
# each step writes a DONE marker under its receipts dir and is skipped if
# already present (delete the marker to force a re-run of that step).
#
# NUM_GPUS controls what the box actually has (4, 6, or 8) and gates which
# steps run: TP4/PP4-family steps need >=4; TP8 and the 2x-TP4-replica c=16
# cell need >=8 and are skipped (not failed) below that.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/bench"
RECEIPTS_ROOT="${RECEIPTS_ROOT:-/workspace/receipts}"
NUM_GPUS="${NUM_GPUS:-8}"
PY="${PY:-python3}"
STEP_TIMEOUT_S="${STEP_TIMEOUT_S:-1800}"   # 30 min hang guard per bench call
MM_FLAG_TESTED="${MM_FLAG_TESTED:-0}"      # set to 1 once step1 records the mm-profiling result

log() { printf '[run_queue] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }

done_marker() { echo "$1/DONE"; }
skip_if_done() {
  local dir="$1"
  if [[ -f "$(done_marker "$dir")" ]]; then
    log "skip (already done): $dir"
    return 0
  fi
  return 1
}
mark_done() { mkdir -p "$1"; touch "$(done_marker "$1")"; }

wait_ready() {
  local port="$1" timeout="${2:-900}"
  local waited=0
  while ! curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    sleep 5; waited=$((waited + 5))
    if [[ "$waited" -ge "$timeout" ]]; then
      log "ERROR: server on port $port not ready after ${timeout}s"
      return 1
    fi
  done
  log "server on port $port ready after ${waited}s"
}

run_bench() {
  # run_bench <out_dir> <base_url> <model> -- <script args...>
  local out="$1" url="$2" model="$3"; shift 3; shift # drop the literal --
  timeout "$STEP_TIMEOUT_S" "$PY" "$@" --base-url "$url" --model "$model" --out "$out"
}

# --- Step 0: preflight (superset of onstart's; safe to re-run) ---
step_preflight() {
  local d="$RECEIPTS_ROOT/00-preflight"
  skip_if_done "$d" && return 0
  mkdir -p "$d"
  nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,driver_version --format=csv | tee "$d/nvidia-smi-query.csv"
  nvidia-smi topo -m | tee "$d/topo-m.txt"
  nvidia-smi topo -p2p r 2>&1 | tee "$d/topo-p2p-r.txt"
  df -h /workspace | tee "$d/disk.txt"
  # simple local write bandwidth sample (not network; network bw is measured
  # by onstart's checkpoint download time)
  dd if=/dev/zero of="$d/.bwtest" bs=1M count=512 oflag=direct 2>&1 | tail -3 | tee "$d/disk-write-bw.txt"
  rm -f "$d/.bwtest"
  echo "$NUM_GPUS" > "$d/num_gpus.txt"
  mark_done "$d"
}

# --- Step 1: TP4 recipe of record (cards 0-3), no mm-limit flag first ---
step_tp4_record() {
  local d="$RECEIPTS_ROOT/01-tp4-record"
  skip_if_done "$d" && return 0
  mkdir -p "$d"
  local t0=$(date +%s)
  NAME=glm53-tp4 PORT=18098 K=3 DEVICES=0,1,2,3 EXTRA_MM_FLAG='' bash "$HERE/launch_tp4.sh" 2>&1 | tee "$d/launch.log"
  if wait_ready 18098 900; then
    echo "mm_limit_flag_needed=false" > "$d/mm-flag-result.txt"
  else
    log "TP4 record: startup failed without mm-limit flag, retrying WITH it"
    docker rm -f glm53-tp4 >/dev/null 2>&1 || true
    NAME=glm53-tp4 PORT=18098 K=3 DEVICES=0,1,2,3 \
      EXTRA_MM_FLAG='--limit-mm-per-prompt {"image": 0, "video": 0}' \
      bash "$HERE/launch_tp4.sh" 2>&1 | tee -a "$d/launch.log"
    wait_ready 18098 900 || { log "ERROR: TP4 record failed to boot even with mm-limit flag"; docker logs glm53-tp4 2>&1 | tail -100 > "$d/boot-failure.log"; return 1; }
    echo "mm_limit_flag_needed=true" > "$d/mm-flag-result.txt"
  fi
  local t1=$(date +%s)
  echo "boot_time_s=$((t1 - t0))" > "$d/boot-time.txt"
  run_bench "$d" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/bench_glm53.py" -- gate prefill ttft decode_c1
  run_bench "$d" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/c8_stability.py" --concurrency 8 --rounds 3
  run_bench "$d" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/context_sweep.py" --lengths 4096 16384 65536 --reps 3
  run_bench "$d" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/lossless_check.py"
  mark_done "$d"
}

# --- Step 2: TP4 variants (one factor each): P2P path, k=2, k=5 ---
step_tp4_variants() {
  local d="$RECEIPTS_ROOT/02-tp4-variants"
  skip_if_done "$d" && return 0
  mkdir -p "$d"
  local p2p_available=0
  if grep -qi 'OK' "$RECEIPTS_ROOT/00-preflight/topo-p2p-r.txt" 2>/dev/null; then p2p_available=1; fi

  if [[ "$p2p_available" == "1" ]]; then
    local sub="$d/p2p-custom-ar"
    docker rm -f glm53-tp4 >/dev/null 2>&1 || true
    NAME=glm53-tp4 PORT=18098 K=3 DEVICES=0,1,2,3 NO_CUSTOM_AR=0 bash "$HERE/launch_tp4.sh"
    if wait_ready 18098 900; then
      run_bench "$sub" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/bench_glm53.py" -- gate decode_c1
      run_bench "$sub" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/c8_stability.py" --concurrency 8 --rounds 1
    else
      log "P2P variant failed to boot"; mkdir -p "$sub"; docker logs glm53-tp4 2>&1 | tail -100 > "$sub/boot-failure.log"
    fi
  else
    log "P2P not available on this box; skipping custom-all-reduce variant"
  fi

  for k in 2 5; do
    local sub="$d/k$k"
    docker rm -f glm53-tp4 >/dev/null 2>&1 || true
    NAME=glm53-tp4 PORT=18098 K=$k DEVICES=0,1,2,3 bash "$HERE/launch_tp4.sh"
    if wait_ready 18098 900; then
      run_bench "$sub" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/bench_glm53.py" -- gate decode_c1
      run_bench "$sub" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/c8_stability.py" --concurrency 8 --rounds 1
    else
      log "k=$k variant failed to boot"; mkdir -p "$sub"; docker logs glm53-tp4 2>&1 | tail -100 > "$sub/boot-failure.log"
    fi
  done
  mark_done "$d"
}

# --- Step 3: PP4 + MTP x5 verdict ---
step_pp4() {
  local d="$RECEIPTS_ROOT/03-pp4"
  skip_if_done "$d" && return 0
  mkdir -p "$d"
  docker rm -f glm53-tp4 glm53-pp4 >/dev/null 2>&1 || true
  DEVICES=0,1,2,3 bash "$HERE/launch_pp4.sh" 2>&1 | tee "$d/launch.log"
  if wait_ready 18099 900; then
    run_bench "$d" http://127.0.0.1:18099 glm-5.3-flash -- "$BENCH/bench_glm53.py" -- gate decode_c1
    docker logs glm53-pp4 2>&1 | grep -Ei 'accept|draft' | tail -50 > "$d/acceptance-log-grep.txt" || true
  else
    log "PP4 failed to boot"
    docker logs glm53-pp4 2>&1 | tail -150 > "$d/boot-failure.log"
  fi
  mark_done "$d"
}

# --- Step 4: TP8 + 2xTP4 replicas for c=16 (needs 8 cards) ---
step_tp8() {
  local d="$RECEIPTS_ROOT/04-tp8"
  if [[ "$NUM_GPUS" -lt 8 ]]; then
    log "NUM_GPUS=$NUM_GPUS < 8, skipping TP8 step"
    mkdir -p "$d"; echo "skipped: NUM_GPUS=$NUM_GPUS < 8" > "$d/SKIPPED"
    return 0
  fi
  skip_if_done "$d" && return 0
  mkdir -p "$d"
  docker rm -f glm53-pp4 glm53-tp4 glm53-tp8 >/dev/null 2>&1 || true
  NAME=glm53-tp8 PORT=18098 K=3 DEVICES=0,1,2,3,4,5,6,7 bash "$HERE/launch_tp8.sh" 2>&1 | tee "$d/launch.log"
  if wait_ready 18098 900; then
    run_bench "$d" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/bench_glm53.py" -- gate decode_c1
    run_bench "$d" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/c8_stability.py" --concurrency 8 --rounds 1
  else
    log "TP8 failed to boot"; docker logs glm53-tp8 2>&1 | tail -150 > "$d/boot-failure.log"
    mark_done "$d"; return 0
  fi
  docker rm -f glm53-tp8 >/dev/null 2>&1 || true

  # 2x TP4 replicas (0-3, 4-7) for aggregate c=16
  NAME=glm53-tp4a PORT=18098 K=3 DEVICES=0,1,2,3 bash "$HERE/launch_tp4.sh" 2>&1 | tee -a "$d/launch.log"
  NAME=glm53-tp4b PORT=18100 K=3 DEVICES=4,5,6,7 bash "$HERE/launch_tp4.sh" 2>&1 | tee -a "$d/launch.log"
  if wait_ready 18098 900 && wait_ready 18100 900; then
    run_bench "$d/replica-a" http://127.0.0.1:18098 glm-5.3-flash -- "$BENCH/c8_stability.py" --concurrency 8 --rounds 1 &
    run_bench "$d/replica-b" http://127.0.0.1:18100 glm-5.3-flash -- "$BENCH/c8_stability.py" --concurrency 8 --rounds 1 &
    wait
  else
    log "2x TP4 replica boot failed"
  fi
  docker rm -f glm53-tp4a glm53-tp4b >/dev/null 2>&1 || true
  mark_done "$d"
}

# --- Step 5: collect ---
step_collect() {
  bash "$HERE/collect.sh"
}

main() {
  log "starting queue, NUM_GPUS=$NUM_GPUS"
  step_preflight
  step_tp4_record
  step_tp4_variants
  step_pp4
  step_tp8
  step_collect
  log "queue complete"
}

main "$@"
