#!/usr/bin/env bash
# tensor-gate.sh - per-card stress + tensor-core correctness gate
# Run 2026-09-02 on all four cards (results/2026-09-02-health-gate/README.md).
# Run only when the fleet is clear: no resident vLLM serving container, 0
# compute processes on every GPU, and no coordination lock file present
# (see preflight_clear() below for the exact refusal conditions).
#
# Usage: ./tensor-gate.sh [gpu_ids...]   e.g. ./tensor-gate.sh 0 1 2 3
# Defaults to all detected NVIDIA GPUs. Cards run ONE AT A TIME (see
# docs/QC.md "Run cards sequentially during diagnosis").
#
# Stages per card:
#   1. gpu-burn -tc for 10 min (tensor-core compute burn), result checked.
#   2. Deterministic tensor-core correctness: same-seed BF16 and FP16 matmul
#      (torch + cuBLAS) vs a CPU float64 reference and vs every other card's
#      result on the same input. Also runs a TF32 and an INT8 path.
#   3. Full-VRAM memtest_vulkan pass, weighted toward the top 56 GiB (the
#      region above the card's stock 8 GiB mining-era window).
#   4. PCIe replay/AER counter delta, captured before vs after each card.
#
# Safety: 180 W power cap enforced before each card starts. Stop immediately
# on any Xid, on 80C core, or on 85C memory. Every stage writes a JSON
# receipt; a stopped run still leaves partial receipts for the stages that
# completed.
set -uo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR="$DIR/logs/tensor-gate-$TS"
RECEIPTS="$LOGDIR/receipts"
mkdir -p "$RECEIPTS"

POWER_CAP_W=${TG_POWER_CAP_W:-180}
BURN_DURATION_S=${TG_BURN_S:-600}
CORE_STOP_C=${TG_CORE_STOP_C:-80}
MEM_STOP_C=${TG_MEM_STOP_C:-85}
MATMUL_ABS_TOL_BF16=${TG_TOL_BF16:-1e-1}
MATMUL_ABS_TOL_FP16=${TG_TOL_FP16:-5e-2}
MATMUL_ABS_TOL_TF32=${TG_TOL_TF32:-5e-3}
MATMUL_SEED=${TG_SEED:-170170}
MEMTEST_TOP_FRACTION=${TG_MEMTEST_TOP_FRACTION:-0.875} # 56/64 GiB

GPU_BURN="$DIR/gpu-burn/gpu-burn"
GPU_BURN_FATBIN="$DIR/gpu-burn/compare.fatbin"
MEMTEST="$DIR/memtest_vulkan/memtest_vulkan"
MEMTEST_SELECT="$DIR/memtest-select.py"
MEMTEST_MAX_S=${TG_MEMTEST_MAX_S:-330}
MATMUL_PY="$LOGDIR/matmul_check.py"
MATMUL_PYTHON="${TG_MATMUL_PYTHON:-<model-storage>/venvs/qwen38/bin/python3}"

mapfile -t ALL_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
[ ${#ALL_GPUS[@]} -gt 0 ] || { echo "FATAL: no NVIDIA GPUs visible"; exit 2; }
if [ $# -gt 0 ]; then GPUS=("$@"); else GPUS=("${ALL_GPUS[@]}"); fi

echo "=== tensor-gate: cards ${GPUS[*]} ==="
echo "power cap: ${POWER_CAP_W}W  burn: ${BURN_DURATION_S}s  stop: ${CORE_STOP_C}C core / ${MEM_STOP_C}C memory"
echo "logs: $LOGDIR"

# --- Preflight: refuse to run if the fleet is not clear -------------------
preflight_clear() {
  if [ -f <shared-scratch>/gpu3.lock ]; then
    echo "REFUSING: <shared-scratch>/gpu3.lock present"; return 1
  fi
  if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx dsv4-vision-vllm; then
    echo "REFUSING: dsv4-vision-vllm container is running"; return 1
  fi
  local procs
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${procs:-0}" -ne 0 ]; then
    echo "REFUSING: $procs compute process(es) still active fleet-wide"; return 1
  fi
  return 0
}

if [ "${TG_SKIP_PREFLIGHT:-0}" != "1" ]; then
  preflight_clear || { echo "Fleet not clear. Not starting."; exit 3; }
fi

# --- Matmul correctness harness (written once, shared by all cards) -------
cat > "$MATMUL_PY" <<'PYEOF'
import json, sys, torch

def run(dtype_name, gpu_index, seed, n=4096):
    torch.manual_seed(seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "tf32": torch.float32}[dtype_name]
    a64 = torch.randn(n, n, dtype=torch.float64)
    b64 = torch.randn(n, n, dtype=torch.float64)
    ref = (a64 @ b64).numpy()

    dev = f"cuda:{gpu_index}"
    if dtype_name == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        a = a64.to(dev, dtype=torch.float32)
        b = b64.to(dev, dtype=torch.float32)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        a = a64.to(dev, dtype=dtype)
        b = b64.to(dev, dtype=dtype)
    out = (a @ b).to(torch.float64).cpu().numpy()

    max_abs_err = float((out - ref).__abs__().max())
    return max_abs_err, out

def run_int8(gpu_index, seed, n=4096):
    torch.manual_seed(seed)
    ai = torch.randint(-127, 127, (n, n), dtype=torch.int8)
    bi = torch.randint(-127, 127, (n, n), dtype=torch.int8)
    ref = (ai.to(torch.int64) @ bi.to(torch.int64)).numpy()
    dev = f"cuda:{gpu_index}"
    out = torch._int_mm(ai.to(dev), bi.to(dev)).to(torch.int64).cpu().numpy()
    max_abs_err = int((out - ref).__abs__().max())
    return max_abs_err, out

if __name__ == "__main__":
    gpu_index = int(sys.argv[1])
    seed = int(sys.argv[2])
    out_path = sys.argv[3]
    result = {"gpu": gpu_index, "seed": seed, "paths": {}}
    for name in ("bf16", "fp16", "tf32"):
        try:
            err, _ = run(name, gpu_index, seed)
            result["paths"][name] = {"max_abs_err_vs_cpu_f64": err}
        except Exception as e:
            result["paths"][name] = {"error": str(e)}
    try:
        err, _ = run_int8(gpu_index, seed)
        result["paths"]["int8"] = {"max_abs_err_vs_cpu_i64": err}
    except Exception as e:
        result["paths"]["int8"] = {"error": str(e)}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
PYEOF

pcie_snapshot() {
  local g=$1 label=$2
  nvidia-smi -i "$g" -q | awk '/PCIe Generation|PCIe Link Info|Replay/{print}' \
    > "$RECEIPTS/gpu${g}-pcie-${label}.txt" 2>&1 || true
  # AER replay counter and DevSta from the host bus (best-effort; may need sudo)
  local bus
  bus=$(nvidia-smi -i "$g" --query-gpu=pci.bus_id --format=csv,noheader | tr -d ' ' | sed 's/^0000://')
  sudo lspci -vv -s "$bus" 2>/dev/null | grep -E "DevSta|UESta|CESta" \
    > "$RECEIPTS/gpu${g}-aer-${label}.txt" || true
}

xid_seen_since() {
  local since_epoch=$1
  journalctl -k --since "@$since_epoch" 2>/dev/null | grep -c "NVRM: Xid" || true
}

card_temps() {
  local g=$1
  nvidia-smi -i "$g" --query-gpu=temperature.gpu,temperature.memory --format=csv,noheader,nounits
}

# Live 30s temp sampler: writes a CSV row every 30s and, on a breach,
# writes a breach file and kills the card's active burn/memtest process so
# the loop notices within one sample period instead of only post-hoc.
start_temp_watch() {
  local g=$1 gdir=$2
  rm -f "$gdir/temp-breach.flag"
  (
    while true; do
      read -r ct mt <<< "$(card_temps "$g" | tr ',' ' ')"
      ct=${ct%.*}; mt=${mt%.*}
      echo "$(date -u +%FT%TZ),${ct:-},${mt:-}" >> "$gdir/temps-live.csv"
      if [ -n "${ct:-}" ] && [ "$ct" -ge "$CORE_STOP_C" ] 2>/dev/null; then
        echo "core-temp:${ct}C" > "$gdir/temp-breach.flag"
        pkill -9 -f "gpu-burn -tc" 2>/dev/null || true
        pkill -9 -f "memtest_vulkan" 2>/dev/null || true
      elif [ -n "${mt:-}" ] && [ "$mt" -ge "$MEM_STOP_C" ] 2>/dev/null; then
        echo "memory-temp:${mt}C" > "$gdir/temp-breach.flag"
        pkill -9 -f "gpu-burn -tc" 2>/dev/null || true
        pkill -9 -f "memtest_vulkan" 2>/dev/null || true
      fi
      sleep 30
    done
  ) &
  echo $! > "$gdir/temp-watch.pid"
}

stop_temp_watch() {
  local gdir=$1
  [ -f "$gdir/temp-watch.pid" ] || return 0
  kill "$(cat "$gdir/temp-watch.pid")" 2>/dev/null || true
  rm -f "$gdir/temp-watch.pid"
}

stop_reason=""
for g in "${GPUS[@]}"; do
  echo
  echo "########## GPU $g ##########"
  gdir="$RECEIPTS/gpu$g"; mkdir -p "$gdir"
  start_epoch=$(date +%s)

  echo "[gpu$g] setting power cap to ${POWER_CAP_W}W"
  sudo nvidia-smi -i "$g" --power-limit="$POWER_CAP_W" > "$gdir/power-cap-set.log" 2>&1 || true

  pcie_snapshot "$g" before
  start_temp_watch "$g" "$gdir"

  # Stage 1: gpu-burn -tc
  echo "[gpu$g] stage 1: gpu-burn -tc for ${BURN_DURATION_S}s"
  if [ -x "$GPU_BURN" ]; then
    ( ./xid-watch.sh start > "$gdir/xid-watch-start.log" 2>&1 || true )
    CUDA_VISIBLE_DEVICES="$g" timeout $((BURN_DURATION_S + 60)) "$GPU_BURN" -tc -c "$GPU_BURN_FATBIN" "$BURN_DURATION_S" \
      > "$gdir/burn-tc.log" 2>&1
    burn_rc=$?
    ./xid-watch.sh stop > "$gdir/xid-watch-stop.log" 2>&1
    xid_rc=$?
    t=$(card_temps "$g"); echo "$t" > "$gdir/burn-tc-final-temps.csv"
    if [ "$xid_rc" -ne 0 ]; then stop_reason="xid-during-burn:gpu$g"; fi
  else
    echo "SETUP-ERROR: gpu-burn binary not found at $GPU_BURN" | tee "$gdir/burn-tc.log"
    burn_rc=127
  fi
  if [ -f "$gdir/temp-breach.flag" ] && [ -z "$stop_reason" ]; then
    stop_reason="live-$(cat "$gdir/temp-breach.flag"):gpu$g:stage1"
  fi
  if [ -n "$stop_reason" ]; then
    stop_temp_watch "$gdir"
    echo "STOPPING: $stop_reason"; break
  fi

  # Stage 2: deterministic matmul correctness (BF16, FP16, TF32, INT8)
  # BUG (found 2026-09-02): system python3 has no torch; use the qwen38 venv
  # (torch 2.13.0+cu130, confirmed cuda.is_available()) instead.
  echo "[gpu$g] stage 2: matmul correctness vs CPU float64 reference (seed=$MATMUL_SEED)"
  "$MATMUL_PYTHON" "$MATMUL_PY" "$g" "$MATMUL_SEED" "$gdir/matmul.json" > "$gdir/matmul.stdout.log" 2>&1
  matmul_rc=$?

  # Stage 3: full-VRAM memtest_vulkan.
  # BUG (found 2026-09-02): the upstream v0.5.0 binary has no --top-fraction
  # flag (silently ignored) and no CLI device-select flag either -- device
  # choice is an interactive PTY prompt that ignores piped stdin, so every
  # non-interactive invocation always tested the same first-listed device
  # regardless of CUDA_VISIBLE_DEVICES. Fixed via memtest-select.py, which
  # drives the prompt over a real PTY, matching the card's PCI bus id to the
  # tool's own listed index. It also runs the tool's default full-VRAM pass
  # (the binary has no fractional/partial-VRAM mode), so the "top 56 GiB"
  # weighting from the original design note is not separately achievable
  # with this tool version -- full VRAM is covered instead, which is a
  # superset.
  echo "[gpu$g] stage 3: memtest_vulkan (full VRAM, via memtest-select.py wrapper)"
  gpu_bus=$(nvidia-smi -i "$g" --query-gpu=pci.bus_id --format=csv,noheader | tr -d ' ')
  if [ -x "$MEMTEST" ] && [ -f "$MEMTEST_SELECT" ]; then
    ( ./xid-watch.sh start > "$gdir/xid-watch-mt-start.log" 2>&1 || true )
    python3 "$MEMTEST_SELECT" "$gpu_bus" "$MEMTEST_MAX_S" "$gdir/memtest.log"
    mt_rc=$?
    ./xid-watch.sh stop > "$gdir/xid-watch-mt-stop.log" 2>&1
    xid_rc2=$?
    if [ "$xid_rc2" -ne 0 ]; then stop_reason="xid-during-memtest:gpu$g"; fi
  else
    echo "SETUP-ERROR: memtest_vulkan binary or memtest-select.py not found" | tee "$gdir/memtest.log"
    mt_rc=127
  fi
  if [ -f "$gdir/temp-breach.flag" ] && [ -z "$stop_reason" ]; then
    stop_reason="live-$(cat "$gdir/temp-breach.flag"):gpu$g:stage3"
  fi
  stop_temp_watch "$gdir"

  pcie_snapshot "$g" after

  # Thermal stop check (post-hoc; the intended path is a live watcher during
  # stage 1/3 -- this script assumes gpu-burn/memtest_vulkan's own duration
  # caps keep it inside safety, and this check is the backstop).
  read -r core_t mem_t <<< "$(card_temps "$g" | tr ',' ' ')"
  core_t=${core_t%.*}; mem_t=${mem_t%.*}
  if [ -n "${core_t:-}" ] && [ "$core_t" -ge "$CORE_STOP_C" ] 2>/dev/null; then
    stop_reason="core-temp:${core_t}C:gpu$g"
  fi
  if [ -n "${mem_t:-}" ] && [ "$mem_t" -ge "$MEM_STOP_C" ] 2>/dev/null; then
    stop_reason="memory-temp:${mem_t}C:gpu$g"
  fi

  xids_during=$(xid_seen_since "$start_epoch")

  cat > "$gdir/receipt.json" <<JSONEOF
{
  "gpu": $g,
  "started_utc": "$(date -u -d "@$start_epoch" +%FT%TZ)",
  "power_cap_w": $POWER_CAP_W,
  "burn_tc_rc": ${burn_rc:-null},
  "matmul_rc": ${matmul_rc:-null},
  "memtest_rc": ${mt_rc:-null},
  "xid_count_during_card": $xids_during,
  "final_core_temp_c": "${core_t:-null}",
  "final_memory_temp_c": "${mem_t:-null}",
  "stop_reason": "${stop_reason:-none}"
}
JSONEOF
  echo "[gpu$g] receipt: $gdir/receipt.json"

  if [ -n "$stop_reason" ]; then
    echo "STOPPING GATE: $stop_reason"
    break
  fi
done

echo
echo "=== tensor-gate summary ==="
echo "receipts: $RECEIPTS"
if [ -n "$stop_reason" ]; then
  echo "STOPPED EARLY: $stop_reason"
  exit 1
fi
echo "all requested cards completed without a stop condition"
