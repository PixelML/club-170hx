#!/bin/bash
# Qwen3.8-27B W4A16 (dbirks AutoRound, stock checkpoint) on the club's SM80 vLLM image.
# usage: qwen_vllm_run.sh <devices e.g. 0,1> <tp> <outdir> [mtp]
set -euo pipefail
# MODEL_STORAGE: directory holding Qwen3.8-27B-W4A16-AutoRound; CACHE_DIR: compile cache;
# RECIPE_DIR: recipes/qwen3.8-27b-dflash2 (for live_benchmark.py). Run from this tools/ directory.
: "${MODEL_STORAGE:?}" "${CACHE_DIR:?}" "${RECIPE_DIR:?}"
DEVS=$1; TP=$2; OUT=$3; SPEC=${4:-none}
IMAGE=ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905
mkdir -p "$OUT"
docker rm -f qwen >/dev/null 2>&1 || true
SPECARGS=()
[ "$SPEC" = mtp ] && SPECARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
t0=$(date +%s)
docker run -d --name qwen --gpus "\"device=$DEVS\"" --ipc=host --shm-size 16g \
  -v $MODEL_STORAGE:/models -v $CACHE_DIR:/root/.cache -p 127.0.0.1:18020:8000 -e VLLM_ENABLE_CUDA_COMPATIBILITY=0 "$IMAGE" \
  --model /models/Qwen3.8-27B-W4A16-AutoRound --served-model-name qwen3.8-27b \
  --tensor-parallel-size "$TP" --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' "${SPECARGS[@]}" >/dev/null
until curl -fsS --max-time 3 http://127.0.0.1:18020/health >/dev/null 2>&1; do
  if [ "$(docker inspect -f '{{.State.Running}}' qwen)" != "true" ]; then
    echo "SERVER EXITED"; docker logs --tail 80 qwen > "$OUT/server-fail.log" 2>&1; grep -E "Error|error" "$OUT/server-fail.log" | tail -8; exit 1
  fi
  [ $(( $(date +%s) - t0 )) -gt 2400 ] && { echo TIMEOUT; exit 1; }
  sleep 5
done
echo "{\"devices\":\"$DEVS\",\"tp\":$TP,\"spec\":\"$SPEC\",\"cold_boot_s\":$(( $(date +%s) - t0 ))}" > "$OUT/boot.json"
docker inspect qwen --format '{{json .Config.Cmd}}' > "$OUT/launch-args.json"
docker logs qwen 2>&1 | grep -E "KV cache|Maximum concurrency|backend|Using .* for" | sed -E 's/^.*(INFO|WARNING) [0-9-]+ [0-9:]+ //' | head -20 > "$OUT/server-key-lines.txt" || true
./temp_guard.sh qwen "$OUT/telemetry.csv" &
G=$!
VLLM_API_KEY=none python3 ./run_qwen_suite.py $RECIPE_DIR "$OUT/suite.jsonl" "$DEVS" | tail -5
python3 ./conc_sweep.py --host http://127.0.0.1:18020 --model qwen3.8-27b --levels 1,4,8,16 --out "$OUT/conc.json" | tail -3
docker logs qwen 2>&1 | grep -iE "Xid|Traceback|CUDA error" | tail -5 > "$OUT/server-errors.txt" || true
docker rm -f qwen >/dev/null
kill $G 2>/dev/null || true
echo DONE
