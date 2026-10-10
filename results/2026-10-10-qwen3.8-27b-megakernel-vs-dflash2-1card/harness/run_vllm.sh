set -x
R=/work/receipts-180w; mkdir -p $R
nvidia-smi --query-gpu=power.limit --format=csv,noheader | grep -q "^180" || { echo CAP_NOT_180; exit 1; }
RD=/work/club-170hx/recipes/qwen3.8-27b-dflash2; RT=/work/qwen-serving
$RT/venv/bin/python $RD/prepare_pinned_models.py --runtime-root $RT --model-root /models > $R/vllm-prepare.log 2>&1 || { echo PREPARE_FAILED; exit 1; }
sleep 60
cd $RT
env CUDA_VISIBLE_DEVICES=0 VLLM_API_KEY=local SPEC=dflash2 CTX=fast MAX_SEQS=1 DFLASH_TOKENS=7 PORT=18020 \
    GPU_UTIL=0.90 KV_MEM= MODEL=/models/Qwen3.8-27B-W4A16-AutoRound-fast DRAFT=/models/Qwen3.8-27B-DFlash2-W4A16 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_V2_CUDAGRAPH_MEM_MIB=1400 \
    setsid bash single-user/start_qwen.sh > $R/vllm-server.log 2>&1 &
echo $! > /work/engine.pids
bash /work/guard.sh $R/telemetry-vllm.csv > /dev/null 2>&1 &
t0=$(date +%s)
for i in $(seq 900); do curl -sf -H "Authorization: Bearer local" localhost:18020/v1/models >/dev/null && break; kill -0 $(cat /work/engine.pids) 2>/dev/null || { echo SERVER_DIED; break; }; sleep 2; done
echo "BOOT_S $(( $(date +%s) - t0 ))"
cd /work
# club recipe suite, unchanged (decode256 / decode900 / prefill_long, /v1/completions, ignore_eos)
$RT/venv/bin/python -c "
import sys; sys.path.insert(0, '$RD')
from pathlib import Path; from live_benchmark import run_suite
run_suite('http://127.0.0.1:18020', 'local', Path('$R/vllm-club-suite.jsonl'), lambda: None)" 2>&1 | tee $R/vllm-club-suite.log
# same chat suite as the megakernel arm
python3 /work/chat_bench.py --url http://127.0.0.1:18020 --engine vllm-dflash2-k7 --out $R/vllm.jsonl \
   --suites greedy,think,prefill --long-file /work/open-jet/megakernel/prefill.cuh --long-chars 8000,24000,48000 2>&1 | tee $R/vllm-bench.log
kill -- -$(cat /work/engine.pids) 2>/dev/null; kill $(cat /work/engine.pids) 2>/dev/null; kill $(cat /work/guard.pid)
echo VLLM_ARM_DONE
