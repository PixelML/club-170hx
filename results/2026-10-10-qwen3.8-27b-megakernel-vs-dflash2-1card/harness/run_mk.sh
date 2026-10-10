set -x
pip install -q --break-system-packages jinja2==3.1.6 2>&1 | tail -1
MK=/work/open-jet/megakernel; GGUF=/models/qwen3.8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf
R=/work/receipts-180w; mkdir -p $R; nvidia-smi --query-gpu=power.limit --format=csv,noheader | grep -q "^180" || { echo CAP_NOT_180; exit 1; }
for d in 4 3 0; do sleep 60
  python3 $MK/server/mk_server.py -m $GGUF --port 18080 -c 65536 -d $d > $R/mk-d$d-server.log 2>&1 &
  echo $! > /work/engine.pids
  bash /work/guard.sh $R/telemetry-mk-d$d.csv > /dev/null 2>&1 &
  for i in $(seq 300); do curl -sf localhost:18080/health >/dev/null && break; sleep 2; done
  S=greedy,think,prefill; [ $d = 0 ] && S=greedy,prefill
  python3 /work/chat_bench.py --url http://127.0.0.1:18080 --engine mk-d$d --out $R/mk.jsonl --suites $S \
     --long-file $MK/prefill.cuh --long-chars 8000,24000,48000 2>&1 | tee $R/mk-d$d-bench.log
  kill $(cat /work/engine.pids); kill $(cat /work/guard.pid); sleep 5
done
echo MK_ARM_DONE
