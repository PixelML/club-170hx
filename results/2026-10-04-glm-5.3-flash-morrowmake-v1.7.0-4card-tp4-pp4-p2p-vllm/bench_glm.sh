#!/bin/bash
# usage: bench_glm.sh NAME  -- decode (MiaAI protocol) + prefill against :8030 into runs/NAME/receipts
N=$1; D=<bench>/$N; mkdir -p $D/receipts; cd $D
cp <bench>/glm53-v170-tp4-p2p/bench_decode.py <bench>/glm53-v170-tp4-p2p/bench_long.py .
nvidia-smi --query-gpu=index,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv > receipts/nvidia-smi-start.csv
nohup nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,utilization.gpu --format=csv -l 2 > receipts/telemetry.csv 2>&1 & TEL=$!
for c in 1 8; do for p in structured coding prose; do
  flag=""; [ $p = structured ] && flag=--structured; [ $p = coding ] && flag=--coding
  python3 bench_decode.py --phase $p-c$c $flag --concurrency $c --max-tokens 400 --runs $([ $c = 1 ] && echo 5 || echo 3) --out receipts/decode-$p-c$c.json > receipts/decode-$p-c$c.stdout 2>&1
done; done
python3 bench_long.py prefill receipts/prefill.json > receipts/prefill.out 2>&1
kill $TEL
