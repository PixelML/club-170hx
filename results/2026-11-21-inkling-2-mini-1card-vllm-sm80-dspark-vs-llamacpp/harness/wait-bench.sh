#!/bin/bash
# wait-bench.sh <port> <engine-label> <out-stem> [bench.py args]  — run on the GPU host
P=$1; E=$2; O=$3; shift 3
cd /work
for i in $(seq 120); do
  curl -sf -m 3 "http://127.0.0.1:$P/v1/models" >/dev/null && break; sleep 5
done
python3 bench.py --url "http://127.0.0.1:$P" --engine "$E" --out "receipts/$O.jsonl" "$@" > "receipts/$O.log" 2>&1
echo "DONE $E exit=$?" >> receipts/$O.log
