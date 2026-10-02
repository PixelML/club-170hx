#!/usr/bin/env bash
# Decode matrix: 3 prompt types x {1,8} users, 400 tok, T=0, median of 5
set -u
R=receipts; mkdir -p $R
for c in 1 8; do
  for p in structured coding prose; do
    flag=""; [ $p = structured ] && flag=--structured; [ $p = coding ] && flag=--coding
    python3 bench_decode.py --phase $p-c$c $flag --runs 5 --max-tokens 400 --skip-coherence --concurrency $c --out $R/decode-$p-c$c.json > $R/decode-$p-c$c.stdout 2>&1
    echo "$p c=$c exit=$?"
  done
done
echo DONE
