#!/usr/bin/env python3
"""Summarize bench receipts (*.jsonl) into one CSV row per (engine, suite, case, temperature)."""
import collections, csv, json, statistics as st, sys

rows = collections.defaultdict(list)
for f in sys.argv[1:]:
    for line in open(f):
        r = json.loads(line)
        rows[(r["engine"], r["suite"], r["case"], r.get("temperature"))].append(r)
w = csv.writer(sys.stdout)
w.writerow(["engine", "suite", "case", "temperature", "n", "median_tok_s", "min_tok_s", "max_tok_s",
            "median_ttft_s", "prompt_tokens", "median_completion_tokens", "median_prefill_tok_s"])
for (eng, suite, case, t), v in rows.items():
    d = [x["decode_tok_s"] for x in v]
    w.writerow([eng, suite, case, t, len(v), round(st.median(d), 2), min(d), max(d),
                round(st.median([x["ttft_s"] for x in v if x.get("ttft_s") is not None] or [0]), 3),
                v[0].get("prompt_tokens", ""), st.median([x["completion_tokens"] for x in v]),
                round(st.median([x["prefill_tok_s"] for x in v]), 1) if "prefill_tok_s" in v[0] else ""])
