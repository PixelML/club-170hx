#!/usr/bin/env python3
"""Uncached single-request prefill probe: unique ~21.5k-token prompts,
max_tokens=1, streaming, TTFT from the first streamed chunk, prompt tokens
from the final usage object. One JSON line per sample."""
import argparse
import json
import time
import urllib.request
import uuid

FILLER = ("The quarterly maintenance log records pump pressure, valve state, bearing "
          "temperature and operator notes for each station along the line. ")


def once(host, model, repeats):
    prompt = f"[{uuid.uuid4()}] " + FILLER * repeats + "\nSummarise the log in one sentence."
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    t0 = time.perf_counter()
    ttft = None
    usage = None
    req = urllib.request.Request(host + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            msg = json.loads(payload)
            if msg.get("usage"):
                usage = msg["usage"]
    return {"prompt_tokens": usage["prompt_tokens"], "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(usage["prompt_tokens"] / ttft, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--repeats", type=int, default=880)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    once(a.host, a.model, 50)  # warmup, not recorded
    rows = []
    with open(a.out, "w") as f:
        for i in range(a.samples):
            r = once(a.host, a.model, a.repeats)
            r["sample"] = i + 1
            f.write(json.dumps(r) + "\n")
            rows.append(r)
    rows.sort(key=lambda r: r["prefill_tok_s"])
    print("median prefill tok/s", rows[len(rows) // 2]["prefill_tok_s"], "at", rows[0]["prompt_tokens"], "prompt tokens")


if __name__ == "__main__":
    main()
