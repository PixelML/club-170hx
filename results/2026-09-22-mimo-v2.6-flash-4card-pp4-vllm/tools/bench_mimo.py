#!/usr/bin/env python3
"""P1/P2 decode benchmark for the MiMo-V2.6-Flash PP4 lane.

Protocol (mirrors the 2026-09-05 GLM-5.3-Flash lane):
  P1: greedy (temperature 0), 512 output tokens, 5 workloads, 3 reps, median
      reported; a repeat guard flags collapsed completions and those cells are
      NOT read as throughput.
  P2: temperature 1.0 / top_p 0.95 (vendor-recommended), ignore_eos, one fixed
      long-form prompt, 5 reps (first rep cold), median reported.

Token counts always come from the final streamed `usage` object, never from
counting stream events.

Usage:
  python3 bench_mimo.py --host http://localhost:8000 --model mimo-v2.6-flash \
      --protocol p1 --out p1.json
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

WORKLOADS = {
    "code": "Write a Python function that merges two sorted lists without duplicates. Include a short docstring and one test call.",
    "json": 'Produce a JSON object with keys "name", "version", "tags" (list of 5 strings) and "stable": true describing a release of a fictional CLI tool called "bandcatch".',
    "counting": "Count from 1 to 120, comma separated, then state how many numbers you listed.",
    "math": "A train covers 3 sections of 47 km, 88 km and 65 km at 60, 80 and 50 km/h respectively. Give total time in minutes, rounded to one decimal, then verify by recomputing.",
    "prose": "Explain how a four-stage assembly line changes throughput when the third stage is twice as slow as the others. Two paragraphs.",
}
P2_PROMPT = (
    "You are maintaining a large codebase. Describe, step by step, how to move a "
    "widely-imported utility module into a package without breaking callers, "
    "including how to keep the old import path working during migration, how to "
    "run verification, and how to communicate the change. Be exhaustive."
)

REPEAT_LINE = re.compile(r"(.{20,}?)\1{2,}", re.S)


def repeat_guard(text: str) -> bool:
    """True when the completion looks repetition-collapsed."""
    if REPEAT_LINE.search(text[-2048:]):
        return True
    tail = text[-256:]
    if len(tail) >= 32 and len(set(tail)) <= 8:
        return True
    return False


def stream_once(host, model, messages, temperature, top_p, ignore_eos, max_tokens):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(
        host.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    text_parts = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(delta)
    total = time.perf_counter() - t0
    completion_tokens = usage["completion_tokens"] if usage else 0
    decode_s = max(total - (ttft or 0.0), 1e-9)
    text = "".join(text_parts)
    return {
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_s": round(total, 4),
        "prompt_tokens": usage["prompt_tokens"] if usage else None,
        "completion_tokens": completion_tokens,
        "decode_tok_s": round(completion_tokens / decode_s, 2),
        "degenerate": repeat_guard(text),
        "text_chars": len(text),
        "text_head": text[:160],
    }


def run(args):
    out = {
        "protocol": args.protocol,
        "model": args.model,
        "host": args.host,
        "reps": args.reps,
        "out_tokens": args.out_tokens,
        "sampling": {"temperature": 0.0 if args.protocol == "p1" else 1.0,
                     "top_p": 1.0 if args.protocol == "p1" else 0.95,
                     "ignore_eos": args.protocol == "p2"},
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
        "runs": {},
    }
    if args.protocol == "p1":
        sets = {k: [{"role": "user", "content": v}] for k, v in WORKLOADS.items()}
    else:
        sets = {"longform": [{"role": "user", "content": P2_PROMPT}]}
    for name, messages in sets.items():
        reps = []
        for i in range(args.reps):
            r = stream_once(args.host, args.model, messages,
                            out["sampling"]["temperature"], out["sampling"]["top_p"],
                            out["sampling"]["ignore_eos"], args.out_tokens)
            print(f"  {name} rep{i}: {r['decode_tok_s']} tok/s ttft={r['ttft_s']}s "
                  f"out={r['completion_tokens']} degenerate={r['degenerate']}")
            reps.append(r)
        clean = [r["decode_tok_s"] for r in reps if not r["degenerate"]]
        out["runs"][name] = {
            "reps": reps,
            "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in reps),
            "median_ttft_s": statistics.median(r["ttft_s"] for r in reps if r["ttft_s"] is not None),
            "clean_median_decode_tok_s": statistics.median(clean) if clean else None,
            "flagged_reps": [i for i, r in enumerate(reps) if r["degenerate"]],
        }
    out["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--protocol", choices=["p1", "p2"], required=True)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--out-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.reps is None:
        args.reps = 3 if args.protocol == "p1" else 5
    run(args)
