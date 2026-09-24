#!/usr/bin/env python3
"""Aggregate-throughput sweep: N concurrent streams, sampled, ignore_eos.

Each level runs `c` workers for a fixed number of rounds; aggregate tok/s is
total completion tokens (from final usage) divided by wall time of the level.
"""
import argparse
import json
import threading
import time
import urllib.request

PROMPT = ("Write a detailed, step-by-step technical explanation of how a hash map "
          "handles collisions, with examples in Python.")


def one(host, model, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(host + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["usage"]["completion_tokens"]


def level(host, model, c, rounds, max_tokens):
    toks = []
    lock = threading.Lock()

    def worker():
        for _ in range(rounds):
            n = one(host, model, max_tokens)
            with lock:
                toks.append(n)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(c)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    return {"concurrency": c, "requests": len(toks), "completion_tokens": sum(toks),
            "wall_s": round(wall, 2), "aggregate_tok_s": round(sum(toks) / wall, 2),
            "per_stream_tok_s": round(sum(toks) / wall / c, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--levels", default="1,4,8,16,32")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    one(a.host, a.model, 16)  # warmup
    res = []
    for c in (int(x) for x in a.levels.split(",")):
        r = level(a.host, a.model, c, a.rounds, a.max_tokens)
        print(json.dumps(r), flush=True)
        res.append(r)
    json.dump({"prompt": PROMPT, "sampling": {"temperature": 1.0, "top_p": 0.95, "ignore_eos": True},
               "max_tokens": a.max_tokens, "rounds": a.rounds, "levels": res},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
