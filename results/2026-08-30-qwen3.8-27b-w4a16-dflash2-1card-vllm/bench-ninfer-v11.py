#!/usr/bin/env python3
"""bench-ninfer.py — v11 measurement harness for NInfer (engine=ninfer).

Protocol-identical to the v9/v10 vLLM harness (warmup 1 + 3 samples,
decode256/decode900/prefill_long, temperature 0.0, ignore_eos, streaming,
usage.completion_tokens counted, nvidia-smi peak sampler) with two engine
mandated deltas:
  - POSTs /v1/chat/completions: the fork has NO /v1/completions route
    (verified in src/serve/http_server.cpp register_routes()).
  - Token counts come from usage in the final SSE chunk; /tokenize is
    never called (no such endpoint in the fork).

Writes nothing to disk; stdout only (one JSON object per line).
"""

import json
import subprocess
import threading
import time
import urllib.request

BASE_URL = "http://127.0.0.1:18020"
MODEL = "qwen3.8-27b"
KEY = "<bench-api-key>"
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

peak = {"util": 0, "power": 0, "mem": 0, "clk": 0, "temp": 0}
stop = False


def sample():
    global stop
    while not stop:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            u, p, m, c, t = map(float, out.split(","))
            peak["util"] = max(peak["util"], u)
            peak["power"] = max(peak["power"], p)
            peak["mem"] = max(peak["mem"], m)
            peak["clk"] = max(peak["clk"], c)
            peak["temp"] = max(peak["temp"], t)
        except Exception:
            pass
        time.sleep(1)


def req_stream(prompt, maxtok):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "temperature": 0.0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    t0 = time.perf_counter()
    ttft = None
    n = 0
    ctok = None
    ptok = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if line.startswith(b"data: "):
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                try:
                    obj = json.loads(payload)
                    u = obj.get("usage") or {}
                    if u.get("completion_tokens") is not None:
                        ctok = u["completion_tokens"]
                    if u.get("prompt_tokens") is not None:
                        ptok = u["prompt_tokens"]
                except Exception:
                    pass
    if ctok is None:
        ctok = n
    return ttft, time.perf_counter() - t0, ctok, ptok


def bench(name, prompt, maxtok, warm, samples):
    for i in range(warm):
        req_stream(prompt, maxtok)
    res = []
    for i in range(samples):
        ttft, total, n, ptok = req_stream(prompt, maxtok)
        row = {"engine": "ninfer", "run": name, "i": i,
               "ttft_ms": round(ttft * 1000, 1), "total_s": round(total, 3),
               "out_tok": n, "decode_tok_s": round((n - 1) / (total - ttft), 2)}
        if ptok is not None:
            row["prompt_tokens"] = ptok
        print(json.dumps(row), flush=True)
        res.append((ttft, total, n, ptok))
    ttft = sum(r[0] for r in res) / len(res) * 1000
    tot = sum(r[1] for r in res) / len(res)
    nn = sum(r[2] for r in res) / len(res)
    ptoks = [r[3] for r in res if r[3] is not None]
    summary = {"engine": "ninfer", "summary": name,
               "mean_ttft_ms": round(ttft, 1), "mean_total_s": round(tot, 3),
               "mean_out_tok": round(nn, 1),
               "decode_tok_s": round((nn - 1) / (tot - ttft / 1000), 2)}
    if ptoks:
        summary["prompt_tokens"] = ptoks[0]
        summary["prefill_tok_s"] = round(ptoks[0] / (ttft / 1000), 1)
    print(json.dumps(summary), flush=True)


def main():
    print(json.dumps({"engine": "ninfer", "bench": "v11", "base_url": BASE_URL,
                      "model": MODEL}), flush=True)
    threading.Thread(target=sample, daemon=True).start()
    P256 = "Write a story about a robot who learns to paint."
    LONG = ("summarize the following text. " +
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega ") * 200
    bench("decode256", P256, 256, 1, 3)
    bench("decode900", P256, 900, 1, 3)
    bench("prefill_long", LONG, 8, 1, 3)
    time.sleep(2)
    stop = True
    time.sleep(1)
    print(json.dumps({"engine": "ninfer", "gpu_peak": peak}), flush=True)


if __name__ == "__main__":
    main()
