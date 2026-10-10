#!/usr/bin/env python3
"""One OpenAI chat-completions client for every engine (megakernel mk_server, llama-server, vLLM).

Streams /v1/chat/completions, takes token counts from the final usage object, and writes one
JSON line per request. decode tok/s = (completion_tokens - 1) / (total - ttft).

  python3 chat_bench.py --url http://127.0.0.1:18080 --engine mk-d4 --out receipts.jsonl
"""
import argparse
import json
import statistics
import time
import urllib.request

# megakernel bench/prompts.txt (greedy, thinking off)
GREEDY = [
    ("code_prime", "Write a Python function that checks whether a number is prime, with a docstring and a few doctests."),
    ("hashmap", "Explain how a hash map works internally, including collision handling and resizing, for a junior developer."),
    ("story_lighthouse", "Write a short story (about 300 words) about a lighthouse keeper who discovers a message in a bottle."),
    ("flask_fastapi", "Give me a step-by-step plan for migrating a Flask REST API to FastAPI, including testing and deployment."),
    # club-170hx recipes/qwen3.8-27b-dflash2 P256
    ("story_robot", "Write a story about a robot who learns to paint."),
]
# megakernel bench/think_bench.py PROMPTS (thinking on, open-jet sampling preset)
THINK = [
    ("intervals", "Write a Python function that merges overlapping intervals in a list of [start, end] pairs, with a few pytest tests."),
    ("bash_largest", "Write a bash script that finds the 10 largest files under a directory (argument), skipping .git, and prints sizes in human-readable form."),
    ("c_ringbuf", "In C, implement a fixed-capacity ring buffer of ints with push, pop and is_full, safe for one producer and one consumer thread."),
    ("mutex_sem", "Explain the difference between a mutex and a semaphore, and when to use each, with a short Python example of each."),
]


def stream_chat(url, body, timeout=1800):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.perf_counter()
    ttft, usage, text, reasoning = None, None, [], []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            msg = json.loads(payload)
            if msg.get("usage"):
                usage = msg["usage"]
            for ch in msg.get("choices") or []:
                d = ch.get("delta") or {}
                c, rc = d.get("content"), d.get("reasoning_content") or d.get("reasoning")
                if (c or rc) and ttft is None:
                    ttft = time.perf_counter() - t0
                if c:
                    text.append(c)
                if rc:
                    reasoning.append(rc)
    total = time.perf_counter() - t0
    if usage is None or ttft is None:
        raise RuntimeError("stream ended without usage or without a first token")
    ct = int(usage["completion_tokens"])
    return dict(ttft_s=round(ttft, 4), total_s=round(total, 4), prompt_tokens=int(usage["prompt_tokens"]),
                completion_tokens=ct, decode_tok_s=round((ct - 1) / (total - ttft), 2),
                prefill_tok_s=round(int(usage["prompt_tokens"]) / ttft, 1),
                text_head=("".join(text))[:400], reasoning_chars=len("".join(reasoning)))


def long_prompt(path, approx_chars):
    src = open(path).read()
    while len(src) < approx_chars:
        src += "\n" + src
    return "Here is a source file:\n\n" + src[:approx_chars] + "\n\nSummarise what this file does in one sentence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine", required=True, help="label written into every receipt")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="greedy,think,prefill")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--greedy-n", type=int, default=512)
    ap.add_argument("--think-n", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--long-file", default="")
    ap.add_argument("--long-chars", default="8000,24000,48000")
    ap.add_argument("--ignore-eos", action="store_true", help="only for engines that accept it (vLLM, llama-server)")
    a = ap.parse_args()
    out = open(a.out, "a")

    def emit(rec):
        rec.update(engine=a.engine, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"[{a.engine}] {rec['suite']}/{rec['case']} rep{rec.get('rep', 0)}: "
              f"{rec['completion_tokens']} tok, decode {rec['decode_tok_s']} tok/s, ttft {rec['ttft_s']} s", flush=True)

    base = dict(model=a.model)
    if a.ignore_eos:
        base["ignore_eos"] = True
    suites = a.suites.split(",")
    # warm-up (not recorded)
    stream_chat(a.url, dict(base, messages=[{"role": "user", "content": "Say hi."}], max_tokens=16, temperature=0.0,
                            chat_template_kwargs={"enable_thinking": False}))
    if "greedy" in suites:
        for name, p in GREEDY:
            for rep in range(a.reps):
                r = stream_chat(a.url, dict(base, messages=[{"role": "user", "content": p}], max_tokens=a.greedy_n,
                                            temperature=0.0, chat_template_kwargs={"enable_thinking": False}))
                emit(dict(r, suite="greedy", case=name, rep=rep, max_tokens=a.greedy_n))
    if "think" in suites:
        for name, p in THINK:
            for s in range(a.seeds):
                r = stream_chat(a.url, dict(base, messages=[{"role": "user", "content": p}], max_tokens=a.think_n,
                                            temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, seed=1000 + s,
                                            chat_template_kwargs={"enable_thinking": True}))
                emit(dict(r, suite="think", case=name, rep=s, max_tokens=a.think_n))
    if "prefill" in suites and a.long_file:
        for n in [int(x) for x in a.long_chars.split(",")]:
            p = long_prompt(a.long_file, n)
            for rep in range(a.reps):
                # vary the first line so no engine can reuse a cached prefix
                msg = f"[run {a.engine} {rep} {time.time_ns()}]\n" + p
                r = stream_chat(a.url, dict(base, messages=[{"role": "user", "content": msg}], max_tokens=128,
                                            temperature=0.0, chat_template_kwargs={"enable_thinking": False}))
                emit(dict(r, suite="prefill", case=f"chars{n}", rep=rep, max_tokens=128))
    out.close()


if __name__ == "__main__":
    main()
