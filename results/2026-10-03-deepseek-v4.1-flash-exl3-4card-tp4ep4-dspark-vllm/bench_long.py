#!/usr/bin/env python3
"""Cold prefill (max_tokens=1) and long-context decode for the GLM-5.3 recipe on :8000.

Prompts are real text (Python stdlib source + markdown docs) with a unique nonce
first, so nothing hits the prefix cache. Token counts come from usage.
"""
import glob, json, os, sys, time, uuid, urllib.request

BASE, MODEL = "http://127.0.0.1:8040", "deepseek-v4.1-flash"
files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True)) + sorted(glob.glob("/models/miaai-bench/**/*.md", recursive=True))
corpus = "".join(open(f, errors="ignore").read() for f in files[:4000])
CHARS_PER_TOK = 3.4  # refined per request from usage

def stream(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None; buf = b""
    with urllib.request.urlopen(req, timeout=3600) as r:
        while (piece := r.read1(256)):
            buf += piece
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1); line = line.strip()
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]": continue
                o = json.loads(line[5:])
                usage = o.get("usage") or usage
                d = ((o.get("choices") or [{}])[0].get("delta") or {})
                if first is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content")):
                    first = time.perf_counter()
    t1 = time.perf_counter()
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": first - t0,
            "prefill_tok_s": pt / (first - t0), "decode_tok_s": (ct - 1) / (t1 - first) if ct > 1 else None}

def prompt_of(target_tokens, offset, task):
    n = int(target_tokens * CHARS_PER_TOK)
    return f"[run {uuid.uuid4()}]\n" + corpus[offset:offset + n] + "\n\n" + task

mode, out = sys.argv[1], sys.argv[2]
res = []
if mode == "prefill":   # median of 2 per size, max_tokens=1
    for i, tgt in enumerate([8000, 30000, 54000]):
        for rep in range(2):
            r = stream(prompt_of(tgt, (i * 2 + rep) * 200000, "Summarize the text above in one line."), 1)
            r.update(target=tgt, rep=rep); res.append(r); print(json.dumps(r), flush=True)
else:                   # long-context decode curve, 400 tokens
    for i, tgt in enumerate([1000, 8000, 32000, 54000]):
        r = stream(prompt_of(tgt, i * 50000 % max(1, len(corpus) - 800000), "Explain in detail, step by step, what the code and text above do."), 400)
        r.update(target=tgt); res.append(r); print(json.dumps(r), flush=True)
json.dump({"mode": mode, "corpus_chars": len(corpus), "results": res}, open(out, "w"), indent=1)
