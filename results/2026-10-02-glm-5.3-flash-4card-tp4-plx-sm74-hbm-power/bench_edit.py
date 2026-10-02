#!/usr/bin/env python3
"""Edit-reply decode bench: the model returns a given source file with one small
change, so most of the reply repeats the prompt (copy-draft territory).

Decode tok/s = (completion_tokens - 1) / (end - first_token_time); temp 0, thinking
off, median of --runs. Writes each reply's sha256 so two servers can be checked
for identical output.
"""
import argparse, hashlib, json, os, statistics, time, urllib.request

BASE, MODEL = os.environ.get("GLM_BASE", "http://127.0.0.1:8000"), "glm-5.3-flash"
SRC = "/usr/lib/python3.13/textwrap.py"

TASKS = {
    "rename": "Rename the class TextWrapper to LineWrapper everywhere (definition, "
              "uses, docstrings). Return the complete modified file and nothing else, "
              "in one ```python block.",
    "docstring": "Add a one-line comment `# edited` as the first line of every function "
                 "body. Return the complete modified file and nothing else, in one "
                 "```python block.",
}


def stream(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None; text = []; buf = b""
    with urllib.request.urlopen(req, timeout=3600) as r:
        while (piece := r.read1(4096)):
            buf += piece
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1); line = line.strip()
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                    continue
                o = json.loads(line[5:])
                usage = o.get("usage") or usage
                for c in o.get("choices") or []:
                    d = (c.get("delta") or {}).get("content")
                    if d:
                        if first is None:
                            first = time.perf_counter()
                        text.append(d)
    end = time.perf_counter()
    n = usage["completion_tokens"]
    return {"tok_s": (n - 1) / (end - first), "ttft_s": first - t0, "tokens": n,
            "sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16], "text": "".join(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = open(SRC).read()
    rec = {}
    for name, task in TASKS.items():
        prompt = f"{task}\n\n```python\n{src}```"
        stream(prompt, 16)  # warm the prefix cache
        runs = [stream(prompt, a.max_tokens) for _ in range(a.runs)]
        rec[name] = {"tok_s_median": statistics.median(r["tok_s"] for r in runs),
                     "tokens": [r["tokens"] for r in runs],
                     "shas": [r["sha"] for r in runs], "runs": runs}
        open(a.out.replace(".json", f"-{name}.txt"), "w").write(runs[0]["text"])
        print(name, round(rec[name]["tok_s_median"], 1), rec[name]["tokens"], rec[name]["shas"], flush=True)
    json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
