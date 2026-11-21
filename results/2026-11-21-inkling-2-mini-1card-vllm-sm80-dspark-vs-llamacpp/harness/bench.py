#!/usr/bin/env python3
"""Inkling-2-Mini bench client for one OpenAI-compatible server (vLLM or llama-server).

Streams /v1/chat/completions. Token counts come from the final usage object, never from stream events.
decode tok/s = (completion_tokens - 1) / (total - ttft). One JSON line per request.

Suites:
  c1      single user, 3 prompt types (code, prose, structured JSON), reps each, at T=0 and T=1 (model default)
  conc    concurrency sweep (aggregate output tok/s), mixed prompts, T=1
  prefill uncached long prompts (prefix salted), TTFT and prefill tok/s
"""
import argparse, json, threading, time, urllib.request

PROMPTS = {
    "code": "Write a Python module that implements an LRU cache class with get, put and a capacity limit, "
            "using an OrderedDict, with type hints, docstrings and five pytest tests.",
    "prose": "Write an essay of about 600 words on why lighthouses were built where they were, and what "
             "replaced them, for a general reader.",
    "struct": "Return a JSON array of 12 objects for fictional bookstore inventory. Each object has: id (int), "
              "title, author, genre, year (int), price (float), in_stock (bool), tags (array of 3 strings). "
              "Output only the JSON.",
}
FILLER = open(__file__).read()


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
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            m = json.loads(p)
            if m.get("usage"):
                usage = m["usage"]
            for ch in m.get("choices") or []:
                d = ch.get("delta") or {}
                c, rc = d.get("content"), d.get("reasoning_content") or d.get("reasoning")
                if (c or rc) and ttft is None:
                    ttft = time.perf_counter() - t0
                if c: text.append(c)
                if rc: reasoning.append(rc)
    total = time.perf_counter() - t0
    if usage is None or ttft is None:
        raise RuntimeError("stream ended without usage or first token")
    ct, pt = int(usage["completion_tokens"]), int(usage["prompt_tokens"])
    return dict(ttft_s=round(ttft, 4), total_s=round(total, 4), prompt_tokens=pt, completion_tokens=ct,
                decode_tok_s=round((ct - 1) / max(total - ttft, 1e-9), 2), prefill_tok_s=round(pt / ttft, 1),
                text_head="".join(text)[:300], text_chars=len("".join(text)), reasoning_chars=len("".join(reasoning)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="c1,conc,prefill")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--temps", default="0,1")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--conc", default="1,4,8,16")
    ap.add_argument("--prefill", default="4000,16000,32000", help="approx prompt tokens")
    ap.add_argument("--ignore-eos", action="store_true")
    a = ap.parse_args()
    out = open(a.out, "a")
    lock = threading.Lock()

    def emit(rec):
        rec.update(engine=a.engine, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with lock:
            out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"[{a.engine}] {rec['suite']}/{rec['case']} T{rec.get('temperature')} r{rec.get('rep',0)}: "
              f"{rec['completion_tokens']} tok {rec['decode_tok_s']} tok/s ttft {rec['ttft_s']}s", flush=True)

    def body(prompt, n, t, seed=None):
        b = dict(model=a.model, messages=[{"role": "user", "content": prompt}], max_tokens=n, temperature=t,
                 reasoning_effort=a.effort, chat_template_kwargs={"reasoning_effort": a.effort})
        if seed is not None: b["seed"] = seed
        if a.ignore_eos: b["ignore_eos"] = True
        return b

    suites = a.suites.split(",")
    stream_chat(a.url, body("Say hi.", 16, 0.0))  # warm-up, not recorded
    if "c1" in suites:
        for t in [float(x) for x in a.temps.split(",")]:
            for case, p in PROMPTS.items():
                for rep in range(a.reps):
                    r = stream_chat(a.url, body(p, a.n, t, seed=100 + rep))
                    emit(dict(r, suite="c1", case=case, rep=rep, temperature=t, max_tokens=a.n, effort=a.effort))
    if "conc" in suites:
        cases = list(PROMPTS.items())
        for c in [int(x) for x in a.conc.split(",")]:
            res, errs = [], []
            def worker(i):
                case, p = cases[i % len(cases)]
                try:
                    r = stream_chat(a.url, body(p, a.n, 1.0, seed=500 + i)); r.update(case=case); res.append(r)
                except Exception as e:
                    errs.append(repr(e))
            ths = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
            t0 = time.perf_counter()
            [th.start() for th in ths]; [th.join() for th in ths]
            wall = time.perf_counter() - t0
            tot = sum(r["completion_tokens"] for r in res)
            rec = dict(suite="conc", case=f"c{c}", concurrency=c, wall_s=round(wall, 3), completion_tokens=tot,
                       decode_tok_s=round(tot / wall, 2), ttft_s=round(max(r["ttft_s"] for r in res), 3) if res else None,
                       per_user_tok_s=[r["decode_tok_s"] for r in res], errors=errs, temperature=1.0, max_tokens=a.n)
            emit(rec)
    if "prefill" in suites:
        for n in [int(x) for x in a.prefill.split(",")]:
            src = FILLER
            while len(src) < n * 4: src += "\n" + FILLER
            for rep in range(2):
                p = f"[salt {time.time_ns()}]\nHere is a source file:\n\n{src[:n*4]}\n\nSummarise it in one sentence."
                r = stream_chat(a.url, body(p, 64, 0.0))
                emit(dict(r, suite="prefill", case=f"tok{n}", rep=rep, temperature=0.0, max_tokens=64))
    out.close()


if __name__ == "__main__":
    main()
