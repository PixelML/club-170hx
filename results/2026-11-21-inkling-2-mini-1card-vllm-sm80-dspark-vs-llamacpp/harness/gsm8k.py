#!/usr/bin/env python3
"""GSM8K accuracy gate (first N test questions, openai/gsm8k main test split) against one chat endpoint.
Answer = last number in the reply content. One JSON line per question."""
import argparse, concurrent.futures as cf, json, re, time, urllib.request


def ask(url, model, q, effort, n):
    body = {"model": model, "messages": [{"role": "user", "content": q + "\nGive the final answer as a number on the last line."}],
            "max_tokens": n, "temperature": 0.0, "reasoning_effort": effort,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return d["choices"][0]["message"].get("content") or "", d["usage"], time.time() - t0, d["choices"][0]["finish_reason"]


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--engine", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="inkling"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--effort", default="low"); ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data)][: a.n]
    out = open(a.out, "w"); ok = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ask, a.url, a.model, r["question"], a.effort, a.max_tokens): i for i, r in enumerate(rows)}
        for f in cf.as_completed(futs):
            i = futs[f]; gold = rows[i]["answer"].split("####")[-1].strip().replace(",", "")
            text, usage, dt, fin = f.result(); pred = num(text)
            good = pred is not None and abs(float(pred) - float(gold)) < 1e-6
            ok += good
            out.write(json.dumps(dict(engine=a.engine, idx=i, gold=gold, pred=pred, correct=good, finish=fin,
                                      completion_tokens=usage["completion_tokens"], seconds=round(dt, 2))) + "\n")
    print(f"{a.engine} gsm8k first {len(rows)}: {ok}/{len(rows)} = {ok/len(rows):.3f}")


if __name__ == "__main__":
    main()
