#!/usr/bin/env python3
"""GLM-5.3-Flash (vLLM sm80) prefill/TTFT/c=1 confirmation harness.

Adapted from the club-170hx 4x-box protocol (results/2026-09-03-glm-5.3-flash-
vllm-sm80-4gpu/README.md): decode phase is temperature 0.7, 512 output
tokens, ignore_eos, 5 reps, first rep treated as cold.

Usage:
    python3 bench_glm53.py --base-url http://127.0.0.1:18098 --model glm-5.3-flash \
        --out /workspace/receipts/tp4-record gate prefill ttft decode_c1
"""
import sys
import uuid

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import Client, base_parser, build_fixture, now, save  # noqa: E402

PREFILL_TOKENS = 2900
REPS = 3


def gate(client, out):
    ids = client.models()
    det = [client.completion("The capital of France is", 1) for _ in range(3)]
    texts = [d.get("text_head") for d in det]
    rec = {
        "phase": "gate", "utc": now(), "models": ids, "deterministic_reps": det,
        "deterministic_identical": len(set(texts)) == 1 and all(d["ok"] for d in det),
        "verdict": "PASS" if client.model in ids and len(set(texts)) == 1 and all(d["ok"] for d in det) else "FAIL",
    }
    save(out, "gate.json", rec)
    print(rec["verdict"], ids, texts)
    return rec["verdict"] == "PASS"


def prefill(client, out):
    reps = []
    for i in range(REPS):
        prefix = f"[run {uuid.uuid4().hex}] "
        text, _ = build_fixture(client, PREFILL_TOKENS, prefix)
        r = client.completion(text, 1)
        pt = r.get("usage", {}).get("prompt_tokens")
        r["prefill_tok_s"] = round(pt / r["wall_s"], 2) if r["ok"] and pt else None
        reps.append(r)
        print("prefill rep", i, r.get("usage"), r["wall_s"], r.get("prefill_tok_s"), flush=True)
    ok = [x for x in reps if x["ok"]]
    rec = {
        "phase": "uncached_prefill", "utc": now(), "target_prompt_tokens": PREFILL_TOKENS, "max_tokens": 1, "reps": reps,
        "prompt_tokens_exact": all(x.get("usage", {}).get("prompt_tokens") == PREFILL_TOKENS for x in ok) and len(ok) == REPS,
        "median_wall_s": sorted(x["wall_s"] for x in ok)[len(ok) // 2] if ok else None,
        "median_prefill_tok_s": sorted(x["prefill_tok_s"] for x in ok)[len(ok) // 2] if ok else None,
    }
    save(out, "prefill.json", rec)


def ttft(client, out):
    import json
    import time
    import urllib.request

    text, c = build_fixture(client, PREFILL_TOKENS, "[ttft-fixture] ")
    client.completion(text, 1)  # prime the prefix cache
    reps = []
    for i in range(REPS):
        body = {"model": client.model, "prompt": text, "max_tokens": 32, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        req = urllib.request.Request(client.base_url + "/v1/completions", json.dumps(body).encode(),
                                      {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first = None
        usage = None
        ntok = 0
        err = None
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    d = json.loads(payload)
                    if d.get("choices") and d["choices"][0].get("text"):
                        if first is None:
                            first = time.perf_counter() - t0
                        ntok += 1
                    if d.get("usage"):
                        usage = d["usage"]
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:1000]
        total = time.perf_counter() - t0
        reps.append({"ok": err is None, "ttft_s": round(first, 4) if first else None, "total_s": round(total, 4),
                     "stream_chunks_with_text": ntok, "usage": usage, "error": err})
        print("ttft rep", i, reps[-1], flush=True)
    ok = [x for x in reps if x["ok"] and x["ttft_s"]]
    rec = {"phase": "warm_streaming_ttft", "utc": now(), "fixture_tokens": c, "reps": reps,
           "median_ttft_s": sorted(x["ttft_s"] for x in ok)[len(ok) // 2] if ok else None}
    save(out, "ttft.json", rec)


def decode_c1(client, out):
    prompts = [
        "Explain in careful detail how a modern operating system kernel implements virtual memory.",
        "Write an original short story about a lighthouse keeper.",
        "Write a complete Python implementation of a red-black tree with insert, delete, and search.",
        "Describe the full lifecycle of an HTTP request from browser to origin server and back.",
        "Write a detailed essay on the causes and consequences of the 1973 oil shock.",
    ]
    reps = []
    for i, p in enumerate(prompts):
        r = client.completion(f"[c1-{uuid.uuid4().hex}] " + p, 512, temperature=0.7, ignore_eos=True)
        ct = r.get("usage", {}).get("completion_tokens")
        r["decode_tok_s"] = round(ct / r["wall_s"], 2) if r["ok"] and ct else None
        r["cold"] = (i == 0)
        reps.append(r)
        print("decode rep", i, r.get("usage"), r["wall_s"], r.get("decode_tok_s"), flush=True)
    ok = [x for x in reps if x["ok"]]
    warm = [x for x in ok if not x["cold"]]
    rec = {
        "phase": "c1_decode_confirm", "utc": now(), "reps": reps,
        "median_tok_s_warm": sorted(x["decode_tok_s"] for x in warm)[len(warm) // 2] if warm else None,
        "peak_tok_s_warm": max((x["decode_tok_s"] for x in warm), default=None),
        "cold_rep_tok_s": reps[0].get("decode_tok_s") if reps else None,
    }
    save(out, "decode_c1.json", rec)


PHASES = {"gate": gate, "prefill": prefill, "ttft": ttft, "decode_c1": decode_c1}


def main():
    p = base_parser(__doc__)
    p.add_argument("phases", nargs="*", default=["gate", "prefill", "ttft", "decode_c1"], choices=list(PHASES))
    args = p.parse_args()
    client = Client(args.base_url, args.model)
    for ph in args.phases:
        PHASES[ph](client, args.out)


if __name__ == "__main__":
    main()
