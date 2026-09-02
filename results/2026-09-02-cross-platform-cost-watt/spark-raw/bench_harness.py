#!/usr/bin/env python3
"""Canonical 4-card text benchmark harness for DeepSeek-V4-Flash-Vision-Exp on the
0731-fork SM80 vLLM recipe. Uses only the OpenAI-compatible HTTP API.

Phases (each writes a raw JSON receipt into OUT/):
  gate     : /v1/models + deterministic greedy token check
  prefill  : uncached prefill, 2941 prompt tokens, max_tokens=1, 3 reps, unique prefix each
  ttft     : warm streaming TTFT, 3 reps, same fixture (cached prefix)
  ladder   : C1,C2,C4,C8,C16 greedy, ignore_eos, 400 completion tokens, 1 warmup + 3 measured reps
"""
import json, os, sys, time, uuid, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.error

URL = os.environ.get("DSV4_URL", "http://127.0.0.1:18098")
MODEL = os.environ.get("DSV4_MODEL_NAME", "dsv4v")
OUT = os.environ.get("DSV4_OUT", ".")
PREFILL_TOKENS = 2941
LADDER_TOKENS = 400
LADDER = [1, 2, 4, 8, 16]
REPS = 3

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def post(path, body, timeout=1800, stream=False):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def completion(prompt, max_tokens, ignore_eos=False, timeout=1800, seed=None):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    if ignore_eos: body["ignore_eos"] = True
    if seed is not None: body["seed"] = seed
    t0 = time.perf_counter()
    try:
        r = json.load(post("/v1/completions", body, timeout))
        dt = time.perf_counter() - t0
        return {"ok": True, "wall_s": round(dt, 4), "usage": r["usage"], "text_head": r["choices"][0]["text"][:80],
                "finish_reason": r["choices"][0].get("finish_reason")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}"}
    except Exception as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"{type(e).__name__}: {e}"[:2000]}

def tokenize(text):
    r = json.load(post("/tokenize", {"model": MODEL, "prompt": text}, 120))
    return r["count"]

def save(name, obj):
    p = os.path.join(OUT, name)
    with open(p + ".tmp", "w") as f: json.dump(obj, f, indent=2)
    os.replace(p + ".tmp", p)
    print("saved", p, flush=True)

def smi():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=index,power.draw,temperature.gpu,utilization.gpu,memory.used",
                                        "--format=csv,noheader"], text=True).strip().split("\n")
    except Exception as e:
        return [str(e)]

# ---------------- fixture -----------------
BASE = ("The following is a technical reference on distributed systems. Section {n}: consensus protocols. "
        "A replicated state machine applies a deterministic sequence of commands to identical copies of state on every node. "
        "Leader election chooses one node to order commands; followers acknowledge log entries; commitment requires a quorum. "
        "Failure detectors use heartbeats and timeouts; network partitions cause split votes; term numbers resolve stale leaders. ")

def build_fixture(target, prefix=""):
    """Grow a text until the tokenizer reports exactly `target` tokens (including prefix)."""
    body = ""
    n = 0
    while tokenize(prefix + body) < target:
        n += 1
        body += BASE.format(n=n)
    words = body.split(" ")
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        if tokenize(prefix + " ".join(words[:mid])) < target: lo = mid + 1
        else: hi = mid
    text = prefix + " ".join(words[:lo])
    c = tokenize(text)
    tries = 0
    while c != target and tries < 40:
        tries += 1
        if c > target: text = text[:-1]
        else: text += " a"
        c = tokenize(text)
    return text, c

# ---------------- phases -----------------
def gate():
    r = json.load(urllib.request.urlopen(URL + "/v1/models", timeout=60))
    ids = [m["id"] for m in r["data"]]
    det = [completion("The capital of France is", 1) for _ in range(3)]
    texts = [d.get("text_head") for d in det]
    rec = {"phase": "gate", "utc": now(), "models": ids, "deterministic_reps": det,
           "deterministic_identical": len(set(texts)) == 1 and all(d["ok"] for d in det),
           "verdict": "PASS" if MODEL in ids and len(set(texts)) == 1 and all(d["ok"] for d in det) else "FAIL"}
    save("gate.json", rec); print(rec["verdict"], ids, texts); return rec["verdict"] == "PASS"

def prefill():
    reps = []
    for i in range(REPS):
        prefix = f"[run {uuid.uuid4().hex}] "
        text, c = build_fixture(PREFILL_TOKENS, prefix)
        r = completion(text, 1)
        r["fixture_tokens_by_tokenizer"] = c
        pt = r.get("usage", {}).get("prompt_tokens")
        r["prefill_tok_s"] = round(pt / r["wall_s"], 2) if r["ok"] and pt else None
        reps.append(r); print("prefill rep", i, r.get("usage"), r["wall_s"], r.get("prefill_tok_s"), flush=True)
    ok = [x for x in reps if x["ok"]]
    rec = {"phase": "uncached_prefill", "utc": now(), "target_prompt_tokens": PREFILL_TOKENS, "max_tokens": 1, "reps": reps,
           "prompt_tokens_exact": all(x.get("usage", {}).get("prompt_tokens") == PREFILL_TOKENS for x in ok) and len(ok) == REPS,
           "median_wall_s": sorted(x["wall_s"] for x in ok)[len(ok)//2] if ok else None,
           "median_prefill_tok_s": sorted(x["prefill_tok_s"] for x in ok)[len(ok)//2] if ok else None,
           "gpu_after": smi()}
    save("prefill.json", rec)

def ttft():
    text, c = build_fixture(PREFILL_TOKENS, "[ttft-fixture] ")
    completion(text, 1)  # prime the prefix cache
    reps = []
    for i in range(REPS):
        body = {"model": MODEL, "prompt": text, "max_tokens": 32, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.perf_counter(); first = None; usage = None; ntok = 0; err = None
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data:"): continue
                    payload = line[5:].strip()
                    if payload == "[DONE]": break
                    d = json.loads(payload)
                    if d.get("choices") and d["choices"][0].get("text"):
                        if first is None: first = time.perf_counter() - t0
                        ntok += 1
                    if d.get("usage"): usage = d["usage"]
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:1000]
        total = time.perf_counter() - t0
        reps.append({"ok": err is None, "ttft_s": round(first, 4) if first else None, "total_s": round(total, 4),
                     "stream_chunks_with_text": ntok, "usage": usage, "error": err})
        print("ttft rep", i, reps[-1], flush=True)
    ok = [x for x in reps if x["ok"] and x["ttft_s"]]
    rec = {"phase": "warm_streaming_ttft", "utc": now(), "fixture_tokens": c, "reps": reps,
           "median_ttft_s": sorted(x["ttft_s"] for x in ok)[len(ok)//2] if ok else None}
    save("ttft.json", rec)

PROMPTS = [
    "Explain in careful detail how a modern operating system kernel implements virtual memory: page tables, the TLB, page faults, demand paging, copy-on-write, and swapping.",
    "Write an original short story about a lighthouse keeper who discovers something unexpected washed up on the shore one winter morning.",
    "Write a complete Python implementation of a red-black tree with insert, delete, and search, including the rebalancing cases, with comments.",
    "Describe the full lifecycle of an HTTP request from browser to origin server and back, including DNS, TLS, proxies, and caching.",
    "Write a detailed essay on the causes and consequences of the 1973 oil shock, with specific countries and dates.",
]

def ladder_level(c, rep_tag):
    samples = []
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            samples.append({"t": now(), "gpu": smi()}); stop.wait(2)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    with ThreadPoolExecutor(max_workers=c) as ex:
        t0 = time.perf_counter()
        res = list(ex.map(lambda i: completion(f"[{rep_tag}-{i}] " + PROMPTS[i % len(PROMPTS)], LADDER_TOKENS, ignore_eos=True), range(c)))
        wall = time.perf_counter() - t0
    stop.set(); th.join(timeout=5)
    ok = [r for r in res if r["ok"]]
    toks = sum(r["usage"]["completion_tokens"] for r in ok)
    per = sorted(r["usage"]["completion_tokens"] / r["wall_s"] for r in ok)
    return {"concurrency": c, "requests": c, "succeeded": len(ok), "failed": c - len(ok),
            "success_rate": round(len(ok) / c, 3), "total_completion_tokens": toks, "wall_s": round(wall, 3),
            "aggregate_tok_s": round(toks / wall, 2) if wall > 0 else None,
            "all_exactly_400": all(r["usage"]["completion_tokens"] == LADDER_TOKENS for r in ok) and len(ok) == c,
            "per_request_tok_s": {"median": round(per[len(per)//2], 2) if per else None, "min": round(per[0], 2) if per else None,
                                  "max": round(per[-1], 2) if per else None, "all": [round(x, 2) for x in per]},
            "per_request_wall_s": [r["wall_s"] for r in res], "errors": [r["error"] for r in res if not r["ok"]],
            "finish_reasons": [r.get("finish_reason") for r in ok], "gpu_samples": samples}

def ladder(levels):
    rec = {"phase": "concurrency_ladder", "utc": now(), "max_tokens": LADDER_TOKENS, "sampling": "temperature 0, ignore_eos",
           "warmup_per_level": 1, "measured_reps_per_level": REPS, "levels": []}
    for c in levels:
        entry = {"concurrency": c, "reps": [], "warmup": None, "status": "running"}
        rec["levels"].append(entry); save("ladder.json", rec)
        w = ladder_level(c, f"warm-c{c}"); w.pop("gpu_samples", None); entry["warmup"] = w
        print(f"C{c} warmup: {w['aggregate_tok_s']} tok/s ok={w['succeeded']}/{c}", flush=True)
        if w["failed"] == c:
            entry["status"] = "FAIL_warmup"; save("ladder.json", rec); print(f"C{c} warmup failed; stopping ladder", flush=True); break
        for i in range(REPS):
            r = ladder_level(c, f"rep{i}-c{c}"); entry["reps"].append(r); save("ladder.json", rec)
            print(f"C{c} rep{i}: {r['aggregate_tok_s']} tok/s agg, per-req median {r['per_request_tok_s']['median']}, ok={r['succeeded']}/{c}, wall {r['wall_s']}s", flush=True)
            if r["failed"] == c:
                entry["status"] = "FAIL"; break
            time.sleep(2)
        if entry["status"] == "running":
            aggs = sorted(x["aggregate_tok_s"] for x in entry["reps"]); meds = sorted(x["per_request_tok_s"]["median"] for x in entry["reps"])
            entry["status"] = "PASS" if all(x["failed"] == 0 for x in entry["reps"]) else "PARTIAL"
            entry["summary"] = {"aggregate_tok_s_median": aggs[len(aggs)//2], "aggregate_tok_s_min": aggs[0], "aggregate_tok_s_max": aggs[-1],
                                "per_request_tok_s_median_of_medians": meds[len(meds)//2],
                                "success_rate": round(sum(x["succeeded"] for x in entry["reps"]) / (c * len(entry["reps"])), 3)}
        save("ladder.json", rec)
        if entry["status"] == "FAIL":
            print(f"C{c} failed; stopping ladder", flush=True); break
        time.sleep(3)

if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "gate": sys.exit(0 if gate() else 1)
    elif phase == "prefill": prefill()
    elif phase == "ttft": ttft()
    elif phase == "ladder": ladder([int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else LADDER)])
