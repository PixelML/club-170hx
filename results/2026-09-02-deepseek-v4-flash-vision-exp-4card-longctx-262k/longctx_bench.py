#!/usr/bin/env python3
"""Long-context bench for dsv4v at max-model-len 262144.
Reuses the fixture/tokenize/completion helpers and conventions from
club-170hx/results/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm/bench_harness.py
(greedy temp 0, final usage object for token counts, unique-prefix fixtures so nothing
prefix-caches, 1 warmup + 3 reps).

Phases:
  prefill_ladder   : max_tokens=1 at 2941 / 16k / 32k / 65k / 131k / 200k / 250k prompt tokens
  needle           : fact retrieval at 25/50/75% depth for 32k, 131k, largest-passing length
  decode_longctx   : C1 x3 greedy 400 tok after 131k and largest-passing prompt; C2 x3 at 32k
  vision_longctx   : one text+image request at 131k context
"""
import json, os, sys, time, uuid, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.error

URL = os.environ.get("DSV4_URL", "http://127.0.0.1:18099")
MODEL = os.environ.get("DSV4_MODEL_NAME", "dsv4v")
OUT = os.environ.get("DSV4_OUT", ".")
REPS = 3

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def has_tokenize():
    try:
        json.load(post("/tokenize", {"model": MODEL, "prompt": "hello"}, 30))
        return True
    except Exception:
        return False

def tokenize(text):
    r = json.load(post("/tokenize", {"model": MODEL, "prompt": text}, 300))
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

BASE = ("The following is a technical reference on distributed systems. Section {n}: consensus protocols. "
        "A replicated state machine applies a deterministic sequence of commands to identical copies of state on every node. "
        "Leader election chooses one node to order commands; followers acknowledge log entries; commitment requires a quorum. "
        "Failure detectors use heartbeats and timeouts; network partitions cause split votes; term numbers resolve stale leaders. ")

def build_fixture(target, prefix=""):
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

def completion(prompt, max_tokens, ignore_eos=False, timeout=1800, extra=None):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    if ignore_eos: body["ignore_eos"] = True
    if extra: body.update(extra)
    t0 = time.perf_counter()
    try:
        r = json.load(post("/v1/completions", body, timeout))
        dt = time.perf_counter() - t0
        return {"ok": True, "wall_s": round(dt, 4), "usage": r["usage"], "text": r["choices"][0]["text"],
                "finish_reason": r["choices"][0].get("finish_reason")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}"}
    except TimeoutError as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"TimeoutError: {e}"}
    except Exception as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"{type(e).__name__}: {e}"[:2000]}

def chat_completion(text, image_data_url, max_tokens, ignore_eos=False, timeout=1800):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}}]}]}
    if ignore_eos: body["ignore_eos"] = True
    t0 = time.perf_counter()
    try:
        r = json.load(post("/v1/chat/completions", body, timeout))
        dt = time.perf_counter() - t0
        return {"ok": True, "wall_s": round(dt, 4), "usage": r["usage"],
                "text": r["choices"][0]["message"].get("content") or "",
                "finish_reason": r["choices"][0].get("finish_reason")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}"}
    except Exception as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"{type(e).__name__}: {e}"[:2000]}

LADDER_LENGTHS = [2941, 16000, 32000, 65000, 131000, 200000, 250000]

def prefill_ladder():
    rec = {"phase": "prefill_ladder", "utc": now(), "reps_per_level": REPS, "max_tokens": 1, "levels": []}
    save("prefill_ladder.json", rec)
    for target in LADDER_LENGTHS:
        entry = {"target_prompt_tokens": target, "reps": [], "status": "running"}
        rec["levels"].append(entry); save("prefill_ladder.json", rec)
        try:
            build_t0 = time.perf_counter()
            base_text, base_c = build_fixture(target, "[base] ")
            print(f"level {target}: built fixture {base_c} tokens in {time.perf_counter()-build_t0:.1f}s", flush=True)
        except Exception as e:
            entry["status"] = "FAIL_BUILD"; entry["error"] = f"{type(e).__name__}: {e}"
            save("prefill_ladder.json", rec)
            print(f"level {target}: FIXTURE BUILD FAILED: {e}", flush=True)
            break
        failed = False
        for i in range(REPS):
            prefix = f"[run {uuid.uuid4().hex}] "
            text = prefix + base_text[len("[base] "):]
            t0 = time.perf_counter()
            r = completion(text, 1, timeout=900)
            r["wall_s_total"] = round(time.perf_counter() - t0, 4)
            pt = r.get("usage", {}).get("prompt_tokens")
            r["prefill_tok_s"] = round(pt / r["wall_s"], 2) if r["ok"] and pt else None
            r["gpu_after"] = smi()
            entry["reps"].append(r); save("prefill_ladder.json", rec)
            print(f"level {target} rep{i}: ok={r['ok']} wall={r['wall_s']}s tok/s={r.get('prefill_tok_s')} err={r.get('error')}", flush=True)
            if not r["ok"]:
                failed = True
                if "Timeout" in str(r.get("error","")) or r["wall_s"] > 900:
                    entry["status"] = "FAIL_TIMEOUT"
                elif "400" in str(r.get("error","")):
                    entry["status"] = "FAIL_400"
                elif "OOM" in str(r.get("error","")).upper() or "out of memory" in str(r.get("error","")).lower():
                    entry["status"] = "FAIL_OOM"
                else:
                    entry["status"] = "FAIL_OTHER"
                save("prefill_ladder.json", rec)
                break
        if not failed:
            ok = entry["reps"]
            entry["status"] = "PASS"
            entry["median_wall_s"] = sorted(x["wall_s"] for x in ok)[len(ok)//2]
            entry["median_prefill_tok_s"] = sorted(x["prefill_tok_s"] for x in ok)[len(ok)//2]
        save("prefill_ladder.json", rec)
        if entry["status"] != "PASS":
            print(f"level {target}: STOP LADDER ({entry['status']})", flush=True)
            break
    return rec

NEEDLE_FACT = "The vault code is 7391-XK."
NEEDLE_Q = "\n\nWhat is the vault code mentioned earlier in this document? Answer with just the code."

def build_needle_prompt(target_tokens, depth_frac, tag):
    prefix = f"[needle-{tag}] "
    body = ""
    n = 0
    insert_at_words = None
    while True:
        n += 1
        body += BASE.format(n=n)
        cur = tokenize(prefix + body)
        if cur >= target_tokens:
            break
    words = body.split(" ")
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        if tokenize(prefix + " ".join(words[:mid])) < target_tokens: lo = mid + 1
        else: hi = mid
    words = words[:lo]
    ins = int(len(words) * depth_frac)
    words = words[:ins] + [NEEDLE_FACT] + words[ins:]
    text = prefix + " ".join(words) + NEEDLE_Q
    return text

def needle_test(lengths):
    rec = {"phase": "needle_in_haystack", "utc": now(), "fact": NEEDLE_FACT, "depths": [0.25, 0.5, 0.75],
           "max_tokens": 32, "lengths": []}
    save("needle.json", rec)
    for target in lengths:
        entry = {"target_prompt_tokens": target, "depths": {}}
        for depth in [0.25, 0.5, 0.75]:
            tag = f"{target}-{depth}"
            try:
                prompt = build_needle_prompt(target, depth, tag)
            except Exception as e:
                entry["depths"][str(depth)] = {"ok": False, "error": f"build failed: {e}"}
                continue
            r = completion(prompt, 32, timeout=900)
            passed = r.get("ok") and "7391-xk" in (r.get("text") or "").lower().replace(" ", "")
            entry["depths"][str(depth)] = {"ok": r.get("ok"), "response_text": r.get("text"), "pass": bool(passed),
                                            "wall_s": r.get("wall_s"), "usage": r.get("usage"), "error": r.get("error")}
            print(f"needle {target}@{depth}: pass={passed} resp={r.get('text','')[:60]!r}", flush=True)
        rec["lengths"].append(entry); save("needle.json", rec)
    return rec

def decode_longctx(prompt_lengths_c1, length_c2):
    rec = {"phase": "decode_with_long_context", "utc": now(), "max_tokens": 400, "levels": []}
    save("decode_longctx.json", rec)
    for target in prompt_lengths_c1:
        prefix_text, _ = build_fixture(target, f"[decodeC1-{target}-{uuid.uuid4().hex[:8]}] ")
        reps = []
        for i in range(REPS):
            r = completion(prefix_text, 400, ignore_eos=True, timeout=900)
            per_tok_s = None
            if r.get("ok") and r.get("usage"):
                per_tok_s = round(r["usage"]["completion_tokens"] / r["wall_s"], 2)
            r["per_request_tok_s"] = per_tok_s
            reps.append(r)
            print(f"decode C1 @{target} rep{i}: ok={r['ok']} tok/s={per_tok_s} wall={r['wall_s']}", flush=True)
        rec["levels"].append({"concurrency": 1, "target_prompt_tokens": target, "reps": reps})
        save("decode_longctx.json", rec)
    # C2 at 32k
    reps_c2 = []
    for i in range(REPS):
        texts = [build_fixture(length_c2, f"[decodeC2-{length_c2}-{i}-{j}-{uuid.uuid4().hex[:6]}] ")[0] for j in range(2)]
        with ThreadPoolExecutor(max_workers=2) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(lambda t: completion(t, 400, ignore_eos=True, timeout=900), texts))
            wall = time.perf_counter() - t0
        ok = [r for r in res if r["ok"]]
        toks = sum(r["usage"]["completion_tokens"] for r in ok) if ok else 0
        agg = round(toks / wall, 2) if wall > 0 else None
        reps_c2.append({"wall_s": round(wall, 3), "aggregate_tok_s": agg, "results": res})
        print(f"decode C2 @{length_c2} rep{i}: agg_tok/s={agg} ok={len(ok)}/2", flush=True)
    rec["levels"].append({"concurrency": 2, "target_prompt_tokens": length_c2, "reps": reps_c2})
    save("decode_longctx.json", rec)
    return rec

def vision_longctx(target, image_data_url):
    prefix_text, c = build_fixture(target - 50, f"[vision-longctx-{uuid.uuid4().hex[:8]}] ")
    question = "\n\nBased on the text above and the attached image, name the two colors the image blends between, then summarize the document's topic in one sentence."
    full_text = prefix_text + question
    r = chat_completion(full_text, image_data_url, 64, timeout=900)
    rec = {"phase": "vision_longctx", "utc": now(), "target_prompt_tokens": target, "text_fixture_tokens": c, "result": r}
    save("vision_longctx.json", rec)
    print(f"vision @{target}: ok={r['ok']} usage={r.get('usage')} text={r.get('text','')[:120]!r}", flush=True)
    return rec

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("tokenize endpoint present:", has_tokenize(), flush=True)
    if phase in ("all", "prefill"):
        prefill_ladder()
    if phase in ("all", "needle"):
        pass  # driven from run script with dynamic lengths
    if phase in ("all", "decode"):
        pass
