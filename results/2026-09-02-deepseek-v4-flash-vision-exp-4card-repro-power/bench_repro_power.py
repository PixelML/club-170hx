#!/usr/bin/env python3
"""Reproducibility + power-arm benchmark for DeepSeek-V4-Flash-Vision-Exp,
4-card CMP 170HX, vLLM Path 3 (SM80). Text-only prompts, thinking off.

Two independent measurement protocols, run side by side at each concurrency
level:

  ours  : greedy, ignore_eos, exactly 400 completion tokens, non-streaming
          /v1/completions over our 2,941-token fixture prompt. Tokens taken
          from the final `usage` object. aggregate_tok_s = total_completion_
          tokens / synchronized wall time (time for the whole concurrent
          batch to finish).

  mia   : greedy, ignore_eos, streaming /v1/completions. Each request uses a
          fresh 256-token prompt with a unique cold prefix (uuid4). Forces a
          128-token decode window. Per-stream decode tok/s is measured AFTER
          the first token (excludes TTFT): (total_tokens - 1) / (t_last -
          t_first). TTFT is reported separately (median). aggregate_tok_s =
          sum of per-stream decode tok/s (not tokens/wall). Median across
          reps, thinking off (no reasoning params sent).

Levels: C1, C2, C4, C8, C16. Reps: warmup(1) + 5 measured at C1/C2, warmup(1)
+ 3 measured at C4/C8, warmup(1) + 1 attempt at C16.

Also samples nvidia-smi power/clocks/temps at 1 Hz for the whole run and
scrapes /metrics before/after each level for DSpark accept/draft counters.
"""
import json, os, sys, time, uuid, subprocess, threading, csv
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.error

URL = os.environ.get("DSV4_URL", "http://127.0.0.1:18099")
MODEL = os.environ.get("DSV4_MODEL_NAME", "deepseek-v4-flash-vision-exp")
OUT = os.environ.get("DSV4_OUT", ".")
ARM = os.environ.get("DSV4_ARM", "180w")

OURS_PROMPT_TOKENS = 2941
OURS_COMPLETION_TOKENS = 400
MIA_PROMPT_TOKENS = 256
MIA_COMPLETION_TOKENS = 128

LEVELS = [1, 2, 4, 8, 16]
REPS_BY_LEVEL = {1: 5, 2: 5, 4: 3, 8: 3, 16: 1}

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def tokenize(text):
    r = json.load(post("/tokenize", {"model": MODEL, "prompt": text}, 120))
    return r["count"]

def save(name, obj):
    p = os.path.join(OUT, name)
    with open(p + ".tmp", "w") as f: json.dump(obj, f, indent=2)
    os.replace(p + ".tmp", p)
    print("saved", p, flush=True)

def smi_row():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,power.draw,power.limit,clocks.sm,temperature.gpu,temperature.memory,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True).strip().split("\n")
        return [ [x.strip() for x in row.split(",")] for row in out ]
    except Exception as e:
        return [["ERR", str(e)]]

def scrape_metrics():
    try:
        txt = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    except Exception as e:
        return {"error": str(e)}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"): continue
        for key in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                    "vllm:spec_decode_num_drafts_total"):
            if line.startswith(key):
                try:
                    val = float(line.rsplit(" ", 1)[1])
                    out[key] = out.get(key, 0.0) + val
                except Exception:
                    pass
    return out

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

# --- shared fixed 2941-token fixture for "ours" protocol (same prompt reused each rep, unique launch-uuid prefix per request so it's a distinct request id but same length) ---
_OURS_FIXTURE = None
def ours_fixture():
    global _OURS_FIXTURE
    if _OURS_FIXTURE is None:
        text, c = build_fixture(OURS_PROMPT_TOKENS, "[ours-fixture] ")
        _OURS_FIXTURE = (text, c)
        print("ours fixture built:", c, "tokens", flush=True)
    return _OURS_FIXTURE

PROMPTS = [
    "Explain in careful detail how a modern operating system kernel implements virtual memory: page tables, the TLB, page faults, demand paging, copy-on-write, and swapping.",
    "Write an original short story about a lighthouse keeper who discovers something unexpected washed up on the shore one winter morning.",
    "Write a complete Python implementation of a red-black tree with insert, delete, and search, including the rebalancing cases, with comments.",
    "Describe the full lifecycle of an HTTP request from browser to origin server and back, including DNS, TLS, proxies, and caching.",
    "Write a detailed essay on the causes and consequences of the 1973 oil shock, with specific countries and dates.",
]

# ---------------- ours-short protocol: exact original condition that produced the 163 tok/s c=1 figure ----------------
def ours_short_level(c):
    m0 = scrape_metrics()
    with ThreadPoolExecutor(max_workers=c) as ex:
        t0 = time.perf_counter()
        res = list(ex.map(lambda i: ours_request(f"[req{uuid.uuid4().hex[:8]}-{i}] " + PROMPTS[i % len(PROMPTS)]), range(c)))
        wall = time.perf_counter() - t0
    m1 = scrape_metrics()
    ok = [r for r in res if r["ok"]]
    toks = sum(r["usage"]["completion_tokens"] for r in ok)
    per = sorted(r["usage"]["completion_tokens"] / r["wall_s"] for r in ok)
    dspark = {k: round(m1.get(k, 0) - m0.get(k, 0), 1) for k in m1} if "error" not in m1 and "error" not in m0 else {"error": "metrics unavailable"}
    return {"concurrency": c, "requests": c, "succeeded": len(ok), "failed": c - len(ok),
            "success_rate": round(len(ok) / c, 3), "total_completion_tokens": toks, "wall_s": round(wall, 3),
            "aggregate_tok_s": round(toks / wall, 2) if wall > 0 else None,
            "all_exactly_target": all(r["usage"]["completion_tokens"] == OURS_COMPLETION_TOKENS for r in ok) and len(ok) == c,
            "per_request_tok_s": {"median": round(per[len(per)//2], 2) if per else None,
                                   "min": round(per[0], 2) if per else None, "max": round(per[-1], 2) if per else None},
            "errors": [r["error"] for r in res if not r["ok"]], "dspark_delta": dspark}

# ---------------- ours protocol (2,941-token fixture) ----------------
def ours_request(prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": OURS_COMPLETION_TOKENS, "temperature": 0, "ignore_eos": True}
    t0 = time.perf_counter()
    try:
        r = json.load(post("/v1/completions", body, 1800))
        dt = time.perf_counter() - t0
        return {"ok": True, "wall_s": round(dt, 4), "usage": r["usage"], "finish_reason": r["choices"][0].get("finish_reason")}
    except urllib.error.HTTPError as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"}
    except Exception as e:
        return {"ok": False, "wall_s": round(time.perf_counter() - t0, 4), "error": f"{type(e).__name__}: {e}"[:500]}

def ours_level(c):
    text, _ = ours_fixture()
    m0 = scrape_metrics()
    with ThreadPoolExecutor(max_workers=c) as ex:
        t0 = time.perf_counter()
        res = list(ex.map(lambda i: ours_request(f"[req{uuid.uuid4().hex[:8]}] " + text), range(c)))
        wall = time.perf_counter() - t0
    m1 = scrape_metrics()
    ok = [r for r in res if r["ok"]]
    toks = sum(r["usage"]["completion_tokens"] for r in ok)
    per = sorted(r["usage"]["completion_tokens"] / r["wall_s"] for r in ok)
    dspark = {k: round(m1.get(k, 0) - m0.get(k, 0), 1) for k in m1} if "error" not in m1 and "error" not in m0 else {"error": "metrics unavailable"}
    return {"concurrency": c, "requests": c, "succeeded": len(ok), "failed": c - len(ok),
            "success_rate": round(len(ok) / c, 3), "total_completion_tokens": toks, "wall_s": round(wall, 3),
            "aggregate_tok_s": round(toks / wall, 2) if wall > 0 else None,
            "all_exactly_target": all(r["usage"]["completion_tokens"] == OURS_COMPLETION_TOKENS for r in ok) and len(ok) == c,
            "per_request_tok_s": {"median": round(per[len(per)//2], 2) if per else None,
                                   "min": round(per[0], 2) if per else None, "max": round(per[-1], 2) if per else None},
            "errors": [r["error"] for r in res if not r["ok"]], "dspark_delta": dspark}

# ---------------- mia protocol ----------------
def mia_unique_prompt():
    text, c = build_fixture(MIA_PROMPT_TOKENS, f"[cold-{uuid.uuid4().hex}] ")
    return text

def mia_request(prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": MIA_COMPLETION_TOKENS, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; t_last = None; ntok = 0; usage = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:"): continue
                payload = line[5:].strip()
                if payload == "[DONE]": break
                d = json.loads(payload)
                if d.get("choices") and d["choices"][0].get("text") is not None and d["choices"][0]["text"] != "":
                    t_now = time.perf_counter()
                    if t_first is None: t_first = t_now
                    t_last = t_now
                    ntok += 1
                if d.get("usage"): usage = d["usage"]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:500]
    total = time.perf_counter() - t0
    ttft = round(t_first - t0, 4) if t_first else None
    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    ctoks = usage["completion_tokens"] if usage else ntok
    decode_tok_s = round((ctoks - 1) / decode_s, 2) if decode_s and ctoks > 1 else None
    return {"ok": err is None and usage is not None, "ttft_s": ttft, "total_s": round(total, 4),
            "decode_tok_s": decode_tok_s, "completion_tokens": ctoks, "usage": usage, "error": err}

def mia_level(c):
    prompts = [mia_unique_prompt() for _ in range(c)]
    m0 = scrape_metrics()
    with ThreadPoolExecutor(max_workers=c) as ex:
        t0 = time.perf_counter()
        res = list(ex.map(mia_request, prompts))
        wall = time.perf_counter() - t0
    m1 = scrape_metrics()
    ok = [r for r in res if r["ok"]]
    dr = sorted(r["decode_tok_s"] for r in ok if r["decode_tok_s"])
    ttfts = sorted(r["ttft_s"] for r in ok if r["ttft_s"])
    agg = round(sum(dr), 2) if dr else None
    dspark = {k: round(m1.get(k, 0) - m0.get(k, 0), 1) for k in m1} if "error" not in m1 and "error" not in m0 else {"error": "metrics unavailable"}
    return {"concurrency": c, "requests": c, "succeeded": len(ok), "failed": c - len(ok),
            "success_rate": round(len(ok) / c, 3), "wall_s": round(wall, 3),
            "aggregate_decode_tok_s": agg,
            "per_stream_decode_tok_s": {"median": round(dr[len(dr)//2], 2) if dr else None,
                                         "min": round(dr[0], 2) if dr else None, "max": round(dr[-1], 2) if dr else None},
            "ttft_s": {"median": round(ttfts[len(ttfts)//2], 4) if ttfts else None,
                       "min": round(ttfts[0], 4) if ttfts else None, "max": round(ttfts[-1], 4) if ttfts else None},
            "errors": [r["error"] for r in res if not r["ok"]], "dspark_delta": dspark}

# ---------------- telemetry ----------------
def telemetry_writer(stop_evt, csv_path):
    header_written = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not header_written:
            w.writerow(["t_utc", "arm", "gpu_index", "power_draw_w", "power_limit_w", "clocks_sm_mhz", "temp_gpu_c", "temp_mem_c", "util_pct"])
        while not stop_evt.is_set():
            t = now()
            for row in smi_row():
                if row[0] == "ERR": continue
                w.writerow([t, ARM] + row)
            f.flush()
            stop_evt.wait(1.0)

# ---------------- driver ----------------
FNS = {"ours_short": ours_short_level, "ours_long": ours_level, "mia": mia_level}

def run_level(protocol, c, reps_n):
    fn = FNS[protocol]
    warm = fn(c)
    print(f"[{protocol}] C{c} warmup: {warm}", flush=True)
    reps = []
    for i in range(reps_n):
        r = fn(c)
        reps.append(r)
        print(f"[{protocol}] C{c} rep{i}: succ={r['succeeded']}/{c} " +
              (f"agg_decode={r.get('aggregate_decode_tok_s')}" if protocol == "mia" else f"agg={r.get('aggregate_tok_s')}"),
              flush=True)
        if r["failed"] == c:
            print(f"[{protocol}] C{c} rep{i} total failure, stopping level", flush=True)
            break
        time.sleep(1.5)
    return {"concurrency": c, "warmup": warm, "reps": reps}

def main():
    proto_filter = sys.argv[1] if len(sys.argv) > 1 else "both"
    levels_arg = sys.argv[2] if len(sys.argv) > 2 else None
    levels = [int(x) for x in levels_arg.split(",")] if levels_arg else LEVELS

    stop_evt = threading.Event()
    telem_path = os.path.join(OUT, f"telemetry-1s-{ARM}.csv")
    th = threading.Thread(target=telemetry_writer, args=(stop_evt, telem_path), daemon=True)
    th.start()

    rec = {"phase": "repro_power_ladder", "arm": ARM, "utc_start": now(), "levels_requested": levels,
           "reps_by_level": REPS_BY_LEVEL, "ours_short": [], "ours_long": [], "mia": []}
    save(f"repro_ladder_{ARM}.json", rec)

    protos = ["ours_short", "ours_long", "mia"] if proto_filter == "both" else [proto_filter]
    wedged = False
    for c in levels:
        if wedged: break
        reps_n = REPS_BY_LEVEL[c]
        for protocol in protos:
            try:
                entry = run_level(protocol, c, reps_n)
            except Exception as e:
                entry = {"concurrency": c, "error": f"{type(e).__name__}: {e}"}
                print(f"[{protocol}] C{c} EXCEPTION: {e}", flush=True)
                if c == 16:
                    wedged = True
            rec[protocol].append(entry)
            save(f"repro_ladder_{ARM}.json", rec)
        time.sleep(2)

    rec["utc_end"] = now()
    rec["wedged_at_c16"] = wedged
    save(f"repro_ladder_{ARM}.json", rec)
    stop_evt.set()
    th.join(timeout=5)
    print("DONE", ARM, flush=True)

if __name__ == "__main__":
    main()
