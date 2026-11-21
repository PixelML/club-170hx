"""Load every receipt of this experiment into plain Python structures (no third-party deps)."""
import csv, json, re, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "receipts"

# engine label in receipts -> (display name, card, card copy bandwidth GB/s)
ENGINES = {
    "llama-b12600-iq4nl": ("llama.cpp b12600, IQ4_NL GGUF, 4 slots", 1),
    "llama-b12600-iq4nl-ub2048-np1": ("llama.cpp b12600, IQ4_NL GGUF, 1 slot, ub 2048", 1),
    "vllm-sm80p-base": ("vLLM + sm80 patch, no speculation (slow-HBM card)", 0),
    "vllm-sm80p-base-gpu3": ("vLLM + sm80 patch, no speculation", 3),
    "vllm-sm80p-dspark7-greedydraft": ("vLLM + sm80 patch + DSpark k=7, greedy draft", 1),
    "vllm-sm80p-dspark7": ("vLLM + sm80 patch + DSpark k=7, probabilistic draft", 2),
}
FILES = ["llama-iq4", "llama-ub2048", "vllm-base", "vllm-base-g3", "vllm-dspark7greedy", "vllm-dspark7"]


def records():
    out = []
    for f in FILES:
        out += [json.loads(l) for l in open(R / f"{f}.jsonl")]
    return out


def table():
    """{(engine, suite, case, temperature): dict(median, min, max, n, ttft, prompt_tokens, prefill)}"""
    g = {}
    for r in records():
        g.setdefault((r["engine"], r["suite"], r["case"], r.get("temperature")), []).append(r)
    t = {}
    for k, v in g.items():
        d = [x["decode_tok_s"] for x in v]
        t[k] = dict(n=len(v), median=round(st.median(d), 1), min=min(d), max=max(d),
                    ttft=round(st.median([x["ttft_s"] for x in v]), 3),
                    prompt_tokens=v[0].get("prompt_tokens"),
                    prefill=round(st.median([x["prefill_tok_s"] for x in v]), 0) if "prefill_tok_s" in v[0] else None,
                    completion=st.median([x["completion_tokens"] for x in v]))
    return t


def spec_metrics(name):
    m = {}
    for line in open(R / name):
        k = re.match(r"vllm:(\w+)\{([^}]*)\} ([0-9.e+]+)", line)
        if k and not k.group(1).endswith("_created"):
            pos = re.search(r'position="(\d+)"', k.group(2))
            m[k.group(1) + (f"[{pos.group(1)}]" if pos else "")] = float(k.group(3))
    return m


def gsm8k(name):
    rs = [json.loads(l) for l in open(R / name)]
    return sum(r["correct"] for r in rs), len(rs), sorted(r["idx"] for r in rs if not r["correct"])


def telemetry():
    rows = []
    for f in sorted((R / "telemetry").glob("guard-*.csv")):
        s = [l.split(",") for l in open(f) if re.match(r"^\d+,", l)]
        if not s:
            continue
        rows.append(dict(file=f.name, samples=len(s), core=max(float(x[2]) for x in s),
                         mem=max(float(x[3]) for x in s), power=max(float(x[4]) for x in s),
                         stops=sum(1 for l in open(f) if l.startswith("STOP"))))
    return rows


def hbm():
    return {l.split()[0]: float(l.split("copy_bw_GBps=")[1]) for l in open(R / "hbm-copy-bw.txt") if l.startswith("gpu")}
