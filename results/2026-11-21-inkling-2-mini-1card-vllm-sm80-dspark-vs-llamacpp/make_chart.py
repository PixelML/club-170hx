"""Build assets/charts/<experiment>.png from the receipts. Run from the repo root."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize as S

EXP = HERE.name
OUT = HERE.parents[1] / "assets" / "charts" / f"{EXP}.png"
t = S.table()
ARMS = [("llama-b12600-iq4nl", "llama.cpp IQ4_NL", "#9aa0a6"),
        ("vllm-sm80p-base-gpu3", "vLLM sm80, no spec", "#5b8def"),
        ("vllm-sm80p-dspark7-greedydraft", "vLLM sm80 + DSpark\n(greedy draft)", "#f2a33a"),
        ("vllm-sm80p-dspark7", "vLLM sm80 + DSpark\n(probabilistic draft)", "#1f9d55")]
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
cases = ["code", "struct", "prose"]
w = 0.2
for i, (e, lab, c) in enumerate(ARMS):
    ys = [t[(e, "c1", k, 0.0)]["median"] for k in cases]
    xs = [j + (i - 1.5) * w for j in range(len(cases))]
    bars = a.bar(xs, ys, w, color=c, label=lab.replace("\n", " "))
    for x, y in zip(xs, ys):
        a.text(x, y + 2, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
a.set_xticks(range(len(cases)), ["code (LRU cache)", "structured JSON", "prose essay"])
a.set_ylabel("decode tok/s, 1 user, temperature 0 (median of 3)")
a.set_title("Single user, 768 output tokens")
a.legend(fontsize=8, loc="upper left", ncol=2)
a.set_ylim(0, 250)
for e, lab, c in ARMS:
    cs = sorted({k[2] for k in t if k[0] == e and k[1] == "conc"}, key=lambda s: int(s[1:]))
    xs = [int(s[1:]) for s in cs]
    ys = [t[(e, "conc", s, 1.0)]["median"] for s in cs]
    b.plot(xs, ys, "o-", color=c, label=lab.replace("\n", " "))
    b.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -3), fontsize=8)
b.set_xscale("log", base=2)
b.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
b.set_xlabel("concurrent users (temperature 1)")
b.set_ylabel("aggregate output tok/s")
b.set_title("Throughput vs concurrency")
b.grid(alpha=0.3)
fig.suptitle("Inkling-2-Mini on 1x CMP 170HX (SM80, 140 W) — measured 2026-11-21", fontsize=12)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130)
print(OUT)
