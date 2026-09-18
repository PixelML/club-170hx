#!/usr/bin/env python3
"""Chart source for ...-packing-tradeoff.{png,svg} — bpw vs decode across platforms.

Regenerate:
    python3 assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-packing-tradeoff.py

Reads the committed receipts
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/llama-bench-pq2_0.txt
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/llama-bench-ptq1_0.txt
and writes the PNG and SVG beside this script. No network, no GPU.

Upstream points are community-reported (model card throughput table).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REC = os.path.join(REPO, "results", "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp", "receipts")
STEM = os.path.join(HERE, "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-packing-tradeoff")

PQ2, PTQ = 2.13, 1.75  # true bits per weight


def bench_tg(fname):
    with open(os.path.join(REC, fname)) as fh:
        for line in fh:
            if "tg128" in line:
                return float(line.rstrip().split("|")[-2].split()[0])
    raise SystemExit(f"no tg128 row in {fname}")


def main():
    ours = [
        ("CMP 170HX 180 W (this run, measured)", [(PQ2, bench_tg("llama-bench-pq2_0.txt")),
                                                  (PTQ, bench_tg("llama-bench-ptq1_0.txt"))],
         "o-", "#1f6feb"),
        ("RTX 5090 32 GB (community-reported)", [(PQ2, 129.9), (PTQ, 120.5)], "s--", "#adb5bd"),
        ("H100 SXM 80 GB (community-reported)", [(PQ2, 113.9), (PTQ, 86.9)], "^--", "#adb5bd"),
        ("A100 SXM 80 GB (community-reported)", [(PQ2, 73.9), (PTQ, 54.7)], "v--", "#adb5bd"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, pts, style, color in ours:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, style, color=color, label=name)
        for x, y in pts:
            ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(5, -3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("true bits per weight (2.13 = PQ2_0, 1.75 = PTQ1_0)")
    ax.set_ylabel("llama-bench tg128 (tokens/second, batch 1)")
    ax.set_xticks([PQ2, PTQ])
    ax.set_xticklabels(["PQ2_0 (2.13 bpw, 6.7 GiB)", "PTQ1_0 (1.75 bpw, 5.5 GiB)"])
    ax.set_title("Bonsai 2 27B: the two ternary packings trade footprint against unpack cost\n"
                 "dense-trit PTQ1_0 loses on A100-class parts where decode is instruction-throughput-bound")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print("saved", STEM + ".png")


if __name__ == "__main__":
    main()
