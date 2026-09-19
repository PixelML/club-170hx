#!/usr/bin/env python3
"""Chart source for 2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-bench-pp.{png,svg}.

Regenerate:
    python3 assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-bench-pp.py

Reads the committed receipts
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/llama-bench-*.txt
and writes the PNG and SVG beside this script. No network, no GPU.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REC = os.path.join(REPO, "results", "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp", "receipts")
STEM = os.path.join(HERE, "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-bench-pp")

REFS = {
    "A100 SXM 80 GB\n(community-reported)": 1328,
    "H100 SXM 80 GB\n(community-reported)": 2830,
    "RTX 5090 32 GB\n(community-reported)": 3893,
}


def bench_pp(fname):
    with open(os.path.join(REC, fname)) as fh:
        for line in fh:
            if "pp512" in line:
                return float(line.rstrip().split("|")[-2].split()[0])
    raise SystemExit(f"no pp512 row in {fname}")


def main():
    ours = [
        ("PQ2_0\n8 threads, ub 512", bench_pp("llama-bench-pq2_0.txt"), "#1f6feb"),
        ("PTQ1_0\n8 threads, ub 512", bench_pp("llama-bench-ptq1_0.txt"), "#0ca678"),
        ("PQ2_0\n8 threads, ub 2048", bench_pp("llama-bench-pq2_0-ub2048.txt"), "#e8590c"),
    ]
    labels = [n for n, _, _ in ours] + list(REFS)
    values = [v for _, v, _ in ours] + list(REFS.values())
    colors = [c for _, _, c in ours] + ["#adb5bd"] * len(REFS)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    for rect, val in zip(bars, values):
        ax.annotate(f"{val:,.0f}", xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("llama-bench pp512 (tokens/second)")
    ax.set_title("Bonsai 2 27B ternary prompt processing by packing and flags, 1x CMP 170HX (SM80, 180 W cap)\n"
                 "llama.cpp PrismML fork b10685, -ngl 99, flash attention")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print("saved", STEM + ".png")


if __name__ == "__main__":
    main()
