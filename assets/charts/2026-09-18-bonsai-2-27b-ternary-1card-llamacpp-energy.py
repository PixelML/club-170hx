#!/usr/bin/env python3
"""Chart source for ...-energy.{png,svg} — energy per generated token.

Regenerate:
    python3 assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-energy.py

Reads the committed receipts
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/nvidia-serving.csv
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/serve-pq2_0.jsonl
and writes the PNG and SVG beside this script. No network, no GPU.

Method (indicative, same caveat as the GLM-5.3 result): J/tok = mean board
power over the serving telemetry window divided by the median served decode
rate. nvidia-smi board power includes the HBM rail; it is not integrated
energy and not a lower bound.
"""
import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REC = os.path.join(REPO, "results", "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp", "receipts")
STEM = os.path.join(HERE, "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-energy")

# community-reported from the model card throughput table (llama-bench tg128)
REFS = [
    ("RTX 5090 32 GB", 1.95),
    ("H100 SXM 80 GB", 2.69),
    ("A100 SXM 80 GB", 3.43),
]


def main():
    powers = []
    with open(os.path.join(REC, "nvidia-serving.csv")) as fh:
        for row in csv.DictReader(fh):
            powers.append(float(row[" power.draw [W]"].split()[0]))
    mean_w = st.mean(powers)

    with open(os.path.join(REC, "serve-pq2_0.jsonl")) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    dec = [r["decode_tok_s"] for r in rows
           if r["tag"].startswith("decode256-sample") and r.get("decode_tok_s")]
    tps = st.median(dec)
    j_per_tok = mean_w / tps

    labels = [f"CMP 170HX 180 W\n(this run, {mean_w:.0f} W avg / {tps:.1f} tok/s)"] + \
             [n + "\n(community-reported)" for n, _ in REFS]
    values = [j_per_tok] + [v for _, v in REFS]
    colors = ["#1f6feb"] + ["#adb5bd"] * len(REFS)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    for rect, val in zip(bars, values):
        ax.annotate(f"{val:.2f}", xy=(rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("joules per generated token (board power / tok/s)")
    ax.set_title("Bonsai 2 27B (PQ2_0) decode energy per token, 1x CMP 170HX (SM80, 180 W cap)\n"
                 "ours: mean 1 Hz board power over the serving window / median served decode rate — indicative")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print(f"saved {STEM}.png  (mean_w={mean_w:.1f} W, tps={tps:.2f}, J/tok={j_per_tok:.2f})")


if __name__ == "__main__":
    main()
