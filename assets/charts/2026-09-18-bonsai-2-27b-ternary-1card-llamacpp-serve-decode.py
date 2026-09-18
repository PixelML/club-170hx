#!/usr/bin/env python3
"""Chart source for 2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-serve-decode.{png,svg}.

Regenerate:
    python3 assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-serve-decode.py

Reads the committed receipts
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/serve-pq2_0.jsonl
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/serve-ptq1_0.jsonl
and writes the PNG and SVG beside this script. No network, no GPU.
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REC = os.path.join(REPO, "results", "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp", "receipts")
STEM = os.path.join(HERE, "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-serve-decode")

PACKS = [("serve-pq2_0.jsonl", "PQ2_0", "#1f6feb"), ("serve-ptq1_0.jsonl", "PTQ1_0", "#0ca678")]
COHORTS = [("decode256", "decode, 256-token cohort"), ("decode900", "decode, 900-token cohort")]


def med(rows, tag_prefix, key):
    vals = [r[key] for r in rows if r["tag"].startswith(tag_prefix) and r.get(key)]
    return st.median(vals) if vals else None


def main():
    data = {}
    for fname, _, _ in PACKS:
        with open(os.path.join(REC, fname)) as fh:
            data[fname] = [json.loads(l) for l in fh if l.strip()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.35
    for ax, (cohort, title) in zip(axes, COHORTS):
        for k, (fname, label, color) in enumerate(PACKS):
            rows = data[fname]
            warm = med(rows, f"{cohort}-warmup", "decode_tok_s")
            samps = [r["decode_tok_s"] for r in rows
                     if r["tag"].startswith(f"{cohort}-sample") and r.get("decode_tok_s")]
            if warm is None and not samps:
                continue
            x = k * width
            if warm is not None:
                ax.bar(x, warm, width * 0.9, color=color, alpha=0.35)
                ax.annotate(f"{warm:.1f}", xy=(x, warm), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8, alpha=0.8)
            if samps:
                m = st.median(samps)
                ax.bar(x + width / 2, m, width * 0.9, color=color, label=f"{label} (median of {len(samps)})")
                ax.annotate(f"{m:.1f}", xy=(x + width / 2, m), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("decode tok/s (single stream, usage-counted)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Bonsai 2 27B served decode, 1x CMP 170HX (SM80, 180 W cap), greedy, streaming\n"
                 "light bars: warmup rep; solid: median of measured reps", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print("saved", STEM + ".png")


if __name__ == "__main__":
    main()
