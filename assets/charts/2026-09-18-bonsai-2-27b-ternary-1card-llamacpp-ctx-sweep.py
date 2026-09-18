#!/usr/bin/env python3
"""Chart source for 2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-ctx-sweep.{png,svg}.

Regenerate:
    python3 assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-ctx-sweep.py

Reads the committed receipt
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/ctx-sweep.json
and writes the PNG and SVG beside this script. No network, no GPU.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REC = os.path.join(REPO, "results", "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp", "receipts")
STEM = os.path.join(HERE, "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-ctx-sweep")


def main():
    with open(os.path.join(REC, "ctx-sweep.json")) as fh:
        doc = json.load(fh)

    xs = [r["prompt_tokens"] for r in doc["results"]]
    dec = [r["median_decode_tok_s"] for r in doc["results"]]
    pre = [r["median_prefill_tok_s"] for r in doc["results"]]
    ttft = [r["median_ttft_s"] * 1000 for r in doc["results"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(xs, dec, "o-", color="#1f6feb", label="decode tok/s")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{x:,}" for x in xs], fontsize=8)
    ax1.set_xlabel("prompt tokens (uncached)")
    ax1.set_ylabel("decode tok/s, 128-token window")
    ax1.set_title("Decode vs filled context")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    for x, y in zip(xs, dec):
        ax1.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=8)

    ax2.plot(xs, pre, "s-", color="#e8590c", label="prefill tok/s")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{x:,}" for x in xs], fontsize=8)
    ax2.set_xlabel("prompt tokens (uncached)")
    ax2.set_ylabel("prefill tok/s")
    ax2b = ax2.twinx()
    ax2b.plot(xs, ttft, "^--", color="#9c36b5", label="TTFT ms")
    ax2b.set_ylabel("TTFT (ms)", color="#9c36b5")
    ax2.set_title("Prefill and TTFT vs prompt length")
    ax2.grid(True, alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.suptitle("Bonsai 2 27B (PQ2_0) context sweep, llama-server, 1x CMP 170HX (SM80, 180 W cap)\n"
                 "uncached prefill (cache_prompt false), 128 output tokens, median of 3, ignore_eos", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print("saved", STEM + ".png")


if __name__ == "__main__":
    main()
