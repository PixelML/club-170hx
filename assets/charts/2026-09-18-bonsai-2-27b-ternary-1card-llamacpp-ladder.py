#!/usr/bin/env python3
"""Chart source for 2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-ladder.{png,svg}.

Regenerate:
    python3 assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-ladder.py

Reads the committed receipt
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/serve-pq2_0.jsonl
and writes the PNG and SVG beside this script. No network, no GPU.
"""
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REC = os.path.join(REPO, "results", "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp", "receipts")
STEM = os.path.join(HERE, "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-ladder")


def main():
    with open(os.path.join(REC, "serve-pq2_0.jsonl")) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]

    by_c = {}
    for r in rows:
        m = re.fullmatch(r"ladder-c(\d+)-r(\d+)-SUMMARY", r["tag"])
        if m:
            by_c.setdefault(int(m.group(1)), []).append(r["decode_tok_s"])
    cs = sorted(by_c)
    agg = [st.median(by_c[c]) for c in cs]
    per = [a / c for a, c in zip(agg, cs)]

    # single-stream reference from the decode256 measured samples
    d256 = [r["decode_tok_s"] for r in rows
            if r["tag"].startswith("decode256-sample") and r.get("decode_tok_s")]
    c1 = st.median(d256) if d256 else None

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(cs, agg, "o-", color="#1f6feb", label="aggregate tok/s")
    ax.plot(cs, per, "s--", color="#0ca678", label="per-stream tok/s (aggregate / c)")
    if c1:
        ax.axhline(c1, color="#adb5bd", ls=":", label=f"c=1 decode256 median ({c1:.1f} tok/s)")
    for x, y in zip(cs, agg):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xticks(cs)
    ax.set_xlabel("concurrency (requested; server clamps to a single slot for this arch)")
    ax.set_ylabel("tok/s")
    ax.set_title("Bonsai 2 27B (PQ2_0) concurrency ladder, 1x CMP 170HX (SM80, 180 W cap)\n"
                 "the fork clamps Bonsai 2 to n_slots=1 (despite -np 8 and its env override), so requests queue:\n"
                 "aggregate stays at the single-stream rate while per-request TTFT grows ~5.2 s per queue position")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print("saved", STEM + ".png")


if __name__ == "__main__":
    main()
