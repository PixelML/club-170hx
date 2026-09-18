#!/usr/bin/env python3
"""Chart source for ...-power-trace.{png,svg} — board power over the measured window.

Regenerate:
    python3 assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-power-trace.py

Reads the committed receipts
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/nvidia.csv
    results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/receipts/nvidia-serving.csv
and writes the PNG and SVG beside this script. No network, no GPU.
"""
import csv
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REC = os.path.join(REPO, "results", "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp", "receipts")
STEM = os.path.join(HERE, "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-power-trace")

FMT = "%Y/%m/%d %H:%M:%S.%f"


def load(fname):
    out = []
    with open(os.path.join(REC, fname)) as fh:
        for row in csv.DictReader(fh):
            ts = datetime.datetime.strptime(row["timestamp"].strip(), FMT)
            w = float(row[" power.draw [W]"].split()[0])
            out.append((ts, w))
    return out


def main():
    bench = load("nvidia.csv")
    serving = load("nvidia-serving.csv")
    # serving file starts later; plot both on one axis with a visual gap if needed
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for rows, color, label in ((bench, "#1f6feb", "llama-bench window"),
                               (serving, "#e8590c", "llama-server serving window")):
        if not rows:
            continue
        t0 = rows[0][0]
        xs = [(t - t0).total_seconds() for t, _ in rows]
        ys = [w for _, w in rows]
        ax.plot(xs, ys, lw=0.8, color=color, alpha=0.8, label=label)
        peak = max(ys)
        ax.annotate(f"peak {peak:.1f} W", xy=(xs[ys.index(peak)], peak),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color=color)
    ax.axhline(180, color="#c92a2a", ls="--", lw=1, label="180 W power cap")
    ax.set_xlabel("seconds since window start (per segment)")
    ax.set_ylabel("board power draw (W)")
    ax.set_title("Bonsai 2 27B benchmark power draw, 1x CMP 170HX (SM80), 1 Hz telemetry\n"
                 "segments reset at their own t=0; batch-1 decode stays well under the cap")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 200)
    fig.tight_layout()
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print("saved", STEM + ".png")


if __name__ == "__main__":
    main()
