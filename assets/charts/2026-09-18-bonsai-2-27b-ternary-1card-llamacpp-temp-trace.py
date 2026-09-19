#!/usr/bin/env python3
"""Chart source for ...-temp-trace.{png,svg} — core and memory temperature over the window.

Regenerate:
    python3 assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-temp-trace.py

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
STEM = os.path.join(HERE, "2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-temp-trace")

FMT = "%Y/%m/%d %H:%M:%S.%f"


def load(fname):
    out = []
    with open(os.path.join(REC, fname)) as fh:
        for row in csv.DictReader(fh):
            ts = datetime.datetime.strptime(row["timestamp"].strip(), FMT)
            core = int(row[" temperature.gpu"])
            mem = int(row[" temperature.memory"])
            out.append((ts, core, mem))
    return out


def main():
    bench = load("nvidia.csv")
    serving = load("nvidia-serving.csv")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    peaks = {"core": 0, "mem": 0}
    for rows, tag in ((bench, "llama-bench window"), (serving, "llama-server serving window")):
        if not rows:
            continue
        t0 = rows[0][0]
        xs = [(t - t0).total_seconds() for t, _, _ in rows]
        core = [c for _, c, _ in rows]
        mem = [m for _, _, m in rows]
        peaks["core"] = max(peaks["core"], max(core))
        peaks["mem"] = max(peaks["mem"], max(mem))
        ax.plot(xs, core, lw=0.8, color="#e8590c", alpha=0.85,
                label=f"core C ({tag})" if tag == "llama-bench window" else None)
        ax.plot(xs, mem, lw=0.8, color="#1f6feb", alpha=0.85, ls="--",
                label=f"memory C ({tag})" if tag == "llama-bench window" else None)
    ax.axhline(80, color="#c92a2a", ls=":", lw=1, label="80 C core stop condition")
    ax.axhline(85, color="#1971c2", ls=":", lw=1, label="85 C memory stop condition")
    ax.set_xlabel("seconds since window start (per segment)")
    ax.set_ylabel("temperature (C)")
    ax.set_title(f"Bonsai 2 27B benchmark temperatures, 1x CMP 170HX (SM80), forced airflow\n"
                 f"peak core {peaks['core']} C, peak memory {peaks['mem']} C — both under the stop conditions")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(20, 95)
    fig.tight_layout()
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print("saved", STEM + ".png")


if __name__ == "__main__":
    main()
