#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Production-load chart for the Qwen3.8-27B Jev-endpoint experiment.

Reads only committed receipts under ../receipts and writes the figure to
assets/charts/ as PNG and SVG:

  left   annotations per minute across the whole production run (the pilot's
         own pacing; the engine sample window is marked)
  right  answer time per engine request in ms, log scale: idle c=1 reads
         (latency.json) against the loaded engine (metrics-delta.json)

  python make_chart_load.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE.parent / "receipts"
CHARTS = HERE.parents[2] / "assets" / "charts"
STEM = "2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm"


def receipt(rel):
    with open(RECEIPTS / rel) as f:
        return json.load(f)


def main():
    summary = receipt("load/load-summary.json")
    delta = receipt("load/metrics-delta.json")
    minutes = receipt("load/load-ledger-minutes.json")["minutes"]
    lat = receipt("latency.json")
    os.makedirs(CHARTS, exist_ok=True)

    ledger = summary["ledger"]
    window = summary["engine_window"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})

    # --- left: annotations per minute ---------------------------------------
    def minute_index(m):
        h, mm = m["minute_utc"].split(":")
        return int(h) * 60 + int(mm)

    xs = [minute_index(m) for m in minutes]
    base = xs[0]
    xs = [x - base for x in xs]
    ys = [m["reads"] for m in minutes]
    ax1.plot(xs, ys, color="#1f6feb", lw=1.4)
    ax1.fill_between(xs, 0, ys, color="#1f6feb", alpha=0.15)
    mean_rate = ledger["annotations_per_h_avg"] / 60
    ax1.axhline(mean_rate, color="#e8590c", lw=1.2, ls="--",
                label=f"run average {mean_rate:.0f}/min")
    steady = ledger["steady_last30_annotations_per_min"]
    ax1.axhline(steady, color="#0ca678", lw=1.2, ls=":",
                label=f"steady state {steady:.0f}/min (last 30 min)")
    # mark the 292 s engine sample window
    end_min = ledger["end_utc"][11:16]
    h, mm = end_min.split(":")
    win_end = (int(h) * 60 + int(mm)) - base
    win_start = win_end - window["window_s"] / 60
    ax1.axvspan(win_start, win_end, color="#9c36b5", alpha=0.18,
                label=f"engine sample window ({window['window_s']:.0f} s)")
    ax1.set_xlabel("Minute of the run (UTC minutes since first annotation, 2026-09-21)")
    ax1.set_ylabel("Annotations completed per minute")
    ax1.grid(alpha=0.3)
    ax1.set_title(f"{ledger['annotations']:,} annotations in "
                  f"{ledger['span_h']:.1f} h, {ledger['failed']} failures\n"
                  f"engine prefill {window['prompt_tokens_per_s']:.0f} tok/s "
                  f"in the sample window")

    # --- right: answer time in ms -------------------------------------------
    bars = [
        ("idle c=1\nwarm cache\np50", lat["warm"]["p50_ms"], "#1f6feb", ""),
        ("idle c=1\ncache-busted\np50", lat["busted"]["p50_ms"], "#4c8ef7", ""),
        ("loaded, 16 clients\nmean", delta["e2e_latency_mean_s"] * 1000,
         "#e8590c", ""),
        ("loaded, 16 clients\n99.8% answered within",
         15_000, "#f0975c", "//"),
    ]
    xs2 = range(len(bars))
    values = [b[1] for b in bars]
    ax2.bar(xs2, values, color=[b[2] for b in bars],
            hatch=[b[3] for b in bars], edgecolor="white")
    ax2.set_yscale("log")
    for x, (label, v, _, _) in zip(xs2, bars):
        text = f"{v:,.0f} ms" if v >= 1000 else f"{v:.1f} ms"
        ax2.annotate(text, xy=(x, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5)
    ax2.set_xticks(list(xs2))
    ax2.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax2.set_ylabel("Answer time per engine request (ms, log scale)")
    ax2.set_title("The answer, in ms: idle probe vs production load\n"
                  "(one annotation = 5 permutation reads in flight together; "
                  "queue-bound at 6 running / 58 waiting)")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_ylim(80, 40_000)

    ax2.set_title("The answer, in ms: idle vs production load")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "svg"):
        fig.savefig(CHARTS / f"{STEM}-load.{ext}", dpi=150)
    print("wrote", CHARTS / f"{STEM}-load.png")
    print("wrote", CHARTS / f"{STEM}-load.svg")


if __name__ == "__main__":
    main()
