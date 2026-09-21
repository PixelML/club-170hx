#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Chart for the Qwen3.8-27B Jev-endpoint experiment.

Reads only committed receipts under ../receipts and writes the figure to
assets/charts/ as PNG and SVG:

  left   reliability diagram: binned confidence against empirical accuracy on
         the labelled reads, at T=1 and at the NLL-fitted temperature
  right  option-order rotation: the probability of each option when the option
         order is rotated, i.e. the position bias the permutations option
         exists to average over

  python make_chart.py
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


def receipt(name):
    with open(RECEIPTS / name) as f:
        return json.load(f)


def reliability(metrics, key):
    xs, ys, ns = [], [], []
    for b in metrics[key]["reliability"]:
        if b["n"]:
            xs.append(b["conf"] / b["n"])
            ys.append(b["correct"] / b["n"])
            ns.append(b["n"])
    return xs, ys, ns


def main():
    m = receipt("metrics.json")
    perm = receipt("permutations.json")
    os.makedirs(CHARTS, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # --- left: reliability -------------------------------------------------
    ax1.plot([0, 1], [0, 1], "--", color="#adb5bd", lw=1, label="perfect calibration")
    for key, colour, label in (
        ("at_T1", "#1f6feb", f"T=1.00 (ECE {m['at_T1']['ece_10bin']:.3f})"),
        ("at_fitted_T", "#0ca678",
         f"T={m['fitted_temperature']:.2f} fitted (ECE {m['at_fitted_T']['ece_10bin']:.3f})"),
    ):
        xs, ys, ns = reliability(m, key)
        ax1.plot(xs, ys, "o-", color=colour, lw=1.6, ms=6,
                 label=label)
        for x, y, n in zip(xs, ys, ns):
            ax1.annotate(f"n={n}", (x, y), xytext=(4, -10), textcoords="offset points",
                         fontsize=6.5, color=colour)
    ax1.set_xlabel("Mean predicted confidence in bin (max option probability)")
    ax1.set_ylabel("Empirical accuracy in bin (fraction correct)")
    ax1.set_title(f"Reliability on {m['n_scored']} labelled reads\n"
                  f"accuracy {m['at_T1']['accuracy']:.3f} "
                  f"(95% CI {m['at_T1']['accuracy_wilson95'][0]:.2f}-"
                  f"{m['at_T1']['accuracy_wilson95'][1]:.2f}), 10 bins")
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(-0.03, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    # --- right: option-order rotation --------------------------------------
    options = perm["per_rotation"][0]["option_order"]
    rotations = [r["rotation"] for r in perm["per_rotation"]]
    width = 0.8 / len(options)
    colours = ["#1f6feb", "#9c36b5", "#e8590c", "#0ca678"]
    for i, name in enumerate(options):
        vals = []
        for r in perm["per_rotation"]:
            order = r["option_order"].index(name)
            vals.append(r["probabilities"][order])
        xs = [r - 0.4 + width * (i + 0.5) for r in rotations]
        bars = ax2.bar(xs, vals, width=width, color=colours[i % len(colours)],
                       label=name)
        for rect, v in zip(bars, vals):
            ax2.annotate(f"{v:.3f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", fontsize=7)
    for r in perm["per_rotation"]:
        order = [f"{n} {p:.3f}" for n, p in sorted(
            zip(r["option_order"], r["probabilities"]), key=lambda kv: -kv[1])]
        ax2.annotate("argmax: " + order[0],
                     xy=(r["rotation"], 1.0), xytext=(0, -14), textcoords="offset points",
                     ha="center", fontsize=7.5,
                     color="#c92a2a" if not r["argmax_vs_rotation0"] else "#2b8a3e")
    ax2.set_xticks(rotations)
    ax2.set_xticklabels([f"rotation {r}\n{', '.join(perm['per_rotation'][r]['option_order'])}"
                         for r in rotations], fontsize=8)
    ax2.set_xlabel("Option order presented in the prompt")
    ax2.set_ylabel("Probability read for the option (T=1)")
    ax2.set_title("Option-order rotation on one choice question\n"
                  f"max L1 shift {perm['max_l1_vs_rotation0']:.3f}; "
                  f"argmax stable: {perm['argmax_stable']}")
    ax2.set_ylim(0, 0.75)
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=8, title="option", title_fontsize=8)

    fig.suptitle("Qwen3.8-27B W4A16 Jev-compatible endpoint on 1x CMP 170HX (SM80), vLLM",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "svg"):
        fig.savefig(CHARTS / f"{STEM}.{ext}", dpi=150)
    print("wrote", CHARTS / f"{STEM}.png")
    print("wrote", CHARTS / f"{STEM}.svg")


if __name__ == "__main__":
    main()
