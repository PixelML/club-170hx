#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"   # 180 W
ORANGE = "#d6722a" # 250 W
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e2e2e2"

# aggregate tok/s medians, ours_short protocol, per level (C4/C8/C16 blank at 250W = crashed)
levels = ["C1", "C2", "C4"]
v180 = [118.95, 176.59, 207.59]
v250 = [120.42, 158.75, None]
err180_lo = [48.46, 71.36, 124.46]
err180_hi = [161.72, 190.34, 290.73]
err250_lo = [86.85, 102.27, None]
err250_hi = [154.72, 201.27, None]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#fcfcfb")

# --- Panel 1: aggregate tok/s by concurrency, both arms ---
ax = axes[0]
ax.set_facecolor("#fcfcfb")
x = np.arange(len(levels))
w = 0.32
for i, (lv, v, elo, ehi, color, label) in enumerate([
        (levels, v180, err180_lo, err180_hi, BLUE, "180 W"),
        (levels, v250, err250_lo, err250_hi, ORANGE, "250 W")]):
    xs = x + (w/2 if label == "250 W" else -w/2)
    vals = [vv if vv is not None else 0 for vv in v]
    bars = ax.bar(xs, vals, width=w, color=color, label=label, zorder=3)
    for j, vv in enumerate(v):
        if vv is None:
            ax.text(xs[j], 5, "crashed", ha="center", va="bottom", fontsize=8, color=MUTED, rotation=90)
            continue
        lo, hi = elo[j], ehi[j]
        ax.plot([xs[j], xs[j]], [lo, hi], color=INK, lw=1.2, zorder=4)
        ax.plot([xs[j]-0.05, xs[j]+0.05], [lo, lo], color=INK, lw=1.2, zorder=4)
        ax.plot([xs[j]-0.05, xs[j]+0.05], [hi, hi], color=INK, lw=1.2, zorder=4)
ax.set_xticks(x); ax.set_xticklabels(levels)
ax.set_ylabel("aggregate tok/s (median of reps, min-max whisker)", color=INK, fontsize=9)
ax.set_title("Text c=1..4, our fixture prompt: 180 W vs 250 W", fontsize=10, color=INK, loc="left")
ax.grid(axis="y", color=GRID, zorder=0)
for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]: ax.spines[spine].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.axhline(163.06, color=MUTED, lw=1, linestyle="--", zorder=2)
ax.text(2.05, 163.06, "prior C1 figure\n163.1 tok/s", fontsize=7, color=MUTED, va="center")

# --- Panel 2: tokens per Wh at C1/C2 (measured active-load power) ---
ax2 = axes[1]
ax2.set_facecolor("#fcfcfb")
levels2 = ["C1", "C2"]
pwr180, pwr250 = 414.8, 428.8  # measured active-load draw, 4 cards, W
twh180 = [v * 3600 / pwr180 for v in v180[:2]]
twh250 = [v * 3600 / pwr250 for v in v250[:2]]
x2 = np.arange(len(levels2))
ax2.bar(x2 - w/2, twh180, width=w, color=BLUE, label="180 W", zorder=3)
ax2.bar(x2 + w/2, twh250, width=w, color=ORANGE, label="250 W", zorder=3)
ax2.set_xticks(x2); ax2.set_xticklabels(levels2)
ax2.set_ylabel("tokens / Wh (measured active-load power)", color=INK, fontsize=9)
ax2.set_title("Energy efficiency: 250 W buys no headroom here", fontsize=10, color=INK, loc="left")
ax2.grid(axis="y", color=GRID, zorder=0)
for spine in ["top", "right"]: ax2.spines[spine].set_visible(False)
for spine in ["left", "bottom"]: ax2.spines[spine].set_color(GRID)
ax2.tick_params(colors=MUTED, labelsize=8)
ax2.legend(frameon=False, fontsize=8, loc="upper left")

fig.suptitle("DeepSeek-V4-Flash-Vision-Exp, 4x CMP 170HX: power-cap reproducibility (2026-09-02)",
             fontsize=11, color=INK, x=0.02, ha="left")
fig.text(0.02, 0.01, "Median of 5 reps (C1/C2) / 3 reps (C4), text-only, greedy, ignore_eos. C4+ crashed at 250 W (EngineCore).",
          fontsize=7.5, color=MUTED)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])

fig.savefig("./power_arms_chart.png", dpi=200, facecolor=fig.get_facecolor())
fig.savefig("./power_arms_chart.svg", facecolor=fig.get_facecolor())
print("charts written")
