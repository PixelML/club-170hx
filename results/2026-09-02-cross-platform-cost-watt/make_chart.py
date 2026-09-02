#!/usr/bin/env python3
"""Cross-platform chart: 4x CMP 170HX vs 2x DGX Spark, DeepSeek-V4-Flash-Vision-Exp.
Reads the committed JSON receipts in this directory; writes PNG+SVG to assets/charts/.
Palette: dataviz skill default categorical slots 1 (blue) and 2 (orange), validated
with scripts/validate_palette.js (light mode PASS on both hard gates).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"    # CMP 170HX
ORANGE = "#eb6834"  # DGX Spark
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"

cmp_levels = json.load(open("cmp-level-power-summary.json"))
spark_levels = json.load(open("spark-power/level-power-summary.json"))
cmp_levels = [l for l in cmp_levels if l["concurrency"] <= 8]  # C16 failed, excluded

cost = json.load(open("cross-platform-summary.json"))["cost_per_million_output_tokens_usd"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

# Panel 1: tokens/Wh vs concurrency (measured)
cx = [l["concurrency"] for l in cmp_levels]
cy = [l["tok_per_Wh"] for l in cmp_levels]
sx = [l["concurrency"] for l in spark_levels]
sy = [l["tok_per_Wh"] for l in spark_levels]

ax1.plot(cx, cy, marker="o", ms=7, lw=2, color=BLUE, label="4x CMP 170HX")
ax1.plot(sx, sy, marker="o", ms=7, lw=2, color=ORANGE, label="2x DGX Spark")
for x, y in zip(cx, cy):
    ax1.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8, color=INK)
for x, y in zip(sx, sy):
    ax1.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, -14),
                 ha="center", fontsize=8, color=INK)
ax1.set_xticks([1, 2, 4, 8])
ax1.set_xlabel("concurrency (completion requests in flight)", color=MUTED, fontsize=9)
ax1.set_ylabel("tokens per Wh (whole measured node power)", color=MUTED, fontsize=9)
ax1.set_title("Measured efficiency: tokens per Wh vs concurrency", color=INK, fontsize=11, loc="left")
ax1.legend(frameon=False, fontsize=9, loc="upper left")

# Panel 2: $ per million output tokens, stacked hardware + energy, at c=8, $0.15/kWh
platforms = ["4x CMP 170HX\n(c=8, measured)", "2x DGX Spark\n(c=8, measured)"]
hw = [cost["cmp_170hx_c8"]["amortized_hw"], cost["dgx_spark_c8"]["amortized_hw"]]
en15 = [cost["cmp_170hx_c8"]["energy_at_0.15"], cost["dgx_spark_c8"]["energy_at_0.15"]]
en30 = [cost["cmp_170hx_c8"]["energy_at_0.30"], cost["dgx_spark_c8"]["energy_at_0.30"]]
colors = [BLUE, ORANGE]
x = [0, 1]
bars_hw = ax2.bar(x, hw, width=0.5, color=colors, label="amortized hardware (assumed)")
bars_en = ax2.bar(x, en15, width=0.5, bottom=hw, color=colors, alpha=0.55,
                   hatch="///", edgecolor=SURFACE, linewidth=1,
                   label="energy @ $0.15/kWh (assumed rate)")
ax2.set_ylim(0, max(h + e30 for h, e30 in zip(hw, en30)) * 1.35)
for xi, h, e15, e30 in zip(x, hw, en15, en30):
    total15 = h + e15
    ax2.annotate(f"${total15:.2f}", (xi, total15), textcoords="offset points", xytext=(0, 6),
                 ha="center", fontsize=9, color=INK, fontweight="bold")
    ax2.annotate(f"(${h + e30:.2f} @ $0.30/kWh)", (xi, total15), textcoords="offset points",
                 xytext=(0, 22), ha="center", fontsize=7.5, color=MUTED)
ax2.set_xticks(x)
ax2.set_xticklabels(platforms, fontsize=9, color=INK)
ax2.set_ylabel("USD per million output tokens", color=MUTED, fontsize=9)
ax2.set_title("Modeled cost: $/M output tokens (hardware + energy)", color=INK, fontsize=11, loc="left", pad=14)
ax2.legend(frameon=False, fontsize=8, loc="upper left")

fig.suptitle("DeepSeek-V4-Flash-Vision-Exp: 4x CMP 170HX vs 2x DGX Spark (2026-09-02)",
             fontsize=12, color=INK, x=0.02, ha="left")
fig.text(0.02, 0.01,
         "Measured: throughput, GPU power. Assumed: hardware list price, electricity rate, 3yr/50% utilization lifetime. "
         "DGX Spark GPU-power-only reading is a lower bound on whole-node energy.",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])

fig.savefig("../../assets/charts/2026-09-02-cross-platform-cmp170hx-vs-dgxspark.png", dpi=200)
fig.savefig("../../assets/charts/2026-09-02-cross-platform-cmp170hx-vs-dgxspark.svg")
print("wrote charts")
