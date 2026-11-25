"""Build assets/charts/<experiment>.png from the receipts via summarize.ROUNDS. Run from the repo root."""
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize as S

EXP = HERE.name
OUT = HERE.parents[1] / "assets" / "charts" / f"{EXP}.png"

S.write_summary_csv()
sb = S.scoreboard()
n_rounds = len(sb)

fig, (ax1, ax2, ax3) = plt.subplots(
    1,
    3,
    figsize=(24.0, 5.9) if n_rounds >= 9 else ((20.2, 5.7) if n_rounds >= 8 else ((18.8, 5.6) if n_rounds >= 7 else (17.2, 5.5))),
    gridspec_kw={"width_ratios": [1.30, 1.34, 1.28] if n_rounds >= 9 else ([1.22, 1.34, 1.20] if n_rounds >= 8 else ([1.18, 1.32, 1.16] if n_rounds >= 7 else [1.14, 1.34, 1.12]))},
)

# Panel 1: 6-prompt single-user decode progression (T=0.6)
xs1 = list(range(n_rounds))
c1_meds = [r["c1_med"] for r in sb]
c1_peaks = [r["c1_peak"] for r in sb]
colors = [r["color"] for r in sb]
ax1.bar(xs1, c1_meds, width=0.54 if n_rounds >= 7 else 0.56, color=colors, edgecolor="#222222", linewidth=0.6)
ax1.scatter(xs1, c1_peaks, color="#111111", marker="D", s=24, zorder=5, label="run peak")
ax1.axhline(
    S.CEILING_SUMMARY["c1_mixed_4k"],
    color="#b03a2e",
    linestyle="--",
    linewidth=1.1,
    label=f"HBM ceiling ({S.CEILING_SUMMARY['c1_mixed_4k']:.0f} tok/s)",
)
for i, (m, pk) in enumerate(zip(c1_meds, c1_peaks)):
    gain = f"\n(+{sb[i]['c1_gain_pct']:.1f}%)" if i > 0 else ""
    m_txt = f"{m:.2f}" if i == n_rounds - 1 else f"{m:.1f}"
    ax1.text(
        i,
        max(m, pk) + 4.0,
        f"{m_txt}{gain}",
        ha="center",
        va="bottom",
        fontsize=6.5 if n_rounds >= 7 else (6.9 if n_rounds >= 6 else 7.5),
        fontweight="bold" if i == n_rounds - 1 else "normal",
    )
ax1.set_xticks(xs1, [r["label"] for r in sb], fontsize=5.2 if n_rounds >= 9 else (5.6 if n_rounds >= 7 else (5.9 if n_rounds >= 6 else 7.0)))
ax1.set_ylabel("decode tok/s, 1 user, 6-prompt median (T=0.6)")
ax1.set_title("Hill-climb c=1 decode (262K ctx)")
ax1.set_ylim(0, 272)
ax1.grid(axis="y", alpha=0.28)
ax1.legend(fontsize=7.3, loc="upper left")

# Panel 2: PR #90 apples-to-apples suite on stock card @ 262K ctx
cases = [("code", "code (LRU)"), ("struct", "structured JSON"), ("prose", "prose essay")]
w = min(0.18, 0.82 / n_rounds)
for i, r in enumerate(sb):
    vals = [r["pr90"].get(k, {}).get("pooled", 0.0) or 0.0 for k, _ in cases]
    pos = [j + (i - (n_rounds - 1) / 2.0) * w for j in range(len(cases))]
    ax2.bar(pos, vals, width=w * 0.90, color=r["color"], edgecolor="#222222", linewidth=0.5, label=r["short"].replace("`", ""))
    for x, y in zip(pos, vals):
        ax2.text(x, y + 2.5, f"{y:.0f}", ha="center", va="bottom", fontsize=5.8 if n_rounds >= 7 else (6.2 if n_rounds >= 6 else 6.8))
# PR #90 +200 MHz SM offset card reference markers (T=0, 32K ctx)
ref_vals = [
    S.PR90_PUBLISHED_OFFSET_CARD["code_t0"],
    S.PR90_PUBLISHED_OFFSET_CARD["struct_t0"],
    S.PR90_PUBLISHED_OFFSET_CARD["prose_t0"],
]
ax2.scatter(
    range(len(cases)),
    ref_vals,
    color="#c0392b",
    marker="_",
    s=260,
    linewidths=2.0,
    zorder=6,
    label="PR #90 (+200 MHz SM offset, 32K, T=0)",
)
ax2.set_xticks(range(len(cases)), [lab for _, lab in cases], fontsize=8.5)
ax2.set_ylabel("decode tok/s, 768 output tokens (pooled T=0,1 median)")
ax2.set_title("PR #90 harness on stock card (262K ctx)")
ax2.set_ylim(0, 305)
ax2.grid(axis="y", alpha=0.28)
ax2.legend(fontsize=6.5 if n_rounds >= 7 else 6.8, loc="upper left", ncol=2)

# Panel 3: Unified-server c=16 aggregate throughput
c16_meds = [r["c16_med"] for r in sb]
c16_peaks = [r["c16_peak"] for r in sb]
ax3.bar(xs1, c16_meds, width=0.54 if n_rounds >= 7 else 0.56, color=colors, edgecolor="#222222", linewidth=0.6)
ax3.scatter(xs1, c16_peaks, color="#111111", marker="D", s=24, zorder=5, label="run peak")
ax3.axhline(
    S.PR90_PUBLISHED_OFFSET_CARD["c16_t1"],
    color="#c0392b",
    linestyle=":",
    linewidth=1.3,
    label=f"PR #90 c=16 ({S.PR90_PUBLISHED_OFFSET_CARD['c16_t1']:.1f}*, +200 MHz)",
)
for i, (m, pk) in enumerate(zip(c16_meds, c16_peaks)):
    gain = f"\n(+{sb[i]['c16_gain_pct']:.1f}%)" if i > 0 else ""
    ax3.text(
        i,
        max(m, pk) + 15.0,
        f"{m:.1f}{gain}",
        ha="center",
        va="bottom",
        fontsize=6.5 if n_rounds >= 7 else (6.9 if n_rounds >= 6 else 7.5),
        fontweight="bold" if i == n_rounds - 1 else "normal",
    )
ax3.set_xticks(xs1, [r["label"] for r in sb], fontsize=5.2 if n_rounds >= 9 else (5.6 if n_rounds >= 7 else (5.9 if n_rounds >= 6 else 7.0)))
ax3.set_ylabel("aggregate output tok/s, 16 concurrent users")
ax3.set_title("Unified server c=16 throughput")
ax3.set_ylim(0, 835)
ax3.grid(axis="y", alpha=0.28)
ax3.legend(fontsize=7.2, loc="upper left")

champ = sb[-1]
fig.suptitle(
    f"Inkling-2-Mini on 1x stock CMP 170HX (SM80, 140 W, 262,144 ctx) — "
    f"{sb[0]['c1_med']:.1f} → {champ['c1_med']:.2f} tok/s c=1, {sb[0]['c16_med']:.1f} → {champ['c16_med']:.1f} tok/s c=16 (measured 2026-11-25)",
    fontsize=11,
)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=135)
print(OUT)
