# %% [markdown]
# # Qwen3.8-27B + LLKVApprox on one CMP 170HX
#
# **Measured verdict: 1.71x faster prefill; quality trade-off remains.**
#
# A recorded comparison across separate receipt files shows zero-shot LLKVApprox with an exact
# suffix of 256 tokens at **1.944 s versus 3.329 s** for full BF16 prefill at 6,603 tokens
# (3,396 versus 1,983 tok/s; 1.712x). Quality was measured on a separate
# 23-prompt cohort: the 15 prompts at or below the suffix took the exact path, while the 8
# approximation-active prompts reached **13.3% zero-shot** and **16.0% stage-1-trained** greedy
# token agreement, with 0/8 full matches. Greedy agreement is a diagnostic, **not task accuracy**.
#
# ![Evidence-backed LLKVApprox chart](../assets/charts/2026-09-11-qwen3.8-27b-llkvapprox-1card-hf.png)
#
# Evidence class: **MEASURED**. Hardware: **1x CMP 170HX, 64 GiB, 180 W cap**. Model:
# **Qwen/Qwen3.8-27B BF16** on the HF Transformers path. Exact model/projector revisions and the
# immutable runtime commit were not recorded in the committed receipts, so this notebook is a CPU-only
# receipt replay—not a full live GPU reproduction.
#
# Mechanism credit: [kishida's LLKVApprox demo](https://kishida.github.io/webdemos/llkvapprox/),
# [reference engine](https://github.com/kishida/webdemos/blob/main/llkvapprox/engine.js), and
# [write-up](https://nowokay.hatenablog.com/entry/2026/09/11/120001).

# %% [markdown]
# ## Requirements and configure
#
# Run from the repository root or `notebooks/` with Python 3, `matplotlib`, and IPython.
# `LIVE` is intentionally false: these cells only verify and visualize committed public receipts.

# %%
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from IPython.display import Markdown, display

EXPERIMENT = "2026-09-12-qwen3.8-27b-llkvapprox-1card-hf"
LIVE = False

root_candidates = (Path.cwd(), Path.cwd().parent)
ROOT = next(
    path for path in root_candidates
    if (path / "receipts" / EXPERIMENT / "bench-prefill.json").is_file()
)
RECEIPT_DIR = ROOT / "receipts" / EXPERIMENT
EVIDENCE_PATH = ROOT / "results" / EXPERIMENT / "publication-evidence.json"
CHART_PATH = ROOT / "assets" / "charts" / "2026-09-11-qwen3.8-27b-llkvapprox-1card-hf.png"

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def receipt(name: str) -> dict:
    return load_json(RECEIPT_DIR / name)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def markdown_table(headers: list[str], rows: list[list[object]]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    display(Markdown("\n".join(lines)))

evidence = load_json(EVIDENCE_PATH)
for item in evidence["receipts"]:
    path = ROOT / item["path"]
    assert path.is_file(), f"missing receipt: {item['path']}"
    assert sha256(path) == item["sha256"], f"receipt hash mismatch: {item['path']}"

print("receipt replay : ready")
print("evidence class :", evidence["evidence_class"])
print("LIVE           :", LIVE)
print("receipt hashes : verified")

# %% [markdown]
# ## Preflight: verify the selected headline fields
#
# The assertions below gate the numeric fields used in this notebook's headline tables and chart.
# Card/video copy and media properties are checked separately during the publication review.

# %%
bench = receipt("bench-prefill.json")
bench_zs = receipt("bench-prefill-zs-s256.json")
quality_zs = receipt("quality-eval-zs-s256.json")
quality_trained = receipt("quality-trained-s256.json")
oracle = receipt("oracle-test.json")
stage1 = receipt("stage1_cached_mlp.json")

def timed_row(data: dict, mode: str, tokens: int) -> dict:
    return next(row for row in data["rows"] if row.get("mode", "student") == mode and row["tokens"] == tokens)

full_6603 = timed_row(bench, "full", 6603)
oracle_6603 = timed_row(bench, "oracle", 6603)
student_6603 = timed_row(bench_zs, "student", 6603)
speedup = full_6603["mean_s"] / student_6603["mean_s"]

def quality_split(data: dict, mode: str) -> dict:
    exact_rows = [row for row in data["rows"] if row["prompt_tokens"] <= 256]
    active_rows = [row for row in data["rows"] if row["prompt_tokens"] > 256]
    pooled_match = sum(row["modes"][mode]["match_tokens"] for row in data["rows"])
    pooled_total = sum(row["modes"][mode]["total"] for row in data["rows"])
    exact_match = sum(row["modes"][mode]["match_tokens"] for row in exact_rows)
    exact_total = sum(row["modes"][mode]["total"] for row in exact_rows)
    active_match = sum(row["modes"][mode]["match_tokens"] for row in active_rows)
    active_total = sum(row["modes"][mode]["total"] for row in active_rows)
    return {
        "pooled_match": pooled_match,
        "pooled_total": pooled_total,
        "exact_prompts": len(exact_rows),
        "exact_match": exact_match,
        "exact_total": exact_total,
        "active_prompts": len(active_rows),
        "active_match": active_match,
        "active_total": active_total,
        "active_agreement": active_match / active_total,
        "active_full_matches": sum(row["modes"][mode]["match_rate"] == 1 for row in active_rows),
    }

zs_split = quality_split(quality_zs, "student_zeroshot")
trained_split = quality_split(quality_trained, "student")

assert round(speedup, 4) == round(evidence["prefill_speed"]["speedup"], 4)
assert (full_6603["mean_s"], student_6603["mean_s"]) == (3.3293, 1.9444)
assert (zs_split["pooled_match"], zs_split["pooled_total"]) == (2056, 2944)
assert (trained_split["pooled_match"], trained_split["pooled_total"]) == (2084, 2944)
assert (zs_split["exact_match"], zs_split["exact_total"]) == (1920, 1920)
assert (trained_split["exact_match"], trained_split["exact_total"]) == (1920, 1920)
assert (zs_split["active_match"], zs_split["active_total"]) == (136, 1024)
assert (trained_split["active_match"], trained_split["active_total"]) == (164, 1024)
assert zs_split["exact_prompts"] == trained_split["exact_prompts"] == 15
assert zs_split["active_prompts"] == trained_split["active_prompts"] == 8
assert zs_split["active_full_matches"] == trained_split["active_full_matches"] == 0

print("recorded speed fields  : verified")
print("quality cohort split   : verified")
print("unsupported decode data: excluded")

# %% [markdown]
# ## Benchmark: measured prefill and separate quality diagnostic
#
# The speed and quality rows below must not be read as one deployment operating point. The
# 6,603-token timing cohort has no corresponding quality measurement.

# %%
markdown_table(
    ["Prefill mode", "Prompt tokens", "Mean seconds", "Throughput", "Status"],
    [
        ["Full BF16", "6,603", f"{full_6603['mean_s']:.4f}", f"{full_6603['tok_s']:,.1f} tok/s", "MEASURED"],
        ["Zero-shot suffix 256", "6,603", f"{student_6603['mean_s']:.4f}", f"{student_6603['tok_s']:,.1f} tok/s", f"MEASURED · {speedup:.2f}x"],
        ["Trained suffix 256", "6,603", "UNVERIFIED", "UNVERIFIED", "Repeated timing failures"],
    ],
)

markdown_table(
    ["Quality diagnostic", "All 23 prompts", "8 active prompts", "Active full matches"],
    [
        ["Zero-shot suffix 256", f"{zs_split['pooled_match']}/{zs_split['pooled_total']} · 69.8%", f"{zs_split['active_match']}/{zs_split['active_total']} · {100 * zs_split['active_agreement']:.1f}%", "0/8"],
        ["Stage-1 suffix 256", f"{trained_split['pooled_match']}/{trained_split['pooled_total']} · 70.8%", f"{trained_split['active_match']}/{trained_split['active_total']} · {100 * trained_split['active_agreement']:.1f}%", "0/8"],
    ],
)

display(Markdown(
    "**Interpretation:** the pooled 69.8%/70.8% values include 15 prompts that take the exact "
    "path and contribute 1,920/1,920 matching tokens per mode. On the prompts that actually use "
    "approximation, agreement is 13.3%/16.0%. This is not task accuracy."
))

# %% [markdown]
# ## Chart replay
#
# The left panel compares recorded 6,603-token timing arms from separate receipt files. The right panel uses only the 8 prompts
# longer than suffix 256. The panels are explicitly marked as separate cohorts.

# %%
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.labelcolor": "#dce7f5",
    "xtick.color": "#aab6c8",
    "ytick.color": "#aab6c8",
    "text.color": "#f4f7fb",
})

fig, (ax_speed, ax_quality) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=100)
fig.patch.set_facecolor("#07111f")
for axis in (ax_speed, ax_quality):
    axis.set_facecolor("#111e30")
    axis.grid(axis="y", color="#aab6c8", alpha=0.16, linewidth=0.8)
    axis.spines[:].set_visible(False)

speed_labels = ["Full BF16", "Zero-shot\nsuffix 256"]
speed_seconds = [full_6603["mean_s"], student_6603["mean_s"]]
speed_bars = ax_speed.bar(speed_labels, speed_seconds, color=["#64748b", "#36d7b0"], width=0.62)
ax_speed.set_title("RECORDED PREFILL COMPARISON · 6,603 TOKENS", fontsize=14, pad=18)
ax_speed.set_ylabel("Prefill seconds · lower is better")
ax_speed.set_ylim(0, 3.8)
for bar, seconds in zip(speed_bars, speed_seconds):
    ax_speed.text(bar.get_x() + bar.get_width() / 2, seconds + 0.08, f"{seconds:.3f} s", ha="center", fontsize=13, fontweight="bold")
ax_speed.text(0.5, 3.55, f"{speedup:.2f}x faster", ha="center", color="#36d7b0", fontsize=18, fontweight="bold")
ax_speed.text(0.5, -0.17, "3 samples/arm · warmup only notebook-reported", transform=ax_speed.transAxes, ha="center", color="#aab6c8", fontsize=10)

quality_labels = ["Zero-shot", "Stage 1"]
quality_values = [100 * zs_split["active_agreement"], 100 * trained_split["active_agreement"]]
quality_bars = ax_quality.bar(quality_labels, quality_values, color=["#5c8dff", "#f6b83f"], width=0.62)
ax_quality.set_title("QUALITY DIAGNOSTIC · 8 ACTIVE PROMPTS", fontsize=14, pad=18)
ax_quality.set_ylabel("Greedy token agreement · not task accuracy")
ax_quality.set_ylim(0, 25)
for bar, value in zip(quality_bars, quality_values):
    ax_quality.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}%", ha="center", fontsize=15, fontweight="bold")
ax_quality.text(0.5, 22.5, "0 / 8 full matches", ha="center", color="#f4d58d", fontsize=13, fontweight="bold")
ax_quality.text(0.5, -0.17, "Separate cohort · 1,024 generated tokens per mode", transform=ax_quality.transAxes, ha="center", color="#aab6c8", fontsize=10)

fig.suptitle("Qwen3.8-27B + LLKVApprox · 1x CMP 170HX · HF BF16", fontsize=19, fontweight="bold", y=0.97)
fig.text(0.5, 0.02, "1.71x faster prefill; quality trade-off remains", ha="center", color="#36d7b0", fontsize=16, fontweight="bold")
fig.tight_layout(rect=(0.03, 0.08, 0.97, 0.91), w_pad=3.5)
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(CHART_PATH, facecolor=fig.get_facecolor())
plt.show()
print("chart regenerated: assets/charts/2026-09-11-qwen3.8-27b-llkvapprox-1card-hf.png")

# %% [markdown]
# ## Oracle diagnostic and training status
#
# The oracle speed path uses teacher captures and is a diagnostic ceiling—not deployable speed or
# TTFT. Its speed comparison is at 6,603 prompt tokens, while fidelity was measured separately at
# 1,024 prompt tokens. Fidelity is not bit-exact: teacher-forced argmax agreement is 31/32, and
# free-running greedy agreement is 18/64. Neither metric is task accuracy. Stage 1 completed;
# stage 2 and the stacked-GEMM projector remain unvalidated.

# %%
oracle_speedup = full_6603["mean_s"] / oracle_6603["mean_s"]
markdown_table(
    ["Item", "Receipt-backed result", "Interpretation"],
    [
        ["Oracle prefill", f"{oracle_6603['mean_s']:.4f} s · {oracle_speedup:.2f}x", "Teacher-capture ceiling"],
        ["Oracle teacher-forced fidelity", f"{oracle['drift_argmax_match_count']}/32 argmax at {oracle['prompt_tokens']:,} prompt tokens", "Not bit-exact; not task accuracy"],
        ["Oracle free-running fidelity", f"{oracle['greedy_agree_tokens']}/{oracle['greedy_total']} greedy tokens at {oracle['prompt_tokens']:,} prompt tokens", "Separate negative diagnostic"],
        ["Stage 1", f"{stage1['args']['steps']} steps · {stage1['wall_hours']:.2f} h", "Completed"],
        ["Trained prefill", "No successful timing receipt", "UNVERIFIED"],
        ["Stage 2 / stacked GEMM", "No validated receipt", "UNVALIDATED"],
    ],
)

# %% [markdown]
# ## Reproduction boundary and limitations
#
# This notebook provides a runnable, CPU-only reader path: it verifies receipt hashes, recomputes
# the speed ratio and quality cohort split, renders the tables, and regenerates the chart. It does
# **not** claim a full live GPU reproduction because the public evidence does not include every
# benchmark script or immutable base-model, projector, and runtime revision.
#
# Additional limits:
#
# - Speed and quality use separate cohorts; no quality result exists at 6,603 tokens.
# - Trained-projector throughput is unverified after repeated illegal-memory timing failures.
# - The 2.01x oracle result is not deployable throughput or TTFT.
# - Greedy token agreement is not task accuracy.
# - No end-to-end TTFT or video-understanding result is included.
#
# The structured evidence and exact receipt hashes are in
# `results/2026-09-12-qwen3.8-27b-llkvapprox-1card-hf/publication-evidence.json`.
