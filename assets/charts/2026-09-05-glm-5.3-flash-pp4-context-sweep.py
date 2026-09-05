#!/usr/bin/env python3
"""Chart source for 2026-09-05-glm-5.3-flash-pp4-context-sweep.{png,svg}.

Regenerate:
    python3 assets/charts/2026-09-05-glm-5.3-flash-pp4-context-sweep.py

Reads the committed receipt
    results/2026-09-05-glm-5.3-flash-4card-pp4-vllm/receipts/k3/sweep/context_sweep.json
and writes the PNG and SVG beside this script. No network, no GPU.
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RECEIPT = os.path.join(
    REPO, "results", "2026-09-05-glm-5.3-flash-4card-pp4-vllm",
    "receipts", "k3", "sweep", "context_sweep.json")
STEM = os.path.join(HERE, "2026-09-05-glm-5.3-flash-pp4-context-sweep")

ARMS = [("thinking_off", "thinking off", "#1f77b4", "o"),
        ("thinking_on", "thinking on", "#ff7f0e", "s")]


def med(reps, key):
    vals = [r[key] for r in reps if r.get("ok") and r.get(key) is not None]
    return st.median(vals) if vals else None


def main():
    with open(RECEIPT) as fh:
        doc = json.load(fh)

    labels, series = [], {a: {} for a, _, _, _ in ARMS}
    fields = ["prompt_tps", "gen_tps", "cold_ttft_s", "warm_ttft_s",
              "tpot_ms", "prompt_kb_s", "gen_bytes_s", "ratio"]
    for arm, _, _, _ in ARMS:
        for f in fields:
            series[arm][f] = []

    for row in doc["results"]:
        arms = row["arms"]
        if not all(arms[a].get("cold", {}).get("ok") for a, _, _, _ in ARMS):
            continue  # unmeasured length, see the note printed on the figure
        n = arms["thinking_off"]["cold"]["prompt_tokens"]
        labels.append(f"{n:,}")
        for arm, _, _, _ in ARMS:
            v = arms[arm]
            reps = v["reps"]
            series[arm]["prompt_tps"].append(v["median_prompt_tps"])
            series[arm]["gen_tps"].append(v["median_gen_tps"])
            series[arm]["cold_ttft_s"].append(v["cold_ttft_s"])
            series[arm]["warm_ttft_s"].append(v["warm_ttft_s"])
            series[arm]["tpot_ms"].append(v["median_tpot_ms"])
            series[arm]["prompt_kb_s"].append(med(reps, "prompt_kb_s"))
            series[arm]["gen_bytes_s"].append(med(reps, "gen_bytes_s"))
            series[arm]["ratio"].append(v["cold_warm_ttft_ratio"])

    x = list(range(len(labels)))
    panels = [
        ("prompt_tps", "Prompt processing (prompt tok/s)", "tokens/s", False),
        ("gen_tps", "Generation (tok/s)", "tokens/s", False),
        ("cold_ttft_s", "Cold TTFT (s, log scale)", "seconds", True),
        ("warm_ttft_s", "Warm TTFT (s)", "seconds", False),
        ("tpot_ms", "Time per output token (ms)", "ms/token", False),
        (None, "Accepted tokens per pass", "tokens", False),
        ("prompt_kb_s", "Prompt throughput (KB/s)", "KB/s", False),
        ("gen_bytes_s", "Generation throughput (B/s)", "bytes/s", False),
        ("ratio", "Cold / warm TTFT ratio", "ratio", False),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17.0, 12.5))
    for ax, (key, title, ylab, logy) in zip(axes.flat, panels):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_xlabel("prompt tokens", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if key is None:
            ax.text(0.5, 0.5,
                    "untested (pending)\n\nNo SpecDecoding metrics line fell inside\n"
                    "any sample window in this run, so acceptance\n"
                    "per pass is not derivable from this receipt.\n"
                    "Depth-sweep acceptance is in the k-sweep table.",
                    ha="center", va="center", fontsize=10, color="#555555",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        for arm, lab, color, marker in ARMS:
            ax.plot(x, series[arm][key], marker=marker, color=color, label=lab,
                    linewidth=1.8, markersize=5)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=9)

    fig.suptitle(
        "GLM-5.3-Flash AWQ W4A16 + MTP k=3 on 4x CMP 170HX (PP4 14,12,12,7) "
        "— thinking on vs off",
        fontsize=16, fontweight="bold", y=0.988)
    fig.text(
        0.5, 0.958,
        "vLLM sm80 (pp-dflash2 overlay) | MTP k=3 | KV dtype auto | gpu-mem-util 0.90 | "
        "max-model-len 393,216 | prefix caching OFF | temp 0, 512 output tokens | "
        "3 reps + cold/warm pair per point | ONE boot, arms alternated | median of reps",
        ha="center", fontsize=9.5)
    fig.text(
        0.5, 0.930,
        "MEASURED ON A DEGRADED LINK: GPU1 PCIe Gen1 x1 (slot ceiling x8), GPU0 x8, GPU2/3 x16, no NVLink. "
        "Prompt-processing and TTFT panels are lower bounds;\n"
        "generation tok/s and ms/token are link-insensitive under PP4 (~50 KB per decode step). "
        "Warm == cold because prefix caching is off in the recipe of record.\n"
        "The two arms are identical pending thinking-switch verification. "
        "258k point: untested (prompt calibration overshot the 393,216-token limit).",
        ha="center", va="top", fontsize=8.5, color="#8a2b06", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.900))
    fig.savefig(STEM + ".png", dpi=150)
    fig.savefig(STEM + ".svg")
    print("wrote", STEM + ".png")
    print("wrote", STEM + ".svg")
    print("points:", labels)


if __name__ == "__main__":
    main()
