#!/usr/bin/env python3
"""Chart source for 2026-09-19-bonsai-2-27b-w4a16-dflash2-1card-vllm-*.{png,svg}.

Regenerate each with:
    python3 assets/charts/2026-09-19-bonsai-2-27b-w4a16-dflash2-1card-vllm-<name>.py

Reads committed receipts; no network, no GPU.
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
REC = os.path.join(REPO, "results", "2026-09-19-bonsai-2-27b-w4a16-dflash2-1card-vllm", "receipts")
STEM = os.path.join(HERE, "2026-09-19-bonsai-2-27b-w4a16-dflash2-1card-vllm")


def rows():
    return [json.loads(l) for l in open(os.path.join(REC, "serve-w4a16.jsonl")) if l.strip()]


def med(vals):
    return st.median(vals) if vals else None


# ---- 1. served decode cohorts -------------------------------------------------
def chart_decode():
    r = rows()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cohorts = [("decode256", "256-token cohort", "#1f6feb"), ("decode900", "900-token cohort", "#e8590c")]
    width = 0.35
    for i, (tag, label, color) in enumerate(cohorts):
        samples = [x["decode_tok_s"] for x in r if x["tag"].startswith(f"{tag}-sample")]
        wall = [256 / x["wall_s"] if tag == "decode256" else 900 / x["wall_s"] for x in r if x["tag"].startswith(f"{tag}-sample")]
        ax.bar(i - width / 2, med(samples), width, color=color, label=f"{label} (ttft-excl. {med(samples):.0f})")
        ax.bar(i + width / 2, med(wall), width, color=color, alpha=0.45, label=f"{label} (wall {med(wall):.0f})")
    ax.axhline(147.7, color="#adb5bd", ls="--", lw=1.2, label="Qwen3.8-27B W4A16 + DFlash2 receipt (147.7, community/own)")
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([c[1] for c in cohorts])
    ax.set_ylabel("decode tok/s (single stream, usage-counted)")
    ax.set_title("Bonsai-2 27B W4A16 + DFlash2 k=7 on vLLM 0.28.0, 1x CMP 170HX (180 W cap)\n"
                 "greedy, ignore_eos, 10-token prompt; solid = ttft-excluded, light = wall rate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(STEM + "-serve-decode.png", dpi=150)
    fig.savefig(STEM + "-serve-decode.svg")
    print("saved serve-decode")


# ---- 2. acceptance profile ----------------------------------------------------
def chart_acceptance():
    m = json.load(open(os.path.join(REC, "summary.json")))
    per_pos = m["results"]["acceptance"]["accepted_per_pos"]
    drafts = m["results"]["acceptance"]["drafts"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar([f"pos {i}" for i in range(len(per_pos))], [p / drafts for p in per_pos], color="#1f6feb")
    for rect, p in zip(bars, per_pos):
        ax.annotate(f"{p/drafts:.2f}", xy=(rect.get_x() + rect.get_width()/2, p/drafts),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("accepted tokens per draft position")
    ax.set_ylim(0, 1)
    ax.set_title(f"Bonsai-2 27B + DFlash2: acceptance per draft position\n"
                 f"mean {m['results']['acceptance']['mean_accepted_per_draft']:.2f} accepted tokens/draft "
                 f"({m['results']['acceptance']['accepted_total']}/{m['results']['acceptance']['draft_tokens']} = "
                 f"{m['results']['acceptance']['acceptance_rate']:.1%}), base-calibrated drafter, zero-shot for Bonsai-2")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(STEM + "-acceptance.png", dpi=150)
    fig.savefig(STEM + "-acceptance.svg")
    print("saved acceptance")


# ---- 3. ladder ----------------------------------------------------------------
def chart_ladder():
    r = rows()
    lad = {}
    for x in r:
        if "SUMMARY" in x["tag"]:
            lad.setdefault(x["concurrency"], []).append(x["decode_tok_s"])
    cs = sorted(lad)
    agg = [st.median(lad[c]) for c in cs]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(cs, agg, "o-", color="#1f6feb")
    for c, v in zip(cs, agg):
        ax.annotate(f"{v:.0f}", (c, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(cs)
    ax.set_xlabel("concurrency (server slots, MAX_SEQS=1 recipe)")
    ax.set_ylabel("aggregate tok/s (wall)")
    ax.set_title("Bonsai-2 27B W4A16 + DFlash2 concurrency ladder, 1x CMP 170HX (180 W cap)\n"
                 "single-slot recipe: requests queue, aggregate = the single-stream wall rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(STEM + "-ladder.png", dpi=150)
    fig.savefig(STEM + "-ladder.svg")
    print("saved ladder")


# ---- 4. power -----------------------------------------------------------------
def chart_power():
    import csv
    import datetime
    ws, ts = [], []
    with open(os.path.join(REC, "nvidia-vllm.csv")) as fh:
        for row in csv.DictReader(fh):
            ts.append(datetime.datetime.strptime(row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f"))
            ws.append(float(row[" power.draw [W]"].split()[0]))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    t0 = ts[0]
    xs = [(t - t0).total_seconds() for t in ts]
    ax.plot(xs, ws, lw=0.8, color="#e8590c")
    ax.axhline(180, color="#c92a2a", ls="--", lw=1, label="180 W cap")
    ax.axhline(st.mean(ws), color="#1f6feb", ls=":", lw=1, label=f"mean {st.mean(ws):.0f} W")
    ax.set_xlabel("seconds since telemetry start")
    ax.set_ylabel("board power draw (W)")
    ax.set_title("Bonsai-2 27B W4A16 + DFlash2 serving window, 1 Hz telemetry, 1x CMP 170HX\n"
                 "peak 219 W transient; decode sits well under the cap")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 240)
    fig.tight_layout()
    fig.savefig(STEM + "-power.png", dpi=150)
    fig.savefig(STEM + "-power.svg")
    print("saved power")


if __name__ == "__main__":
    chart_decode()
    chart_acceptance()
    chart_ladder()
    chart_power()
