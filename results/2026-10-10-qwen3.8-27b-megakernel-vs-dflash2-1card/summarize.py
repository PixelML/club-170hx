#!/usr/bin/env python3
"""Build summary.csv and the chart from the committed receipts (no GPU, no network).

  python3 results/2026-10-10-qwen3.8-27b-megakernel-vs-dflash2-1card/summarize.py
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RX = HERE / "receipts"
CHART = REPO / "assets" / "charts" / "2026-10-10-qwen3.8-27b-megakernel-vs-dflash2-1card.png"
ENGINES = {"mk-d4": "megakernel MTP d=4", "mk-d3": "megakernel MTP d=3", "mk-d0": "megakernel no spec",
           "vllm-dflash2-k7": "vLLM DFlash2 k=7"}


def load(path):
    rows = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[(r["engine"], r["suite"], r["case"])].append(r)
    return rows


def summary(rows, cap_w):
    out = []
    for (engine, suite, case), v in sorted(rows.items()):
        out.append(dict(cap_w=cap_w, engine=engine, suite=suite, case=case, n=len(v),
                        prompt_tokens=v[0]["prompt_tokens"],
                        completion_tokens_mean=round(statistics.mean(x["completion_tokens"] for x in v), 1),
                        decode_tok_s_mean=round(statistics.mean(x["decode_tok_s"] for x in v), 2),
                        decode_tok_s_min=min(x["decode_tok_s"] for x in v),
                        decode_tok_s_max=max(x["decode_tok_s"] for x in v),
                        ttft_s_mean=round(statistics.mean(x["ttft_s"] for x in v), 3),
                        prefill_tok_s_mean=round(statistics.mean(x["prefill_tok_s"] for x in v), 1)))
    return out


def telemetry(path):
    rows = list(csv.DictReader(path.open()))
    busy = [r for r in rows if float(r["util_pct"]) > 50]
    return dict(file=path.name, samples=len(rows), peak_core_c=max(float(r["temp_gpu_c"]) for r in rows),
                peak_mem_c=max(float(r["temp_mem_c"]) for r in rows),
                peak_w=max(float(r["power_w"]) for r in rows),
                mean_busy_w=round(statistics.mean(float(r["power_w"]) for r in busy), 1) if busy else None)


def build():
    s = summary(load(RX / "180w" / "mk.jsonl"), 180) + summary(load(RX / "180w" / "vllm.jsonl"), 180) \
        + summary(load(RX / "250w-thermal-stop" / "mk.jsonl"), 250)
    with (HERE / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(s)
    club = [json.loads(l) for l in (RX / "180w" / "vllm-club-suite.jsonl").read_text().splitlines()]
    club = [r for r in club if r.get("type") == "summary"]
    tel = [telemetry(p) for p in sorted((RX / "180w").glob("telemetry-*.csv"))] + \
          [dict(telemetry(p), file="250w/" + p.name) for p in sorted((RX / "250w-thermal-stop").glob("telemetry-*.csv"))]
    return s, club, tel


def chart(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = {(r["engine"], r["suite"], r["case"]): r for r in s if r["cap_w"] == 180}
    rows = [("greedy", "code_prime", "Code: prime function"), ("greedy", "hashmap", "Explain: hash map"),
            ("greedy", "flask_fastapi", "Plan: Flask to FastAPI"), ("greedy", "story_robot", "Story: robot (club prompt)"),
            ("greedy", "story_lighthouse", "Story: lighthouse")]
    labels = [r[2] for r in rows] + ["Thinking on, 4 prompts (mean)"]

    def think_mean(engine):
        return statistics.mean(r["decode_tok_s_mean"] for k, r in idx.items() if k[0] == engine and k[1] == "think")

    mk = [idx[("mk-d4", a, b)]["decode_tok_s_mean"] for a, b, _ in rows] + [think_mean("mk-d4")]
    vl = [idx[("vllm-dflash2-k7", a, b)]["decode_tok_s_mean"] for a, b, _ in rows] + [think_mean("vllm-dflash2-k7")]
    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=surface)
    ax.set_facecolor(surface)
    h = 0.36
    ys = list(range(len(labels)))[::-1]
    ax.barh([y + h / 2 + 0.01 for y in ys], vl, height=h, color="#eb6834", label="vLLM 0.27.1 + DFlash2 k=7 (W4A16)")
    ax.barh([y - h / 2 - 0.01 for y in ys], mk, height=h, color="#2a78d6", label="megakernel, MTP d=4 (Q4_K_M)")
    for y, a, b in zip(ys, vl, mk):
        ax.text(a + 4, y + h / 2, f"{a:.0f}", va="center", fontsize=9, color=ink)
        ax.text(b + 4, y - h / 2, f"{b:.0f}", va="center", fontsize=9, color=ink)
    ax.set_yticks(ys, labels, color=ink, fontsize=9)
    ax.set_xlabel("Single-stream decode (tok/s, usage-counted, mean of 3 greedy / 2 seeds thinking)", color=ink2, fontsize=9)
    ax.set_title("Qwen3.8-27B on 1x CMP 170HX at 180 W: megakernel vs vLLM DFlash2", color=ink, fontsize=11, loc="left")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", colors=ink2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_xlim(0, max(vl) * 1.12)
    fig.tight_layout()
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART, dpi=160, facecolor=surface)
    return fig


if __name__ == "__main__":
    s, club, tel = build()
    chart(s)
    print(f"wrote {HERE / 'summary.csv'} and {CHART}")
