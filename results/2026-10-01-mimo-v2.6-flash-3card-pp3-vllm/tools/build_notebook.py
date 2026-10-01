#!/usr/bin/env python3
"""Build notebooks/2026-10-01-mimo-v2.6-flash-3card-pp3-vllm.ipynb from the receipts in
results/2026-10-01-mimo-v2.6-flash-3card-pp3-vllm/ and the prior x16 receipts in
results/2026-09-22-mimo-v2.6-flash-4card-pp4-vllm/receipts/x16/. Run from the repo root.
Hero numbers are computed from the receipts at build time."""
import json
import os
import statistics as st

import nbformat as nbf

EXP = "2026-10-01-mimo-v2.6-flash-3card-pp3-vllm"
PRIOR = "results/2026-09-22-mimo-v2.6-flash-4card-pp4-vllm/receipts/x16"
NEW = f"results/{EXP}/runs"
NB = f"notebooks/{EXP}.ipynb"
IMAGE = "ghcr.io/pixelml/club-170hx@sha256:62f612b49614523e6a46e1493d35d3efd1f363917129d38cc923a31053693bfb"
CFGS = [c for c in ("pp3", "pp3-mtp2") if os.path.exists(f"{NEW}/{c}/p1.json")]


def p1(path):
    d = json.load(open(path))
    return st.median(v["median_decode_tok_s"] for v in d["runs"].values())


def p2(path):
    reps = json.load(open(path))["runs"]["longform"]["reps"]
    return st.median(r["decode_tok_s"] for r in reps[1:])


def conc(path):
    return {l["concurrency"]: l["aggregate_tok_s"] for l in json.load(open(path))["levels"]}


def prefill(path):
    rows = [json.loads(l) for l in open(path)]
    return st.median(r["prefill_tok_s"] for r in rows), rows[0]["prompt_tokens"], st.median(r["ttft_s"] for r in rows)


h = {}
for c in CFGS:
    h[c] = {"p1": p1(f"{NEW}/{c}/p1.json"), "p2": p2(f"{NEW}/{c}/p2.json"), "conc": conc(f"{NEW}/{c}/conc.json"),
            "prefill": prefill(f"{NEW}/{c}/prefill.jsonl") if os.path.exists(f"{NEW}/{c}/prefill.jsonl") else None,
            "prior_p1": p1(f"{PRIOR}/{c}/p1.json"), "prior_p2": p2(f"{PRIOR}/{c}/p2.json"), "prior_conc": conc(f"{PRIOR}/{c}/conc.json")}

best = max(((c, lvl, v) for c in CFGS for lvl, v in h[c]["conc"].items()), key=lambda x: x[2])
lead = "pp3-mtp2" if "pp3-mtp2" in CFGS else "pp3"
pf = h[lead]["prefill"]
hero_rows = [
    f"| Decode, c=1, greedy (P1 median, 5 workloads) | " + " · ".join(f"**{h[c]['p1']:.1f} tok/s** {c.upper().replace('-MTP2', ' + MTP k=2')} (was {h[c]['prior_p1']:.1f})" for c in reversed(CFGS)) + " |",
    f"| Best aggregate | {best[2]:.0f} tok/s at c={best[1]} ({best[0].upper().replace('-MTP2', ' + MTP k=2')}) |",
    f"| Prefill | {pf[0]:,.0f} tok/s at {pf[1]:,} prompt tokens (c=1, uncached) |" if pf else "| Prefill | untested |",
    f"| TTFT | {pf[2]:.2f} s at {pf[1]:,} prompt tokens |" if pf else "| TTFT | untested |",
]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
    md(f"""# MiMo-V2.6-Flash-RL on 3× CMP 170HX (SM80) — vLLM PP3, Gen2 x16 behind PLX switches

| Metric | Value |
|---|---|
""" + "\n".join(hero_rows) + f"""

![decode and aggregate vs the 2026-09-24 x16 run](../assets/charts/{EXP}.png)

```bash
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --revision 5711b268169967567844e1e560e8a3966da959b1 --local-dir <weights>
```

[Previous run and full lane history: 2026-09-22 MiMo notebook](2026-09-22-mimo-v2.6-flash-4card-pp4-vllm.ipynb)"""),
    code(f"""# --- Status cell -------------------------------------------------------
# LIVE = False replays the committed receipts. The live path is the previous
# notebook's tools (gate.py, bench_mimo.py, conc_sweep.py) against a server
# started as in section 3.
import os

EXPERIMENT = "{EXP}"
RESULTS_DIR = os.path.join("..", "results", EXPERIMENT)
PRIOR_DIR = os.path.join("..", "{PRIOR}")
LIVE = False
print("LIVE =", LIVE, "| receipts:", RESULTS_DIR, "| prior:", PRIOR_DIR)"""),
    code(f"""import json, statistics as st
from IPython.display import display, Markdown

CFGS = {CFGS!r}
LABEL = {{"pp3": "PP3", "pp3-mtp2": "PP3 + MTP k=2"}}


def J(*p):
    return json.load(open(os.path.join(*p)))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    display(Markdown("\\n".join(out)))


def p1m(d):
    return st.median(v["median_decode_tok_s"] for v in d["runs"].values())


def p2m(d):
    return st.median(r["decode_tok_s"] for r in d["runs"]["longform"]["reps"][1:])


new = {{c: {{k: J(RESULTS_DIR, "runs", c, k + ".json") for k in ("p1", "p2", "conc", "gate", "boot")}} for c in CFGS}}
old = {{c: {{k: J(PRIOR_DIR, c, k + ".json") for k in ("p1", "p2", "conc", "gate")}} for c in CFGS}}
pre = {{}}
for c in CFGS:
    p = os.path.join(RESULTS_DIR, "runs", c, "prefill.jsonl")
    pre[c] = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
env = J(RESULTS_DIR, "environment.json")
print("configs:", CFGS)"""),
    md("""## 1. TL;DR

Same image, patches, launch arguments and protocol as the 2026-09-24 three-card x16 re-run
(`receipts/x16/` of the previous notebook); what changed is the host. The earlier run had three
cards on direct CPU root ports at **Gen1 x16** in a VM; this one has three cards at **Gen2 x16**
behind two PLX PEX 8747 switches, bare metal, no P2P, same 180 W cap.

**Measured verdict: no gain from the faster link.** Single-stream decode is within 1–3 % of the
prior run (PP3 74.6 vs 75.5, PP3 + MTP k=2 113.8 vs 117.7 tok/s greedy), and aggregate at c=32 is
450 vs 455 (PP3) and 518 vs 561 (MTP). PP decode is HBM-bound and hands only small activations
between stages, so doubling the link rate does not show. All gates 4/4 correct, no thermal stop,
uncached prefill 4,107 tok/s (PP3) / 4,082 tok/s (MTP) at ~20.3k prompt tokens."""),
    code("""rows = []
for c in CFGS:
    rows.append([LABEL[c], "measured 2026-10-01", "all correct" if all(g["correct"] for g in new[c]["gate"]) else "FAIL",
                 f"{p1m(new[c]['p1']):.1f}", f"{p2m(new[c]['p2']):.1f}",
                 max(l["aggregate_tok_s"] for l in new[c]["conc"]["levels"]), new[c]["boot"]["cold_boot_s"]])
    rows.append([LABEL[c], "2026-09-24 x16 (prior)", "all correct" if all(g["correct"] for g in old[c]["gate"]) else "FAIL",
                 f"{p1m(old[c]['p1']):.1f}", f"{p2m(old[c]['p2']):.1f}",
                 max(l["aggregate_tok_s"] for l in old[c]["conc"]["levels"]), "—"])
table(["Config", "Run", "Gate (4 prompts × 3)", "P1 greedy c=1 (tok/s)", "P2 T=1.0 c=1 (tok/s)", "Best aggregate (tok/s)", "Cold boot (s)"], rows)
table(["Pin", "Value"], list(env.items()))"""),
    md("## 2. Visible results\n\n### P1 per workload (greedy, 512 tokens, median of 3; tok/s)"),
    code("""for c in CFGS:
    rows = [[w, v["median_decode_tok_s"], old[c]["p1"]["runs"][w]["median_decode_tok_s"], v["flagged_reps"]] for w, v in new[c]["p1"]["runs"].items()]
    display(Markdown(f"**{LABEL[c]}**"))
    table(["Workload", "This run (tok/s)", "Prior x16 run (tok/s)", "Repeat-flagged reps"], rows)"""),
    md("### Concurrency (256 tokens, T=1.0, `ignore_eos`, 2 rounds; aggregate tok/s)"),
    code("""levels = sorted({l["concurrency"] for c in CFGS for l in new[c]["conc"]["levels"]})
rows = []
for lv in levels:
    r = [lv]
    for c in CFGS:
        n = {l["concurrency"]: l["aggregate_tok_s"] for l in new[c]["conc"]["levels"]}
        o = {l["concurrency"]: l["aggregate_tok_s"] for l in old[c]["conc"]["levels"]}
        r += [n.get(lv, "not run"), o.get(lv, "—")]
    rows.append(r)
table(["Concurrency"] + [f"{LABEL[c]} {w}" for c in CFGS for w in ("this run", "prior")], rows)"""),
    md("### Prefill (uncached unique prompt, `max_tokens=1`, c=1)"),
    code("""rows = [[LABEL[c], r["sample"], r["prompt_tokens"], r["ttft_s"], r["prefill_tok_s"]] for c in CFGS for r in pre[c]]
table(["Config", "Sample", "Prompt tokens", "TTFT (s)", "Prefill (tok/s)"], rows) if rows else print("prefill: untested")"""),
    code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
x = range(len(CFGS)); w = 0.38
a1.bar([i - w / 2 for i in x], [p1m(old[c]["p1"]) for c in CFGS], w, label="2026-09-24: Gen1 x16, root ports, VM", color="#8c8c8c")
a1.bar([i + w / 2 for i in x], [p1m(new[c]["p1"]) for c in CFGS], w, label="2026-10-01: Gen2 x16, PLX switches, bare metal", color="#4c72b0")
for i, c in enumerate(CFGS):
    a1.text(i - w / 2, p1m(old[c]["p1"]) + 1, f"{p1m(old[c]['p1']):.1f}", ha="center", fontsize=9)
    a1.text(i + w / 2, p1m(new[c]["p1"]) + 1, f"{p1m(new[c]['p1']):.1f}", ha="center", fontsize=9)
a1.set_xticks(list(x)); a1.set_xticklabels([LABEL[c] for c in CFGS]); a1.set_ylabel("Decode, c=1 greedy (tokens/second)")
a1.set_title("P1 median of 5 workloads"); a1.legend(fontsize=7); a1.grid(axis="y", alpha=.3)
for c, col in zip(CFGS, ["#4c72b0", "#c44e52"]):
    n = new[c]["conc"]["levels"]; o = old[c]["conc"]["levels"]
    a2.plot([l["concurrency"] for l in n], [l["aggregate_tok_s"] for l in n], marker="o", color=col, label=f"{LABEL[c]} this run")
    a2.plot([l["concurrency"] for l in o], [l["aggregate_tok_s"] for l in o], marker="x", ls="--", color=col, alpha=.6, label=f"{LABEL[c]} prior x16")
a2.set_xlabel("Concurrency (simultaneous requests)"); a2.set_ylabel("Aggregate decode (tokens/second)")
a2.set_title("Aggregate throughput"); a2.legend(fontsize=7); a2.grid(alpha=.3)
fig.suptitle("MiMo-V2.6-Flash-RL, 3x CMP 170HX PP3, 180 W — measured 2026-10-01 vs 2026-09-24", fontsize=10)
fig.tight_layout()
chart = os.path.join("..", "assets", "charts", EXPERIMENT)
fig.savefig(chart + ".png", dpi=130); fig.savefig(chart + ".svg"); plt.close(fig)
display(Image(chart + ".png"))"""),
    md(f"""## 3. Reproduce

Identical to section 5 of the [2026-09-22 notebook](2026-09-22-mimo-v2.6-flash-4card-pp4-vllm.ipynb)
(weights, `VLLM_PP_LAYER_PARTITION=17,16,15`, patch mounts, `--gpu-memory-utilization 0.97
--max-model-len 32768 --max-num-seqs 32`), with the image pinned by digest:

```bash
P=$PWD/results/2026-09-22-mimo-v2.6-flash-4card-pp4-vllm/patches
docker run -d --name mimo --gpus all --shm-size 32g \\
  -e VLLM_PP_LAYER_PARTITION=17,16,15 -e VLLM_USE_V2_MODEL_RUNNER=1 \\
  -v <weights>:/models \\
  -v $P/mimo_v2.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py:ro \\
  -v $P/mimo_v2_mtp.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_mtp.py:ro \\
  -p 127.0.0.1:8000:8000 {IMAGE} \\
  --model /models/mimo-v2.6-flash-rl --served-model-name mimo-v2.6-flash \\
  --pipeline-parallel-size 3 --trust-remote-code --gpu-memory-utilization 0.97 \\
  --reasoning-parser mimo --max-num-seqs 32 --max-model-len 32768 \\
  --speculative-config '{{"method":"mimo_mtp","num_speculative_tokens":2}}'
```

Plain PP3: drop `VLLM_USE_V2_MODEL_RUNNER`, the MTP mount and `--speculative-config`. Bench with
the previous notebook's `tools/gate.py`, `tools/bench_mimo.py --protocol p1|p2`,
`tools/conc_sweep.py`, and this bundle's `tools/prefill_probe.py`.

Host notes: the unlocked cards share PLX switches here, so the unlock needs the BAR1-resize
serialization patch from the 2026-10-01 interconnect notebook; the WPR fix from the previous
notebook is already upstream in the unlock version used."""),
    md("""## 4. Appendix

<details>
<summary>Differences from the prior run, limitations</summary>

- Prior run: 3 cards at Gen1 x16 on CPU root ports, inside a VM (passthrough), 180 W. This run:
  3 cards at Gen2 x16 behind two PLX switches, bare metal, 180 W, no P2P. Pipeline stages hand
  activations between cards through host memory in both cases.
- Chassis fans were held at a fixed high duty for the run (one card's slot has weak airflow;
  a separate run on that card hit the 80 °C stop). The 80 °C / Xid guard was active; see
  `runs/*/telemetry.csv`.
- Prefill here uses a fresh ~21.5k-token prompt per sample (`tools/prefill_probe.py`); the
  prior run's prefill receipts are not in the repository, so no prefill comparison is made.
- DFlash k=7 was not re-run.
- MTP k=2 mean acceptance length during the concurrency sweep: 2.20–2.24 (`runs/pp3-mtp2/spec-stats.txt`), in line with the prior run's ~2.1.

</details>"""),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, NB)
print("wrote", NB, "configs:", CFGS)
