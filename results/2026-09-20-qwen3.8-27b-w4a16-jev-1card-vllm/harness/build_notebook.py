#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the Jev-endpoint notebook from a cell spec.

The notebook is generated rather than hand-edited so the receipts it reads and
the numbers it states stay in one place.

  python build_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "notebooks" / "2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm.ipynb"

CELLS = [
    ("md", r"""# Qwen3.8-27B (W4A16 GPTQ) as a no-train Jev endpoint on 1x CMP 170HX (SM80) — 1card, vLLM

| Metric | Value |
|---|---|
| Answer, idle c=1 (p50 warm prefix cache / cache-busted) | 141.7 ms / 133.0 ms |
| Answer, production load (16 clients, 5 reads per annotation) | mean 12.0 s; 99.8% answered within 15 s |
| First production run (2026-09-21) | 19,045 annotations in 4.5 h, 0 failures, 1,952 tok/s prefill at 178 W |
| Training | **none** — no-train method: stock checkpoint served read-only, nothing fitted that generalises |
| Labelled accuracy (42 author-labelled reads, T=1) | 0.571 (95% CI 0.42-0.71) |
| Calibration | ECE 0.274 at T=1; the fitted T does not generalise (leave-one-out 0.301) |
| Option-order sensitivity (max probability shift) | 0.639, argmax unstable |
| Read determinism (sequential / interleaved / 2 concurrent) | identical / identical / identical |

![production load: throughput and answer time](../assets/charts/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm-load.png)

```
hf download Qwen/Qwen3.8-27B-GPTQ-4bit --local-dir <weights>
```

Model guide: [Qwen3.8-27B](../docs/models/qwen3.8-27b.md) · serving recipe credit: [Kis's DFlash2 notebook](2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb)
"""),
    ("md", r"""This notebook shows that **Qwen3.8-27B can serve as a Jev-compatible
endpoint** on one CMP 170HX: a `POST /v1/systemone` request returns a calibrated
probability per option, for `choice`, `noul` (yes/no) and `score` questions,
without generating a single token. It is a **no-train method**: the stock
W4A16 checkpoint is served read-only, and everything that shapes the answer is
a read-time choice (which tokens are allowed, what order the options are
presented in, a temperature divisor applied after the read). No head, no
projector, no fine-tune — the sibling
[LLKVApprox notebooks](2026-09-11-qwen3.8-27b-llkvapprox-1card-hf.ipynb) train
a projector; this one trains nothing. And the one quantity that *could* have
been fitted — a single calibration temperature — failed leave-one-out at n=42
(section 2.6), so the configuration this notebook ships learns nothing at all.
Qwen3.8-27B is the checkpoint this no-train recipe is measured on here.

The serving stack is the one Kis established in
[`2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm`](2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb)
— his notebook is the reference for *how this checkpoint is served on this
card* (W4A16 on one 170HX, the syv-ai recipe lineage, 180 W-safe, ~140 tok/s
generation, ~1,955 tok/s prefill at a 180 W cap). This notebook does **not**
re-measure generation; it adds the classification path on top of that runtime
and reports what that path measures — first as probes (2026-09-20), then under
a real workload (2026-09-21).

The Jev contract itself is documented in the
[`jev` branch of kishida's llama.cpp fork](https://github.com/kishida/llama.cpp/blob/jev/docs/jev.md)
— shape-compatible with [TypeSafe's System One API](https://docs.typesafe.ai/api):
the contract and the read mechanic, **not** the calibration — TypeSafe trains
Jev for that (RLCD), and this stack ships uncalibrated reads at T=1 (appendix,
"Positioning vs TypeSafe Jev"). Credit for the design — one prompt
evaluation, label logits, `confidence = 1 - normalized entropy`, expected-level
`score`, temperature as a calibration divisor, permutations to average out
option order — belongs to kishida's work. What is ours here is the vLLM
implementation of it and the findings in section 2 that a vLLM host has to
know.

On 2026-09-21 the endpoint served its **first production workload**: an
LLM-as-judge annotation pilot speaking this exact contract — 19,045
annotations, five permutation reads each, 16 clients, 4.5 h, zero failures,
1,952 tok/s prefill at 178 W. Section 2.9 carries those receipts; the answer
time goes from ~142 ms idle to a 12.0 s mean under that load, and the chart in
the hero cell shows both.

Executed on a single CMP 170HX (SM80) on 2026-09-20, with production-load
receipts frozen on 2026-09-21. `LIVE = False`: every number below is read from
the committed receipts under
`results/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm/`.
"""),
    ("code", r"""# --- Status cell -------------------------------------------------------
# LIVE = False replays the committed receipts under results/<experiment>/.
# LIVE = True runs the same harness against a running pair of endpoints whose
# URL comes from the environment. Never commit a notebook executed with LIVE = True.
import os

EXPERIMENT = "2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm"
RESULTS_DIR = os.path.join("..", "results", EXPERIMENT)
RECEIPTS = os.path.join(RESULTS_DIR, "receipts")
LIVE = False
JEV_ENDPOINT_ENV_VAR = "JEV_API"      # the interposer, /v1/systemone
VLLM_ENDPOINT_ENV_VAR = "JEV_UPSTREAM"  # the vLLM OpenAI server

print(f"experiment : {EXPERIMENT}")
print(f"receipts   : {RECEIPTS}")
print(f"LIVE       : {LIVE}")
if LIVE:
    for var in (JEV_ENDPOINT_ENV_VAR, VLLM_ENDPOINT_ENV_VAR):
        if not os.environ.get(var):
            raise RuntimeError(f"LIVE=True but {var} is not set")
    print("endpoints  : from the environment (not printed)")
"""),
    ("code", r"""# --- Helpers ------------------------------------------------------------
import json
import os


def receipt(name):
    # One JSON receipt from this experiment's receipts directory.
    with open(os.path.join(RECEIPTS, name)) as f:
        return json.load(f)


def jsonl(name):
    rows = []
    with open(os.path.join(RECEIPTS, name)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text(name):
    with open(os.path.join(RECEIPTS, name), errors="replace") as f:
        return f.read()


from IPython.display import display, Markdown


def render_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    display(Markdown("\n".join(lines)))
"""),
    ("md", r"""## 1. TL;DR

**Verdict: the classification contract works on this card, it survived its
first production workload, and the label probabilities are exact and
repeatable — all with no training. The honest reading of the probabilities is
the open part:** on 42 author-labelled reads the endpoint is right 57% of the
time, over-confident at T=1 (ECE 0.274), and the answer changes with the order
the options are written in.

Three things a vLLM host has to know, each with a receipt in section 2:

1. **`logprobs_mode` decides whether the label distribution exists at all.**
   vLLM's default (`raw_logprobs`) computes logprobs *before*
   `allowed_token_ids`, so a label-masked read returns the unmasked top-k and
   the labels are simply missing. `--logprobs-mode processed_logprobs` fixes
   it. This is not a documentation footnote — it is the difference between a
   working endpoint and a 502.
2. **The checkpoint's own `generation_config` silently filters the labels.**
   It ships `top_k=20, top_p=0.95`, vLLM adopts those as server defaults, and
   in `processed_logprobs` mode the returned logprobs are post-filter: on a
   four-option question one label is dropped and the survivors are inflated.
   The read path sets `top_p=1.0, top_k=-1` per request.
3. **Option order moves the answer.** Rotating a three-option question's order
   shifts the probability mass by up to 0.639 and flips the argmax. Jev's
   `permutations` option exists for exactly this and the endpoint implements
   it; the un-averaged read is the one to be careful with.

What the first production workload added (section 2.9): on 2026-09-21 an
LLM-as-judge annotation pilot drove this endpoint for 4.5 h — 19,045
annotations (five permutation reads each, 16 clients), zero failures,
1,952 tok/s prefill at a 178 W mean, within 0.2% of the ~1,955 tok/s Kis
measured on a short synthetic at the same 180 W cap. The **answer in ms**:
~142 ms idle at c=1, **12.0 s mean under load** (99.8% of reads within 15 s) —
queue-bound, not compute-bound. And one operational surprise: the engine's
prefix cache hit **0.0%** on this traffic, so every read honestly pays its
~1.5k-token prefill.

Then the calibration picture, stated plainly: the probabilities are
*measurable* and *reproducible* — identical to the last bit on repeat and
across concurrency — but on a 42-example descriptive set they are
over-confident, and **the fitted temperature does not generalise at this
sample size**. In-sample, fitting T by NLL improves ECE from 0.274 to 0.223;
under leave-one-out fitting it comes out *worse* than leaving T=1 alone (ECE
0.301). Read that as: the fitting loop works and is worth running, but n=42
cannot certify a temperature, and the in-sample number is not the one to
quote.

One caveat on the word *reproducible*: the endpoint serves one read per
request and those are bit-identical on repeat. Reads placed in the **same
batch** are not — two copies of the same prompt agreed exactly over HTTP and
differed by up to 0.067 in logprob when batched together in-process (section
4). Bit-identity is a property of a fixed batch composition, not of the model.
"""),
    ("md", r"""### Pins

From `receipts/env.json`, which records the runtime versions, the model config
hash, the resolved engine configuration and the line of engine log that names
the architecture.
"""),
    ("code", r"""import re

env = receipt("env.json")
weights_gib = None
for line in env["log_excerpt"]:
    m = re.search(r"Model loading took ([\d.]+) GiB", line)
    if m:
        weights_gib = m.group(1)
pins = [
    ("Model (served)", "Qwen3.8-27B, W4A16 GPTQ (checkpoint `config.json` hash in the receipt)"),
    ("Architecture resolved by the engine", env["model"]["architectures"][0]),
    ("Quantization", f"{env['model']['quantization_config']['quant_method']} "
                     f"{env['model']['quantization_config']['bits']}-bit, "
                     f"group_size {env['model']['quantization_config']['group_size']}, "
                     f"desc_act {env['model']['quantization_config']['desc_act']}"),
    ("Linear kernel", "MarlinLinearKernel (from the serve log)"),
    ("Attention backend", "FLASH_ATTN (FlashAttention version 2)"),
    ("vLLM", env["measured"]["vllm"]),
    ("torch / CUDA", f"{env['measured']['torch']} / {env['measured']['cuda']}"),
    ("transformers", env["measured"]["transformers"]),
    ("Hardware", "1x NVIDIA CMP 170HX, 64 GiB HBM2e, SM80"),
    ("Topology", "1card, tensor_parallel_size=1, no speculative decoding"),
    ("Serving shape", f"max_model_len {env['serve']['max_model_len']}, "
                      f"gpu_memory_utilization {env['serve']['gpu_memory_utilization']}, "
                      f"max_logprobs {env['serve']['max_logprobs']}, "
                      f"prefix caching on"),
    ("Jev read mode", "`--logprobs-mode processed_logprobs` (not the default)"),
    ("Weights held by the engine", f"{weights_gib} GiB (from the serve log)"),
]
render_table(["Pin", "Value"], pins)
"""),
    ("md", r"""### Protocol

*Every* read is one `POST /v1/completions` with `max_tokens=1`: the prompt is
evaluated once and the logprobs of the first generated position are read. The
sampled token is discarded and nothing is generated, so the answer does not
depend on sampling.

| Setting | Value | Why |
|---|---|---|
| `temperature` (upstream) | 1.0 | identity; a requested temperature is applied afterwards as `softmax(logprobs / T)` |
| `top_p` / `top_k` (upstream) | 1.0 / -1 | neutralises the checkpoint's `generation_config` nucleus default |
| `allowed_token_ids` | the option symbols | makes the returned logprobs the label-set distribution |
| `logprobs` | number of labels | one logprob per label; 62 single-token symbols max |
| `return_tokens_as_token_ids` | true | label logprobs are matched by token id, not by text |
| prompt | Jev's format, model chat template | assistant generation prompt, `enable_thinking=False` |

Prompt format, label selection and the temperature/permutations semantics are
the ones in kishida's `jev` docs, with one deviation to be explicit about:
`enable_thinking=False` on this checkpoint still renders an empty
`<think>\n\n</think>\n\n` block before the label position (visible in
`receipts/prompts.json`). The read works because the empty block is part of the
generation prompt and the label follows it, but a prompt built by hand without
it is not the same prompt.
"""),
    ("code", r"""prompts = receipt("prompts.json")
rows = []
for f in prompts["fixtures"]:
    rows.append([f["case"], f["question"]["type"], f["prompt_tokens"],
                 len(f["option_names"]), " ".join(f["label_symbols"]),
                 f["prompt_sha256"][:16] + "..."])
render_table(["Fixture", "Question type", "Prompt tokens", "Options",
              "Label symbols", "Prompt sha256"], rows)
print("rendered prompt for the noul fixture:")
print(prompts["fixtures"][1]["rendered_prompt"])
"""),
    ("md", r"""## 2. Visible results

Every table and the chart are computed from the committed receipts. Raw
payloads, including full logprob dictionaries, are in the receipts themselves.
"""),
    ("md", r"""![reliability on 42 labelled reads and option-order rotation](../assets/charts/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm.png)
"""),
    ("md", r"""### 2.1 The finding that decides the whole thing: `logprobs_mode`

The same prompt, read twice with `allowed_token_ids=[labels]`. With vLLM's
default mode the payload holds the **unmasked** ranking — the labels are not in
it, so the endpoint cannot answer at all. With `processed_logprobs` the payload
is exactly the label set, and the mass over it sums to 1.

`labels_visible_unmasked` counts how many labels also appear in an unmasked
top-20 read; where they do, the *pairwise differences* of the logprobs are
identical between the two reads (they differ only by the normalisation
constant), which is the invariant that proves the whitelist precedes the
gather rather than following it.
"""),
    ("code", r"""raw = receipt("mask-raw_logprobs.json")
proc = receipt("mask-processed_logprobs.json")

rows = []
for r in raw["rows"]:
    rows.append(["raw_logprobs (default)", r["case"], len(r["masked_returned_ids"]),
                 r["returned_ids_equal_allowed"], r["count_equals_n_labels"],
                 "not auditable"])
for r in proc["rows"]:
    rows.append(["processed_logprobs", r["case"], len(r["masked_returned_ids"]),
                 r["returned_ids_equal_allowed"], r["count_equals_n_labels"],
                 f"{r['sum_exp_masked']:.6f}"])
render_table(["Mode", "Fixture", "Logprobs returned", "Returned == allowed",
              "Count == labels", "Mass over labels"], rows)

shared = [r for r in proc["rows"] if r["labels_visible_unmasked"]]
print("offset-invariance where labels are visible in both reads:")
for r in shared:
    print(f"  {r['case']}: {r['labels_visible_unmasked']} shared labels, "
          f"max |Δ(lp_i - lp_j)| = {r['max_abs_offset_delta']:.3e}")
"""),
    ("md", r"""### 2.2 The checkpoint's `generation_config` filters the labels

vLLM adopts the model's `generation_config.json` (`temperature 1.0, top_k 20,
top_p 0.95`) as the server's default sampling parameters — the serve log says
so. In `processed_logprobs` mode the returned logprobs are post-filter, so a
read that does not neutralise those defaults is a nucleus read: on the
four-option question the low-mass label is **dropped entirely** and the
surviving probabilities are inflated.

This is silent. Nothing errors; the endpoint returns a probability
distribution that is simply not the one the model computed.
"""),
    ("code", r"""nuc = receipt("nucleus.json")
rows = []
for r in nuc["rows"]:
    g = r["generation_config_default"]
    rows.append([r["case"], r["question_type"],
                 len(g["label_logprobs"]), len(g["dropped_labels"]),
                 ", ".join(map(str, g["dropped_labels"])) or "none",
                 f"{r['max_abs_probability_delta']:.4f}"
                 if r["max_abs_probability_delta"] is not None else "labels dropped"])
render_table(["Fixture", "Question type", "Labels returned under defaults",
              "Labels dropped", "Dropped ids",
              "Max |Δprobability| vs neutralised"], rows)
print("neutralised vs generation_config defaults, per fixture:")
for r in nuc["rows"]:
    print(f"  {r['case']}: neutralised {[round(p, 4) for p in r['neutralised']['probabilities']]}"
          f"  dropped={r['generation_config_default']['dropped_labels']}")
"""),
    ("md", r"""### 2.3 Determinism: the same prompt gives the same distribution

Three reads back to back, two reads around a different prompt, and two reads
fired concurrently. The label logprobs are compared; the sampled token is not
part of the answer, so its randomness is irrelevant.

This is a claim about an idle, single-tenant server at c=1. Continuous
batching changes kernel shapes, and some kernels are not batch-invariant, so
"identical under concurrency" here is measured for two concurrent requests and
is not a licence to assume it at high concurrency. The appendix measures the
effect directly: two copies of the same prompt agree exactly when they are
separate requests and differ by up to 0.067 in logprob when they share a
batch. Bit-identity belongs to a batch composition, not to the model.
"""),
    ("code", r"""det = receipt("determinism.json")
rows = [
    ["Sequential (3 reads, back to back)", det["sequential_identical"],
     "exact equality of label logprobs"],
    ["Interleaved (a different prompt in between)", det["interleaved_identical"],
     "same"],
    ["Concurrent (2 in flight)", det["concurrent_identical"],
     f"max |Δlogprob| = {det['concurrent_max_abs_delta']:.1e}"],
]
render_table(["Condition", "Identical", "Note"], rows)
print("label logprobs, sequential reads:")
for i, lp in enumerate(det["sequential_logprobs"]):
    print(f"  read {i}: yes {lp['yes']:.10f}  no {lp['no']:.10f}")
"""),
    ("md", r"""### 2.4 Temperature is a calibration divisor, applied after the read

Every read runs at T=1. A requested temperature is applied to the label
logprobs as `softmax(logprobs / T)`, so the same read serves any calibration
and changing T cannot change what was read. Note what T does and does not
move: it cannot change the argmax (dividing by a positive scalar preserves
order), but it does change `confidence` **and** the expected level of a
`score` question, because a flatter distribution shifts the probability-
weighted mean. The table checks the endpoint's probabilities against that
formula computed client-side.
"""),
    ("code", r"""temp = receipt("temperature.json")
rows = []
for case, blob in temp["cases"].items():
    for r in blob["rows"]:
        rows.append([case, blob["question_type"], r["temperature"],
                     " ".join(f"{p:.6f}" for p in r["api_probabilities"]),
                     f"{r['max_abs_delta']:.1e}"])
render_table(["Fixture", "Question type", "T",
              "Probabilities at T", "|API - client formula|"], rows)
"""),
    ("md", r"""### 2.5 Option order changes the answer

A three-option routing question, read once per rotation of the option order.
The probability read for "billing" stays between 0.481 and 0.536 — but it wins
in one rotation and loses in the other two, because the gap between the top two
options is smaller than the shift the order induces. Jev's `permutations`
option averages over rotations; the endpoint implements it and the last row
confirms the averaged answer is the mean of the individually measured
rotations.
"""),
    ("code", r"""perm = receipt("permutations.json")
rows = []
for r in perm["per_rotation"]:
    order = r["option_order"]
    pretty = "  ".join(f"{n}={p:.4f}" for n, p in zip(order, r["probabilities"]))
    rows.append([r["rotation"], ", ".join(order), pretty,
                 f"{r['l1_vs_rotation0']:.4f}", r["argmax_vs_rotation0"]])
render_table(["Rotation", "Order presented", "Probability per option",
              "L1 vs rotation 0", "Argmax same"], rows)
print(f"argmax stable across rotations: {perm['argmax_stable']}")
print(f"API permutations={len(perm['per_rotation'])} vs mean of measured rotations: "
      f"max |Δ| = {perm['api_vs_mean_max_abs_delta']:.4f}")
"""),
    ("md", r"""### 2.6 Labelled reads and calibration

A 42-example author-labelled set (`receipts/labeled.jsonl`): 20 routing
questions over billing/technical/sales, 14 sentiment yes/no, 8 urgency levels.
Labels are the author's reading of each text, not a gold corpus; the set is
small and in-domain, so read the numbers as descriptive.

Reported: accuracy with a Wilson 95% interval, multiclass Brier, NLL of the
true class, and 10-bin ECE at T=1 and at the NLL-fitted temperature. The
leave-one-out row fits the temperature on the other 41 examples for each held
out example, which is the honest version of the ECE — and it is **worse than
not fitting at all** (0.301 against 0.274). With 42 examples a single
temperature does not generalise; the in-sample gain (0.223) is the fit
absorbing its own sample. Anything built on this should either fit T on a
separate, larger calibration split or leave it at 1.
"""),
    ("code", r"""m = receipt("metrics.json")
rows = []
for key, label in (("at_T1", "T = 1 (as served)"),
                   ("at_fitted_T", f"T = {m['fitted_temperature']:.2f} (fitted)")):
    d = m[key]
    rows.append([label, f"{d['accuracy']:.3f}",
                 f"{d['accuracy_wilson95'][0]:.3f}-{d['accuracy_wilson95'][1]:.3f}",
                 f"{d['brier_multiclass']:.4f}", f"{d['nll_true_class']:.4f}",
                 f"{d['ece_10bin']:.4f}"])
loo = m["leave_one_out"]
rows.append(["leave-one-out fitting", f"{loo['accuracy']:.3f}", "-", "-", "-",
             f"{loo['ece_10bin']:.4f}"])
render_table(["Reads", "Accuracy", "95% CI", "Brier", "NLL", "ECE (10 bins)"], rows)
print("by question type at T=1:")
for t, d in m["by_question_type_at_T1"].items():
    print(f"  {t:8s} n={d['n']:2d}  accuracy {d['accuracy']:.3f}  "
          f"ECE {d['ece_10bin']:.3f}")
print(f"\nfitted temperature: {m['fitted_temperature']:.3f} "
      "(NLL-minimising on the same 42 examples)")
"""),
    ("md", r"""### 2.7 Latency

Thirty reads of one three-option question with a warm prefix cache, then
thirty with the prefix cache reset before each read. The cache-busting uses
`POST /reset_prefix_cache` rather than a prompt nonce, because a nonce changes
the prompt and therefore measures something else.

These are client-side end-to-end wall times at c=1 against a single-tenant
server, and the card is shared with no other job. They do not generalise to
concurrency or to a different card.
"""),
    ("code", r"""lat = receipt("latency.json")
rows = []
for key, label in (("warm", "Warm prefix cache"), ("busted", "Cache reset per read")):
    d = lat[key]
    rows.append([label, d["n"], f"{d['p50_ms']:.1f}", f"{d['p95_ms']:.1f}",
                 f"{d['min_ms']:.1f}", f"{d['max_ms']:.1f}"])
render_table(["Condition", "Reads", "p50 (ms)", "p95 (ms)", "min (ms)", "max (ms)"],
             rows)
print("GPU during the warm run (util %, SM clock, power, temp):", lat["gpu_mid_run"])
"""),
    ("md", r"""### 2.8 Question types and rejects

`choice`, `noul` and `score` in one request each, and the endpoint's error
behaviour. Every negative case returns the status a client should expect, and
the last row is the engine's own cap, not the interposer's.
"""),
    ("code", r"""neg = receipt("negatives.json")
rows = [[c["case"], c["status"], c["expected_status"],
         "ok" if c["status"] == c["expected_status"] else "MISMATCH"]
        for c in neg["cases"]]
render_table(["Case", "Status", "Expected", ""], rows)
print("all negatives as expected:", neg["all_as_expected"])
"""),
    ("md", r"""### 2.9 The first production workload (2026-09-21)

The probes above ran against an idle single-tenant server. On 2026-09-21 the
same serve line carried its first real job: an LLM-as-judge annotation pilot
speaking this exact contract over a private pool of items. Each annotation is
one Jev read per rotation — **five permutation reads per annotation** (the
permutations semantics from kishida's `jev` docs, now in production), driven
by 16 client workers. The receipts here are aggregates: the pilot's
per-annotation ledger stays outside this repository (it names pool items);
what is committed are per-minute counts, the engine's `/metrics` counter
deltas over a sample window, and `nvidia-smi dmon` telemetry. How they were
frozen: `harness/collect_load_receipts.py`.
"""),
    ("code", r"""load = receipt("load/load-summary.json")
led = load["ledger"]
rows = [
    ["Annotations", f"{led['annotations']:,}"],
    ["Failures", str(led['failed'])],
    ["Wall time (UTC)", f"{led['start_utc'][11:16]}-{led['end_utc'][11:16]} = {led['span_h']:.2f} h"],
    ["Prompt tokens", f"{led['input_tokens_total']:,} (mean {led['input_tokens_mean']:.0f}/annotation, "
                      f"p95 {led['input_tokens_p95']})"],
    ["Billed", f"${led['usd_billed']:.2f}"],
    ["Rate, run average", f"{led['annotations_per_h_avg']:,.0f} annotations/h"],
    ["Rate, steady (last 30 min)", f"{led['steady_last30_annotations_per_min']:.1f}/min, "
                                   f"{led['steady_last30_input_tokens_per_s']:.0f} tok/s prefill"],
]
render_table(["Production run, one card", "Value"], rows)

print("per UTC hour:")
rows = [[h["hour_utc"], f"{h['annotations']:,}", f"{h['mean_input_tokens']:.0f}"]
        for h in load["per_hour_utc"]]
render_table(["Hour", "Annotations", "Mean prompt tokens"], rows)
"""),
    ("code", r"""win = load["engine_window"]
d = receipt("load/metrics-delta.json")
g = load["gpu_window"]
rows = [
    ["Engine requests finished", f"{win['requests_finished']:.0f} in {win['window_s']:.0f} s"],
    ["Requests per annotation", f"{win['requests_per_annotation']:.2f} (the 5 permutation reads)"],
    ["Generated tokens per request", f"{win['generation_tokens_per_request']:.2f} "
                                     f"(max_tokens=1: nothing generated)"],
    ["Prompt tokens processed", f"{win['prompt_tokens']:,.0f} -> {win['prompt_tokens_per_s']:.0f} tok/s prefill"],
    ["Answer time, mean e2e", f"{d['e2e_latency_mean_s']*1000:,.0f} ms per request"],
    ["Answer time bounds", f"{d['e2e_latency_bounds']['within_15s']:.0f} of {d['requests_finished']:.0f} "
                           f"within 15 s; all within 30 s"],
    ["Prefix cache hit rate", f"{d['prefix_cache_hit_rate_pct']:.1f}% "
                              f"({d['prefix_cache_hits']:.0f} hits / {d['prefix_cache_queries']:.0f} queried)"],
    ["In flight at sample", f"{d['num_requests_running_at_t1']:.0f} running / "
                            f"{d['num_requests_waiting_at_t1']:.0f} waiting"],
    ["GPU over the telemetry window", f"{g['power_w_mean']:.0f} W mean "
                                      f"({g['power_w_min']:.0f}-{g['power_w_max']:.0f}), SM {g['sm_util_pct_mean']:.0f}%, "
                                      f"core {g['gpu_temp_c_range'][0]:.0f}-{g['gpu_temp_c_range'][1]:.0f} C, "
                                      f"mem {g['mem_temp_c_range'][0]:.0f}-{g['mem_temp_c_range'][1]:.0f} C"],
]
render_table(["Engine window under load", "Value"], rows)

littles = d["num_requests_running_at_t1"] + d["num_requests_waiting_at_t1"]
arrival = d["requests_finished"] / d["window_s"]
print(f"Little's law check: {littles:.0f} requests in flight / {arrival:.1f} req/s arrival "
      f"= {littles/arrival:.1f} s expected wait+compute; measured mean {d['e2e_latency_mean_s']:.1f} s")
"""),
    ("md", r"""![production load: throughput and answer time](../assets/charts/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm-load.png)

**The answer in ms.** Idle at c=1 the read is 141.7 ms p50 (133.0 ms
cache-busted) — the honest cost of one forward pass over a short prompt. Under
the pilot's 16 clients the mean is **12.0 s** and every request still finished
within 30 s: at ~1.5k prompt tokens the engine computes a read in well under a
second (1,952 tok/s across the window), so nearly all of the 12 s is queue.
The five permutation reads of one annotation are in flight together, so an
annotation's answer arrives with them — the wall-clock answer per annotation
is the ~12 s, not five times that. For capacity planning: low concurrency buys
latency, high concurrency buys throughput; the card is saturated either way.

**Kis's 180 W learning, confirmed in production.** His
[08-30 notebook](2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb)
measured ~1,955 tok/s prefill for this checkpoint on a short synthetic at a
180 W cap. The sample window holds 1,952 tok/s, and the run's steady state
matches it too (1,952 tok/s ledger-side over the last 30 min) — for hours,
under permutation-heavy judge traffic, at a 178 W mean with 71-72 C core and
74-76 C memory temperature. The cap remains nearly free.

**Prefix cache: measured 0.0%.** The serve line has `--enable-prefix-caching`,
and the counters show 570,234 queried tokens with zero hits in the window:
this pilot's prompts share no leading KV blocks, so every read pays its full
prefill. Do not bank on prefix caching for per-item judge traffic without
checking the hit rate; prompts that do share a long fixed preamble only make
this card's effective rate better.

Rate shape: the two dips in the timeline are the pilot's own pacing
(client-side scheduling and pool mix), not engine errors — zero failures
across the whole run.
"""),
    ("md", r"""## 3. Reproduce

**Hardware.** One NVIDIA CMP 170HX (SM80, 64 GiB HBM2e) with forced airflow,
180 W power cap is enough. No second card is needed for serving; the
cross-check in section 4 optionally uses one.

**Weights.** The W4A16 GPTQ checkpoint, 18.79 GiB held by the engine on load:

```bash
hf download Qwen/Qwen3.8-27B-GPTQ-4bit --local-dir <weights>
```

Verify before serving that the config declares `quantization_config.method ==
"gptq"`, 4-bit, `group_size 32`, `desc_act false` — that is the shape the
Marlin kernel serves on SM80.

**Serve.** The runtime is the project's SM80 vLLM build in a venv
(`<venv>/bin/vllm`, version and CUDA in `receipts/env.json`). Two flags are not
defaults and both are required for the Jev path:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
<venv>/bin/vllm serve <weights> \
  --served-model-name qwen3.8-27b-jev \
  --port 18030 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-logprobs 128 \
  --logprobs-mode processed_logprobs \
  --enable-prefix-caching
```

`--logprobs-mode processed_logprobs` is the finding in section 2.1. Without it
the endpoint cannot read labels at all. `FLASHINFER_DISABLE_VERSION_CHECK=1`
silences a `flashinfer-cubin`/`flashinfer` version mismatch that aborts engine
startup (`cubin 0.6.13` vs `flashinfer 0.6.16.post3`) — the same workaround
Kis's recipe uses.

**Interposer.** `harness/jev_server.py` is the Jev endpoint; it needs the model
directory only for the tokenizer:

```bash
<venv>/bin/python harness/jev_server.py \
  --upstream http://127.0.0.1:18030 \
  --tokenizer <weights> \
  --model qwen3.8-27b-jev \
  --port 18031
```

**Expected boot time.** Model load 115 s, engine init and CUDA-graph capture
about 6 minutes total on a warm compile cache; the first request after that
pays a Triton JIT for the read shape (1.6 s), then reads settle at ~140 ms.

**Measurements.** Every receipt in this notebook is produced by
`harness/jev_probe.py`, one probe per file:

```bash
MODEL_DIR=<weights> <venv>/bin/python harness/jev_probe.py \
    env prompts mask nucleus determinism temperature permutations latency \
    negatives labeled
```

`make_chart.py` rebuilds the reliability figure and `make_chart_load.py` the
production-load figure from the receipts. The load receipts are frozen from
the pilot's ledger, two `/metrics` snapshots and a `dmon` capture (paths
supplied at run time; the raw ledger stays outside the repository):

```bash
python harness/collect_load_receipts.py <pilot-ledger> <metrics-t0> <metrics-t1> \
    <window-seconds> <window-end-utc> <dmon> <gpu-index>
```

A smoke run of the same stack, without the receipts, is one request:

```bash
curl -s http://127.0.0.1:18031/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b-jev",
  "state": "I have been trying to connect my Stripe account for 3 days and it keeps failing.",
  "questions": {"department": {"type": "choice",
     "instructions": "Which team should handle this",
     "criteria": {"billing": "Payment or subscription issues",
                  "technical": "Bugs or integration problems",
                  "sales": "Pricing or account questions"}}},
  "options": {"temperature": 1.0, "return_logits": true}}'
```
"""),
    ("md", r"""## 4. Appendix

<details>
<summary>Failed attempts, the engine-vs-API cross-check, safety notes, and limitations (click to expand)</summary>
"""),
    ("md", r"""### Failures on the way here

Every one of these is preserved because each cost time and each has a fix.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `402`-style refusal at boot: `Free memory on device cuda:0 (35.43/63.53 GiB) ... less than desired (0.9, 57.17 GiB)` | another job on the card held ~12.7 GiB per GPU | size `--gpu-memory-utilization` to the free memory, or serve on the other card |
| 2 | `RuntimeError: flashinfer-cubin version (0.6.13) does not match flashinfer version (0.6.16.post3)` at engine start | the venv's two flashinfer packages are out of step | `FLASHINFER_DISABLE_VERSION_CHECK=1` (Kis's recipe does the same) |
| 3 | Interposer answered `502 upstream_error: upstream returned no logprobs for label ids [32, 33]` | vLLM's default `raw_logprobs` mode computes logprobs before `allowed_token_ids` | `--logprobs-mode processed_logprobs` |
| 4 | The same 502 with the mode fixed | logprob keys are `token_id:32` in this build, not `token:32`; the parser matched the wrong prefix | parse the integer after the last `:` (both spellings) |
| 5 | Labels missing from an otherwise working read (`choice-longshot`, all four options) | the checkpoint's `generation_config` ships `top_k=20, top_p=0.95` and vLLM adopts it; post-filter logprobs drop the low-mass label | send `top_p=1.0, top_k=-1` on the read |
| 6 | Orphaned `VLLM::EngineCore` holding ~15.7 GiB after a killed boot, making the next boot fail on free memory | engine core survives when the API server is killed mid-startup | kill the engine core before relaunching |
| 7 | `serve.sh` exits and takes the engine with it | the launcher backgrounds its children and returns | run the engine and the interposer as supervised processes, not as background children of a script that exits |
"""),
    ("md", r"""### Engine-vs-API cross-check, and what batching does to a read

The interposer trusts that the HTTP completions path returns what the engine
computes. `crosscheck.json` reads the same prompts twice — in-process with
vLLM's `LLM.generate(...)`, all fixtures in **one batch**, and over HTTP, one
request per read — and compares the label logprobs. `analyze_crosscheck.py`
derives the two questions this answers.

Two of the fixtures carry *the same prompt* as another fixture, which turns the
receipt into a batching probe: identical prompts agree exactly over HTTP (they
are separate requests) and disagree in-process (they sit at different batch
positions). The differences are small — up to ~8e-2 in logprob — and they are
the engine's batch-composition dependence, not a transport defect: nothing
about the HTTP path loses information, and the reads the endpoint serves are
one per request.
"""),
    ("code", r"""ca = receipt("crosscheck-analysis.json")
rows = [[r["case"], r["labels"], f"{r['max_abs_delta_http_vs_inprocess']:.4f}"
         if r["max_abs_delta_http_vs_inprocess"] is not None else "n/a",
         r["inprocess_only"], r["http_only"]] for r in ca["transport_fidelity"]["rows"]]
render_table(["Fixture", "Labels", "|Δ logprob| HTTP vs in-process (batched)",
              "In-process only", "HTTP only"], rows)
print(f"exact matches: {ca['transport_fidelity']['exact_matches']} of "
      f"{ca['transport_fidelity']['of']}")

rows = []
for r in ca["batch_position_sensitivity"]["rows"]:
    rows.append([r["prompt"], r["duplicate_fixture"], r["same_prompt"],
                 f"{r['max_abs_delta_inprocess_pair']:.4f}",
                 f"{r['max_abs_delta_http_pair']:.4f}"])
render_table(["Prompt", "Repeated as", "Same label ids",
              "|Δ| between the two, in one batch",
              "|Δ| between the two, over HTTP"], rows)
print("in-process, same prompt at two batch positions, choice fixture:")
for lp in ca["batch_position_sensitivity"]["rows"][0]["inprocess_logprobs"]:
    print("   ", {k: round(v, 6) for k, v in lp.items()})
print("over HTTP, the same two requests:")
for lp in ca["batch_position_sensitivity"]["rows"][0]["http_logprobs"]:
    print("   ", {k: round(v, 6) for k, v in lp.items()})
"""),
    ("md", r"""### What this does not show

- **Calibration.** n=42 in-domain author-labelled examples with an in-sample
  temperature fit. The ECE improvement is a demonstration of Jev's fitting
  loop, not certification — and the leave-one-out row (0.301) shows the fitted
  T is worse than T=1 on unseen examples at this sample size. A calibration
  claim needs hundreds of examples and a held-out split; until then, treat the
  fitted temperature as unfitted.
- **Generation.** This notebook serves reads. Throughput, TTFT and
  speculative-decode behaviour for *generation* on this checkpoint and card
  are Kis's notebook's subject, not this one's.
- **Images.** The endpoint returns 422 `images_not_supported`; this checkpoint
  is served text-path only.
- **High concurrency.** Determinism is measured for two concurrent requests on
  an idle server. Batch-composition-dependent kernels can change numerics at
  higher concurrency, and the appendix measures that effect (~0.07 logprob on
  two copies of one prompt sharing a batch). The c=1 latency table does not
  generalise to a loaded server — section 2.9 now measures the loaded case
  for answer time (12.0 s mean at 16 clients); determinism at high
  concurrency is still unmeasured.
- **Other models.** The `processed_logprobs` and `generation_config` findings
  are properties of a vLLM host, so they should hold for any checkpoint served
  this way — but they were measured on this one checkpoint and this engine
  build.
"""),

    ("md", r"""### Positioning vs TypeSafe Jev

[TypeSafe's Jev](https://docs.typesafe.ai/) (jev-1.13.0) is a hosted System
One model **trained with RLCD** — its probabilities are optimized against
outcomes, so "0.8" means ~80% of such predictions come true. That
calibration is their product. kishida's `jev` branch showed the *contract*
can be served from any model by reading label logits at one prompt
evaluation; this bundle is a vLLM implementation of that contract, and the
honest boundary is:

| Claim | Status | Receipt |
|---|---|---|
| Contract shape: `state` + `questions` map, `choice`/`noul`/`score` answers, probabilities summing to 1, `confidence`, `usage`, 422 error contract | **measured** — implemented and probed | `negatives.json`, `mask-processed_logprobs.json` |
| Read mechanic: one prompt evaluation, no answer generated, bit-identical at c=1, permutations average position bias | **measured** | `determinism.json`, `permutations.json` |
| No-train operation at production rate | **measured** (2026-09-21 pilot) | `load/` |
| Calibrated probabilities (Jev's differentiator) | **not claimed** — reads ship at T=1 and are over-confident on our labelled set: accuracy 0.571, ECE 0.274, fitted T fails leave-one-out | `metrics.json`, `labeled.jsonl` |
| Confidence numeric parity with jev-1.13 | **measured: no parity** — ours `1 − normalized entropy` vs jev-1.13 on the same 28 choice/score reads: mean |Δ| 0.649, Pearson r −0.23 | `positioning-bench/jev-leg.json` |
| Accuracy parity with jev-1.13 | **measured on n=42: no parity** — jev-1.13.0 scores 0.881 (choice 0.95, noul 0.93, score 0.63) against our 0.571 on the same labelled set; top-1 agreement 0.619, choice-distribution JS 0.254 | `positioning-bench/jev-leg.json` |
| More than 62 Choice options (Jev allows 255) | **not supported** — the label mask needs single-token symbols | `jev_server.py` |
| 64k context (deployed: 8k) and multi-question single-call latency | **untested** — deployment choices and per-question reads | — |

Closing the calibration gap is future work, not a property of this bundle:
fit a temperature on a proper held-out split (kishida measures ECE 0.02-0.03
after fitting on several models) or fine-tune the checkpoint for the read
format, as his 4B poc does (accuracy 0.872, ECE 0.022). Until then: our
probabilities are honest *rankings* from a general instruct model — usable
for routing with thresholds you validate on your own data — not certified
uncertainties, and not interchangeable with jev-1.13's confidence numbers.

**Measured alignment** (2026-09-21/22, n=42, this bundle's labelled set;
scripts and raw outputs in `positioning-bench/`): jev-1.13.0 (live API)
**0.881** — choice 0.95, noul 0.93, score 0.63; Laya base (421M RLCD
encoder, zero-shot) **0.786** — choice 0.90, noul 0.64, score 0.75; Laya
typed-decisions 0.786 (agrees with Jev on 81.0% of reads — the highest
Jev-alignment of any self-host option); Laya multilingual 0.595; our raw
read 0.571 — choice 0.60, noul 0.50, score 0.62; GLiNER 2.5 Multi 0.476.
Alignment-with-Jev tells the same story: Laya base 78.6% / typed 81.0% vs
our read 61.9%, and choice-distribution JS 0.135-0.170 vs our 0.254. Two
honest reads: a 6x-smaller RLCD-trained encoder nearly matches Jev on this
task even zero-shot — the trained-decision direction is validated twice
over — and its 512-1024 token context plus per-task fine-tuning needs are
the trade. Our raw read keeps the zero-training, big-context, 27B-quality
niche but is the weakest *judge* of the group. Confidence parity is
measured too: our entropy confidence vs jev-1.13's on the same 28
choice/score reads — mean |Δ| 0.649, Pearson r −0.23: the two confidence
numbers are unrelated, and only Jev's carry a calibration claim.

**Production-facet alignment** (150 real pilot states x 5 facets, top-1
agreement with live jev-1.13.0; noul excluded): **our raw read 83.5%**
(task_family 0.740, method_family 0.727, modality 0.900,
code_release_evidence 0.987, evaluation_type 0.820) vs Laya typed-decisions
**33.5%** and Laya base 31.3% (modality 12-15%, near random). The 0.786
Laya advantage from the labelled bench was on short ticket-style states —
its home turf; on long scientific abstracts with 7-way rubrics, only the
27B read tracks Jev. Re-annotating production with Laya would diverge from
Jev, not converge. Receipts: `positioning-bench/production-facets/`.

The read trick itself is folk knowledge — community tutorials do it with
plain llama.cpp (`max_tokens=1`, `top_logprobs`, `e^logprob`), sometimes
dressing several yes/no questions into one 16-way label. What separates the
implementations is everything around the trick: a label mask that survives
the logprob gather, neutralized nucleus defaults, rotation averaging, and —
decisively — calibration. kishida productized the trick; TypeSafe trained
the real thing; this bundle ports it to vLLM and states which half it has.

Two self-hosted neighbours bracket this bundle's position. [Solomon](https://huggingface.co/DoccyHealth/Solomon)
puts trained linear answer heads on the *same* Qwen3.8-27B base — the read
becomes trained, with per-type temperature artifacts and a runtime-identity
binding, at the cost of a custom (non-vLLM) runtime. [Laya](https://huggingface.co/convaiinnovations/laya)
goes further down the size axis: a 421M encoder trained with RLCD, ~33 ms
per question, Apache-2.0 — and its own card admits the base checkpoint is
near chance zero-shot and ships over-confident until a temperature is fitted
on your data. Our bundle is the zero-training end: no heads, no fit, big
context, production throughput on one card — and the weakest calibration
claim of the three. A fourth neighbour does a different job: [GLiNER 2.5
Multi](https://huggingface.co/fastino/gliner2.5-multi-v1) (287M, Apache-2.0,
231k downloads) is a schema-based *extraction* model — spans, records,
relations, multi-label with per-label confidence and constrained decoding —
no probability-per-option judgment, but a natural upstream partner: extract
the entities and records, then let the read judge them. Benchmarking
Solomon, Laya and GLiNER on this bundle's labelled set is the obvious
follow-up.
"""),
    ("md", r"""### Safety

Card stayed within the 80 C core / 85 C memory stop conditions during the
serving window and the whole production run; no Xid, ECC or GPU-disappearance
events. The card ran at its 180 W cap — 178 W mean with 100% SM under the
pilot, 71-72 C core and 74-76 C memory (telemetry in
`receipts/load/gpu-telemetry.json`). Forced airflow was in place throughout.

### Evidence

- Receipts, harness, and the raw payloads:
  `results/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm/`
- The serving recipe this builds on (credit: Kis):
  [`2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm`](2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb)
- The Jev contract:
  [`kishida/llama.cpp`, branch `jev`, `docs/jev.md`](https://github.com/kishida/llama.cpp/blob/jev/docs/jev.md)
- The production-load receipts (aggregates only; the raw pilot ledger names
  pool items and stays outside the repository): `receipts/load/`
</details>
"""),
]


def main():
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        cid = f"jev-{i:02d}"
        if kind == "md":
            cells.append({"cell_type": "markdown", "id": cid, "metadata": {},
                          "source": src})
        else:
            cells.append({
                "cell_type": "code", "id": cid, "execution_count": None,
                "metadata": {}, "outputs": [], "source": src,
            })
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", OUT, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
