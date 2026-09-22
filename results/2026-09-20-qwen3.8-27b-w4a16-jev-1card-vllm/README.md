# Qwen3.8-27B (W4A16 GPTQ) as a Jev-compatible endpoint — 1x CMP 170HX (SM80), vLLM

Measured 2026-09-20 on one CMP 170HX (SM80, 64 GiB HBM2e) at the 180 W club
cap, with production-load receipts frozen on 2026-09-21. Notebook:
[`notebooks/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm.ipynb`](../../notebooks/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm.ipynb).

**Question.** Can this checkpoint serve as a Jev-compatible classification
endpoint on this card — a `POST /v1/systemone` request returning a calibrated
probability per option, for `choice`, `noul` and `score` questions, with no
token generated?

**Answer.** Yes, mechanically, and the mechanics have three sharp edges a vLLM
host has to know about. Reads are exact, repeatable and fast (~140 ms p50 at
c=1). On 42 author-labelled reads the endpoint scores 0.571 accuracy and is
over-confident (ECE 0.274 at T=1; the NLL-fitted T does not generalise,
leave-one-out ECE 0.301). Option order moves the answer (max shift 0.639,
argmax flips). And it is a **no-train method**: the stock W4A16 checkpoint is
served read-only — nothing fine-tuned, nothing fitted that generalises. The
day after the probes it carried its first production workload: 19,045
annotations in 4.5 h, zero failures, 1,952 tok/s prefill at a 178 W mean.

## Findings

1. **`--logprobs-mode processed_logprobs` is required.** vLLM's default
   (`raw_logprobs`) computes logprobs *before* `allowed_token_ids` applies, so
   a label-masked read returns the unmasked top-k and the labels are absent —
   the endpoint cannot answer at all (`502 upstream_error`). With the flag, the
   returned payload is exactly the label set and the mass over it sums to 1.
   Receipts: `mask-raw_logprobs.json` (the failure), `mask-processed_logprobs.json`
   (the fix, plus the offset-invariance check that proves the ordering).
2. **The checkpoint's `generation_config` filters labels unless neutralised.**
   It ships `top_k=20, top_p=0.95`; vLLM adopts those as server defaults; in
   the processed mode the returned logprobs are post-filter, so a four-option
   question loses its low-mass label and the survivors are inflated. The read
   path sends `top_p=1.0, top_k=-1`. Receipt: `nucleus.json`.
3. **Option order is a real effect, not a rounding error.** Rotating a
   three-option question shifts the read mass by up to 0.639 (L1) and changes
   the argmax. Jev's `permutations` option averages over rotations; the
   endpoint implements it and the receipt confirms the averaged answer is the
   mean of the individually measured rotations. Receipt: `permutations.json`.
4. **Bit-identity belongs to a batch composition, not to the model.** Two
   copies of the same prompt agree exactly when served as separate requests and
   differ by up to 0.067 in logprob when placed in one batch. The endpoint
   serves one read per request, so its reads are repeatable; a batched client
   should not expect bit-identical logits. Receipts: `crosscheck-analysis.json`.
5. **The first production workload confirmed the serving recipe and exposed
   the cache.** An LLM-as-judge annotation pilot (five permutation reads per
   annotation, 16 clients, 4.5 h) ran with zero failures; engine prefill held
   1,952 tok/s — within 0.2% of Kis's synthetic 1,955 tok/s at the same cap —
   and the answer went from 141.7 ms idle to a 12.0 s mean under load
   (99.8% of reads within 15 s; queue-bound, not compute-bound). The prefix
   cache hit **0.0%** on this traffic: per-item judge prompts share no leading
   KV blocks, so every read pays its full prefill. Receipts: `load/`.

## Measured

| Quantity | Value | Receipt |
|---|---|---|
| Read latency, c=1, warm prefix cache | p50 141.7 ms, p95 160.3 ms (n=30) | `latency.json` |
| Read latency, c=1, cache reset before each read | p50 133.0 ms, p95 150.6 ms (n=30) | `latency.json` |
| Label distribution repeatability | identical sequential / interleaved / 2 concurrent; max |Δlogprob| 0.0 | `determinism.json` |
| Temperature handling | API probabilities equal `softmax(logprobs(T=1)/T)`; max |Δ| 8.3e-17 | `temperature.json` |
| Labelled accuracy (n=42, author labels) | 0.571 (Wilson 95% CI 0.422-0.709) | `metrics.json` |
| Brier / NLL / ECE at T=1 | 0.5494 / 0.9543 / 0.2736 | `metrics.json` |
| Brier / NLL / ECE at fitted T=2.16 (in-sample) | 0.5356 / 0.8747 / 0.2228 | `metrics.json` |
| Leave-one-out ECE (honest read of the fit) | 0.301 — worse than T=1, so the fitted temperature does not generalise at n=42 | `metrics.json` |
| Rejected requests (8 schema/image cases + engine cap) | all as expected (422 / 400) | `negatives.json` |
| Boot | model load 18.79 GiB in 115 s; engine init ~6 min cold | `serve.log` |
| Production run (2026-09-21, first real workload) | 19,045 annotations in 4.53 h, 0 failures, 27.87M prompt tokens, $1.17 billed | `load/load-summary.json` |
| Answer time under load (16 clients) | mean 12,032 ms/request; 1,890 of 1,893 within 15 s; all within 30 s | `load/metrics-delta.json` |
| Prefill under load | 1,952 tok/s in the 292 s sample window; steady state 76.6 annotations/min = 1,952 tok/s ledger-side | `load/load-summary.json` |
| Reads per annotation / gen tokens per request | 5.01 / 1.02 (max_tokens=1: nothing generated) | `load/metrics-delta.json` |
| Prefix cache hit rate on judge traffic | 0.0% (0 hits / 570,234 queried tokens) | `load/metrics-delta.json` |
| GPU under sustained judge load | 178 W mean (172-186) at the 180 W cap, SM 100%, 71-72 C core, 74-76 C memory | `load/gpu-telemetry.json` |

## Receipts

| File | What it is |
|---|---|
| `serve.log` | engine boot log for the served configuration (the run the numbers come from) |
| `serve-rawmode.log` | engine boot log for the default-mode comparison |
| `env.json` | runtime versions, model config + file hashes, engine pins, log excerpt |
| `prompts.json` | rendered prompt, token ids, sha256 and label symbols per question type |
| `mask-raw_logprobs.json` | the label-mask failure under the default `logprobs_mode` |
| `mask-processed_logprobs.json` | the fix, with the ordering proof |
| `nucleus.json` | neutralised vs `generation_config`-default reads |
| `determinism.json` | sequential / interleaved / concurrent repeat reads |
| `temperature.json` | API vs client-side `softmax(logits/T)` at four temperatures |
| `permutations.json` | per-rotation reads and the endpoint's averaged answer |
| `latency.json`, `latency.jsonl` | warm and cache-busted read latencies |
| `negatives.json` | rejected requests and status codes |
| `labeled.jsonl` | per-example labelled predictions with label logprobs |
| `metrics.json` | accuracy, Brier, NLL, ECE, NLL curve, per-type metrics |
| `load/load-summary.json` | frozen aggregate of the pilot's per-annotation ledger (counts, tokens, per-hour, per-minute; no pool items) |
| `load/metrics-delta.json` | engine `/metrics` counter deltas over a 292 s window under load, incl. e2e latency bounds and prefix-cache counters |
| `load/load-ledger-minutes.json` | annotations and prompt tokens per UTC minute for the whole run |
| `load/gpu-telemetry.json` | `nvidia-smi dmon` samples (45 x 2 s) during the window |
| `crosscheck.json`, `crosscheck-analysis.json` | same prompts read in-process (vLLM `LLM.generate`) vs over HTTP, plus the duplicate-prompt pair that isolates batch-position sensitivity |

## Harness

| File | What it does |
|---|---|
| `jev_server.py` | the Jev endpoint: `/v1/systemone` in front of vLLM's OpenAI server |
| `jev_probe.py` | produces every probe receipt above, one probe per file |
| `make_chart.py` | rebuilds the notebook figure from the receipts |
| `make_chart_load.py` | rebuilds the production-load figure from the receipts |
| `collect_load_receipts.py` | freezes `receipts/load/` from the pilot ledger, two `/metrics` snapshots and a `dmon` capture |
| `serve.sh` | launches the engine and the endpoint with the required flags |
| `build_notebook.py` | builds the notebook from the same numbers |

## Reconstruct

```bash
MODEL_DIR=<weights> ./serve.sh                       # engine + endpoint
MODEL_DIR=<weights> <venv>/bin/python jev_probe.py all   # every receipt
python make_chart.py && python make_chart_load.py && python build_notebook.py  # figures + notebook
```


The serving recipe for this checkpoint on this card — W4A16 on one 170HX, the
syv-ai lineage, the 180 W cap being nearly free — is Kis's:
[`2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm`](../../notebooks/2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb).
[kishida's `jev` branch docs](https://github.com/kishida/llama.cpp/blob/jev/docs/jev.md)
— shape-compatible with [TypeSafe's System One API](https://docs.typesafe.ai/api)
(the contract and the read mechanic; **not** the calibration — see below).
This bundle adds the vLLM implementation and the findings above, the
production-load validation, and the no-train reading of the method.

## Positioning vs TypeSafe Jev

TypeSafe's Jev (jev-1.13.0) is a hosted System One model **trained with
RLCD** — probabilities optimized against outcomes. That calibration is their
product; kishida's `jev` branch showed the contract itself can be served
from any model by reading label logits at one prompt evaluation. The honest
boundary for this bundle:

| Claim | Status | Receipt |
|---|---|---|
| Contract shape (state + questions, choice/noul/score, probabilities, confidence, usage, 422s) | **measured** | `negatives.json`, `mask-processed_logprobs.json` |
| Read mechanic (one evaluation, no generation, bit-identical at c=1, permutation averaging) | **measured** | `determinism.json`, `permutations.json` |
| No-train production operation | **measured** | `load/` |
| Calibrated probabilities (Jev's differentiator) | **not claimed** — T=1 reads are over-confident: acc 0.571, ECE 0.274, fitted T fails LOO | `metrics.json` |
| Confidence numeric parity with jev-1.13 | **not claimed** — ours is `1 − normalized entropy`; theirs is undocumented | — |
| Accuracy parity with jev-1.13 | **untested** — no common benchmark | — |
| >62 Choice options (Jev: 255) | **not supported** — single-token label mask | `jev_server.py` |

**Measured vs jev-1.13.0 and self-hostable options** (n=42, this bundle's
labelled set; `receipts/positioning-bench/`): jev-1.13.0 (live API) **0.881**
— choice 0.95, noul 0.93, score 0.63; Laya base (421M RLCD encoder,
zero-shot) **0.786** — choice 0.90, noul 0.64, score 0.75; Laya
typed-decisions 0.786 (Jev-agreement 81.0%, the highest of any self-host
option); Laya multilingual 0.595; our raw read 0.571 — choice 0.60, noul
0.50, score 0.62; GLiNER 2.5 Multi 0.476. Jev-alignment: Laya base 78.6% /
typed 81.0% vs our read 61.9%; choice-distribution JS 0.135-0.170 vs our
0.254. A 6x-smaller RLCD-trained encoder nearly matches Jev on this task
even zero-shot — the trained-decision direction is validated twice over;
our raw read keeps the zero-training, big-context niche but is the weakest
judge of the group. Confidence parity: our entropy confidence vs jev-1.13's
on the same 28 choice/score reads — mean |Δ| 0.649, Pearson r −0.23:
unrelated, and only Jev's carry a calibration claim. Closing the gap
(temperature fit on a held-out split, trained heads, task fine-tune) is
future work.

**Production-facet alignment** (150 real pilot states x 5 facets vs live
jev-1.13.0): our raw read agrees with Jev on **83.5%** of choice reads
(task_family 0.740, method_family 0.727, modality 0.900,
code_release_evidence 0.987, evaluation_type 0.820) vs Laya typed-decisions
33.5% and base 31.3% — the labelled-bench advantage of the small RLCD model
does not transfer to the production task. Re-annotating production with it
would diverge from Jev. Receipts:
`receipts/positioning-bench/production-facets/`.

## Limitations

- n=42, in-domain, author-labelled; the temperature fit is in-sample and the
  leave-one-out ECE (0.301) is the honest number. Not a calibration claim.
- Classification reads only. Generation throughput on this checkpoint and card
  is Kis's notebook's subject.
- Text path only; images return 422.
- Determinism is measured at c=1 on an idle, single-tenant server (two
  concurrent requests); answer time under load is measured in `load/` but
  bit-identity at high concurrency is not.
- The production receipts are aggregates; the raw pilot ledger stays outside
  the repository because it names pool items.
