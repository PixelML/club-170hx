# Qwen3.8-27B (W4A16 GPTQ) as a Jev-compatible endpoint — 1x CMP 170HX (SM80), vLLM

Measured 2026-09-20 on one CMP 170HX (SM80, 64 GiB HBM2e) at the 180 W club
cap. Notebook:
[`notebooks/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm.ipynb`](../../notebooks/2026-09-20-qwen3.8-27b-w4a16-jev-1card-vllm.ipynb).

**Question.** Can this checkpoint serve as a Jev-compatible classification
endpoint on this card — a `POST /v1/systemone` request returning a calibrated
probability per option, for `choice`, `noul` and `score` questions, with no
token generated?

**Answer.** Yes, mechanically, and the mechanics have three sharp edges a vLLM
host has to know about. Reads are exact, repeatable and fast (~140 ms p50 at
c=1). On 42 author-labelled reads the endpoint scores 0.571 accuracy and is
over-confident (ECE 0.274 at T=1, 0.223 at the NLL-fitted T=2.16); option
order moves the answer (max shift 0.639, argmax flips). Treat it as a working
classification surface whose calibration is a project, not a property.

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
| `crosscheck.json`, `crosscheck-analysis.json` | same prompts read in-process (vLLM `LLM.generate`) vs over HTTP, plus the duplicate-prompt pair that isolates batch-position sensitivity |

## Harness

| File | What it does |
|---|---|
| `jev_server.py` | the Jev endpoint: `/v1/systemone` in front of vLLM's OpenAI server |
| `jev_probe.py` | produces every receipt above, one probe per file |
| `serve.sh` | launches the engine and the endpoint with the required flags |
| `make_chart.py` | rebuilds the notebook figure from the receipts |
| `build_notebook.py` | builds the notebook from the same numbers |

## Reconstruct

```bash
MODEL_DIR=<weights> ./serve.sh                       # engine + endpoint
MODEL_DIR=<weights> <venv>/bin/python jev_probe.py all   # every receipt
python make_chart.py && python build_notebook.py     # figure + notebook
```

## Credits

The serving recipe for this checkpoint on this card — W4A16 on one 170HX, the
syv-ai lineage, the 180 W cap being nearly free — is Kis's:
[`2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm`](../../notebooks/2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb).
The Jev contract (one prompt evaluation, label logits, confidence as
`1 - normalized entropy`, expected-level `score`, temperature as a calibration
divisor, permutations for position bias) is
[kishida's `jev` branch docs](https://github.com/kishida/llama.cpp/blob/jev/docs/jev.md),
API-compatible with TypeSafe System One. This bundle adds the vLLM
implementation and the three findings above.

## Limitations

- n=42, in-domain, author-labelled; the temperature fit is in-sample and the
  leave-one-out ECE (0.301) is the honest number. Not a calibration claim.
- Classification reads only. Generation throughput on this checkpoint and card
  is Kis's notebook's subject.
- Text path only; images return 422.
- Determinism and latency are measured at c=1 on an idle, single-tenant
  server; the concurrency claim covers two concurrent requests.
