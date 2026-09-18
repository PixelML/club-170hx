# Workload — Bonsai 2 27B (ternary g128) on llama.cpp (llama-server), 1 × CMP 170HX

Status: measured
Date: 2026-09-18

## Hardware

- Cards: 1 × CMP 170HX (64 GiB HBM2e), card 0 of a multi-card node; the second card was idle and unused
- Anonymous card labels: single card under test, homogeneous with its idle sibling
- Topology / PCIe links: not sampled this run; single-card serving, no P2P involved
- Power limit and measured peak draw: 180.00 W limit read back on every 1 Hz sample; measured peak draw 213.5 W serving / 186.1 W bench (transients above the average-based cap, same pattern as earlier CMP 170HX receipts)
- Cooling and peak core/memory temperatures: forced airflow; peak 73 °C core / 77 °C memory against the 80/85 °C stop conditions

## Software

- OS / kernel: Ubuntu 22.04, kernel 6.8
- NVIDIA driver / CUDA: driver 610.43.03 (UMD reports CUDA 13.3); binary built for CUDA 12.8
- Runtime repository + exact commit: PrismML-Eng/llama.cpp fork, prebuilt release `prism-b10685-7dffb15`, build `7dffb158d` (10685). Stock llama.cpp cannot run these files — it refuses `PQ2_0`/`PTQ1_0` as unknown types, and loads the (unshipped) legacy `Q2_0` band without the required Hadamard activation transform
- Model repository + exact revision: prism-ml/Ternary-Bonsai-2-27B-gguf @ `6ed5e12bf84b7a63069882c91dd9e9218647d17b` (PQ2_0 6.70 GiB / PTQ1_0 5.53 GiB, SHA-256 in receipts)
- Quantization / dtype: ternary g128 {−1, 0, +1} with FP16 group scales in a Hadamard-rotated basis; PQ2_0 = 2.13 true bpw, PTQ1_0 = 1.75 true bpw; Q8_0 vision projector downloaded but not loaded (text-only)

## Command

```text
llama-server -m <weights>/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -fa on --no-mmap -c 40960 -np 1 -t 8 -ub 2048 --host 127.0.0.1 --port 8082
llama-bench -m <weights>/Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -fa 1 -t 8 --load-mode none
```

## Method

- Warmup: 1 discarded rep per cohort
- Samples: 3 measured reps per cohort; llama-bench 5 repetitions (3 for flag A/Bs)
- Input/output tokens: decode cohorts 10-token prompt / 256 and 900 completion tokens, `ignore_eos`; prefill cohort 6,601-token prompt (usage-reported) / 11 out; context sweep 991-31,711 token prompts (tokenizer-calibrated) / 128 out, `cache_prompt: false`; ladder 2/4/8 concurrent 256-token requests
- Metric calculation: tokens from the final usage object only (never SSE events); greedy (`temperature: 0`) throughout; decode rate = completion tokens / (wall − TTFT); prefill rate = prompt tokens / TTFT; J/tok = mean 1 Hz board power over the serving window / median served decode rate (indicative)
- Executed notebook: `notebooks/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp.ipynb` (LIVE = False, replays these receipts)

## Results

| Metric | Value |
|---|---:|
| llama-bench tg128, PQ2_0 (tok/s) | 54.52 ± 0.35 (rerun 53.34 ± 0.14) |
| llama-bench tg128, PTQ1_0 (tok/s) | 40.07 ± 0.14 (rerun 40.04 ± 0.09) |
| llama-bench pp512, PQ2_0 / PTQ1_0 (tok/s) | 872.9 / 442.7 |
| Served decode 256-token cohort, PQ2_0 / PTQ1_0 (tok/s, median) | 52.6 / 52.2 |
| Served decode 900-token cohort, PQ2_0 / PTQ1_0 (tok/s, median) | 51.9 / 51.4 |
| Served prefill, 6,601-token prompt (tok/s) | 815.6 |
| TTFT, 10-token prompt (ms, client-observed median) | 274 |
| Context sweep decode, 991 → 31,711 prompt tokens (tok/s) | 51.5 → 46.8 |
| Concurrency ladder aggregate, c=2/4/8 (tok/s, median) | 49.2 / 49.1 / 48.7 (single-slot queueing — see below) |
| Boot to healthy (warm page cache) | 6-8 s |
| Mean board power over serving window (W) / J per token | 133.5 / 2.54 |
| Peak core / memory temp (°C) | 73 / 77 |

Measured notes:

- **The packing ranking flips between measurement surfaces.** `llama-bench` puts PTQ1_0 25-33 % behind PQ2_0 on decode (reproduced twice, ±0.1 tok/s), matching the community-reported A100 ordering — but through `llama-server` the two packs serve at parity (52.2 vs 52.6 tok/s). Both instruments are stable; this is a real path difference, cause not identified. PQ2_0 stays the recipe of record on prefill (2×) and upstream-verified ranking; the bench-implied decode penalty for PTQ1_0 does not exist on the served path.
- **The server clamps this model to a single slot.** With `-np 8` (and with the `LLAMA_ARG_N_PARALLEL=8` env override) the boot banner reads `n_slots = 1` and every request logs onto slot 0 — the concurrency ladder therefore measures queueing: aggregate stays at the single-stream rate while per-request TTFT grows ~5.2 s per queue position. Multi-slot serving of the hybrid-attention backbone is upstream work, not a flag.
- **Speculative decoding is unavailable**: the fork's DSpark path has no Bonsai 2 drafter (the upstream demo's own downloader states this), so plain decoding is the fastest measured configuration.
- Threads (8 vs 16) and prefill ubatch (512 vs 2048) are inert: tg128 54.5 vs 54.4, pp512 872.9 vs 868.2.
- The third c=8 ladder rep was aborted by a boot recycle and is excluded from the receipts.

## Correctness and failures

- Output validation: greedy sanity requests before measurement ("capital of France" → `Paris`; a one-line Pythagoras function → correct code); the executed notebook's final editable request records a coherent 256-token answer with a clean usage object
- Xid/ECC/AER scan: no Xid lines; post-run `nvidia-smi -q` snapshot in receipts
- Known caveats: TTFT on the OpenAI-compat endpoint carries a ~0.2-0.3 s client-side first-chunk constant on 10-token prompts — the prefill rates are the trustworthy latency figures; boot-time receipts reflect warm host page cache, not cold reads from the shared model library

## Evidence

- Raw receipts: `receipts/` in this directory (llama-bench tables incl. reruns, serve JSONL, context sweep, 1 Hz power/thermal telemetry, load-gate responses, boot times, weight SHA-256, run metadata)
- Charts: `assets/charts/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp-*.png` with per-chart regenerator scripts reading these receipts
- Upstream sources: [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (throughput table, community-reported), [PrismML-Eng/llama.cpp releases](https://github.com/PrismML-Eng/llama.cpp/releases), [Bonsai-demo AGENTS.md](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/AGENTS.md) (drafter availability)

Inferred (not measured): the bench-vs-served PTQ1_0 gap plausibly comes from different CUDA graph shapes between the two decode loops interacting with the dense-trit unpack path; magnitudes on the A100 comparison row are cross-power-envelope and should be read as ordering, not equality.
