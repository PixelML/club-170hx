# GLM-5.3-Flash EXL3 4.05bpw, 4x CMP 170HX, exllamav3 + TabbyAPI

Status: measured
Date: 2026-09-02 (250 W, accidental default); re-measured 2026-09-03 (180 W, verified cap, canonical)

**Power cap correction (2026-09-03).** The 2026-09-02 ladder ran at the
vBIOS default 250 W by accident -- no per-card cap had been set that
session. The identical protocol was re-run against the same standing
server (not restarted) at the verified 180 W club-standard cap. No level
showed a consistent throughput difference outside run-to-run noise.
**180 W is canonical; the summary table below is the 180 W re-measure**,
with 250 W figures kept for comparison. See `ladder-180w.json`,
`prefill-ttft-180w.json`, `power-180w.json`, `power-temp-sample-180w.csv`.

## Hardware

- Cards: 4 x CMP 170HX (64 GiB HBM2e each, 256 GiB pool)
- Anonymous card labels: gpu0-gpu3
- Topology / PCIe links: single host, manual `gpu_split` (not tensor-parallel)
- Power limit and measured peak draw: **180 W cap (verified, canonical)** -- measured group peak 352.4 W total (coincident-peak artifact; no single 1 s sample exceeded 302 W across all four cards), peak per-card 172.6 W, all samples under the 180 W cap. 250 W run (2026-09-02, accidental default): measured group peak 302.3 W total during a c=4 load run.
- Cooling and peak core/memory temperatures: 180 W run peak 51 C core; 250 W run peak 49 C core; no throttle at either cap.

## Software

- OS / kernel: standard club node image (see `docs/HARDWARE.md`)
- NVIDIA driver / CUDA: exllamav3 1.4.6+cu128.torch2.10.0
- Runtime repository + exact commit or image digest: exllamav3 1.4.6, served via TabbyAPI (OpenAI-compatible server)
- Model repository + exact revision: `turboderp/GLM-5.3-Flash-exl3`, branch `4.05bpw`, revision `2a30229e67012798ba9f0cd832bb78abf4c363d5`
- Quantization / dtype: EXL3, 4.05 bits/weight; KV cache mode Q8 for standing service (FP16 only for the one prefill measurement below)

## Command

```text
# gpu_split is a list of GB per card, not tensor_parallel — TP raises
# NotImplementedError for Glm5NextForConditionalGeneration in exllamav3 1.4.6
gpu_split: [48, 48, 48, 48]
max_seq_len: 32768
cache_size: 32768
cache_mode: Q8
reasoning: true
```

## Method

- Warmup: 1 warmup rep discarded per concurrency level
- Samples: 3 measured reps per concurrency level (C1/C2/C4/C8)
- Input/output tokens: exactly 400 completion tokens, greedy decoding; separate prefill measurement at a 2,941-token prompt (exact count verified against the tokenizer)
- Metric calculation: aggregate tok/s = total completion tokens / wall time, from the final usage object; per-request tok/s = aggregate / concurrency

## Results

See `ladder-180w.json`, `prefill-ttft-180w.json`, `power-180w.json`, `power-temp-sample-180w.csv` (180 W, canonical), and `ladder.json`, `prefill.json`, `power.json`, `speculative.json`, `gpu-final.csv`, `run-manifest.json` (250 W, comparison) in this directory.

**180 W (verified cap, canonical):**

| Metric | Value |
|---|---:|
| Decode, c=1 (mean of 3 reps) | 25.2 tok/s |
| Decode, c=2 (mean of 3 reps) | 35.3 tok/s |
| Decode, c=4 (mean of 3 reps) | 43.2 tok/s |
| Best aggregate, c=8 (mean of 3 reps) | 44.6 tok/s |
| Prefill, warm (2,954-token prompt post chat-template) | ~358.5 tok/s |
| TTFT, 3 reps | 0.73 s / 1.41 s / 1.78 s |
| Golden corpus (20 prompts, keyword-match) | 20/20 (unchanged, not re-run at 180 W) |
| n-gram speculative decoding | -7.2% vs. no draft; shipped with drafting disabled (unchanged, not re-run at 180 W) |

**250 W (accidental default, 2026-09-02, retained for comparison):**

| Metric | Value |
|---|---:|
| Decode, c=1 (mean of 3 reps) | 26.9 tok/s |
| Best aggregate, c=8 (mean of 3 reps) | 44.8 tok/s |
| Prefill, warm (2,941-token prompt) | ~354 tok/s |
| TTFT proxy, cold boot | 5.57 s |
| TTFT proxy, warm | 0.39 s |

Delta (180 W vs 250 W): C1 -6.3%, C2 +13.5%, C4 +3.6%, C8 -0.4% -- read as
run-to-run noise, not a directional power effect. 180 W remains the
standing cap.

## Correctness and failures

- Output validation: 20/20 golden-corpus prompts passed (short_factual, reasoning, code, json, multilingual categories, keyword-match scoring, `max_tokens=512`).
- Xid/ECC/AER scan: none observed through the full concurrency ladder (2s telemetry samples).
- Known caveats: with `cache_mode: Q8` (the standing config), GLM-5.3-Flash's DeepSeek Sparse Attention (DSA, `index_topk: 2048`) activates past ~2048 tokens of context and hits an explicit exllamav3 assertion against a quantized MLA cache — any request whose context exceeds ~2048 tokens fails with a 503 (the server process itself stays up). `cache_mode: FP16` removes the assertion and fits the same `gpu_split`, but was not load-tested at concurrency. Reasoning must be enabled (`reasoning: true`) or chain-of-thought leaks into `content` and burns the token budget; `max_tokens=32` is not enough even with reasoning parsed correctly, `max_tokens=128` is.

## Evidence

- Full attempt history, including the failed `gpu_split` boot attempts and the compatibility-only prior finding on this checkpoint family, lives in [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX).
- Executed notebook: [notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb](../../notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb).
