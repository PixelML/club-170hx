# GLM-5.3-Flash EXL3 4.05bpw, 4x CMP 170HX, exllamav3 + TabbyAPI

Status: measured
Date: 2026-09-02

## Hardware

- Cards: 4 x CMP 170HX (64 GiB HBM2e each, 256 GiB pool)
- Anonymous card labels: gpu0-gpu3
- Topology / PCIe links: single host, manual `gpu_split` (not tensor-parallel)
- Power limit and measured peak draw: no per-card power cap changed this run; measured group peak 302.3 W total during a c=4 load run
- Cooling and peak core/memory temperatures: peak 49 C core through the full concurrency ladder, no throttle

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

See `ladder.json`, `prefill.json`, `power.json`, `speculative.json`, `gpu-final.csv`, and `run-manifest.json` in this directory. Summary:

| Metric | Value |
|---|---:|
| Decode, c=1 (mean of 3 reps) | 26.9 tok/s |
| Best aggregate, c=8 (mean of 3 reps) | 44.8 tok/s |
| Prefill, warm (2,941-token prompt) | ~354 tok/s |
| TTFT proxy, cold boot | 5.57 s |
| TTFT proxy, warm | 0.39 s |
| Golden corpus (20 prompts, keyword-match) | 20/20 |
| n-gram speculative decoding | -7.2% vs. no draft; shipped with drafting disabled |

## Correctness and failures

- Output validation: 20/20 golden-corpus prompts passed (short_factual, reasoning, code, json, multilingual categories, keyword-match scoring, `max_tokens=512`).
- Xid/ECC/AER scan: none observed through the full concurrency ladder (2s telemetry samples).
- Known caveats: with `cache_mode: Q8` (the standing config), GLM-5.3-Flash's DeepSeek Sparse Attention (DSA, `index_topk: 2048`) activates past ~2048 tokens of context and hits an explicit exllamav3 assertion against a quantized MLA cache — any request whose context exceeds ~2048 tokens fails with a 503 (the server process itself stays up). `cache_mode: FP16` removes the assertion and fits the same `gpu_split`, but was not load-tested at concurrency. Reasoning must be enabled (`reasoning: true`) or chain-of-thought leaks into `content` and burns the token budget; `max_tokens=32` is not enough even with reasoning parsed correctly, `max_tokens=128` is.

## Evidence

- Full attempt history, including the failed `gpu_split` boot attempts and the compatibility-only prior finding on this checkpoint family, lives in [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX).
- Executed notebook: [notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb](../../notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb).
