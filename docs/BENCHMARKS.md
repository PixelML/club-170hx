# Benchmarks

Results here are application measurements on the tested CMP 170HX setup. They are not vendor specifications or theoretical estimates. This page is the normalized comparison index; raw manifests, commands, and receipts stay in the model repositories.

## Evidence tiers

A row is **publication-safe** only when the full sanitized receipt chain (manifest, commands, redacted outputs, exact model/runtime pins) is merged in the owning model repository. Rows whose receipts are unmerged are **pending evidence repair**: their owning tickets repair the evidence chain, and the numbers are not decision-grade until then. The README scoreboard mirrors these statuses.

## Normalized matrix

`—` = not presented without sanitized stable evidence. Do not interpolate or compare unlike metrics.

| Workload | Quant / runtime | Cards | Context | Concurrency | Prefill | Decode | Aggregate | TTFT | ITL | Quality / success | Power / thermals | Energy | Status | Evidence |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| GLM-5.3-Flash | UD-IQ4_XS · llama.cpp (0069971) | 4 · layer split | 16,384 | c=1; ladder 1/2/4; soak c=2 | — | 17.73 tok/s median (c=1) | ~17.5–17.7 tok/s (c=2/4) | — | — | 21/26 local tasks · 41/41 soak reps | Snapshot only: 40.10–44.49 W, core 41–43 °C, mem 45–55 °C, VRAM 32,656–42,312 / 65,536 MiB; limit not queried | Snapshot proxy only, not integrated | Publication-safe | [Run manifest](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/blob/7fc71e00925f7b7902764aab7d08b6d923aaaea4/results/phase63/run-manifest.json) · [Result card](results/2026-08-30-glm-5.3-flash-ud-iq4xs-llamacpp-cmp170hx.md) |
| GLM-5.3-Flash | NVFP4 | 3 | — | — | — | — | — | — | — | — | — | — | Not compatible (SM121 weights on SM80) | [Negative results](#negative-results-matter) |
| Qwen3.8-27B | NVFP4 · vLLM | 1 (× 3 runs) | — | — | — | — | — | — | — | — | — | — | Pending evidence repair · [repo PR 1](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX/pull/1) | [Repo](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| DeepSeek-V4-Flash-0731 | FP8 · vLLM pipeline | 3 | 16,384 | — | — | — | — | — | — | — | — | — | Pending evidence repair · [repo PR 1](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX/pull/1) | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 · SM80 vLLM fork | 4 · PP4 | 16,384 | c=1; ladder 1/2/4/8 | 325.5 tok/s (2,941 input tokens) | 59.78 tok/s warm; 56.6 tok/s sustained | 169.65 tok/s @ c=4 | 0.163 s warm | — | Text passed; image rejected (HTTP 400) | Loaded sample: 114–137 W/card peak, core ≤46 °C, no throttle flags | — | Provisional measured text baseline; vision unavailable and runtime source revision unpinned | [Summary](../results/2026-08-31-deepseek-v4-flash-vision-exp-cmp170hx.md) · [Redacted data](../results/2026-08-31-deepseek-v4-flash-vision-exp-cmp170hx.json) |

## GLM-5.3-Flash UD-IQ4_XS, four cards (publication-safe)

**Measured:** 4 × 64 GiB CMP 170HX, layer split (`--tensor-split 1,1,1,1`), 16,384 context, unslothai/llama.cpp at commit 00699716c275498ff84d71e329178fe21cba56a6, driver 610.43.03, kernel 6.8, Ubuntu 22.04. Model: unsloth/GLM-5.3-Flash-GGUF at revision 2975ab414d30340466d8c51533c6e91f0cca64c1, 5-shard UD-IQ4_XS (~146 GiB).

| Metric | Value |
|---|---:|
| Decode, c=1 median (5 reps after 3 warmups) | 17.73 tok/s |
| End-to-end per task, c=1 | 14.44 s |
| Aggregate, c=2 and c=4 ladder (2 reps each) | ~17.5–17.7 tok/s (flat) |
| Soak, 20 min at c=2 | 41/41 reps ok, stable 17.5→17.7 tok/s |
| Quality, corrected 26-task pack | 21/26 (math 8/8, instruction 4/5, long-context 3/3, held-out math 4/4, held-out code 1/1, coding 1/3, held-out instruction 0/2) |
| Prefill / TTFT / ITL | — (not captured in sanitized receipts) |
| Power, end-of-run snapshot per card | 40.10 / 44.49 / 40.64 / 41.82 W (configured limit not queried) |
| Temperatures, end-of-run snapshot | core 41–43 °C, memory 45–55 °C (continuous/peak not recorded) |
| VRAM, end-of-run snapshot per card | 32,656–42,312 of 65,536 MiB |
| Energy | single-snapshot power proxy; not integrated energy and not a lower bound |
| Fault telemetry | continuous throttle/Xid/thermal telemetry not recorded this run; no fault-free-operation claim is made |

Aggregate curve flatness beyond c=4 and at longer contexts is untested. Result card: [results/2026-08-30-glm-5.3-flash-ud-iq4xs-llamacpp-cmp170hx.md](results/2026-08-30-glm-5.3-flash-ud-iq4xs-llamacpp-cmp170hx.md).

## Qwen3.8-27B NVFP4, one card — pending evidence repair

**Status:** the canonical sanitized receipts for the three-card runs are unmerged ([repo PR 1](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX/pull/1)). The numbers below are platform context, not decision-grade, until that repair lands.

**Measured (context):** three separate CMP 170HX cards at a 180 W limit.

| Card | Decode, 256 output tokens | Decode, 900 output tokens | TTFT | Prefill |
|---|---:|---:|---:|---:|
| 0 | 135.31 tok/s | 121.28 tok/s | 201.2 ms | 1,957.3 tok/s |
| 1 | 140.27 tok/s | 124.78 tok/s | 189.7 ms | 1,954.7 tok/s |
| 2 | 133.57 tok/s | 119.94 tok/s | 181.4 ms | 1,926.0 tok/s |
| Mean | **136.38 tok/s** | **122.00 tok/s** | **190.8 ms** | **1,946.0 tok/s** |

Peak observed core temperature was 51 °C; peak observed memory temperature across the three runs was 61 °C. A single-card hosted reference reported 147.7 tok/s at 255 W; it is context, not a controlled apples-to-apples comparison.

Reproduction and raw outputs: [PixelML/Qwen3.8-27B-CMP-170HX](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX).

## DeepSeek-V4-Flash-0731, three cards — pending evidence repair

**Status:** canonical sanitized receipts are unmerged ([repo PR 1](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX/pull/1)). Numbers below are platform context until the repair lands.

**Measured (context):** 3 × 64 GiB cards, pipeline parallel size 3, 180 W/card, FP8 KV cache, 16,384 maximum sequence length, and speculative decoding with `k=5`.

| Prompt class | Decode throughput |
|---|---:|
| Technical | 73.4 tok/s |
| Prose | 72.4 tok/s |
| Code | 116.6 tok/s |
| Aggregate | **83.3 tok/s** |

Measured prefill reached 2,965 tok/s at 5,399 input tokens. Draft-token acceptance ranged from 5.07 to 5.32 tokens, or roughly 81–86%. The 48-shard checkpoint was about 148 GB. Cold startup included approximately 22 minutes of weight loading from shared storage plus approximately seven minutes of CUDA graph capture.

The precompiled runtime image lacked `vllm._C` for this path; a source build was required. Full details: [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX).

## DeepSeek-V4-Flash-Vision-Exp, four cards — provisional text baseline

**Status:** the text path is measured, but this is not a publication-safe
multimodal recipe. The measured SM80 runtime rejected image input, and its
source revision was unavailable from the running image. The result stays
provisional until vision passes and the runtime is immutably pinned.

**Measured:** 4 × 64 GiB CMP 170HX, pipeline parallel size 4 with layer
partition `11,11,11,10`, 16,384 maximum model length, FP8 KV cache, and
DSpark speculative decoding with `k=6`. Model revision:
`86f746b36186f0e567729a5c06a8c918caba82a9`.

| Metric | Value |
|---|---:|
| Warm single-stream decode | **59.78 tok/s** |
| Sustained decode, 800 completion tokens | **56.6 tok/s** |
| Uncached prefill, 2,941 input tokens | **325.5 tok/s** |
| Warm TTFT | **0.163 s** |
| Aggregate decode, c=1 / 2 / 4 / 8 | 101.21 / 114.68 / **169.65** / 133.95 tok/s |
| Image request | Rejected, HTTP 400 |

Concurrency 4 was the stable throughput peak in the measured ladder. The
concurrency-16 attempt wedged in the speculative draft path; it is outside the
published stable envelope. The vision failure is a runtime compatibility gate,
not evidence that the checkpoint itself is text-only.

Public redacted snapshot: [result summary](../results/2026-08-31-deepseek-v4-flash-vision-exp-cmp170hx.md)
and [structured data](../results/2026-08-31-deepseek-v4-flash-vision-exp-cmp170hx.json).

## Negative results matter

### GLM-5.3-Flash NVFP4

**Compatibility result, not a performance result:** the published NVFP4 artifact targets an SM121 runtime path and is not directly compatible with the SM80 CMP 170HX. An AWQ INT4 alternative was measured at about 198.1 GiB of safetensors, or roughly 66 GiB per card under simple three-way sharding before runtime/KV overhead, so it does not fit 3 × 64 GiB as-is.

Do not advertise a throughput number until a compatible model format, runtime path, and memory plan are demonstrated.

## Measurement rules

Every submitted result must include:

- exact model repository + revision and quantization;
- runtime image/commit, CUDA, driver, kernel, and launch command;
- card count, parallelism, PCIe topology, power limit, peak draw, and temperatures;
- input/output tokens, concurrency, batch size, context length, warmup, and sample count;
- raw redacted output and an explanation of the metric calculation.

For server-sent-event APIs, count generated tokens from the final `usage.completion_tokens`. Counting stream events produced incorrect results in an earlier harness because events are not tokens.

Rows marked `pending evidence repair` are restored to full decision-grade status only when the owning model repository merges the sanitized receipt chain, and the README scoreboard is updated in the same workflow.

Use the template in [results/README.md](../results/README.md).
