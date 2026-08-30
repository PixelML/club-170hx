# Benchmarks

Results here are application measurements on the tested CMP 170HX setup. They are not vendor specifications or theoretical estimates.

## Qwen3.8-27B NVFP4, one card

**Measured:** three separate CMP 170HX cards at a 180 W limit.

| Card | Decode, 256 output tokens | Decode, 900 output tokens | TTFT | Prefill |
|---|---:|---:|---:|---:|
| 0 | 135.31 tok/s | 121.28 tok/s | 201.2 ms | 1,957.3 tok/s |
| 1 | 140.27 tok/s | 124.78 tok/s | 189.7 ms | 1,954.7 tok/s |
| 2 | 133.57 tok/s | 119.94 tok/s | 181.4 ms | 1,926.0 tok/s |
| Mean | **136.38 tok/s** | **122.00 tok/s** | **190.8 ms** | **1,946.0 tok/s** |

Peak observed core temperature was 51 °C; peak observed memory temperature across the three runs was 61 °C. A single-card hosted reference reported 147.7 tok/s at 255 W; it is context, not a controlled apples-to-apples comparison.

Reproduction and raw outputs: [PixelML/Qwen3.8-27B-CMP-170HX](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX).

## DeepSeek-V4-Flash-0731, three cards

**Measured:** 3 × 64 GiB cards, pipeline parallel size 3, 180 W/card, FP8 KV cache, 16,384 maximum sequence length, and speculative decoding with `k=5`.

| Prompt class | Decode throughput |
|---|---:|
| Technical | 73.4 tok/s |
| Prose | 72.4 tok/s |
| Code | 116.6 tok/s |
| Aggregate | **83.3 tok/s** |

Measured prefill reached 2,965 tok/s at 5,399 input tokens. Draft-token acceptance ranged from 5.07 to 5.32 tokens, or roughly 81–86%. The 48-shard checkpoint was about 148 GB. Cold startup included approximately 22 minutes of weight loading from shared storage plus approximately seven minutes of CUDA graph capture.

The precompiled runtime image lacked `vllm._C` for this path; a source build was required. Full details: [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX).

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

Use the template in [results/README.md](../results/README.md).
