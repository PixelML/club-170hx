# DeepSeek-V4-Flash-Vision-Exp — provisional text baseline

Status: measured, provisional
Date: 2026-08-31

The four-card text path reached **59.78 tok/s warm decode**, **325.5 tok/s
uncached prefill**, and **169.65 tok/s aggregate at concurrency 4**. The tested
SM80 runtime rejected image input, so this is not yet a successful vision
recipe.

## Hardware

- Cards: 4 × CMP 170HX, 64 GiB each
- Topology: pipeline parallel 4, layer partition `11,11,11,10`
- Loaded telemetry sample: 114–137 W per-card peak; core temperature no higher
  than 46 °C; no throttle flags
- Configured power limit and integrated energy: not recorded in the public
  snapshot

## Software

- Runtime: source-built SM80 vLLM fork
- Runtime image content digest:
  `sha256:0e33b051516583ac8f9d2449a5d9889cadad8e43c121cc454d597c251986ddbe`
- Runtime source revision: unavailable from the measured image; this blocks a
  publication-safe recipe
- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- Model revision: `86f746b36186f0e567729a5c06a8c918caba82a9`
- Checkpoint format: FP8 e4m3 weights with dynamic activations and 128 × 128
  blocks

## Method

- Maximum model length: 16,384
- Maximum batched tokens / sequences: 2,048 / 8
- KV cache: FP8
- Speculative decoding: DSpark, 6 speculative tokens
- Single-stream result: greedy 400-token completions over three prompt classes;
  token counts came from final API usage records
- Sustained result: one 800-completion-token request
- Prefill result: 2,941 input tokens and one output token
- Concurrency ladder: one measured batch each at c=1, 2, 4, and 8 with 400
  completion tokens per request

## Results

| Metric | Value |
|---|---:|
| Cold single-stream decode | 51.06 tok/s |
| Warm single-stream decode | **59.78 tok/s** |
| Sustained decode | **56.6 tok/s** |
| Cold / warm TTFT | 0.214 / **0.163 s** |
| Uncached prefill | **325.5 tok/s** |
| Aggregate c=1 / 2 / 4 / 8 | 101.21 / 114.68 / **169.65** / 133.95 tok/s |

## Correctness and failures

- Text requests completed and returned final usage accounting.
- Image input was rejected with HTTP 400 because the tested SM80 runtime did
  not wire the vision tower.
- Concurrency 4 was the measured stable throughput peak.
- Concurrency 16 wedged in the speculative draft path and is excluded from the
  stable envelope.
- No task-quality evaluation was captured in this snapshot.

## Evidence

- [Redacted structured snapshot](2026-08-31-deepseek-v4-flash-vision-exp-cmp170hx.json)
- Classification: measured text performance; vision compatibility failed;
  immutable runtime source pin untested
