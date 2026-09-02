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
| Qwen3.8-27B | W4A16 AutoRound + DFlash2 k=7 · vLLM | 1 (3 cards tested) | 64k | c=1 | 1,946 tok/s | 136.38 tok/s (256 tok) / 122.00 (900 tok) | — | 190.8 ms | — | 3 cards, 3 runs | 180 W cap; peak 51 °C core / 61 °C memory | — | Measured 2026-08-30 | [Repo](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| DeepSeek-V4-Flash-0731 | FP8 · SM80 vLLM fork · PP3 · DSpark k=5 | 3 | 16,384 | c=1, three prompt classes | 2,965 tok/s (5,399 in) | 83.3 tok/s aggregate (73.4 / 72.4 / 116.6) | — | — | — | 400 tok × 3 classes | 180 W cap | acceptance 5.07–5.32 | Measured 2026-08-30 | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 · SM80 vLLM fork | 4 · PP4 · DSpark k=6 | 16,384 | c=1; ladder 1/2/4/8/16 | 2,352 tok/s warm (362 tok/s first cold prefill) (2,941 input tokens) | 97.4 tok/s (median of 3; 57.6–123.5) (c=1) | 165.5 tok/s (median of 3; 140.3–203.2) @ c=4; failed (device-side assert, reproduced twice) @ c=16 | 0.394 s warm | — | Text passed; image not served on this path | — | — | Benchmark in progress; supersedes the earlier ladder | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) · [Section](#deepseek-v4-flash-vision-exp-four-cards) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 → BF16 fallback · reference TP4 runtime + SM80 patches | 4 · TP4 | ≤ ~1,024 input tokens (OOM above) | batch 1 | 512 tokens in ~8.7–9.8 s | 0.88–0.93 tok/s (3 × 401 tokens) | — | ~2.05–2.08 s proxy | — | Real-image completion PASS; no-image and wrong-image controls PASS | — | — | Correctness evidence only, not a performance result | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) · [Milestone](#vision-correctness-milestone) |

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

## DeepSeek-V4-Flash-Vision-Exp, four cards

**Checkpoint:** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, FP8 e4m3 weights,
48 shards, revision `86f746b36186f0e567729a5c06a8c918caba82a9`.
**Hardware:** 4 × 64 GiB CMP 170HX, driver 610.43.03, kernel 6.8, Ubuntu 22.04.
Detailed receipts, launch scripts, and patches live in
[PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX).

Two runtimes ran on this checkpoint. The text path answers the performance
question. The vision path answers the correctness question. Do not combine
their numbers.

### Text path: PP4 + DSpark k=6 (benchmark in progress)

**Setup:** SM80 vLLM fork, pipeline parallel size 4 with layer partition
`11,11,11,10`, 16,384 maximum model length, 2,048 maximum batched tokens,
FP8 KV cache, DSpark speculative decoding with `k=6`. Greedy decoding, 400
completion tokens per request, warm engine, token counts from final usage
objects, three repetitions per level.

| Metric | Value |
|---|---:|
| Decode, c=1 | 97.4 tok/s (median of 3; 57.6–123.5) |
| Aggregate, c=4 | 165.5 tok/s (median of 3; 140.3–203.2) |
| Aggregate, c=16 | failed (device-side assert, reproduced twice) |
| Uncached prefill, 2,941 input tokens | 2,352 tok/s warm (362 tok/s first cold prefill) |
| Warm TTFT | 0.394 s |
| Aggregate, c=2 | 103.7 tok/s (median of 3; 96.6–159.2) |
| Aggregate, c=8 | 220.2 tok/s (median of 3; 206.3–232.0) |

The placeholders are filled when the normalized run completes and its receipts
merge in the model repository. Until then this table is not decision-grade.

Full tables, per-request spread, telemetry, and the throughput-vs-concurrency
chart are in the executed notebook:
[notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb](../notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb).
Raw receipts: [results/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm/](../results/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm/).

**Earlier run, superseded.** An earlier ladder on the same topology reported
101.21 / 114.68 / 169.65 / 133.95 tok/s aggregate at c=1 / 2 / 4 / 8, with a
c=16 wedge in the speculative draft path. A separate mixed-content single-stream
run reported 59.78 tok/s warm decode, 0.163 s warm TTFT, and 325.5 tok/s
uncached prefill. The two runs used different protocols, and the runtime
source revision was unavailable from the running image. The numbers stay
visible as measured learning, but they are superseded by the normalized run.
Snapshot: [result summary](../results/2026-08-31-deepseek-v4-flash-vision-exp-cmp170hx.md).

### Vision-correctness milestone

**Result:** the first real-image inference of this checkpoint on Ampere (SM80)
hardware. The runtime is the reference TP4 (tensor parallel 4) implementation
plus SM80 fallback patches. The server passed `/v1/models`, a deterministic
text request, a real 64 × 64 gradient-image request (the model named the
gradient colors, which appear nowhere in the prompt), and no-image and
wrong-image controls.

SM80 fallback patches on the reference path:

- block-scale broadcast fix for the FP8 block-scaled weights;
- pure-torch Sylvester Hadamard transform in place of the CUDA
  `fast_hadamard_transform` extension (exactness checked: `H @ H.T = n * I`);
- sinkhorn layout fix in the hyper-connection split fallback (flat
  `pre | post | comb` layout instead of an `(m, m+2)` grid);
- FP4 dequant to BF16 with a BF16 GEMM fallback for linear layers, because
  the tilelang FP8 × FP4 GEMM raises a device assert on SM80 (it requires
  SM89 FP8 MMA); the vision tower runs BF16 SDPA.

| Metric (TP4 reference runtime, NVMe-staged weights) | Value |
|---|---:|
| Decode, c=1, greedy 401 tokens, 3 reps | 0.88–0.93 tok/s (about 0.9 tok/s) |
| Wall time per 401-token completion | 431–454 s |
| Prefill, 512 input tokens | 8.7–9.8 s wall |
| Prefill, 256 input tokens | 7.4–7.5 s wall |
| TTFT proxy, 3 reps | 2.05–2.08 s |
| Single-request prefill, ≥ ~1,024 tokens | OOM (7.8–11.5 GiB sparse-attention allocation against 2–7 GiB free) |
| Weights resident per card | about 44.4 GiB |

**This is correctness evidence, not a performance result.** The reference
runtime serves one request at a time, does not stream, has no speculative
decoding, and dequantizes to BF16 on the fly. Its decode rate says nothing
about what the vLLM text path can reach. Do not place 0.9 tok/s beside any
other row on this page.

### SM80 compatibility gaps in the upstream Vision vLLM path

The pinned upstream Vision vLLM source assumes SM90-class kernels in several
places. Each gap below stopped a four-rank PP4 boot at a different stage. The
gaps were found one at a time, in this order, because each one only appears
after the previous one is fixed.

| # | Stage | Failure | Root cause | Status |
|---|---|---|---|---|
| 0 | Weight load | Workers killed by the host OOM killer at 7 of 48 shards, no traceback | The eager safetensors load strategy reads each whole shard into host RAM; 4 ranks × concurrent whole-shard reads exhausted 94 GiB of host RAM | Fixed: drop the eager strategy and use streaming `safe_open` |
| 1 | KV-cache profiling | `RuntimeError: DeepGEMM backend is unavailable` on the first pipeline rank | The MHC broadcast path calls the DeepGEMM TF32 prenorm GEMM unconditionally; upstream gates the fallback on package presence, not on architecture support | Fixed: Triton fallback, selected with an architecture-aware predicate (`is_deep_gemm_supported()`) |
| 2 | Attention profiling | Triton compile error `type fp8e4nv not supported in this architecture` | The fused inverse-RoPE FP8 quantization kernel for `o_proj` stores `fp8e4nv`, which this Triton build cannot emit for sm_80 | Fixed: torch BF16 fallback, gated on compute capability < 9 |
| 3 | KV-cache warmup | NVVM backend compilation failed for target sm_80 | The CuteDSL `fused_indexer_q` RoPE + FP8 quantization kernel does not compile for sm_80; the Triton alternative also stores `fp8e4nv` | Fixed: pure-torch fallback that reproduces the Triton numerics (GPT-J interleaved RoPE, power-of-two per-token scales); SM90+ path unchanged |
| 4 | First multimodal request | `vision MoE routing requires input_ids` under pipeline parallel | Structural: the vision MoE router reads `input_ids` to route image tokens, and non-first pipeline ranks receive hidden states only, never `input_ids` | Unresolved. The route is frozen; image requests are not served on the PP4 vLLM path |

Gaps 1–3 are capability-gated so that the SM90+ path is byte-identical. Gap 4
is why the performance numbers on this page are text-only and the vision
milestone runs on the TP4 reference runtime instead.

### Negative results, four cards

- **The reference TP4 runtime is batch 1.** It is not a serving engine. No
  concurrency ladder was run on it, and none should be.
- **Shared-storage load is slow.** The checkpoint streamed from NFS at about
  31 MiB/s aggregate. A cold start took about 30–45 minutes of weight loading
  before profiling and graph capture began. Staging the weights on local NVMe
  removed this cost.
- **2,941-token prefill OOM on the reference TP4 path.** With about 44.4 GiB
  of weights per card, the sparse-attention fallback needs 6.8 GiB per rank
  for a 2,941-token request and 7.8 GiB for a 2,048-token request. Both
  exceeded the free memory. Reliable single-request prefill on this path is
  below about 1,024 tokens.
- **NVRM VA-space corruption after an OOM kill storm.** After the four ranks
  were killed mid-prefill, the kernel log showed NVRM assertion failures on
  all four GPUs, and CUDA initialization failed host-wide
  (`CUDA_ERROR_NO_DEVICE`). ECC and retired-page counters stayed clean.
  Reloading `nvidia_uvm` alone did not help. A full `rmmod nvidia_uvm nvidia`
  and `modprobe nvidia nvidia_uvm` sequence restored the devices without a VM
  reboot. Treat this as a driver-state hazard whenever a multi-rank OOM
  occurs.
- **Only one PP4 partition boots.** Layer partitions `12,12,12,7` and
  `12,12,11,8` failed before serving traffic (exit 137 and a first-request
  device-side assert). `11,11,11,10` is the only stable partition found on
  this checkpoint.

### Cross-platform context: two-node DGX Spark

The same checkpoint at the same revision runs on a two-node DGX Spark kit
(GB10, vLLM, TP=2, `fp8_ds_mla` KV cache). Published there (merged evidence): c=1 36.9 tok/s, c=6 112.7 tok/s
aggregate, uncached prefill 1,789 tok/s, streaming TTFT 0.239 s, vision PASS.
A later normalized run on the 2,941-token / 400-token protocol (c=1 48.7,
c=16 106.8 aggregate) is in that repository's open PR and is not yet merged. Results and
receipts: [PixelML/DeepSeek-V4-Flash-Vision-Exp-DGX-Spark](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-DGX-Spark).

This is context, not a head-to-head comparison. The platforms differ in
runtime revision, parallelism, memory budget, interconnect, and power. The
DGX Spark path serves images through upstream vLLM; the CMP path does not.

### Credits

The SM80 vLLM fork, the DSA/MTP kernels, and the four-card reference recipe
come from [allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80)
and [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx).
The SM80 fallback patches above extend that work. The three-card baseline
is [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX)
(83.3 tok/s aggregate decode, PP3, DSpark k=5).

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
