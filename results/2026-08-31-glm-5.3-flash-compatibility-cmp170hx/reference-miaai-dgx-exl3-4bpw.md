# Reference — MiaAI-Lab EXL3/TR3 4 bpw on 2x DGX Spark (external)

Status: external reference; not a CMP attempt
Date: 2026-08-30

## What this is

[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
@ `79f10b91f84779b2b1ff2c9327b1a5847cd97f70` (repo tree HEAD, checked
2026-08-30; MIT license) serves GLM-5.3-Flash on **two GB10 DGX Sparks**
(arm64, CUDA 13.0, native `sm_121a` cubins, TP=2 over CX7). It is a
methodology source and a feasibility datapoint for the EXL3/TR3 4 bpw
checkpoint — **not** a portable CMP recipe.

## Checkpoint provenance (cited, not mirrored)

- Served weights: [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw),
  a self-declared byte-identical mirror of
  [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  snapshot `5ab363a8` (community-reported identity; exact full SHA not shown
  in the README).
- Format: EXL3/TR3, uniform K4 group codebooks, **4 bpw on routed experts
  only**; dense/shared/attention/embeddings/lm_head stay native (bf16).
- Size: ~164 GiB download / ~176 GB installed, 120 shards
  (community-reported; not re-measured here).
- Quality gate (weights-only, external): teacher-logit panel by malaiwah —
  mean KLD(teacher || model) **0.024555 nats** at 4 bpw vs official FP8
  0.020615 / 0.024629 on the same protocol; NVFP4 0.060535
  (community-reported, five cold runs, 25 sealed windows, 51,175 positions).
  This scores the checkpoint, not any CMP runtime, and is not a universal
  quality score.
- License: the DGX repo is MIT; the checkpoint card carries its own terms
  (ShapleyMCG License 1.0 per the earlier
  [EXL3 4 bpw attempt record](../exl3-tr3-4bpw-exllamav3/README.md)).
  Redistribution not verified; nothing is mirrored here.

## Why none of the runtime stack transfers to CMP 170HX

| DGX stack component | SM12x assumption | CMP (SM80) reality |
|---|---|---|
| Target KV `fp8_ds_mla` | packed 656 B NoPE-MLA record (512 latent + scales + RoPE pad), padded into GLM_NSA geometry by the overlay | `pe_dim must be 64 for fp8_ds_mla` — this exact failure is what the overlay exists to patch; no SM80 sparse-MLA backend is known |
| Attention `FLASHINFER_MLA_SPARSE_SM120` | SM120+ FlashInfer sparse MLA | Not built for SM80; no equivalent backend identified |
| ExLlamaV3 `exl3_moe` fused kernels | pin `c5d9c657` (0.0.43), sm_121a cubins, aarch64 allreduce stubs | No SM80 build; gate returns capability 80 (Ampere) and stops |
| DFlash2 k=7 speculator | EAGLE3 aux taps at mHC, padded slot-share page layout, `is_causal: false` draft | Speculator deps are stack-specific; untested and unpinned for CMP |
| Image | `vllm/vllm-openai:glm53-flash-arm64-cu130` overlay, CUDA 13, arm64 | x86_64 CMP node; image explicitly not reusable |

Per delegation: no SM121-only kernel, `fp8_ds_mla` assumption, CX7 topology,
CUDA-13 image, or overlay patch is ported unless a source-level review proves
an Ampere-compatible component. No such component was identified in this pass.

## Feasibility on the CMP node (arithmetic, not measured)

- Four-card node (current), TP=4: 176 GB installed / 4 = ~44 GB/card = ~40.9
  GiB/card at a perfect weight balance, leaving ~23 GiB/card for CUDA context,
  activations, and KV before imbalance. (Three-card era, TP=3: ~54.6 GiB/card,
  ~9 GiB/card headroom — the tightest fit of any candidate at the time.)
  Routed-expert byte skew across TP ranks is unverified at either topology.
- The runtime stack (not the arithmetic) is the binding blocker; see
  [EXL3 4 bpw attempt](../exl3-tr3-4bpw-exllamav3/README.md).

## Measurement ideas adopted for CMP harnesses

Source: `tests/bench_decode.py` @ `79f10b91`, `tests/bench_prefix_cache.py`,
`docs/cold-prefill.md` + `_run_cold_prefill.py` (MIT).

1. **Structured vs prose decode phases** — spec-decode acceptance is highly
   regime-dependent (DGX lab medians: structured 61.7 tok/s at 0.918
   accept/step vs prose 26.9 at 0.332; community-reported). CMP protocol:
   measure both a structured count task and free prose, report acceptance
   separately, never mix them into one headline number.
2. **Median of 5 x 400-token runs** at temperature 0, single stream, with
   warmup discarded; tokens from the final usage object (already used by
   `bench/measure.py`; extend runs 3 -> 5 when time allows).
3. **Cold vs warm TTFT split** — cold prefill on a large context is measured
   separately from warm repeated-prompt TTFT (`docs/cold-prefill.md`).
4. **Prefix-cache hit testing** — repeated-prefix latency vs cold at matched
   lengths, before claiming caching benefits (adapt `bench_prefix_cache.py`
   ideas; llama.cpp server exposes prompt caching but not the same API).
5. **Failure classification** — every failed run keeps HTTP/error class,
   stage (load/prefill/decode), and raw error text in evidence, not just a
   boolean.

## Externally reported performance (context only, not CMP)

DGX Spark kit, DFlash2 k=7, high-accept structured tasks, warm/empty KV:
62.9 tok/s single stream, 719 ms TTFT, 146.5 tok/s aggregate at concurrency 4
(community-reported). Long-context mixed 24-27 tok/s. Not comparable to CMP
hardware, runtime, or topology; do not cite next to CMP numbers except as
labeled external context.

## Re-run instructions

Not runnable on CMP. On the intended GB10 kit, follow the upstream README
@ `79f10b91` directly. Any future CMP EXL3 work first needs an SM80 build of
`exllamav3_ext` plus an SM80 attention backend for NoPE MLA — see the
[EXL3 4 bpw attempt record](../exl3-tr3-4bpw-exllamav3/README.md).
