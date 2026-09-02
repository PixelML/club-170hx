# Attempt — NVFP4 checkpoint on vLLM SM121 fork

Status: compatibility-only, failed
Date: 2026-08-30

## Checkpoint

- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) @ `11d73216cd636238e82e1d77fe1042ffab36e7fa`
- ~194.4 GiB download across 120 shards (community-reported, from the DGX recipe documentation; not re-measured here)
- NVFP4 ModelOpt weights, marlin MoE backend
- License: base-model terms apply; verify before any redistribution

## Runtime

- The DGX recipe serves this checkpoint on an SM121/arm64 GB10 image
  ([reference implementation](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark)).
- SM80 support: **no** — the serving path uses sparse-MLA attention backends
  (FLASHINFER_MLA_SPARSE_SM120/SM90 class) that do not exist for SM80
  (measured: registry grep + PR review, 2026-08-30).

## Static fit calculation

~194.4 GiB across 4 cards at TP=4 = ~48.6 GiB/card before CUDA context,
activations, or KV — fits on paper on the current 4-card node. (The three-card
era fit was ~64.8 GiB/card at TP=3, over budget.) The runtime incompatibility
is unchanged and remains the blocker.
(Fit number inferred from community-reported size; per-card arithmetic is exact.)

## Execution status and outcome

Not executed on CMP hardware — blocked before download by the runtime
incompatibility and the preserve-gate on the live test node. No bytes were
downloaded; no attempt was forced.

## Blocker

`glm5_next` (`Glm5NextForConditionalGeneration`) is absent from upstream
vLLM's model registry (measured 2026-08-30 against main). The only known
support PR, [vllm#53906](https://github.com/vllm-project/vllm/pull/53906), is
open, unmerged, and SM90+ only.

## Evidence

- Upstream registry (checked 2026-08-30, no `glm5` match):
  [vllm/model_executor/models/registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)
- Support PR (open, SM90+): [vllm#53906](https://github.com/vllm-project/vllm/pull/53906)
- DGX reference recipe: [PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark)

## Re-run instructions

Blocked until upstream lands an SM80-capable `glm5_next` backend. Once it
does: serve this checkpoint with TP=4 across the four-card node, 180 W power
policy, forced airflow, stop at 80 C core / 85 C memory, abort on any Xid, and
measure per the [AWQ attempt's methodology section](../awq-int4-vllm/README.md#re-run-instructions).
