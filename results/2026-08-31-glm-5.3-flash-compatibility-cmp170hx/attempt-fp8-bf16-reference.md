# Attempt — official FP8 and BF16 checkpoints

Status: static-fit-only, failed
Date: 2026-08-30

## Checkpoint

- [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (FP8) and
  [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)
- Sizes: >= ~328 GB FP8; BF16 larger (community-reported sizes; the FP8 release
  notes reference this magnitude — not re-measured here)
- Official zai-org releases

## Runtime / SM80 / fit

Moot on memory grounds: even at TP=4 these need ~76+ GiB per card (FP8; BF16
larger) against a 64 GiB physical budget. No fit calculation beyond that is
meaningful at either the three- or four-card topology.

## Execution status and outcome

Not executed. Size alone rules the node out.

## Blocker

Static fit, by a factor of ~1.2x per card at TP=4 (~2.6x at the three-card
TP=3 era) even before overhead.

## Evidence

- HF model cards (community-reported sizes)
- This conclusion does not depend on runtime compatibility, which is
  separately blocked per the [NVFP4 attempt](../nvfp4-vllm-sm121/README.md).
