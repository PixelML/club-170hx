# Candidate — a checkpoint that would actually fit

Status: not-attempted (no runtime-compatible candidate existed as of 2026-08-30)
Date: 2026-08-30

## What would qualify

A W4A16/AWQ/GPTQ-class quant whose **exact blob total is <= ~220 GiB**
(55 GiB x 4 cards), leaving per-card room for CUDA context, activations, and a
usable KV pool at TP=4. Practically: a true 3-bit quant of GLM-5.3-Flash, or a
4-bit quant with expert-layer exclusions. Two checkpoints already fit the
four-card budget on paper — the 198.1 GiB AWQ INT4 (49.5 GiB/card; blocked by
runtime support, see [AWQ attempt](../awq-int4-vllm/README.md)) and the
146.05 GiB UD-IQ4_XS GGUF (36.5 GiB/card; since measured on 2026-08-30, see
[GGUF attempt](../gguf-ud-iq4xs-llamacpp/README.md)). Size alone is therefore
not the open question here; no runtime-compatible higher-quality lane was identified or tested.

## Search performed

HF search for `GLM-5.3-Flash` on 2026-08-30 (community-visible quants
enumerated): NVFP4, EXL3/TR3 4bpw, AWQ INT4, FP8, BF16, GGUF, MLX. None of
the higher-bit-width candidates met the byte budget; the one fitting
candidate with a usable runtime found at that date is the measured
llama.cpp UD-IQ4_XS GGUF lane (see gguf-ud-iq4xs-llamacpp). Re-run this search
before assuming the conclusion still holds — new quants appear frequently.

## Also required

Upstream vLLM support for `glm5_next` — currently absent (see
[NVFP4 attempt](../nvfp4-vllm-sm121/README.md)). Both blockers must clear
independently.

## Re-run trigger

When a candidate appears: verify exact blob bytes from the HF API (not the
model card), apply the [static-fit method](../awq-int4-vllm/README.md#static-fit-calculation),
then the [benchmark methodology](../awq-int4-vllm/README.md#re-run-instructions).
