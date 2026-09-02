# Quantization plan: DeepSeek-V4-Flash-Vision-Exp W4A16 for Ampere

Status: draft, not committed. Owner of commits on this branch is a separate agent.
Evidence labels follow repo convention: **measured**, **inferred**, **community-reported**, **untested**.

## 1. Recommendation

**Go**, with conditions. Build one artifact:

`PixelML/DeepSeek-V4-Flash-Vision-Exp-W4A16-Ampere`

No published, well-adopted W4A16/AWQ/GPTQ artifact of this model family exists yet on
Hugging Face. Two amateur attempts exist for the older `-0731` text-only checkpoint
(`baicai1145/DeepSeek-V4-Flash-0731-W4A16`, `True2456/DeepSeek-V4-Flash-0731-AWQ`); neither
targets Marlin/vLLM on SM80, and one is nearly as large as the FP8 source (166 GB) because it
upcast too much of the non-expert weight to BF16. That is the gap this artifact fills, and the
mistake to avoid.

**Top risk:** two open vLLM bugs threaten correctness/stability on Ampere specifically for
MoE W4A16, independent of whether quantization itself succeeds:
- [vllm#35922](https://github.com/vllm-project/vllm/issues/35922): `fused_marlin_moe`
  `cudaErrorIllegalAddress` for W4A16 INT4 MoE on A100/SM80.
- [vllm#41511](https://github.com/vllm-project/vllm/issues/41511): compressed-tensors W4A16
  MoE under tensor parallelism reports `group_size=128` but computes `group_size=16`
  (an 8x mismatch tracking TP degree), corrupting weight-scale sharding.

Both must be tested against on real SM80 hardware (the 4x170HX box) before this artifact is
called usable. If either reproduces, the fallback is MoE-WNA16 kernels (vLLM default for
&gt;32 experts) instead of Marlin-MoE, and TP=1 per node with pipeline/data parallel across
nodes instead of tensor parallel across cards.

## 2. Tool and recipe

**Tool: `llm-compressor` (vllm-project)**, not AutoAWQ (archived, no deepseek_v4 support),
not AutoRound (its own README says MoE/VLM support is "currently limited"; a third-party
`Intel/DeepSeek-V4-Flash-W4A16-AutoRound` card states vLLM/SGLang do not run it), and not
GPTQModel as primary (it lists DeepSeek V4 support as of a 2026-05-13 dev release: newer
and less proven than llm-compressor's dedicated, documented DeepSeek-V4 page). Keep
GPTQModel as the fallback if llm-compressor's oneshot run fails on the vision-tower path.

llm-compressor references:
- [DeepSeek-V4 docs page](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/deepseek-v4/)
- [MoE quantization guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantizing_moe/)
- [Multimodal vision quantization guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/multimodal_vision/)
- [Memory guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/memory/)
- RedHat has already shipped `DeepSeek-V4-Flash-NVFP4-FP8` with this tool; that confirms the
  architecture path works end to end (different target precision, same loader/recipe shape).

Loading `AutoModelForCausalLM.from_pretrained` on the FP8-block checkpoint dequantizes to
BF16 automatically; no separate dequant step needed.

### Recipe skeleton (`recipe.yaml`)

```yaml
# DeepSeek-V4-Flash-Vision-Exp -> W4A16, Marlin/MoE-WNA16 target, SM80
quant_stage:
  quant_modifiers:
    GPTQModifier:
      ignore:
        - "lm_head"
        - "re:.*vision_tower.*"
        - "re:.*multi_modal_projector.*"
        - "re:.*mtp.*"          # keep the 3 DSpark/MTP draft layers out of this pass
        - "re:.*hash_layer.*"   # keep gate bias/router logic unquantized -- untested risk, see 5
      config_groups:
        group_0:
          weights:
            num_bits: 4
            type: "int"
            symmetric: true
            strategy: "group"
            group_size: 128
          targets: ["Linear"]
          # applies to routed-expert Linear layers only, via the ignore list above
```

Design goal, informed by the `baicai1145` size regression: do **not** blanket-upcast
non-expert weight to BF16. Where llm-compressor's exporter allows it, re-express attention,
shared-expert, and embedding weights that were FP8 in the source as FP8 W8A8 in the output
rather than BF16; this alone is the difference between a ~166 GB artifact and a ~100 GB one.
This step is **untested**; validate the exporter actually supports mixed INT4/FP8/BF16 output
before committing to the size estimate below.

Calibration: 512 samples x 2048 tokens is the confirmed llm-compressor/production default
([MoE quantization guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantizing_moe/)).
For the Vision-Exp variant, mix in image-grounded samples (the multimodal vision guide's
pattern) at roughly 20-30% of the calibration set so the vision-tower-adjacent projector
layers see real activations, even though the vision tower itself is excluded from
quantization.

## 3. Where to run the quantization job

**Run on one node of the DGX Spark pair (GB10, aarch64, ~120 GiB unified memory, ~1.4 TB free
NVMe)**, not the 4x170HX box. Reasoning:

- llm-compressor's sequential onloading needs the full BF16-dequantized model resident
  somewhere (measured comparable: GLM-4.6, a similarly-sized MoE, needed 768 GB RAM + 300 GB
  swap and was CPU-memory-bound, not GPU-bound; [GLM-4.6-AWQ card](https://huggingface.co/bullpoint/GLM-4.6-AWQ)).
  The DGX Spark unified memory architecture (GPU and CPU share the same 120 GiB pool, no PCIe
  copy) is a better fit for this access pattern than a discrete-GPU rig with fixed VRAM per
  card.
- The DGX Spark pair already has a local 157 GB snapshot of the exact target revision on both
  nodes (**measured**, confirmed under a per-user Hugging Face cache directory on each node),
  avoiding a second 168 GB download.
  However, 120 GiB unified memory alone will not hold the full ~330 GB BF16-dequantized
  intermediate; the job needs disk-offload (`offload_folder` in `from_pretrained`) onto the
  1.4+ TB free NVMe. Confirm llm-compressor's disk-offload path (used for GLM-5.2, &lt;2 hrs)
  before committing to this node, and budget for a slower run if it falls back to a smaller
  effective working set.
- The 4x170HX box is reserved for **inference-side validation** (the SM80/Marlin/MoE-WNA16
  target runtime), not the quantization compute, since neither DGX Spark node is SM80 and the
  quantization step itself does not depend on Marlin kernels.

Neither DGX Spark node currently has `llmcompressor` or `auto_round` installed, in the system
Python or any existing venv/container (**measured**, checked via `python3 -c "import ..."`
and `pip list` across the local Python environments on both nodes). Installing
`llm-compressor` is a required setup step before any job launches; watch for aarch64 wheel
availability: `llm-compressor`, `auto-gptq`-style native-kernel dependencies, and CUDA
extensions are commonly x86_64-only, and this is the single biggest unresolved feasibility
question. **Untested. Do not launch until wheel availability is confirmed on GB10.**

## 4. Estimated wall time

3-6 hours on a well-optimized multi-GPU/expert-sharded pipeline, or 8-15+ hours on a
single-accelerator CPU/disk-offload path, by extrapolation from:
- DeepSeek-V3/R1 671B GPTQ, optimized: ~2 hrs on 8xH100 ([IST-DASLab MoE-Quant](https://github.com/IST-DASLab/MoE-Quant))
- GLM-4.6 714B-class AWQ via llm-compressor: ~5 hrs on one 96 GB GPU, CPU-memory-bound
  ([GLM-4.6-AWQ card](https://huggingface.co/bullpoint/GLM-4.6-AWQ))
- Mixtral 8x7B GPTQ: 30 min-1.5 hrs on H100/A100

This model's BF16-dequantized size (~330 GB) sits below GLM-4.6/DeepSeek-V3 and above
Mixtral, but a single DGX Spark node's GB10 unified-memory profile is closer to the GLM-4.6
single-GPU case than the 8-GPU DeepSeek-V3 case. Plan for the higher end (~10-15 hrs) and
treat it as a background job with checkpointing, not an interactive session.

## 5. Estimated artifact size and card fit

- Routed experts (85-95% of ~163-170B total params) at INT4 + group-128 scale overhead:
  **~74-82 GB**.
- Remaining ~8-25B params (attention, embeddings, lm_head, shared experts, 32-layer vision
  tower, 3 MTP/DSpark layers) at BF16: **~16-50 GB**, or less if the FP8-preservation design
  goal in section 2 works.
- **Best case: ~95-110 GB** (lean non-expert path). **Worst case: ~160-170 GB** (naive
  blanket BF16 upcast, matching the community precedent).
- **2x 64 GiB (128 GiB total) fit: only viable in the best case, with thin KV headroom.**
  Do not advertise 2-card support until the actual artifact size is measured. **Recommend
  publishing the model card as a 3-4 card requirement by default**, and adding a 2-card note
  only after a real 128 GiB two-card load test succeeds with the intended context length and
  concurrency.

## 6. Validation plan (on the 4x170HX box)

Run only after explicit authorization per `club-170hx/AGENTS.md` (no autonomous GPU launches).

1. **Load/serve smoke test**: vLLM boots the artifact across the target card count (start at
   4 cards, then retest at 2-3 if size allows), confirms Marlin or MoE-WNA16 kernel selection
   in logs, and completes a single-token generation without `cudaErrorIllegalAddress` or the
   TP group-size mismatch from vllm#41511.
2. **Text correctness gate**: perplexity on a held-out WikiText2/C4 slice, plus a GSM8K
   subset (50-100 problems), compared against the FP8 baseline's published numbers. Set an
   explicit pass threshold (for example, GSM8K accuracy within 3 points of FP8) before running,
   not after seeing results.
3. **Vision correctness gate**: a small real-image prompt set (not synthetic-only) exercising
   the 32-layer vision tower end to end, spot-checked for caption/VQA quality against the FP8
   baseline. This is the artifact's differentiator versus text-only W4A16 attempts, so do not
   skip it.
4. **DSpark/MTP gate**: confirm the 3 draft layers still produce a usable acceptance rate with
   the quantized main model; vLLM's own MTP docs warn accuracy is not guaranteed for
   `num_speculative_tokens &gt; 1` on DeepSeek-family MTP, so measure rather than assume parity
   with the FP8 baseline's 55-70% acceptance (measured on the DGX Spark pair's GGUF/DSpark path, not this
   artifact).
5. **Stability/thermal gate**: per `club-170hx/docs/HARDWARE.md`, stop on 80 C core /
   85 C memory, Xid, or GPU disappearance during any of the above.

Record every result as measured/untested per the repo's evidence rules, with card count,
power limit, driver, vLLM commit, and quant recipe hash attached.

## 7. Hugging Face model card skeleton

```yaml
---
license: mit  # inherited from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp; verify base license text before publishing
base_model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
base_model_relation: quantized
tags:
  - sm80
  - ampere
  - cmp-170hx
  - a100
  - rtx-3090
  - awq       # or gptq/w4a16 -- set to match final compressed-tensors config
  - w4a16
  - vllm
  - moe
  - vision-language
  - deepseek_v4
---
```

Card body sections: architecture summary (43 layers, 256 routed experts, 32-layer vision
tower, 3 MTP/DSpark draft layers, hash-layer gates); exact source revision
(`86f746b36186f0e567729a5c06a8c918caba82a9`); quantization recipe and calibration set used
(link the recipe YAML); measured artifact size; measured minimum card count and total VRAM;
known-issue links (vllm#35922, vllm#41511) with current status; validation results from
section 6 with evidence labels; explicit statement that vision-tower and MTP/DSpark layers
were left unquantized and why.

## 8. Open questions before launch

1. Does `llm-compressor` (or its CUDA-kernel dependencies) have working aarch64/GB10 wheels?
   **Untested; resolve first.**
2. Does the exporter support mixed INT4/FP8/BF16 output, or does everything non-INT4 default
   to BF16 (the `baicai1145` outcome)? Determines whether the artifact lands at ~100 GB or
   ~165 GB. **Untested.**
3. Do vllm#35922 and vllm#41511 reproduce on the actual 170HX SM80 cards with this specific
   MoE shape (256 experts, group-128)? **Untested; this is the go/no-go gate for calling the
   artifact usable, independent of whether quantization itself succeeds.**
