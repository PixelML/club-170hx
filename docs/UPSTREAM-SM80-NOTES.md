# SM80 compatibility notes for upstream

This page collects five SM80 (compute capability 8.0, NVIDIA GA100) blockers
found while running vLLM on CMP 170HX cards, each with the fallback that
worked. It is written to paste into the vLLM PR thread and the
[allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80)
fork. Full detail and measured numbers are in
[LESSONS.md](LESSONS.md#a-kernel-and-format-compatibility-on-sm80). No
private hosts, IPs, or account names appear below.

## 1. CuteDSL kernels fail to compile for `sm_80`

**Symptom.** The CuteDSL `fused_indexer_q` RoPE and FP8-quant kernel fails at
the NVVM backend compile step when the target is `sm_80`.

**Fallback used.** A pure-torch implementation reproducing the same
numerics: GPT-J interleaved RoPE with power-of-two per-token scales. Checked
against the CuteDSL output on a supported card before use.

**File.** See the indexer module referenced in `docs/LESSONS.md`, row
"CuteDSL kernels".

## 2. tilelang FP8 x FP4 GEMM asserts on SM80

**Symptom.** A device-side assert fires because the kernel needs SM89 FP8
matrix-multiply-accumulate instructions, which SM80 does not have.

**Fallback used.** Dequantize FP4 weights to BF16, then run a standard BF16
GEMM. For the vision tower, standard BF16 scaled dot-product attention
replaces the fused FP8 x FP4 path entirely.

## 3. `fast_hadamard_transform` CUDA extension has no SM80 build

**Symptom.** The Vision-Exp indexer path calls a compiled Hadamard-transform
extension that is not usable on SM80.

**Fallback used.** A pure-torch Sylvester Hadamard transform, verified
exact against the identity `H @ H.T = n * I` before use in the indexer path.

## 4. NVFP4 / MXFP4 native tensor-core execution does not exist on SM80

**Symptom.** NVFP4 checkpoints (for example GLM-5.3-Flash NVFP4) target
SM12x FlashInfer backends only; native FP4 matrix-multiply hardware is
Blackwell-only and absent on SM80.

**Fallback used.** None available for native execution. The working paths
on SM80 are: (a) a checkpoint that stores MXFP4 expert weights but
dequantizes them before a BF16 GEMM (works, see DeepSeek-V4-Flash-0731), or
(b) a different quant format entirely, such as AWQ/GPTQ INT4 through Marlin,
or a llama.cpp GGUF quant on a different runtime.

## 5. MTP speculative decoding needs two separate patch sets under pipeline parallelism

**Symptom.** Stock vLLM blocks multi-token-prediction (MTP) speculative
decoding under pipeline parallelism in three places. A newer
`GPUModelRunnerV2` code path enforces the same restriction a second time,
independently of the first three.

**Fallback used.** Apply both patch sets from
[allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80).
Verify each patch with an import check plus an undefined-name check
(pyflakes), in addition to `py_compile`, since a patch that only compiles
can still reference a name that does not exist at import time.

## Shared root cause, worth flagging upstream

Several of the fallbacks above exist only because the precompiled vLLM
wheel ships without the SM80-specific custom op
(`VLLM_USE_PRECOMPILED=1` silently omits it), and the failure surfaces late,
at CUDA-graph capture, not at import time. A full source build is required
whenever a patch touches `csrc/`. An architecture-aware predicate at import
time (checking compute capability before choosing a kernel, the same
pattern already used for `is_deep_gemm_supported()`) would surface these
gaps earlier and make them easier to patch around.
