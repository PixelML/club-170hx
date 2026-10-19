# Third-party notices

Current vllm-exl3 releases are distributed as a whole under **AGPL-3.0-only**. Earlier vllm-exl3 releases were Apache-2.0; that historical license text is preserved in `LICENSE.APACHE-2.0`.

This project contains and derives from third-party MIT- and Apache-licensed work. Those original copyright, permission, and attribution notices remain in force for the material they cover and are reproduced below. The project-level AGPL license does not erase or replace those notices.

---

## Mia's AI Lab, GLM-5.3-Flash-EXL3-2x-DGX-Sparks

Authors: Mia's AI Lab ([@MiaAI-Lab](https://github.com/MiaAI-Lab)) and
[@plotarmordev](https://github.com/plotarmordev).

https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks

**Files copied from that project into this one:**

- `csrc/exl3_fat_gemm.cu` and `csrc/exl3_fat_gemm.cuh` are taken from `overlay/exl3_fat_gemm.cu`
  and `overlay/exl3_fat_gemm.cuh` at commit `4b8d3c7`, "perf(exl3): accelerate fat-expert prefill".
  The only modification is three `#include` paths adjusted for this project's layout. The kernel is
  otherwise unchanged.

**Files in this project substantially derived from that project:**

- `src/vllm_exl3/exl3.py` derives from `overlay/exl3.py`, first published there on 2026-08-27,
  which precedes this project's first commit. Substantial portions of the routed-expert EXL3/MCG
  path, including its pointer-table construction, expert-map pinning and diagnostic strings,
  originate there.

For later optimization work, `docs/provenance.md` distinguishes historical MIT source material from newer upstream work and records whether local changes are copied/derived, adapted designs, or independent implementations.

```
MIT License

Copyright (c) 2026 Mia's AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Turboderp, ExLlamaV3

Author: [@turboderp](https://github.com/turboderp).

https://github.com/turboderp-org/exllamav3

The CUDA sources in `csrc/` build against ExLlamaV3's extension headers, including
`quant/exl3_gemv_kernel.cuh`, `exl3_dq.cuh`, `hadamard_inner.cuh`, `util.h`, `util.cuh` and
`ptx.cuh`, and the GEMV and batched kernels reuse its kernel body. The EXL3 trellis format, the MCG
codebook and the quantization method itself are ExLlamaV3's work.

### ExLlamaV3 n-gram embedding codec

`ngram_dequant_rows_torch` and `ngram_mul1_codebook` in `src/vllm_exl3/exl3.py` reimplement, as a
pure-Python/CPU fallback, the row layout and mul1 codebook arithmetic of ExLlamaV3's
`exllamav3/modules/quant/exl3_lib/ngram_codec.py` and its `ngram_dequant` CUDA kernel. The serving
path itself calls ExLlamaV3's own `ngram_dequant` kernel through `exllamav3_ext`; the torch
reimplementation exists for correctness cross-checking and for environments without the compiled
kernel.

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## vLLM

Author: the vLLM project ([vllm-project/vllm](https://github.com/vllm-project/vllm)).

`_exl3_routed_experts_loader` in `src/vllm_exl3/exl3.py` mirrors the checkpoint-name resolution of
vLLM's `RoutedExperts.load_weights`, adapted to load one EXL3 tensor per expert instead of taking
vLLM's fused (3-D) branch. This plugin's custom ops are also registered through vLLM's
`direct_register_custom_op`. That source lineage is Apache-2.0.
