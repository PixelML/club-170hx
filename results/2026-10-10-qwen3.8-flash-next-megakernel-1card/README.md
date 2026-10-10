# Qwen3.8-Flash-Next decode megakernel on 1x CMP 170HX — receipts

**Credits.** The kernel design (one persistent cooperative kernel per decode step, phases split by a software grid barrier, dp4a on int8 activations, weights kept in GGUF quant formats) is authored by **L-Forster** ([@L-Forster](https://github.com/L-Forster)) in the [open-jet megakernel](https://github.com/L-Forster/open-jet/tree/93c2b9abee50ea2ea41981c406cfd201989e4d58/megakernel) for Qwen3.8-27B (AGPL-3.0, commit `93c2b9abee50ea2ea41981c406cfd201989e4d58`). The forward pass follows the `qwen4exp` graph authored by **ggml-org** ([@ggml-org](https://github.com/ggml-org)) in [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/aa94f2086147cd4bfbcff818ce998dffce5fcb39/src/models/qwen4exp.cpp) (MIT, commit `aa94f2086147cd4bfbcff818ce998dffce5fcb39`); the IQ3_XXS / IQ4_NL lookup tables in `src/fnx_tables.h` come from its `ggml-common.h`. The GGUF quants are authored by **Unsloth** ([@unslothai](https://github.com/unslothai)), [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6). The model is authored by **Qwen** ([@QwenLM](https://github.com/QwenLM)), [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) (licence `qwen-community-1.0`).

Notebook: [notebooks/2026-10-10-qwen3.8-flash-next-megakernel-1card.ipynb](../../notebooks/2026-10-10-qwen3.8-flash-next-megakernel-1card.ipynb)

**Licence of `src/`:** `src/fnx_mk.cu` is AGPL-3.0-or-later because it follows the open-jet megakernel (AGPL-3.0); `src/fnx_tables.h` is MIT (ggml). The rest of this folder follows the repository licence.

## Files

| Path | What it is |
|---|---|
| `summary.json` | every number in the notebook, computed by `src/summarize.py` from `raw/` |
| `src/fnx_mk.cu`, `src/fnx_tables.h`, `src/Makefile` | the engine (one `.cu` file, ~1,500 lines) |
| `src/run.sh` | the exact launch line, paths as environment variables |
| `src/extract_ple.py` | copies the 26.8 GiB PLE n-gram table out of the GGUF onto a local disk |
| `src/ref_llamacpp.py`, `src/ref_margins.py` | llama.cpp greedy references and top-2 log-prob gaps |
| `src/prompts.tsv`, `src/gpu_sampler.sh` | the 4 prompts; 1 Hz `nvidia-smi` sampler |
| `raw/fnx_bench.jsonl` / `.log` | megakernel: 4 prompts x 3 reps, token ids and timings |
| `raw/ref512_r{0,1,2}/` | llama-server: 3 rounds of the same 4 prompts (r0 = cold) |
| `raw/llama-bench.md` | `llama-bench` pp512 / tg128 on the same card |
| `raw/score512.txt`, `raw/margins.json` | teacher-forced agreement and the llama.cpp top-2 gap at each disagreement |
| `raw/profile.log`, `raw/phases.txt`, `raw/barrier.txt` | per-phase timing, phase micro-benchmarks, grid-barrier cost |
| `raw/*_gpu.csv`, `raw/env.txt` | power / clocks / temperatures during each run; driver, CUDA, kernel |
| `raw/ref/` | first 256-token reference round (superseded by `ref512_r1`, kept for the record) |
