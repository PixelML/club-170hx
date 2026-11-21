# Inkling-2-Mini on 1x CMP 170HX (SM80), 140 W — vLLM with an SM80 patch and DSpark, vs llama.cpp

Status: measured
Date: 2026-11-21

Notebook: [`notebooks/2026-11-21-inkling-2-mini-1card-vllm-sm80-dspark-vs-llamacpp.ipynb`](../../notebooks/2026-11-21-inkling-2-mini-1card-vllm-sm80-dspark-vs-llamacpp.ipynb)

**Question.** Does Thinking Machines' Inkling-2-Mini (26.4B looped MoE, 3.4B active, NVFP4 + MXFP8 checkpoint) run on one CMP 170HX, and what is the fastest way to serve it?

**Answer (measured): yes, and vLLM with the bundled DSpark draft head is the fast path.** The `vllm/vllm-openai:inkling-2-mini` image is built for Blackwell and Hopper only. Six Python patches (no rebuild, `patches/`) make it run on SM80. One user, temperature 0: 166.9 tok/s code, 189.1 tok/s structured JSON, 94.0 tok/s prose. Aggregate 510.9 tok/s at 16 users. Prefill 4,700 tok/s at 23,806 tokens. GSM8K first 100: 99/100. llama.cpp b12600 runs the official IQ4_NL GGUF with no patch at 42.7 tok/s and 315–627 tok/s prefill (98/100 GSM8K).

## Results (tok/s, median of 3, 768 output tokens, temperature 0)

| Engine (card copy bandwidth) | Code | Structured | Prose | c=16 aggregate (T=1) | Prefill 23.8k |
|---|---:|---:|---:|---:|---:|
| llama.cpp b12600 IQ4_NL (1,603 GB/s) | 42.7 | 42.8 | 42.7 | — (c=4: 99.8) | 413 |
| vLLM + sm80 patch, no speculation (1,602 GB/s) | 79.7 | 78.8 | 79.3 | 376.5 | 4,599 |
| vLLM + sm80 patch, no speculation (1,313 GB/s) | 76.3 | 76.3 | 76.3 | 380.7 | 4,781 |
| vLLM + DSpark k=7, greedy draft (1,603 GB/s) | 154.5 | 175.1 | 87.6 | 480.5 | — |
| **vLLM + DSpark k=7, probabilistic draft (1,597 GB/s)** | **166.9** | **189.1** | **94.0** | **510.9** | 4,700 |

DSpark acceptance: 1.68 accepted tokens per draft (probabilistic) vs 1.44 (greedy), pooled over both temperatures. k is fixed at 7 by the checkpoint.

**Caution:** the two DSpark rows ran on different cards. The probabilistic card has a 170tune SM offset (+200 at 1,410 MHz) and runs about 5% higher SM clock at the 140 W cap. On one card (no offset) the two methods are about equal at temperature 0, and probabilistic is 6–17% faster at temperature 1 (`receipts/vllm-dspark7-vm504-card86.jsonl`).

## Files

- `harness/` — `bench.py` (chat bench), `gsm8k.py`, `guard.sh` (80 °C / 85 °C / Xid stop), `run-vllm.sh`, `run-llama.sh`, `build-llama-b12600.sh`, `hbm_bw.py`, `oprobe_sitecustomize.py` (finds ops that leave a CUDA error), `digest_check.sh`, `summarize_csv.py`, `wait-bench.sh`.
- `patches/` — `make-sm80-patch.sh` (all six steps), `sm80_moe_align.py`, `patch_dspark.py`, `inkling2-sm80.diff` (full diff against the image's `vllm/models/inkling2`).
- `receipts/` — one JSON line per request (`*.jsonl`), bench logs, server logs, DSpark `/metrics`, GSM8K answers, `hbm-copy-bw.txt`, `telemetry/` (2 s guard samples), `port-walls/` (logs of each failed SM80 boot).
- `summarize.py`, `make_chart.py`, `make_notebook.py` — rebuild the chart and the notebook from the receipts.

## Credits

- **Thinking Machines Lab** ([@thinking-machines-lab](https://github.com/thinking-machines-lab)) authored [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its DSpark draft head (Apache-2.0), revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`.
- The **vLLM team** ([@vllm-project](https://github.com/vllm-project)) authored vLLM and the image `vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e` (Apache-2.0); `patches/inkling2-sm80.diff` edits its `vllm/models/inkling2` package.
- **ggml-org** ([@ggml-org](https://github.com/ggml-org)) authored [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and the [Inkling-2-Mini GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) @ `db61c5b492a6a034d986328d28274af0a439aa6c`; **Pedro Cuenca** ([@pcuenca](https://github.com/pcuenca)) authored the merged Inkling 2 Mini support (PR #32694, b12600 `29b52d398b294509746ec0a3f28948a3107c0213`).
- **OpenAI** authored [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (MIT).
