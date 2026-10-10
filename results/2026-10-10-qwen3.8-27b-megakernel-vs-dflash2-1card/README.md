# Qwen3.8-27B megakernel vs vLLM DFlash2 — 1x CMP 170HX (SM80), 180 W

Status: measured
Date: 2026-10-10

Megakernel authored by **Louis Forster** ([@L-Forster](https://github.com/L-Forster)): [L-Forster/open-jet `megakernel/`](https://github.com/L-Forster/open-jet/tree/93c2b9abee50ea2ea41981c406cfd201989e4d58/megakernel) @ `93c2b9abee50ea2ea41981c406cfd201989e4d58` (AGPL-3.0; no code copied here). vLLM + DFlash2 recipe authored by **syv.ai** ([@syv-ai](https://github.com/syv-ai)); W4A16 AutoRound by **David Birks** ([@dbirks](https://github.com/dbirks)); GGUF by **Unsloth** ([@unslothai](https://github.com/unslothai)).

Notebook: [`notebooks/2026-10-10-qwen3.8-27b-megakernel-vs-dflash2-1card.ipynb`](../../notebooks/2026-10-10-qwen3.8-27b-megakernel-vs-dflash2-1card.ipynb)

**Question.** Does the open-jet Qwen3.8-27B megakernel (one persistent cooperative kernel per speculative cycle, MTP drafts) beat the club's vLLM + DFlash2 recipe on one CMP 170HX?

**Answer (measured): no.** The test used the same card, the same 180 W cap, the same day and the same client. vLLM DFlash2 k=7 decodes 1.6-2.5x faster on every prompt (greedy mean of 5 prompts 181.8 vs 94.1 tok/s; thinking mean 156.3 vs 75.6). It also prefills 25% faster (1,915 vs 1,528 tok/s at about 3.2K tokens). The megakernel builds for SM80 with zero source changes and needs seconds to start instead of 427 s.

## Results (tok/s, mean of 3 greedy reps / 2 thinking seeds)

| Prompt | mk no-spec | mk d=3 | mk d=4 | vLLM DFlash2 k=7 |
|---|---:|---:|---:|---:|
| code_prime | 51.8 | 123.1 | 128.5 | **317.5** |
| hashmap | 51.0 | 106.9 | 105.9 | **193.9** |
| flask_fastapi | 50.6 | 102.0 | 99.4 | **165.8** |
| story_robot (club prompt) | 50.4 | 68.3 | 68.2 | **120.3** |
| story_lighthouse | 50.5 | 69.3 | 68.6 | **111.2** |
| thinking, 4 prompts (mean) | untested | 77.1 | 75.6 | **156.3** |
| prefill ~3.2K tok | 1,532 | 1,531 | 1,528 | **1,915** |
| prefill 9.5K / 19K tok | 1,489 / 1,400 | 1,489 / 1,400 | 1,490 / 1,400 | Xid 31 crash / untested |

Club suite, vLLM arm, unchanged (`recipes/qwen3.8-27b-dflash2/live_benchmark.py`): decode256 136.96 (record 136.38), decode900 117.92 (record 122.00), prefill 6,603 tok 1,929.8 (record 1,946).

Telemetry at 180 W: peak core 76-77 °C, peak memory 77-80 °C, busy mean 172-180 W.

## Pins

- Container (both arms): `docker.io/nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66` (12.8.1-devel-ubuntu24.04)
- Megakernel: `L-Forster/open-jet@93c2b9abee50ea2ea41981c406cfd201989e4d58`, `make ARCH=sm_80`, nvcc 12.8.93; tokenizer library `ggml-org/llama.cpp` b10246 = `39eab74a05d3e68ac822b6dc6cd78c90cb985c19`
- Megakernel model: `unsloth/Qwen3.8-27B-GGUF@4121cb19390a7984a0e6dc0f46bea9177b846f15` `Qwen3.8-27B-Q4_K_M.gguf`, sha256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` (Apache-2.0; deleted from `main` at `e1d8a26782d4ce57b3c612e29a58cfb8f491c2fc`, so pin the revision)
- vLLM: `syv-ai/qwen38-27b-rtx3090@69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c`, vLLM 0.27.1, `recipes/qwen3.8-27b-dflash2/requirements.lock`; models per `recipes/qwen3.8-27b-dflash2/recipe.json`
- Serve: `mk_server.py -c 65536 -d {4,3,0}`; vLLM `SPEC=dflash2 CTX=fast MAX_SEQS=1 DFLASH_TOKENS=7 GPU_UTIL=0.90` (BF16 KV, 65,536 ctx)
- Host: guest driver 580.178.04 (stock), host-side unlock `amoghmunikote/cmpunlocker@88e39ce`, HBM 1,728 MHz, 74 SMs, PCIe Gen2 x16; tuning `nvidia-smi -pl 180`

## Files

- `receipts/180w/` — `mk.jsonl`, `vllm.jsonl` (one line per request, usage-counted), `vllm-club-suite.jsonl`, 1 Hz telemetry, bench logs, pip freezes, source SHAs
- `receipts/250w-thermal-stop/` — negative result: megakernel at the 250 W VBIOS default reached the 80 °C stop; greedy receipts that completed before the stop
- `harness/` — `build.sh`, `chat_bench.py` (one client for both engines), `guard.sh` (80 °C / 85 °C fail-closed), `run_mk.sh`, `run_vllm.sh`, `install_vllm.sh`
- `summarize.py` → `summary.csv` and `assets/charts/2026-10-10-qwen3.8-27b-megakernel-vs-dflash2-1card.png`

## Negative results kept

1. 250 W default: megakernel busy mean 240-249 W, 80 °C stop within about a minute. The 180 W cap costs it 6-9% decode.
2. vLLM: Xid 31 (`FAULT_PDE`, virtual read) and an illegal memory access on a 9,472-token chat prompt; engine died; cause not identified; all GPU load stopped.
3. Recipe install: `nvidia-cuda-nvcc` 13.4.92 (not pinned in the lock) moved to `nvidia/cu13/`, so the recipe's `nvidia.cuda_nvcc` step fails. The manual fix is in the notebook appendix.

## Limitations

One card, one run per configuration. Different 4-bit checkpoints per arm (GGUF Q4_K_M vs W4A16 AutoRound). Concurrency not tested (the megakernel serves one sequence at a time).
