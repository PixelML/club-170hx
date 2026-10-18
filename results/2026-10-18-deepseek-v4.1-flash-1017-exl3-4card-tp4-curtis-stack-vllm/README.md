# DeepSeek-V4.1-Flash-1017 EXL3 3 bpw on 4× CMP 170HX: TP4 P2P matrix + Curtis White's serving stack on our quant

Notebook: [`notebooks/2026-10-18-deepseek-v4.1-flash-1017-exl3-4card-tp4-curtis-stack-vllm.ipynb`](../../notebooks/2026-10-18-deepseek-v4.1-flash-1017-exl3-4card-tp4-curtis-stack-vllm.ipynb).

## Licence of this folder

| Path | Licence |
|---|---|
| `build/tcmoe1/exl3.py`, `build/tcmoe1/exl3-tcmoe.patch`, `build/v3p1/exl3-1017.patch` | **AGPL-3.0** (modified vllm-exl3; text in `build/tcmoe1/vllm-exl3.LICENSE`, keep `vllm-exl3.NOTICE` and `vllm-exl3.THIRD_PARTY_NOTICES.md`) |
| `build/serve/patch/*.patch`, `build/serve/hostar/*` | Apache-2.0 (vLLM, Morrowmake) |
| sources fetched by `build/tcmoe1/fetch_sources.sh` | Apache-2.0 (Curtis White) and MIT (exllamav3), at the pins in Credits below |
| everything else (scripts, receipts) | Apache-2.0 (this repository's licence) |

## Layout

- `receipts/<arm>/`: decode, prefill, known-answer and greedy-reference JSON, telemetry, launch line, `pins.json` (host details masked). `receipts/fail-*/`: boots that failed, with the error lines.
- `build/v3p1/`: Dockerfile of the P2P-matrix image (public Oct 4 image v3 + `exl3-1017.patch`), `prepare_1017.py` (overlay view), `engram_int4.py`.
- `build/tcmoe1/`: Dockerfile of the Curtis-stack image, our `exl3.py` (coop route, aligned MoE TP, TC hook calls), coop side-module build, `fetch_sources.sh` + `SOURCES.gitblob` (git blob hash of every upstream file that went into the measured image).
- `build/serve/`: launch/run/bench/matrix scripts for both images, 1017 loader patches, host all-reduce files. `build/bench/`: bench tools.

Images: see the notebook pins table (both pinned by full 64-hex digest in `ghcr.io/pixelml/club-170hx`).

## Credits (authored by)

- Curtis White ([@CurtisAccelerate](https://github.com/CurtisAccelerate)): tensor-core EXL3 MoE port for DeepSeek V4.1, row dispatch, shared workspace, EXL3 hook, `gate0.py` — [DeepSeekv41-Flash-EXL3-TC-MoE @ cd83ade14c43a42babad39895d75bdd0e3453d49](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE/tree/cd83ade14c43a42babad39895d75bdd0e3453d49) (Apache-2.0); 1017 serve flags and numbers in [DeepSeekv41-Flash-1017-EXL3-CMP170HX @ f5cb5051f0](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/tree/f5cb5051f0) (no licence file: flags used as facts, no files copied).
- Morrowmake ([@Morrowmake](https://github.com/Morrowmake)): tensor-core EXL3 MoE kernels and host-staged all-reduce, [vllm-cmp170hx @ 18305948acba3f4c4d912cdd56f2202d2ebd94e8](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8) (Apache-2.0). `build/serve/hostar/dsv41_host_shm_all_reduce.py` is their file with the P2P hand-over probe stubbed.
- Victor Cruz ([@vcruz305](https://github.com/vcruz305)): [vllm-exl3](https://github.com/vcruz305/vllm-exl3/tree/2a740a3773601c994072295fde45738ef8cea55e) (AGPL-3.0); coop route [#27](https://github.com/vcruz305/vllm-exl3/pull/27) by @kaka86mm, aligned uneven MoE TP split [#36](https://github.com/vcruz305/vllm-exl3/pull/36) and per-layer rotation [#40](https://github.com/vcruz305/vllm-exl3/pull/40) by @fattchris.
- turboderp ([@turboderp](https://github.com/turboderp)): EXL3 format and `exl3_moe_coop`, [exllamav3 @ d3739fd393337b1ff4d6c2a342b12f0c87a9592f](https://github.com/turboderp-org/exllamav3/tree/d3739fd393337b1ff4d6c2a342b12f0c87a9592f) (MIT).
- 0xSero ([@0xSero](https://github.com/0xSero)): base serving image, [dsv41-flash-offload @ 2e9da5ecdab37c232c2da887e25c33a99da738d3](https://github.com/0xSero/dsv41-flash-offload/tree/2e9da5ecdab37c232c2da887e25c33a99da738d3) (MIT).
- sfxnz ([@sfxnz](https://github.com/sfxnz)): `tools/quantize_experts_exl3.py`, [DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ 3de914682d0880c8e478712a60ada930339c22e4](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/tree/3de914682d0880c8e478712a60ada930339c22e4) (MIT).
- DeepSeek-AI: [DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) (MIT). MiaAI-Lab ([@MiaAI-Lab](https://github.com/MiaAI-Lab)): decode bench prompts.
