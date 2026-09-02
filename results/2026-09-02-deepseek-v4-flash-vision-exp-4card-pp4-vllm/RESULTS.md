# DeepSeek-V4-Flash-Vision-Exp — canonical 4-card text benchmark (2026-09-02)

## Identity
| item | value |
|---|---|
| model | deepseek-ai/DeepSeek-V4-Flash-Vision-Exp rev `86f746b36186f0e567729a5c06a8c918caba82a9` |
| snapshot | `<model-storage>/models/deepseek-v4-flash-vision-exp/deepseek-ai-86f746b36186f0e567729a5c06a8c918caba82a9` |
| snapshot verification | 48 shards; model_type deepseek_v4; num_nextn_predict_layers 3; repo bytes 167,831,846,872 exact (HF `.cache`/`.prefetch` bookkeeping adds 29,746 B) — PASS (`receipts/snapshot-verification.json`) |
| image | `<image>` = `sha256:90a1419e8ceaad3542153ef4e2a1d94a69b9af03cce7b0a1b267dd1dad55b9d7` |
| image build | `Dockerfile.fullbuild16` on `<workdir>/repos/dsv4-vllm` @ f8ea5bb16 + 8 patched files (`receipts/vllm-source-patches.diff`, `receipts/source-sha256.txt`); TORCH_CUDA_ARCH_LIST=8.0; all layers cache-hit, image created 2026-09-01T05:22:44Z; in-image dspark.py sha a1192b41 == patched source |
| runtime patches bind-mounted | 7 vLLM files from the source tree + `patches/safetensors_torch.py` (adds F8_E8M0 dtype map) |
| launch | `receipts/launch-command.sh` — PP4, VLLM_PP_LAYER_PARTITION=11,11,11,10, dspark k=6, kv fp8, block 256, max-model-len 16384, gpu-mem-util 0.90, max-num-batched-tokens 2048, max-num-seqs 8, port 18098, container <container>, `--gpus device=0,1,2,3` |
| hardware | 4x CMP 170HX (SM80, 64 GiB), power limit 180 W each; model on NFS <model-storage> |

## Boot
Launched 04:42:11Z, ready 05:27:08Z (2515 s = 42 min). Shard stream 18:43 (48/48, ~23 s/shard). Model load: PP0-2 1146 s; PP3 2270 s (drafter). Graph capture: 17 piecewise + 8 full.

## Gate
`/v1/models` -> `dsv4v`. Greedy `The capital of France is` -> ` Paris` on 3/3 reps (identical). PASS.

## Headline table
| level | status | warmup agg tok/s | measured agg tok/s (3 reps) | agg median | per-request median tok/s | per-request min - max | success |
|---|---|---|---|---|---|---|---|
| C1 | PASS | 117.89 | 57.58 / 123.45 / 97.41 | **97.41** | 97.42 | 57.58 - 123.47 | 3/3 |
| C2 | PASS | 98.54 | 103.66 / 159.24 / 96.58 | **103.66** | 57.37 | 48.3 - 106.51 | 6/6 |
| C4 | PASS | 164.7 | 140.32 / 203.17 / 165.53 | **165.53** | 66.92 | 35.08 - 104.07 | 12/12 |
| C8 | PASS | 216.5 | 220.17 / 231.99 / 206.33 | **220.17** | 35.96 | 25.8 - 62.76 | 24/24 |
| C16 | **FAIL** | 238.39 (16/16 ok) | rep0: 0/16 ok, wall 300.1 s | n/a | n/a | n/a | 0/16 |

| metric | value |
|---|---|
| uncached prefill, 2941 prompt tokens, max_tokens 1 | rep0 8.130 s (361.8 tok/s, first long prefill after boot); rep1 1.250 s (2352.2 tok/s); rep2 1.218 s (2414.4 tok/s); median 1.250 s / 2352.2 tok/s; usage.prompt_tokens = 2941 on 3/3 |
| warm streaming TTFT, same 2941-token fixture (prefix cached) | 0.389 / 0.394 / 0.418 s; median 0.394 s |

Aggregate decode = sum of `usage.completion_tokens` (all exactly 400, finish_reason length, ignore_eos) / synchronized wall time of the level. Greedy (temperature 0). Warmup 1 + 3 measured reps per level. Prompts carry a unique tag per request so no prefix-cache reuse across reps.

Comparison with attempt 12 (2026-08-31, unverified provenance, 1 rep per level): C1 101.21, C2 114.68, C4 169.65, C8 133.95. This run (median of 3): C1 97.41, C2 103.66, C4 165.53, C8 220.17. C1/C4 reproduce within the per-prompt DSpark acceptance spread; C8 is higher here.

## C16 failure (recorded, not retried)
- Warmup: 16/16 ok, 238.39 tok/s aggregate.
- rep0: 0/16 ok after 300.1 s. Every request: `HTTP 500: {"error":{"message":"EngineCore encountered an issue. See stack trace (above) for the root cause.","type":"InternalServerError","param":null,"code":500}}`
- Container log (`logs/container.log` line 606): `[rank3] ProcessGroupNCCL.cpp:2174 Process group watchdog thread terminated with exception: CUDA error: device-side assert triggered` (cudaErrorAssert) -> `terminate called after throwing an instance of 'c10::DistBackendError'`; EngineCore: `TimeoutError: RPC call to sample_tokens timed out.` -> `EngineDeadError`.
- dmesg: `NVRM: Xid (<pci>): 43, pid=<pid>, name=python3.12, channel 0x00000002` — one software-caused Xid 43, no ECC events. GPU bus map: 0, <pci> 1, <pci> 2, <pci> 3, <pci> 
- Same failure class as attempt 12 (device-side assert on rank 3 in the draft path at C16 > max-num-seqs 8). C16 is outside the stable envelope of this recipe.

## Telemetry
| point | power W (GPU0..3) | temp C | memory MiB |
|---|---|---|---|
| before launch | 34 / 38 / 34 / 36 | 41 / 40 / 40 / 39 | 0 / 0 / 0 / 0 |
| loaded, idle | 41 / 45 / 41 / 42 | 44 / 44 / 44 / 41 | 50606 / 49700 / 51196 / 59998 |
| peak during ladder (2 s samples) | 194 / 241 / 211 / 223 (instantaneous samples above the 180 W averaged cap) | 62 / 60 / 61 / 60 | — |
| after bench (engine dead) | 61 / 78 / 41 / 42 | 49 / 49 / 43 / 42 | 51308 / 50276 / 51836 / 60576 |
| after stop+rm | — | 46 / 46 / 42 / 41 | 0 / 0 / 0 / 0 |
Max temperature 63 C (5 s sampler, `telemetry-5s.csv`, 627 samples). No thermal or root-disk stop conditions were hit (root 54 GB free throughout; docker root is <fast-cache>).

## Files
- `gate.json`, `prefill.json`, `ttft.json`, `ladder.json` (per-request tok/s, walls, errors, 2 s GPU samples per rep)
- `receipts/` snapshot-verification.json, launch-command.sh, launch-started-utc.txt, ready-utc.txt, vllm-source-head.txt, vllm-source-patches.diff, source-sha256.txt, Dockerfile.fullbuild16, safetensors_torch.py, nvidia-smi-{before,loaded,after-bench,final}.csv, dmesg-xid-ecc.txt, gpu-bus-ids.csv, summary.json
- `logs/` docker-build.log, container.log (full serve log incl. C16 traceback), protocol.log
- `bench_harness.py`, `run_protocol.sh` (the exact harness and driver)
