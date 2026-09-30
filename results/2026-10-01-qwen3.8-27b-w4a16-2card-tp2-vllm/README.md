# Qwen3.8-27B W4A16 — TP1 vs TP2 (same / cross PLX switch) and TP2 + MTP, no P2P

Status: measured
Date: 2026-10-01

Notebook: [`notebooks/2026-10-01-qwen3.8-27b-w4a16-2card-tp2-vllm.ipynb`](../../notebooks/2026-10-01-qwen3.8-27b-w4a16-2card-tp2-vllm.ipynb)

## Hardware

- Cards: CMP 170HX (GA100, SM80), memory-unlocked to 64 GB; 1 card (TP1) or 2 cards (TP2); anonymous labels GPU0–GPU2
- Topology / PCIe links: Gen2 x16; GPU0+GPU1 behind one PLX PEX 8747 switch, GPU2 behind a second (same CPU socket); no P2P (no BAR1-P2P driver patches), NCCL SHM transport
- Power limit and measured peak draw: 180 W; 182.8–187.4 W instantaneous peak during the suite
- Cooling and peak core temperatures: forced airflow, chassis fans at a fixed high duty; 67–74 °C during the suite; 80 °C stop enforced (one earlier GPU0 run stopped — see below)

## Software

- OS / kernel: Proxmox VE 9 (Debian 13), kernel 7.0.2, bare metal
- NVIDIA driver: open kernel modules 615.71.09 with host-side cmpunlocker unlock (upstream 88e39ce) + BAR1-resize serialization patch
- Runtime: `ghcr.io/pixelml/club-170hx@sha256:62f612b49614523e6a46e1493d35d3efd1f363917129d38cc923a31053693bfb` (tag `vllm-glm53-sm80-pp-20260905`), vLLM `v0.1.dev20051+g487ecf187`
- Model: `dbirks/Qwen3.8-27B-W4A16-AutoRound` @ `1f05c441c4e64ae0549de44fa9ea5a6d43610314` (compressed-tensors W4A16), BF16 KV
- Serve flags: `--max-model-len 65536 --gpu-memory-utilization 0.90 --max-num-seqs 32 --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}'` (+ `--tensor-parallel-size 2`; MTP run adds `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`)

## Method

- Suite: `recipes/qwen3.8-27b-dflash2/live_benchmark.py` unchanged — `decode256`, `decode900` (11-token prompt, greedy, `ignore_eos`), `prefill_long` (6,603 prompt tokens, 8 output); 1 warm-up + 3 samples, mean; tokens from the final streamed usage object.
- Concurrency: `tools/conc_sweep.py`, c = 1/4/8/16, 256 output tokens, T=1.0, top_p 0.95, `ignore_eos`, 2 rounds.
- Guard: `tools/temp_guard.sh` stops the container at 80 °C core or on a new Xid.

## Results (measured)

| Layout | Decode 256 | Decode 900 | Prefill 6,603 | TTFT 6,603 | Agg c=4 | Agg c=16 | KV cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP1 (GPU2) | 54.0 | 53.4 | 1,914 | 3.45 s | 188 | 546 | 588k tok |
| TP2 same switch (GPU0+1) | 72.0 | 71.0 | 1,695 | 3.90 s | 221 | 530 | 1.51M tok |
| TP2 cross switch (GPU1+2) | 71.9 | 71.8 | 1,736 | 3.80 s | 223 | 535 | 1.47M tok |
| TP2 cross + MTP k=3 | 103.5 | 97.6 | 1,658 | 3.98 s | 117 | 385 | not captured |

All throughput in tok/s. vLLM selected `PYNCCL` as the only TP all-reduce backend (custom all-reduce requires P2P).

## Negative result kept

`runs/thermal-stop-gpu0-tp1/`: the first TP1 control on GPU0 hit the 80 °C stop three times (250 W; 180 W with quiet fans; 180 W with full fans at c=32). Its decode (55.2 / 53.6 tok/s) matches the GPU2 control; its prefill (~28k tok/s) is invalid because prefix caching was on.

## Limitations

- One run per layout. MTP acceptance not captured. Not comparable to the DFlash2 recipe (different runtime, drafter and checkpoint preparation).
