# DeepSeek-V4-Flash-Vision: x16 rerun, NCCL P2P level, PP4 after the fix, receipts

Notebook: [`notebooks/2026-10-04-deepseek-v4-flash-vision-exp-4card-x16-nccl-sys-pp4-vllm.ipynb`](../../notebooks/2026-10-04-deepseek-v4-flash-vision-exp-4card-x16-nccl-sys-pp4-vllm.ipynb)

Image (unchanged from the autoresearch notebook): `ghcr.io/pixelml/club-170hx@sha256:97ecb29396abc87a9a4f99e18e561f97dc610ca1d79788819d374f5fff1205ba`. Model `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` @ `86f746b36186f0e567729a5c06a8c918caba82a9`. Build recipe and bench: [`results/2026-10-04-deepseek-v4-flash-vision-exp-4card-tp4-autoresearch-vllm/`](../2026-10-04-deepseek-v4-flash-vision-exp-4card-tp4-autoresearch-vllm/).

| Path | What |
|---|---|
| `chain.sh` | the three arms (restart, `protocol.sh`, 80 °C watchdog) |
| `receipts/x16-tp4-sys/` | TP4, `NCCL_P2P_LEVEL=SYS` (the published command) |
| `receipts/x16-tp4-nosys/` | TP4, no `NCCL_P2P_LEVEL` |
| `receipts/x16-pp4-nosys/` | PP4, `VLLM_PP_LAYER_PARTITION=11,11,11,10`, no `NCCL_P2P_LEVEL` |

Each folder: `gate.json`, `correctness.json`, `probe.json`, `vision-check.json`, `prefill.json`, `ttft.json`, `ladder.json`, `ladder_image.json`, `telemetry.csv`, `nvidia-smi-loaded.csv` (all four cards Gen2 x16), `serve-config.json`, `dmesg-xid.txt`, `protocol-log.txt`.

Measured: TP4 prefill 2,008–2,009 tok/s at x16 (1,292 at x8), SYS on = SYS off; 5 / 5 known answers and 5 / 5 short prompts on topic in every run; PP4 crashes at four concurrent image requests (Xid 43).
