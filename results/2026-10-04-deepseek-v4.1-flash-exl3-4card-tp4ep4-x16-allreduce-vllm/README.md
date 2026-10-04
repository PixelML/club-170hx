# DeepSeek-V4.1-Flash: x16 rerun and all-reduce choice, receipts

Notebook: [`notebooks/2026-10-04-deepseek-v4.1-flash-exl3-4card-tp4ep4-x16-allreduce-vllm.ipynb`](../../notebooks/2026-10-04-deepseek-v4.1-flash-exl3-4card-tp4ep4-x16-allreduce-vllm.ipynb)

Image (unchanged from the 2026-10-03 notebook): `ghcr.io/pixelml/club-170hx@sha256:b8b7a1699bf272855a9b8c8b316f796b3ef917696302baf2a62c856707d69932`. Build recipe, patches, model and Engram pins: [`results/2026-10-03-deepseek-v4.1-flash-exl3-4card-tp4ep4-dspark-vllm/`](../2026-10-03-deepseek-v4.1-flash-exl3-4card-tp4ep4-dspark-vllm/).

| Path | What |
|---|---|
| `chain.sh` | the three arms (restart, wait for `/health`, bench, known answers, 80 °C watchdog) |
| `receipts/int4-hybrid-x16/` | image defaults: custom all-reduce below 128 KiB, NCCL above |
| `receipts/int4-nccl-x16/` | `CUSTOM_AR=0 VLLM_FORCE_CUSTOM_AR_PCIE=0` |
| `receipts/int4-custom-x16/` | `VLLM_CUSTOM_AR_MAX_BYTES=0` (custom for every size) |

Each receipt folder: `prefill.json`, `decode-*-c{1,4}.json`, `correctness.json`, `telemetry.csv`, `nvidia-smi-start.csv` (all four cards Gen2 x16, 140 W), `launch-lines.txt` (all-reduce backends, KV pool).

Measured: NCCL only decodes one user at 119.9 / 87.4 / 62.5 tok/s (structured / code / prose), +21–25% against the image default at the same draft acceptance; four users and prefill unchanged; 5 / 5 known answers in all three.
