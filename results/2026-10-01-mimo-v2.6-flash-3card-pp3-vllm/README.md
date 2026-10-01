# MiMo-V2.6-Flash-RL — PP3 and PP3 + MTP k=2 re-run at Gen2 x16 behind PLX switches

Status: measured
Date: 2026-10-01

Notebook: [`notebooks/2026-10-01-mimo-v2.6-flash-3card-pp3-vllm.ipynb`](../../notebooks/2026-10-01-mimo-v2.6-flash-3card-pp3-vllm.ipynb) · prior run: [`2026-09-22` receipts/x16](../2026-09-22-mimo-v2.6-flash-4card-pp4-vllm/receipts/x16/README.md)

## Hardware / software

See `environment.json`: 3 × CMP 170HX at 180 W, Gen2 x16 behind two PLX PEX 8747 switches, bare metal, no P2P; same image (pinned by digest), patches, launch arguments and tools as the 2026-09-24 x16 re-run.

## Results (measured)

| Config | P1 greedy c=1 | P2 T=1.0 c=1 | Agg c=32 | Prefill ~20.3k (uncached) | Gate |
|---|---:|---:|---:|---:|---|
| PP3 | 74.6 (prior 75.5) | 73.8 (74.9) | 450 (455) | 4,107 tok/s, TTFT 4.94 s | 4/4 |
| PP3 + MTP k=2 | 113.8 (117.7) | 86.9 (90.1) | 518 (561) | 4,082 tok/s, TTFT 4.97 s | 4/4 |

All in tok/s. MTP mean acceptance length 2.20–2.24. No thermal stop (fans at full duty, 80 °C / Xid guard).

## Files

`runs/<config>/` — `gate.json`, `p1.json`, `p2.json`, `conc.json`, `prefill.jsonl`, `boot.json`, `launch-args.json`, `launch-env.json`, `telemetry.csv`, `spec-stats.txt`; `tools/prefill_probe.py`, `tools/temp_guard.sh`, `tools/build_notebook.py`.
