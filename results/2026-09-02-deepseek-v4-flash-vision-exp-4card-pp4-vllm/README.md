# DeepSeek-V4-Flash-Vision-Exp — 4-card concurrency ladder (2026-09-02)

Public-safe evidence bundle for a canonical text-only benchmark of
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (rev `86f746b36186f0e567729a5c06a8c918caba82a9`)
on 4x CMP 170HX (SM80, 64 GiB) with pipeline parallelism 4
(`VLLM_PP_LAYER_PARTITION=11,11,11,10`) and DSpark speculative decoding (k=6),
kv-cache fp8, block 256, max-model-len 16384, max-num-batched-tokens 2048,
max-num-seqs 8. Internal infrastructure details (storage paths, container and
image names, PCI bus ids, pids, IPs, hostnames) are masked with
`<...>` placeholders; sha256 digests are kept unmodified so the artifacts remain
verifiable.

## Protocol

All phases drive the engine's OpenAI-compatible HTTP API on loopback
(`bench_harness.py`), sequentially: readiness poll → gate → prefill → TTFT →
ladder.

- **Gate** — `/v1/models` plus a deterministic greedy check
  ("The capital of France is" → " Paris", 3/3 identical reps). Required to PASS
  before any measurement.
- **Sampling** — greedy (temperature 0) everywhere; ladder requests use
  `ignore_eos` and exactly **400 completion tokens** (all requests finish with
  `finish_reason=length`, `all_exactly_400=true` verified per rep).
- **Ladder** — concurrency levels C1, C2, C4, C8, C16; **1 warmup + 3 measured
  reps** per level; per-request prompts carry a unique tag so there is no
  prefix-cache reuse across reps. Aggregate tok/s = sum of completion tokens /
  wall time of the synchronized level; token counts are taken from the **final
  `usage` object** of each response, never from client-side text counting.
- **Uncached prefill fixture** — a **2,941-token** prompt (tokenizer-verified
  exactly 2941 prompt tokens on 3/3 reps), `max_tokens=1`, unique prefix per
  rep so the prefill is genuinely uncached. Reports wall time and prefill tok/s;
  rep0 is the first long prefill after boot and is reported separately in the
  receipt.
- **Warm streaming TTFT** — same 2,941-token fixture with the prefix already
  cached, streamed (`stream=true`), time-to-first-token over 3 reps.
- **Telemetry** — `nvidia-smi` sampled every 5 s throughout
  (`telemetry-5s.csv`), plus point-in-time snapshots and a post-run
  dmesg Xid/ECC check.

## Headline results (from RESULTS.md)

| level | status | warmup agg tok/s | measured agg tok/s (3 reps) | agg median | per-request median tok/s | per-request min - max | success |
|---|---|---|---|---|---|---|---|
| C1 | PASS | 117.89 | 57.58 / 123.45 / 97.41 | **97.41** | 97.42 | 57.58 - 123.47 | 3/3 |
| C2 | PASS | 98.54 | 103.66 / 159.24 / 96.58 | **103.66** | 57.37 | 48.3 - 106.51 | 6/6 |
| C4 | PASS | 164.7 | 140.32 / 203.17 / 165.53 | **165.53** | 66.92 | 35.08 - 104.07 | 12/12 |
| C8 | PASS | 216.5 | 220.17 / 231.99 / 206.33 | **220.17** | 35.96 | 25.8 - 62.76 | 24/24 |
| C16 | **FAIL** | 238.39 (16/16 ok) | rep0: 0/16 ok, wall 300.1 s | n/a | n/a | n/a | 0/16 |

| metric | value |
|---|---|
| uncached prefill, 2941 prompt tokens, max_tokens 1 | rep0 8.130 s (361.8 tok/s, first long prefill after boot); rep1 1.250 s (2352.2 tok/s); rep2 1.218 s (2414.4 tok/s); median 1.250 s / 2352.2 tok/s |
| warm streaming TTFT, same 2941-token fixture (prefix cached) | 0.389 / 0.394 / 0.418 s; median 0.394 s |

C16 is outside the stable envelope of this recipe: warmup passed 16/16, then
rep0 lost all 16 requests after ~300 s with a device-side assert on rank 3 in
the draft path (`CUDA error: device-side assert triggered` → watchdog
termination → engine dead). Recorded, not retried. See `RESULTS.md` and the
incident lines in `logs/container-tail.log`.

## Files

| file | contents |
|---|---|
| `RESULTS.md` | full run report (sanitized) |
| `gate.json` | deterministic greedy gate receipt |
| `prefill.json` | uncached 2,941-token prefill receipt (3 reps + GPU state) |
| `ttft.json` | warm streaming TTFT receipt (3 reps) |
| `ladder.json` | per-level, per-rep, per-request ladder receipt incl. 2 s GPU samples |
| `summary.json` | machine-readable run summary (snapshot verification, image, boot, phases) |
| `launch-command.sh` | exact `docker run` invocation (names/paths masked) |
| `bench_harness.py` | the benchmark harness (as run, unchanged logic) |
| `run_protocol.sh` | the driver script (readiness poll, phase order, teardown) |
| `telemetry-5s.csv` | 627 five-second power/thermal/memory samples |
| `logs/container-tail.log` | last 120 lines of the serve log + every line matching `Xid\|assert\|EngineDead\|Timeout\|ready\|Application startup` |

Notes: paths beginning with `/library/`, `/models/`, and `/home/<user>/` are
replaced by `<model-storage>/`, `<fast-cache>/`, and `<workdir>/`;
`127.0.0.1` is retained intentionally. Because sanitized placeholders use
angle brackets, `run_protocol.sh` and `launch-command.sh` are records and will
not parse as-is under bash.
