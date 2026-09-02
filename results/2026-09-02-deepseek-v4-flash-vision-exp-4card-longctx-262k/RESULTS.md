# DeepSeek-V4-Flash-Vision-Exp — long-context bench at max-model-len 262,144 (2026-09-02)

## Identity and pins

| item | value |
|---|---|
| model | `dsv4v` (DeepSeek-V4-Flash-Vision-Exp, FP8 e4m3, 48 shards) |
| container | resident vision server, relaunched at `--max-model-len 262144` |
| launch script | `launch-vision-server.sh` (path3 recipe) |
| hardware | 4x CMP 170HX (SM80, 64 GiB each), pipeline parallel 4, partition `11,11,11,10` |
| speculative decoding | DSpark k=6 |
| KV cache | fp8, block size 256 |
| other flags | `--max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 --max-num-seqs 8` |
| harness | `longctx_bench.py` / `run_longctx.py`, extending the existing `bench_harness.py` conventions (greedy, temperature 0, final `usage` object for token counts, unique-prefix fixtures per rep, 1 warmup + 3 reps) |

## Boot and KV pool

Server answered `/v1/models` 200 within the bounded readiness loop. From `docker logs`:

```
GPU KV cache size: 1,621,821 tokens
Maximum concurrency for 262,144 tokens per request: 6.19x
```

Gate check: greedy `The capital of France is` -> ` Paris`. PASS.

## Step 1 — prefill ladder (max_tokens=1, 3 reps, unique prefix per rep)

| prompt tokens | status | median wall time (s) | median prefill tok/s |
|---:|---|---:|---:|
| 2,941 | PASS | 1.24 | 2,397 |
| 16,000 | PASS | 3.43 | 4,665 |
| 32,000 | PASS | 6.18 | 5,182 |
| 65,000 | PASS | 12.36 | 5,261 |
| 131,000 | **FAIL** | — | — |
| 200,000 | not reached | — | — |
| 250,000 | not reached | — | — |

Wall time here is also the effective TTFT proxy: each request asked for exactly 1 completion token, so wall time measures time-to-first-and-only-token.

**Largest verified passing length: 65,000 prompt tokens** (25% of the configured 262,144 max-model-len).

### Failure at 131,000 tokens (verbatim)

While the harness was building the 131,000-token fixture (a sequence of `/tokenize` calls against the running server), the connection was reset:

```
level 131000: FIXTURE BUILD FAILED: [Errno 104] Connection reset by peer
```

The container log shows the root cause, on the PP3 rank (the DSpark drafter shard), inside the DFlash/DSpark speculator's input-preparation Triton kernel:

```
(Worker_PP3 pid=517) ERROR ... File "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py", line 674, in prepare_dflash_inputs
(Worker_PP3 pid=517) ERROR ...     _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](...)
(Worker_PP3 pid=517) ERROR ... RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
[rank3]:[W902 17:07:37...] CachingHostAllocator.cpp:26] Warning: Exception in pinned allocator free(), rethrowing (function free)
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: an illegal memory access was encountered
```

The fault fired inside the speculative-decode draft-token-preparation kernel, not inside prefill or attention proper, while a request was mid-flight on the engine (the log shows a running request and its spec-decode acceptance stats immediately before the fault). This killed the PP3 worker, which cascaded into an `EngineDeadError` on the API server and a full engine-core shutdown. The container exited (`Exited (0)`) on its own; it was not stopped by the harness or the operator. Full excerpt: `receipts/crash-excerpt-clean.txt`.

**Classification: stability failure (engine crash), not thermal, not a driver Xid, not an OOM or HTTP-level failure.** GPU telemetry across the whole run shows a peak temperature of 51 °C (well under the 80 °C stop threshold), no NVIDIA Xid lines in `dmesg`, and no ECC error counters set. Card 3 (the drafter shard) was the busiest rank throughout the ladder (99% utilization, up to 173 W instantaneous during earlier levels), consistent with it being the one that hit the illegal-memory-access fault. This looks like the same class of EngineCore instability already tracked on #79 (crash under load on the vision build), now reproduced specifically as a long-context / large-fixture-construction trigger rather than a concurrency trigger.

Per the operating authorization for this run, **the server was not restarted** after the crash. All downstream phases that depend on prompt lengths above 65,000 tokens, or on the server being alive after 17:08 UTC, could not be executed.

## Step 2 — needle-in-haystack

Planned at 32k, 131k, and the largest passing length. The needle phase started immediately after the prefill ladder and every request failed with the same `Connection reset by peer` because the server had already crashed:

```json
"32000": {"0.25": "build failed: Connection reset by peer", "0.5": "...", "0.75": "..."}
"65000": {"0.25": "build failed: Connection reset by peer", "0.5": "...", "0.75": "..."}
```

**Result: untested.** No needle depth at any length produced a real request/response pair. This is a negative result caused entirely by the step-1 crash, not a needle-retrieval failure — no valid completion was ever returned to grade.

## Step 3 — decode with long context

Also untested for the same reason: `decode_longctx.json` shows an empty `levels` list — the first fixture-build call for the 65,000-token C1 case hit the same dead server before any decode request was issued.

## Step 4 — vision at 131k context

Not attempted. The driver script gates this step on the prefill ladder reaching 131,000 tokens, which failed. Per protocol step 4's dependency on a working long-context path, this is **untested**, not failed — the vision encoder itself was never exercised at long context in this run.

## Step 5 — comparison with the 16k baseline

| metric | 16k baseline (2026-09-02, max-model-len 16384) | 262k run (this bench) | note |
|---|---:|---:|---|
| prefill tok/s at 2,941 tokens | 2,352 | 2,397 | within run-to-run noise; longer max-model-len did not regress short-prompt prefill |
| TTFT at 2,941 tokens | 0.386 s (streamed, prefix cached) | 1.24 s (uncached, max_tokens=1, unique prefix) | not directly comparable — the baseline number is a *cached* streaming TTFT; this run's number is uncached wall time. Both are reported as measured, with the method difference stated so nobody merges them by mistake. |
| C1 text decode | 119 tok/s median (5 reps, 16k config) | not measured | decode phase never ran (see step 3) |
| largest context actually exercised | 16,384 (max-model-len ceiling) | 65,000 (prefill only; ladder failed before higher lengths) | the 262k boot proves the KV pool and engine start correctly at 262,144, but only 65,000 of that was verified to serve a real request |

Prefill throughput scaled up through the tested range (2,397 -> 4,665 -> 5,182 -> 5,261 tok/s from 2,941 to 65,000 tokens), i.e., longer uncached prefills are *more* tok/s-efficient up to 65k, consistent with better GPU utilization at bigger batches during the prefill phase alone. No degradation was observed inside the verified range; the only degradation observed is binary (works up to 65k, crashes attempting to build/serve 131k).

## Power and thermal summary

- Sampled continuously via `nvidia-smi --query-gpu=power.draw,temperature.gpu,memory.used --format=csv -l 1` across all 4 cards for the duration of the run (3,037 rows, `receipts/nvidia-smi-longctx.csv`).
- Max temperature observed: 51 °C. Min: 33 °C. No card approached the 80 °C stop threshold at any point.
- No Xid or ECC events in `dmesg` for the run window.
- Memory per card under load (65,000-token level): roughly 50.6–60.8 GiB, consistent with the 16k-config baseline plus the larger KV pool reservation implied by `--max-model-len 262144`.

## Files

- `receipts/prefill_ladder.json` — full per-rep raw data for the passing levels and the 131,000 failure record.
- `receipts/needle.json`, `receipts/decode_longctx.json` — negative-result records (server dead before any real request).
- `receipts/full-container-log.txt`, `receipts/crash-excerpt.txt` — full and excerpted container log including the crash traceback.
- `receipts/boot-kv-lines.txt` — the two required boot log lines (KV cache size, max concurrency).
- `receipts/nvidia-smi-longctx.csv` — 1 Hz telemetry across all 4 cards for the run.
- `receipts/ready-utc.txt` — readiness timestamp.
- `receipts/prefill_ttft_vs_length.png` / `.svg` — chart (prefill tok/s and TTFT-proxy vs prompt length).
- `longctx_bench.py`, `run_longctx.py` — the harness used for this run.

## Limitations and what remains untested

- The 262,144-token max-model-len configuration boots, reports the expected KV pool size, and correctly serves at least three ladder rungs up to 65,000 prompt tokens.
- Everything at or above 131,000 prompt tokens is **untested**, not measured-and-failing-gracefully: the engine crashed while the harness was still constructing the test fixture (a sequence of `/tokenize` calls), which itself is a legitimate way to trigger a fault under this recipe, but it means no clean 400/OOM/timeout boundary was found — the boundary found is "engine survives 65k, does not survive whatever load pattern building/serving ~131k produces."
- Needle-in-haystack correctness, long-context decode speed, and vision-plus-long-context are all untested this run, purely as a consequence of the crash. None of these should be read as quality or capability findings.
- The server was left down at the end of this run's live phase; the task's "leave the server up" requirement could not be honored because the crash was outside the harness's control and the harness was authorized not to restart it. See the tracking issue comment for the exact state handoff.
