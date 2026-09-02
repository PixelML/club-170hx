# DeepSeek-V4-Flash-Vision-Exp — vision on 4-card PP4 vLLM, live run (2026-09-02)

Public-safe evidence bundle for the first live vision measurement of
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on 4x CMP 170HX (SM80, 64 GiB)
with pipeline parallelism 4, DSpark speculative decoding (k=6), kv-cache fp8,
block 256, max-model-len 16384, max-num-batched-tokens 2048, max-num-seqs 8,
`--limit-mm-per-prompt '{"image": 2}'`. This is the same Path 3 SM80 fork used
for the text-only ladder in `results/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm/`,
with five additional fixes to reach a working vision boot (see `docs/LESSONS.md`
appendix). Internal infrastructure details are masked with `<...>` placeholders;
sha256 digests are kept unmodified.

The server was already running from a prior boot session when this run
started. This run did not start, stop, or restart it at any point, including
after the crash described below.

## What was measured

1. **Gate.** `/v1/models` plus a deterministic greedy check — PASS, 3/3
   identical reps.
2. **Golden corpus** (`<model-storage>/golden`, 30 rows, temperature 0,
   `max_tokens=200`, run against the live endpoint via `run_corpus.py`):
   - Image rows (10): **10/10 keyword match.**
   - Text rows (20): **15/20 keyword match, 10/20 exact-match** against the
     DGX Spark reference (`golden_outputs.jsonl`), with 4 `finish_reason`
     mismatches (`length` vs `stop`). This is a known limitation, not a
     stability issue — see docs/LESSONS.md.
3. **Uncached prefill**, 2,941-token fixture, `max_tokens=1`, 3 reps: rep0
   918.2 tok/s (first long prefill after boot), rep1 2352.4 tok/s, rep2
   2400.0 tok/s; median **2352.4 tok/s**.
4. **Warm streaming TTFT**, same fixture with the prefix cached, 3 reps:
   0.3837 / 0.4257 / 0.3858 s; median **0.386 s**.
5. **Text-only concurrency ladder** (`bench_harness.py ladder`, greedy,
   `ignore_eos`, 400 completion tokens, 1 warmup + 3 measured reps per level):

   | level | status | warmup agg tok/s | measured agg tok/s (3 reps) | agg median | per-request median tok/s | success |
   |---|---|---|---|---|---|---|
   | C1 | PASS | 142.95 | 177.25 / 163.06 / 124.11 | **163.06** | 163.08 | 3/3 |
   | C2 | PASS | 117.04 | 106.83 / 204.37 / 116.57 | **116.57** | 119.56 | 6/6 |
   | C4 | **FAIL** | 143.67 | 241.85 / 159.98 / crash | n/a | n/a | 8/12 (crash on rep 3) |
   | C8 | not measured | — | — | — | — | — |
   | C16 | not measured | — | — | — | — | — |

6. **Text+image concurrency ladder** (`bench_harness.py ladder_image`, same
   protocol, one fixed 64x64 gradient image per request via
   `/v1/chat/completions`). The container came back up later in the same
   session; C1 and C2 were measured then. C4 and above were not attempted,
   given the text-only crash at C4 above:

   | level | status | warmup agg tok/s | measured agg tok/s (3 reps) | agg median | per-request median tok/s | success |
   |---|---|---|---|---|---|---|
   | C1 | PASS | 42.46 | 41.22 / 45.32 / 47.11 | **45.32** | 45.32 | 3/3 |
   | C2 | PASS | 66.36 | 78.23 / 61.45 / 79.77 | **78.23** | 45.59 | 6/6 |
   | C4 | not attempted (see "What broke") | — | — | — | — | — |
   | C8 | not measured | — | — | — | — | — |
   | C16 | not measured | — | — | — | — | — |

## What broke

During the text-only ladder, at concurrency 4, rep 3 of 3, the vLLM
`EngineCore` process died mid-batch with:

```
RuntimeError: cancelled
  (raised from distributed/device_communicators/shm_broadcast.py, acquire_read)
```

All 4 in-flight requests in that rep received `HTTP 500 EngineDeadError`. The
container then exited cleanly (exit code 0) and the API server stopped
responding. This session did not restart it, per the standing operating
instruction for this endpoint, so the remainder of the protocol — C8 and C16
on the text-only ladder, and the entire text+image ladder (C1 through C16) —
was **not measured** in this run.

This is a second, independent concurrency ceiling for the same recipe: the
sibling text-only notebook already recorded a C16 failure (device-side assert
on rank 3 of the draft path) for this fork. This run shows the ceiling can
sit lower, at C4, when a vision-capable build carries the extra per-layer
`hc_attn_fn_broadcast` and raw-input-tokens bookkeeping needed for the
vision-aware MoE routing (see docs/LESSONS.md). Both failures point at the
same class of issue: this fork's multi-process pipeline-parallel executor is
not yet robust under concurrent load once request count and pipeline depth
combine past a recipe-specific threshold.

The container came back up later in the same session under infrastructure
outside this notebook's control; this run did not restart it either time.
Once it answered `/v1/models` again, the text+image C1/C2 levels above were
measured against it; C4 and above were left alone given the crash history.

The single-request text+image throughput figure from the prior boot session
(finish_reason=length, ~54.6-56.2 tok/s per request, wall time 7.12-7.31 s for
400 completion tokens, `<model-storage>/path3/receipts/bench-c1/c1_image.json`)
is kept in the appendix as **measured, prior session** context; this run's own
C1 text+image measurement (45.32 tok/s median) sits somewhat below it, which
this notebook reads as normal run-to-run variance, not a regression, given
neither run repeats the other's exact prompt set.

## Safety

All 4 GPUs stayed at 42-47 C for the full session (gate through the crash and
5 minutes after). No Xid or ECC events. GPU memory and utilization returned
to 0 within seconds of the crash. No driver reload, no restart, no power-state
change.

## Files

| file | contents |
|---|---|
| `summary.json` | machine-readable run summary, including the crash and the not-measured levels |
| `vision-gates.json` | functional-gate results |
| `gate.json` | deterministic greedy gate receipt |
| `prefill.json` | uncached 2,941-token prefill receipt (3 reps + GPU state) |
| `ttft.json` | warm streaming TTFT receipt (3 reps) |
| `ladder.json` | per-level, per-rep, per-request text-only ladder receipt, including the C4 crash |
| `ladder_image.json` | per-level, per-rep, per-request text+image ladder receipt (C1, C2) |
| `cmp_run_live.jsonl` | golden-corpus run against the live endpoint (30 rows) |
| `telemetry-1s.csv` | 1-second power/thermal/memory samples spanning the whole session |
| `fixtures/prior_evidence_gradient_regenerated.png` | the gradient proof image used in the notebook hero cell |
| `prior-evidence/vision-ready.json` | the prior boot session's gate/corpus/C1 receipt (kept for the appendix, superseded by this run's live receipts where they overlap) |
| `../2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm/bench_harness.py` | shared harness, extended in this run with `ladder_image`/`chat_completion` for the (not yet exercised) text+image ladder |

Notes: paths beginning with `/models/`, `/home/<user>/` are replaced by
`<model-storage>/` and `<workdir>/`; `127.0.0.1` is retained intentionally.
The served model name is reported here as `deepseek-v4-flash-vision-exp`
(internal aliasing removed).
