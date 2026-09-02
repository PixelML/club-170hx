# DeepSeek-V4-Flash-Vision-Exp — 163 tok/s reproducibility + 180 W vs 250 W (2026-09-02)

Public-safe evidence bundle. Server: resident `deepseek-v4-flash-vision-exp` on
4x CMP 170HX (SM80, 64 GiB), PP4, DSpark k=6, kv-cache fp8. Text-only prompts,
thinking off (not exposed by this server; no reasoning params sent). Not
restarted at the start of this run; restarted twice by this run after known
EngineCore crashes (see "What broke"), per the standing runbook for this
crash class.

## Method

Two protocols run side by side at each concurrency level, greedy sampling,
warmup(1) + 5 measured reps at C1/C2, warmup(1) + 3 at C4/C8, one attempt at
C16:

- **ours_short** — the exact prompt set that produced the prior 163.1 tok/s
  C1 figure (`club-170hx/docs/BENCHMARKS.md`): 5 short (~30-45 token) prompts,
  400 completion tokens, `ignore_eos`, non-streaming `/v1/completions`.
  `aggregate_tok_s = total_completion_tokens / synchronized wall time`.
- **ours_long** — same protocol, but every request uses our shared
  2,941-token fixture prompt (unique per-request id prefix, same prompt
  body) instead of the short prompts. Isolates the effect of prompt length
  on decode throughput.
- **mia** — Mia-style: each request gets a fresh 256-token prompt with a
  unique cold prefix (`uuid4`), 128-token forced decode window
  (`ignore_eos`), streaming. Per-stream decode tok/s is measured **after**
  the first token: `(completion_tokens - 1) / (t_last_token - t_first_token)`.
  TTFT reported separately (median). `aggregate = sum of per-stream decode
  tok/s`, median across reps.

Tokens are taken from the API's final `usage` object in all three protocols.
DSpark accept/draft token deltas are scraped from `/metrics`
(`vllm:spec_decode_num_*_tokens_total`) immediately before and after each
level. Power/clock/temp intended at 1 Hz for the whole run; see "What broke
in the harness" below — the continuous log has a gap, filled with discrete
load samples.

## Results — text c=1..4, our prompt sets (aggregate tok/s, median of reps)

| level | protocol | 180 W median (min-max) | 250 W median (min-max) | delta |
|---|---|---|---|---|
| C1 | ours_short | **118.95** (48.5-161.7) | **120.42** (86.9-154.7) | +1.2% |
| C1 | ours_long (2,941-tok prompt) | **94.03** (82.9-97.6) | **98.80** (98.4-100.6) | +5.1%, tighter spread |
| C1 | mia (decode-only, 256-tok prompt) | **134.67** (57.8-141.0) | **138.10** (79.0-145.6) | +2.5% |
| C2 | ours_short | **176.59** (71.4-190.3) | **158.75** (102.3-201.3) | -10.1% |
| C2 | ours_long | **159.57** (109.2-161.4) | **164.19** (151.1-165.6) | +2.9%, tighter spread |
| C2 | mia | **235.83** (141.0-239.4) | **241.38** (177.2-267.8) | +2.4% |
| C4 | ours_short | **207.59** (124.5-290.7) | crashed (see below) | n/a |
| C4 | ours_long | **210.39** (203.3-216.2) | crashed | n/a |
| C4 | mia | **363.51** (328.5-402.0) | crashed | n/a |
| C8 | any | 1/2 reps succeeded at 180 W (193.46 tok/s), then EngineCore crash | crashed on warmup | n/a |
| C16 | any | wedged (all 16 requests failed, broken pipe — server already down) | wedged (same) | n/a |

TTFT (mia, warm streaming path, median): 180 W — 0.334 s (C1), 0.383 s (C2),
0.548 s (C4). 250 W — 0.331 s (C1), 0.388 s (C2).

DSpark accept ratio (accepted/draft tokens, mia protocol, per-rep, median):
180 W C1 0.775, C2 0.811, C4 0.605; 250 W C1 0.790, C2 0.793. Ratio swings
from ~0.83 down to ~0.20-0.40 on individual reps, and those low-ratio reps
are the same reps with the lowest tok/s — draft acceptance variance, not
power cap, explains most of the run-to-run spread at fixed concurrency.

## The 163 tok/s verdict: reproduced, with a wide spread — not a stable point estimate

The prior figure (`ladder.json`, 2026-09-02, `club-170hx/docs/BENCHMARKS.md`
line 141) was 163.1 tok/s, median of 3 reps, range 124.1-177.3, C1,
ours_short-equivalent prompts. This run's same protocol, 5 reps instead of 3:

- 180 W: median **118.95** tok/s, range 48.5-161.7. The top of this run's
  range (161.7) sits within 1% of the prior median (163.1); the median of
  5 reps sits ~27% below it.
- 250 W: median 120.42 tok/s, range 86.9-154.7 — statistically
  indistinguishable from 180 W at this concurrency.

**Verdict: yes, reproducible as an achievable rate (peak-to-peak agreement),
not reproducible as a stable central tendency.** C1 aggregate tok/s on this
recipe has a genuinely wide run-to-run spread (roughly 50-180 tok/s band
across both arms and both short/long prompt lanes), correlated with the
DSpark accept-ratio swings above. Three reps (the original sample) landed
in the upper half of that band by chance; five reps pulled the median down.
Anyone quoting "163 tok/s" for this recipe should quote the range, not the
point estimate.

## The 250 W delta: no measurable throughput gain, worse tokens/Wh

At C1/C2 — the only levels both arms completed cleanly — 250 W bought
+1% to +5% on aggregate tok/s medians, well inside the run-to-run noise
band above, and *lost* -10% on one lane (C2 ours_short, likely noise in
the same direction as the wide spread already documented, not a real
regression). Measured active-load GPU power (util > 10%, C2 ours_short
burst, 4-card total): **414.8 W at the 180 W cap, 428.8 W at the 250 W
cap** — only +3.4% more power drawn, because this decode workload
(concurrency <= 2, small batches) is latency/memory-bound, not
power-bound: SM clocks under load average ~1443 MHz at *both* caps,
identical within measurement noise. The 250 W ceiling is never actually
being pushed against at this concurrency.

**Tokens per Wh** (C1/C2, ours_short, using the measured active-load power
above): 180 W — 1,032 tok/Wh (C1), 1,533 tok/Wh (C2). 250 W — 1,011 tok/Wh
(C1), 1,333 tok/Wh (C2). 250 W is flat-to-worse on efficiency at this
concurrency band; it buys nothing measured here.

Whether 250 W would matter at C4+ (heavier per-step compute, closer to
actually saturating the power cap) is **untested** — both arms lost the
server to the same EngineCore crash class at C4 or C8 before that could be
measured cleanly (see below). This is the open question a future run needs
a more stable serve path to answer.

## What broke

**EngineCore crash, recurring, independent of power cap.** Same crash class
already on file for this recipe (`RuntimeError('cancelled')` in
`shm_broadcast.acquire_read`, container exits 0):

- 180 W: survived C1, C2, C4 cleanly; crashed mid-C8 (1 of 2 reps succeeded
  at 193.46 tok/s before the crash). C16 then found the server already down
  (all 16 requests failed with broken-pipe errors) — this is the "attempt
  C16 once, record if wedged" case from the protocol; recorded, ladder
  stopped.
- 250 W: crashed earlier, at C4 warmup (0/4 succeeded), before any C4 data
  could be collected. C8 and C16 both found the server already down.
- **Both crashes happened with all 4 cards at 41-58 C core / 47-63 C
  memory** — nowhere near the 80 C/85 C stop thresholds. This is not a
  thermal event at either power cap; it is the same concurrency-scaling
  instability already documented for this fork, and it surfaced at a lower
  concurrency (C4 instead of C8) under the 250 W cap. One data point is not
  enough to claim power cap causes the earlier crash, but it is the only
  variable that changed between the two runs, so it is flagged for a future
  run to isolate.
- Each crash was recovered with `docker start dsv4-vision-vllm`, ~5 minutes
  to `/v1/models` readiness both times, matching the documented recovery
  time. No Xid/ECC events, no driver reload, no restart of anything beyond
  the container.

**Harness telemetry bug (self-inflicted, disclosed).** The 1 Hz
power/clock/temp sampler queried the wrong nvidia-smi field name
(`memory.temperature` instead of `temperature.memory`), so it silently
produced empty CSVs for both full-ladder arm runs — every sample failed and
was dropped rather than crashing loudly. Fixed in `bench_repro_power.py`
after the fact. Continuous 1 Hz telemetry for the two full ladder runs was
**not captured**; it is not being claimed. In its place: (a) 60 s-cadence
manual `nvidia-smi` checks during both live runs, used to enforce the 250 W
stop condition in real time (max observed 58 C core / 63 C memory — no stop
triggered); (b) two short, correctly-instrumented supplementary C2 bursts
run immediately after each arm, with working 1 Hz telemetry, used for the
power/clock/tok-per-Wh figures reported above. These are disclosed as
supplementary, same-day, same-config samples, not part of the original
ladder's own telemetry stream.

## Safety

250 W arm: max core temp observed 58 C, max memory temp 63 C (manual
60 s-cadence checks during the live run) — both stop thresholds (80 C /
85 C) never approached. No Xid or ECC events in either arm. 180 W restored
and verified on all 4 cards at the end of the run (`power.limit` readback:
180.00 W x4).

## Files

| file | contents |
|---|---|
| `bench_repro_power.py` | benchmark harness (this run) |
| `180w/repro_ladder_180w.json` | full 180 W arm receipt: all levels, all 3 protocols, per-rep |
| `250w/repro_ladder_250w.json` | full 250 W arm receipt |
| `180w/telemetry-1s-180w.csv`, `250w/telemetry-1s-250w.csv` | intended 1 Hz telemetry — empty, see "What broke" |
| `summary_computed.json` | computed medians/min/max per level/protocol/arm |
| `power_arms_chart.png` / `.svg` | two-arm comparison chart |
| `make_chart.py` | chart source |

Notes: served model reported here as `deepseek-v4-flash-vision-exp`
(internal aliasing removed). `127.0.0.1` retained intentionally.
