# RESULTS — GLM-5.3-Flash on 4x CMP 170HX: compatibility review + fallback baseline

Date: 2026-08-31 (compatibility finding dated 2026-08-30 upstream; fallback measured 2026-08-30)
Node: 4x NVIDIA CMP 170HX (GA100, SM80, 64 GiB HBM2e each, 256 GiB aggregate), forced airflow
Status: **negative result (compatibility-only) with one measured fallback baseline**

## Headline

**FAIL — the NVFP4 lane is incompatible with SM80.** The
[LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
checkpoint cannot be served on CMP 170HX by any known runtime, for two
independent reasons, either one fatal:

1. **No model support.** `glm5_next` (`Glm5NextForConditionalGeneration`) is
   absent from upstream vLLM's model registry (measured: registry grep,
   2026-08-30). The only known support PR,
   [vllm#53906](https://github.com/vllm-project/vllm/pull/53906), is open,
   unmerged, and SM90+ only.
2. **No SM80 attention backend.** The DGX recipe's serving path uses
   sparse-MLA attention backends (FLASHINFER_MLA_SPARSE_SM120/SM90 class)
   that do not exist for SM80 (measured: registry grep + PR review, 2026-08-30).

The checkpoint was **never downloaded and never executed** — blocked before
download by the runtime incompatibility and the preserve-gate on the live
test node. There is consequently **no runtime error log for this lane**; the
negative result rests on the registry/PR review above, not on a failed boot.
Static fit is not the blocker: ~194.4 GiB (community-reported size) is
~48.6 GiB/card at TP=4, which fits the 64 GiB budget on paper (inferred from
the community-reported size; per-card arithmetic exact).

**The one positive data point:** the fallback lane —
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
UD-IQ4_XS (146.05 GiB, measured HF blob sum, 5 SHA-256-verified shards) on
the unslothai/llama.cpp fork at commit `00699716c275498ff84d71e329178fe21cba56a6`
built with `-DCMAKE_CUDA_ARCHITECTURES=80` — **served and completed a full
benchmark** on 2026-08-30: **17.73 tok/s median single-stream decode**
(5 reps, 400-token prompt / 256-token completion), flat ~17.5-17.7 tok/s
aggregate at c=2 and c=4, a clean 41-rep soak at c=2 over 20 minutes, and a
corrected 26-task quality pack scoring 21/26 (quality index 0.783).

## Static-fit table (weights-only, per card)

Budget: 64 GiB = 68,719,476,736 bytes per card. "Static fit" excludes CUDA
context (~0.5-1 GiB), activations, and KV cache. Three-card-era (TP=3)
arithmetic is kept where it differs, as history.

| # | Candidate (revision) | Exact bytes | Size status | Per card @ TP=4 | Per card @ TP=3 (historical) | SM80 runtime | Outcome |
|---|---|---:|---|---:|---:|---|---|
| 1 | unsloth GLM-5.3-Flash-GGUF UD-IQ4_XS @ `2975ab41` | 156,822,111,075 B = 146.05 GiB | measured (HF blob sum) | **36.51 GiB** (+27.5 GiB margin) | 48.68 GiB (+15.3 GiB) | **yes** — llama.cpp fork `00699716`, sm_80 build (measured) | **served + benchmarked** |
| 2 | Mia-AiLab EXL3-TR3-4bpw @ `25a44fdb` | 175,642,157,752 B = 163.58 GiB | measured (HF blob sum at pinned rev) | 40.90 GiB (+~23 GiB, expert-skew unverified) | ~54.5 GiB | untested — `sm_121a` cubins only, no SM80 build | blocked (compatibility-only) |
| 3 | LibertAIDAI NVFP4 @ `11d73216` | ~194.4 GiB | community-reported | ~48.6 GiB (fits on paper) | ~64.8 GiB (over budget) | **no** — `glm5_next` absent; PR #53906 SM90+; sparse-MLA SM12x backends | **failed (compatibility-only)** |
| 4 | cyankiwi AWQ-INT4 @ `3999f9bf` | 212,721,952,636 B = 198.1 GiB | measured (HF blob sum) | 49.53 GiB (+14.5 GiB) | **66.04 GiB (−2.04 GiB, over)** | no — same `glm5_next` blocker | failed static fit (3-card era); runtime-blocked on 4 cards |
| 5 | zai-org official FP8 | >= ~328 GB | community-reported | ~76+ GiB (**no fit**) | no fit | n/a — memory | failed (static fit) |
| 6 | 0xSero EXL3 3.0 bpw @ `8b099bf2` | not established (community-reported) | community-reported | not established | not established | none — `requires_custom_loader`, `runtime_status: pending_full_server` | **not attempted**; repo-reported quality gate FAIL (KL 0.153, ppl delta 0.093, top-1 agree 0.873) |

Reading: on the four-card node, size alone kills only the official FP8/BF16
lane. Every 4-bit lane fits on paper; what actually distinguishes row 1 is
the **runtime** — it is the only candidate with a buildable SM80 serving
path at all.

## Fallback measured results (phase C, 2026-08-30)

From `run-manifest.json`, `speed-c1.json`, `ladder.json`, `soak.json`,
`gpu-final.csv`, `summary.csv`:

| Metric | Value | Source |
|---|---|---|
| Decode, c=1 median (5 reps, 400 in / 256 out) | **17.73 tok/s** (reps: 17.80 / 17.59 / 17.73 / 17.98 / 17.73) | measured, `speed-c1.json` |
| End-to-end per task, c=1 | 14.44 s median | measured, `speed-c1.json` |
| Aggregate, c=2 (2 reps) | 17.50 / 17.53 tok/s | measured, `ladder.json` |
| Aggregate, c=4 (2 reps) | 17.73 / 17.69 tok/s (flat — compute-bound) | measured, `ladder.json` |
| Soak, c=2, 20 min | 41/41 reps ok; per-rep aggregate 17.3-18.09 tok/s (first 3: 17.51-17.64, last 3: 17.60-17.74; upstream characterizes the band as stable 17.5 -> 17.7) | measured, `soak.json` |
| Corrected quality pack | 21/26 tasks, quality index 0.783 (math 8/8, instruction 4/5, long-ctx 3/3, held-out math 4/4, held-out code 1/1; coding 1/3, held-out instruction 0/2) | measured, `summary.csv` |
| VRAM per card, end-of-run snapshot | 32,656 / 33,342 / 41,646 / 42,312 MiB used of 65,536 MiB | measured, `gpu-final.csv` |
| Core / memory temps (snapshot) | 41-43 °C / 53-55 °C | measured, `gpu-final.csv` |
| Power per card (snapshot) | 40.10-44.49 W | measured, `gpu-final.csv` |
| Driver / kernel / OS | 610.43.03 / 6.8 / Ubuntu 22.04 | measured, `run-manifest.json` |

Decision recorded upstream for the baseline row: `baseline-below-90pct-gate`
(the throughput is far below the club's 90%-of-reference gate; it is a
compatibility fallback, not a performance result).

## Receipt gaps (stated plainly)

- **No NVFP4 runtime log exists.** The lane was never executed; the
  incompatibility is established by registry/PR review (measured
  2026-08-30), not by a failed boot.
- **NVFP4 checkpoint size is community-reported** (~194.4 GiB, from the DGX
  recipe documentation), not re-measured via the HF API.
- **Official FP8/BF16 sizes are community-reported** (>= ~328 GB FP8).
- **EXL3 3.0 bpw size/quality are community-reported** from the checkpoint
  card at the pinned commit; not re-measured.
- **No continuous thermal/Xid telemetry for the fallback run.** Safety
  telemetry is a single end-of-run snapshot plus bounded guard calls between
  phases; throttle events, continuous kernel-log Xid capture, and the
  per-card configured power limit were not recorded this phase
  (`run-manifest.json` safety block records these as `not-recorded`).
  No fault-free-operation claim is made for the soak window.
- **Two harness defects were found and fixed during the quality phase** (a
  sandbox wrapper dropping the candidate module; completion budgets starving
  reasoning output). The corrected 21/26 pack is the valid one; remaining
  misses lack finish/reasoning fields in receipts, so no per-miss diagnosis
  is claimed.

## Evidence

- Upstream evidence repository: [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX)
  (attempts/ and results/ copied here; phase-C receipts under `results/phase63/` upstream)
- vLLM registry check: [vllm/model_executor/models/registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py) (no `glm5` match, 2026-08-30)
- Support PR (open, SM90+): [vllm#53906](https://github.com/vllm-project/vllm/pull/53906)
- llama.cpp GLM-DSA support PR (open): [ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754)
- DGX reference recipe: [PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark)
