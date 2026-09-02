# GLM-5.3-Flash on CMP 170HX — compatibility review (2026-08-31)

Public-safe evidence bundle for a **negative/compatibility-only** finding:
every vLLM lane for `zai-org/GLM-5.3-Flash` on 4x CMP 170HX (SM80, 64 GiB
each) is blocked — the NVFP4 checkpoint headline among them — and the one
lane that clears both blockers (static fit + an SM80 runtime) is the
unsloth GGUF UD-IQ4_XS checkpoint on the sm_80 llama.cpp fork, measured at
17.73 tok/s median single-stream decode.

Source: evidence receipts copied from the upstream repository
[PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX)
(attempts/, results/, results/phase63/). The
[PixelML/GLM-5.3-Flash-vLLM-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-vLLM-CMP-170HX)
repository is a superseded redirect shell and holds no evidence; its PLAN.md
candidate matrix is planning-only and is cited from the upstream repo's
history, not copied here.

## Verdict summary

| Question | Answer | Status |
|---|---|---|
| NVFP4 checkpoint on SM80 vLLM | **INCOMPATIBLE** — never booted, blocked before download | measured (registry + PR review); no runtime error log exists |
| Any vLLM lane on SM80 | blocked — `glm5_next` absent from upstream vLLM registry; support PR vllm#53906 open, SM90+ only | measured (registry check 2026-08-30) |
| EXL3/TR3 4 bpw on SM80 ExLlamaV3 | blocked — `sm_121a` cubins only, no SM80 build | verified via source/distribution review |
| Official FP8 / BF16 on 4x64 GiB | no static fit (~76+ GiB/card at TP=4, FP8) | community-reported sizes; arithmetic exact |
| GGUF UD-IQ4_XS on sm_80 llama.cpp fork | **WORKS** — served, benchmarked, soaked | measured (phase C, 2026-08-30): 17.73 tok/s median c=1, 21/26 quality |

## Files

| file | contents |
|---|---|
| `RESULTS.md` | full sanitized compatibility report: static-fit table, NVFP4 blocker, fallback measurements |
| `attempt-nvfp4-vllm-sm121.md` | NVFP4 attempt record (compatibility-only, failed; the SM121 DGX recipe's runtime does not exist for SM80) |
| `attempt-gguf-ud-iq4xs-llamacpp.md` | UD-IQ4_XS GGUF attempt record (measured; the working fallback) |
| `attempt-awq-int4-vllm.md` | AWQ INT4 attempt record (static-fit failure on the 3-card era; runtime-blocked on 4 cards) |
| `attempt-exl3-tr3-4bpw-exllamav3.md` | EXL3/TR3 4 bpw attempt record (compatibility-only, failed) |
| `attempt-exl3-3bpw-0xsero.md` | EXL3 3.0 bpw record (not attempted; no servable artifact, repo-reported quality gate FAIL) |
| `attempt-fp8-bf16-reference.md` | official FP8/BF16 record (static-fit failure at either topology) |
| `attempt-future-small-quant.md` | what a qualifying checkpoint would need (not attempted) |
| `reference-miaai-dgx-exl3-4bpw.md` | external DGX Spark EXL3 reference (methodology source, not a CMP attempt) |
| `run-manifest.json` | phase-C run manifest: runtime commit, model revision, serve config, protocol, safety note |
| `warmups.json` | 3 discarded warm-up reps at c=1 (JSON array; converted from upstream JSONL) |
| `speed-c1.json` | 5 measured single-stream reps, 400-tok prompt / 256-tok completion (JSON array; converted from the upstream JSONL receipt, contents unchanged) |
| `ladder.json` | concurrency ladder 1/2/4 x 2 reps (JSON array; converted from upstream JSONL) |
| `soak.json` | 41-repetition soak at c=2 over 20 minutes (JSON array; converted from upstream JSONL) |
| `gpu-final.csv` | end-of-run per-card memory/thermal/power snapshot |
| `expected-sha256.txt` | pinned SHA-256 of the 5 GGUF shards |
| `experiments.csv` | phase-C experiment row (decision: baseline-below-90pct-gate) |
| `summary.csv` | machine-readable phase-C summary incl. quality index 0.7833 |

Note for committers: the repo-level `.gitignore` excludes `*.csv`, so the
two CSV receipts and `gpu-final.csv` need `git add -f` (same precedent as
`results/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm/telemetry-5s.csv`).
The notebook reads `gpu-final.csv` and `summary.csv` at run time; both must
be committed for `LIVE = False` replay to work.

## Sanitization

Receipts were fetched from the already-public upstream evidence repository
and re-scanned before copying here: no private IPs, hostnames, container
names, PIDs, PCI bus ids, MAC/serial identifiers, or storage paths are
present (automated scans for IPv4 literals, `/home/`-class paths, PCI
address patterns, serial/UUID fields — all clean). The run manifest's
storage field is the generic label "canonical shared model library (not
guest root)". Card identities in `gpu-final.csv` are generic indices 0-3.
Claims inside the receipts keep their original measured / inferred /
community-reported labels.
