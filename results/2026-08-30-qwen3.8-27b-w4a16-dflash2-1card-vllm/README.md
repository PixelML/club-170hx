# Qwen3.8-27B W4A16 + DFlash2 — single-card vLLM, 180 W vs 255 W, v10/v11 A/Bs (2026-08-28 → 2026-08-30)

Public-safe evidence bundle for single-card CMP 170HX (SM80, 64 GiB) runs of
`Qwen3.8-27B` W4A16 + DFlash2 speculative decoding on vLLM 0.27.1 (syv-ai
recipe). The measured story spans three sessions:

1. **v6 → v9** (2026-08-28, rented single-card host, 255 W): the syv-ai
   recipe brought up on a public CUDA base image; v9 closes the decode gap
   (147.7 tok/s) after fixing a silently-failing quantize step, flashinfer
   topk, and wrong token counting.
2. **v10 / v11** (2026-08-28, same rented host): the Ninfer config-trick A/B
   (lm-head-draft, fp8-KV long context) and the Ithrial Ninfer sm_80 fork
   head-to-head against the vLLM DFlash2 stack.
3. **Local rerun** (2026-08-29/30, own 3-card host, one card at a time,
   **180 W power cap**): v9 recipe reproduced at home; 95% of the rented
   255 W decode at ~71% of the power.

`RESULTS.md` is the full sanitized run report; `UPSTREAM-README.md` is the
sanitized upstream project README (RTX 3090 baseline included). Notebook:
`notebooks/2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb`.

## Headline results (all measured, greedy, single stream, usage-token-counted)

| run | config | decode 256 tok (tok/s) | decode 900 tok (tok/s) | TTFT 11-tok (ms) | prefill 6.6K (tok/s) | power |
|---|---|---|---|---|---|---|
| v9 (rented) | dflash2 k=7, bf16 KV, 64k ctx | **147.7** | 134.5 | 76 | 2156 | 255 W cap (peak 255.1 W, 73 C) |
| v10 boot A (rented) | v9 config re-run, different card | 140.5 | 126.6 | 107 | 2161 | peak 261.9 W |
| v10 boot B | mtp k=4, truncated 40k draft head | 133.1 | 119.8 | 81 | 2155 | peak 258.4 W |
| v10 boot C | mtp k=4, full 248k lm-head draft | 119.2 | 113.1 | 67 | 2145 | peak 256.1 W |
| v10 boot D | dflash2 k=7, fp8 KV, 131k ctx | 135.8 | 121.2 | 110 | 1741 | peak 261.6 W |
| v11 Ninfer fork (rented, same node as A) | mtp k=3, int8 KV, 64k ctx | 39.6 | 41.2 | 1.8 (63-tok chat prompt) | 419,177 | peak 247.4 W |
| local gpu0 (180 W cap) | v9 recipe | 135.3 | 121.3 | 201 | 1957 | 180 W cap (peak 185.5 W) |
| local gpu1 (180 W cap) | v9 recipe | **140.3** | 124.8 | 190 | 1955 | 180 W cap (peak 222.9 W) |
| local gpu2 (180 W cap) | v9 recipe | 133.6 | 119.9 | 181 | 1926 | 180 W cap (peak 195.3 W) |

Key A/B findings (measured, same node/protocol unless noted):

- **180 W vs 255 W:** best local card (gpu1) reaches 140.3 tok/s = 95% of the
  rented v9 run at a 180 W cap (255 W upstream); prefill ~91%. The earlier
  "power envelope" explanation for a 47.0 tok/s local number was wrong — that
  run overlapped a host docker build and pinned a KV slab. See RESULTS.md.
- **Ninfer `--lm-head-draft` hurts on this stack:** full 248k-head drafting is
  −10.4% decode256 vs the truncated 40k draft head (119.2 vs 133.1 tok/s, C vs B).
- **fp8 KV + 131k context costs 3.4% decode / 19.5% prefill** vs bf16 64k (D vs A).
- **Ninfer sm_80 fork:** builds and runs, 3.5× behind vLLM DFlash2 on decode,
  but ~194× faster prefill (419k tok/s) and 1.8 ms TTFT. Caveat: v11 was
  benchmarked over `/v1/chat/completions` (63-token templated prompt vs 11 raw),
  and the fork ignores `ignore_eos`.

## Protocol

Greedy (temperature 0), streaming, single stream (`MAX_SEQS=1`), 1 warmup +
3 measured samples per cohort; 11-token prompt for decode cohorts (256 and 900
completion tokens), 6,603-token prompt for prefill. **Token counts come from
the final `usage` object** (`stream_options.include_usage`), never from SSE
event counts — the v6/v8 numbers counted events and understate real throughput
by roughly the acceptance length (~2.5-3×); they are preserved below as
negative results. GPU state sampled via `nvidia-smi`; acceptance metrics from
server-side SpecDecoding log lines.

## Files

| file | contents |
|---|---|
| `RESULTS.md` | full sanitized run report (v1 failure → v9 → v10 → v11 → local rerun) |
| `UPSTREAM-README.md` | sanitized upstream project README (incl. RTX 3090 baseline A/B) |
| `attempt-history-v7.md` | W8A16 (v7e) negative result + failed v7 sub-attempts |
| `bench-v9.json` / `bench-v9-samples.jsonl` | v9 receipt + per-sample raw (rented, 255 W) |
| `bench-v10.json` / `bench-v10-samples.jsonl` | v10 four-boot A/B receipt + boots C/D per-sample raw |
| `bench-v11-ninfer.json` / `bench-v11-ninfer-samples.jsonl` | Ninfer sm_80 fork receipt + per-sample raw |
| `bench-v6-run1/2/3.json`, `bench-v7e.json`, `bench-v8.json` | earlier attempts incl. event-counted numbers (kept for the record) |
| `onstart-v9.sh`, `onstart-v10.sh`, `onstart-v11-ninfer.sh` | exact launch/build scripts as run (masked) |
| `run-local-card.sh` | local 180 W rerun driver (health gate, bench, telemetry, kernel-log check) |
| `bench-usage.py` | usage-token-counted streaming bench harness (as run locally) |
| `bench-ninfer-v11.py` | v11 bench harness (`/v1/chat/completions` path) |
| `local-rerun-gpu0/1/2/` | per-card raw evidence: `bench.jsonl`, `metadata.txt`, `specdec.log`, `nvidia-after.txt` |
| `rtx3090-baseline/` | RTX 3090 run of the identical recipe (`results.json`, `bench-stdout.jsonl`) |
| `logs/v9-serve-tail.log` | filtered + tail lines of the v9 serve log (quantize steps, acceptance, readiness) |

## Sanitization notes

Internal infrastructure details (rental instance/offer ids, public IPs, the
local hostname, pids, PCI bus ids, GPU UUIDs, storage paths, bench API-key
values, billing figures) are masked with `<...>` placeholders in the same
style as the other bundles in this repo: instance ids → `<instance-N>`,
offers → `<offer-a/b>`, rented hosts → `<rented-host-ip>`,
`/library/` → `<model-storage>/`, `/home/<user>/` → `<workdir>/`,
`/models/venvs/` → `<fast-cache>/venvs/`. Loopback `127.0.0.1` and the
container-internal `/app` tree are retained intentionally. `run-local-card.sh`
was renamed from its original (which embedded a private hostname). Because of
masked placeholders, `run-local-card.sh` and the `onstart-*.sh` scripts are
records and will not parse as-is under bash. vLLM SpecDecoding pids appear as
`pid=<pid>`. The upstream evidence repo is
[PixelML/Qwen3.8-27B-CMP-170HX](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX);
probe documents not copied here (image-manifest, ssh-auth, instance metadata)
remain upstream only, with their findings summarized in `RESULTS.md`.
