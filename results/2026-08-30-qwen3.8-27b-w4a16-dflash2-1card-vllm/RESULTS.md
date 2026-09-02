# RESULTS — Qwen3.8-27B + DFlash2 on a rented CMP 170HX

**Date:** 2026-08-28
**Working instance:** rented single-card CMP 170HX 64 GB host, run label v9c (instance <instance-15>), on-demand
**Earlier failure:** instance <instance-1> — private GHCR image, documented below

## Bottom line

## v10 — Ninfer trick A/B (2026-08-28, instance <instance-16>)

Question: do the two Ninfer config tricks from the Reddit 5090 post transfer
to the syv vLLM stack on a 170HX? Four boots, one rental, same protocol as
v9 (streaming, `usage.completion_tokens`, temp 0, warmup 1 + 3 samples):

| Boot | Config | decode256 tok/s | decode900 tok/s | TTFT ms | prefill tok/s |
|---|---|---|---|---|---|
| A | dflash2 k=7, bf16 64k (v9 config re-run) | **140.5** | 126.6 | 107 | 2161 |
| B | mtp k=4, truncated 40k draft head | 133.1 | 119.8 | 81 | 2155 |
| C | mtp k=4, **MTP_DRAFT_VOCAB=0** (full-head draft) | 119.2 | 113.1 | 67 | 2145 |
| D | dflash2 k=7, **fp8 KV 131k ctx** | 135.8 | 121.2 | 110 | 1741 |

**lm-head-draft hurts here: −10.4% decode** (C vs B, 119.2 vs 133.1). The
truncated 40k-row draft head is the better design on this stack — the full
248k head's wider acceptance does not pay for its 6× wider draft GEMM. Do
not port Ninfer's `--lm-head-draft` to vLLM.

**fp8-KV long context costs 3.4% decode / 19.5% prefill** vs bf16 and buys
131k context on one card (D vs A). Ninfer's `--host-kv-mib` trick itself is
unnecessary at 64 GB — everything fits in VRAM.

Cross-node note: boot A re-measured the v9 config at 140.5 vs v9's 147.7
tok/s on a different physical card. Same config, same protocol — treat ~5%
as node-to-node spread when comparing runs.

Raw JSON: `bench-v10.json`, samples `bench-v10-samples.jsonl`,
onstart `onstart-v10.sh`.

## v11 — Ithrial's Ninfer sm_80 fork (2026-08-28, instance <instance-17>)

A team member flagged [`Ithrial/ninfer-cmp170hx`](https://github.com/Ithrial/ninfer-cmp170hx) —
a fork of Ninfer retargeted from Blackwell to sm_80 (Ampere). Tested head-to-head
against the v10 boots on the same offer (<offer-b>, same physical node as boot A):
native cmake build with `CMAKE_CUDA_ARCHITECTURES=80` (not
`NINFER_CUDA_ARCHITECTURES` — that one is Dockerfile sugar), fork's own
`neroued/Qwen3.8-27B-NInfer` artifact, served with
`--spec mtp --draft-tokens 3 --lm-head-draft --kv-dtype int8 --max-context 65536`
to match the vLLM CTX=fast setup.

**It builds and runs.** The scary-looking stall at `[237/245]`
(`nvfp4_small_t.cu`) is real compile time, not a hang — that one file takes
~20 minutes of template instantiation; the rest of the build is quick. Serve
boots, the 16.96 GiB artifact loads, health gate passes, bench completes.

| Metric (same node, same protocol) | vLLM DFlash2 k=7 (v10 boot A) | Ninfer fork (v11) |
|---|---|---|
| decode256 tok/s | **140.5** | 39.6 |
| decode900 tok/s | **126.6** | 41.2 |
| TTFT | 107 ms | **1.8 ms** |
| prefill (6.6K prompt) | 2,161 tok/s | **419,177 tok/s** |
| GPU at peak | 262 W / 57.6 GiB | 247 W / 20.1 GiB |

**Verdict: the fork is real but not competitive for decode.** Its sm_80
groupwise-int + w8 kernels land 3.5× behind our DFlash2 stack on single-stream
decode. What is genuinely impressive is the prefill path — 194× our vLLM
prefill and near-zero TTFT. If anything is worth porting from the fork it is
that prefill kernel design, not the decode tricks.

Caveats: bench went through `/v1/chat/completions` (the fork has no
`/v1/completions` route) so the prompt is 63 chat-templated tokens vs vLLM's
11 raw tokens; the fork silently ignores `ignore_eos` so decode runs can stop
at EOS (formula unchanged, still comparable); acceptance metrics are not
exposed in the serve log.

Raw JSON: `bench-v11-ninfer.json`, samples
`bench-v11-ninfer-samples.jsonl`, onstart
`onstart-v11-ninfer.sh`, bench harness `bench-ninfer-v11.py`.

The stack **works** on the CMP 170HX: public CUDA base image + syv-ai repo +
vLLM 0.27.1 + DFlash2 (7 draft tokens) served the model and survived a
controlled benchmark. Measured, single request, greedy, streaming:

| Metric | v9 (fast variant) | v6/v8 (base target) | LocalMaxxing reference |
|---|---|---|---|
| Output tok/s (256-tok gen) | **147.7** | 43.3–47.5* | 212.68 |
| Output tok/s (900-tok gen) | **134.5** | — | — |
| TTFT (11-tok prompt) | **76 ms** | 517–621 ms* | 72 ms |
| Prefill tok/s (6.6K prompt) | **2156** | 1705–1712 | 1221.6 |
| Mean acceptance length | **2.56–2.80** | 2.51–3.38 | — |
| Peak GPU | 255 W, 1455 MHz, 57.8 GiB | — | — |

\*v6/v8 counted SSE events as tokens; with speculative decoding each event
carries ~2.5-3 tokens, so those numbers understate real throughput by roughly
the acceptance length. v9 counts `usage.completion_tokens`.

v9 decode (147.7 tok/s) matches the syv repo's own community 170HX datapoint
(133.7 tok/s median, 3x900 tok, greedy, fast target) and the 3090's ~120-133
tok/s single-stream. The repo's tables show 212 is an **8-stream aggregate**,
not single-stream; single-stream on a 3090 is ~120 tok/s. Against that
yardstick the 170HX at 147.7 tok/s is fully explained.
| KV cache tokens | 69,758 (v6) / GPU_UTIL-sized 33.6 GiB (v8) | ~65,536 context | fine |

Full context is usable in v8: the empty `KV_MEM=` disables the 24-GB-card
5.2 GiB pin and lets GPU_UTIL=0.90 size the pool (33.63 GiB for KV), so the
full 65,536-token context fits with headroom.

DFlash2 is confirmed active: server-side SpecDecoding metrics show mean
acceptance length 3.38, per-position acceptance
[0.789, 0.539, 0.368, 0.25, 0.184, 0.145, 0.105], full CUDA graphs captured
for both model and drafter.

## Root causes of the 4.5x gap (verified from logs, not speculation)

(Superseded by v9 — see "v9: what actually closed the gap" below. The three
items here were real but secondary.)

## v9: what actually closed the gap (verified from logs)

1. **The fast variant was never built in v6-v8.** `docker/prepare.sh`
   `cd /app` then calls `prepare/quant_lm_head.py`; our clone lives in
   `/app/qwen-serving`, so prepare.sh died at `can't open file
   '/app/prepare/quant_lm_head.py'` — silently (`|| echo continue`). No
   int8/packed lm_head, no int8 embed/MTP, and crucially no
   **fast variant** (int4-GPTQ lm_head/MTP, +15% decode). v9 symlinks
   `/app/prepare` -> the clone's prepare dir; the log then shows every
   quantization step completing and `FAST-VARIANT-PRESENT`.
2. **Token counting was wrong.** v6/v8/v7e counted SSE events. Spec decode
   emits one event per accepted batch (~2.5-3 tokens). v9 uses
   `stream_options: {include_usage: true}` and `usage.completion_tokens`.
   This alone turns 43.3 "tok/s" into ~110-130 real tok/s.
3. **flashinfer topk fixed.** v8's JIT failed (`curand.h: No such file`)
   -> torch.topk fallback. v9 symlinks curand.h from the pip nvidia cu13
   package into /usr/local/cuda/include and clears the flashinfer cache.

1. **KV pool mis-sized.** The launcher pinned `KV_MEM=5583457484` (5.2 GiB),
   a 24 GB-card value from the syv repo defaults. The log confirms:
   "reserved 5.2 GiB memory for KV Cache ... This does not respect the
   gpu_memory_utilization config." ~58 GiB of the 64 GB card sat idle.
2. **flashinfer topk fell back to torch.topk.** `/usr/local/cuda/bin/nvcc`
   missing in the base image; flashinfer JIT could not compile its fast
   topk. The repo docs say flashinfer makes the DFlash2 selector ~2x faster.
3. **W4A16 vs W8A16 — tested directly (v7e).** Serving the official W8A16
   checkpoint dropped throughput to 31.6 tok/s (from 47.5 W4A16) and
   acceptance length to 2.9 (from 3.38). The card is bandwidth-bound, so
   denser weights are strictly slower. The reference's 212 tok/s cannot be
   explained by weight precision alone on this host.

## v6 configuration

- Image: `nvidia/cuda:13.0.1-base-ubuntu24.04` (public base, not the private GHCR)
- Stack: syv-ai/qwen38-27b-rtx3090 depth-1 clone, vLLM 0.27.1 + repo patches,
  flashinfer 0.6.16.post3, torch 2.13.0
- Launch: `SPEC=dflash2 CTX=fast MAX_SEQS=1 DFLASH_TOKENS=7 PORT=18020
  VLLM_V2_CUDAGRAPH_MEM_MIB=1400 KV_MEM=5583457484`
- Endpoint: http://<rented-host-ip>:40226 (host port 40226 <- container 18020)

## Benchmark protocol

Single request, greedy (temperature 0), streaming, 88-token prompt,
256 max tokens, ignore_eos. 1 warmup + 3 measured samples per run; three
runs total. Protocol matches the LocalMaxxing reference shape.

| Run | Mean output tok/s | Mean TTFT | Output tokens |
|---|---|---|---|
| 1 (ignore_eos) | 47.49 | 518.8 ms | 77 |
| 2 (eos honored) | 47.35 | 521.2 ms | 103 |
| 3 (repeat) | 47.22 | 517.5 ms | 103 |

v7e (W8A16, official recipe): **31.6 tok/s, 608-621 ms TTFT**, acceptance 2.9.
Raw: [bench-v7e.json](bench-v7e.json). Attempt log:
[attempt-history-v7.md](attempt-history-v7.md).

v8 (W4A16 + GPU_UTIL-sized KV pool, full 64K context): decode **43.3 tok/s /
610 ms TTFT**, prefill **1705-1712 tok/s on an 8,192-token prompt** (1 output
token, non-streaming, 2 samples each pass; repeat pass within 0.5%). The
prefill number beats the reference's 1221.6 by 1.4x — prefill is
compute-bound and this host's 170HX handles it well; decode remains
bandwidth-bound at ~1/4 of reference. Raw:
[bench-v8.json](bench-v8.json).

v9 (W4A16 + **fast variant** + curand.h fix + correct token counting,
instance <instance-15>): decode **147.7 tok/s** (256-tok) / **134.5 tok/s**
(900-tok), TTFT **76 ms**, prefill **2156 tok/s** (6,603-tok prompt).
Peak: 57.8 GiB VRAM, 255 W, 1455 MHz SM clock, 73 C, 100% util.
Acceptance length 2.56-2.80. Raw: [bench-v9.json](bench-v9.json),
samples: [bench-v9-samples.jsonl](bench-v9-samples.jsonl),
serve-log tail: `logs/v9-serve-tail.log` (full log kept upstream only).

### v9 benchmark samples (greedy, streaming, usage-token-counted)

| Run | i | TTFT | total | out tok | tok/s |
|---|---|---|---|---|---|
| decode256 | 1 | 75.9 ms | 1.803 s | 256 | 147.6 |
| decode256 | 2 | 75.7 ms | 1.804 s | 256 | 147.6 |
| decode900 | 1 | 93.4 ms | 6.779 s | 900 | 134.5 |
| decode900 | 2 | 93.0 ms | 6.797 s | 900 | 134.1 |
| prefill 6.6K | 1 | 3062.8 ms | 3.126 s | 8 | 2156 (prefill) |

## v9 settings audit vs the syv launcher (single-user/start_qwen.sh)

Applied exactly as the repo intends (verified in the v9 server log, not just
the script):

- Fast variant model (`...W4A16-AutoRound-fast` — confirmed in the engine
  init line) + W4A16 DFlash2 drafter, k=7
- `CTX=fast`: FLASH_ATTN, bf16 KV, 65,536 context
- Lookup-augmented drafting ON (log: "k=7 nmin=6 nmax=12 nstrong=6")
- All 20 patches applied, incl. split-KV verify attention and the sm80
  Marlin repack staging (auto-ON for compute capability 8.0)
- Tool parser `qwen3_coder`, reasoning parser `qwen3`, mamba state fp16,
  vision tower dropped (`--language-model-only`), async scheduling default,
  `max_num_batched_tokens 2048`

Three deliberate deviations:

1. **`MAX_SEQS=1` vs his default 8** — pinned for a single-stream benchmark.
   The syv tables show 212 tok/s is an 8-stream aggregate; this config would
   queue 8 concurrent streams. Main knob to chase 212.
2. **`GPU_UTIL=0.90` vs his 0.93** — his value tunes a 24 GB card's transient
   margin; on 64 GB the difference is ~1 GB of unused headroom. Negligible.
3. **`KV_MEM=` empty vs his pinned 5.2 GiB** — correct per his own guidance
  (the pin is a 24-GB-card constant; issue #40 un-pins it on bigger cards).
   Ours sized a 33.6 GiB pool holding the full 64K context.

Known-benign warning: "Unknown vLLM environment variable
VLLM_SPEC_DECODE_ATTN_QMAX" — that is the launcher's own variable, read by
the patched spec-decode attention code; identical on the author's 3090.

Raw JSON: [bench-v6-run1.json](bench-v6-run1.json),
[bench-v6-run2.json](bench-v6-run2.json),
[bench-v6-run3.json](bench-v6-run3.json).

## Attempt history

| # | Instance | Image | Outcome |
|---|---|---|---|
| v1 | <instance-1> | private GHCR | image 401, never ran |
| v2 | <instance-2> | public base + clone | prepare.sh path bugs |
| v3 | <instance-3> | public base + clone | same |
| v4 | <instance-4> | public base + clone | same |
| v5 | <instance-5> | public base + clone | venv symlink fix |
| v6 | <instance-6> | public base + clone | **working server + benchmark** |
| v7e | <instance-11> | public base + clone, W8A16 | working, 31.6 tok/s |
| v8 | <instance-12> | public base + clone | working, 43.3 tok/s (event-counted) |
| v9a/v9b | <instance-13>/<instance-14> | create returned stopped | destroyed immediately |
| v9c | <instance-15> | public base + clone, prepare fix | **147.7 tok/s, gap closed** |

## Local rerun — own 3-card CMP 170HX host (2026-08-29/30)

Same v9 recipe rerun on our own 3-card box, power-capped at **180 W**
(VAST ran 255 W), one card. Weights rebuilt once to
<model-storage>/models/qwen38/bench-2026-08-29 (base dbirks W4A16-AutoRound +
syv post-quant chain + DFlash2 drafter), stored on the 8.4 TB library
NFS for reuse.

**Correction (2026-08-30):** an earlier writeup reported 47.0 tok/s from a
single-card run. That run overlapped the DSV4 docker build on the host and
pinned a KV slab (KV_MEM set), so it under-measured. A clean rerun of the
same v9 recipe - KV_MEM empty, GPU_UTIL 0.90, host idle - ran once per card
sequentially (gpu0 02:30, gpu1 02:40, gpu2 02:43 UTC). Those numbers
supersede 47.0.

| card | decode 256 tok | decode 900 tok | TTFT | prefill 6.6K | SM peak |
|---|---|---|---|---|---|
| gpu0 | 135.3 tok/s | 121.3 tok/s | 201 ms | 1957 tok/s | 1395 MHz |
| gpu1 | **140.3 tok/s** | 124.8 tok/s | 190 ms | 1955 tok/s | 1500 MHz |
| gpu2 | 133.6 tok/s | 119.9 tok/s | 181 ms | 1926 tok/s | 1395 MHz |

vs VAST v9 at 255 W: decode 147.7 -> best local 140.3 (**95% of VAST at
180 W, 71% of VAST's power**); prefill 2156 -> ~1955 (91%); TTFT 76 ->
~190 ms (PCIe Gen2 x4 host link). Acceptance length 2.14-3.00, draft
acceptance 16-29% per request - in family with v9c (2.56-2.80). Peak card
power 185-223 W instantaneous (180 W cap with transients), 100% util,
51 C core / 52-61 C mem, no Xid, no power-brake. Weights:
<model-storage>/models/qwen38/bench-2026-08-29 (W4A16-AutoRound-fast + DFlash2
drafter). Harness: `run-local-card.sh` +
[bench-usage.py](bench-usage.py) (v9-usage-token-counted
protocol, identical to v9c). Raw evidence archived per card:
[local-rerun-gpu0](local-rerun-gpu0/bench.jsonl) /
[gpu1](local-rerun-gpu1/bench.jsonl) /
[gpu2](local-rerun-gpu2/bench.jsonl) - bench.jsonl, metadata.txt,
specdec.log, nvidia-after.txt in each.

**Verdict: recipe fully reproduced at home. The earlier "power envelope"
explanation was wrong - the 180 W cap costs only ~5% decode vs 255 W.
The 47.0 number was host contention + pinned KV, not silicon.**

Earlier depressed runs (kept for the record): 47.0 tok/s / 1927.6 prefill
(contention + pinned KV); run 1 was 30.9 tok/s (lued INT8-W8A16, wrong
checkpoint).

DeepSeek-V4-Flash-0731 on all 3 cards is split to its own repo:
[DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX)
(83.3 tok/s decode).

## Cost record

Omitted from this sanitized bundle (billing detail). The upstream receipts retain the
per-attempt cost record.

## Failure evidence from v1 (instance <instance-1>)

1. `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest` requires auth: anonymous
   manifest request -> HTTP 401.
2. Container filesystem ~1.2 MiB; /app held only onstart.sh/ports.log.
3. SSH denied separately (authorized_keys mode/ownership on team account).

Details: [container-filesystem.md](container-filesystem.md),
[image-manifest-probe.md](image-manifest-probe.md).
