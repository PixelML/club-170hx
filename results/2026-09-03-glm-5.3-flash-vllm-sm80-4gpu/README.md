# GLM-5.3-Flash · vLLM sm80 · 4x CMP 170HX

Measured 2026-09-03 on four NVIDIA CMP 170HX cards (SM80), using the
PixelML `sm80vllm` fork, TP4, native MTP speculation, and the club-standard
180 W/card cap.

**Evidence class: measured public replay.** The companion notebook is
`LIVE=False` and replays the committed receipts below. It does not claim a
new live endpoint run.

## Headline

| Measurement | Result | Evidence |
|---|---:|---|
| Decode, C1 | **56.4 tok/s median** (56.9 peak) | Measured; five repetitions, MTP depth sweep |
| Aggregate, C8 | **37.0 tok/s** | Measured; same TP4 boot |
| MTP selection | **k=3** | Measured; k=2/3/5 sweep |
| Prefill / TTFT | untested | Not in this receipt |

C1 is the useful single-stream result: 56.4 tok/s is 2.1x the EXL3 C1
comparison (25.2 tok/s) under its separate short-context protocol. The
server was configured with `max-model-len=524,288`; long-context stress was
not part of this receipt. At C8,
TP4 all-reduce traffic across the PCIe fabric makes the aggregate
communication-bound; the canonical 180 W EXL3 contextual comparison is
44.6 tok/s (the earlier 250 W run was 44.8 tok/s).

## Pinned receipt

| Field | Value |
|---|---|
| Checkpoint | [`wtdcode/GLM-5.3-Flash-AWQ-W4A16`](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16) @ `abd7b07719111f137e1de8a0c1b7e01c11b74d1a` |
| Quantization | AWQ W4A16 (compressed-tensors), 190,843,146,533 bytes |
| Runtime | vLLM, `PixelML/sm80vllm`, branch `glm53-sm80`, commit `f6fbf3b854` |
| Topology | TP4, four CMP 170HX (SM80), 180 W/card |
| Max model length | `524,288` configured; long-context stress test untested |
| MTP | native, `num_speculative_tokens=3` (swept 2/3/5) |
| Protocol | temperature 0.7, 512 output tokens, `ignore_eos`, first repetition retained as cold reference |
| Token accounting | final `usage.completion_tokens` divided by wall time |
| Health | 0 Xid/ECC events; numeric peak temperature was not captured in this receipt |

## Structured data

- [`summary.json`](summary.json) — model, hardware, runtime, protocol, headline, and health pins.
- [`mtpsweep.json`](mtpsweep.json) — the complete MTP depth sweep.
- [`ladder.json`](ladder.json) — the aggregate C1/C8 ladder and comparison note.
- [`attempts.json`](attempts.json) — fixed, blocked, and unsafe-path findings.
- [`summary.csv`](summary.csv) — chart source; no values are generated in the chart code.

## Reproduce

1. Stage the pinned checkpoint on local NVMe. The 190.8 GB copy loaded at
   about 54 s/shard warm in the measured environment; network storage was
   materially slower.
2. Pull `ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903` and launch
   vLLM with TP4, automatic KV mode, max model length 524,288, and native
   MTP k=3. The full command is in the [executed notebook](../../notebooks/2026-09-03-glm-5.3-flash-4card-tp4-vllm.ipynb).
3. Warm once, then run five C1 repetitions and the C1/C8 ladder. Count only
   the final `usage.completion_tokens` field. Keep FlashInfer autotune
   enabled with MTP-3.

The public notebook includes download, launch, preflight, benchmark, and an
editable OpenAI-compatible request cell. Replace `<weights>` and set
`GLM_VLLM_BENCH_ENDPOINT_URL` only for an intentional live run.

## Known findings and limits

- PP4 + MTP is the next aggregate optimization; it was not ported in this
  run. The current C8 score is therefore a real TP4 result, not a PP claim.
- Do **not** combine `--no-enable-flashinfer-autotune` with MTP-3; the
  attempt log records a reproducible startup failure for that combination.
- FP8 KV cache and block size 256 hit a Triton MLA limitation on this model;
  the measured recipe keeps automatic KV mode.
- The EXL3 comparison values use a separate short-context protocol and are
  labeled as context, not a like-for-like replacement.

## Publication artifacts

- [Executed notebook](../../notebooks/2026-09-03-glm-5.3-flash-4card-tp4-vllm.ipynb)
- [Sweep chart PNG](../../assets/charts/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu-sweep.png) · [SVG](../../assets/charts/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu-sweep.svg)
- [Motion source and render notes](../../assets/video/glm53-vllm-sm80-motion/README.md)
- [Poster PNG](../../assets/video/glm53-vllm-sm80-motion/glm53-vllm-sm80-motion-poster.png)
- [Square MP4](../../assets/video/glm53-vllm-sm80-motion/glm53-vllm-sm80-motion-1080x1080.mp4) · [landscape MP4](../../assets/video/glm53-vllm-sm80-motion/glm53-vllm-sm80-motion-1920x1080.mp4) · [vertical MP4](../../assets/video/glm53-vllm-sm80-motion/glm53-vllm-sm80-motion-1080x1920.mp4)

The poster and MP4s are generated from the local HyperFrames-style motion
source and summarize the measured receipt; they are publication media, not
additional benchmark runs.
