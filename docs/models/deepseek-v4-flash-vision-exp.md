# DeepSeek-V4-Flash-Vision-Exp on CMP 170HX

`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` is DeepSeek's vision-capable Flash checkpoint: FP8 e4m3 weights, 48 shards, about 156 GiB. On this club's 4-card CMP 170HX rig it serves both text and images from one SM80 vLLM fork, pipeline parallel 4, DSpark speculative decoding at k=6. The text path is a measured performance benchmark (220.2 tok/s aggregate decode at c=8). The vision path is a measured but partial benchmark: functional gates and image correctness pass, and the concurrency ladder tops out at c=2 before the server's known crash point at c=4. A separate reference TP4 runtime, kept as history, produced the first real-image completion of this checkpoint on Ampere hardware at about 0.9 tok/s — a correctness result, not a speed result.

Read this page for the settings and commands. Read the executed notebooks for the full protocol and raw receipts.

## Run on CMP 170HX

| Cards | VRAM/card | Format | Runtime | Image tag | Partition | Speculative k | KV cache | Context | Measured decode | Status |
|---|---:|---|---|---|---|---:|---|---:|---|---|
| 4 | ~51–61 GiB under load | FP8 e4m3 | SM80 vLLM fork, PP4 | `vllm-deepseek-v4-sm80-20260902` | `11,11,11,10` | 6 (DSpark) | fp8 | 16,384 | 97.4 tok/s c=1, 220.2 tok/s c=8 (median of 3) | Measured, text path |
| 4 | ~51–61 GiB under load | FP8 e4m3 | SM80 vLLM fork, PP4, vision-enabled (Path 3) | `vllm-deepseek-v4-vision-sm80-20260902` | `11,11,11,10` | 6 (DSpark) | fp8 | 16,384 | 119 tok/s median of 5 reps (peak 162), c=1 text-only; 45.3 tok/s c=1 text+image | Measured, partial — stable to c=2, crashes at c=4 to c=8 |
| 4 | ~44.4 GiB weights resident | FP8 → BF16 fallback | Reference TP4 runtime + SM80 patches | — (source build, no published image) | TP4 | none | BF16 | ≤ ~1,024 input tokens (OOM above) | ~0.9 tok/s, batch 1 | Correctness only, history |
| 3 | untested | FP8 e4m3 | SM80 vLLM fork, PP3 | — | `15,15,13` | 5 (DSpark) | fp8 | 16,384 | untested on this checkpoint (see `deepseek-v4-flash-0731.md` for the sibling checkpoint's 3-card numbers) | Untested |

Only `11,11,11,10` boots cleanly on 4 cards. Two other partitions failed before serving traffic: `12,12,12,7` exits 137, `12,12,11,8` hits a device-side assert on the first request.

## Quick start

### 1. Pull the image

```bash
docker pull ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902
```

For the vision-enabled build, pull the vision tag instead:

```bash
docker pull ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902
```

### 2. Download the weights, pinned to the exact revision

```bash
pip install -U huggingface_hub
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --revision 86f746b36186f0e567729a5c06a8c918caba82a9 \
  --local-dir <weights>
```

Verify shard count (48) and total bytes against a manifest before serving. Do not trust an unverified cache — see Troubleshooting below for why.

### 3. Launch — text path

```bash
docker run -d --name <container> --gpus '"device=0,1,2,3"' \
  -e VLLM_PP_LAYER_PARTITION=11,11,11,10 \
  -v <model-storage>/deepseek-v4-flash-vision-exp@86f746b3:/model:ro \
  --shm-size=16g -p 18098:8000 \
  ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902 \
  vllm serve /model --served-model-name dsv4v \
  --pipeline-parallel-size 4 --kv-cache-dtype fp8 \
  --block-size 256 --max-model-len 16384 --max-num-batched-tokens 2048 \
  --max-num-seqs 8 --tokenizer-mode deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":6}'
```

This image serves text only. It does not carry the vision encoder.

### 4. Launch — vision path (confirmed working, boot attempt 5)

```bash
docker run -d --name <container> --gpus '"device=0,1,2,3"' \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=11,11,11,10 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  -v <model-storage>/deepseek-v4-flash-vision-exp@86f746b3:/model:ro \
  --shm-size=16g -p 18099:8000 \
  ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902 \
  vllm serve /model --served-model-name deepseek-v4-flash-vision-exp \
  --pipeline-parallel-size 4 --kv-cache-dtype fp8 \
  --block-size 256 --max-model-len 16384 --max-num-batched-tokens 2048 \
  --trust-remote-code --gpu-memory-utilization 0.90 --max-num-seqs 8 \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":6}' \
  --hf-overrides '{"architectures":["DeepseekV4ForConditionalGeneration"]}' \
  --limit-mm-per-prompt '{"image": 2}'
```

Expect a 30–45 minute cold boot from network storage, or 8–15 minutes once weights are staged on local NVMe (see Troubleshooting).

### 5. First request — text

```bash
curl http://localhost:18098/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4v",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "temperature": 0,
    "max_tokens": 8
  }'
```

### 6. First request — text and image

```bash
curl http://localhost:18099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-vision-exp",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Name the two colors this gradient blends between, left color first."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_IMAGE>"}}
      ]
    }],
    "temperature": 0,
    "max_tokens": 32
  }'
```

A 64x64 gradient image, encoded as a `data:image/png;base64,...` URL, is what the vision golden corpus uses. This is the request shape that returned `"Red, green"` on the measured 2026-09-02 run.

## Recommended settings

- **Sampling for correctness checks:** `temperature=0`, greedy, so output is comparable to the golden corpus.
- **Sampling for throughput benches:** greedy decoding, `ignore_eos=true`, 400 completion tokens per request, three repetitions per concurrency level, one warmup rep discarded. Count tokens from the final `usage.completion_tokens` object, not from streamed events.
- **Prompt limits:** the measured prefill benchmark uses a 2,941-token uncached prompt. The reference TP4 runtime OOMs above roughly 1,024 input tokens (see Troubleshooting); the vLLM PP4 path does not share that ceiling.
- **Image limits:** the vision launch command sets `--limit-mm-per-prompt '{"image": 2}'`. The measured golden corpus uses one image per request.
- **Concurrency, vision build:** stable through c=2 on the vision-enabled image; the crash point sits somewhere in c=4 to c=8 and is tracked, not fixed. A 180 W-cap run stayed up through c=4 and crashed mid-c=8; a 250 W-cap run of the same recipe crashed earlier, at c=4 on warmup. Both crashes are the same `EngineCore` failure (`RuntimeError: cancelled` in the shared-memory broadcast queue, `shm_broadcast.acquire_read`). Keep concurrency at c=2 or below until this is fixed. Text+image concurrency above c=2 was not attempted given that crash history. The text-only image (no vision encoder) has its own, separate ceiling at c=16.
- **Power cap:** 250 W per card buys no measured throughput gain over 180 W at c=1/c=2 (+1% to +5%, inside run-to-run noise) and is flat to worse on tokens/Wh; this decode workload is latency-bound at low concurrency, not power-bound. 180 W is the standing default. Detail: `docs/BENCHMARKS.md`, "Reproducibility and power cap."
- **c=1 throughput spread:** a 5-rep check found the text-only c=1 aggregate swinging 48.5-161.7 tok/s run to run (median 119, peak 162), tracked to DSpark draft-acceptance ratio swings (0.20-0.83), not to the power cap. Quote the median and range, not a single point estimate.
- **Context:** `--max-model-len 16384`, `--max-num-batched-tokens 2048`. Not tested past this window on this checkpoint.
- **KV cache:** `fp8`. Cuts memory versus BF16 KV at some decode/prefill cost — see `docs/LESSONS.md` section d for the measured trade on a sibling checkpoint (Qwen3.8-27B): -3.4% decode, -19.5% prefill.

## Troubleshooting

**1. Eager weight load exhausts host RAM.** The default eager `safetensors` load reads each shard whole into host RAM. Four pipeline ranks loading a 156 GiB checkpoint concurrently exhausted a 94 GiB host and the kernel silently killed workers at 7 of 48 shards, no traceback. Fix: use the default streaming `safe_open` load instead of `--safetensors-load-strategy eager`.

**2. Missing multimodal plan API.** The pinned model code's processor calls `_plan_prompt_updates`, a method this vLLM fork does not carry. Fix: a compatibility shim in `vllm/multimodal/processing/processor.py`.

**3. Processor never returns `input_ids`.** `DeepseekV4VLProcessor.__call__` tokenized nothing, causing `KeyError: 'input_ids'` during profiling. Fix: override `_call_hf_processor` to tokenize and merge `input_ids` in, and override `_hf_processor_applies_updates` to return `False`.

**4. Broadcast weights never finalized.** `DeepseekV4ForCausalLM.load_weights()` skipped `process_weights_after_loading()`, so `hc_attn_fn_broadcast` stayed `None` and image profiling tripped an assertion. Fix: call it, matching the pinned reference's own load order.

**5. CUDA-graph capture nulls `input_ids` on non-first ranks.** DeepSeek V4 Vision's MoE router needs raw `input_ids` on every rank (`requires_raw_input_tokens = True`); the graph-capture path did not carry the same guard the normal forward path had. Fix: added the guard to `cudagraph_utils.py`. This was the fifth and last fix before the vision build reached READY.

**Eager vs. streaming load.** Prefer the default streaming `safe_open` load over `--safetensors-load-strategy eager` on any multi-rank boot; the eager path is a host-RAM-exhaustion hazard proportional to rank count, not just model size.

**NFS vs. NVMe boot times.** A cold boot from NFS-backed shared storage measured 42 minutes (2,515 s: 1,146 s for ranks 0–2, 2,270 s for rank 3, which also carries the draft head). NFS throughput measured about 31 MiB/s aggregate. Staging the same checkpoint on local NVMe cuts this to roughly 8–15 minutes.

**`--gpus device=...` list after a crash.** After an OOM crash, `--gpus all` can assign zero devices to a new container — stale cgroup state left over from the crash. Use an explicit device list, `--gpus '"device=0,1,2,3"'`, instead of `all`.

**Driver recovery after an OOM storm.** A multi-rank OOM kill can leave the kernel log showing NVRM assertion failures on every GPU, with `cuInit` returning `CUDA_ERROR_NO_DEVICE` host-wide. Reloading `nvidia_uvm` alone does not clear it. The full sequence — `rmmod nvidia_uvm nvidia` then `modprobe nvidia nvidia_uvm` — restores all devices without a VM reboot.

**2,941-token prefill OOM on the reference TP4 runtime.** The reference runtime (not the vLLM path) needs 6.8 GiB per rank for a 2,941-token prompt and 7.8 GiB for a 2,048-token prompt; both exceeded free memory with about 44.4 GiB of weights already resident per card. Reliable single-request prefill on that runtime stays below about 1,024 tokens. This limit does not apply to the vLLM PP4 path, whose measured prefill benchmark uses the full 2,941-token prompt without issue.

**c=4 crash on the vision build.** At c=4, rep 3 of 3, the vLLM `EngineCore` process died mid-batch: `RuntimeError: cancelled` from `shm_broadcast.py`, `acquire_read`. All 4 in-flight requests got `HTTP 500`; the container exited cleanly (code 0). GPUs stayed at 42–47°C through the crash, no Xid or ECC events. Per this club's standing operating instruction, the session did not restart the server automatically — restart it manually and stay at c=2 or below.

## Benchmarks

### Text path, four cards, PP4 + DSpark k=6

Greedy decoding, 400 completion tokens, `ignore_eos`, one warmup plus three reps per level, tokens counted from the final `usage` object. Measured 2026-09-02.

| Concurrency | Aggregate decode | Notes |
|---|---:|---|
| c=1 | 97.4 tok/s (median of 3; range 57.6–123.5) | |
| c=2 | 103.7 tok/s (median of 3; range 96.6–159.2) | |
| c=4 | 165.5 tok/s (median of 3; range 140.3–203.2) | |
| c=8 | 220.2 tok/s (median of 3; range 206.3–232.0) | Best measured aggregate |
| c=16 | Failed | Device-side assert in the draft-decode path, reproduced twice |

Uncached prefill, 2,941 input tokens: 2,352 tok/s warm (362 tok/s first cold prefill). Warm TTFT: 0.394 s.

### Vision path, four cards, PP4 + DSpark k=6 (Path 3)

Measured 2026-09-02, partial.

| Metric | Value |
|---|---:|
| Functional gates | PASS, 3/3 identical reps |
| Golden corpus, image rows (10) | PASS, 10/10 keyword match |
| Golden corpus, text rows (20) | 15/20 keyword match, 10/20 exact-match vs. DGX Spark reference |
| Decode, c=1, text-only | 119 tok/s median of 5 reps (peak 162), aggregate |
| Decode, c=2, text-only | 116.57 tok/s aggregate (median of 3) |
| Decode, c=4, text-only | FAIL — server crashed, rep 3 of 3 |
| Decode, c=8 / c=16, text-only | Not measured |
| Decode, c=1, text+image | 45.32 tok/s aggregate (median of 3) |
| Decode, c=2, text+image | 78.23 tok/s aggregate (median of 3) |
| Decode, c=4 and above, text+image | Not attempted, given the c=4 text-only crash |
| Uncached prefill, 2,941 tokens | 2,352.42 tok/s (median of 3) |
| Warm streaming TTFT | 0.386 s (median of 3) |

### Long context, max-model-len 262,144 (measured 2026-09-02)

Same recipe (4x CMP 170HX, PP4, DSpark k=6, fp8 KV), relaunched with
`--max-model-len 262144`. Prefill ladder, greedy, `max_tokens=1`, one
warmup plus three reps per level, unique prompt prefix per rep.

**Power cap: 250 W.** This ladder ran after a card reboot and before the
180 W cap was re-applied. The numbers below are 250 W; 180 W re-measure
pending.

| Prompt tokens | Status | Median wall time (s) | Median prefill tok/s |
|---:|---|---:|---:|
| 2,941 | PASS | 1.24 | 2,397 |
| 16,000 | PASS | 3.43 | 4,665 |
| 32,000 | PASS | 6.18 | 5,182 |
| 65,000 | PASS | 12.36 | 5,261 |
| 131,000 | FAIL — engine crash | — | — |
| 200,000 / 250,000 | Not reached | — | — |

| Item | Value |
|---|---:|
| KV pool at boot | 1,621,821 tokens |
| Reported max concurrency at 262,144 tokens/request | 6.19x |
| Largest verified passing prompt | 65,000 tokens |
| Needle-in-haystack (32k / 65k, three depths each) | Untested — server crashed before any needle request ran |
| Long-context decode (C1/C2) | Untested — same cause |
| Vision at 131k context | Not attempted — gated on the 131k prefill rung, which failed |

The engine died while the harness built the 131,000-token fixture: a Triton
kernel inside the DSpark/DFlash speculator's input-preparation step
(`prepare_dflash_inputs`, `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`)
raised `RuntimeError: Triton Error [CUDA]: an illegal memory access was
encountered` on the PP3 (drafter) rank, which cascaded to an
`EngineDeadError` and a clean container exit. Peak temperature during the
run was 51°C, and no Xid or ECC events appeared — this is a stability
failure, not a thermal or hardware fault. Per the run's operating
authorization, the server was not restarted after the crash, so all phases
that need prompt lengths at or above 131,000 tokens, or a live server after
the crash, stay untested. Full writeup, verbatim crash excerpt, and chart:
`results/2026-09-02-deepseek-v4-flash-vision-exp-4card-longctx-262k/`.

### Reproducibility and 180 W vs. 250 W (measured 2026-09-02)

A follow-up 5-rep check on the text-only c=1 recipe above: 118.95 tok/s
median at 180 W (range 48.5-161.7), 120.42 tok/s median at 250 W (range
86.9-154.7) — statistically indistinguishable at this concurrency. The wide
range is driven by DSpark draft-acceptance ratio swinging 0.20-0.83 across
reps, not by the power cap. 250 W bought +3.4% more measured active-load
power and flat-to-worse tokens/Wh; concurrency stayed stable to c=2 on both
arms, with 180 W crashing mid-c=8 and 250 W crashing earlier at c=4 (same
`EngineCore`/`shm_broadcast` failure). Full protocol, both-protocol table,
and chart: `docs/BENCHMARKS.md`, "Reproducibility and power cap"; raw
receipts: `results/2026-09-02-deepseek-v4-flash-vision-exp-4card-repro-power/`.

### Reference TP4 runtime — vision correctness milestone (history)

Not a serving benchmark. Batch 1, no streaming, no speculative decoding.

| Metric | Value |
|---|---:|
| Decode, c=1, greedy 401 tokens, 3 reps | 0.88–0.93 tok/s |
| Wall time per 401-token completion | 431–454 s |
| Prefill, 512 input tokens | 8.7–9.8 s wall |
| Prefill, 256 input tokens | 7.4–7.5 s wall |
| Weights resident per card | ~44.4 GiB |

Real-image completion: PASS. The model named colors present in a 64x64 gradient image and absent from the text prompt. No-image and wrong-image controls also passed.

### Cross-platform context (not a head-to-head)

The same checkpoint at the same revision also runs on a two-node DGX Spark kit (GB10, vLLM, TP=2): c=1 36.9 tok/s, c=6 112.7 tok/s aggregate, uncached prefill 1,789 tok/s, vision PASS. Full numbers and cost/watt comparison: `docs/BENCHMARKS.md#cross-platform-4x-cmp-170hx-vs-2x-dgx-spark`. The two platforms differ in runtime, parallelism, and memory budget — read this as context, not a controlled comparison.

## Artifacts

- **GHCR images:** `ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902` (text path), `ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902` (vision path). A pinned digest for a related club image: `ghcr.io/pixelml/club-170hx@sha256:90a1419e8ceaad3542153ef4e2a1d94a69b9af03cce7b0a1b267dd1dad55b9d7` — pin the digest for the tag you pull before relying on it in production.
- **Evidence repository:** [PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) — full receipts, launch scripts, and patches.
- **Hugging Face collection:** [PixelML/club-170hx: verified on CMP 170HX (SM80)](https://huggingface.co/collections/PixelML/club-170hx-verified-on-cmp-170hx-sm80-6a97bf4edc20b52c5cf454e3).
- **Checkpoint:** [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp), revision `86f746b36186f0e567729a5c06a8c918caba82a9`.
- **Patch series:** the five vision boot fixes and the SM80 fallback patches build on [allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80) and [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx). Full patch detail: `docs/LESSONS.md` section f.1.
- **Executed notebooks:** [text path](../../notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb), [vision path](../../notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb).

## Changelog

- **2026-09-02** — Long-context check at `max-model-len 262144`: prefill
  ladder passes cleanly through 65,000 prompt tokens, then the engine
  crashes on a Triton fault in the DSpark/DFlash speculator while building
  the 131,000-token case. Not thermal, not Xid. Needle, decode, and vision
  phases above 65k stay untested; server left down (not restarted, per
  operating authorization). See "Long context, max-model-len 262,144" above.
- **2026-09-02** — Reproducibility and power-cap check on the text-only c=1
  recipe: 5 reps put the median at 119 tok/s (peak 162), correcting the
  earlier 163.1 tok/s median-of-3 figure, which was a peak-adjacent sample,
  not a stable central tendency. 250 W per card measured no throughput gain
  over 180 W; 180 W kept as the standing default. Concurrency ceiling
  narrowed to c=4-c=8 depending on power cap (previously reported only at
  c=4 for this build). Detail: `docs/BENCHMARKS.md`, "Reproducibility and
  power cap."
- **2026-09-02** — Vision path measured on the SM80 vLLM PP4 fork (Path 3): functional gates and image correctness pass, text-only ladder crashes at c=4, text+image ladder measured through c=2. Text path re-measured on the normalized protocol through c=16 (220.2 tok/s aggregate at c=8, device-side assert at c=16). Published as release [`dsv4-vision-exp-4card-2026-09-02`](https://github.com/PixelML/club-170hx/releases/tag/dsv4-vision-exp-4card-2026-09-02), a full release, not a pre-release. Two result videos are attached: [text-only motion](https://github.com/PixelML/club-170hx/releases/download/dsv4-vision-exp-4card-2026-09-02/dsv4-vision-4card-motion-1920x1080.mp4) and [text+image motion](https://github.com/PixelML/club-170hx/releases/download/dsv4-vision-exp-4card-2026-09-02/dsv4-vision-4card-vision-motion-1920x1080.mp4).
- **2026-08-31** — Earlier text-path ladder and single-stream run, superseded by the 2026-09-02 normalized text benchmark. Numbers kept in `docs/BENCHMARKS.md` as historical, not current.
- **2026-08-31 (approx.)** — Reference TP4 runtime achieves the first real-image completion of this checkpoint on Ampere hardware, at about 0.9 tok/s. Kept as the correctness-only history milestone.
