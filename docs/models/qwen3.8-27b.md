# Qwen3.8-27B on CMP 170HX

`Qwen3.8-27B`, W4A16-AutoRound quantized with DFlash2 speculative decoding, is the fastest working recipe this club has measured on a single CMP 170HX card: 140.3 tok/s decode at a 180 W local power cap, 95% of a 255 W rented card's decode rate at 71% of its power draw. The card is bandwidth-bound, so the denser W8A16 checkpoint measured slower (31.6 tok/s), not faster. Evidence lives in [PixelML/Qwen3.8-27B-CMP-170HX](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX); the sanitized receipt chain for the three-card local runs is still an open PR there, so the numbers below are platform context until that repair lands.

## Run on CMP 170HX

| Cards | VRAM/card | Format | Runtime | Power | k (DFlash2) | KV cache | Context | Measured decode | Status |
|---|---:|---|---|---:|---:|---|---:|---|---|
| 1 (rented) | ~57.8 GiB peak | W4A16-AutoRound ("fast variant") | vLLM 0.27.1 + DFlash2 | 255 W | 7 | BF16 | 65,536 | 147.7 tok/s @ 256 tok, 134.5 tok/s @ 900 tok | Measured |
| 3 (local, one card each) | untested | W4A16-AutoRound (dbirks) | vLLM 0.27.1 + DFlash2 | 180 W | 7 | BF16 | untested | 133.6–140.3 tok/s decode, 136.38 tok/s mean | Measured 2026-08-30, pending evidence repair |
| 1 (rented) | untested | W8A16 (official checkpoint) | vLLM 0.27.1 + DFlash2 | 255 W | 7 | BF16 | untested | 31.6 tok/s decode, acceptance 2.9 | Measured, negative vs. W4A16 |
| 1 | untested | W4A16-AutoRound | vLLM 0.27.1 + DFlash2, FP8 KV | 180 W | 7 | FP8 | 131,072 | -3.4% decode, -19.5% prefill vs. BF16 KV at the same context | Measured trade-off, not a standalone recipe |

No club-published GHCR image exists for this recipe as of this writing; the reference run used a public CUDA base plus `pip install vllm==0.27.1` and source patches, not a prebuilt club image. Treat the docker step below as the pattern this club would ship, not a pulled artifact.

## Quick start

### 1. Build or pull a vLLM 0.27.1 image with the DFlash2 patch set applied

No club GHCR tag is published for this recipe. Reproduce it from a public CUDA base:

```bash
docker pull nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04
# then: pip install vllm==0.27.1 and apply the DFlash2 patches from the
# upstream syv-ai recipe referenced in the evidence repository.
```

### 2. Download the weights

```bash
pip install -U huggingface_hub
hf download dbirks/Qwen3.8-27B-W4A16-AutoRound --revision <pin-before-use> --local-dir <weights>
hf download syvai/Qwen3.8-27B-DFlash2-W4A16 --revision <pin-before-use> --local-dir <draft>
```

`dbirks/Qwen3.8-27B-W4A16-AutoRound` is the main checkpoint for this recipe (the onstart scripts in the evidence repository confirm it; `lued/Qwen3.8-27B-INT8-W8A16-MTP` is a different, unrelated checkpoint and does not apply here). Neither the base nor the draft `hf download` call in the onstart scripts pins a revision — pin one from your own `hf download` output before relying on it.

Package versions measured for this recipe: vLLM 0.27.1, torch 2.13.0, flashinfer 0.6.16.post3, flashinfer-cubin 0.6.13.

### 3. Launch

```bash
docker run -d --name <container> --gpus '"device=0"' \
  -v <weights>:/model:ro -v <draft>:/draft:ro \
  --shm-size=8g -p 18020:8000 \
  <your-image> \
  vllm serve /model --served-model-name qwen3.8-27b \
  --speculative-config '{"method":"dflash2","draft_model":"/draft","num_speculative_tokens":7}' \
  --kv-cache-dtype bf16 --gpu-memory-utilization 0.90 --max-num-seqs 1 \
  --max-model-len 65536
```

### 4. First request

```bash
curl http://localhost:18020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "temperature": 0,
    "max_tokens": 8
  }'
```

Qwen3.8-27B is text-only on this recipe; no image-input example applies.

## Recommended settings

- **Sampling for throughput benches:** greedy decoding, fixed output lengths (256 and 900 tokens measured), tokens counted from the final usage object.
- **Speculative decoding:** DFlash2 at `k=7`. Measured acceptance length 2.56–2.80 (255 W rented) and 2.85–3.32 (180 W local, better than the rented run). A 2.30x decode speedup over plain decoding was measured on the RTX 3090 baseline at the same recipe.
- **KV cache:** BF16 is the default and the faster choice. FP8 KV was measured at -3.4% decode and -19.5% prefill versus BF16 KV at 131k context on the same server — FP8 KV buys long context, it does not come free.
- **Power:** 180 W local reaches 95% of the 255 W rented card's decode rate at 71% of the power draw. Prefill reaches 91% of the 255 W rate at 180 W. TTFT is worse locally (about 190 ms vs. 76 ms), attributed to the PCIe Gen2 x4 host link, not the power cap.
- **Context:** 65,536 measured on the rented single-card run; 131,072 measured for the FP8-KV comparison. Not tested beyond that on this checkpoint.
- **Concurrency:** every run in this guide used `--max-num-seqs 1`. Multi-stream throughput is untested for this recipe.

## Troubleshooting

**Private GHCR image returns 401.** An earlier attempt tried a prebuilt image (`syv-ai/qwen38-27b-rtx3090`) that turned out to be private: an anonymous manifest pull returned 401, and the container held no model. Check image pullability with an anonymous token request before renting a GPU around an image you have not verified is public.

**SSH auth failure from an `authorized_keys` permission problem.** Blocked the first rented-node attempt entirely. Check key file permissions (`600`) and ownership before assuming a network or firewall issue.

**Multi-GPU attempts died on disk size, not compute.** Two unrelated multi-GPU runs (GLM-5.3-Flash-AWQ at 8x, Qwen3.8-Flash-Next-AWQ at 4x) failed because rented instances shipped 17–20 GiB disks against 168–176 GiB models. Check disk size against model size before renting, independent of GPU count or VRAM.

**A surprising low-power reading was contamination, not a real 3x power penalty.** A single-card run once reported 47.0 tok/s at 180 W, read at the time as "the power cap costs 3x." The number was contaminated by an overlapping model build on the same host and a pinned KV-cache allocation left over from a different card's default. A clean rerun at the same 180 W cap reached 135.3–140.3 tok/s — the real 180 W vs. 255 W gap is about 5%. Re-verify any surprising power number before trusting it.

## Benchmarks

### Single card, 255 W rented (measured)

| Metric | Value |
|---|---:|
| Decode, 256 output tokens | 147.7 tok/s |
| Decode, 900 output tokens | 134.5 tok/s |
| TTFT | 76 ms |
| Prefill, 6.6k-token prompt | 2,156 tok/s |
| Peak VRAM | 57.8 GiB |
| Peak power / clock / temp | 255 W / 1,455 MHz / 73°C |
| Acceptance length | 2.56–2.80 |

### Three cards, 180 W local, one card each (measured 2026-08-30, pending evidence repair)

| Card | Decode, 256 tok | Decode, 900 tok | TTFT | Prefill |
|---|---:|---:|---:|---:|
| 0 | 135.31 tok/s | 121.28 tok/s | 201.2 ms | 1,957.3 tok/s |
| 1 | 140.27 tok/s | 124.78 tok/s | 189.7 ms | 1,954.7 tok/s |
| 2 | 133.57 tok/s | 119.94 tok/s | 181.4 ms | 1,926.0 tok/s |
| Mean | **136.38 tok/s** | **122.00 tok/s** | **190.8 ms** | **1,946.0 tok/s** |

Peak core temperature 51°C, peak memory temperature 61°C across the three runs.

### RTX 3090 baseline, same recipe (measured, comparison only)

122.42 tok/s decode (256 tok), 111.22 tok/s (900 tok), TTFT 181.5 ms, prefill 1,341.9 tok/s, 21.9 GiB VRAM, 390 W. The CMP 170HX measured about 1.21x faster on decode and 1.61x faster on prefill at the same recipe, and roughly 2.4x better tokens per watt.

## Runtime A/B: vLLM vs. Ninfer sm_80 fork

The `Ithrial/ninfer-cmp170hx` sm_80 fork ships its own speculative-decoding
path (MTP, `--draft-tokens 3 --lm-head-draft`) and was worth a controlled
comparison against this page's vLLM + DFlash2 recipe on the same card. A
prior attempt had crashed at warmup; this run checked whether that was fixed
before measuring anything.

| Metric | vLLM + DFlash2 (control) | Ninfer spec-on (MTP) | Ninfer spec-off |
|---|---:|---:|---:|
| decode256 | 138.6 tok/s | 38.16 tok/s | 29.95 tok/s |
| decode900 | 123.4 tok/s | 39.15 tok/s | 29.55 tok/s |
| Peak power | 190.5 W | 195.9 W | 203.0 W |
| Peak SM clock | 1170-1200 MHz sustained (1470 MHz sampled peak) | 1455 MHz | 1455 MHz |

Full protocol, per-sample data, and the bandwidth-ceiling estimate:
[results/2026-09-02-qwen3.8-27b-ninfer-ab](../../results/2026-09-02-qwen3.8-27b-ninfer-ab/README.md).

**Verdict: stay on vLLM + DFlash2.** Ninfer's own MTP speculation is
functioning — it gives a real ~1.28x uplift over its own spec-off run — but
Ninfer's base per-pass throughput on this build is far below vLLM's, not a
spec-acceptance problem. Spec-on is 3.6x slower than the control; spec-off
is 4.6x slower. The prior warmup crash did not recur on this attempt, so the
build is at least stable enough to bench, just not fast enough to recommend.

One open question, not folded into the speed verdict above: Ninfer's peak
power reading (195.9-203.0 W) came in above the 180 W cap configured on the
card, at a higher clock than the control run's sustained clock. This looks
like a difference in how the two engines interact with the driver's power
limit and is worth a follow-up, not a claim about either engine's speed.

## Artifacts

- **Evidence repository:** [PixelML/Qwen3.8-27B-CMP-170HX](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX).
- **GHCR image:** none published for this recipe as of this writing.
- **Checkpoints:** [dbirks/Qwen3.8-27B-W4A16-AutoRound](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound) (main), [syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16) (draft) — pin an exact revision before use; neither onstart script in the evidence repository records one.
- **Package pins:** vLLM 0.27.1, torch 2.13.0, flashinfer 0.6.16.post3, flashinfer-cubin 0.6.13.
- **Patch series:** DFlash2 patches from the syv-ai reference stack ([syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)), applied at container start.

## Changelog

- **2026-09-02** — Runtime A/B against the Ninfer sm_80 fork measured: 3.6x slower on decode with the fork's own MTP speculation on, 4.6x slower with it off; verdict is to stay on vLLM + DFlash2.
- **2026-08-30** — Three-card local runs at 180 W measured (136.38 tok/s mean decode at 256 tokens); sanitized receipt chain still pending merge in the evidence repository.
- **2026-08-28** — Single rented-card run at 255 W measured (147.7 tok/s decode); first-attempt private-image and SSH failures documented.
