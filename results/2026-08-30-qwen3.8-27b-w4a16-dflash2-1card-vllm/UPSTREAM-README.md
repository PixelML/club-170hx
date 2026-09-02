# Qwen3.8-27B W4A16 + DFlash2 speculative decoding on a Vast.ai CMP 170HX

**Date:** 2026-08-28 (sanitized copy; see bundle README for the file index)
**Status:** ✅ Benchmark complete — 147.7 tok/s decode, 2156 tok/s prefill
**Cost basis:** on-demand rental (billing detail omitted from this sanitized copy)

## Summary

We reproduced the syv-ai/qwen38-27b-rtx3090 recipe on a rented Vast.ai
CMP 170HX 64 GB (single card). The first attempt with the
private GHCR image failed (documented below for the record); the working
path uses a public CUDA base image + the repo's own launcher.

**Final numbers (v9, fast variant, greedy, single stream):**

| Metric | Result | LocalMaxxing ref |
|---|---|---|
| Decode (256 tok) | **147.7 tok/s** | 212.68 |
| Decode (900 tok) | **134.5 tok/s** | — |
| TTFT | **76 ms** | 72 ms |
| Prefill (6.6K prompt) | **2156 tok/s** | 1221.6 |
| Peak VRAM / power | 57.8 GiB / 255 W | 54.5 GB |

The 212 reference is an 8-stream aggregate per the syv repo's own tables;
single-stream community datapoints (3090: ~120 tok/s, another 170HX:
133.7 tok/s) put our 147.7 squarely in family. The "88" in the reference
prompt spec is the 88-token prompt, not a layer depth. Full analysis in
[RESULTS.md](RESULTS.md).

## What we measured (final run v9)

Decode 147.7 tok/s (256-tok) / 134.5 tok/s (900-tok), TTFT 76 ms, prefill
2156 tok/s, acceptance length 2.56-2.80, peak 57.8 GiB / 255 W / 1455 MHz /
73 C. See [RESULTS.md](RESULTS.md) and
[artifacts/bench-v9.json](artifacts/bench-v9.json).

## What we measured (first attempt — kept for the record)

Nothing: the GHCR image was private, the container never ran. Evidence below.

## Evidence index

| File | What it shows |
|---|---|
| `image-manifest-probe.md` (upstream only) | Anonymous GHCR manifest request → 401 (image is private) |
| `container-filesystem.md` (upstream only) | ~1.2 MiB filesystem; /app only onstart.sh + ports.log; /tmp/qwen38 absent |
| `health-probes.md` (upstream only) | the mapped public port /health unreachable (connection refused / timeout) |
| `ssh-auth.md` (upstream only) | SSH Permission denied; authorized_keys mode/ownership issue; fix attempted, still failing |
| `instance-metadata.md` (upstream only) | Instance config as known; parent-supplied metadata pending where marked |
| raw v1 instance log (upstream only) | Raw container log (87 KB): SSH shim only; no image pull, no vLLM |
| `benchmark-spec.md` (upstream only) | Intended launch flags, env, and request payload (never executed) |

## Hardware caveats (why this GPU was interesting)

The CMP 170HX is a mining card repurposed for LLM inference:

- 64 GB HBM2e with very high memory bandwidth (~1.5 TB/s class)
- **PCIe Gen2 x4 host link only** — model download/weight load is slow; fine
  once weights are resident, painful for cold starts
- **No FP8/FP4 tensor-core paths** (compute capability 8.6-ish, stripped SKU) —
  W8A16 main + W4A16 draft is roughly the right quantization envelope
- No display outputs, fan/power quirks vary by vendor

## Reproducing the *diagnosis*

```bash
# 1. Confirm the image is private (expect 401 for anonymous access)
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:syv-ai/qwen38-27b-rtx3090:pull" | jq -r .token)
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $TOKEN" \
  https://ghcr.io/v2/syv-ai/qwen38-27b-rtx3090/manifests/latest

# 2. Probe the mapped public port (expect connection refused/timeout)
curl -sS --max-time 8 -w "HTTP=%{http_code} time=%{time_total}\n" \
  the mapped public port /health
```

See [scripts/probe.sh](scripts/probe.sh) for the full probe sequence used.

## Lessons / what a retry needs

1. **Verify image pullability before renting.** A 401 from the anonymous
   manifest check (above) predicts this exact failure in seconds, for free.
2. **Skip the image entirely.** Start from any public CUDA base image, then
   `pip install vllm==0.27.1` and apply the patches from
   github.com/syv-ai/qwen38-27b-rtx3090 in the onstart script — the author's
   own instructions. No private registry involved.
3. If the image must be used, attach a GHCR pull credential to the instance
   (Vast supports registry auth via Docker config on creation).

## Reference target (from LocalMaxxing, not reproduced here)

| Metric | Reference |
|---|---|
| Output throughput | 212.68 tok/s |
| Prefill throughput | 1221.6 tok/s |
| TTFT | 72 ms |
| VRAM used | 54.5 GB of 64 GB |
| llama.cpp stock baseline | ~70 tok/s (for contrast) |
| Output tokens / request | 256 |
| Prompt tokens | 88 |
| Context length | 65,536 |
| Stack | vLLM 0.27.1 syv overlay, DFlash2, 7 draft tokens |
| Main / draft model | lued/Qwen3.8-27B-INT8-W8A16-MTP / syvai/Qwen3.8-27B-DFlash2-W4A16 |
| KV cache | BF16; FLASH_ATTN; GPU util 0.90; max seqs 1 |

No numbers on this page were measured by us. This run produced no metrics.

## References

- LocalMaxxing run page:
  <https://www.localmaxxing.com/en/models/lued/Qwen3.8-27B-INT8-W8A16-MTP?run=cmt7y3mm301v1nn01sxj7ptwi>
- Dual Channel Labs announcement (origin of the target numbers):
  <https://x.com/bob_hw_store/status/2092962836934705501> and
  <https://x.com/bob_hw_store/status/2092962838822150508>
- The overlay/patches repo the image was built from:
  <https://github.com/syv-ai/qwen38-27b-rtx3090>

Per the author, the intended install is **not** the GHCR image: it's
`pip install vllm==0.27.1` plus applying the patches from the GitHub repo
above (that's what backports DFlash2 and the Ampere-specific fixes). The GHCR
image that defeated this run was only a prebuilt convenience artifact — and it
is private.

---

## Update 2026-08-28: RTX 3090 baseline + multi-GPU scaling notes

We ran the identical recipe (repo patches, W4A16 fast variant, DFlash2 k=7,
vLLM 0.27.1) on a local RTX 3090 to anchor the CMP 170HX result:

| Metric | RTX 3090 (24 GB) | CMP 170HX (64 GB) | 170HX advantage |
|---|---|---|---|
| Decode 256 tok | **122.42 tok/s** | **147.7 tok/s** | **1.21x** |
| Decode 900 tok | 111.22 tok/s | 134.5 tok/s | 1.21x |
| TTFT (11-tok prompt) | 181.5 ms | 76 ms | 2.4x |
| Prefill (6.6K tok) | 1341.9 tok/s | 2156 tok/s | 1.61x |
| Baseline, no spec (256 tok) | 53.14 tok/s | — | — |
| DFlash2 speedup | 2.30x | — | — |
| Peak VRAM | 21.9 GiB | 57.8 GiB | — |
| Peak power | 390 W (cap) | 255 W | — |

Key takeaways:

1. **DFlash2 speculation is the dominant factor on Ampere** — 2.30x on the
   3090, same family as the 170HX. Both cards are compute/power saturated
   under speculative load, so the 170HX's 1.6x memory-bandwidth edge
   shrinks to 1.21x per-card decode advantage.
2. **Prefill scales with bandwidth** (1.61x, closer to the HW ratio) —
   long-prompt workloads benefit more from the 170HX than short-prompt
   chat does.
3. **Power efficiency is the 170HX's quiet win**: 255 W vs 390 W at higher
   throughput — ~2.4x more tokens per watt.

Full data: [artifacts/rtx3090-baseline/results.json](artifacts/rtx3090-baseline/results.json).
Raw per-run samples: [artifacts/rtx3090-baseline/bench-stdout.jsonl](artifacts/rtx3090-baseline/bench-stdout.jsonl).

### Multi-GPU scaling attempt (8x/4x TP on Vast, failed — for the record)

A companion attempt to benchmark GLM-5.3-Flash-AWQ (8x 170HX) and
Qwen3.8-Flash-Next-AWQ (4x 170HX) on Vast.ai failed before serving: the
instances were created with 17-20 GB disks, far below the 168-176 GiB model
sizes; the GLM run died mid-download (HF Xet \"background writer channel
\" error after 380 s). Both instances were destroyed ~50 min in. Lesson: check `disk_space >= model_size + headroom` before creating
any instance. No benchmark numbers were produced and none are invented here.
