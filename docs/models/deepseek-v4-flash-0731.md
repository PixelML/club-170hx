# DeepSeek-V4-Flash-0731 on CMP 170HX

`deepseek-ai/DeepSeek-V4-Flash-0731` ships native MXFP4 experts plus FP8 e4m3 attention, 48 shards, about 148 GB. This club's own 3-card measurement reached 83.3 tok/s aggregate decode at 180 W with DSpark speculative decoding at k=5. A separate community 4-card reference run (not this club's own, from allover326) reached 98.1 tok/s and verified the full 1,047,736-token context window. This checkpoint is also the baseline the vision-capable `DeepSeek-V4-Flash-Vision-Exp` fork builds on — see `deepseek-v4-flash-vision-exp.md` for the vision recipe.

## Run on CMP 170HX

| Cards | VRAM/card | Format | Runtime | Partition | k (DSpark) | KV cache | Context | Measured decode | TTFT | Status |
|---|---:|---|---|---|---:|---|---:|---|---|---|
| 3 (180 W, local) | untested | MXFP4 experts + FP8 e4m3 | SM80 vLLM fork (source build), PP3 | `15,15,13` | 5 | fp8 | 16,384 | 83.3 tok/s aggregate (technical 73.4 / prose 72.4 / code 116.6) | untested | Measured, pending evidence repair |
| 4 (community reference) | untested | Same | SM80 vLLM fork, PP4 | untested split | 5 | fp8 | 1,047,736 (full context verified) | 98.1 tok/s aggregate | Measured, upstream reference — not this club's own run |
| 4 (PP4, acceptance study) | untested | Same | SM80 vLLM fork, PP4 | untested split | 5 | fp8 | untested | Not a throughput row; acceptance length 3.03 (per-position 0.730/0.569/0.372/0.226/0.131) | Measured, from `docs/LESSONS.md` |
| 4 (PP4, k=7 comparison) | untested | Same | SM80 vLLM fork, PP4 | untested split | 7 | fp8 | untested | 60.3 tok/s aggregate — worse than k=5; acceptance never extends past ~3 tokens | Measured, negative result: do not raise k above the checkpoint's block size |

`num_speculative_tokens` must be at least the checkpoint's `dspark_block_size` (5, for this checkpoint). Below it, output is garbled, not merely lower-acceptance. Above it measures worse, not better — see the k=7 row.

## Quick start

### 1. Build the image

No prebuilt club GHCR image is published for this checkpoint. The precompiled vLLM wheel (`VLLM_USE_PRECOMPILED=1`) ships without `vllm._C`, the custom op this SM80 fork's patches need — the engine fails at CUDA-graph capture, not at import, if you try to skip the build. A full source build is required:

```bash
export TORCH_CUDA_ARCH_LIST=8.0
# build from nvidia/cuda:13.0.2-cudnn-devel or an equivalent real CUDA
# toolkit image; pip CUDA wheels alone leave an nvcc/FlashInfer header
# mismatch. Measured build time: ~62 min compile + ~13 min export, ~78 min
# total wall time from scratch.
```

### 2. Download the weights

```bash
pip install -U huggingface_hub
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 --local-dir <weights>
```

No revision hash was recorded in the launch script or receipts for the 2026-08-30 run. The hash above (`7872f01b1d1fe23eabc4c98b48bffcef5a386062`) is resolved after the fact: it is the newest commit on the `main` branch on Hugging Face at or before the run date, dated 2026-08-01, per the [HF Hub commits API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731/commits/main). It is a reconstruction, not a value captured at run time — pin your own revision from a fresh `hf download` and compare.

### 3. Launch — 3 cards, PP3

```bash
docker run -d --name <container> --gpus '"device=0,1,2"' \
  -e VLLM_PP_LAYER_PARTITION=15,15,13 \
  -v <weights>:/model:ro --shm-size=16g -p 18010:8000 \
  <your-built-image> \
  vllm serve /model --served-model-name dsv4-0731 \
  --pipeline-parallel-size 3 --kv-cache-dtype fp8 \
  --max-model-len 16384 --gpu-memory-utilization 0.95 \
  --tokenizer-mode deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

`VLLM_PP_LAYER_PARTITION=15,15,13` is required on 3 cards, not optional — the default even split (`15,14,14`) fails during the Marlin FP4 expert repack because the last rank also carries `lm_head` and the DSpark drafter. `gpu-memory-utilization 0.95` is required too: 0.85 and 0.93 both fail KV allocation on that same last rank.

### 4. First request

```bash
curl http://localhost:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-0731",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "temperature": 0,
    "max_tokens": 8
  }'
```

Text-only. No image-input example applies to this checkpoint — see `deepseek-v4-flash-vision-exp.md` for the vision-capable sibling.

## Recommended settings

- **Sampling for throughput benches:** greedy decoding, three prompt classes (technical, prose, code), 400 completion tokens per request.
- **Speculative decoding:** DSpark at k=5, matched to this checkpoint's `dspark_block_size`. Do not raise it — k=7 measured worse (60.3 vs. 83.3–98.1 tok/s).
- **KV cache:** fp8. Required by this fork's `fp8_ds_mla` layout assertion, not optional.
- **Topology:** pipeline parallel, not tensor parallel. On this fabric (no NVLink, no P2P over PCIe Gen2 x4), PP beats TP by 6.6x on prefill (5,321 vs. 801 tok/s at 77k context) and roughly 2x on aggregate decode.
- **gpu-memory-utilization:** 0.95 on 3 cards, 0.85 on 4 cards. Higher is not always safer — see Troubleshooting.
- **Context:** 16,384 measured locally; the community 4-card reference verified the full 1,047,736-token window.

## Troubleshooting

**Precompiled wheel silently lacks the custom op.** `VLLM_USE_PRECOMPILED=1` installs without `vllm._C`; the failure surfaces at CUDA-graph capture, far from the actual cause. Build from source whenever a patch touches `csrc/`.

**Editable installs avoid most rebuilds.** vLLM installed with `pip install -e .` lets patched Python files be bind-mounted read-only over the checkout, taking effect with no rebuild. Only `csrc/` changes force a real rebuild (~62 minutes). This recipe ships five patched files this way.

**Verify a patch with three checks, not one.** `py_compile` (syntax only) and an import smoke test both passed on a real shipped bug; only an undefined-name scan (pyflakes) caught it. Run all three before trusting a patched build.

**Uneven layer partition costs KV capacity.** Rebalancing the 4-card partition away from an even split grew the KV pool about 85% (798,660 to 1,476,563 tokens at `max-model-len 163840`) by removing an 8.7 GiB rank imbalance on the `lm_head`-heavy last rank.

**Wrong `gpu-memory-utilization` fails KV allocation, not the model.** On 3 cards, 0.85 and 0.93 both fail; 0.95 works (leaves 7.7 GiB of KV against the last rank's 51.8 GiB of weights plus activations and CUDA graphs). On 4 cards, going above 0.85 takes headroom from activations and graph capture — 0.90 with the DSpark draft resident OOMs at capture time.

**NFS vs. NVMe boot times.** This checkpoint is about 148 GB. Loading from NFS-backed shared storage measured about 31 MiB/s aggregate, roughly 22 minutes for weight loading plus about 7 minutes of CUDA graph capture. Local NVMe-class storage cuts the load leg by 16–25x.

**`--gpus device=...` list after a crash.** After an OOM crash, `--gpus all` can silently assign zero devices to the next container — stale cgroup state from the crash. Use `--gpus '"device=0,1,2"'` (or your card list) explicitly.

**Driver recovery after an OOM storm.** A multi-rank OOM kill can leave every GPU showing NVRM assertion failures and `cuInit` returning `CUDA_ERROR_NO_DEVICE` host-wide. `rmmod nvidia_uvm nvidia` then `modprobe nvidia nvidia_uvm` restores the devices without a VM reboot; reloading `nvidia_uvm` alone does not.

**2,941-token prefill OOM does not apply here.** That limit is specific to the reference TP4 runtime used for the vision-capable sibling checkpoint's correctness milestone, not to this PP3/PP4 vLLM recipe — see `deepseek-v4-flash-vision-exp.md` for that runtime's limits.

## Benchmarks

### Three cards, PP3, DSpark k=5, 180 W (measured 2026-08-30, pending evidence repair)

| Prompt class | Decode throughput |
|---|---:|
| Technical | 73.4 tok/s |
| Prose | 72.4 tok/s |
| Code | 116.6 tok/s |
| Aggregate | **83.3 tok/s** |

Uncached prefill: 2,965 tok/s at 5,399 input tokens. Draft-token acceptance: 5.07–5.32 tokens (81–86%).

### Four cards, PP4, DSpark k=5 (community reference, not this club's own run)

| Metric | Value |
|---|---:|
| Aggregate decode | 98.1 tok/s |
| Prefill (77k context) | 5,321 tok/s |
| Context verified | Full 1,047,736 tokens |

### Speculative-decoding sensitivity (PP4, from `docs/LESSONS.md`)

| k | Mean acceptance length | Aggregate decode |
|---:|---|---:|
| 5 | 3.03 | 98.1 tok/s |
| 7 | 1.43–2.51 | 60.3 tok/s — worse, do not use |

### Concurrency behavior (PP4, from `docs/LESSONS.md`)

DSpark keeps winning through 64 concurrent requests on pipeline parallel (712.8 vs. 472.0 tok/s plain decode at c=64). On tensor parallel, the same technique goes negative above about 8 concurrent requests — a pipeline-parallel-specific result, not a general DSpark property.

## Artifacts

- **Evidence repository:** [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX). Sanitized receipts for the 3-card run are an open PR there as of this writing.
- **GHCR image:** none published for this recipe. No image lineage exists for this checkpoint on its own; it shares the "fullbuild" source-build lineage documented for the vision-capable sibling (`Dockerfile.fullbuild` in the allover326 stack, later `Dockerfile.fullbuild16` for the vision fork — see `deepseek-v4-flash-vision-exp.md`). Build from source per the Quick start above.
- **Checkpoint:** [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` (resolved after the fact, see Quick start above).
- **Patch series:** SM80 vLLM fork and patches from [allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80) and [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx).
- **Executed notebook:** [notebooks/2026-08-30-deepseek-v4-flash-0731-3card-pp3-vllm.ipynb](../../notebooks/2026-08-30-deepseek-v4-flash-0731-3card-pp3-vllm.ipynb).

## Changelog

- **2026-08-30** — 3-card local run measured (83.3 tok/s aggregate decode, DSpark k=5, 180 W); sanitized receipt chain pending merge in the evidence repository.
- **Undated (community reference)** — 4-card reference run from allover326 measured 98.1 tok/s aggregate decode and verified the full context window; kept as upstream context, not this club's own measurement.
