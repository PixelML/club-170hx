# GLM-5.3-Flash on vLLM sm80 (`glm53-sm80`) — 4x CMP 170HX

Measured 2026-09-03, 180 W per-card club-standard cap. Runtime: the
PixelML `sm80vllm` fork, branch `glm53-sm80` (wtdcode/vllm-backport
GLM enablement vendored onto the club image lineage; provenance in
`docs/SM80.md` on that branch). Image `ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903` (local build tag `glm53-sm80:test`)
(f7fe5d02c295), built from `docker/Dockerfile.glm53-sm80` (full source
build, `TORCH_CUDA_ARCH_LIST=8.0`).

## First boot (hypothesis 1, TP4)

- Checkpoint: `wtdcode/GLM-5.3-Flash-AWQ-W4A16` @ `abd7b07719111f137e1de8a0c1b7e01c11b74d1a`
  (compressed-tensors AWQ W4A16, 190,843,146,533 bytes, byte-verified).
- Attention: `TRITON_MLA_SPARSE` (Triton sparse-MLA fallback), kpool
  indexer on the Triton fp8 MQA-logits fallback (DeepGEMM unavailable on
  SM80 — expected). MoE: Marlin WNA16 (`CompressedTensorsWNA16MoEMethod`).
- MTP draft (`Glm5NextMTPModel`, single MTP layer) with
  `num_speculative_tokens=3`. TileLang KDA/hyper-connection kernels
  compiled on-device.
- KV cache: 7.98 GiB/card → 650,916 tokens; `--max-model-len 524288`.
- Cold boot: ~41 min from NFS; ~10 min from the NVMe-staged copy
  (weights 54 s/shard warm).

## c=1 single-stream, MTP depth sweep (5 measured reps each, greedy protocol off — temperature 0.7, 512 output tokens, `ignore_eos`, cold rep noted)

| num_speculative_tokens | median tok/s | peak | cold rep |
|---|---:|---:|---:|
| 2 | 51.1 | 54.7 | 38.7 |
| **3** | **56.4** | **56.9** | 41.0 |
| 5 | 47.1 | 52.3 | 35.0 |

MTP-3 is the accepted optimum (same shape as the DSpark k-sweep finding
in `docs/LESSONS.md` §d: depth past the acceptance cliff buys pure
verification cost).

## Aggregate (same boot, TP4)

| Concurrency | Aggregate tok/s |
|---|---:|
| c=1 | 54.7 |
| c=8 | 37.0 |

c=8 aggregate is **below** the exllamav3 EXL3 baseline (44.8 tok/s at
180 W): TP4 runs an all-reduce per layer over PCIe Gen1 with no P2P, and
the ladder is communication-bound. See `docs/LESSONS.md` §c — PP moves
~28x less wire data than TP on this fabric. Aggregate is the open item;
the DSV4 playbook's aggregate numbers were all PP4.

## Known limitation found while iterating (do not reproduce)

`--no-enable-flashinfer-autotune` combined with MTP-3 crashes at engine
startup (`cudaErrorLaunchFailure` in mm-encoder profiling), reproduced on
a clean post-reboot boot. The crash wedges a GPU at the PCIe level
(`rev ff`); only a VM reboot recovered it. The measured recipe above
runs flashinfer autotune at its default (on).

## Follow-up (tracked)

PP4 + MTP (the aggregate fix) is stock-blocked in vLLM; it needs the
`vllm-dsa-mtp-sm80` PR #46994 patch set ported to `glm53-sm80`
(12 hunks in `spec_decode/autoregressive/speculator.py` + 6 in
`pp_utils.py`, different upstream era; PR-referenced runtime knobs:
`--enforce-eager`, gpu-memory-utilization 0.80). Not yet ported.

## Launch recipe

```bash
# weights staged at /models/model-cache/glm-5.3-flash-awq-w4a16
docker run -d --name glm53-vllm --gpus '"device=0,1,2,3"' \
  --shm-size 16g --ipc=host -p 127.0.0.1:18098:8000 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  -e NCCL_VERSION=2.28.3-1 -e TORCH_CUDA_ARCH_LIST=8.0 \
  -e VLLM_TARGET_DEVICE=cuda \
  --mount type=bind,src=<weights>,dst=/model,readonly \
  ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903 \
  vllm serve /model --served-model-name glm-5.3-flash \
  --tensor-parallel-size 4 --max-model-len 524288 \
  --gpu-memory-utilization 0.92 --max-num-seqs 16 \
  --max-num-batched-tokens 8192 --enable-prefix-caching \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
```

Live evidence thread: [seanphan/pixelml#103](https://github.com/seanphan/pixelml/issues/103)
(first boot: comment 5527939703; depth sweep: comment 5530227651).
