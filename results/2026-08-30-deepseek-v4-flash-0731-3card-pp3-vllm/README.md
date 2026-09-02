# DeepSeek-V4-Flash-0731 on 3x CMP 170HX (SM80)

> Sanitized receipt for the `2026-08-30-deepseek-v4-flash-0731-3card-pp3-vllm`
> experiment, copied from the upstream evidence repository
> [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX).
> See the sanitization note at the end of this file.

Official **deepseek-ai/DeepSeek-V4-Flash-0731** (48 shards, 148 GB)
served with vLLM on 3x NVIDIA CMP 170HX mining cards (64 GB HBM2e each,
SM80, no FP4 tensor cores), power-capped at **180 W** per card.

**Result: 83.3 tok/s aggregate decode** (technical 73.4 / prose 72.4 /
code 116.6, 400 tok each, greedy, single stream) and up to **2965 tok/s
prefill** at 5.4K context. The upstream reference config
(allover326/deepseek-v4-cmp170hx, 4-card) is ~78.8 tok/s.

## The config

| setting | value |
|---|---|
| parallelism | pipeline-parallel 3, VLLM_PP_LAYER_PARTITION=15,15,13 |
| speculative | DSpark, k=5 (acceptance length 5.07-5.32, draft acc 81-86%) |
| KV cache | fp8, block-size 256, maxlen 16384 |
| gpu-memory-utilization | **0.95** (0.85/0.93 fail KV allocation on the lm_head-heavy last rank) |
| max-num-batched-tokens / seqs | 2048 / 8 |
| image | full CUDA source build, TORCH_CUDA_ARCH_LIST=8.0 (see build note) |
| launch | [launch-dsv4-3card.sh](launch-dsv4-3card.sh) on port 8098 |

## What does not work

- **Precompiled vLLM image** (VLLM_USE_PRECOMPILED): silently ships
  without vllm._C, so fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out
  fails at graph capture. Full source build required - exactly the
  warning in the allover326 Dockerfile.fullbuild.
- **gpu-memory-utilization below 0.95** on 3 cards: KV allocation fails
  on rank 2 (weights 51.8 GiB + activations + CUDA graphs leave ~9 GiB;
  0.95 makes it fit with 7.7 GiB KV).
- **NVFP4 recipes** (GLM-5.3-Flash, Qwen3.8-Flash-Next DGX Spark kits):
  CMP 170HX is SM80 with no FP4 path; those checkpoints (~181 GiB) do not
  fit 3x64 GB anyway.

## Reproduce

    # weights once, on a big library volume
    <model-storage>/deepseek-ai/DeepSeek-V4-Flash-0731/   # 48 shards, 148 GB

    # serve (cold boot ~30 min: 22 min shard load from NFS + 7 min graph capture)
    DSV4_MAXLEN=16384 DSV4_GPU_UTIL=0.95 bash launch-dsv4-3card.sh
    curl http://127.0.0.1:8098/health

    # bench (allover326 harnesses)
    python3 bench_decode3.py run
    python3 bench_prefill.py run 42

Numbers: [RESULTS.md](RESULTS.md). Setup credit and base patches:
[allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx)
and [allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80).

## Hardware

3x CMP 170HX @ 64 GB HBM2e, locked 180 W (nvidia-smi verified before and
during runs), PCIe Gen2 x4, model library on 8.4 TB NFS (<model-storage>).

---

Sanitization note: storage paths beginning with `/library/` and
`/home/<user>/` are replaced by `<model-storage>/` and `<workdir>/`; local
container and image names are replaced by `<container>` and `<image>`.
`127.0.0.1` is retained intentionally. The masked `launch-dsv4-3card.sh` in
this bundle is a record and will not parse as-is under bash. This bundle
contains no raw JSON or log receipts; the upstream evidence repository holds
only the three files copied here (README, RESULTS.md, launch script).
