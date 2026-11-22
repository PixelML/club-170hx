# Inkling-2-Mini on 1x CMP 170HX (SM80), 140 W — Round 1–4 SM80 CUDA megakernels, monolithic CUDA graphs, 18-split decode attention, 49K DSpark pruning & unit-scale FP8 KV

Status: measured
Date: 2026-11-22

Follow-up to [`2026-11-21-inkling-2-mini-1card-vllm-sm80-dspark-vs-llamacpp`](../2026-11-21-inkling-2-mini-1card-vllm-sm80-dspark-vs-llamacpp/README.md) (PR #90, commit `2f23339b66e6bba963b8b145f739ff8b78ecc927`).

Notebook: [`notebooks/2026-11-22-inkling-2-mini-1card-sm80-opt-vllm.ipynb`](../../notebooks/2026-11-22-inkling-2-mini-1card-sm80-opt-vllm.ipynb) · Upstream-rebased SM80 fork port: [`PixelML/sm80vllm#3`](https://github.com/PixelML/sm80vllm/pull/3) (`20a5fdccd5badfcc6c6b502cedc5e77be4d8d05e`)

**Question.** PR #90 got `thinkingmachines/Inkling-2-Mini` running on SM80 with six minimal compatibility patches (`108.56 tok/s` 6-prompt median / `146.0 tok/s` code on a stock CMP 170HX, `491.1 tok/s` at `c=16`, `432.5 s` cold startup, `32,768` context). How far can a stock-clocked CMP 170HX (`1,140 MHz` SM / `1,728 MHz` HBM2e, `140 W` TDP, `1,600.5 GB/s` D2D copy bandwidth) push single-user decode, `c=16` throughput, and full `262,144`-token context (including multi-stream `>690K` KV via unit-scale FP8 KV) from one unified server without changing the target output distribution?

**Answer (measured): `108.56 → 134.98 → 155.48 → 164.25 → 167.24 tok/s` median (`169.20` peak, `+54.1%` same-card / `+55.9%` peak) single-user decode and `491.1 → 676.0 tok/s` (`697.4` peak, `+37.7%` / `+42.0%`) `c=16` aggregate on the same unified server at full `262,144` context.** On PR #90's exact benchmark harness (`harness/bench.py`, 768 output tokens) at `262,144` context on the stock card, Round-4 Best reaches **`214.6 tok/s` code (`228.6` at `T=0`), `199.6 tok/s` structured JSON (`216.2` `T=1` median, `225.9` peak), `128.9 tok/s` prose (`132.1` peak), and `646.1 tok/s` at `c=16`** — beating both PR #90's stock-card baseline (`146.0 / 136.8 / 87.5`) by `+46–47%` and PR #90's `+200 MHz SM offset` card at `32K` context (`166.9 / 189.1 / 94.0` at `T=0`, `510.9` at `c=16`) by `+37.0%` code (`T=0`), `+37.1%` prose (`T=0`), and `+26.5%` (`+32.3%` unified) at `c=16`. Long-context needle retrieval is **`3/3` at `131,227` tokens (`194.22 tok/s` decode, `201.98` peak) and `231,044` tokens (`154.92 tok/s` decode, `+11.8%` over Round 3)**, GSM8K-50 is **`50/50` (`100%`)** on both BF16-KV and FP8-KV, and enabling **`--kv-cache-dtype=fp8`** expands the `262K` KV pool from `563,878` (`2.15x`) to **`805,605` tokens (`3.07x`)**, passing **`4 × 172.9K` = `692,992` simultaneous active KV tokens (`4/4` streams, `12/12` needles, `0` errors)**.

## Scoreboard (`stock CMP 170HX 64 GiB @ 140 W TDP, 1,140 MHz SM / 1,728 MHz HBM2e, 1,600.5 GB/s D2D copy BW`, `262,144` ctx)

| Metric | Baseline (`ctl-01/02`, `k=7`) | Round 1 (`cand-10`, `k=7`) | Round 2 (`cand-16`, `k=4`) | Round 3 (`cand-25`, unified) | **Round 4 Best (`cand-30`, unified)** | Theoretical Ceiling |
|---|---:|---:|---:|---:|---:|---:|
| `c=1` decode median (6 prompts, `T=0.6`) | `108.56 tok/s` | `134.98 tok/s` (`+24.3%`) | `155.48 tok/s` (`+43.2%`) | `164.25` (`164.94` pk) | **`167.24 tok/s` (`169.20` pk, `+54.1%`)** | `238` (mixed) / `298` (code) |
| `c=16` aggregate (same unified server) | `491.10 tok/s` | `521.90 tok/s` (`+6.3%`) | `604.21 tok/s` (`+23.0%`) | `669.47` (`681.49` pk) | **`676.00 tok/s` (`697.44` pk, `+37.7%`)** | `15,181 tok/s` (HBM roof) |
| Cold prefill `~4K` / `~32K` (`tok/s`) | `4,735.8` / `5,039.8` | `5,660.1` / `5,664.0` | `6,012.8` / `5,603.6` | `5,865.6` / `5,596.7` | **`5,943.0` / `5,697.8 tok/s`** | `5,731` / `5,026` (50% MFU) |
| Short-prompt TTFT (`ms`) | `130.8 ms` | `87.4 ms` | `93.4 ms` (`87.0` best) | `95.8 ms` (`71.0` warm) | **`104.0 ms` (`71.0` warm)** | — |
| Server startup (`s`, → `/health` 200) | `432.5 s` (`207.5` warm) | `145.5 s` | `120.3 s` (`108.3` best) | `133.4 s` (`128.3` best) | **`164.4 s` (monolithic FULL CG)** | — |
| Long-context `131,227` tok (prefill / decode / needles) | — | `3,249.5` / `107.78` (`3/3`) | `3,556.3` / `193.60` (`3/3`) | `3,544.9` / `187.48` (`3/3`) | **`3,523.5` / `194.22 tok/s` (`201.98` pk, `3/3`)** | `3,534` / `185 tok/s` |
| Long-context `231,044` tok (prefill / decode / needles) | — | `2,586.0` / `141.45` (`3/3`) | `2,543.9` / `146.76` (`3/3`) | `2,542.8` / `138.62` (`3/3`) | **`2,529.8` / `154.92 tok/s` (`156.76` pk, `3/3`)** | `2,532` / `150 tok/s` |
| Concurrent long soak (`262K` server) | — | `2/2` (`335.7K` KV) | `2/2` (`335.7K` KV) | `2/2` (`335.7K` KV) | **`2/2` (`335.7K` BF16) · `4/4` (`693.0K` FP8 KV, `12/12` needles)** | `563.9K` BF16 / `805.6K` FP8 |
| PR #90 `c=1` `code` / `prose` / `struct` (pooled med) | `146.0` / `87.5` / `136.8` | `177.1` / `103.5` / `206.0` | `188.5` / `121.6` / `185.2` | `204.0` / `127.8` / `183.6` | **`214.6` (`228.6` `T=0`) / `128.9` / `199.6` (`225.9` pk)** | — |
| PR #90 `conc/c16` (`12,288` tok) & GSM8K-50 | `510.9*` / `48/50` (`96%`) | `539.3` / `48/50` (`96%`) | `604.2` / `50/50` (`100%`) | `650.1` / `50/50` (`100%`) | **`646.11 tok/s` / `50/50` (`100%` BF16 & FP8)** | — |

*\*PR #90 `510.9 tok/s` c=16 (`166.9 / 189.1 / 94.0` c=1 at `T=0`) was measured on PR #90's `+200 MHz SM offset` card at `32,768` context; all Round 1–4 scoreboard medians ran on the exact same stock-clocked CMP 170HX at full `262,144` context (`169.20 / 697.44` peak measured on a second stock 170HX card).*

## Winning stack summary

1. **Zero-ULP-diff Cold-HBM SM80 Humming v3 MoE buckets (`patches/overlay/nvidia/ops/moe_sm90.py`):** 8-warp `num_warps_k=1` `[8, 256, 64] w=[8, 32, 64] stg=4 cta=2 evict=True` on `NVFP4 w13` (`torch.equal == True`, `-0.409 ms` cold-HBM at `M=5`), `[8, 256, 128] cta=2 evict=True` on `NVFP4 w2`, `stg=4 cta=3` `[8, 128, 64]` on `MXFP8 w2` (`1.553 → 0.863 ms`, fixing `stg=5` SMEM spill), plus restored `BM=96` prefill buckets above `2,048` rows.
2. **Custom SM80 CUDA megakernels (`patches/sm80_fast_v5.cu`, `patches/overlay/sm80_ops_v2.py`, 0 spills):** compiled inside the container with `nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC` (`moe_align_v3_sm80`, `select_and_align_32cta_128_sm80`, `<1 KB`-smem `conv_mix_2304_sm80`, `swiglu_1152_sm80`, `pair/blend_rms_norm_2304_sm80`, `markov_rnn_gate_256_sm80`, and 1-launch bitwise-exact `markov_fold_gumbel_cache_r1_sm80` + `OnlineAcceptanceEstimator` fused variant).
3. **SM80 low-M Tensor-Core linears (`patches/overlay/nvidia/ops/ll_gemm_sm80.py` + `nvidia/model.py`):** ported & retuned `_ll_gemm_kernel` + 1-launch `_qkv_bmm_kernel` from `vllm` PR `#69168` (`BN=128` `8600×2304` `loc_qkvr`, batched `MP=64` `loop.blend`, `cuBLAS` `M>=5` `lm_head`).
4. **Lossless DSpark `K=49,152` pruning + zero-scatter `-inf` `DraftLogitsCache` + bidirectional `k=4` + joint block verification (`"block"`) + dynamic batch `k` (`[[1,2,4],[3,6,3],[7,256,2]]`):** `fold_weight` shrinks `102.9 → 25.17 MB` (fits inside the `32.0 MiB` physical L2 cache); `k=4` concentrates DSpark bidirectional block attention; Sun et al. joint block verification lifts prose `+13.3%`; dynamic `k` unifies `c=1` (`k=4`) and `c=16` (`k=2`) in one server.
5. **Monolithic `M <= 21` FULL CUDA graphs + 18-split SM80 global decode attention + optional unit-scale SM80 E4M3 FP8 KV (`patches/overlay/nvidia/{attention.py,dspark.py}` + `nvidia/ops/{attention.py,attention_sm90.py,sconv.py}`):** eliminates piecewise graph breaks across all 48 layer visits (`228.6 tok/s` `T=0` code), raises global decode `head_splits` `9 → 18` (`1,329.8 GB/s` = `89.1%` raw HBM2e peak at `131K`, `+11.8%` at `231K`), and adds branchless 9-instruction PTX `e4m3x2 → f16x2` + IEEE RNE `_fp32_to_e4m3fn_u8` (`--kv-cache-dtype=fp8`: `805,605` KV tok = `3.07x` @ `262K` with lossless BF16 DSpark KV).

## Reproducibility pins

- Container image: `vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`
- Model & drafter: `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7` (Apache-2.0)
- Upstream vLLM commits ported: `vllm-project/vllm@ab18334a4f60da74db7de89fe95594e5ed0cda36` (PR #69028) and `8c17e4692a7cf393e3d1dacf6404fa4d48d993a1` (PR #69168)
- Rebased `PixelML/sm80vllm` fork PR: [`PixelML/sm80vllm#3`](https://github.com/PixelML/sm80vllm/pull/3) @ `20a5fdccd5badfcc6c6b502cedc5e77be4d8d05e` (rebased onto `vllm-project/vllm` `b976fff86af82c559b19aa676f52ea193eb0c188`)
- Driver stack: guest NVIDIA `580.178.04` over host unlock `amoghmunikote/cmpunlocker@88e39ce67488796b2c6c716fe8f9b4e6e943a55e` + `PixelML/cmpunlocker@6ca4da72f078553d37089ee741cd128aa7804504`
- Harnesses: `sybil-solutions/inference-optimisation-skill@fda968e7fc1996aa23c67139c755fe541c9d4be0`, `openai/gsm8k@e53f048856ff4f594e959d75785d2c2d37b678ee`

## Files

- `patches/` — `sm80_fast_v5.cu` + `overlay/` Python sources + `install-sm80-r4.sh` / `install-sm80-r3.sh` (compiles `libsm80_fast_v5.so` inside the container via `nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC`; zero `.so` binaries or `__pycache__`) + `inkling2-sm80-r3.diff` (complete unified diff against the image's `vllm/models/inkling2`).
- `harness/` — `run-vllm-r3.sh` (one-command Docker launcher), `onstart.sh` (Vast.ai / cloud GPU rental one-command bring-up; supports `KV_CACHE_DTYPE=fp8`), `run_gates.py` (5 capability gates + `2 × 167.8K` concurrent soak), `bench.py` (PR #90 bench client), `hillclimb_bench.py`, `ceiling.py`, `ceiling-inkling2-cmp170hx.json`, `gsm8k.py`, `guard.sh`, `hbm_bw.py`.
- `receipts/` — `ledger.jsonl` (all 82 runs across Waves 0–11 incl. negative results), `round4-spec.json`, `pr90-dspark7-stock.jsonl`, `pr90-r1best.jsonl`, `pr90-r2best.jsonl`, `pr90-r3best.jsonl`, `pr90-r4best.jsonl`, `gsm8k-r1best.jsonl`, `gsm8k-r2best.jsonl`, `gsm8k-r3best.jsonl`, `gsm8k-r4best.jsonl`, `gsm8k-r4best-fp8.jsonl`, `r1-gates-summary.json`, `r2-gates-summary.json`, `r3-gates-summary.json`, `r4-gates-summary.json`, `soak-4x173k-fp8kv.json`, `r1-winner-metrics.txt`, `r2-winner-metrics.txt`, `r3-winner-metrics.txt`, `r4-winner-metrics.txt`, `r1-prof-phases-c1.json`, `r3-prof-phases-c1.json`, `r3-prof-phases-c16.json`, `srv-r3-winner-final.log`, `summary.csv`.
- `summarize.py`, `make_chart.py`, `make_notebook.py` — parameterized across `ROUNDS` to rebuild `receipts/summary.csv`, `assets/charts/`, and `notebooks/`.

## Credits

- **Thinking Machines Lab** ([@thinking-machines-lab](https://github.com/thinking-machines-lab)) authored [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its DSpark draft head (Apache-2.0), revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`.
- **Woosuk Kwon** ([@WoosukKwon](https://github.com/WoosukKwon)) and the **vLLM team** ([@vllm-project](https://github.com/vllm-project)) authored vLLM Inkling-2-Mini support ([PR #69028](https://github.com/vllm-project/vllm/pull/69028), `ab18334a4f60da74db7de89fe95594e5ed0cda36`, Apache-2.0) and `vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`.
- **Thien Tran** ([@gau-nernst](https://github.com/gau-nernst)) authored the Triton low-M BF16 GEMM (`_ll_gemm_kernel`, `_qkv_bmm_kernel`) in [vllm-project/vllm PR #69168](https://github.com/vllm-project/vllm/pull/69168) (`8c17e4692a7cf393e3d1dacf6404fa4d48d993a1`, Apache-2.0).
- **0xSero** ([@0xSero](https://github.com/0xSero)) and **Sybil Solutions** ([@sybil-solutions](https://github.com/sybil-solutions)) authored [sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill) @ `fda968e7fc1996aa23c67139c755fe541c9d4be0` (MIT).
- **Ziteng Sun**, **Ananda Theertha Suresh**, **Jae Hun Ro**, **Ahmad Beirami**, **Himanshu Jain**, and **Felix Yu** authored joint block-level draft verification ([arXiv:2403.10444](https://arxiv.org/abs/2403.10444)).
- **Ash** ([@ashhart](https://github.com/ashhart)) authored [TensorFold](https://github.com/ashhart/TensorFold) @ `34af03a12aa388130b71909c9d7b30481f6ec9da`; **OpenAI** authored [GSM8K](https://huggingface.co/datasets/openai/gsm8k) @ `e53f048856ff4f594e959d75785d2c2d37b678ee` (MIT).
