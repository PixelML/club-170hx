# Workload — Bonsai-2 27B (ternary → W4A16 conversion) + DFlash2 on vLLM 0.28, 1 × CMP 170HX

Status: measured
Date: 2026-09-19

## Hardware

- Cards: 1 × CMP 170HX (64 GiB HBM2e), card 0 of a multi-card node; a second card hosted an idle llama-server (hence GPU_UTIL 0.78)
- Power limit and measured peak draw: 180.00 W read back on every 1 Hz sample; peak 219 W instantaneous transient above the average-based cap
- Cooling and peak core/memory temperatures: forced airflow; 68 °C core / 78 °C memory vs the 80/85 °C stop conditions

## Software

- OS / kernel: Ubuntu 22.04, kernel 6.8
- NVIDIA driver / CUDA: 610.43.03; vLLM 0.28.0 wheel (torch cu130), flashinfer 0.6.16.post3 + cubin 0.6.13 (JIT topk compiled with the toolkit's own nvcc 13.0), flash-linear-attention 0.5.2 (GDN prefill kernel)
- Runtime repository + exact commit: syv-ai qwen-serving patch series (33 of 38 patches applied to the vLLM tree; the dflash2 backport is retired — DFlash2 is native in 0.28.0)
- Model repository + exact revision: converted from prism-ml/Ternary-Bonsai-2-27B-gguf @ `6ed5e12bf84b7a63069882c91dd9e9218647d17b` (F16 GGUF), vision tower + MTP module grafted from Qwen/Qwen3.8-27B
- Quantization / dtype: W4A16 GPTQ (llmcompressor 0.13, pack-quantized, 256 × 1024-token open_platypus samples) + int8 lm_head/embed/MTP (syv prepare chain) + 40960-token draft head; visual tower bf16

## Command

```text
SPEC=dflash2 CTX=fast MAX_SEQS=1 DFLASH_TOKENS=7 GPU_UTIL=0.78 KV_MEM= MODEL=<w4a16-dir> PORT=18020 bash single-user/start_qwen.sh
```

## Method

- Warmup: 1 discarded rep per cohort
- Samples: 3 measured reps per cohort; ladder = 3 reps per level
- Input/output tokens: decode 10-token prompt / 256 and 900 completion (`ignore_eos`); prefill ~6.6k-token prompt (nonce-varied, per-request) / 11 out; ladder 2/4/8 × 256 out
- Metric calculation: tokens from the final usage object only; decode = completion/(wall − TTFT); acceptance from `/metrics` spec-decode counters diffed across the decode window
- Executed notebook: `notebooks/2026-09-19-bonsai-2-27b-w4a16-dflash2-1card-vllm.ipynb` (LIVE = False)

## Results

| Metric | Value |
|---|---:|
| Decode 256-token cohort (tok/s, ttft-excl. / wall) | 155.7 / 136.4 |
| Decode 900-token cohort (tok/s, ttft-excl. / wall) | 251.6 / 238.0 |
| Prefill, ~6.6k-token prompt (tok/s) | 1876 |
| TTFT, 10-token prompt (ms, client-observed) | 226 |
| Accepted tokens per draft (zero-shot base drafter) | 4.17 (59.6% per position) |
| Aggregate c=2/4/8 (tok/s, wall) | 137.6 / 138.2 / 136.9 |
| Power during serving (W, mean / peak) | 101 / 219 |
| Boot to healthy (warm page cache) | ~90 s incl. torch.compile |

Measured notes:

- **The mechanism transfers.** The ternary-retrained weights, converted out of the
  Hadamard-folded GGUF layout, run the exact DFlash2 recipe of the 147.7-tok/s
  Qwen3.8-27B receipts without recalibrating the drafter: acceptance 59.6% per
  position (0.88 at position 0 decaying to 0.46 at position 6), and decode lands
  ~5% above that receipt on the 256-token cohort. The 900-token cohort's 252
  tok/s reflects DFlash2's adaptive draft depth (nmin 6 / nmax 12) stretching on
  predictable `ignore_eos` text.
- **The conversion required four inverse transforms**, all verified against
  runtime source or the base model: Hadamard unrotation (`W_hf = W_stored @ T`,
  T = blockdiag(H·diag(s_b)/32) from the file's own sign manifest), GDN v-layout
  reorder (rep-major 3×16 → group-major 16×3, mirrors the Bonsai-demo
  `runtime.py` `reorder()`, 0.0 diff against the gen-1 unpacked/F16 pair),
  delta-norms (`g_hf = g_gguf + 1` for input/post/final/q_norm/k_norm),
  conv1d rank reshape, and verbatim grafts of the base `model.visual.*`/`mtp.*`
  (absent from the ternary GGUF; the MTP module is what DFlash2 drafts with).
- The bench co-hosted an idle llama-server (GPU_UTIL 0.78 instead of 0.90); idle,
  it does not affect decode. Aggregate holds at the single-stream rate across
  c=2/4/8 because the recipe pins a single slot.
- This lane runs the **int8** lm_head/embed/MTP variant — no Bonsai-2 fast
  variant (int4) exists on the Hub, which is base-model-only.

## Correctness and failures

- Output validation: greedy gate ("capital of France" → `Paris` with 24 reasoning tokens), coherent 256-token answer in the notebook's final editable request, all cohort requests returned valid usage objects
- Xid/ECC scan: none recorded
- Known caveats: client TTFT carries a ~0.2 s SSE first-chunk constant; boot time reflects warm page cache; one GPU of the node wedged during pre-bench debugging (unrelated to this workload's serving; PCI remove/rescan recovered it) — the serving run itself was fault-free

## Evidence

- Raw receipts: `receipts/` (serve JSONL, `/metrics` snapshots before/after decode and ladder, 1 Hz power/thermal telemetry, gate response, summary with the full conversion recipe)
- Charts: `assets/charts/2026-09-19-bonsai-2-27b-w4a16-dflash2-1card-vllm-*.png` + regenerator scripts
- Companion ternary lane: `results/2026-09-18-bonsai-2-27b-ternary-1card-llamacpp/`
- Upstream: [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), [syv-ai qwen-serving](https://github.com/syv-ai/qwen38-27b-rtx3090), [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo)

Inferred (not measured): the drafter's zero-shot acceptance would rise above 0.6
with a Bonsai-2-recalibrated draft head (syv `drafter/capture_dflash2.py`
pipeline); the conversion's Hadamard convention was verified against runtime
source and the base model rather than against an official Bonsai-2 HF export,
which does not exist at time of writing.
