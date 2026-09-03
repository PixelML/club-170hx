# Model guides

One page per model family this club has measured on CMP 170HX. Each guide is the durable how-to: quick-start commands, recommended settings, troubleshooting, and the current benchmark numbers, kept in sync with the receipts in the model's evidence repository. The executed notebook under `notebooks/` is the proof that a specific run happened; the guide here is what you run next.

| Model | Best measured | Status | Guide |
|---|---|---|---|
| DeepSeek-V4-Flash-Vision-Exp | 220.2 tok/s aggregate decode (c=8, text path) | Text measured; vision measured, partial (c=2 ceiling) | [deepseek-v4-flash-vision-exp.md](deepseek-v4-flash-vision-exp.md) |
| DeepSeek-V4-Flash-0731 | 83.3 tok/s aggregate decode (3 cards, local) | Measured, pending evidence repair | [deepseek-v4-flash-0731.md](deepseek-v4-flash-0731.md) |
| Qwen3.8-27B | 147.7 tok/s decode (1 card, 255 W rented) | Measured, pending evidence repair | [qwen3.8-27b.md](qwen3.8-27b.md) |
| GLM-5.3-Flash | 44.8 tok/s aggregate decode (c=8, 4 cards, EXL3 4.05bpw); 262,144-token context validated (250,000 prompt tokens, `cache_mode: FP16`) | Measured; the earlier ~2,048-token cap under `cache_mode: Q8` is resolved — see the guide | [glm-5.3-flash.md](glm-5.3-flash.md) |

For every model this club has ever tried, including negative results and planned attempts, see [docs/MODEL-STATUS.md](../MODEL-STATUS.md). For the normalized cross-model benchmark ledger, see [docs/BENCHMARKS.md](../BENCHMARKS.md).
