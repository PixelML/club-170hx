# Model guides

One page per model family this club has measured on CMP 170HX. Each guide is the durable how-to: quick-start commands, recommended settings, troubleshooting, and the current benchmark numbers, kept in sync with the receipts in the model's evidence repository. The executed notebook under `notebooks/` is the proof that a specific run happened; the guide here is what you run next.

| Model | Best measured | Status | Guide |
|---|---|---|---|
| DeepSeek-V4-Flash-Vision-Exp | 220.2 tok/s aggregate decode (c=8, text path) | Text measured; vision measured, partial (c=2 ceiling) | [deepseek-v4-flash-vision-exp.md](deepseek-v4-flash-vision-exp.md) |
| DeepSeek-V4-Flash-0731 | 83.3 tok/s aggregate decode (3 cards, local) | Measured, pending evidence repair | [deepseek-v4-flash-0731.md](deepseek-v4-flash-0731.md) |
| Qwen3.8-27B | 147.7 tok/s decode (1 card, 255 W rented) | Measured, pending evidence repair | [qwen3.8-27b.md](qwen3.8-27b.md) |
| GLM-5.3-Flash | 56.4 tok/s median decode (c=1, 4 cards, AWQ W4A16 + vLLM MTP-3); 37.0 tok/s aggregate at c=8; `max-model-len=524,288` configured | Measured, partial; c=8 is TP4 communication-bound; EXL3 remains the stronger c=8 lane at 44.6 tok/s — see the guide | [glm-5.3-flash.md](glm-5.3-flash.md) |

For every model this club has ever tried, including negative results and planned attempts, see [docs/MODEL-STATUS.md](../MODEL-STATUS.md). For the normalized cross-model benchmark ledger, see [docs/BENCHMARKS.md](../BENCHMARKS.md).
