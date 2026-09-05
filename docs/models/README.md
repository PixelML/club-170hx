# Model guides

One page per model family this club has measured on CMP 170HX. Each guide is the durable how-to: quick-start commands, recommended settings, troubleshooting, and the current benchmark numbers, kept in sync with the receipts in the model's evidence repository. The executed notebook under `notebooks/` is the proof that a specific run happened; the guide here is what you run next.

| Model | Best measured | Status | Guide |
|---|---|---|---|
| DeepSeek-V4-Flash-Vision-Exp | 220.2 tok/s aggregate decode (c=8, text path) | Text measured; vision measured, partial (c=2 ceiling) | [deepseek-v4-flash-vision-exp.md](deepseek-v4-flash-vision-exp.md) |
| DeepSeek-V4-Flash-0731 | 83.3 tok/s aggregate decode (3 cards, local) | Measured, pending evidence repair | [deepseek-v4-flash-0731.md](deepseek-v4-flash-0731.md) |
| Qwen3.8-27B | 147.7 tok/s decode (1 card, 255 W rented) | Measured, pending evidence repair | [qwen3.8-27b.md](qwen3.8-27b.md) |
| GLM-5.3-Flash | 87.6 tok/s c=1 decode (4 cards, AWQ W4A16 · vLLM sm80 PP4 + native MTP k=3, temp 0, median of 3, clean) / 67.9 tok/s (temp 0.7 + `ignore_eos`, median of 5); 78.4 tok/s aggregate at c=16 on a degraded PCIe link | Measured 2026-09-05 — recipe of record; the 2026-09-03 TP4 lane (56.4 tok/s c=1) is superseded — see the guide | [glm-5.3-flash.md](glm-5.3-flash.md) |

For every model this club has ever tried, including negative results and planned attempts, see [docs/MODEL-STATUS.md](../MODEL-STATUS.md). For the normalized cross-model benchmark ledger, see [docs/BENCHMARKS.md](../BENCHMARKS.md).
