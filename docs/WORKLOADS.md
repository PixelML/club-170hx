# Workload matrix

The repository covers more than LLM inference. Each workload track starts with compatibility and memory-fit evidence, then adds a reproducible recipe and measured result.

| Track | Current state | Next evidence needed |
|---|---|---|
| LLM inference | Verified on one-, three-, and four-card workloads; real-image vision verified on four cards | More model families, concurrency, energy/token, a performance-grade SM80 vision path |
| Image generation | Planned | Reproducible SM80 pipeline, images/minute, peak VRAM and power |
| Video generation | Measured on one card: MiniMax H3 video + audio, ComfyUI/SGLang/diffusers, power cap sweep ([notebook](../notebooks/2026-10-06-minimax-h3-video-1card-comfyui.ipynb)) | Sustained 200 W soak, multi-card sequence parallel, lip-sync and identity scores, 768-px native resolution |
| CUDA/QC | Initial tools included | Near-full HBM and sustained compute reports from more cards |
| Fine-tuning/training | Untested | Memory plan, optimizer/quantization, interconnect scaling |
| Multi-node | Untested | Network/topology and end-to-end scaling data |

Follow the track guides:

- [LLM inference](../workloads/llm/README.md)
- [Image generation](../workloads/image/README.md)
- [Video generation](../workloads/video/README.md)

## Qualification order

1. Confirm CUDA/SM80 compatibility.
2. Calculate static weight fit and reserve runtime/KV/activation memory.
3. Prefer a single-card correctness run when possible.
4. Select pipeline/data/tensor parallelism from communication behavior.
5. Benchmark with thermal, power, quality, and correctness data—not throughput alone.
