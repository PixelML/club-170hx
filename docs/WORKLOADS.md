# Workload matrix

The repository covers more than LLM inference. Each workload track starts with compatibility and memory-fit evidence, then adds a reproducible recipe and measured result.

| Track | Current state | Next evidence needed |
|---|---|---|
| LLM inference | Verified on one- and three-card workloads | More model families, concurrency, energy/token |
| Image generation | Planned | Reproducible SM80 pipeline, images/minute, peak VRAM and power |
| Video generation | Planned | Reproducible model, resolution/frames, seconds/frame, peak VRAM |
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
