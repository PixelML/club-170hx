# LLM inference

LLM inference is the first verified workload track.

## Proven paths

- **Single card:** Qwen3.8-27B NVFP4, measured across three separate cards.
- **Three cards:** DeepSeek-V4-Flash-0731 using pipeline parallelism.
- **Compatibility rejection:** GLM-5.3-Flash NVFP4 is an SM121 path, not an SM80 artifact; the examined AWQ alternative does not statically fit three 64 GiB cards with runtime headroom.

See [Benchmarks](../../docs/BENCHMARKS.md) for results and evidence links.

## Recipe order

1. Measure checkpoint size with filesystem bytes, not the model card estimate.
2. Run `scripts/model-fit.py` with a realistic per-card runtime reserve.
3. Check runtime support for SM80 and the model architecture/quantization.
4. Prefer a one-card correctness test; use pipeline parallel first for oversized models on slow PCIe.
5. Record TTFT/prefill, decode, concurrency, acceptance rate if speculative, power, and temperatures.

Example static check:

```bash
python3 scripts/model-fit.py 148 --gpus 3 --vram-gib 64 --reserve-gib 8
```

Static fit is necessary but insufficient. CUDA graphs, KV cache, expert placement, and uneven layer sizes can still cause out-of-memory failures.
