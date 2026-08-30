# Video generation

Status: **planned; no public CMP 170HX result yet**.

Video pipelines combine large weights, long-lived activations, host-memory pressure, and expensive output validation. Capacity alone does not guarantee useful throughput.

## Minimum benchmark record

- model repository + exact revision and component dtypes;
- runtime commit, attention backend, and offload settings;
- width, height, frame count, FPS, steps, guidance, seed, and batch;
- load time, generation wall time, seconds/frame, peak VRAM, host RAM, power, and temperatures;
- output checksum, decode success, duration, and a license-compatible sample;
- storage read/write volume when components are offloaded.

## Qualification target

1. Preflight static weights and host-memory requirements.
2. Single-card short-clip correctness.
3. Long-clip thermal and memory-stability run.
4. Repeatability at fixed seed and settings.
5. Multi-card component/pipeline experiment only after measuring transfer cost.

Do not publish a speed number without resolution, frame count, steps, and quality settings beside it.
