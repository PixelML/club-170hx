# Image generation

Status: **planned; no public CMP 170HX result yet**.

The first recipe should favor a widely redistributable SM80-compatible pipeline and record both performance and output correctness.

## Minimum benchmark record

- model and VAE repository + exact revisions;
- runtime/attention implementation and CUDA versions;
- resolution, steps, sampler/scheduler, guidance, batch size, seed, and dtype;
- cold start, warm latency, images/minute, peak VRAM, power, core temperature, and memory temperature;
- output hashes and a small license-compatible sample set;
- any CPU/NVMe offload and its effect on latency.

Do not compare images/second across different step counts, resolutions, or quality settings as if they were equivalent.

## Qualification target

1. Single-card correctness at 125 W.
2. Repeated warm run at 180 W with forced airflow.
3. Batch/VRAM scaling curve.
4. Comparison with a controlled reference GPU.
5. Multi-card independent-job throughput before attempting communication-heavy parallelism.
