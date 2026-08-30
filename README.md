# Club CMP 170HX

Community-tested recipes, diagnostics, and reproducible benchmarks for running AI workloads on NVIDIA CMP 170HX cards.

This project is building a practical, low-cost SM80 compute pool for:

- large-language-model inference;
- image generation;
- video generation;
- CUDA validation and memory-heavy research workloads;
- single-node and distributed experiments.

The CMP 170HX shares useful traits with A100-class hardware, including SM80 compute capability and high-capacity HBM. It is **not an A100 replacement**: it has an unsupported software path, no display output, no NVLink, limited PCIe behavior in common passthrough setups, and unusual cooling and power requirements.

## Start here

| Goal | Guide |
|---|---|
| Understand the card and trade-offs | [Hardware](docs/HARDWARE.md) |
| Install it in a Proxmox VM | [Installation](docs/INSTALLATION.md) |
| Validate a new or used card | [QC and acceptance testing](docs/QC.md) |
| Control heat and noise | [Cooling and power](docs/COOLING-AND-POWER.md) |
| Plan a multi-card node | [Cluster design](docs/CLUSTER.md) |
| Diagnose failures | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| Compare measured results | [Benchmarks](docs/BENCHMARKS.md) |
| Choose an AI workload | [Workload matrix](docs/WORKLOADS.md) |

## Verified baseline

Our current repeatable baseline is:

- 3 × CMP 170HX, each reporting 64 GiB VRAM;
- Ubuntu 22.04 guest on a Proxmox Q35 VM with SeaBIOS;
- Linux 6.8 and NVIDIA 610.43.03 open kernel modules;
- pinned `cmpunlocker` v0.1 patch set;
- 125 W quiet/idle policy and 180 W benchmark policy;
- forced airflow across every passive heatsink.

Exact versions matter. Treat this as a known-good reference, not a claim that every card, board, BIOS, kernel, or driver combination will work.

## Published workload results

| Workload | Topology | Result | Evidence |
|---|---:|---:|---|
| Qwen3.8-27B NVFP4 | 1 card | 136.38 tok/s mean at 180 W across three cards | [Benchmark repo](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| DeepSeek-V4-Flash-0731 | 3 cards, pipeline parallel | 83.3 tok/s aggregate decode at 180 W/card | [Benchmark repo](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) |
| GLM-5.3-Flash (all quants) | 3 cards | Not compatible as of 2026-08-30: no SM80 runtime; smallest checkpoint 198.1 GiB exceeds 192 GiB node total | [GLM attempt repository](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) |

These are measured application results, not theoretical peaks. The linked repositories contain commands, model/runtime pins, per-run outputs, and known caveats.

## Repository map

```text
docs/       Hardware, setup, QC, operations, troubleshooting, and results
scripts/    Read-only inventory, model-fit, and card-validation helpers
workloads/  LLM, image, and video workload recipes and status
results/    Submission format for reproducible community results
```

## Model-family repositories

Experiments live in one dedicated repository per model family or workload —
for example [GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX)
holds every GLM-5.3-Flash attempt (NVFP4, AWQ, GPTQ, EXL3, FP8/BF16, all runtimes,
successes, and failures). Never one repository per quantization, checkpoint, runtime,
or machine; this repository indexes results and holds cross-workload guidance.

## Safety first

CMP 170HX cards use passive server heatsinks. Do not run sustained workloads without directed, monitored airflow. Our default stop thresholds are 80 °C core or 85 °C memory. A cold power cycle may be required after a card falls off the PCIe bus.

The unlock path is community-maintained and unsupported by NVIDIA. Back up the machine, pin known-good artifacts, verify module provenance, and expect recovery work.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md). Benchmark claims need raw, redacted evidence and full environment metadata. Security and privacy rules are in [AGENTS.md](AGENTS.md) and [SECURITY.md](SECURITY.md).

Inspired by the community-first documentation style of [club-3090](https://github.com/noonghunna/club-3090).

## License

Apache-2.0. See [LICENSE](LICENSE).
