# Result submission format

Create one Markdown file per result. Store small, redacted machine-readable evidence beside it when useful; do not commit model outputs containing private prompts or data.

```markdown
# Workload — short result name

Status: measured | compatibility-only | failed
Date: YYYY-MM-DD

## Hardware

- Cards: N × CMP 170HX
- Anonymous card labels:
- Topology / PCIe links:
- Power limit and measured peak draw:
- Cooling and peak core/memory temperatures:

## Software

- OS / kernel:
- NVIDIA driver / CUDA:
- Runtime repository + exact commit or image digest:
- Model repository + exact revision:
- Quantization / dtype:

## Command

```text
copy-pasteable redacted command
```

## Method

- Warmup:
- Samples:
- Input/output tokens, frames, resolution, steps, batch, concurrency:
- Metric calculation:

## Results

| Metric | Value |
|---|---:|
| ... | ... |

## Correctness and failures

- Output validation:
- Xid/ECC/AER scan:
- Known caveats:

## Evidence

- Links to raw redacted files or upstream benchmark repository
```

Before committing, run the publication gate in [AGENTS.md](../AGENTS.md).
