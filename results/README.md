# Result submission format

Create one Markdown file per result, plus a runnable artifact for it. Store small, redacted machine-readable evidence beside it; do not commit model outputs containing private prompts or data.

The runnable artifact is either an executed notebook under `notebooks/` with its receipts in a dated `results/` directory (the convention used by the GLM-5.3-Flash and DeepSeek lanes), or a self-contained reproduction folder under `recipes/<model-runtime>/` containing `recipe.json`, `reproduce.ipynb`, `results/summary.csv`, and a generated chart. Either way the notebook must preserve clean output from the measured run, execute in order after one configuration cell, and finish with an editable request that prints the response and final usage object.

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

Also run `python3 scripts/validate_recipe_notebooks.py` from the repository root.
