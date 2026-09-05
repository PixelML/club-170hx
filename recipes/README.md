# Reproducible benchmark notebooks

This directory is the runnable public entry point for measured CMP 170HX recipes. Each published recipe is a self-contained folder that a reader can open in Jupyter, configure once, and run from top to bottom.

## Published recipes

| Model | Runtime | Hardware | Notebook | Headline result |
|---|---|---|---|---:|
| Qwen3.8-27B W4A16 + DFlash2 | vLLM 0.27.1 + pinned public patch set | 1 × CMP 170HX, 180 W | [Open notebook](qwen3.8-27b-dflash2/reproduce.ipynb) | 136.38 decode tok/s mean across three cards |

## Folder contract

```text
recipes/<model-runtime>/
├── README.md             short verdict and notebook link
├── recipe.json           immutable pins and artifact index
├── requirements.lock     exact direct serving dependencies
├── reproduce.ipynb       configure → preflight → install → serve → benchmark → curl
├── assets/performance.png
└── results/              sanitized receipts plus the CSV used by the notebook and chart
```

A published notebook must preserve clean outputs from the measured environment, regenerate its chart from committed results, fail closed when prerequisites are missing, and finish with an editable `curl` request that prints the model response and final usage accounting.

Validate every notebook before publication:

```bash
python3 scripts/validate_recipe_notebooks.py
python3 scripts/render_recipe_chart.py \
  --spec recipes/qwen3.8-27b-dflash2/chart-spec.json \
  --results recipes/qwen3.8-27b-dflash2/results/summary.csv \
  --output recipes/qwen3.8-27b-dflash2/assets/performance.png
```

## Not yet packaged as a recipe folder

The GLM-5.3-Flash vLLM sm80 lanes publish as executed notebooks with dated
receipt directories rather than as `recipes/` folders. They are reproducible
from those pages today; packaging them to this folder contract is pending.

| Date | Lane | Notebook | Receipts |
|---|---|---|---|
| 2026-09-05 | GLM-5.3-Flash PP4 + native MTP k=3, AWQ W4A16 — recipe of record | [notebook](../notebooks/2026-09-05-glm-5.3-flash-4card-pp4-vllm.ipynb) | [results](../results/2026-09-05-glm-5.3-flash-4card-pp4-vllm/README.md) |
| 2026-09-03 | GLM-5.3-Flash TP4 + native MTP k=3, AWQ W4A16 — superseded | [notebook](../notebooks/2026-09-03-glm-5.3-flash-4card-tp4-vllm.ipynb) | [results](../results/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu/README.md) |

The full index of executed notebooks is [notebooks/README.md](../notebooks/README.md).

Large logs and detailed experiment history remain outside the club recipe. Link them only after their complete public boundary is independently validated. The club notebook contains the sanitized file-level receipts needed to verify the published result.
