# %% [markdown] cell:0
# Qwen3.8-Flash-Next — vLLM production-kernel prefill baseline on DGX Spark (apollo-2, GB10 128GB unified), NVFP4 checkpoint

| tokens served | TTFT median (s) | prefill tok/s |
|---:|---:|---:|
| 509 | 0.435 | 1,170 |
| 1,001 | 0.608 | 1,647 |
| 2,001 | 0.967 | 2,069 |
| 6,481 | 1.542 | **4,203** |

| Item | Value |
|---|---|
| Stack | vLLM (MiaAI-Lab `qwen38-flash-next` image), NVFP4 checkpoint `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`, PLE packed/offloaded, TP=1 |
| Host | apollo-2 DGX Spark, GB10 128GB unified, arm64 |
| Method | streaming completions, temp 0, `max_tokens=1`, `include_usage`, median of 3, prompts token-trimmed with the model tokenizer |
| Purpose | **production-kernel BASELINE** for the LLKVApprox CED comparison; next experiment applies the trained projector ([PixelML/Qwen3.8-Flash-Next-KVA-Projector](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-KVA-Projector)) as a same-stack overlay |

Eager-engine comparison (different stack, labeled): our correctness-first eager engine measures 53.4 tok/s at ~6.5k — a ~79x kernel-path gap to this vLLM baseline, attributed by component profile to the MoE path (per-expert Python loop + per-pass int4 dequant, ~73% of the pass). The CED ratio claim (2.19-2.27x, same-stack) is separate and unchanged.
# %% [code] cell:1
# --- Status -------------------------------------------------------------
import os, json
EXPERIMENT = "2026-09-17-qwen3.8-flash-next-vllm-baseline-apollo2"
RESULTS_DIR = os.path.join("..", "receipts", EXPERIMENT)
LIVE = False
print(f"experiment : {EXPERIMENT}")
print(f"LIVE       : {LIVE}")
# %% [code] cell:2
# --- Baseline table -----------------------------------------------------
import json, os
r = json.load(open(os.path.join(RESULTS_DIR, "vllm-baseline.json")))
lines = ["| tokens served | TTFT median s | prefill tok/s |", "|---:|---:|---:|"]
for row in r["rows"]:
    lines.append(f"| {row['tokens_served']} | {row['ttft_median_s']} | {row['prefill_tok_s']} |")
from IPython.display import display, Markdown
display(Markdown("\n".join(lines)))
print("receipt:", os.path.join(RESULTS_DIR, "vllm-baseline.json"))
# %% [markdown] cell:3
# ### Next experiment
#
# Apply the trained LLKVApprox projector as a same-stack overlay (layers 24..47 skip +
# fill injection inside the qwen4_exp forward on this vLLM build), then re-run this exact
# grid. Matched-stack comparison only; the eager-stack 2.19-2.27x is a separate claim.
