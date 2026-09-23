#!/usr/bin/env bash
# SM80 import probe for the MiMo-V2.6 PP4 lane. No GPU time, no weights:
# answers "does this image register mimo_v2 and can its quant path run on cc 8.0"
# before any expensive load. ~15 min budget.
set -euo pipefail
IMAGE="${1:-vllm/vllm-openai:mimov25-cu129}"

docker pull "$IMAGE"

docker run --rm --gpus all --entrypoint python3 "$IMAGE" - <<'PY'
import json
import torch

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("device cap:", torch.cuda.get_device_capability(0))

import vllm
print("vllm:", vllm.__version__)

from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print("mimo_v2 registered:", "MiMoV2ForCausalLM" in archs)

# quantization method coverage on this build
try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    print("quant methods:", sorted(m for m in QUANTIZATION_METHODS))
except Exception as e:  # noqa: BLE001
    print("quant method listing unavailable:", e)
PY
