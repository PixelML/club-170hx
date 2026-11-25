# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

from .nvidia.ops import is_sm80

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class Inkling2DSparkSpeculatorConfig(DSparkSpeculatorConfig):
    speculator_class = "vllm.models.inkling2.speculator.Inkling2DSparkSpeculator"
    adaptive_verification_uses_draft_logits = True

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        super().adapt_draft_model_config(spec)
        if (
            spec.draft_model_config is not None
            and spec.num_speculative_tokens is not None
        ):
            hf_cfg = spec.draft_model_config.hf_config
            if (
                hasattr(hf_cfg, "draft_block_size")
                and 1 <= spec.num_speculative_tokens < hf_cfg.draft_block_size - 1
            ):
                hf_cfg.draft_block_size = spec.num_speculative_tokens + 1
                hf_cfg.dspark_block_size = spec.num_speculative_tokens
        if current_platform.is_device_capability_family(120):
            # The target's clipped, unit-scale FP8 cache does not describe the
            # draft's unnormalized values. Keep the draft cache in BF16.
            if spec.kv_cache_dtype is None:
                spec.kv_cache_dtype = "auto"
            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):
                raise ValueError("Inkling2 DSpark on SM12 requires BF16 draft KV")
        elif is_sm80():
            from .sm80_tp_ops import install_sm80_spec_schedule_hooks

            install_sm80_spec_schedule_hooks()
            # Keep DSpark's 3 draft layers in BF16 even when target uses FP8 KV.
            if spec.kv_cache_dtype is None:
                spec.kv_cache_dtype = "bfloat16"
            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):
                raise ValueError("Inkling2 DSpark on SM80 requires BF16 draft KV")
