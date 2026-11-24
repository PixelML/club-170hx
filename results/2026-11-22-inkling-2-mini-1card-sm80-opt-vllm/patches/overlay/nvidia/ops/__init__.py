# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    return current_platform.is_device_capability_family(100)


@functools.cache
def is_sm80() -> bool:
    """Whether to run the SM8x (Ampere / Ada, pre-Hopper) kernels."""
    return (
        current_platform.is_cuda()
        and current_platform.has_device_capability(80)
        and not current_platform.has_device_capability(90)
    )


# Programmatic Dependent Launch (`griddepcontrol.*` PTX) requires Hopper+
# (SM90+). Query via NVML-backed `has_device_capability` so importing this
# module in the API-server parent process never initializes CUDA early.
LAUNCH_PDL: bool = (
    current_platform.is_cuda() and current_platform.has_device_capability(90)
)
_USE_PDL: tl.constexpr = tl.constexpr(LAUNCH_PDL)


@triton.jit
def gdc_wait() -> None:
    if _USE_PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents() -> None:
    if _USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
