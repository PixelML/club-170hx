# SPDX-License-Identifier: Apache-2.0
"""Native SM80 CUDA moe_align_v3 (1-CTA shared-atomic + 8-warp scan up to 1024)."""
import ctypes
import os
import torch
from vllm.utils.math_utils import round_up

_LIB = None
_FN = None
_WS: dict[torch.device, torch.Tensor] = {}


def _get_fn():
    global _LIB, _FN
    if _FN is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "libsm80_fast_v4.so")
        if not os.path.exists(cand):
            cand = "/work/libsm80_fast_v4.so"
        _LIB = ctypes.CDLL(cand)
        _FN = _LIB.launch_moe_align_v3_sm80
        _FN.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int32] * 6 + [ctypes.c_void_p]
        _FN.restype = None
    return _FN


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
    return_scatter_idx: bool = False,
):
    if expert_map is not None or return_scatter_idx or num_experts > 256:
        raise NotImplementedError("sm80_moe_align CUDA: unsupported args")
    flat = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    n = flat.numel()
    max_pad = n + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_pad = round_up(max_pad, block_size)
    if n < num_experts:
        max_pad = min(n * block_size, max_pad)
    n_blocks = (max_pad + block_size - 1) // block_size
    dev = flat.device
    ws = _WS.get(dev)
    if ws is None:
        ws = torch.empty((2 * 64 * 256,), dtype=torch.int32, device=dev)
        _WS[dev] = ws
    sorted_ids = torch.empty((max_pad,), dtype=torch.int32, device=dev)
    expert_ids = torch.empty((n_blocks,), dtype=torch.int32, device=dev)
    num_post_pad = torch.empty((1,), dtype=torch.int32, device=dev)
    _get_fn()(
        flat.data_ptr(),
        sorted_ids.data_ptr(),
        expert_ids.data_ptr(),
        num_post_pad.data_ptr(),
        ws.data_ptr(),
        n,
        block_size,
        num_experts,
        max_pad,
        n_blocks,
        int(ignore_invalid_experts),
        torch.cuda.current_stream(dev).cuda_stream,
    )
    return sorted_ids, expert_ids, num_post_pad
