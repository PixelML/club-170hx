# SPDX-License-Identifier: Apache-2.0
"""PixelML sm80 patch: pure-torch moe_align_block_size for GPUs where the image's compiled
`_moe_C.moe_align_block_size` has no kernel image (CMP 170HX, sm_80; image built for sm_100).

Same outputs as vllm.model_executor.layers.fused_moe.moe_align_block_size for the call Inkling2's
Hopper expert path makes (ignore_invalid_experts=True, no expert_map): token rows sorted by expert
(stable), each expert's segment padded to a multiple of block_size, padding = topk_ids.numel().
No host sync and fixed output shapes, so it is CUDA-graph capturable.
"""
import torch
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def moe_align_block_size(topk_ids, block_size, num_experts, expert_map=None, pad_sorted_ids=False,
                         ignore_invalid_experts=False, return_scatter_idx=False):
    if expert_map is not None or return_scatter_idx:
        raise NotImplementedError("sm80 moe_align_block_size: expert_map / scatter_idx not supported")
    dev = topk_ids.device
    n = topk_ids.numel()
    max_pad = n + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_pad = round_up(max_pad, block_size)
    if n < num_experts:
        max_pad = min(n * block_size, max_pad)
    n_blocks = triton.cdiv(max_pad, block_size)

    flat = topk_ids.reshape(-1).to(torch.int64)
    valid = (flat >= 0) & (flat < num_experts)
    if not ignore_invalid_experts:
        valid = torch.ones_like(valid)  # every id counts (caller guarantees range)
    key = flat.masked_fill(~valid, num_experts)  # invalid -> bucket E
    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=dev)
    counts.scatter_add_(0, key, torch.ones_like(key))  # bincount without a host sync
    counts = counts[:num_experts]
    padded = (counts + block_size - 1) // block_size * block_size
    pad_end = torch.cumsum(padded, 0)
    pad_start = pad_end - padded
    cnt_start = torch.cumsum(counts, 0) - counts

    order = torch.sort(key, stable=True).indices
    skey = key[order]
    is_valid = skey < num_experts
    e = skey.clamp(max=num_experts - 1)
    pos = torch.arange(n, device=dev, dtype=torch.int64) - cnt_start[e]
    dest = (pad_start[e] + pos).masked_fill(~is_valid, max_pad)  # invalid -> dump slot
    buf = torch.full((max_pad + 1,), n, dtype=torch.int32, device=dev)
    buf.scatter_(0, dest, order.to(torch.int32))
    sorted_ids = buf[:max_pad].contiguous()

    starts = torch.arange(n_blocks, device=dev, dtype=torch.int64) * block_size
    expert_ids = torch.searchsorted(pad_end, starts, right=True).clamp(max=num_experts - 1)
    total = pad_end[-1:]
    expert_ids = expert_ids.masked_fill(starts >= total, 0).to(torch.int32)
    return sorted_ids, expert_ids, total.to(torch.int32).reshape(1)
