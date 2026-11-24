# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 Triton stand-in for `_C::vocab_parallel_embedding` (no sitecustomize required)."""

import torch

from vllm.triton_utils import tl, triton

from .nvidia.ops import is_sm80

_INSTALLED = False
_LIBS: list[torch.library.Library] = []


@triton.jit
def _vocab_parallel_embedding_kernel(
    Out,
    Ids,
    W,
    out_t,
    w_v,
    org_start,
    org_end,
    add_start,
    add_end,
    add_offset,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    d_blk = tl.program_id(1)
    tok = tl.load(Ids + t).to(tl.int64)
    in_org = (tok >= org_start) & (tok < org_end)
    in_add = (tok >= add_start) & (tok < add_end)
    valid = in_org | in_add
    row = tl.where(in_org, tok - org_start, tl.where(in_add, tok - add_offset, 0))
    cols = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    cmask = cols < D
    vals = tl.load(W + row * w_v + cols, mask=valid & cmask, other=0.0)
    tl.store(Out + t * out_t + cols, vals, mask=cmask)


def vocab_parallel_embedding(
    out: torch.Tensor,
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> None:
    n = input_ids.shape[0]
    if n == 0:
        return
    d = weight.shape[1]
    assert out.stride(1) == 1 and weight.stride(1) == 1
    add_offset = (
        int(added_vocab_start_index)
        - (int(org_vocab_end_index) - int(org_vocab_start_index))
        - int(num_org_vocab_padding)
    )
    block_d = 256
    _vocab_parallel_embedding_kernel[(n, triton.cdiv(d, block_d))](
        out,
        input_ids,
        weight,
        out.stride(0),
        weight.stride(0),
        int(org_vocab_start_index),
        int(org_vocab_end_index),
        int(added_vocab_start_index),
        int(added_vocab_end_index),
        add_offset,
        D=d,
        BLOCK_D=block_d,
        num_warps=4,
    )


def install_sm80_tp_ops() -> None:
    """Register `_C::vocab_parallel_embedding` CUDA impl on sm_80 (idempotent)."""
    global _INSTALLED
    if _INSTALLED or not is_sm80():
        return
    import vllm._custom_ops  # noqa: F401

    lib_c = torch.library.Library("_C", "IMPL")
    lib_c.impl("vocab_parallel_embedding", vocab_parallel_embedding, "CUDA")
    _LIBS.append(lib_c)
    _INSTALLED = True
