# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark + Hybrid SuffixDecoding speculator for Inkling2 on SM80:
stores raw target features, runs DSpark's recurrent Markov head, and
blends/shortcuts lossless exact-point-mass SuffixDecoding proposals on GPU."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import async_tensor_h2d, current_stream
from vllm.v1.worker.gpu.sample.gumbel import (
    _gumbel_combine_kernel,
    _log1p_neg_stable,
    _murmur3_fmix32,
    _murmur3_mix,
    gumbel_block_argmax,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

from .nvidia.ops import is_sm80

_SUF_RING_SIZE = 4096
_SUF_RING_MASK = 4095
_SUF_RING_BLOCK = 256
_SUF_RING_BLOCKS = _SUF_RING_SIZE // _SUF_RING_BLOCK


@triton.jit
def _suffix_sync_hbm_ring_kernel(
    uva_token_ids_ptr,  # *int32  [max_num_reqs, uva_stride]
    uva_stride,
    idx_mapping_ptr,  # *int32  [B]
    total_len_ptr,  # *int32  [max_num_reqs]
    hbm_ring_ptr,  # *int32  [max_num_reqs, RING_SIZE]
    hbm_base_ptr,  # *int64  [max_num_reqs]
    hbm_len_ptr,  # *int64  [max_num_reqs]
    RING_SIZE: tl.constexpr,
    RING_MASK: tl.constexpr,
    SYNC_BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    if req < 0:
        return
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    prev_len = tl.load(hbm_len_ptr + req)
    prev_base = tl.load(hbm_base_ptr + req)

    is_inc = (prev_len > 0) & (seq_len >= prev_len) & ((seq_len - prev_len) <= 16)
    copy_start = tl.where(is_inc, prev_len, tl.maximum(seq_len - RING_SIZE, 0))
    new_base = tl.where(
        is_inc, tl.maximum(prev_base, seq_len - RING_SIZE), copy_start
    )

    uva_row = uva_token_ids_ptr + req * uva_stride
    ring_row = hbm_ring_ptr + req * RING_SIZE
    for p in range(copy_start, seq_len, SYNC_BLOCK):
        offs = p + tl.arange(0, SYNC_BLOCK).to(tl.int64)
        m = offs < seq_len
        tok = tl.load(uva_row + offs, mask=m, other=-1)
        tl.store(ring_row + (offs & RING_MASK), tok, mask=m)

    tl.store(hbm_base_ptr + req, new_base)
    tl.store(hbm_len_ptr + req, seq_len)


@triton.jit
def _suffix_local_scan_ring_kernel(
    hbm_ring_ptr,  # *int32  [max_num_reqs, RING_SIZE]
    idx_mapping_ptr,  # *int32  [B]
    hbm_base_ptr,  # *int64  [max_num_reqs]
    hbm_len_ptr,  # *int64  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    scratch_ptr,  # *int64  [B, NUM_BLOCKS]
    scratch_stride,
    Lp1,
    RING_SIZE: tl.constexpr,
    RING_MASK: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    scratch_off = b * scratch_stride + blk
    if req < 0:
        tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
        return
    base = tl.load(hbm_base_ptr + req)
    seq_len = tl.load(hbm_len_ptr + req)
    num_sampled = tl.load(num_sampled_ptr + b)

    if base + blk * BLOCK >= seq_len:
        tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
        return

    ring_row = hbm_ring_ptr + req * RING_SIZE
    pos = base + blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    active = (pos >= base + 1) & (pos < seq_len) & (num_sampled > 0)
    match_len = tl.zeros([BLOCK], dtype=tl.int64)
    for t in range(MAX_DEPTH):
        tail_pos = seq_len - 1 - t
        tail_ok = tail_pos >= base
        tail_tok = tl.load(ring_row + (tail_pos & RING_MASK), mask=tail_ok, other=-2)
        src_pos = pos - 1 - t
        ok = active & (src_pos >= base) & tail_ok
        tok = tl.load(ring_row + (src_pos & RING_MASK), mask=ok, other=-1)
        active = ok & (tok == tail_tok)
        match_len += active.to(tl.int64)

    score = tl.where(match_len > 0, match_len * Lp1 + pos, 0)
    tl.store(scratch_ptr + scratch_off, tl.max(score, axis=0))


@triton.jit
def _hybrid_suffix_finalize_ring_kernel(
    hbm_ring_ptr,  # *int32  [max_num_reqs, RING_SIZE]
    idx_mapping_ptr,  # *int32  [B]
    hbm_len_ptr,  # *int64  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    last_sampled_ptr,  # *int64  [max_num_reqs]
    temp_ptr,  # *fp32   [max_num_reqs]
    scratch_ptr,  # *int64  [B, scratch_stride]
    scratch_stride,
    Lp1,
    drafts_ptr,  # *int64  [B, drafts_stride]
    drafts_stride,
    num_valid_ptr,  # *int32  [B]
    match_len_ptr,  # *int64  [B]
    skip_flag_ptr,  # *int32  [1]
    min_match_t0,
    min_match_tpos,
    RING_SIZE: tl.constexpr,
    RING_MASK: tl.constexpr,
    WRITE_SKIP_FLAG: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    NB_PO2: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    if req < 0:
        return
    seq_len = tl.load(hbm_len_ptr + req)
    num_sampled = tl.load(num_sampled_ptr + b)
    last_tok = tl.load(last_sampled_ptr + req)
    temp = tl.load(temp_ptr + req)
    ring_row = hbm_ring_ptr + req * RING_SIZE

    nb = tl.arange(0, NB_PO2).to(tl.int64)
    l_score = tl.max(tl.load(scratch_ptr + b * scratch_stride + nb), axis=0)
    l_len = l_score // Lp1
    l_pos = l_score - l_len * Lp1

    k_iota = tl.arange(0, K_PO2).to(tl.int64)
    k_mask = k_iota < K
    idx = l_pos + k_iota
    draft_ok = k_mask & (idx < seq_len) & (l_len > 0) & (num_sampled > 0)
    draft = tl.load(ring_row + (idx & RING_MASK), mask=draft_ok, other=0).to(tl.int64)

    num_valid = tl.sum(draft_ok.to(tl.int32), axis=0)
    req_min = tl.where(temp == 0.0, min_match_t0, min_match_tpos)
    eff_match = tl.where((num_sampled > 0) & (l_len >= req_min), l_len, 0)
    out = tl.where(draft_ok, draft, last_tok)
    tl.store(drafts_ptr + b * drafts_stride + k_iota, out, mask=k_mask)
    tl.store(num_valid_ptr + b, num_valid)
    tl.store(match_len_ptr + b, eff_match)
    if WRITE_SKIP_FLAG:  # noqa: SIM102
        if b == 0:
            do_skip = (num_valid >= K) & (eff_match > 0)
            tl.store(skip_flag_ptr, do_skip.to(tl.int32))


@triton.jit
def _pruned_gumbel_scatter_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_cache_row_mapping_ptr,
    logits_ptr,
    logits_stride,
    keep_ids_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    pruned_vocab_size,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    pruned_offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pruned_offs < pruned_vocab_size
    target_keys = tl.load(keep_ids_ptr + pruned_offs, mask=mask, other=0)
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + pruned_offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        target_keys,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        logits_cache_ptr,
        logits_cache_stride_0,
        logits_cache_stride_1,
        logits_cache_col_ptr,
        None,
        0,
        logits_cache_row_mapping_ptr,
        pruned_vocab_size,
        IS_DRAFTING=True,
        APPLY_TEMPERATURE=True,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=False,
    )
    token_id = tl.load(
        keep_ids_ptr + block_idx * BLOCK_SIZE + idx,
        mask=(block_idx * BLOCK_SIZE + idx) < pruned_vocab_size,
        other=0,
    )
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def pruned_gumbel_sample_and_cache(
    logits: torch.Tensor,
    keep_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    logits_cache: torch.Tensor,
    logits_cache_col: torch.Tensor,
    logits_cache_row_mapping: torch.Tensor | None,
    use_fp64: bool = False,
) -> torch.Tensor:
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    logits_cache_col = logits_cache_col.contiguous()
    num_tokens, pruned_vocab = logits.shape
    block_size = 1024
    num_blocks = triton.cdiv(pruned_vocab, block_size)
    local_argmax = logits.new_empty((num_tokens, num_blocks), dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty((num_tokens, num_blocks), dtype=local_max_dtype)
    _pruned_gumbel_scatter_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0),
        logits_cache.stride(1),
        logits_cache_col,
        logits_cache_row_mapping,
        logits,
        logits.stride(0),
        keep_ids,
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        pruned_vocab,
        BLOCK_SIZE=block_size,
        USE_FP64=use_fp64,
    )
    sampled = logits.new_empty((num_tokens,), dtype=torch.int64)
    _gumbel_combine_kernel[(num_tokens,)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        sampled,
        num_blocks,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
    )
    return sampled


_MARKOV_BATCH_WS: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_RETIRED_MARKOV_WS: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []


def _get_markov_batch_ws(
    stream_id: int, device: torch.device, num_reqs: int, num_blocks: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (stream_id, device.index or 0)
    ws = _MARKOV_BATCH_WS.get(key)
    if (
        ws is None
        or ws[0].shape[0] < num_reqs
        or ws[0].shape[1] < num_blocks
    ):
        if ws is not None:
            _RETIRED_MARKOV_WS.append(ws)
        cap_b = max(num_reqs, 64)
        cap_n = max(num_blocks, 512)
        ws = (
            torch.empty((cap_b, cap_n), dtype=torch.int64, device=device),
            torch.empty((cap_b, cap_n), dtype=torch.float32, device=device),
            torch.empty((cap_b,), dtype=torch.int64, device=device),
        )
        _MARKOV_BATCH_WS[key] = ws
    return ws[0][:num_reqs, :num_blocks], ws[1][:num_reqs, :num_blocks], ws[2][:num_reqs]


@triton.jit
def _markov_fold_softcap_gumbel_batch_kernel(
    LATENT,
    FOLD_W,
    FOLD_B,
    BASE,
    KEEP_IDS,
    IDX_MAP,
    TEMP,
    SEEDS,
    POS,
    ROW_MAP,
    CACHE,
    LOCAL_ARGMAX,
    LOCAL_MAX,
    lat_stride,
    fw_stride,
    base_stride,
    idx_stride,
    pos_stride,
    cache_s0,
    cache_s1,
    l_stride,
    step_col,
    B,
    K_vocab,
    softcap,
    RANK: tl.constexpr,
    MP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    HAS_ROW_MAP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    rows = tl.arange(0, MP)
    cols = pid_n * BN + tl.arange(0, BN)
    row_mask = rows < B
    col_mask = cols < K_vocab

    acc = tl.zeros((MP, BN), dtype=tl.float32)
    for k0 in tl.static_range(0, RANK, BK):
        ks = k0 + tl.arange(0, BK)
        lat_tile = tl.load(
            LATENT + rows[:, None] * lat_stride + ks[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        w_tile = tl.load(
            FOLD_W + cols[None, :].to(tl.int64) * fw_stride + ks[:, None],
            mask=col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(lat_tile, w_tile, acc)

    fb = tl.load(FOLD_B + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = (acc + fb[None, :]).to(tl.bfloat16).to(tl.float32)
    base = tl.load(
        BASE + rows[:, None] * base_stride + cols[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    x = (base + bias).to(tl.bfloat16)
    lg = (softcap * tldevice.tanh(x.to(tl.float32) / softcap)).to(tl.bfloat16)

    target_keys = tl.load(KEEP_IDS + cols, mask=col_mask, other=0)
    req_idx = tl.load(IDX_MAP + rows * idx_stride, mask=row_mask, other=-1).to(tl.int64)
    is_valid_req = row_mask & (req_idx >= 0)
    temp = tl.load(TEMP + req_idx, mask=is_valid_req, other=0.0).to(tl.float32)

    if HAS_CACHE:
        if HAS_ROW_MAP:
            cache_row = tl.load(
                ROW_MAP + req_idx, mask=is_valid_req, other=-1
            ).to(tl.int64)
        else:
            cache_row = req_idx
        tl.store(
            CACHE
            + cache_row[:, None] * cache_s0
            + tl.cast(step_col, tl.int64) * cache_s1
            + target_keys[None, :],
            lg,
            mask=(cache_row[:, None] >= 0) & col_mask[None, :],
        )

    raw_seed = tl.load(SEEDS + req_idx, mask=is_valid_req, other=0).to(tl.int64)
    seed = raw_seed ^ (tl.cast(step_col, tl.int64) << 32) if HAS_CACHE else raw_seed
    pos = (
        tl.load(POS + rows * pos_stride, mask=row_mask, other=0).to(tl.int64)
        - 1
        + (1 << 30)
    )

    h = (seed ^ seed).to(tl.uint32)
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, (pos & 0xFFFFFFFF).to(tl.uint32))
    h2 = _murmur3_mix(h[:, None], target_keys[None, :].to(tl.uint32))
    random32 = _murmur3_fmix32(h2 ^ 16)
    hi16 = (random32 >> 16).to(tl.int32)
    lo16 = (random32 & 0xFFFF).to(tl.int32)
    u = (
        hi16.to(tl.float32) * 1.52587890625e-05
        + (lo16.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )
    gumbel_noise = -tl.log(-_log1p_neg_stable(u))

    score = lg.to(tl.float32)
    is_nz = temp != 0.0
    safe_temp = tl.where(is_nz, temp, 1.0)
    score = tl.where(is_nz[:, None], score / safe_temp[:, None] + gumbel_noise, score)
    score = tl.where(row_mask[:, None] & col_mask[None, :], score, float("-inf"))

    block_max, block_local_idx = tl.max(score, axis=1, return_indices=True)
    token_id = tl.load(
        KEEP_IDS + pid_n * BN + block_local_idx,
        mask=row_mask & ((pid_n * BN + block_local_idx) < K_vocab),
        other=0,
    )
    tl.store(LOCAL_ARGMAX + rows * l_stride + pid_n, token_id, mask=row_mask)
    tl.store(LOCAL_MAX + rows * l_stride + pid_n, block_max, mask=row_mask)


@triton.jit
def _markov_gumbel_combine_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    out_draft_col_ptr,
    out_draft_stride,
    out_prev_ptr,
    num_blocks,
    PADDED_NUM_BLOCKS: tl.constexpr,
    WRITE_PREV: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, PADDED_NUM_BLOCKS)
    mask = block < num_blocks
    values = tl.load(
        local_max_ptr + row * local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    nan_block = tl.min(tl.where(values != values, block, PADDED_NUM_BLOCKS), axis=0)
    if nan_block < PADDED_NUM_BLOCKS:
        winner = nan_block
    else:
        maximum = tl.max(values, axis=0)
        winner = tl.min(
            tl.where(mask & (values == maximum), block, PADDED_NUM_BLOCKS), axis=0
        )
    sampled = tl.load(local_argmax_ptr + row * local_argmax_stride + winner)
    tl.store(out_draft_col_ptr + row * out_draft_stride, sampled)
    if WRITE_PREV:
        tl.store(out_prev_ptr + row, sampled)


def sm80_markov_sample_batch(
    state: torch.Tensor,
    prev: torch.Tensor,
    markov_w1: torch.Tensor,
    ctx_rnn: torch.Tensor,
    rnn_w_step: torch.Tensor,
    fold_w: torch.Tensor,
    fold_b: torch.Tensor,
    base_logits: torch.Tensor,
    keep_ids: torch.Tensor,
    idx_map: torch.Tensor,
    temp: torch.Tensor,
    seeds: torch.Tensor,
    sample_pos: torch.Tensor,
    logits_cache: torch.Tensor | None,
    softcap: float,
    out_draft: torch.Tensor,
    row_mapping: torch.Tensor | None = None,
) -> None:
    """Fused batched Markov RNN + fold GEMM + softcap + pruned Gumbel sample + cache."""
    from .sm80_ops_v2 import markov_rnn_gate_256

    num_reqs, rank = state.shape
    n_spec = out_draft.shape[1]
    k_vocab = fold_w.shape[0]
    assert rank == 256 and 0 < num_reqs <= 64
    mp = 16 if num_reqs <= 16 else (32 if num_reqs <= 32 else 64)
    bn = 64 if mp == 64 else 128
    bk = 128
    num_blocks = triton.cdiv(k_vocab, bn)
    padded_blocks = triton.next_power_of_2(num_blocks)
    stream_id = current_stream().cuda_stream
    local_argmax, local_max, prev_buf = _get_markov_batch_ws(
        stream_id, state.device, num_reqs, num_blocks
    )
    has_cache = logits_cache is not None
    has_row_map = row_mapping is not None
    cache_s0 = logits_cache.stride(0) if logits_cache is not None else 0
    cache_s1 = logits_cache.stride(1) if logits_cache is not None else 0
    cur_prev = prev
    for i in range(n_spec):
        state, latent = markov_rnn_gate_256(
            state, cur_prev, markov_w1, ctx_rnn[i], rnn_w_step
        )
        _markov_fold_softcap_gumbel_batch_kernel[(num_blocks,)](
            latent,
            fold_w,
            fold_b,
            base_logits[:, i],
            keep_ids,
            idx_map[:, i],
            temp,
            seeds,
            sample_pos[:, i],
            row_mapping,
            logits_cache,
            local_argmax,
            local_max,
            latent.stride(0),
            fold_w.stride(0),
            base_logits.stride(0),
            idx_map.stride(0),
            sample_pos.stride(0),
            cache_s0,
            cache_s1,
            local_argmax.stride(0),
            i,
            num_reqs,
            k_vocab,
            softcap,
            RANK=256,
            MP=mp,
            BN=bn,
            BK=bk,
            HAS_CACHE=has_cache,
            HAS_ROW_MAP=has_row_map,
            num_warps=4,
            num_stages=2,
        )
        _markov_gumbel_combine_kernel[(num_reqs,)](
            local_argmax,
            local_argmax.stride(0),
            local_max,
            local_max.stride(0),
            out_draft[:, i],
            out_draft.stride(0),
            prev_buf,
            num_blocks,
            PADDED_NUM_BLOCKS=padded_blocks,
            WRITE_PREV=(i + 1 < n_spec),
            num_warps=4,
        )
        cur_prev = prev_buf


class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: Any = None,
    ):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None and (
            is_sm80() or not self.enable_adaptive_verification
        )
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            len(config.target_layer_ids) * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self._aux_staging = None
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )
        self._zero_markov_state = torch.zeros(
            self.max_num_reqs, config.markov_rank, dtype=self.dtype, device=device
        )
        self._sm80_tp1 = (
            is_sm80() and vllm_config.parallel_config.tensor_parallel_size == 1
        )

        # HBM-ring Hybrid SuffixDecoding + DSpark (4 MiB HBM, 0 host sync)
        self._req_states = req_states
        self._hybrid_suffix_enabled = (
            is_sm80()
            and req_states is not None
            and os.environ.get("VLLM_INKLING2_HYBRID_SUFFIX", "0") == "1"
            and self.num_speculative_steps > 0
        )
        self._suf_min_match_t0 = int(
            os.environ.get("VLLM_INKLING2_SUFFIX_MIN_MATCH_T0", "8")
        )
        self._suf_min_match_tpos = int(
            os.environ.get("VLLM_INKLING2_SUFFIX_MIN_MATCH_TPOS", "20")
        )
        self._suf_gpu_skip_markov = (
            os.environ.get("VLLM_INKLING2_SUFFIX_GPU_SKIP_MARKOV", "1") == "1"
        )
        self._suf_max_depth = int(
            os.environ.get("VLLM_INKLING2_SUFFIX_MAX_DEPTH", "24")
        )

        if self._hybrid_suffix_enabled:
            self._hbm_ring = torch.full(
                (self.max_num_reqs, _SUF_RING_SIZE),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self._hbm_base = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=device
            )
            self._hbm_len = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=device
            )
            self._suf_reset_slots: list[int] = []
            self._suf_scratch = torch.zeros(
                (self.max_num_reqs, _SUF_RING_BLOCKS),
                dtype=torch.int64,
                device=device,
            )
            self._suf_drafts = torch.zeros(
                (self.max_num_reqs, self.num_speculative_steps),
                dtype=torch.int64,
                device=device,
            )
            self._suf_num_valid = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            self._suf_match_len = torch.zeros(
                self.max_num_reqs, dtype=torch.int64, device=device
            )
            max_cache_rows = (
                self.draft_logits.values.shape[0]
                if self.draft_logits is not None
                else self.max_num_reqs
            )
            self._prev_suffix_tok = torch.full(
                (max_cache_rows, self.num_speculative_steps),
                -1,
                dtype=torch.int64,
                device=device,
            )
            self._suf_skip_flag = torch.zeros(1, dtype=torch.int32, device=device)
            self._suf_keep_dirty = torch.ones(
                max_cache_rows, dtype=torch.int32, device=device
            )
            self._suf_just_scrubbed = torch.zeros(
                max_cache_rows, dtype=torch.int32, device=device
            )

    def update_max_model_len(self, max_model_len: int) -> None:
        super().update_max_model_len(max_model_len)
        self.max_model_len = max_model_len

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        if self._hybrid_suffix_enabled:
            self._suf_reset_slots.append(req_index)

    def reset_state(self) -> None:
        super().reset_state()
        if self._hybrid_suffix_enabled:
            if self.draft_logits is not None:
                self.draft_logits.values.fill_(float("-inf"))
            self._hbm_ring.fill_(-1)
            self._hbm_base.zero_()
            self._hbm_len.zero_()
            self._suf_reset_slots.clear()
            self._prev_suffix_tok.fill_(-1)
            self._suf_skip_flag.zero_()
            self._suf_keep_dirty.fill_(1)
            self._suf_just_scrubbed.zero_()

    def _eff_k_for_batch(self, num_reqs: int, n_spec: int) -> int:
        spec_cfg = self.vllm_config.speculative_config
        if (
            self._sm80_tp1
            and n_spec >= 2
            and spec_cfg is not None
            and spec_cfg.uses_dynamic_speculative_decoding()
        ):
            if os.environ.get("VLLM_SM80_DYN_SPEC") == "1":
                if num_reqs >= 7:
                    return min(
                        max(1, int(os.environ.get("VLLM_SM80_C16_SPEC_K", "1"))),
                        n_spec,
                    )
                if 3 <= num_reqs <= 6:
                    return min(2, n_spec)
                return n_spec
            # Match speculative_config.num_speculative_tokens_per_batch_size for num_reqs >= 3
            # so c=16 (k=2) runs ONLY 2 Markov batch steps + 32 base_logits rows instead of 4 + 64!
            from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

            if getattr(self, "_static_batch_sd_lookup", None) is None:
                self._static_batch_sd_lookup = build_dynamic_sd_schedule_lookup(
                    spec_cfg.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=max(self.max_num_reqs, 256),
                    vllm_num_speculative_tokens=n_spec,
                )
            if 3 <= num_reqs < len(self._static_batch_sd_lookup):
                return min(max(1, self._static_batch_sd_lookup[num_reqs]), n_spec)
        return n_spec

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        super().on_requests_finished(finished_req_ids)
        if not self._hybrid_suffix_enabled:
            return
        req_states = self._req_states
        for req_id in finished_req_ids:
            slot = req_states.req_id_to_index.get(req_id)
            if slot is not None:
                self._suf_reset_slots.append(slot)

    def _launch_suffix_scan(
        self,
        input_batch,
        num_sampled,
        last_sampled,
        temperature,
        eff_k: int,
        write_skip: bool,
    ) -> None:
        num_reqs = input_batch.num_reqs
        req_states = self._req_states
        if self._suf_reset_slots:
            slots_gpu = async_tensor_h2d(
                self._suf_reset_slots, self.device, torch.int64
            )
            self._hbm_len.index_fill_(0, slots_gpu, 0)
            self._suf_reset_slots.clear()
        token_ids = req_states.all_token_ids.gpu
        lp1 = self.max_model_len + 1
        _suffix_sync_hbm_ring_kernel[(num_reqs,)](
            token_ids,
            token_ids.stride(0),
            input_batch.idx_mapping,
            req_states.total_len.gpu,
            self._hbm_ring,
            self._hbm_base,
            self._hbm_len,
            RING_SIZE=_SUF_RING_SIZE,
            RING_MASK=_SUF_RING_MASK,
            SYNC_BLOCK=256,
            num_warps=4,
        )
        _suffix_local_scan_ring_kernel[(num_reqs, _SUF_RING_BLOCKS)](
            self._hbm_ring,
            input_batch.idx_mapping,
            self._hbm_base,
            self._hbm_len,
            num_sampled,
            self._suf_scratch,
            self._suf_scratch.stride(0),
            lp1,
            RING_SIZE=_SUF_RING_SIZE,
            RING_MASK=_SUF_RING_MASK,
            MAX_DEPTH=self._suf_max_depth,
            BLOCK=_SUF_RING_BLOCK,
            num_warps=4,
        )
        _hybrid_suffix_finalize_ring_kernel[(num_reqs,)](
            self._hbm_ring,
            input_batch.idx_mapping,
            self._hbm_len,
            num_sampled,
            last_sampled,
            temperature,
            self._suf_scratch,
            self._suf_scratch.stride(0),
            lp1,
            self._suf_drafts,
            self._suf_drafts.stride(0),
            self._suf_num_valid,
            self._suf_match_len,
            self._suf_skip_flag,
            self._suf_min_match_t0,
            self._suf_min_match_tpos,
            RING_SIZE=_SUF_RING_SIZE,
            RING_MASK=_SUF_RING_MASK,
            WRITE_SKIP_FLAG=write_skip,
            K=eff_k,
            K_PO2=triton.next_power_of_2(eff_k),
            NB_PO2=_SUF_RING_BLOCKS,
            num_warps=4,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        use_hybrid = (
            self._hybrid_suffix_enabled
            and not self.context_kv_only
            and not dummy_run
            and not is_profile
            and (num_speculative_tokens is None or num_speculative_tokens > 0)
            and self.draft_watermarker is None
            and self.acceptance_estimator is None
            and self._d2t_scatter_index is not None
        )
        if not use_hybrid:
            if self._hybrid_suffix_enabled:
                self._suf_skip_flag.zero_()
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state=dp_sync_state,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )

        from .sm80_ops_v2 import (
            has_sm80_suffix_pointmass,
            suffix_apply_pointmass,
            suffix_cleanup_prev,
        )

        if not has_sm80_suffix_pointmass():
            self._suf_skip_flag.zero_()
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state=dp_sync_state,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )

        num_reqs = input_batch.num_reqs
        max_spec = self.num_speculative_steps
        eff_k = (
            max_spec
            if num_speculative_tokens is None
            else min(num_speculative_tokens, max_spec)
        )
        cache_vals = self.draft_logits.values if self.draft_logits is not None else None
        row_map = (
            self.draft_logits.row_mapping if self.draft_logits is not None else None
        )

        write_skip = (
            self._suf_gpu_skip_markov
            and num_reqs == 1
            and eff_k == max_spec
            and dp_sync_state is None
        )
        if not write_skip:
            self._suf_skip_flag.zero_()

        self._launch_suffix_scan(
            input_batch,
            num_sampled,
            last_sampled,
            temperature,
            eff_k,
            write_skip,
        )
        suffix_cleanup_prev(
            cache_vals,
            input_batch.idx_mapping,
            row_map,
            self._prev_suffix_tok,
            self._suf_keep_dirty,
            self._suf_just_scrubbed,
            num_reqs,
            max_spec,
        )
        out = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )
        suffix_apply_pointmass(
            cache_vals,
            self._d2t_scatter_index,
            input_batch.idx_mapping,
            row_map,
            self._suf_drafts,
            self._suf_num_valid,
            self._suf_match_len,
            self._prev_suffix_tok,
            self._suf_keep_dirty,
            self._suf_just_scrubbed,
            self.draft_tokens,
            num_reqs,
            eff_k,
            max_spec,
            1,
        )
        return out

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        if is_sm80():
            return self.dtype, float("-inf")
        return self.dtype, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        if self.use_draft_logits and getattr(model, "keep_ids", None) is not None:
            self._d2t_scatter_index = model.keep_ids
        return model

    def _store_hidden_states(
        self,
        num_tokens: int,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_graph_tokens: int = 0,
    ) -> None:
        if num_tokens <= 0:
            return
        if self._aux_staging is None and aux_hidden_states:
            torch.cat(
                [h[:num_tokens] for h in aux_hidden_states],
                dim=-1,
                out=self.hidden_states[:num_tokens],
            )
            return
        super()._store_hidden_states(
            num_tokens, last_hidden_states, aux_hidden_states, num_graph_tokens
        )

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        if (
            is_sm80()
            and getattr(self.model.model, "dflare_ws", None) is not None
            and len(self.model.model.layers) == 3
            and os.environ.get("VLLM_SM80_NO_FUSED_FRNN") != "1"
        ):
            self.model._pending_frnn = (
                self.hidden_states,
                num_reqs,
                query_start_loc,
                num_rejected,
                self.f_rnn[:num_reqs],
            )
            return
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        pending_frnn = getattr(self.model, "_pending_frnn", None)
        if pending_frnn is not None:
            self.model._pending_frnn = None
            frnn_base, nr, qsl, num_rej, out_frnn_buf = pending_frnn
            ends = qsl[1 : nr + 1] - num_rej[:nr]
            rows = frnn_base[(ends - 1).clamp_min(0).long()]
            out_frnn_buf.copy_(self.model.compute_f_rnn(rows))
        n_spec = self.num_speculative_steps
        eff_k = self._eff_k_for_batch(num_reqs, n_spec)
        num_all = num_reqs * n_spec
        if eff_k < n_spec:
            s_idx = (
                self.sample_indices[:num_all]
                .view(num_reqs, n_spec)[:, :eff_k]
                .reshape(-1)
            )
            idx_map = self.sample_idx_mapping[:num_all].view(num_reqs, n_spec)[
                :, :eff_k
            ]
            sample_pos = self.sample_pos[:num_all].view(num_reqs, n_spec)[:, :eff_k]
        else:
            s_idx = self.sample_indices[:num_all]
            idx_map = self.sample_idx_mapping[:num_all].view(num_reqs, n_spec)
            sample_pos = self.sample_pos[:num_all].view(num_reqs, n_spec)
        hidden = head_hidden[s_idx]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, eff_k, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        state = self._zero_markov_state[:num_reqs]
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        keep_ids = self._d2t_scatter_index
        if is_sm80() and keep_ids is not None:
            head = self.model.model.markov_head
            if (
                num_reqs == 1
                and self.draft_logits is not None
                and self.draft_watermarker is None
                and not self.use_fp64_gumbel
                and head.markov_rank == 256
                and head.rnn_w_step is not None
                and head.fold_weight is not None
                and head.fold_bias is not None
                and state.dtype == torch.bfloat16
            ):
                from .sm80_ops_v2 import (
                    has_fused_markov_est,
                    markov_fold_gumbel_cache_r1,
                    markov_fold_gumbel_est_cache_r1,
                    markov_rnn_gate_256,
                )

                est = self.acceptance_estimator
                if est is None or has_fused_markov_est():
                    softcap = self.model.config.final_logit_softcapping
                    cache_vals = self.draft_logits.values
                    row_map = self.draft_logits.row_mapping
                    skip_f = (
                        self._suf_skip_flag
                        if (
                            self._hybrid_suffix_enabled
                            and self._suf_gpu_skip_markov
                            and est is None
                        )
                        else None
                    )
                    dirty_f = (
                        self._suf_keep_dirty
                        if (self._hybrid_suffix_enabled and est is None)
                        else None
                    )
                    for i in range(eff_k):
                        state, latent = markov_rnn_gate_256(
                            state,
                            prev,
                            head.markov_w1.weight,
                            context[i],
                            head.rnn_w_step,
                            skip_flag=skip_f,
                        )
                        if est is None:
                            prev = markov_fold_gumbel_cache_r1(
                                latent,
                                head.fold_weight,
                                head.fold_bias,
                                base_logits[0, i],
                                keep_ids,
                                idx_map[:, i],
                                self.temperature,
                                self.seeds,
                                sample_pos[:, i],
                                row_map,
                                cache_vals,
                                i,
                                softcap,
                                self.draft_tokens[0, i : i + 1],
                                skip_flag=skip_f,
                                keep_dirty=dirty_f if i == 0 else None,
                            )
                        else:
                            prev = markov_fold_gumbel_est_cache_r1(
                                latent,
                                head.fold_weight,
                                head.fold_bias,
                                base_logits[0, i],
                                keep_ids,
                                idx_map[:, i],
                                self.temperature,
                                self.seeds,
                                sample_pos[:, i],
                                row_map,
                                cache_vals,
                                i,
                                softcap,
                                self.draft_tokens[0, i : i + 1],
                                est.features,
                                est.predictions,
                                self.draft_token_confidence_probs,
                                est.slope,
                                est.intercepts,
                            )
                    return

            if (
                1 < num_reqs <= 64
                and self.draft_logits is not None
                and self.draft_watermarker is None
                and not self.use_fp64_gumbel
                and self.acceptance_estimator is None
                and head.markov_rank == 256
                and head.rnn_w_step is not None
                and head.fold_weight is not None
                and head.fold_bias is not None
                and state.dtype == torch.bfloat16
            ):
                if self._hybrid_suffix_enabled:
                    self._suf_keep_dirty.fill_(1)
                sm80_markov_sample_batch(
                    state,
                    prev,
                    head.markov_w1.weight,
                    context,
                    head.rnn_w_step,
                    head.fold_weight,
                    head.fold_bias,
                    base_logits,
                    keep_ids,
                    idx_map,
                    self.temperature,
                    self.seeds,
                    sample_pos,
                    self.draft_logits.values,
                    self.model.config.final_logit_softcapping,
                    self.draft_tokens[:num_reqs, :eff_k],
                    row_mapping=self.draft_logits.row_mapping,
                )
                return

            if self._hybrid_suffix_enabled and self.draft_logits is not None:
                self._suf_keep_dirty.fill_(1)

            for i in range(eff_k):
                state, logits = self.model.markov_step_logits(
                    state, prev, context[i], base_logits[:, i]
                )
                if self.draft_logits is None:
                    self._maybe_predict_acceptance(
                        logits, idx_map[:, i], self._step_cols[i]
                    )
                    from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax

                    idx = greedy_argmax(logits)
                    prev = keep_ids[idx]
                elif self.draft_watermarker is None:
                    self._maybe_predict_acceptance(
                        logits, idx_map[:, i], self._step_cols[i]
                    )
                    prev = pruned_gumbel_sample_and_cache(
                        logits,
                        keep_ids,
                        idx_map[:, i],
                        self.temperature,
                        self.seeds,
                        sample_pos[:, i] - 1,
                        self.draft_logits.values,
                        self._step_cols[i],
                        self.draft_logits.row_mapping,
                        use_fp64=self.use_fp64_gumbel,
                    )
                else:
                    prev = self._sample_logits(
                        logits, idx_map[:, i], sample_pos[:, i], i
                    )
                self.draft_tokens[:num_reqs, i] = prev
            return

        for i in range(eff_k):
            state, bias = self.model.markov_step(state, prev, context[i])
            if self.draft_logits is None:
                prev = self.model.markov_argmax(base_logits[:, i], bias)
            else:
                logits = self.model.markov_logits(base_logits[:, i], bias)
                prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev
