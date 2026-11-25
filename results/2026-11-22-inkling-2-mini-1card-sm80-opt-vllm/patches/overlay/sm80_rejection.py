# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-copy BF16 RejectionSampler fast-path for SM80.

Eliminates the per-step `[M, 201024]` FP32 allocation + BF16->FP32 copy +
`_temperature_kernel` pass (`~51.5 MB` HBM traffic at `c=16`) when no masking
or logits processors are active (`not any(may_mask_logits)`), applying `/ temp`
in FP32 registers at 100% 0-ULP parity with `apply_sampling_params`.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import spec_step_seed, tl_rand32
from vllm.v1.worker.gpu.spec_decode.draft_logits import (
    DraftLogitsCache,
    draft_logits_row,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp,
    _compute_global_target_argmax,
    _compute_max_and_sumexp,
    _insert_resampled_kernel,
    _is_bonus_logit,
    _seeded_resample_argmax,
)


@triton.jit
def _bf16_global_logprobs_and_lse(
    token,
    mask,
    logit_idx,
    req_state_idx,
    draft_step,
    temp,
    target_logits_ptr,
    target_logits_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    target_logit = (
        tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + token,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        / temp
    )
    target_lse = _compute_global_logsumexp(
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    target_log_prob = target_logit - target_lse
    if HAS_DRAFT_LOGITS:
        draft_logit = (
            tl.load(
                draft_logits_ptr
                + tl.maximum(
                    draft_logits_row(req_state_idx, draft_logits_row_mapping_ptr), 0
                )
                * draft_logits_stride_0
                + draft_step * draft_logits_stride_1
                + token,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        draft_lse = _compute_global_logsumexp(
            draft_local_max_ptr,
            draft_local_max_stride,
            draft_local_sumexp_ptr,
            draft_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        draft_log_prob = draft_logit - draft_lse
    else:
        draft_log_prob = 0.0
        draft_lse = 0.0
    return target_log_prob, draft_log_prob, target_lse, draft_lse


@triton.jit
def _bf16_global_residual_mass(
    local_residual_mass_ptr,
    local_residual_mass_stride,
    prefix_joint_ratio,
    target_logits_ptr,
    target_logits_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_token,
    logit_idx,
    temp,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    if HAS_DRAFT_LOGITS:
        blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
        mask = blocks < vocab_num_blocks
        partials = tl.load(
            local_residual_mass_ptr + logit_idx * local_residual_mass_stride + blocks,
            mask=mask,
            other=0.0,
        )
        return tl.sum(partials, axis=0)
    else:
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            logit_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        target_logit = (
            tl.load(
                target_logits_ptr + logit_idx * target_logits_stride + draft_token,
            ).to(tl.float32)
            / temp
        )
        m_b = tl.exp(target_logit - target_lse)
        return prefix_joint_ratio * (1.0 - m_b)


@triton.jit(do_not_specialize=["num_logits"])
def _sm80_local_stats_bf16_kernel(
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    target_logits_ptr,
    target_logits_stride,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    temp_ptr,
    vocab_size,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    if _is_bonus_logit(expanded_local_pos_ptr, logit_idx, num_logits):
        return

    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size

    if temp == 0.0:
        target_logits = tl.load(
            target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(
            target_local_argmax_ptr + logit_idx * target_local_argmax_stride + block_idx,
            token_id,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        target_logits = (
            tl.load(
                target_logits_ptr + logit_idx * target_logits_stride + block_offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        target_max, target_sumexp = _compute_max_and_sumexp(target_logits)
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            target_max,
        )
        tl.store(
            target_local_sumexp_ptr + logit_idx * target_local_sumexp_stride + block_idx,
            target_sumexp,
        )
        if HAS_DRAFT_LOGITS:
            draft_logits = (
                tl.load(
                    draft_logits_ptr
                    + tl.maximum(
                        draft_logits_row(req_state_idx, draft_logits_row_mapping_ptr), 0
                    )
                    * draft_logits_stride_0
                    + draft_step_idx * draft_logits_stride_1
                    + block_offsets,
                    mask=mask,
                    other=float("-inf"),
                ).to(tl.float32)
                / temp
            )
            draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
            tl.store(
                draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
                draft_max,
            )
            tl.store(
                draft_local_sumexp_ptr + logit_idx * draft_local_sumexp_stride + block_idx,
                draft_sumexp,
            )


@triton.jit
def _sm80_cum_log_p_bf16_kernel(
    cumulative_log_p_ptr,
    target_logits_ptr,
    target_logits_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temp_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    log_p = tl.zeros((), tl.float32)
    for step in range(num_draft_tokens):
        logit_idx = start_idx + step
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        if draft_token >= 0:
            target_logprob, draft_logprob, _, _ = _bf16_global_logprobs_and_lse(
                draft_token,
                True,
                logit_idx,
                req_state_idx,
                step,
                temp,
                target_logits_ptr,
                target_logits_stride,
                target_local_max_ptr,
                target_local_max_stride,
                target_local_sumexp_ptr,
                target_local_sumexp_stride,
                draft_logits_ptr,
                draft_logits_stride_0,
                draft_logits_stride_1,
                draft_logits_row_mapping_ptr,
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
                HAS_DRAFT_LOGITS,
            )
            log_p = tl.minimum(log_p + (target_logprob - draft_logprob), 0.0)
        tl.store(cumulative_log_p_ptr + logit_idx, log_p)


@triton.jit(do_not_specialize=["num_logits"])
def _sm80_local_res_mass_bf16_kernel(
    local_residual_mass_ptr,
    local_residual_mass_stride,
    cumulative_log_p_ptr,
    target_logits_ptr,
    target_logits_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    draft_sampled_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    temp_ptr,
    vocab_size,
    num_logits,
    vocab_num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx == 0 or _is_bonus_logit(
        expanded_local_pos_ptr, logit_idx, num_logits
    ):
        return
    if tl.load(draft_sampled_ptr + logit_idx + 1) < 0:
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return

    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size
    target_log_probs, draft_log_probs, _, _ = _bf16_global_logprobs_and_lse(
        block_offsets,
        mask,
        logit_idx,
        req_state_idx,
        draft_step_idx,
        temp,
        target_logits_ptr,
        target_logits_stride,
        target_local_max_ptr,
        target_local_max_stride,
        target_local_sumexp_ptr,
        target_local_sumexp_stride,
        draft_logits_ptr,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_logits_row_mapping_ptr,
        draft_local_max_ptr,
        draft_local_max_stride,
        draft_local_sumexp_ptr,
        draft_local_sumexp_stride,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
        True,
    )
    p = tl.exp(tl.load(cumulative_log_p_ptr + logit_idx - 1).to(tl.float32))
    m_b = tl.exp(target_log_probs)
    m_s = tl.exp(draft_log_probs)
    partial = tl.sum(tl.maximum(p * m_b - m_s, 0.0), axis=0)
    tl.store(
        local_residual_mass_ptr + logit_idx * local_residual_mass_stride + block_idx,
        partial,
    )


@triton.jit
def _sm80_rejection_bf16_kernel(
    sampled_ptr,
    sampled_stride,
    rejected_steps_ptr,
    target_rejected_logsumexp_ptr,
    draft_rejected_logsumexp_ptr,
    target_logits_ptr,
    target_logits_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    cumulative_log_p_ptr,
    local_residual_mass_ptr,
    local_residual_mass_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0

    accepted_length = tl.zeros((), tl.int64)
    target_lse = 0.0
    draft_lse = 0.0
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        is_valid_draft = draft_sampled >= 0
        draft_sampled = tl.maximum(0, draft_sampled)
        if not is_greedy:
            verifying &= is_valid_draft
        if verifying:
            target_argmax = tl.full((), 0, tl.int64)
            accepted = False
            pos = tl.load(pos_ptr + logit_idx)
            step_seed = spec_step_seed(seed, i) if USE_BLOCK_VERIFICATION else seed
            u = tl_rand32(step_seed, pos, includes_zero=False)
            if is_greedy:
                target_argmax = _compute_global_target_argmax(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_argmax_ptr,
                    target_local_argmax_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                accepted = (target_argmax == draft_sampled) & is_valid_draft
            elif USE_BLOCK_VERIFICATION:
                prefix_joint_ratio = tl.exp(
                    tl.load(cumulative_log_p_ptr + logit_idx).to(tl.float32)
                )
                next_draft_token = tl.load(
                    draft_sampled_ptr + logit_idx + 2,
                    mask=i < num_draft_tokens - 1,
                    other=-1,
                ).to(tl.int64)
                if next_draft_token >= 0:
                    residual_mass = _bf16_global_residual_mass(
                        local_residual_mass_ptr,
                        local_residual_mass_stride,
                        prefix_joint_ratio,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        next_draft_token,
                        logit_idx + 1,
                        temp,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                    denom = residual_mass + 1.0 - prefix_joint_ratio
                    h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                else:
                    h = prefix_joint_ratio
                accepted_length = tl.where(u <= h, i + 1, accepted_length)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            else:
                target_logprob, draft_logprob, step_target_lse, step_draft_lse = (
                    _bf16_global_logprobs_and_lse(
                        draft_sampled,
                        is_valid_draft,
                        logit_idx,
                        req_state_idx,
                        i,
                        temp,
                        target_logits_ptr,
                        target_logits_stride,
                        target_local_max_ptr,
                        target_local_max_stride,
                        target_local_sumexp_ptr,
                        target_local_sumexp_stride,
                        draft_logits_ptr,
                        draft_logits_stride_0,
                        draft_logits_stride_1,
                        draft_logits_row_mapping_ptr,
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                        HAS_DRAFT_LOGITS,
                    )
                )
                target_lse = step_target_lse
                draft_lse = step_draft_lse
                accepted = target_logprob > tl.log(u) + draft_logprob

            if not USE_BLOCK_VERIFICATION or is_greedy:
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    target_argmax if is_greedy and not accepted else draft_sampled,
                )

    tl.store(rejected_steps_ptr + req_idx, accepted_length)
    if USE_BLOCK_VERIFICATION and not is_greedy and accepted_length < num_draft_tokens:
        rejected_idx = start_idx + accepted_length
        target_lse = _compute_global_logsumexp(
            target_local_max_ptr,
            target_local_max_stride,
            target_local_sumexp_ptr,
            target_local_sumexp_stride,
            rejected_idx,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        if HAS_DRAFT_LOGITS:
            draft_lse = _compute_global_logsumexp(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                rejected_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


@triton.jit
def _sm80_resample_bf16_kernel(
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    resampled_local_max_ptr,
    resampled_local_max_stride,
    target_logits_ptr,
    target_logits_stride,
    target_rejected_logsumexp_ptr,
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    draft_rejected_logsumexp_ptr,
    rejected_step_ptr,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    draft_sampled_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    cumulative_log_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_BLOCK_VERIFICATION: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(rejected_step_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx).to(tl.int64)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        return

    rejected_draft_token = tl.load(
        draft_sampled_ptr + resample_token_idx + 1,
        mask=not is_bonus,
        other=0,
    )
    is_valid_rejected_draft = rejected_draft_token >= 0

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if temp != 0.0:
        target_logits = target_logits / temp

    if is_bonus or not is_valid_rejected_draft:
        residual_logits = target_logits
    elif HAS_DRAFT_LOGITS:
        draft_logits = (
            tl.load(
                draft_logits_ptr
                + tl.maximum(
                    draft_logits_row(req_state_idx, draft_logits_row_mapping_ptr), 0
                )
                * draft_logits_stride_0
                + resample_idx * draft_logits_stride_1
                + block,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            / temp
        )
        target_lse = tl.load(target_rejected_logsumexp_ptr + req_idx)
        draft_lse = tl.load(draft_rejected_logsumexp_ptr + req_idx)
        target_log_probs = target_logits - target_lse
        if USE_BLOCK_VERIFICATION:
            log_p_tau = 0.0
            if resample_idx > 0:
                log_p_tau = tl.load(cumulative_log_p_ptr + resample_token_idx - 1).to(
                    tl.float32
                )
            target_log_probs += log_p_tau
        draft_log_probs = draft_logits - draft_lse
        ratio = tl.exp(draft_log_probs - target_log_probs)
        residual_logits = tl.where(
            ratio < 1.0,
            target_log_probs + tldevice.log1p(-ratio),
            float("-inf"),
        ).to(tl.float32)
    else:
        residual_logits = tl.where(
            block != rejected_draft_token,
            target_logits,
            float("-inf"),
        ).to(tl.float32)

    value, idx = _seeded_resample_argmax(
        residual_logits,
        block,
        mask,
        resample_token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seed_ptr,
        pos_ptr,
        vocab_size,
        USE_FP64=USE_FP64,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(
        resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + block_idx,
        token_id,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


_REJ_WS: dict[tuple[int, bool], dict[str, torch.Tensor]] = {}


def _get_rej_ws(
    dev: torch.device,
    num_logits: int,
    num_reqs: int,
    vocab_num_blocks: int,
    resample_num_blocks: int,
    use_fp64: bool,
) -> dict[str, torch.Tensor]:
    key = (dev.index or 0, use_fp64)
    ws = _REJ_WS.get(key)
    max_l = max(num_logits, 512)
    max_r = max(num_reqs, 256)
    if (
        ws is None
        or ws["t_argmax"].shape[0] < num_logits
        or ws["t_argmax"].shape[1] < vocab_num_blocks
        or ws["r_argmax"].shape[0] < num_reqs
        or ws["r_argmax"].shape[1] < resample_num_blocks
    ):
        ws = {
            "t_argmax": torch.empty(
                (max_l, vocab_num_blocks), dtype=torch.int64, device=dev
            ),
            "t_max": torch.empty(
                (max_l, vocab_num_blocks), dtype=torch.float32, device=dev
            ),
            "t_sumexp": torch.empty(
                (max_l, vocab_num_blocks), dtype=torch.float32, device=dev
            ),
            "d_max": torch.empty(
                (max_l, vocab_num_blocks), dtype=torch.float32, device=dev
            ),
            "d_sumexp": torch.empty(
                (max_l, vocab_num_blocks), dtype=torch.float32, device=dev
            ),
            "cum_log_p": torch.empty((max_l,), dtype=torch.float32, device=dev),
            "res_mass": torch.empty(
                (max_l, vocab_num_blocks), dtype=torch.float32, device=dev
            ),
            "t_rej_lse": torch.empty((max_r,), dtype=torch.float32, device=dev),
            "d_rej_lse": torch.empty((max_r,), dtype=torch.float32, device=dev),
            "r_argmax": torch.empty(
                (max_r, resample_num_blocks), dtype=torch.int64, device=dev
            ),
            "r_max": torch.empty(
                (max_r, resample_num_blocks),
                dtype=torch.float64 if use_fp64 else torch.float32,
                device=dev,
            ),
        }
        _REJ_WS[key] = ws
    return ws


def sm80_rejection_sample_bf16(
    target_logits: torch.Tensor,
    draft_logits_cache: DraftLogitsCache | None,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    *,
    use_fp64: bool = False,
    use_block_verification: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    draft_logits = None
    draft_logits_row_mapping = None
    if draft_logits_cache is not None:
        draft_logits = draft_logits_cache.values
        draft_logits_row_mapping = draft_logits_cache.row_mapping
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    draft_logits_stride_0 = 0
    draft_logits_stride_1 = 0
    if has_draft_logits := draft_logits is not None:
        draft_logits_stride_0 = draft_logits.stride(0)
        draft_logits_stride_1 = draft_logits.stride(1)
        vocab_size = min(vocab_size, draft_logits.size(-1))

    vocab_block_size = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, vocab_block_size)
    resample_block_size = 1024
    resample_num_blocks = triton.cdiv(vocab_size, resample_block_size)

    if (
        not use_fp64
        and vocab_num_blocks <= 32
        and resample_num_blocks <= 256
        and os.environ.get("VLLM_SM80_NO_CPP_REJ") != "1"
    ):
        from .sm80_ops_v2 import has_sm80_rejection_r8, sm80_rejection_sample_r8

        if has_sm80_rejection_r8():
            sampled, num_sampled, _ = sm80_rejection_sample_r8(
                target_logits=target_logits,
                draft_logits=draft_logits,
                row_mapping=draft_logits_row_mapping,
                cu_num_logits=cu_num_logits,
                idx_mapping=idx_mapping,
                expanded_idx_mapping=expanded_idx_mapping,
                expanded_local_pos=expanded_local_pos,
                temperature=temperature,
                seeds=seed,
                num_speculative_steps=num_speculative_steps,
                use_block_verification=use_block_verification,
                do_gather=False,
                draft_sampled=draft_sampled,
                pos=pos,
            )
            return sampled, num_sampled

    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    dev = target_logits.device

    ws = _get_rej_ws(
        dev, num_logits, num_reqs, vocab_num_blocks, resample_num_blocks, use_fp64
    )
    target_local_argmax = ws["t_argmax"][:num_logits]
    target_local_max = ws["t_max"][:num_logits]
    target_local_sumexp = ws["t_sumexp"][:num_logits]
    draft_local_max = ws["d_max"][:num_logits]
    draft_local_sumexp = ws["d_sumexp"][:num_logits]

    _sm80_local_stats_bf16_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_logits_row_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_logits,
        BLOCK_SIZE=vocab_block_size,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )

    if use_block_verification:
        cumulative_log_p = ws["cum_log_p"][:num_logits]
        _sm80_cum_log_p_bf16_kernel[(num_reqs,)](
            cumulative_log_p,
            target_logits,
            target_logits.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_logits_row_mapping,
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            num_warps=1,
        )
        if has_draft_logits:
            local_residual_mass = ws["res_mass"][:num_logits]
            _sm80_local_res_mass_bf16_kernel[(num_logits, vocab_num_blocks)](
                local_residual_mass,
                local_residual_mass.stride(0),
                cumulative_log_p,
                target_logits,
                target_logits.stride(0),
                target_local_max,
                target_local_max.stride(0),
                target_local_sumexp,
                target_local_sumexp.stride(0),
                draft_logits,
                draft_logits_stride_0,
                draft_logits_stride_1,
                draft_logits_row_mapping,
                draft_local_max,
                draft_local_max.stride(0),
                draft_local_sumexp,
                draft_local_sumexp.stride(0),
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                vocab_size,
                num_logits,
                vocab_num_blocks,
                BLOCK_SIZE=vocab_block_size,
                PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            )
        else:
            local_residual_mass = None
    else:
        cumulative_log_p = None
        local_residual_mass = None

    sampled = draft_sampled.new_empty(
        (num_reqs, num_speculative_steps + 1), dtype=torch.int64
    )
    num_sampled = sampled.new_empty((num_reqs,), dtype=torch.int32)
    target_rejected_logsumexp = ws["t_rej_lse"][:num_reqs]
    draft_rejected_logsumexp = ws["d_rej_lse"][:num_reqs]

    _sm80_rejection_bf16_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_logits_row_mapping,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        local_residual_mass,
        local_residual_mass.stride(0) if local_residual_mass is not None else 0,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )

    resampled_local_argmax = ws["r_argmax"][:num_reqs]
    resampled_local_max = ws["r_max"][:num_reqs]
    _sm80_resample_bf16_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_logits_row_mapping,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        vocab_size,
        BLOCK_SIZE=resample_block_size,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
    )

    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
        num_warps=1,
    )
    return sampled, num_sampled


def install_sm80_rejection_fastpath() -> None:
    if os.environ.get("VLLM_SM80_NO_FAST_REJ") == "1":
        return
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
        PROCESSED_LOGPROBS_MODES,
        RejectionSampler,
        get_max_chunk_logits,
    )

    if getattr(RejectionSampler, "_sm80_fast_rej_patched", False):
        return
    from .sm80_ops_v2 import has_sm80_rejection_r8, sm80_rejection_sample_r8

    orig_verify = RejectionSampler._verify
    orig_call = RejectionSampler.__call__
    cpp_rej_avail = has_sm80_rejection_r8()

    def _sm80_verify(
        self,
        logits: torch.Tensor,
        draft_logits: DraftLogitsCache | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampler = self.sampler
        if (
            logits.is_cuda
            and logits.dtype in (torch.bfloat16, torch.float32)
            and logits.ndim == 2
            and logits.shape[0] >= 1
            and logits.stride(-1) == 1
            and self.fly_window_size == 0
            and self.synthetic_conditional_rates is None
            and self.watermark_key is None
            and not sampler.return_sampling_mask
            and not bool(np.any(sampler.may_mask_logits[idx_mapping_np]))
            and (
                sampler.logprobs_mode not in PROCESSED_LOGPROBS_MODES
                or sampler.get_logprobs_dims(idx_mapping_np) is None
            )
        ):
            sampled, num_sampled = sm80_rejection_sample_bf16(
                logits,
                draft_logits,
                draft_sampled,
                cu_num_logits,
                pos,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                sampler.sampling_states.temperature.gpu,
                sampler.sampling_states.seeds.gpu,
                self.num_speculative_steps,
                use_fp64=sampler.use_fp64_gumbel,
                use_block_verification=self.use_block_verification,
            )
            if sampler.trace_replay_state is not None:
                sampler.trace_replay_state.apply_trace_to_verified(
                    sampled, num_sampled, draft_sampled, cu_num_logits, idx_mapping
                )
            return logits, sampled, num_sampled

        return orig_verify(
            self,
            logits,
            draft_logits,
            draft_sampled,
            pos,
            cu_num_logits,
            idx_mapping,
            idx_mapping_np,
            expanded_idx_mapping,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )

    def _sm80_call(
        self,
        logits: torch.Tensor,
        input_batch,
        draft_logits: DraftLogitsCache | None = None,
    ):
        sampler = self.sampler
        num_reqs = input_batch.num_reqs
        num_logits = logits.shape[0] if logits.ndim == 2 else 0
        if (
            cpp_rej_avail
            and num_reqs > 0
            and num_logits > 0
            and not sampler.compute_nans
            and not sampler.return_sampling_mask
            and not sampler.use_fp64_gumbel
            and sampler.trace_replay_state is None
            and not self.enable_adaptive_verification
            and self.fly_window_size == 0
            and self.synthetic_conditional_rates is None
            and self.watermark_key is None
            and logits.is_cuda
            and logits.dtype in (torch.bfloat16, torch.float32)
            and logits.ndim == 2
            and logits.stride(-1) == 1
            and num_logits <= (1073741824 // (logits.shape[1] * 4) or 1)
            and int(input_batch.cu_num_logits_np[0]) == 0
            and int(input_batch.cu_num_logits_np[num_reqs]) == num_logits
            and (
                not sampler.may_mask_logits.any()
                or not bool(sampler.may_mask_logits[input_batch.idx_mapping_np].any())
            )
            and sampler.get_logprobs_dims(input_batch.idx_mapping_np) is None
            and os.environ.get("VLLM_SM80_NO_CPP_REJ") != "1"
        ):
            dl_vals = draft_logits.values if draft_logits is not None else None
            dl_rmap = draft_logits.row_mapping if draft_logits is not None else None
            eff_vocab = (
                min(logits.shape[1], dl_vals.shape[-1])
                if dl_vals is not None
                else logits.shape[1]
            )
            if eff_vocab <= 262144:
                temp_st = sampler.sampling_states.temperature
                all_greedy = False
                if num_reqs == 1:
                    all_greedy = float(temp_st.np[input_batch.idx_mapping_np[0]]) == 0.0
                elif not temp_st.np[input_batch.idx_mapping_np].any():
                    all_greedy = True
                sampled, num_sampled, num_rejected = sm80_rejection_sample_r8(
                    target_logits=logits,
                    draft_logits=dl_vals,
                    row_mapping=dl_rmap,
                    cu_num_logits=input_batch.cu_num_logits,
                    idx_mapping=input_batch.idx_mapping,
                    expanded_idx_mapping=input_batch.expanded_idx_mapping,
                    expanded_local_pos=input_batch.expanded_local_pos,
                    temperature=temp_st.gpu,
                    seeds=sampler.sampling_states.seeds.gpu,
                    num_speculative_steps=self.num_speculative_steps,
                    use_block_verification=self.use_block_verification and (not all_greedy),
                    do_gather=True,
                    input_ids=input_batch.input_ids,
                    positions=input_batch.positions,
                    logits_indices=input_batch.logits_indices,
                    prefill_len=sampler.req_states.prefill_len.gpu,
                    seq_lens=input_batch.seq_lens,
                )
                return SamplerOutput(
                    sampled_token_ids=sampled,
                    logprobs_tensors=None,
                    num_nans=None,
                    num_sampled=num_sampled,
                    num_rejected=num_rejected,
                    sampling_mask_tensors=None,
                )
        return orig_call(self, logits, input_batch, draft_logits)

    RejectionSampler._verify = _sm80_verify  # type: ignore[method-assign]
    RejectionSampler.__call__ = _sm80_call  # type: ignore[method-assign]
    RejectionSampler._sm80_fast_rej_patched = True  # type: ignore[attr-defined]
