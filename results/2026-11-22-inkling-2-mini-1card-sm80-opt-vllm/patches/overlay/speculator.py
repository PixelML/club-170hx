# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator for Inkling2 draft with SM80 Zero-Scatter Pruned-Vocab Gumbel Sampler."""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _gumbel_combine_kernel,
    gumbel_block_argmax,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


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
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(pruned_vocab, BLOCK_SIZE)
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
        BLOCK_SIZE=BLOCK_SIZE,
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


class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None
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

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Initialize non-pruned columns once to -inf so sparse 49K cache writes
        # never need a 201K scatter buffer or 201K -inf overwrites!
        return self.dtype, float("-inf")

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        if self.use_draft_logits and getattr(model, "keep_ids", None) is not None:
            self._d2t_scatter_index = model.keep_ids
        return model

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_base_logits(hidden).view(num_reqs, n_spec, -1)
        context = self.model.model.markov_head.context(hidden, self.f_rnn[:num_reqs])
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        state = hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        keep_ids = self._d2t_scatter_index
        head = self.model.model.markov_head
        if (
            num_reqs == 1
            and keep_ids is not None
            and self.draft_logits is not None
            and self.draft_watermarker is None
            and not self.use_fp64_gumbel
            and head.markov_rank == 256
            and head.rnn_w_step is not None
            and head.fold_weight is not None
            and head.fold_bias is not None
            and state.dtype == torch.bfloat16
        ):
            from vllm.models.inkling2.sm80_ops_v2 import (
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
                for i in range(n_spec):
                    state, latent = markov_rnn_gate_256(
                        state, prev, head.markov_w1.weight, context[i], head.rnn_w_step
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
        for i in range(n_spec):
            state, logits = self.model.markov_step_logits(state, prev, context[i], base_logits[:, i])
            if self.draft_logits is None:
                self._maybe_predict_acceptance(logits, idx_map[:, i], self._step_cols[i])
                from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax
                idx = greedy_argmax(logits)
                prev = self.model.keep_ids[idx] if self.model.keep_ids is not None else idx
            elif keep_ids is not None and self.draft_watermarker is None:
                self._maybe_predict_acceptance(logits, idx_map[:, i], self._step_cols[i])
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
                prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev
