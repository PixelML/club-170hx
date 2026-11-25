# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 Triton stand-in for `_C::vocab_parallel_embedding` (no sitecustomize required)."""

import os

import torch

from vllm.triton_utils import tl, triton

from .nvidia.ops import is_sm80

_INSTALLED = False
_SCHED_HOOKED = False
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


def install_sm80_spec_schedule_hooks() -> None:
    """Optional experimental dynamic spec-schedule hook (opt-in via VLLM_SM80_DYN_SPEC=1)."""
    global _SCHED_HOOKED
    if _SCHED_HOOKED or not is_sm80():
        return
    if os.environ.get("VLLM_SM80_DYN_SPEC") != "1":
        return

    from vllm.config import VllmConfig
    from vllm.v1.core.sched.scheduler import Scheduler

    _orig_get_ql = VllmConfig.get_decode_cudagraph_query_lens

    def _sm80_get_decode_cudagraph_query_lens(
        self: VllmConfig, decode_query_len: int
    ) -> dict[int, int]:
        res = dict(_orig_get_ql(self, decode_query_len))
        if os.environ.get("VLLM_SM80_NO_DYN_SPEC") == "1":
            return res
        spec_cfg = self.speculative_config
        n_new = decode_query_len - self.num_speculative_tokens
        if (
            spec_cfg is None
            or spec_cfg.method != "dspark"
            or not spec_cfg.uses_dynamic_speculative_decoding()
            or self.parallel_config.tensor_parallel_size != 1
            or self.num_speculative_tokens < 2
            or n_new <= 0
        ):
            return res
        max_seqs = self.scheduler_config.max_num_seqs
        c16_k = min(
            max(1, int(os.environ.get("VLLM_SM80_C16_SPEC_K", "1"))),
            self.num_speculative_tokens,
        )
        res[c16_k + n_new] = max(res.get(c16_k + n_new, 0), max_seqs)
        res[2 + n_new] = max(res.get(2 + n_new, 0), min(max_seqs, 8))
        if self.num_speculative_tokens >= 3:
            res[3 + n_new] = max(res.get(3 + n_new, 0), min(max_seqs, 8))
        if self.num_speculative_tokens >= 4:
            res[4 + n_new] = max(res.get(4 + n_new, 0), min(max_seqs, 2))
        return dict(sorted(res.items()))

    VllmConfig.get_decode_cudagraph_query_lens = (  # type: ignore[method-assign]
        _sm80_get_decode_cudagraph_query_lens
    )

    _orig_sched_init = Scheduler.__init__
    _orig_sched_schedule = Scheduler.schedule
    _orig_sched_update = Scheduler.update_from_output

    def _sm80_sched_init(self: Scheduler, *args, **kwargs) -> None:
        _orig_sched_init(self, *args, **kwargs)
        spec_cfg = self.vllm_config.speculative_config
        self._sm80_dyn_active = bool(
            os.environ.get("VLLM_SM80_NO_DYN_SPEC") != "1"
            and spec_cfg is not None
            and spec_cfg.method == "dspark"
            and spec_cfg.uses_dynamic_speculative_decoding()
            and self.vllm_config.parallel_config.tensor_parallel_size == 1
            and self.num_spec_tokens >= 2
            and self.dynamic_sd_lookup is not None
        )
        if self._sm80_dyn_active:
            max_seqs = self.scheduler_config.max_num_seqs
            c16_k = min(
                max(1, int(os.environ.get("VLLM_SM80_C16_SPEC_K", "1"))),
                self.num_spec_tokens,
            )
            lookup = [0] * (max_seqs + 1)
            for b in range(1, max_seqs + 1):
                if b <= 2:
                    lookup[b] = min(3, self.num_spec_tokens)
                elif b <= 6:
                    lookup[b] = min(2, self.num_spec_tokens)
                else:
                    lookup[b] = c16_k
            self.dynamic_sd_lookup = lookup
            self._sm80_req_ema: dict[str, float] = {}
            self._sm80_req_k: dict[str, int] = {}

    def _sm80_sched_schedule(self: Scheduler, *args, **kwargs):
        if getattr(self, "_sm80_dyn_active", False) and self.dynamic_sd_lookup is not None:
            if len(self.running) == 1 and not self.waiting:
                req = self.running[0]
                rid = req.request_id
                ema = self._sm80_req_ema.get(rid, 0.48)
                cur_k = self._sm80_req_k.get(rid, min(3, self.num_spec_tokens))
                temp = (
                    req.sampling_params.temperature
                    if req.sampling_params is not None
                    else 0.6
                )
                if temp == 0.0:
                    if ema >= 0.50 and self.num_spec_tokens >= 4:
                        cur_k = 4
                    elif ema >= 0.36:
                        cur_k = min(3, self.num_spec_tokens)
                    else:
                        cur_k = min(2, self.num_spec_tokens)
                else:
                    if cur_k >= 3 and ema < 0.41:
                        cur_k = min(2, self.num_spec_tokens)
                    elif cur_k <= 2 and ema >= 0.46:
                        cur_k = min(3, self.num_spec_tokens)
                    else:
                        cur_k = min(max(2, cur_k), min(3, self.num_spec_tokens))
                self._sm80_req_k[rid] = cur_k
                self.dynamic_sd_lookup[1] = cur_k
            else:
                self.dynamic_sd_lookup[1] = min(3, self.num_spec_tokens)
        return _orig_sched_schedule(self, *args, **kwargs)

    def _sm80_sched_update(self: Scheduler, scheduler_output, model_runner_output):
        res = _orig_sched_update(self, scheduler_output, model_runner_output)
        if getattr(self, "_sm80_dyn_active", False):
            req_ids = model_runner_output.req_ids
            sampled = model_runner_output.sampled_token_ids
            spec_map = scheduler_output.scheduled_spec_decode_tokens
            if len(req_ids) == 1 and sampled and spec_map:
                rid = req_ids[0]
                drafts = spec_map.get(rid)
                toks = sampled[0]
                if drafts and toks:
                    kd = len(drafts)
                    na = min(kd, max(0, len(toks) - 1))
                    r = na / kd
                    prev = self._sm80_req_ema.get(rid, 0.48)
                    self._sm80_req_ema[rid] = 0.72 * prev + 0.28 * r
            for fid in scheduler_output.finished_req_ids or ():
                self._sm80_req_ema.pop(fid, None)
                self._sm80_req_k.pop(fid, None)
            if len(self._sm80_req_ema) > 128:
                self._sm80_req_ema.clear()
                self._sm80_req_k.clear()
        return res

    Scheduler.__init__ = _sm80_sched_init  # type: ignore[method-assign]
    Scheduler.schedule = _sm80_sched_schedule  # type: ignore[method-assign]
    Scheduler.update_from_output = _sm80_sched_update  # type: ignore[method-assign]
    _SCHED_HOOKED = True


def install_sm80_tp_ops() -> None:
    """Register `_C::vocab_parallel_embedding` CUDA impl + SM80 fast hooks (idempotent)."""
    global _INSTALLED
    if not is_sm80():
        return
    try:
        install_sm80_spec_schedule_hooks()
    except Exception:
        pass
    if _INSTALLED:
        return
    import vllm._custom_ops  # noqa: F401

    lib_c = torch.library.Library("_C", "IMPL")
    lib_c.impl("vocab_parallel_embedding", vocab_parallel_embedding, "CUDA")
    _LIBS.append(lib_c)
    try:
        from .sm80_rejection import install_sm80_rejection_fastpath

        install_sm80_rejection_fastpath()
    except Exception:
        pass
    _INSTALLED = True
