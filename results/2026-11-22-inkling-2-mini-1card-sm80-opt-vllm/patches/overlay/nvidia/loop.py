# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2 checkpoints (`loop_config`).

The core layers `[core_start, core_end)` run `num_passes` times: a prelude,
then each pass from the AttnRes state after the prelude, then the coda from
the last pass's state. Each pass of a core layer has its own KV and
conv-state caches. In passes after the first, every core sublayer input
(attention q/k/v and MLP) is blended with a message: a normalized-value
AttnRes read of the previous pass's blocks, merged by a gated low-rank
controller. Each sublayer's short conv runs fused with the next AttnRes mix
(and small batches at TP > 1 with its all-reduce); without sequence parallel.
"""

from collections.abc import Sequence

import torch
from torch import nn

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.triton_utils import tl as _tl
from vllm.triton_utils import tldevice as _tldevice
from vllm.triton_utils import triton as _triton

from ..configs import Inkling2ModelConfig
from ..quant_config import Inkling2QuantConfig
from .model import (
    Inkling2DecoderLayer,
    Inkling2ForConditionalGeneration,
    Inkling2Model,
)
from .ops import is_sm80
from .ops.attn_res import attn_res
from .ops.linear import linear_k_groups
from .ops.ll_gemm_sm80 import sm80_bf16_ll_gemm, use_sm80_bf16_ll_gemm
from .ops.ll_gemm_sm120 import sm12x_bf16_ll_gemm, use_sm12x_bf16_ll_gemm
from .ops.loop import blend_rms_norm, pair_rms_norm
from .ops.norm import rsqrt as _rsqrt
from .sconv import conv_metadata

try:
    from ..sm80_ops_v2 import (
        bank_rank2_finalize_r7 as _sm80_bank_rank2_finalize_r7,
        bank_stats_r7_2304 as _sm80_bank_stats_r7_2304,
        fast_blend_r6_2304 as _sm80_fast_blend_r6_2304,
        pair_rms_norm_rank2_2304_r7 as _sm80_pair_rms_norm_rank2_r7,
        up_blend_rank2_rms_2304_r7 as _sm80_up_blend_rank2_rms_r7,
    )
except Exception:
    _sm80_bank_rank2_finalize_r7 = None
    _sm80_bank_stats_r7_2304 = None
    _sm80_fast_blend_r6_2304 = None
    _sm80_pair_rms_norm_rank2_r7 = None
    _sm80_up_blend_rank2_rms_r7 = None

LOOP_STREAMS = ("attn", "k", "v", "mlp")


@_triton.jit
def _bank_attn_res_1pass_kernel(
    Bank,
    Queries,
    Out,
    stride_b0,
    T,
    eps,
    D: _tl.constexpr,
    Q_PER_CTA: _tl.constexpr,
    BLOCK_A: _tl.constexpr,
    BLOCK_B: _tl.constexpr,
):
    q_base = _tl.program_id(0) * Q_PER_CTA
    t = _tl.program_id(1).to(_tl.int64)
    row_off = t * D
    da = _tl.arange(0, BLOCK_A)
    db = BLOCK_A + _tl.arange(0, BLOCK_B)
    mask_a = da < D
    mask_b = db < D

    xa0 = _tl.load(Bank + row_off + da, mask_a, other=0).to(_tl.float32)
    xb0 = _tl.load(Bank + row_off + db, mask_b, other=0).to(_tl.float32)
    xa1 = _tl.load(Bank + stride_b0 + row_off + da, mask_a, other=0).to(_tl.float32)
    xb1 = _tl.load(Bank + stride_b0 + row_off + db, mask_b, other=0).to(_tl.float32)

    rstd0 = _rsqrt(_tl.fma(_tl.sum(xa0 * xa0) + _tl.sum(xb0 * xb0), 1.0 / D, eps))
    rstd1 = _rsqrt(_tl.fma(_tl.sum(xa1 * xa1) + _tl.sum(xb1 * xb1), 1.0 / D, eps))

    for qi in _tl.static_range(Q_PER_CTA):
        q_off = (q_base + qi) * D
        qa = _tl.load(Queries + q_off + da, mask_a, other=0)
        qb = _tl.load(Queries + q_off + db, mask_b, other=0)
        l0 = (_tl.sum(qa * xa0) + _tl.sum(qb * xb0)) * rstd0
        l1 = (_tl.sum(qa * xa1) + _tl.sum(qb * xb1)) * rstd1
        mx = _tl.maximum(l0, l1)
        e0 = _tldevice.exp(l0 - mx)
        e1 = _tldevice.exp(l1 - mx)
        inv_sum = _tl.div_rn(1.0, e0 + e1)
        p0 = e0 * rstd0
        p1 = e1 * rstd1
        acc_a = (
            _tl.fma(p1, xa1, _tl.fma(p0, xa0, _tl.zeros((BLOCK_A,), _tl.float32)))
            * inv_sum
        )
        acc_b = (
            _tl.fma(p1, xb1, _tl.fma(p0, xb0, _tl.zeros((BLOCK_B,), _tl.float32)))
            * inv_sum
        )
        out_ptr = Out + (q_base + qi).to(_tl.int64) * T * D + row_off
        _tl.store(out_ptr + da, acc_a, mask_a)
        _tl.store(out_ptr + db, acc_b, mask_b)


def sm80_bank_attn_res(bank: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Single-pass 0-ULP SM80 bank AttnRes (N_SRC=2, Q=4, D=2304, fp32 out)."""
    t = bank.shape[1]
    if (
        is_sm80()
        and 1 <= t <= 64
        and bank.shape == (2, t, 2304)
        and queries.shape == (4, 2304)
        and bank.dtype == torch.bfloat16
        and queries.dtype == torch.float32
        and bank.stride(1) == 2304
        and bank.stride(2) == 1
        and queries.is_contiguous()
    ):
        out = torch.empty((4, t, 2304), dtype=torch.float32, device=bank.device)
        qpc = 2 if t > 16 else 1
        _bank_attn_res_1pass_kernel[(4 // qpc, t)](
            bank,
            queries,
            out,
            bank.stride(0),
            t,
            1e-6,
            D=2304,
            Q_PER_CTA=qpc,
            BLOCK_A=2048,
            BLOCK_B=256,
            num_warps=8,
            enable_fp_fusion=False,
        )
        return out
    return attn_res(bank[:1], bank[1], None, queries, None, 0.0, True)


def sm80_fast_blend(
    ctrl_down: torch.Tensor,
    ctrl_up: torch.Tensor,
    dest: torch.Tensor,
    message: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    """2-launch SM80 Tensor-Core fused loop controller:
    (pair_rms_norm + down) + (up + unfused tanh blend + RMSNorm)."""
    if not is_sm80() or _sm80_fast_blend_r6_2304 is None:
        return None
    return _sm80_fast_blend_r6_2304(
        ctrl_down, ctrl_up, dest, message, norm_weight, norm_eps
    )


_Q_SPLIT_CACHE: dict[tuple[int, tuple[int, ...], int, int], torch.Tensor] = {}


def _get_q_split_bf16_t(message_queries: torch.Tensor) -> torch.Tensor:
    key = (
        message_queries.data_ptr(),
        tuple(message_queries.shape),
        message_queries.device.index or 0,
        message_queries._version,
    )
    qt = _Q_SPLIT_CACHE.get(key)
    if qt is None:
        if len(_Q_SPLIT_CACHE) >= 8:
            _Q_SPLIT_CACHE.clear()
        q_flat = message_queries.reshape(-1, message_queries.shape[-1])
        q_hi = q_flat.to(torch.bfloat16)
        q_lo = (q_flat - q_hi.to(torch.float32)).to(torch.bfloat16)
        qt = torch.cat((q_hi, q_lo), dim=0).t().contiguous()
        _Q_SPLIT_CACHE[key] = qt
    return qt


def _sm80_blend_rms_norm_rank2(
    dest: torch.Tensor,
    bank: torch.Tensor,
    c_raw: torch.Tensor,
    c_norm: torch.Tensor,
    ctrl_down: torch.Tensor,
    ctrl_up: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    """3-call zero-spill CUDA MMA SM80 rank-2 prefill controller blend + RMSNorm:
    1) pair_rms_norm_rank2_2304_r7 (288-thr uint4, bcast3-specialized)
    2) torch.bmm(pair, ctrl_down.T)
    3) up_blend_rank2_rms_2304_r7 (1 ctypes call -> CUDA MMA up+tanh+blend + RMSNorm)."""
    if _sm80_pair_rms_norm_rank2_r7 is None or _sm80_up_blend_rank2_rms_r7 is None:
        return None
    s, m, d = dest.shape
    if (
        d != 2304
        or m <= 0
        or ctrl_down.shape != (s, 64, 4608)
        or ctrl_up.shape != (s, 4608, 64)
        or not ctrl_down.is_contiguous()
        or not ctrl_up.is_contiguous()
        or dest.stride(1) != 2304
        or dest.stride(2) != 1
        or norm_weight.stride(1) != 1
    ):
        return None

    pair = _sm80_pair_rms_norm_rank2_r7(dest, bank, c_norm, 1e-6)
    if s == 1:
        hidden = torch.mm(pair[0], ctrl_down[0].t()).unsqueeze(0)
    else:
        hidden = torch.bmm(pair, ctrl_down.transpose(1, 2))
    del pair
    return _sm80_up_blend_rank2_rms_r7(
        hidden, ctrl_up, dest, bank, c_raw, norm_weight, norm_eps
    )


def sm80_prepare_pass_bank(
    bank: torch.Tensor,
    message_queries: torch.Tensor,
    q_split_bf16_t: torch.Tensor | None = None,
) -> tuple:
    """Pass-hoisted SM80 bank preparation:
    - Decode (1 <= T <= 256): single 0-ULP Triton launch over all 48 queries.
    - Prefill (T > 256): rank-2 scalar coefficients (c_raw, c_norm) via split-BF16 GEMM."""
    t = bank.shape[1]
    if (
        is_sm80()
        and bank.ndim == 3
        and bank.shape[0] == 2
        and bank.shape[2] == 2304
        and message_queries.ndim == 3
        and message_queries.shape[2] == 2304
        and bank.dtype == torch.bfloat16
        and message_queries.dtype == torch.float32
        and bank.stride(1) == 2304
        and bank.stride(2) == 1
        and message_queries.is_contiguous()
    ):
        num_layers, num_streams, d = message_queries.shape
        q_tot = num_layers * num_streams
        if 1 <= t <= 256:
            qpc = 2 if t <= 8 else 8
            while q_tot % qpc != 0:
                qpc -= 1
            out = torch.empty(
                (num_layers, num_streams, t, d),
                dtype=torch.float32,
                device=bank.device,
            )
            _bank_attn_res_1pass_kernel[(q_tot // qpc, t)](
                bank,
                message_queries.view(q_tot, d),
                out,
                bank.stride(0),
                t,
                1e-6,
                D=2304,
                Q_PER_CTA=qpc,
                BLOCK_A=2048,
                BLOCK_B=256,
                num_warps=8,
                enable_fp_fusion=False,
            )
            return ("dense", out)
        if (
            t > 256
            and _sm80_bank_stats_r7_2304 is not None
            and _sm80_bank_rank2_finalize_r7 is not None
        ):
            bank_c = bank if bank.is_contiguous() else bank.contiguous()
            stats = _sm80_bank_stats_r7_2304(bank_c, 1e-6)
            if stats is not None:
                qt = (
                    q_split_bf16_t
                    if q_split_bf16_t is not None
                    else _get_q_split_bf16_t(message_queries)
                )
                dots = torch.mm(
                    bank_c.view(2 * t, 2304), qt, out_dtype=torch.float32
                )
                res = _sm80_bank_rank2_finalize_r7(
                    dots, stats, num_layers, num_streams, 1e-6
                )
                if res is not None:
                    c_raw, c_norm = res
                    return ("rank2", bank_c, c_raw, c_norm)
    return ("fallback", bank, message_queries)


def sm80_blend_pass_layer(
    pass_state: tuple,
    layer_rel_idx: int,
    streams: slice,
    ctrl_down: torch.Tensor,
    ctrl_up: torch.Tensor,
    dest: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Blend one layer's streams using pass-hoisted `pass_state`."""
    kind = pass_state[0]
    if kind == "dense":
        message = pass_state[1][layer_rel_idx, streams]
        out = sm80_fast_blend(
            ctrl_down, ctrl_up, dest, message, norm_weight, norm_eps
        )
        if out is not None:
            return out
    elif kind == "rank2":
        _, bank, c_raw_all, c_norm_all = pass_state
        c_raw = c_raw_all[layer_rel_idx, streams]
        c_norm = c_norm_all[layer_rel_idx, streams]
        out = _sm80_blend_rms_norm_rank2(
            dest, bank, c_raw, c_norm, ctrl_down, ctrl_up, norm_weight, norm_eps
        )
        if out is not None:
            return out
        # Fallback if non-standard shape
        message = (
            c_raw[..., :1] * bank[0].to(torch.float32)
            + c_raw[..., 1:] * bank[1].to(torch.float32)
        )
    else:
        _, bank, message_queries = pass_state
        message = sm80_bank_attn_res(bank, message_queries[layer_rel_idx])[streams]

    pair = pair_rms_norm(dest, message)
    if use_sm12x_bf16_ll_gemm(pair, ctrl_down):
        hidden = sm12x_bf16_ll_gemm(pair, ctrl_down)
        gain_forget = sm12x_bf16_ll_gemm(hidden, ctrl_up, out_dtype=torch.float32)
    elif use_sm80_bf16_ll_gemm(pair, ctrl_down):
        hidden = sm80_bf16_ll_gemm(pair, ctrl_down)
        gain_forget = sm80_bf16_ll_gemm(hidden, ctrl_up, out_dtype=torch.float32)
    else:
        hidden = torch.bmm(pair, ctrl_down.transpose(1, 2))
        gain_forget = torch.bmm(
            hidden, ctrl_up.transpose(1, 2), out_dtype=torch.float32
        )
    return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)


class Inkling2MessageController(nn.Module):
    """Blends a loop message into a core sublayer input: from the rank-`r`
    projection `h` of the RMS-normalized `[dest, message]`,
    `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message` in fp32.

    The weights are views of the layer's stream-stacked weights
    (`Inkling2LoopDecoderLayer`), which run every stream at once; `forward`
    is the per-stream reference.
    """

    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)

    def forward(self, dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """fp32 `(T, D)` `dest` and `message` -> fp32 `(T, D)`."""
        pair = torch.stack((dest, message), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        normalized = pair.to(torch.bfloat16).flatten(1)
        hidden = linear_k_groups(normalized, self.down_weight, torch.bfloat16)
        gain = torch.mm(hidden, self.gain_weight.t(), out_dtype=torch.float32)
        forget = torch.mm(hidden, self.forget_weight.t(), out_dtype=torch.float32)
        return (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    """A loop core layer: per-pass caches and a message controller per
    stream (`LOOP_STREAMS`), whose weights are stacked over the streams."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        streams = len(LOOP_STREAMS)
        # down (S, r, 2D) and [gain; forget] (S, 2D, r), bf16.
        self.register_buffer(
            "controller_down",
            torch.empty(streams, rank, 2 * width, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "controller_gain_forget",
            torch.empty(streams, 2 * width, rank, dtype=torch.bfloat16),
            persistent=False,
        )
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2MessageController(
                    self.controller_down[i],
                    self.controller_gain_forget[i, :width],
                    self.controller_gain_forget[i, width:],
                )
                for i, stream in enumerate(LOOP_STREAMS)
            }
        )

    def process_weights_after_loading(self) -> None:
        # The controllers' weights must still alias the stacked ones.
        for i, controller in enumerate(self.controllers.values()):
            down, gain_forget = self.controller_down, self.controller_gain_forget
            assert controller.down_weight.data_ptr() == down[i].data_ptr()
            assert controller.gain_weight.data_ptr() == gain_forget[i].data_ptr()

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The `streams` controllers on fp32 `(S, T, D)` `dest` (may
        broadcast over streams) and `message`, each output RMS-normalized
        with its row of `norm_weight` `(S, D)`: bf16 `(S, T, D)`."""
        down = self.controller_down[streams]
        up = self.controller_gain_forget[streams]
        if is_sm80():
            out = sm80_fast_blend(down, up, dest, message, norm_weight, norm_eps)
            if out is not None:
                return out
        pair = pair_rms_norm(dest, message)
        if use_sm12x_bf16_ll_gemm(pair, down):
            hidden = sm12x_bf16_ll_gemm(pair, down)
            gain_forget = sm12x_bf16_ll_gemm(hidden, up, out_dtype=torch.float32)
        elif use_sm80_bf16_ll_gemm(pair, down):
            hidden = sm80_bf16_ll_gemm(pair, down)
            gain_forget = sm80_bf16_ll_gemm(hidden, up, out_dtype=torch.float32)
        else:
            hidden = torch.bmm(pair, down.transpose(1, 2))
            gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)


class Inkling2LoopModel(Inkling2Model):
    seq_parallel_paths = False

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        # Serving options set through --hf-overrides skip the constructor.
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        # Message pseudo-queries (raw), two per core layer: attn/mlp and k/v.
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        # Per core layer, the `LOOP_STREAMS` queries scaled by D**-0.5.
        self.register_buffer(
            "message_queries",
            torch.zeros(
                len(self.loop_core),
                len(LOOP_STREAMS),
                config.hidden_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self._q_split_bf16_t: torch.Tensor | None = None

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)
        self._q_split_bf16_t = None
        _Q_SPLIT_CACHE.clear()
        if is_sm80():
            self._q_split_bf16_t = _get_q_split_bf16_t(self.message_queries)

    def _get_q_split_bf16_t(self) -> torch.Tensor:
        qt = self._q_split_bf16_t
        if qt is None or qt.device != self.message_queries.device:
            qt = _get_q_split_bf16_t(self.message_queries)
            self._q_split_bf16_t = qt
        return qt

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)
        return super()._make_layer(config, layer_id, quant_config, prefix)

    def _max_loop_blocks(self) -> int:
        """Completed AttnRes blocks a loop forward holds at most: each pass
        restarts from the prelude's, and the coda continues the last pass."""
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs: torch.Tensor | None,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: nn.Module | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
        plan: list[tuple[bool, bool]],
        pass_state: tuple | None = None,
    ) -> tuple[int, torch.Tensor]:
        """Run decoder layer `i` in loop pass `pass_idx` on its `inputs`
        (the attention inputs; for a visit with messages from `bank`, the
        fp32 mixes they blend into). Returns the number of completed AttnRes
        blocks after it and the mixes of its output with `next_query` through
        `next_norm` (fp32 without it), the next visit's inputs."""
        layer = self.layers[i]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        workspace = self.allreduce_workspace
        (attn_fused, attn_two_shot), (moe_fused, moe_two_shot) = plan
        has_msg = bank is not None
        messages = None
        layer_rel_idx = i - self.loop_core.start
        attn_inputs: Sequence[torch.Tensor]
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            if pass_state is not None:
                attn_inputs = sm80_blend_pass_layer(
                    pass_state,
                    layer_rel_idx,
                    slice(0, 3),
                    layer.controller_down[0:3],
                    layer.controller_gain_forget[0:3],
                    dests,
                    norm_weight,
                    norm.eps,
                )
            else:
                # All four streams' messages: a mix of the bank's two blocks.
                q_msg = self.message_queries[layer_rel_idx]
                if is_sm80():
                    messages = sm80_bank_attn_res(bank, q_msg)
                else:
                    messages = attn_res(
                        bank[:1],
                        bank[1],
                        None,
                        q_msg,
                        None,
                        0.0,
                        True,
                    )
                attn_inputs = layer.blend(
                    slice(0, 3), dests, messages[:3], norm_weight, norm.eps
                )
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

        # The MLP input mixes the blocks with (partial +) attn_out.
        first = i % self.block_size == 0
        attn_out = torch.empty_like(partial)
        mix_args = (
            positions,
            attn_out,
            stack[: num_blocks + first],
            None if first else partial,
            self.attn_res.mlp_queries[i],
            None if has_msg else layer.mlp_norm,
        )
        if attn_fused:
            assert workspace is not None
            mlp_in = layer.attn_sconv.all_reduce_mix(
                workspace,
                *mix_args,
                x=attn_partial,
                two_shot=attn_two_shot,
                owner=owner,
            )
        else:
            attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
            mlp_in = layer.attn_sconv.conv_mix(attn_reduced, *mix_args, owner=owner)
        if has_msg:
            assert isinstance(layer, Inkling2LoopDecoderLayer)
            norm = layer.mlp_norm
            if pass_state is not None:
                mlp_in = sm80_blend_pass_layer(
                    pass_state,
                    layer_rel_idx,
                    slice(3, 4),
                    layer.controller_down[3:4],
                    layer.controller_gain_forget[3:4],
                    mlp_in,
                    norm.weight,
                    norm.eps,
                )
            else:
                assert messages is not None
                mlp_in = layer.blend(
                    slice(3, 4), mlp_in, messages[3:], norm.weight, norm.eps
                )

        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        mix_args = (
            positions,
            stack[num_blocks],
            stack[:num_blocks],
            None,
            next_query,
            next_norm,
        )
        if moe_fused:
            assert workspace is not None
            outputs = layer.mlp_sconv.all_reduce_mix(
                workspace,
                *mix_args,
                combine=layer.mlp(mlp_in[0], reduce=False, pass_idx=pass_idx),
                addend=attn_out,
                residual=residual,
                two_shot=moe_two_shot,
                owner=owner,
            )
        else:
            if (
                is_sm80()
                and hasattr(layer.mlp, "experts")
                and layer.mlp.experts.ep_size == 1
            ):
                mlp_down, mlp_src2dst = layer.mlp(
                    mlp_in[0], reduce=False, pass_idx=pass_idx
                )
                outputs = layer.mlp_sconv.conv_mix(
                    mlp_down,
                    *mix_args,
                    addend=attn_out,
                    residual=residual,
                    owner=owner,
                    src2dst=mlp_src2dst,
                )
            else:
                # The MoE output is all-reduced (in bf16) inside layer.mlp.
                outputs = layer.mlp_sconv.conv_mix(
                    layer.mlp(mlp_in[0], pass_idx=pass_idx),
                    *mix_args,
                    addend=attn_out,
                    residual=residual,
                    owner=owner,
                )
        return num_blocks, outputs

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Prelude, `num_passes` core passes, then the coda. Each sublayer's
        short conv runs fused with the next AttnRes mix (and small batches
        with its all-reduce)."""
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = inputs_embeds.new_empty(
            (self._max_loop_blocks() + 1, num_tokens, hidden)
        )
        stack[0].copy_(inputs_embeds)
        tau = self._log_scaling_tau(positions)
        workspace = self.allreduce_workspace
        plan = [(False, False)] * 2
        if (
            workspace is not None
            and conv_metadata(self.layers[0].conv_state) is not None
        ):
            plan = workspace.plan(num_tokens)
        # (layer, pass) visits; the prelude and coda run as pass 0.
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        aux_hidden_states = []
        next_norm: nn.Module | None
        num_blocks = 0
        inputs = bank = pass_state = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) and partial as
                # the messages' bank.
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                pass_state = (
                    sm80_prepare_pass_bank(
                        bank,
                        self.message_queries,
                        q_split_bf16_t=(
                            self._get_q_split_bf16_t() if num_tokens > 256 else None
                        ),
                    )
                    if is_sm80()
                    else None
                )
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        query,
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].clone())
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
                plan,
                pass_state if looped else None,
            )
        assert inputs is not None
        out = inputs[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """The `inkling2_loop` checkpoints' architecture name; the model is
    `Inkling2ForConditionalGeneration` with an `Inkling2LoopModel`."""
