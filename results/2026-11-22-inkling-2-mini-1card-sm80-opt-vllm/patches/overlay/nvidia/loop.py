# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2 checkpoints (`loop_config`)."""

import os
from collections.abc import Sequence

import torch
from torch import nn

from vllm.distributed import tensor_model_parallel_all_reduce

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
from .sconv import conv_metadata

LOOP_STREAMS = ("attn", "k", "v", "mlp")
_SM80_BLEND_MODE = os.environ.get("VLLM_SM80_LOOP_BLEND_V6", "0")


class Inkling2MessageController(nn.Module):
    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)

    def forward(self, dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        pair = torch.stack((dest, message), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        normalized = pair.to(torch.bfloat16).flatten(1)
        hidden = linear_k_groups(normalized, self.down_weight, torch.bfloat16)
        gain = torch.mm(hidden, self.gain_weight.t(), out_dtype=torch.float32)
        forget = torch.mm(hidden, self.forget_weight.t(), out_dtype=torch.float32)
        return (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
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
        self.controller_gf_t: torch.Tensor | None = None
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
        for i, controller in enumerate(self.controllers.values()):
            down, gain_forget = self.controller_down, self.controller_gain_forget
            assert controller.down_weight.data_ptr() == down[i].data_ptr()
            assert controller.gain_weight.data_ptr() == gain_forget[i].data_ptr()
        if (
            is_sm80()
            and _SM80_BLEND_MODE != "0"
            and self.controller_gain_forget.shape == (4, 4608, 64)
        ):
            self.controller_gf_t = (
                self.controller_gain_forget.view(4, 2, 2304, 64)
                .transpose(2, 3)
                .contiguous()
            )

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        pair = pair_rms_norm(dest, message)
        down = self.controller_down[streams]
        up = self.controller_gain_forget[streams]
        if (
            is_sm80()
            and self.controller_gf_t is not None
            and _SM80_BLEND_MODE != "0"
            and message.shape[2] == 2304
            and message.shape[1] > 0
        ):
            from ..sm80_ops_v2 import (
                fused_up_blend_9tile_2304,
                fused_up_blend_rms_2304,
                has_sm80_loop_blend_v6,
                sm80_loop_down_64,
            )

            s, t, _ = message.shape
            gf_t = self.controller_gf_t[streams]
            if (
                has_sm80_loop_blend_v6()
                and ((t == 1 and _SM80_BLEND_MODE == "auto") or (t <= 8 and _SM80_BLEND_MODE == "down_up9"))
                and s * t <= 24
            ):
                hidden = sm80_loop_down_64(pair, down)
                return fused_up_blend_9tile_2304(
                    dest, message, hidden, gf_t, norm_weight, norm_eps
                )
            if has_sm80_loop_blend_v6() and t <= 8 and _SM80_BLEND_MODE == "down_up1":
                hidden = sm80_loop_down_64(pair, down)
                return fused_up_blend_rms_2304(
                    dest, message, hidden, gf_t, norm_weight, norm_eps
                )
            if _SM80_BLEND_MODE == "ll_up1" and use_sm80_bf16_ll_gemm(pair, down) and s * t <= 256:
                hidden = sm80_bf16_ll_gemm(pair, down)
                return fused_up_blend_rms_2304(
                    dest, message, hidden, gf_t, norm_weight, norm_eps
                )
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
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
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

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)

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
    ) -> tuple[int, torch.Tensor]:
        layer = self.layers[i]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        workspace = self.allreduce_workspace
        (attn_fused, attn_two_shot), (moe_fused, moe_two_shot) = plan
        messages = None
        attn_inputs: Sequence[torch.Tensor]
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            messages = attn_res(
                bank[:1],
                bank[1],
                None,
                self.message_queries[i - self.loop_core.start],
                None,
                0.0,
                True,
            )
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
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

        first = i % self.block_size == 0
        attn_out = torch.empty_like(partial)
        mix_args = (
            positions,
            attn_out,
            stack[: num_blocks + first],
            None if first else partial,
            self.attn_res.mlp_queries[i],
            None if messages is not None else layer.mlp_norm,
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
        if messages is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer)
            norm = layer.mlp_norm
            mlp_in = layer.blend(
                slice(3, 4), mlp_in, messages[3:], norm.weight, norm.eps
            )

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
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        aux_hidden_states = []
        next_norm: nn.Module | None
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
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
            )
        assert inputs is not None
        out = inputs[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """The `inkling2_loop` checkpoints' architecture name."""
