# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 mixture-of-experts layer (routed experts only, NVFP4 or MXFP8)."""

import os
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from .ops import is_sm80, use_sm100_kernels
from .ops.activation import swiglu_quantize
from .ops.block_scales import shuffle_rows, swizzle_block_scales
from .ops.linear import k_group_partials
from .ops.moe import NUM_COUNTERS, combine, combine_push, route
from .ops.moe_humming import HummingNVFP4Experts
from .ops.moe_sm90 import HummingExperts, select_experts
from .ops.moe_sm120 import bmm as bmm_sm120
from .ops.seq_parallel import SeqParallel
from .ops.trtllm_bmm import bmm, expert_plan, load_kernels

QuantFormat = Literal["nvfp4", "mxfp8"]

_USE_SM80_MOE_V6 = os.environ.get("VLLM_SM80_MOE_V6", "0") == "1"
_SM80_MOE_V6_MAX_T = int(os.environ.get("VLLM_SM80_MOE_V6_MAX_T", "8"))
_SM80_MOE_V6_CTAS = int(os.environ.get("VLLM_SM80_MOE_V6_CTAS", "296"))
_USE_SM80_ROUTER_64CTA = os.environ.get("VLLM_SM80_ROUTER_64CTA", "0") == "1"
_USE_SM80_PARTIALS_ALIGN = os.environ.get("VLLM_SM80_PARTIALS_ALIGN", "0") == "1"


class ExpertsLoRA(Protocol):
    def add_w13(
        self,
        gate_up: torch.Tensor,
        x: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None: ...

    def add_w2(
        self,
        down: torch.Tensor,
        activated: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None: ...


class Inkling2Gate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.use_gate_bias
        num_experts, hidden = config.n_routed_experts, config.hidden_size
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        for param in (self.weight, self.bias):
            set_weight_attrs(param, {"weight_loader": default_weight_loader})
        self.register_buffer(
            "counters", torch.zeros(NUM_COUNTERS, dtype=torch.int32), persistent=False
        )


class Inkling2Experts(nn.Module):
    supports_lora = False

    def __init__(self, config, quant_format: QuantFormat) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.nvfp4 = quant_format == "nvfp4"
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        self.shapes = ((2 * inter, hidden), (hidden, inter))
        for name, (n, k) in zip(("w13", "w2"), self.shapes):
            self._register(
                f"{name}_weight",
                (n, k // pack),
                weight_dtype,
                loader=self._load_weight,
            )
            self._register(
                f"{name}_weight_scale",
                (n, k // group),
                scale_dtype,
                loader=self._load_block_scales,
            )
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

        start = self.ep_rank * self.num_experts
        self.register_buffer(
            "expert_bounds",
            torch.tensor([start, start + self.num_experts], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "counts", torch.zeros(self.num_experts, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "ones", torch.ones(self.num_experts, dtype=torch.float32), persistent=False
        )
        self.sm100 = use_sm100_kernels()
        self.sm120 = current_platform.is_device_capability_family(120)
        vllm_config = get_current_vllm_config_or_none()
        self.humming_nvfp4 = (
            self.sm120
            and self.nvfp4
            and vllm_config is not None
            and vllm_config.kernel_config.moe_backend == "humming"
        )
        self.humming: HummingExperts | None = None
        if self.sm100:
            load_kernels(
                (self.w13_weight, self.w2_weight),
                (self.w13_weight_scale, self.w2_weight_scale),
                self.topk,
            )

    def _register(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        per_expert: bool = True,
        loader=None,
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        if loader is None:
            loader = self._load_expert_shard if per_expert else self._load_full
        set_weight_attrs(param, {"weight_loader": loader})
        self.register_parameter(name, param)

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    def _load_weight(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)
        if self.sm100:
            param.data.copy_(shuffle_rows(param.data))

    def _load_block_scales(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load_expert_shard(param, loaded)
        if self.sm100:
            param.data.copy_(swizzle_block_scales(shuffle_rows(param.data)))

    @staticmethod
    def _load_full(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def process_weights_after_loading(self) -> None:
        if self.sm100 or (self.sm120 and not self.humming_nvfp4):
            return
        weights = {}
        for name in ("w13", "w2"):
            amax, denominator = self.scales(name)
            global_scale = None
            if amax is not None and denominator is not None:
                global_scale = (
                    self.ones if self.humming_nvfp4 else amax.data / denominator.data
                )
            weights[name] = (
                getattr(self, f"{name}_weight").data,
                getattr(self, f"{name}_weight_scale").data,
                global_scale,
            )
        self.humming = (
            HummingNVFP4Experts(weights, self.shapes)
            if self.humming_nvfp4
            else HummingExperts(weights, self.shapes, self.nvfp4)
        )
        for name in ("w13", "w2"):
            for suffix in ("weight", "weight_scale"):
                param = getattr(self, f"{name}_{suffix}")
                param.data = param.data.new_empty((0,))

    def plan(self, tokens: int):
        return expert_plan(tokens, self.topk, self.num_experts, self.nvfp4, self.shapes)

    def scales(self, name: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.nvfp4:
            return None, None
        return (
            getattr(self, f"{name}_weight_amax"),
            getattr(self, f"{name}_weight_outer_scale_denominator"),
        )

    def _gemm(self, quantized, name: str, tiles, plan, projection: int):
        if self.humming_nvfp4:
            assert isinstance(self.humming, HummingNVFP4Experts)
            return self.humming.bmm(quantized, name, tiles, plan)
        act, act_scale, row_scale = quantized
        gemm = bmm_sm120 if self.sm120 else bmm
        return gemm(
            act,
            act_scale,
            row_scale,
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_weight_scale"),
            self.ones,
            tiles,
            plan,
            projection,
        )

    def forward(
        self,
        x: torch.Tensor,
        routing,
        quantized,
        plan,
        reduce: bool = True,
        push: SeqParallel | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        if self.sm100 or self.sm120:
            src2dst, gammas, rows, tiles = routing
            gate_up = self._gemm(quantized, "w13", tiles, plan, 0)
            m = plan.max_ctas
            activated = swiglu_quantize(
                gate_up,
                gammas,
                self.gate_cap,
                self.up_cap,
                (rows, tiles[2 * m + 2 :], tiles[:m]),
                plan.tile,
                *self.scales("w2"),
            )
            del gate_up
            down = self._gemm(activated, "w2", tiles, plan, 1)
        else:
            assert self.humming is not None, "process_weights_after_loading"
            if len(routing) == 5:
                _, gammas, src2dst, _, worklist = routing
                from ..sm80_ops_v2 import sm80_moe_forward_v6

                down = sm80_moe_forward_v6(
                    x,
                    self.humming.tensors["w13"],
                    self.humming.tensors["w2"],
                    self.nvfp4,
                    gammas,
                    worklist,
                    _SM80_MOE_V6_CTAS,
                    _SM80_MOE_V6_CTAS,
                )
            else:
                if len(routing) == 4:
                    ids, gammas, src2dst, pre_align8 = routing
                else:
                    ids, gammas, src2dst = routing
                    pre_align8 = None
                down = self.humming(
                    x,
                    ids,
                    gammas,
                    src2dst,
                    self.num_experts * self.ep_size,
                    self.gate_cap,
                    self.up_cap,
                    pre_align8=pre_align8,
                )
        if push is not None:
            combine_push(down, src2dst, self.topk, push)
            return None
        if not reduce:
            return down, src2dst
        out = combine(down, src2dst, self.topk, x.dtype)
        return out if self.ep_size == 1 else tensor_model_parallel_all_reduce(out)


class Inkling2RoutingCapture(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.capture_fn: Callable[[torch.Tensor], None] | None = None


class Inkling2MoE(nn.Module):
    def __init__(
        self, config, quant_format: QuantFormat, visit_ids: Sequence[int] = (0,)
    ) -> None:
        super().__init__()
        assert config.n_shared_experts == 0
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config, quant_format)
        self.routing_captures = nn.ModuleList(
            Inkling2RoutingCapture(visit) for visit in visit_ids
        )
        self.register_buffer(
            "all_experts",
            torch.tensor([0, config.n_routed_experts], dtype=torch.int32),
            persistent=False,
        )

    def routed_expert_ids(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate
        ids, _, _ = select_experts(
            x, gate.weight, gate.bias, gate.topk, gate.route_scale, self.all_experts
        )
        return ids

    def forward(
        self,
        x: torch.Tensor,
        reduce: bool = True,
        push: SeqParallel | None = None,
        lora: ExpertsLoRA | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        capture_fn = self.routing_captures[pass_idx].capture_fn
        if capture_fn is not None:
            capture_fn(self.routed_expert_ids(x.contiguous()))
        if lora is not None:
            raise NotImplementedError(
                "Inkling2 expert LoRA needs BF16 experts; the NVFP4 / MXFP8 "
                "experts do not support it"
            )
        x = x.contiguous()
        gate, experts = self.gate, self.experts
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
        if not (experts.sm100 or experts.sm120):
            t_rows = x.shape[0]
            if (
                is_sm80()
                and experts.ep_size == 1
                and experts.num_experts == 128
                and gate.topk == 8
                and x.shape[1] == 2304
            ):
                if (
                    _USE_SM80_MOE_V6
                    and 1 <= t_rows <= _SM80_MOE_V6_MAX_T
                    and experts.gate_cap == 4.0
                    and experts.up_cap == 25.0
                ):
                    from ..sm80_ops_v2 import (
                        has_sm80_moe_v6,
                        select_and_worklist_64cta_128,
                    )

                    if has_sm80_moe_v6():
                        (ids, gammas, src2dst), _, worklist = (
                            select_and_worklist_64cta_128(
                                x,
                                gate.weight,
                                gate.bias,
                                experts.expert_bounds,
                                gate.route_scale,
                                8,
                                is_padding=is_padding,
                                with_align=False,
                                with_worklist=True,
                            )
                        )
                        return experts(
                            x,
                            (ids, gammas, src2dst, None, worklist),
                            None,
                            None,
                            reduce,
                            push,
                        )
                if _USE_SM80_ROUTER_64CTA and 1 <= t_rows <= 16:
                    from ..sm80_ops_v2 import (
                        has_sm80_router_64cta,
                        select_and_worklist_64cta_128,
                    )

                    if has_sm80_router_64cta():
                        (ids, gammas, src2dst), pre_align8, _ = (
                            select_and_worklist_64cta_128(
                                x,
                                gate.weight,
                                gate.bias,
                                experts.expert_bounds,
                                gate.route_scale,
                                8,
                                is_padding=is_padding,
                                with_align=True,
                                with_worklist=False,
                            )
                        )
                        return experts(
                            x,
                            (ids, gammas, src2dst, pre_align8),
                            None,
                            None,
                            reduce,
                            push,
                        )
                if 1 <= t_rows <= 3 and is_padding is None:
                    from ..sm80_ops_v2 import select_and_align_32cta_128

                    (ids, gammas, src2dst), pre_align8 = select_and_align_32cta_128(
                        x,
                        gate.weight,
                        gate.bias,
                        experts.expert_bounds,
                        gate.route_scale,
                        8,
                    )
                    return experts(
                        x, (ids, gammas, src2dst, pre_align8), None, None, reduce, push
                    )
                if _USE_SM80_PARTIALS_ALIGN and 4 <= t_rows <= 8:
                    from ..sm80_ops_v2 import select_partials_and_align_128

                    partials = k_group_partials(x, gate.weight)
                    (ids, gammas, src2dst), pre_align8 = select_partials_and_align_128(
                        partials,
                        gate.bias,
                        experts.expert_bounds,
                        is_padding,
                        gate.route_scale,
                        8,
                    )
                    return experts(
                        x, (ids, gammas, src2dst, pre_align8), None, None, reduce, push
                    )
            selected = select_experts(
                x,
                gate.weight,
                gate.bias,
                gate.topk,
                gate.route_scale,
                experts.expert_bounds,
                is_padding,
            )
            return experts(x, selected, None, None, reduce, push)
        plan = experts.plan(x.shape[0])
        routing, quantized = route(
            x,
            gate.weight,
            gate.bias,
            gate.counters,
            experts.counts,
            gate.topk,
            gate.route_scale,
            experts.expert_bounds,
            plan,
            *experts.scales("w13"),
            is_padding,
        )
        return experts(x, routing, quantized, plan, reduce, push)
