# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM80 CUDA megakernels wrapper (ctypes -> libsm80_fast_v6.so / v5.so)."""

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile

import torch

_LIB: ctypes.CDLL | None = None
_ALIGN_WS: dict[int, torch.Tensor] = {}
_SA32_WS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_SWL64_WS: dict[
    int,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_MFGC_WS: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_MFGEC_WS: dict[
    int,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_BLEND9_WS: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _resolve_so_path() -> str:
    env_so = os.environ.get("VLLM_SM80_FAST_SO", "").strip()
    if env_so:
        if os.path.isfile(env_so):
            return env_so
        raise FileNotFoundError(
            f"VLLM_SM80_FAST_SO={env_so!r} does not point to an existing file."
        )

    here = os.path.dirname(os.path.abspath(__file__))
    for so_name in ("libsm80_fast_v6.so", "libsm80_fast_v5.so", "libsm80_fast_v4.so"):
        prebuilt_so = os.path.join(here, so_name)
        if os.path.isfile(prebuilt_so):
            return prebuilt_so

    for cu_stem in ("sm80_fast_v6", "sm80_fast_v5", "sm80_fast_v4"):
        cu_path = os.path.join(here, f"{cu_stem}.cu")
        if os.path.isfile(cu_path):
            break
    else:
        raise FileNotFoundError(
            f"Missing CUDA source file (sm80_fast_v6.cu / sm80_fast_v5.cu) in {here}"
        )

    with open(cu_path, "rb") as fp:
        digest = hashlib.sha256(fp.read()).hexdigest()[:12]

    cache_root = os.path.expanduser(
        os.environ.get("VLLM_CACHE_ROOT", "").strip() or "~/.cache/vllm"
    )
    os.makedirs(cache_root, exist_ok=True)
    cached_so = os.path.join(cache_root, f"{cu_stem}_{digest}.so")
    if os.path.isfile(cached_so):
        return cached_so

    lock_path = f"{cached_so}.lock"
    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.isfile(cached_so):
                return cached_so

            nvcc = (
                os.environ.get("CUDACXX", "").strip()
                or shutil.which("nvcc")
                or "/usr/local/cuda/bin/nvcc"
            )
            if not (os.path.isfile(nvcc) and os.access(nvcc, os.X_OK)):
                raise RuntimeError(
                    f"Cannot compile {cu_path}: nvcc not found (checked $CUDACXX, "
                    "PATH, and /usr/local/cuda/bin/nvcc), and no prebuilt "
                    "libsm80_fast_v6.so or $VLLM_SM80_FAST_SO was provided."
                )

            fd, tmp_so = tempfile.mkstemp(
                prefix=f".{cu_stem}_{digest}.{os.getpid()}.",
                suffix=".so",
                dir=cache_root,
            )
            os.close(fd)
            try:
                subprocess.run(
                    [
                        nvcc,
                        "-O3",
                        "-arch=sm_80",
                        "--shared",
                        "-Xcompiler",
                        "-fPIC",
                        cu_path,
                        "-o",
                        tmp_so,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                os.replace(tmp_so, cached_so)
            except subprocess.CalledProcessError as exc:
                if os.path.exists(tmp_so):
                    os.unlink(tmp_so)
                raise RuntimeError(
                    f"Failed to compile {cu_path} with {nvcc}:\n{exc.stderr}"
                ) from exc
            except BaseException:
                if os.path.exists(tmp_so):
                    os.unlink(tmp_so)
                raise
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)

    return cached_so


def _get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(_resolve_so_path())
        vp = ctypes.c_void_p
        i32 = ctypes.c_int32
        i64 = ctypes.c_int64
        f32 = ctypes.c_float

        lib.launch_moe_align_v3_sm80.argtypes = [vp] * 5 + [i32] * 6 + [vp]
        lib.launch_moe_align_v3_sm80.restype = None

        lib.launch_select_and_align_32cta_128_sm80.argtypes = (
            [vp] * 13 + [i32] * 4 + [f32, vp]
        )
        lib.launch_select_and_align_32cta_128_sm80.restype = None

        lib.launch_select_and_align_128_sm80.argtypes = (
            [vp] * 10 + [i32] * 3 + [vp] * 3 + [i32] * 5 + [f32, i32, vp]
        )
        lib.launch_select_and_align_128_sm80.restype = None

        if hasattr(lib, "launch_select_partials_and_align_128_sm80"):
            lib.launch_select_partials_and_align_128_sm80.argtypes = (
                [vp] * 10 + [i32] * 5 + [f32, vp]
            )
            lib.launch_select_partials_and_align_128_sm80.restype = None

        if hasattr(lib, "launch_fused_up_blend_rms_2304_sm80"):
            lib.launch_fused_up_blend_rms_2304_sm80.argtypes = (
                [vp] * 6 + [i64, i64, i32, i32, f32, vp]
            )
            lib.launch_fused_up_blend_rms_2304_sm80.restype = None

        lib.launch_swiglu_1152_sm80.argtypes = [vp, vp, vp, vp, i32, vp]
        lib.launch_swiglu_1152_sm80.restype = None

        lib.launch_combine_2304_bf16_sm80.argtypes = [vp, vp, vp, i32, vp]
        lib.launch_combine_2304_bf16_sm80.restype = None

        lib.launch_attn_res_2304_sm80.argtypes = (
            [vp] * 6 + [i64, i64] + [i32] * 5 + [f32, f32, vp]
        )
        lib.launch_attn_res_2304_sm80.restype = i32

        lib.launch_conv_mix_2304_sm80.argtypes = (
            [vp] * 8
            + [i64, i64, i64]
            + [i32] * 5
            + [f32]
            + [vp] * 8
            + [i64, i64]
            + [i32] * 3
            + [f32, f32, vp]
        )
        lib.launch_conv_mix_2304_sm80.restype = i32

        lib.launch_pair_rms_norm_2304_sm80.argtypes = [
            vp,
            vp,
            vp,
            i64,
            i64,
            i64,
            f32,
            vp,
        ]
        lib.launch_pair_rms_norm_2304_sm80.restype = None

        lib.launch_blend_rms_norm_2304_sm80.argtypes = (
            [vp] * 5 + [i64] * 4 + [f32, vp]
        )
        lib.launch_blend_rms_norm_2304_sm80.restype = None

        lib.launch_markov_rnn_gate_256_sm80.argtypes = [vp] * 7 + [i32, i32, vp]
        lib.launch_markov_rnn_gate_256_sm80.restype = None

        lib.launch_markov_fold_softcap_r1_sm80.argtypes = [vp] * 5 + [i32, f32, vp]
        lib.launch_markov_fold_softcap_r1_sm80.restype = None

        lib.launch_markov_fold_gumbel_cache_r1_sm80.argtypes = (
            [vp] * 16 + [i64, i64, i32, i32, f32, vp]
        )
        lib.launch_markov_fold_gumbel_cache_r1_sm80.restype = None

        if hasattr(lib, "launch_markov_fold_gumbel_est_cache_r1_sm80"):
            lib.launch_markov_fold_gumbel_est_cache_r1_sm80.argtypes = (
                [vp] * 23 + [i64, i64, i64, i32, i32, f32, vp]
            )
            lib.launch_markov_fold_gumbel_est_cache_r1_sm80.restype = None

        if hasattr(lib, "set_l2_persisting_window_sm80"):
            u64 = ctypes.c_uint64
            lib.set_l2_persisting_window_sm80.argtypes = [vp, u64, u64, f32, vp]
            lib.set_l2_persisting_window_sm80.restype = i32

        # Round-5 v6 kernels
        if hasattr(lib, "launch_select_and_worklist_64cta_128_sm80"):
            lib.launch_select_and_worklist_64cta_128_sm80.argtypes = (
                [vp] * 16 + [i32] * 4 + [f32, vp]
            )
            lib.launch_select_and_worklist_64cta_128_sm80.restype = None

        if hasattr(lib, "launch_sm80_moe_w13_swiglu"):
            lib.launch_sm80_moe_w13_swiglu.argtypes = [vp] * 9 + [i32, i32, vp]
            lib.launch_sm80_moe_w13_swiglu.restype = None

        if hasattr(lib, "launch_sm80_moe_w2"):
            lib.launch_sm80_moe_w2.argtypes = [vp] * 8 + [i32, i32, vp]
            lib.launch_sm80_moe_w2.restype = None

        if hasattr(lib, "launch_markov_rnn_gate_256_suf_sm80"):
            lib.launch_markov_rnn_gate_256_suf_sm80.argtypes = (
                [vp] * 8 + [i32, i32, vp]
            )
            lib.launch_markov_rnn_gate_256_suf_sm80.restype = None

        if hasattr(lib, "launch_markov_fold_gumbel_cache_r1_suf_sm80"):
            lib.launch_markov_fold_gumbel_cache_r1_suf_sm80.argtypes = (
                [vp] * 18 + [i64, i64, i32, i32, f32, vp]
            )
            lib.launch_markov_fold_gumbel_cache_r1_suf_sm80.restype = None

        if hasattr(lib, "launch_suffix_cleanup_prev_sm80"):
            lib.launch_suffix_cleanup_prev_sm80.argtypes = (
                [vp] * 6 + [i64, i64, i32, i32, vp]
            )
            lib.launch_suffix_cleanup_prev_sm80.restype = None

        if hasattr(lib, "launch_suffix_apply_pointmass_sm80"):
            lib.launch_suffix_apply_pointmass_sm80.argtypes = (
                [vp] * 11 + [i64, i64] + [i32] * 5 + [vp]
            )
            lib.launch_suffix_apply_pointmass_sm80.restype = None

        if hasattr(lib, "launch_sm80_loop_down_64"):
            lib.launch_sm80_loop_down_64.argtypes = [vp, vp, vp, i64, i32, i32, vp]
            lib.launch_sm80_loop_down_64.restype = None

        if hasattr(lib, "launch_fused_up_blend_9tile_2304_sm80"):
            lib.launch_fused_up_blend_9tile_2304_sm80.argtypes = (
                [vp] * 9 + [i64, i64, i64, i32, i32, f32, vp]
            )
            lib.launch_fused_up_blend_9tile_2304_sm80.restype = None

        _LIB = lib
    return _LIB


def _stream(dev: torch.device) -> int:
    return torch.cuda.current_stream(dev).cuda_stream


def _ptr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def has_sm80_router_64cta() -> bool:
    return hasattr(_get_lib(), "launch_select_and_worklist_64cta_128_sm80")


def has_sm80_moe_v6() -> bool:
    lib = _get_lib()
    return hasattr(lib, "launch_sm80_moe_w13_swiglu") and hasattr(
        lib, "launch_select_and_worklist_64cta_128_sm80"
    )


def has_sm80_loop_blend_v6() -> bool:
    lib = _get_lib()
    return hasattr(lib, "launch_sm80_loop_down_64") and hasattr(
        lib, "launch_fused_up_blend_9tile_2304_sm80"
    )


def has_sm80_suffix_pointmass() -> bool:
    return hasattr(_get_lib(), "launch_suffix_apply_pointmass_sm80")


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
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size as _vllm_moe_align_block_size,
        )

        return _vllm_moe_align_block_size(
            topk_ids,
            block_size,
            num_experts,
            expert_map=expert_map,
            pad_sorted_ids=pad_sorted_ids,
            ignore_invalid_experts=ignore_invalid_experts,
            return_scatter_idx=return_scatter_idx,
        )
    flat = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    dev = flat.device
    numel = flat.numel()
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = (
            (max_num_tokens_padded + block_size - 1) // block_size
        ) * block_size
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=dev)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=dev)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=dev)
    if numel == 0:
        sorted_ids.fill_(0)
        expert_ids.fill_(0)
        num_tokens_post_pad.zero_()
        return sorted_ids, expert_ids, num_tokens_post_pad
    ws_ptr = 0
    if numel > 1024:
        d_idx = dev.index or 0
        ws = _ALIGN_WS.get(d_idx)
        if ws is None:
            ws = torch.empty((2 * 64 * 256,), dtype=torch.int32, device=dev)
            _ALIGN_WS[d_idx] = ws
        ws_ptr = ws.data_ptr()
    _get_lib().launch_moe_align_v3_sm80(
        flat.data_ptr(),
        sorted_ids.data_ptr(),
        expert_ids.data_ptr(),
        num_tokens_post_pad.data_ptr(),
        ws_ptr,
        numel,
        block_size,
        num_experts,
        max_num_tokens_padded,
        max_num_m_blocks,
        1 if ignore_invalid_experts else 0,
        _stream(dev),
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def select_and_align_128(
    partials: torch.Tensor,
    bias: torch.Tensor,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None,
    route_scale: float,
    num_local_experts: int,
    block_sizes: tuple[int, ...],
):
    dev = partials.device
    groups, tokens, experts = partials.shape
    assert experts == 128 and tokens <= 256 and groups in (1, 3)
    numel = tokens * 8
    ids = torch.empty((tokens, 8), dtype=torch.int32, device=dev)
    weights = torch.empty((numel,), dtype=torch.float32, device=dev)
    src2dst = torch.empty((numel,), dtype=torch.int32, device=dev)

    def _alloc_align(bm: int):
        mp = numel + num_local_experts * (bm - 1)
        nb = (mp + bm - 1) // bm
        s = torch.empty((mp,), dtype=torch.int32, device=dev)
        e = torch.empty((nb,), dtype=torch.int32, device=dev)
        p = torch.empty((1,), dtype=torch.int32, device=dev)
        return s, e, p, mp, nb

    bm1 = block_sizes[0]
    s1, e1, p1, mp1, nb1 = _alloc_align(bm1)
    if len(block_sizes) > 1 and block_sizes[1] != bm1:
        bm2 = block_sizes[1]
        s2, e2, p2, mp2, nb2 = _alloc_align(bm2)
    else:
        bm2 = 0
        s2 = e2 = p2 = None
        mp2 = nb2 = 0

    _get_lib().launch_select_and_align_128_sm80(
        partials.data_ptr(),
        bias.data_ptr(),
        expert_bounds.data_ptr(),
        _ptr(is_padding),
        ids.data_ptr(),
        weights.data_ptr(),
        src2dst.data_ptr(),
        s1.data_ptr(),
        e1.data_ptr(),
        p1.data_ptr(),
        bm1,
        mp1,
        nb1,
        _ptr(s2),
        _ptr(e2),
        _ptr(p2),
        bm2,
        mp2,
        nb2,
        tokens,
        groups,
        route_scale,
        num_local_experts,
        _stream(dev),
    )
    aligns = {bm1: (s1, e1, p1)}
    if bm2 > 0:
        aligns[bm2] = (s2, e2, p2)
    return ids, weights, src2dst, aligns


def select_partials_and_align_128(
    partials: torch.Tensor,
    bias: torch.Tensor,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None,
    route_scale: float,
    block_size: int = 8,
):
    dev = partials.device
    groups, t, _ = partials.shape
    n = t * 8
    max_pad = n + 128 * (block_size - 1)
    n_blocks = (max_pad + block_size - 1) // block_size
    ids = torch.empty((t, 8), dtype=torch.int32, device=dev)
    weights = torch.empty((n,), dtype=torch.float32, device=dev)
    src2dst = torch.empty((n,), dtype=torch.int32, device=dev)
    sorted_ids = torch.empty((max_pad,), dtype=torch.int32, device=dev)
    expert_ids = torch.empty((n_blocks,), dtype=torch.int32, device=dev)
    num_post_pad = torch.empty((1,), dtype=torch.int32, device=dev)
    _get_lib().launch_select_partials_and_align_128_sm80(
        partials.data_ptr(),
        bias.data_ptr(),
        expert_bounds.data_ptr(),
        _ptr(is_padding),
        ids.data_ptr(),
        weights.data_ptr(),
        src2dst.data_ptr(),
        sorted_ids.data_ptr(),
        expert_ids.data_ptr(),
        num_post_pad.data_ptr(),
        t,
        groups,
        block_size,
        max_pad,
        n_blocks,
        route_scale,
        _stream(dev),
    )
    return (ids, weights, src2dst), (sorted_ids, expert_ids, num_post_pad)


def select_and_worklist_64cta_128(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    expert_bounds: torch.Tensor,
    route_scale: float,
    block_size: int = 8,
    is_padding: torch.Tensor | None = None,
    with_align: bool = False,
    with_worklist: bool = True,
):
    t = x.shape[0]
    n = t * 8
    dev = x.device
    idx = dev.index or 0
    bufs = _SWL64_WS.get(idx)
    if bufs is None:
        bufs = (
            torch.empty((16 * 128,), dtype=torch.float32, device=dev),
            torch.zeros((1,), dtype=torch.int32, device=dev),
            torch.empty((1,), dtype=torch.int32, device=dev),
            torch.empty((64,), dtype=torch.int32, device=dev),
            torch.empty((64 * 8,), dtype=torch.int32, device=dev),
        )
        _SWL64_WS[idx] = bufs
    ws_lg, ctr, num_active, active_experts, active_slots = bufs

    ids = torch.empty((t, 8), dtype=torch.int32, device=dev)
    weights = torch.empty((n,), dtype=torch.float32, device=dev)
    src2dst = torch.empty((n,), dtype=torch.int32, device=dev)

    if with_align:
        max_pad = (
            min(n * block_size, n + 128 * (block_size - 1))
            if n < 128
            else n + 128 * (block_size - 1)
        )
        n_blocks = (max_pad + block_size - 1) // block_size
        sorted_ids = torch.empty((max_pad,), dtype=torch.int32, device=dev)
        expert_ids = torch.empty((n_blocks,), dtype=torch.int32, device=dev)
        num_post_pad = torch.empty((1,), dtype=torch.int32, device=dev)
        s_ptr, e_ptr, p_ptr = (
            sorted_ids.data_ptr(),
            expert_ids.data_ptr(),
            num_post_pad.data_ptr(),
        )
        pre_align8 = (sorted_ids, expert_ids, num_post_pad)
    else:
        max_pad = 0
        n_blocks = 0
        s_ptr = e_ptr = p_ptr = 0
        pre_align8 = None

    assert 1 <= t <= 16
    if with_worklist:
        assert t <= 8
        na_ptr = num_active.data_ptr()
        ae_ptr = active_experts.data_ptr()
        as_ptr = active_slots.data_ptr()
        worklist = (num_active, active_experts, active_slots)
    else:
        na_ptr = ae_ptr = as_ptr = 0
        worklist = None

    _get_lib().launch_select_and_worklist_64cta_128_sm80(
        x.data_ptr(),
        gate_weight.data_ptr(),
        bias.data_ptr(),
        expert_bounds.data_ptr(),
        _ptr(is_padding),
        ws_lg.data_ptr(),
        ctr.data_ptr(),
        ids.data_ptr(),
        weights.data_ptr(),
        src2dst.data_ptr(),
        s_ptr,
        e_ptr,
        p_ptr,
        na_ptr,
        ae_ptr,
        as_ptr,
        t,
        block_size,
        max_pad,
        n_blocks,
        route_scale,
        _stream(dev),
    )
    return (ids, weights, src2dst), pre_align8, worklist


def sm80_moe_forward_v6(
    x: torch.Tensor,
    w13_tensors: dict[str, torch.Tensor],
    w2_tensors: dict[str, torch.Tensor],
    is_nvfp4: bool,
    route_weights: torch.Tensor,
    worklist: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_ctas_w13: int = 222,
    num_ctas_w2: int = 222,
) -> torch.Tensor:
    dev = x.device
    rows = route_weights.numel()
    num_active, active_experts, active_slots = worklist
    act = torch.empty((rows, 1152), dtype=torch.bfloat16, device=dev)
    down = torch.empty((rows, 2304), dtype=torch.bfloat16, device=dev)
    lib = _get_lib()
    st = _stream(dev)
    nv = 1 if is_nvfp4 else 0
    lib.launch_sm80_moe_w13_swiglu(
        x.data_ptr(),
        w13_tensors["weight"].data_ptr(),
        w13_tensors["weight_scale"].data_ptr(),
        _ptr(w13_tensors.get("weight_scale_2")),
        num_active.data_ptr(),
        active_experts.data_ptr(),
        active_slots.data_ptr(),
        route_weights.data_ptr(),
        act.data_ptr(),
        nv,
        num_ctas_w13,
        st,
    )
    lib.launch_sm80_moe_w2(
        act.data_ptr(),
        w2_tensors["weight"].data_ptr(),
        w2_tensors["weight_scale"].data_ptr(),
        _ptr(w2_tensors.get("weight_scale_2")),
        num_active.data_ptr(),
        active_experts.data_ptr(),
        active_slots.data_ptr(),
        down.data_ptr(),
        nv,
        num_ctas_w2,
        st,
    )
    return down


def swiglu_1152(
    gate_up: torch.Tensor, weights: torch.Tensor, src2dst: torch.Tensor
) -> torch.Tensor:
    rows, two_n = gate_up.shape
    n = two_n // 2
    assert n == 1152 and gate_up.is_contiguous()
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=gate_up.device)
    _get_lib().launch_swiglu_1152_sm80(
        gate_up.data_ptr(),
        weights.data_ptr(),
        src2dst.data_ptr(),
        out.data_ptr(),
        rows,
        _stream(gate_up.device),
    )
    return out


def combine_2304_bf16(
    down: torch.Tensor, src2dst: torch.Tensor, topk: int = 8
) -> torch.Tensor:
    tokens = src2dst.numel() // topk
    assert topk == 8 and down.shape[1] == 2304 and down.is_contiguous()
    out = torch.empty((tokens, 2304), dtype=torch.bfloat16, device=down.device)
    _get_lib().launch_combine_2304_bf16_sm80(
        down.data_ptr(),
        src2dst.data_ptr(),
        out.data_ptr(),
        tokens,
        _stream(down.device),
    )
    return out


def attn_res_2304(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    num_tokens, d = last.shape
    num_stack = stack.shape[0]
    num_q = queries.shape[0]
    if (
        d != 2304
        or last.dtype != torch.bfloat16
        or not last.is_contiguous()
        or num_stack > 6
        or num_q > 4
    ):
        return None
    apply_norm = norm_weight is not None
    out_dtype = torch.bfloat16 if apply_norm else torch.float32
    out = torch.empty((num_q, num_tokens, d), dtype=out_dtype, device=last.device)
    ok = _get_lib().launch_attn_res_2304_sm80(
        stack.data_ptr(),
        last.data_ptr(),
        _ptr(addend),
        queries.data_ptr(),
        _ptr(norm_weight),
        out.data_ptr(),
        stack.stride(0),
        num_tokens,
        num_stack,
        num_q,
        1 if addend is not None else 0,
        1 if apply_norm else 0,
        1 if normalize_values else 0,
        eps,
        norm_eps,
        _stream(last.device),
    )
    return out if ok else None


def conv_mix_2304(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    conv_out: torch.Tensor,
    residual: torch.Tensor | None,
    addend: torch.Tensor | None,
    stack: torch.Tensor,
    last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    num_tokens, d = conv_out.shape
    num_stack = stack.shape[0]
    num_q = queries.shape[0]
    if d != 2304 or cache.dtype != torch.bfloat16 or num_stack > 6 or num_q > 4:
        return None
    apply_norm = norm_weight is not None
    out_dtype = torch.bfloat16 if apply_norm else torch.float32
    out = torch.empty((num_q, num_tokens, d), dtype=out_dtype, device=conv_out.device)
    ok = _get_lib().launch_conv_mix_2304_sm80(
        x.data_ptr(),
        conv_weight.data_ptr(),
        cache.data_ptr(),
        positions.data_ptr(),
        block_table.data_ptr(),
        seq_idx.data_ptr(),
        slot_mapping.data_ptr(),
        query_start.data_ptr(),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        block_table.stride(0),
        block_table.shape[1],
        cache.shape[2],
        stream_offset,
        stream_width,
        5.0**-0.5,
        conv_out.data_ptr(),
        _ptr(residual),
        _ptr(addend),
        stack.data_ptr(),
        _ptr(last),
        queries.data_ptr(),
        _ptr(norm_weight),
        out.data_ptr(),
        stack.stride(0),
        num_tokens,
        num_stack,
        num_q,
        1 if apply_norm else 0,
        eps,
        norm_eps,
        _stream(conv_out.device),
    )
    return out if ok else None


def pair_rms_norm_2304(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    s, t, d = message.shape
    out = message.new_empty((s, t, 2 * d), dtype=torch.bfloat16)
    _get_lib().launch_pair_rms_norm_2304_sm80(
        dest.data_ptr(),
        message.data_ptr(),
        out.data_ptr(),
        dest.stride(0),
        s,
        t,
        1e-6,
        _stream(message.device),
    )
    return out


def blend_rms_norm_2304(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    s, t, d = message.shape
    out = message.new_empty((s, t, d), dtype=torch.bfloat16)
    _get_lib().launch_blend_rms_norm_2304_sm80(
        dest.data_ptr(),
        message.data_ptr(),
        gain_forget.data_ptr(),
        norm_weight.data_ptr(),
        out.data_ptr(),
        dest.stride(0),
        norm_weight.stride(0),
        s,
        t,
        norm_eps,
        _stream(message.device),
    )
    return out


def sm80_loop_down_64(pair: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    s, t, _ = pair.shape
    out = pair.new_empty((s, t, 64), dtype=torch.bfloat16)
    _get_lib().launch_sm80_loop_down_64(
        pair.data_ptr(),
        down.data_ptr(),
        out.data_ptr(),
        down.stride(0),
        s,
        t,
        _stream(pair.device),
    )
    return out


def fused_up_blend_9tile_2304(
    dest: torch.Tensor,
    message: torch.Tensor,
    hidden: torch.Tensor,
    gf_t: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    s, t, d = message.shape
    assert s * t <= 24 and d == 2304
    dev = message.device
    idx = dev.index or 0
    ws = _BLEND9_WS.get(idx)
    if ws is None:
        ws = (
            torch.empty((24 * 2304,), dtype=torch.float32, device=dev),
            torch.empty((24 * 9,), dtype=torch.float32, device=dev),
            torch.zeros((24,), dtype=torch.int32, device=dev),
        )
        _BLEND9_WS[idx] = ws
    ws_x, ws_sq, ctrs = ws
    out = message.new_empty((s, t, d), dtype=torch.bfloat16)
    _get_lib().launch_fused_up_blend_9tile_2304_sm80(
        dest.data_ptr(),
        message.data_ptr(),
        hidden.data_ptr(),
        gf_t.data_ptr(),
        norm_weight.data_ptr(),
        ws_x.data_ptr(),
        ws_sq.data_ptr(),
        ctrs.data_ptr(),
        out.data_ptr(),
        dest.stride(0),
        gf_t.stride(0),
        norm_weight.stride(0),
        s,
        t,
        norm_eps,
        _stream(dev),
    )
    return out


def fused_up_blend_rms_2304(
    dest: torch.Tensor,
    message: torch.Tensor,
    hidden: torch.Tensor,
    gf_t: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    s, t, d = message.shape
    out = message.new_empty((s, t, d), dtype=torch.bfloat16)
    _get_lib().launch_fused_up_blend_rms_2304_sm80(
        dest.data_ptr(),
        message.data_ptr(),
        hidden.data_ptr(),
        gf_t.data_ptr(),
        norm_weight.data_ptr(),
        out.data_ptr(),
        dest.stride(0),
        norm_weight.stride(0),
        s,
        t,
        norm_eps,
        _stream(message.device),
    )
    return out


def suffix_cleanup_prev(
    logits_cache: torch.Tensor | None,
    idx_mapping: torch.Tensor,
    row_mapping: torch.Tensor | None,
    prev_suffix_tok: torch.Tensor,
    keep_dirty: torch.Tensor | None,
    just_scrubbed_all: torch.Tensor | None,
    num_reqs: int,
    max_spec: int,
) -> None:
    if logits_cache is None or num_reqs <= 0:
        return
    _get_lib().launch_suffix_cleanup_prev_sm80(
        logits_cache.data_ptr(),
        idx_mapping.data_ptr(),
        _ptr(row_mapping),
        prev_suffix_tok.data_ptr(),
        _ptr(keep_dirty),
        _ptr(just_scrubbed_all),
        logits_cache.stride(0),
        logits_cache.stride(1),
        num_reqs,
        max_spec,
        _stream(logits_cache.device),
    )


def suffix_apply_pointmass(
    logits_cache: torch.Tensor | None,
    keep_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    row_mapping: torch.Tensor | None,
    suffix_drafts: torch.Tensor,
    suffix_num_valid: torch.Tensor,
    suffix_match_len: torch.Tensor,
    prev_suffix_tok: torch.Tensor,
    keep_dirty: torch.Tensor | None,
    just_scrubbed_all: torch.Tensor | None,
    out_draft_tokens: torch.Tensor,
    num_reqs: int,
    n_spec: int,
    max_spec: int,
    min_match_len: int,
) -> None:
    if num_reqs <= 0 or n_spec <= 0:
        return
    dev = out_draft_tokens.device
    s0 = 0 if logits_cache is None else logits_cache.stride(0)
    s1 = 0 if logits_cache is None else logits_cache.stride(1)
    _get_lib().launch_suffix_apply_pointmass_sm80(
        _ptr(logits_cache),
        keep_ids.data_ptr(),
        idx_mapping.data_ptr(),
        _ptr(row_mapping),
        suffix_drafts.data_ptr(),
        suffix_num_valid.data_ptr(),
        suffix_match_len.data_ptr(),
        prev_suffix_tok.data_ptr(),
        _ptr(keep_dirty),
        _ptr(just_scrubbed_all),
        out_draft_tokens.data_ptr(),
        s0,
        s1,
        num_reqs,
        int(keep_ids.shape[0]),
        n_spec,
        max_spec,
        min_match_len,
        _stream(dev),
    )


def markov_rnn_gate_256(
    state: torch.Tensor,
    prev_tokens: torch.Tensor,
    markov_w1_weight: torch.Tensor,
    context: torch.Tensor,
    rnn_w_step: torch.Tensor,
    skip_flag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = state.shape[0]
    state_out = torch.empty_like(state)
    latent_out = torch.empty_like(state)
    lib = _get_lib()
    if skip_flag is not None and hasattr(lib, "launch_markov_rnn_gate_256_suf_sm80"):
        lib.launch_markov_rnn_gate_256_suf_sm80(
            state.data_ptr(),
            prev_tokens.data_ptr(),
            markov_w1_weight.data_ptr(),
            context.data_ptr(),
            rnn_w_step.data_ptr(),
            skip_flag.data_ptr(),
            state_out.data_ptr(),
            latent_out.data_ptr(),
            num_reqs,
            1 if prev_tokens.dtype == torch.int64 else 0,
            _stream(state.device),
        )
        return state_out, latent_out
    lib.launch_markov_rnn_gate_256_sm80(
        state.data_ptr(),
        prev_tokens.data_ptr(),
        markov_w1_weight.data_ptr(),
        context.data_ptr(),
        rnn_w_step.data_ptr(),
        state_out.data_ptr(),
        latent_out.data_ptr(),
        num_reqs,
        1 if prev_tokens.dtype == torch.int64 else 0,
        _stream(state.device),
    )
    return state_out, latent_out


def markov_fold_softcap_r1(
    latent: torch.Tensor,
    fold_weight: torch.Tensor,
    fold_bias: torch.Tensor,
    base_logits_1d: torch.Tensor,
    softcap: float,
) -> torch.Tensor:
    v = fold_weight.shape[0]
    out = torch.empty((1, v), dtype=torch.bfloat16, device=latent.device)
    _get_lib().launch_markov_fold_softcap_r1_sm80(
        latent.data_ptr(),
        fold_weight.data_ptr(),
        fold_bias.data_ptr(),
        base_logits_1d.data_ptr(),
        out.data_ptr(),
        v,
        softcap,
        _stream(latent.device),
    )
    return out


def select_and_align_32cta_128(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    expert_bounds: torch.Tensor,
    route_scale: float,
    block_size: int = 8,
):
    t = x.shape[0]
    n = t * 8
    max_pad = (
        min(n * block_size, n + 128 * (block_size - 1))
        if n < 128
        else n + 128 * (block_size - 1)
    )
    n_blocks = (max_pad + block_size - 1) // block_size
    dev = x.device
    idx = dev.index or 0
    bufs = _SA32_WS.get(idx)
    if bufs is None:
        bufs = (
            torch.empty((16 * 128,), dtype=torch.float32, device=dev),
            torch.zeros((1,), dtype=torch.int32, device=dev),
        )
        _SA32_WS[idx] = bufs
    ws_lg, ctr = bufs
    ids = torch.empty((t, 8), dtype=torch.int32, device=dev)
    weights = torch.empty((n,), dtype=torch.float32, device=dev)
    src2dst = torch.empty((n,), dtype=torch.int32, device=dev)
    sorted_ids = torch.empty((max_pad,), dtype=torch.int32, device=dev)
    expert_ids = torch.empty((n_blocks,), dtype=torch.int32, device=dev)
    num_post_pad = torch.empty((1,), dtype=torch.int32, device=dev)
    _get_lib().launch_select_and_align_32cta_128_sm80(
        x.data_ptr(),
        gate_weight.data_ptr(),
        bias.data_ptr(),
        expert_bounds.data_ptr(),
        0,
        ws_lg.data_ptr(),
        ctr.data_ptr(),
        ids.data_ptr(),
        weights.data_ptr(),
        src2dst.data_ptr(),
        sorted_ids.data_ptr(),
        expert_ids.data_ptr(),
        num_post_pad.data_ptr(),
        t,
        block_size,
        max_pad,
        n_blocks,
        route_scale,
        _stream(dev),
    )
    return (ids, weights, src2dst), (sorted_ids, expert_ids, num_post_pad)


def markov_fold_gumbel_cache_r1(
    latent: torch.Tensor,
    fold_weight: torch.Tensor,
    fold_bias: torch.Tensor,
    base_logits_1d: torch.Tensor,
    keep_ids: torch.Tensor,
    idx_map_step: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    sample_pos_step: torch.Tensor,
    row_mapping: torch.Tensor | None,
    logits_cache: torch.Tensor,
    step_col: int,
    softcap: float,
    out_draft_col: torch.Tensor,
    skip_flag: torch.Tensor | None = None,
    keep_dirty: torch.Tensor | None = None,
) -> torch.Tensor:
    dev = latent.device
    d_idx = dev.index or 0
    ws = _MFGC_WS.get(d_idx)
    if ws is None:
        ws = (
            torch.empty((256,), dtype=torch.float32, device=dev),
            torch.empty((256,), dtype=torch.int64, device=dev),
            torch.zeros((1,), dtype=torch.int32, device=dev),
        )
        _MFGC_WS[d_idx] = ws
    ws_max, ws_id, ctr = ws
    out_sampled = torch.empty((1,), dtype=torch.int64, device=dev)
    lib = _get_lib()
    if (skip_flag is not None or keep_dirty is not None) and hasattr(
        lib, "launch_markov_fold_gumbel_cache_r1_suf_sm80"
    ):
        lib.launch_markov_fold_gumbel_cache_r1_suf_sm80(
            latent.data_ptr(),
            fold_weight.data_ptr(),
            fold_bias.data_ptr(),
            base_logits_1d.data_ptr(),
            keep_ids.data_ptr(),
            idx_map_step.data_ptr(),
            temperature.data_ptr(),
            seeds.data_ptr(),
            sample_pos_step.data_ptr(),
            _ptr(row_mapping),
            _ptr(skip_flag),
            _ptr(keep_dirty),
            logits_cache.data_ptr(),
            ws_max.data_ptr(),
            ws_id.data_ptr(),
            ctr.data_ptr(),
            out_sampled.data_ptr(),
            out_draft_col.data_ptr(),
            logits_cache.stride(0),
            logits_cache.stride(1),
            int(step_col),
            int(fold_weight.shape[0]),
            float(softcap),
            _stream(dev),
        )
        return out_sampled
    lib.launch_markov_fold_gumbel_cache_r1_sm80(
        latent.data_ptr(),
        fold_weight.data_ptr(),
        fold_bias.data_ptr(),
        base_logits_1d.data_ptr(),
        keep_ids.data_ptr(),
        idx_map_step.data_ptr(),
        temperature.data_ptr(),
        seeds.data_ptr(),
        sample_pos_step.data_ptr(),
        _ptr(row_mapping),
        logits_cache.data_ptr(),
        ws_max.data_ptr(),
        ws_id.data_ptr(),
        ctr.data_ptr(),
        out_sampled.data_ptr(),
        out_draft_col.data_ptr(),
        logits_cache.stride(0),
        logits_cache.stride(1),
        int(step_col),
        int(fold_weight.shape[0]),
        float(softcap),
        _stream(dev),
    )
    return out_sampled


def has_fused_markov_est() -> bool:
    return hasattr(_get_lib(), "launch_markov_fold_gumbel_est_cache_r1_sm80")


def markov_fold_gumbel_est_cache_r1(
    latent: torch.Tensor,
    fold_weight: torch.Tensor,
    fold_bias: torch.Tensor,
    base_logits_1d: torch.Tensor,
    keep_ids: torch.Tensor,
    idx_map_step: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    sample_pos_step: torch.Tensor,
    row_mapping: torch.Tensor | None,
    logits_cache: torch.Tensor,
    step_col: int,
    softcap: float,
    out_draft_col: torch.Tensor,
    est_features: torch.Tensor,
    est_preds: torch.Tensor,
    est_conf: torch.Tensor,
    est_slope: torch.Tensor,
    est_intercepts: torch.Tensor,
) -> torch.Tensor:
    dev = latent.device
    d_idx = dev.index or 0
    ws = _MFGEC_WS.get(d_idx)
    if ws is None:
        ws = (
            torch.empty((256,), dtype=torch.float32, device=dev),
            torch.empty((256,), dtype=torch.int64, device=dev),
            torch.empty((256,), dtype=torch.float32, device=dev),
            torch.empty((256,), dtype=torch.float32, device=dev),
            torch.zeros((1,), dtype=torch.int32, device=dev),
        )
        _MFGEC_WS[d_idx] = ws
    ws_max, ws_id, ws_em, ws_ee, ctr = ws
    out_sampled = torch.empty((1,), dtype=torch.int64, device=dev)
    _get_lib().launch_markov_fold_gumbel_est_cache_r1_sm80(
        latent.data_ptr(),
        fold_weight.data_ptr(),
        fold_bias.data_ptr(),
        base_logits_1d.data_ptr(),
        keep_ids.data_ptr(),
        idx_map_step.data_ptr(),
        temperature.data_ptr(),
        seeds.data_ptr(),
        sample_pos_step.data_ptr(),
        _ptr(row_mapping),
        logits_cache.data_ptr(),
        ws_max.data_ptr(),
        ws_id.data_ptr(),
        ws_em.data_ptr(),
        ws_ee.data_ptr(),
        ctr.data_ptr(),
        out_sampled.data_ptr(),
        out_draft_col.data_ptr(),
        est_features.data_ptr(),
        est_preds.data_ptr(),
        est_conf.data_ptr(),
        est_slope.data_ptr(),
        est_intercepts.data_ptr(),
        logits_cache.stride(0),
        logits_cache.stride(1),
        est_features.stride(0),
        int(step_col),
        int(fold_weight.shape[0]),
        float(softcap),
        _stream(dev),
    )
    return out_sampled


def set_l2_persisting_window(
    tensor: torch.Tensor | None,
    limit_bytes: int = 26 * 1024 * 1024,
    hit_ratio: float = 1.0,
    device: torch.device | None = None,
) -> int:
    lib = _get_lib()
    if not hasattr(lib, "set_l2_persisting_window_sm80"):
        return -1
    dev = (
        tensor.device
        if tensor is not None
        else (device or torch.device("cuda", 0))
    )
    ptr = 0 if tensor is None else tensor.data_ptr()
    nbytes = 0 if tensor is None else tensor.numel() * tensor.element_size()
    return int(
        lib.set_l2_persisting_window_sm80(
            ptr, nbytes, limit_bytes, float(hit_ratio), _stream(dev)
        )
    )
