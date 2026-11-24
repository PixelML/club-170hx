# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-M BF16 tensor-core projections for SM80 (ported & tuned from vLLM PR #69168):
batched `x @ w.T` with deterministic split-K for narrow outputs, plus 1-launch
grouped `qkv_bmm`."""

import functools
from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import current_stream

from . import is_sm80

MAX_M = 16
MAX_BATCHED_M = 64
_MP = 16
_MAX_TILES = 8192


@triton.jit
def _ll_gemm_kernel(
    X,
    W,
    Bias,
    Y,
    P,
    C,
    M,
    N,
    K,
    SXB,
    SXM,
    SWB,
    SWN,
    SYB,
    SYM,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    SPLIT: tl.constexpr,
    MP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    s = tl.program_id(2)
    rows = tl.arange(0, MP)
    cols = pid_n * BN + tl.arange(0, BN)
    ks = s * K_PER_SPLIT + tl.arange(0, BK)
    x_ptrs = X + (b * SXB + rows[:, None] * SXM + ks[None, :])
    w_ptrs = W + (b * SWB + cols[None, :].to(tl.int64) * SWN + ks[:, None])
    acc = tl.zeros((MP, BN), tl.float32)
    for _ in range(0, tl.minimum(K - s * K_PER_SPLIT, K_PER_SPLIT), BK):
        w = tl.load(w_ptrs, (cols[None, :] < N) & (ks[:, None] < K), other=0)
        x = tl.load(x_ptrs, (rows[:, None] < M) & (ks[None, :] < K), other=0)
        acc += tl.dot(x, w)
        x_ptrs += BK
        w_ptrs += BK
        ks += BK

    is_last = True
    if SPLIT > 1:
        num_b = tl.num_programs(1)
        tile = b * tl.num_programs(0) + pid_n
        offs = rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols < N)[None, :]
        tl.store(P + ((s * num_b + b) * MP * N + offs), acc, mask)
        tl.debug_barrier()
        is_last = tl.atomic_add(C + tile, 1, sem="acq_rel", scope="gpu") == SPLIT - 1
        if is_last:
            acc = tl.zeros((MP, BN), tl.float32)
            for i in tl.static_range(SPLIT):
                acc += tl.load(
                    P + ((i * num_b + b) * MP * N + offs),
                    mask,
                    cache_modifier=".cg",
                )
            tl.atomic_xchg(C + tile, 0, sem="relaxed", scope="gpu")
    if is_last:
        if Bias is not None:
            acc += tl.load(Bias + cols, cols < N, other=0).to(tl.float32)[None, :]
        tl.store(
            Y + (b * SYB + rows[:, None] * SYM + cols[None, :]),
            acc.to(Y.dtype.element_ty),
            (rows[:, None] < M) & (cols[None, :] < N),
        )


@triton.jit
def _qkv_bmm_kernel(
    XQ,
    XK,
    XV,
    W,
    Y,
    M,
    N,
    K,
    S1,
    S2,
    SXQ,
    SXK,
    SXV,
    SWN,
    SYM,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    rows = tl.arange(0, MP)
    cols = pid_n * BN + tl.arange(0, BN)
    c0 = pid_n * BN
    c1 = c0 + BN - 1
    b_lo = (c0 >= S1).to(tl.int32) + (c0 >= S2).to(tl.int32)
    b_hi = (c1 >= S1).to(tl.int32) + (c1 >= S2).to(tl.int32)
    ks = tl.arange(0, BK)
    w_ptrs = W + (cols[None, :].to(tl.int64) * SWN + ks[:, None])
    x_mask = (rows[:, None] < M) & (ks[None, :] < K)
    x_lo = (
        tl.where(
            b_lo == 0,
            XQ + rows[:, None] * SXQ,
            tl.where(b_lo == 1, XK + rows[:, None] * SXK, XV + rows[:, None] * SXV),
        )
        + ks[None, :]
    )
    if b_lo == b_hi:
        acc = tl.zeros((MP, BN), tl.float32)
        for _ in range(0, K, BK):
            w = tl.load(w_ptrs, (cols[None, :] < N) & (ks[:, None] < K), other=0)
            x = tl.load(x_lo, x_mask, other=0)
            acc += tl.dot(x, w)
            x_lo += BK
            w_ptrs += BK
            ks += BK
        tl.store(
            Y + (rows[:, None] * SYM + cols[None, :]),
            acc.to(Y.dtype.element_ty),
            (rows[:, None] < M) & (cols[None, :] < N),
        )
        return
    x_hi = (
        tl.where(b_hi == 1, XK + rows[:, None] * SXK, XV + rows[:, None] * SXV)
        + ks[None, :]
    )
    acc_lo = tl.zeros((MP, BN), tl.float32)
    acc_hi = tl.zeros((MP, BN), tl.float32)
    for _ in range(0, K, BK):
        x_mask = (rows[:, None] < M) & (ks[None, :] < K)
        w = tl.load(w_ptrs, (cols[None, :] < N) & (ks[:, None] < K), other=0)
        acc_lo += tl.dot(tl.load(x_lo, x_mask, other=0), w)
        acc_hi += tl.dot(tl.load(x_hi, x_mask, other=0), w)
        x_lo += BK
        x_hi += BK
        w_ptrs += BK
        ks += BK
    segment = (cols >= S1).to(tl.int32) + (cols >= S2).to(tl.int32)
    acc = tl.where((segment == b_lo)[None, :], acc_lo, acc_hi)
    tl.store(
        Y + (rows[:, None] * SYM + cols[None, :]),
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


class _Config(NamedTuple):
    bn: int
    bk: int
    split: int
    k_per_split: int
    num_warps: int
    num_stages: int


_SMEM_LIMIT_BYTES = 156 * 1024
_PARTIALS_WS: dict[tuple[int, int], torch.Tensor] = {}
_RETIRED_PARTIALS_WS: list[torch.Tensor] = []


@functools.cache
def _config(nb: int, n: int, k: int) -> _Config:
    num_sms = num_compute_units()
    bk = min(128, max(16, triton.next_power_of_2(k)))
    k_tiles = triton.cdiv(k, bk)
    # On 74-SM CMP 170HX (num_sms <= 80), preserve the exact Round-5 0-ULP
    # tile/split schedule (BN=128 on 8600x2304 = 68 CTAs = 44.0 us) while
    # eliminating per-step torch.empty allocations via _partials_workspace.
    if num_sms <= 80:
        for bn in (16, 32, 64):
            if nb * triton.cdiv(n, bn) > _MAX_TILES:
                continue
            split = min(32, k_tiles, num_sms // max(1, nb * triton.cdiv(n, bn)))
            if split >= 2:
                tiles_per_split = triton.cdiv(k_tiles, split)
                split = triton.cdiv(k_tiles, tiles_per_split)
                if split >= 2:
                    return _Config(bn, bk, split, tiles_per_split * bk, 4, 4)
        if nb == 1 and 4608 < n < 16384:
            bn = 128
        elif n >= 16384 or (nb == 1 and n > 2048):
            bn = 64
        else:
            bn = 32
        return _Config(bn, bk, 1, k_tiles * bk, 4, 4)
    # On 108-SM A100 (num_sms > 80), only split-K when bn=32 cannot fill >= 60% of SMs:
    if nb * triton.cdiv(n, 32) < (num_sms * 3) // 5:
        for bn in (16, 32, 64):
            if nb * triton.cdiv(n, bn) > _MAX_TILES:
                continue
            split = min(32, k_tiles, num_sms // max(1, nb * triton.cdiv(n, bn)))
            if split >= 2:
                tiles_per_split = triton.cdiv(k_tiles, split)
                split = triton.cdiv(k_tiles, tiles_per_split)
                if split >= 2:
                    return _Config(bn, bk, split, tiles_per_split * bk, 4, 4)
    if nb == 1 and n <= 2304:
        if k >= 6144:
            return _Config(16, bk, 1, k_tiles * bk, 2, 5)
        return _Config(32, bk, 1, k_tiles * bk, 4, 5)
    if nb == 1 and n <= 10240:
        return _Config(32, bk, 1, k_tiles * bk, 4, 4)
    if n > 10240:
        return _Config(64, bk, 1, k_tiles * bk, 4, 3)
    bn = 64 if nb == 1 and n > 2304 else 32
    return _Config(bn, bk, 1, k_tiles * bk, 4, 4)


@functools.cache
def _counter(stream: int, device_idx: int) -> torch.Tensor:
    return torch.zeros(
        _MAX_TILES, dtype=torch.int32, device=torch.device("cuda", device_idx)
    )


def _partials_workspace(
    stream: int, device: torch.device, needed: int
) -> torch.Tensor:
    key = (stream, device.index or 0)
    buf = _PARTIALS_WS.get(key)
    if buf is None or buf.numel() < needed:
        if buf is not None:
            _RETIRED_PARTIALS_WS.append(buf)
        cap = max(needed, 2 * 1024 * 1024)
        buf = torch.empty(cap, dtype=torch.float32, device=device)
        _PARTIALS_WS[key] = buf
    return buf[:needed]


def _ll_gemm(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, out: torch.Tensor
) -> None:
    nb, m, k = x.shape
    n = w.shape[1]
    max_m = MAX_BATCHED_M if nb > 1 else MAX_M
    mp = 16 if m <= 16 else (32 if m <= 32 else 64)
    assert 0 < m <= max_m and w.shape == (nb, n, k)
    assert x.stride(2) == w.stride(2) == 1
    assert bias is None or (bias.shape == (n,) and bias.stride(0) == 1)
    assert out.shape == (nb, m, n) and out.stride(2) == 1
    cfg = _config(nb, n, k)
    n_tiles = triton.cdiv(n, cfg.bn)
    partials = counter = None
    stream_id = current_stream().cuda_stream
    if cfg.split > 1:
        assert nb * n_tiles <= _MAX_TILES
        needed = cfg.split * nb * mp * n
        partials = _partials_workspace(stream_id, x.device, needed)
        counter = _counter(stream_id, x.device.index or 0)
    stage_bytes = mp * cfg.bk * 2 + cfg.bn * cfg.bk * 2
    num_stages = max(1, min(cfg.num_stages, _SMEM_LIMIT_BYTES // stage_bytes))
    _ll_gemm_kernel[(n_tiles, nb, cfg.split)](
        x,
        w,
        bias,
        out,
        partials,
        counter,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BN=cfg.bn,
        BK=cfg.bk,
        K_PER_SPLIT=cfg.k_per_split,
        SPLIT=cfg.split,
        MP=mp,
        num_warps=cfg.num_warps,
        num_stages=num_stages,
    )


def use_sm80_qkv_bmm(
    x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, w: torch.Tensor
) -> bool:
    return (
        x_q.is_cuda
        and x_q.ndim == 2
        and 0 < x_q.shape[0] <= MAX_M
        and x_q.shape == x_k.shape == x_v.shape
        and x_q.dtype == x_k.dtype == x_v.dtype == w.dtype == torch.bfloat16
        and x_q.stride(1) == x_k.stride(1) == x_v.stride(1) == w.stride(1) == 1
        and is_sm80()
    )


def sm80_qkv_bmm(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w: torch.Tensor,
    k_start: int,
    v_start: int,
) -> torch.Tensor:
    xs = (x_q, x_k, x_v)
    m, k = x_q.shape
    n = w.shape[0]
    s1, s2 = k_start, v_start
    bn = 64 if (s1 >= 64 and s1 + 64 <= s2 <= n) else 32
    assert 0 < m <= MAX_M and w.shape == (n, k)
    assert all(x.shape == (m, k) and x.stride(1) == 1 for x in xs)
    assert w.stride(1) == 1 and s1 > 0 and s1 + bn <= s2 <= n
    out = torch.empty((m, n), dtype=x_q.dtype, device=x_q.device)
    _qkv_bmm_kernel[(triton.cdiv(n, bn),)](
        x_q,
        x_k,
        x_v,
        w,
        out,
        m,
        n,
        k,
        s1,
        s2,
        x_q.stride(0),
        x_k.stride(0),
        x_v.stride(0),
        w.stride(0),
        out.stride(0),
        BN=bn,
        BK=128,
        MP=_MP,
        num_warps=4,
        num_stages=4,
    )
    return out


def sm80_bf16_ll_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    n = weight.shape[-2]
    if weight.ndim == 3:
        if out is None:
            out = x.new_empty((*x.shape[:2], n), dtype=out_dtype or x.dtype)
        _ll_gemm(x, weight, bias, out)
        return out
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if out is None:
        out = x.new_empty((x2.shape[0], n), dtype=out_dtype or x.dtype)
    if x2.shape[0]:
        _ll_gemm(x2[None], weight[None], bias, out.view(-1, n)[None])
    return out.view(*lead, n)


def use_sm80_bf16_ll_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    if weight.ndim == 2:
        shapes_ok = x.ndim >= 2
        m = x.numel() // max(x.shape[-1], 1)
        max_m = MAX_M
    else:
        shapes_ok = weight.ndim == 3 and x.ndim == 3 and x.shape[0] == weight.shape[0]
        m = x.shape[-2] if x.ndim >= 2 else 0
        max_m = MAX_BATCHED_M if weight.shape[0] > 1 else MAX_M
    return (
        shapes_ok
        and x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[-1] == weight.shape[-1]
        and x.stride(-1) == weight.stride(-1) == 1
        and 0 < m <= max_m
        and (
            bias is None
            or (
                bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
        and is_sm80()
    )
