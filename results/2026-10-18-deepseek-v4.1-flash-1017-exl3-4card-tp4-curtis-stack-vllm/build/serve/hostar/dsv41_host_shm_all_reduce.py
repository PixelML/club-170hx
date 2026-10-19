# dsv41_host_shm_all_reduce.py -- vendored for the Pixel ML TC-MoE port on 4x CMP 170HX (Oct 18 2026).
# The file comes from Morrowmake (@Morrowmake), https://github.com/Morrowmake/vllm-cmp170hx @ 18305948acba3f4c4d912cdd56f2202d2ebd94e8,
# vllm/sm80/distributed/host_shm_all_reduce.py (Apache-2.0). Curtis White (@CurtisAccelerate) cherry-picked it for DSV4.1 TP4
# (DeepSeekv41-Flash-EXL3-TC-MoE @cd83ade1). Only change: the VLLM_CUSTOM_ALLREDUCE_PCIE_P2P hand-over probe (needs the fork's
# vllm.sm80 package) is stubbed to "not requested", which is the peer-off setting both of them measured.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-staged all-reduce for multiple GPUs on one node without peer-to-peer.

Why this exists
---------------
On PCIe-only parts whose driver refuses peer access -- the NVIDIA CMP 170HX
(GA100) is the case this was built for -- every CUDA-IPC collective disables
itself. `CustomAllreduce` returns early ("not supported on more than two
PCIe-only GPUs"), `SymmMemCommunicator` and the FlashInfer PCIe-IPC path need
the same peer mappings, and every all-reduce falls through to NCCL's SHM
transport. NCCL's ring then needs 2(N-1) sequential host-staged hops, so a
32 KiB decode message costs about as much as one 16x larger: it is
latency-bound, not bandwidth-bound.

This communicator replaces the IPC device buffers of vLLM's own one-shot
algorithm (`csrc/custom_all_reduce.cuh`) with **shared host memory**: one POSIX
shm segment mmap'd by all TP ranks, so they share the same physical pages, and
`cudaHostRegister(Mapped|Portable)` in each rank so every GPU has a device
pointer to them. Ranks synchronise on flags in that same host memory and the
kernels read peers' contributions zero-copy over PCIe. No peer access, no IPC
handles.

Three algorithms, dispatched on message size only (so every rank picks the same
one):

  * one-shot below `TWO_SHOT_MIN_BYTES`: publish N bytes, barrier, reduce all
    ranks' slots. 4N host bytes per rank, one barrier.
  * two-shot at or above it: reduce-scatter then all-gather. 2.5N host bytes
    per rank, two barriers. Wins once bandwidth dominates the barrier.
  * at or above `DMA_MIN_BYTES` the publish goes through a D2H
    `cudaMemcpyAsync` instead of kernel stores, because host writes from a
    kernel run at 4.01 GB/s against 6.70 for the copy engine.

Measured on 4x CMP 170HX, PCIe Gen2 x16, against NCCL in the same process:
8 KiB 2.04x, 32 KiB 1.06x, 64 KiB 1.59x, 128 KiB 1.43x, 256 KiB 1.41x; 16 MiB
is left to NCCL (0.99x) because prefill is already near wire speed.

Exactness
---------
Contributions are reduced in fixed rank order 0..N-1 with FP32 accumulation, so
the result is bitwise identical run to run and bitwise identical on every rank.
Because NCCL's RING_LL accumulates in BF16 while this accumulates in FP32, the
error against an exact FP64 sum is *lower* than NCCL's, not higher. No
precision shortcuts of any kind.

CUDA graphs
-----------
Capturable. The generation counter that drives the flag protocol lives in a
device tensor the kernel itself increments, never in a kernel argument, so a
captured graph replays correctly: replay k performs all-reduce k. There is no
host-side spin, no host sync, and no allocation whose address can move. This is
essential here -- the target model issues most of its decode all-reduces from
inside one captured graph.

Safety
------
Every in-kernel spin is bounded by a `clock64()` budget. On timeout the block
sets an error word in the shared header and returns without writing its output,
so a peer that dies turns into a loud error rather than a wedged GPU. The host
side checks those words periodically and raises.

Setup is collective: every step ends in an agreement over the CPU group, so
either every rank enables the host path or every rank falls back to NCCL. A
rank whose setup fails can never go on alone and leave its peers waiting in a
collective it will not join.

Enabling
--------
Off by default. `VLLM_HOST_ALLREDUCE=1` turns it on. It then stands
aside for vLLM's own `CustomAllreduce` only when the operator has asked for
that, by also setting `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1` -- the same env
that lets `CustomAllreduce` accept PCIe peer-to-peer above two GPUs -- and peer
access is really present. The two fast paths are therefore mutually exclusive
by choice, not by what the driver happens to report: installing a P2P-capable
driver on its own does not silently turn `VLLM_HOST_ALLREDUCE=1` into
NCCL. With the flag off, not one byte of this module's state is constructed and
the dispatch chain in `CudaCommunicator.all_reduce` is byte-identical to
upstream.
"""

from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import time

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

SUPPORTED_WORLD_SIZES = (2, 4, 8)

# --------------------------------------------------------------------------
# Tunables, all measured on 4x CMP 170HX.
# The thresholds below use those measurements.
# --------------------------------------------------------------------------
MAX_BLOCKS = 32            # must match MAX_BLOCKS in the CUDA source
MAX_RANKS = 8              # must match MAX_RANKS in the CUDA source
HEADER_BYTES = 256 * 1024  # must match HEADER_BYTES in the CUDA source
FLAG_STRIDE = 128          # bytes per flag line; must match `struct Flag`
N_BARRIERS = 2             # must match the first extent of `Header::arrive`
PAGE = 4096

DEFAULT_THREADS = 1024
# Grid cap.  Every block runs its own barrier, and a barrier costs one 128-byte
# host line written plus NR-1 read *per block*, so blocks are not free: at the
# 32-block cap a single barrier moves 4 KiB of flags per rank and 32 blocks
# shred the host reads into nunits/(nb*NR)-unit runs.  The whole kernel is
# PCIe-bound, not SM-bound, so one wide block wins.  Verified by a parameter sweep.
DEFAULT_BLOCK_CAP = 1
DEFAULT_HOST_CAP = 512 * 1024   # above this, fall back to NCCL
DEFAULT_SPIN_TIMEOUT_S = 5.0
SM_CLOCK_HZ = 2.0e9        # upper bound on the GA100 SM clock, for the spin timeout
# One-shot moves 4N host bytes per rank, two-shot 2.5N but pays a second
# barrier, so two-shot wins once the bandwidth term dominates the ~5-10 us
# barrier.  Measured crossover, not a model.
TWO_SHOT_MIN_BYTES = 64 * 1024
# At or above this, publish with the copy engine instead of kernel stores: a
# D2H cudaMemcpyAsync into the registered shm pages runs at 6.70 GB/s against
# 4.01 for kernel stores with all four cards busy.  The
# copy-engine path also costs ~10 us of setup and forfeits the store-drain
# overlap the in-kernel publish gets for free, so it only pays once the publish
# is large: measured a clear win at 256 KiB and a loss at 64-128 KiB.
DMA_MIN_BYTES = 256 * 1024

_SUPPORTED = (torch.bfloat16, torch.float32)

# Pauses before re-trying a refused `cudaHostRegister` (see `_register`). Setup
# runs once per process, so the worst case costs well under a second.
REGISTER_RETRY_DELAYS_S = (0.05, 0.2, 0.5)

# --------------------------------------------------------------------------
CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _e = (expr);                                                  \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e));   \
  } while (0)

namespace hostshm {

constexpr int MAX_BLOCKS = 32;
constexpr int MAX_RANKS = 8;
constexpr int64_t HEADER_BYTES = 256 * 1024;

// One flag per 128-byte line: four GPUs writing different words of one line
// through the root complex is the classic way to make a PCIe barrier slow.
struct alignas(128) Flag {
  uint32_t v;
  uint32_t pad[31];
};

struct Header {
  // [barrier][generation parity][block][rank].  Two barriers: the one-shot
  // path uses arrive[0] only, the two-shot path uses both.
  Flag arrive[2][2][MAX_BLOCKS][MAX_RANKS];  // 2*2*32*8*128 = 131072 B
  Flag err[MAX_RANKS];                       //         8*128 =   1024 B
};
static_assert(sizeof(Header) <= HEADER_BYTES, "header does not fit");

// Shared-segment layout, in units of slot_bytes after the header:
//
//   [2][NR]  scatter_p    generation-double-buffered, written by kernel stores
//   [2]      gather       generation-double-buffered, one shared buffer
//   [NR]     scatter_dma  NOT double-buffered, written by the copy engine
//
// `scatter_dma` needs no parity, which is what makes a capturable copy-engine
// publish possible at all: a graph node's address cannot depend on a counter
// living in device memory.  It is safe single-buffered because barrier 1 of the
// two-shot already proves every peer has finished reading our scatter slot -
// a peer releases arrive[1] only after its reduce, and its reduce is the only
// thing that reads us.  The gather buffer does need parity, because nothing
// orders a fast rank's next-generation gather write against a slow peer's
// current-generation gather read.  The one-shot has no second barrier, so it
// keeps using `scatter_p`.
__device__ __forceinline__ uint8_t* gather_base(uint8_t* base, int p, int nr,
                                                int64_t slot_bytes) {
  return base + HEADER_BYTES +
         (static_cast<int64_t>(2) * nr + p) * slot_bytes;
}

__device__ __forceinline__ uint8_t* scatter_dma_base(uint8_t* base, int nr,
                                                     int64_t slot_bytes) {
  return base + HEADER_BYTES + (static_cast<int64_t>(2) * nr + 2) * slot_bytes;
}

template <typename T, int N>
struct __align__(sizeof(T) * N) Vec {
  T d[N];
};

__device__ __forceinline__ float up(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float up(float x) { return x; }
__device__ __forceinline__ void down(float x, __nv_bfloat16* o) { *o = __float2bfloat16(x); }
__device__ __forceinline__ void down(float x, float* o) { *o = x; }

__device__ __forceinline__ void st_flag_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(v), "l"(p) : "memory");
}

__device__ __forceinline__ uint32_t ld_flag_acquire(uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Bounded block-scope barrier: publish `gen` for this block, then wait for the
// NR-1 peers.  Plain st.release.sys / ld.acquire.sys, never atomics - global
// atomics on this part collapse under contention.  Returns true if the wait hit
// its clock64() budget, in which case the caller must bail instead of hanging.
template <int NR>
__device__ __forceinline__ bool block_barrier(Flag* row, uint32_t gen, int rank,
                                              unsigned long long spin_cycles,
                                              int* aborted) {
  const int tid = threadIdx.x;
  if (tid == 0) *aborted = 0;
  __syncthreads();
  __threadfence_system();
  if (tid == 0) st_flag_release(&row[rank].v, gen);
  if (tid < NR && tid != rank) {
    uint32_t* f = &row[tid].v;
    const long long t0 = clock64();
    while (ld_flag_acquire(f) != gen) {
      if ((unsigned long long)(clock64() - t0) > spin_cycles) {
        *aborted = 1;
        break;
      }
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
  }
  __syncthreads();
  return *aborted != 0;
}

// Never hang: flag the timeout, advance the counter so the next call is not
// stuck in the same generation, and leave `out` untouched so validation shouts.
__device__ __forceinline__ void bail(Header* h, uint32_t* seq, int b, int rank,
                                     uint32_t gen) {
  if (threadIdx.x == 0) {
    h->err[rank].v = gen | 0x80000000u;
    seq[b] = gen;
  }
}

// One-shot: publish own slot, barrier, reduce all slots in fixed rank order.
template <typename T, int N, int NR>
__global__ void __launch_bounds__(1024, 1)
oneshot(const Vec<T, N>* __restrict__ inp,
        Vec<T, N>* __restrict__ out,
        uint8_t* __restrict__ base,
        uint32_t* __restrict__ seq,
        int rank,
        int64_t nunits,
        int64_t slot_bytes,
        unsigned long long spin_cycles) {
  Header* h = reinterpret_cast<Header*>(base);
  const int b = blockIdx.x;
  const int nb = gridDim.x;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t gen = seq[b] + 1u;
  const int p = static_cast<int>(gen & 1u);

  // Our own contribution is already in device memory, so phase 3 reads it from
  // `inp` instead of pulling our own slot back over PCIe: host reads per rank
  // drop from 4N to 3N.  Same values in the same fixed rank order, so the
  // result stays bitwise identical run-to-run and across ranks.
  uint8_t* const slots =
      base + HEADER_BYTES + static_cast<int64_t>(p) * NR * slot_bytes;
  Vec<T, N>* const mine =
      reinterpret_cast<Vec<T, N>*>(slots + static_cast<int64_t>(rank) * slot_bytes);
  const Vec<T, N>* src[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    src[r] = (r == rank)
                 ? inp
                 : reinterpret_cast<const Vec<T, N>*>(
                       slots + static_cast<int64_t>(r) * slot_bytes);
  }

  const int64_t per = (nunits + nb - 1) / nb;
  const int64_t lo = static_cast<int64_t>(b) * per;
  const int64_t hi = (lo + per) < nunits ? (lo + per) : nunits;

  // ---- phase 1: publish this block's stripe of our contribution ----------
  for (int64_t i = lo + tid; i < hi; i += nt) mine[i] = inp[i];
  __shared__ int aborted;

  // ---- phase 2: bounded wait for the peers' stripe ----------------------
  if (block_barrier<NR>(h->arrive[0][p][b], gen, rank, spin_cycles, &aborted)) {
    bail(h, seq, b, rank, gen);
    return;
  }

  // ---- phase 3: reduce in fixed rank order 0..NR-1, FP32 accumulate ------
  for (int64_t i = lo + tid; i < hi; i += nt) {
    float acc[N];
#pragma unroll
    for (int k = 0; k < N; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      Vec<T, N> v = src[r][i];
#pragma unroll
      for (int k = 0; k < N; ++k) acc[k] += up(v.d[k]);
    }
    Vec<T, N> o;
#pragma unroll
    for (int k = 0; k < N; ++k) down(acc[k], &o.d[k]);
    out[i] = o;
  }

  if (tid == 0) seq[b] = gen;
}

// Two-shot: reduce-scatter into a shared gather buffer, then all-gather.
//
// Host bytes per rank fall from the one-shot's 4N (write N, read 3N) to 2.5N
// (write 3N/4 then N/4, read 3N/4 then 3N/4), at the cost of a second barrier.
// Block b owns stripe [lo, hi); within that stripe rank c owns sub-chunk c:
//   1. publish the stripe minus our own sub-chunk - the peers' sub-chunks are
//      the only part of it anybody ever reads from us;
//   2. barrier, then reduce our own sub-chunk from the NR-1 peer slots plus
//      `inp`, in fixed rank order, and publish that result once;
//   3. barrier, then copy the NR-1 peer-reduced sub-chunks into `out`.
// Sub-chunk c is summed by rank c alone and broadcast verbatim, so all ranks
// see bitwise identical bytes, and the accumulation order is still 0..NR-1.
template <typename T, int N, int NR>
__global__ void __launch_bounds__(1024, 1)
twoshot(const Vec<T, N>* __restrict__ inp,
        Vec<T, N>* __restrict__ out,
        uint8_t* __restrict__ base,
        uint32_t* __restrict__ seq,
        int rank,
        int64_t nunits,
        int64_t slot_bytes,
        unsigned long long spin_cycles) {
  Header* h = reinterpret_cast<Header*>(base);
  const int b = blockIdx.x;
  const int nb = gridDim.x;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t gen = seq[b] + 1u;
  const int p = static_cast<int>(gen & 1u);

  uint8_t* const slots =
      base + HEADER_BYTES + static_cast<int64_t>(p) * NR * slot_bytes;
  Vec<T, N>* const mine =
      reinterpret_cast<Vec<T, N>*>(slots + static_cast<int64_t>(rank) * slot_bytes);
  // Each sub-chunk of the gather buffer is written only by its owner, so the NR
  // ranks write disjoint ranges of one shared buffer - no per-rank copies.
  Vec<T, N>* const gath = reinterpret_cast<Vec<T, N>*>(
      base + HEADER_BYTES + static_cast<int64_t>(2) * NR * slot_bytes +
      static_cast<int64_t>(p) * slot_bytes);
  const Vec<T, N>* src[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    src[r] = (r == rank)
                 ? inp
                 : reinterpret_cast<const Vec<T, N>*>(
                       slots + static_cast<int64_t>(r) * slot_bytes);
  }

  const int64_t per = (nunits + nb - 1) / nb;
  const int64_t lo = static_cast<int64_t>(b) * per;
  const int64_t hi = (lo + per) < nunits ? (lo + per) : nunits;
  const int64_t len = hi > lo ? hi - lo : 0;
  // A function of (nunits, nb, NR) only, so every rank agrees on the split.
  const int64_t m0 = lo + (len * static_cast<int64_t>(rank)) / NR;
  const int64_t m1 = lo + (len * static_cast<int64_t>(rank + 1)) / NR;

  __shared__ int aborted;

  // ---- 1: publish the peers' sub-chunks of our contribution --------------
  for (int64_t i = lo + tid; i < m0; i += nt) mine[i] = inp[i];
  for (int64_t i = m1 + tid; i < hi; i += nt) mine[i] = inp[i];
  if (block_barrier<NR>(h->arrive[0][p][b], gen, rank, spin_cycles, &aborted)) {
    bail(h, seq, b, rank, gen);
    return;
  }

  // ---- 2: reduce our sub-chunk, fixed rank order, FP32 accumulate --------
  for (int64_t i = m0 + tid; i < m1; i += nt) {
    float acc[N];
#pragma unroll
    for (int k = 0; k < N; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      Vec<T, N> v = src[r][i];
#pragma unroll
      for (int k = 0; k < N; ++k) acc[k] += up(v.d[k]);
    }
    Vec<T, N> o;
#pragma unroll
    for (int k = 0; k < N; ++k) down(acc[k], &o.d[k]);
    gath[i] = o;
    out[i] = o;   // our own quarter of the result never goes back over PCIe
  }
  if (block_barrier<NR>(h->arrive[1][p][b], gen, rank, spin_cycles, &aborted)) {
    bail(h, seq, b, rank, gen);
    return;
  }

  // ---- 3: gather the peers' reduced sub-chunks ---------------------------
  for (int64_t i = lo + tid; i < m0; i += nt) out[i] = gath[i];
  for (int64_t i = m1 + tid; i < hi; i += nt) out[i] = gath[i];

  if (tid == 0) seq[b] = gen;
}

// Two-shot whose publish is done by the copy engine, not by kernel stores.
//
// Kernel stores into mapped host memory run at 4.01 GB/s, but a D2H
// cudaMemcpyAsync into the same cudaHostRegister'd pages runs at 6.70 GB/s with
// all four cards transferring at once - 1.67x, on the direction that is over
// half the candidate's cost.  The alternative considered
// was DMA on the *read* side, where zero-copy already matches the copy engine.
//
// So the launcher issues the two contiguous D2H copies of our contribution
// (the stripe minus our own sub-chunk) on the same stream, and this kernel
// starts at the barrier.  Stream ordering guarantees the copies have landed
// before the flag is released, and the kernel has no host stores outstanding at
// that point, so the release fence is cheap here - measurements showed a fence costs
// what it has to drain.
//
// Identical byte counts, accumulation order and owner-computed-and-broadcast
// structure as `twoshot`; only who performs the publish changes.
template <typename T, int N, int NR>
__global__ void __launch_bounds__(1024, 1)
twoshot_dma(const Vec<T, N>* __restrict__ inp,
            Vec<T, N>* __restrict__ out,
            uint8_t* __restrict__ base,
            uint32_t* __restrict__ seq,
            int rank,
            int64_t nunits,
            int64_t slot_bytes,
            unsigned long long spin_cycles) {
  Header* h = reinterpret_cast<Header*>(base);
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t gen = seq[0] + 1u;
  const int p = static_cast<int>(gen & 1u);

  uint8_t* const slots = scatter_dma_base(base, NR, slot_bytes);
  Vec<T, N>* const gath =
      reinterpret_cast<Vec<T, N>*>(gather_base(base, p, NR, slot_bytes));
  const Vec<T, N>* src[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    src[r] = (r == rank)
                 ? inp
                 : reinterpret_cast<const Vec<T, N>*>(
                       slots + static_cast<int64_t>(r) * slot_bytes);
  }

  const int64_t m0 = (nunits * static_cast<int64_t>(rank)) / NR;
  const int64_t m1 = (nunits * static_cast<int64_t>(rank + 1)) / NR;

  __shared__ int aborted;

  // ---- 1: the copy engine already published; just announce it ------------
  if (block_barrier<NR>(h->arrive[0][p][0], gen, rank, spin_cycles, &aborted)) {
    bail(h, seq, 0, rank, gen);
    return;
  }

  // ---- 2: reduce our sub-chunk, fixed rank order, FP32 accumulate --------
  for (int64_t i = m0 + tid; i < m1; i += nt) {
    float acc[N];
#pragma unroll
    for (int k = 0; k < N; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      Vec<T, N> v = src[r][i];
#pragma unroll
      for (int k = 0; k < N; ++k) acc[k] += up(v.d[k]);
    }
    Vec<T, N> o;
#pragma unroll
    for (int k = 0; k < N; ++k) down(acc[k], &o.d[k]);
    gath[i] = o;
    out[i] = o;
  }
  if (block_barrier<NR>(h->arrive[1][p][0], gen, rank, spin_cycles, &aborted)) {
    bail(h, seq, 0, rank, gen);
    return;
  }

  // ---- 3: gather the peers' reduced sub-chunks ---------------------------
  for (int64_t i = tid; i < m0; i += nt) out[i] = gath[i];
  for (int64_t i = m1 + tid; i < nunits; i += nt) out[i] = gath[i];

  if (tid == 0) seq[0] = gen;
}

}  // namespace hostshm

// ---------------------------------------------------------------------------
// A failed registration is not fatal to the process (the caller falls back to
// NCCL), so the error must not stay behind as the runtime's last error: the
// next unrelated kernel-launch check would report it as its own.
int64_t host_register(int64_t ptr, int64_t nbytes) {
  void* p = reinterpret_cast<void*>(ptr);
  cudaError_t e = cudaHostRegister(p, static_cast<size_t>(nbytes),
                                   cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (e != cudaSuccess) {
    (void)cudaGetLastError();
    TORCH_CHECK(false, "cudaHostRegister: ", cudaGetErrorString(e),
                " (", static_cast<int>(e), ")");
  }
  void* dev = nullptr;
  e = cudaHostGetDevicePointer(&dev, p, 0);
  if (e != cudaSuccess) {
    (void)cudaGetLastError();
    cudaHostUnregister(p);
    (void)cudaGetLastError();
    TORCH_CHECK(false, "cudaHostGetDevicePointer: ", cudaGetErrorString(e),
                " (", static_cast<int>(e), ")");
  }
  return reinterpret_cast<int64_t>(dev);
}

void host_unregister(int64_t ptr) {
  cudaHostUnregister(reinterpret_cast<void*>(ptr));
}

#define LAUNCH(KERNEL, T, N, NR)                                               \
  hostshm::KERNEL<T, N, NR><<<blocks, threads, 0, stream>>>(                   \
      reinterpret_cast<const hostshm::Vec<T, N>*>(inp.data_ptr()),             \
      reinterpret_cast<hostshm::Vec<T, N>*>(out.data_ptr()),                   \
      reinterpret_cast<uint8_t*>(dev_base),                                    \
      reinterpret_cast<uint32_t*>(seq.data_ptr()), static_cast<int>(rank),     \
      nunits, slot_bytes, static_cast<unsigned long long>(spin_cycles))

#define DISPATCH_NR(KERNEL, T, N)                                              \
  switch (world) {                                                             \
    case 2: LAUNCH(KERNEL, T, N, 2); break;                                    \
    case 4: LAUNCH(KERNEL, T, N, 4); break;                                    \
    case 8: LAUNCH(KERNEL, T, N, 8); break;                                    \
    default: TORCH_CHECK(false, "unsupported world size ", world);             \
  }

#define DISPATCH_T(KERNEL)                                                     \
  if (inp.scalar_type() == at::kBFloat16) {                                    \
    DISPATCH_NR(KERNEL, __nv_bfloat16, 8);                                     \
  } else if (inp.scalar_type() == at::kFloat) {                                \
    DISPATCH_NR(KERNEL, float, 4);                                             \
  } else {                                                                     \
    TORCH_CHECK(false, "unsupported dtype ", inp.dtype());                     \
  }

void all_reduce(at::Tensor inp, at::Tensor out, int64_t dev_base, int64_t host_base,
                at::Tensor seq,
                int64_t rank, int64_t world, int64_t slot_bytes, int64_t blocks,
                int64_t threads, int64_t spin_cycles, int64_t algo) {
  TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(inp.dtype() == out.dtype() && inp.numel() == out.numel(), "shape/dtype mismatch");
  TORCH_CHECK(seq.is_cuda() && seq.scalar_type() == at::kInt, "seq must be an int32 CUDA tensor");
  TORCH_CHECK(blocks >= 1 && blocks <= hostshm::MAX_BLOCKS, "bad block count ", blocks);
  TORCH_CHECK(threads >= 32 && threads <= 1024 && (threads % 32) == 0, "bad thread count");
  TORCH_CHECK(seq.numel() >= hostshm::MAX_BLOCKS, "seq too small");

  const int64_t nbytes = inp.numel() * inp.element_size();
  TORCH_CHECK(nbytes % 16 == 0, "host path needs a 16-byte multiple, got ", nbytes);
  TORCH_CHECK(nbytes <= slot_bytes, "message ", nbytes, " exceeds slot ", slot_bytes);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(inp.data_ptr()) % 16 == 0, "inp not 16B aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "out not 16B aligned");

  const int64_t nunits = nbytes / 16;
  auto stream = at::cuda::getCurrentCUDAStream();
  // algo: 0 one-shot, 1 two-shot, 2 two-shot with a copy-engine publish.
  if (algo == 2) {
    // The publish is the stripe minus our own sub-chunk: two contiguous runs,
    // issued on this stream so the kernel below sees them completed.  The
    // destination is the single-buffered `scatter_dma` slot, so the address does
    // not depend on the device-side generation counter and the copy is a plain
    // graph node under capture.
    TORCH_CHECK(blocks == 1, "dma publish needs a single block, got ", blocks);
    TORCH_CHECK(host_base != 0, "dma publish needs the host base pointer");
    const int64_t m0 = (nunits * rank) / world;
    const int64_t m1 = (nunits * (rank + 1)) / world;
    uint8_t* const dst = reinterpret_cast<uint8_t*>(host_base) +
                         hostshm::HEADER_BYTES +
                         (2 * world + 2 + rank) * slot_bytes;
    const uint8_t* const srcb = reinterpret_cast<const uint8_t*>(inp.data_ptr());
    if (m0 > 0)
      CUDA_CHECK(cudaMemcpyAsync(dst, srcb, static_cast<size_t>(m0 * 16),
                                 cudaMemcpyDeviceToHost, stream));
    if (m1 < nunits)
      CUDA_CHECK(cudaMemcpyAsync(dst + m1 * 16, srcb + m1 * 16,
                                 static_cast<size_t>((nunits - m1) * 16),
                                 cudaMemcpyDeviceToHost, stream));
    DISPATCH_T(twoshot_dma);
  } else if (algo == 1) {
    DISPATCH_T(twoshot);
  } else {
    DISPATCH_T(oneshot);
  }
  CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("host_register", &host_register, "cudaHostRegister + GetDevicePointer");
  m.def("host_unregister", &host_unregister, "cudaHostUnregister");
  m.def("all_reduce", &all_reduce,
        "host-staged one-shot / two-shot / dma-publish two-shot all-reduce");
}
"""

# --------------------------------------------------------------------------
_MODULE = None


def _build_dir() -> str:
    """Where the JIT extension is cached.

    Never inside the vLLM source tree: a build artefact there would show up as a
    dirty working tree and be shipped in a wheel. TORCH_EXTENSIONS_DIR, else
    ~/.cache/vllm/host_shm_all_reduce.
    """
    d = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "vllm", "host_shm_all_reduce"
    )
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def _load_module():
    """Compile (or reuse) the extension.  Frozen across calls."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    build = _build_dir()
    tag = hashlib.sha1(CUDA_SRC.encode()).hexdigest()[:10]
    name = f"hostshm_allreduce_{tag}"
    src = os.path.join(build, f"{name}.cu")
    if not os.path.exists(src):
        tmp = src + f".{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(CUDA_SRC)
        os.replace(tmp, src)
    _MODULE = load(
        name=name,
        sources=[src],
        build_directory=build,
        extra_cflags=["-O3", "-std=c++17"],
        # No --use_fast_math: it would change rounding, and precision is frozen.
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                           "-gencode", "arch=compute_80,code=sm_80"],
        verbose=False,
    )
    return _MODULE



def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _p2p_unavailable(world_size: int) -> bool:
    """True when no pair of the visible devices can access each other.

    A probe, not a policy: `_stand_aside_for_custom_allreduce` decides what to
    do with the answer. Checked with `torch.cuda.can_device_access_peer`, which
    is what `_can_p2p` in custom_all_reduce.py trusts under
    VLLM_SKIP_P2P_CHECK.
    """
    try:
        for i in range(world_size):
            for j in range(world_size):
                if i == j:
                    continue
                vi = current_platform.logical_device_id_to_visible_device_id(i)
                vj = current_platform.logical_device_id_to_visible_device_id(j)
                if torch.cuda.can_device_access_peer(vi, vj):
                    return False
        return True
    except Exception as exc:  # a probe failure is not a reason to take over
        logger.debug("host-shm all-reduce: P2P probe failed (%s)", exc)
        return False


def _p2p_handover_requested() -> bool:
    """Pixel ML vendored copy: hand-over to the P2P custom all-reduce is never requested (peer access off)."""
    return False


def _stand_aside_for_custom_allreduce(world_size: int) -> bool:
    """True when the operator has handed the fast path to `CustomAllreduce`.

    `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P` is that hand-over: it is the same
    env that lets `CustomAllreduce` count PCIe peer-to-peer as fully connected
    above two GPUs, so setting it means "use the device-memory path". Without
    it this communicator keeps serving even where peer access is advertised,
    because on a PCIe-only node above two GPUs upstream's `CustomAllreduce`
    disables itself anyway and standing aside would only hand the messages to
    NCCL. With it set, we stand aside only if peer access is genuinely there --
    if it is not, `CustomAllreduce` will refuse too and something has to serve.
    """
    from vllm.sm80.distributed.pcie_p2p import pcie_p2p_custom_allreduce_allowed

    if not pcie_p2p_custom_allreduce_allowed():
        return False
    return not _p2p_unavailable(world_size)


class HostShmAllreduce:
    """No-P2P all-reduce over shared host memory, shaped like CustomAllreduce.

    Mirrors the contract the other backends in `CudaCommunicator.all_reduce`
    use: construct it, check `disabled`, ask `should_host_ar(tensor)`, and if it
    says yes call `host_all_reduce(tensor)`. Anything it declines stays with
    NCCL -- this class never calls a collective of its own on the fast path.

    `should_host_ar` is a pure function of `(dtype, nbytes, device)` and of
    nothing rank-local. That is load-bearing: the in-kernel barrier requires
    **every rank to take the same branch on every call**, and it would deadlock
    (until the spin bound fires) if one rank used the host path while another
    went to NCCL.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = DEFAULT_HOST_CAP,
        spin_timeout_s: float = DEFAULT_SPIN_TIMEOUT_S,
        check_every: int = 512,
    ) -> None:
        self.disabled = True
        self._registered = False
        self._mm = None
        self._cbuf = None
        self.mod = None
        self.name: str | None = None
        self._created = False
        self._failed_ranks: list[int] = []
        self._local_error: str | None = None
        self._register_attempts = 0
        self.calls = 0
        self.check_every = max(1, int(check_every))

        if not current_platform.is_cuda():
            return

        self.group = group
        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "HostShmAllreduce should be attached to a non-NCCL (CPU) group"
        )
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size == 1:
            return
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            logger.warning_once(
                "Host-shm all-reduce is disabled: unsupported world size %d "
                "(supported: %s).",
                self.world_size,
                str(SUPPORTED_WORLD_SIZES),
            )
            return

        # One node only: the segment is shared memory, which does not span hosts.
        from vllm.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning_once(
                "Host-shm all-reduce is disabled: the group spans more than one "
                "node, and a /dev/shm segment cannot."
            )
            return

        # Stand-aside is decided per rank from a peer-access probe, so a probe
        # that answers differently on one rank would split the group: that rank
        # returns here while the others go on into the setup collectives below
        # and wait for it forever. The probe only runs when the operator has
        # set the hand-over env, so only then is the answer agreed: any rank
        # standing aside makes every rank stand aside.
        if _p2p_handover_requested():
            stand_aside = _stand_aside_for_custom_allreduce(self.world_size)
            if not self._agree(not stand_aside):
                logger.info_once(
                    "Host-shm all-reduce stands aside: "
                    "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=1 and this platform has "
                    "working GPU peer-to-peer, so the device-memory custom "
                    "all-reduce owns the fast path."
                )
                return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.threads = DEFAULT_THREADS
        self.block_cap = DEFAULT_BLOCK_CAP
        self.two_shot_min_bytes = TWO_SHOT_MIN_BYTES
        self.dma_min_bytes = DMA_MIN_BYTES
        self.host_cap = max(0, int(max_size))
        if self.host_cap < 16:
            logger.warning_once(
                "Host-shm all-reduce is disabled: max_size=%d leaves nothing to "
                "serve.",
                self.host_cap,
            )
            return
        self.slot_bytes = _round_up(max(self.host_cap, PAGE), PAGE)
        self.seg_bytes = HEADER_BYTES + (3 * self.world_size + 2) * self.slot_bytes
        self.spin_cycles = int(spin_timeout_s * SM_CLOCK_HZ)

        if not self._setup():
            where = (
                "rank(s) " + ", ".join(str(r) for r in self._failed_ranks)
                if self._failed_ranks
                else "a rank (the agreement collective itself failed)"
            )
            logger.warning(
                "Host-shm all-reduce is disabled: setup failed on %s%s. Every "
                "rank of the group falls back to NCCL for every message.",
                where,
                f" -- here: {self._local_error}" if self._local_error else "",
            )
            self.close()
            return

        self.disabled = False
        retried = (
            f", registration retried {self._register_attempts - 1}x on this rank"
            if self._register_attempts > 1
            else ""
        )
        logger.info_once(
            "Host-shm all-reduce enabled for messages <= %d B "
            "(world=%d, threads=%d, blocks=%d, two-shot >= %d B, DMA publish "
            ">= %d B, segment %.1f MiB; setup agreed by all %d ranks, "
            "registration one rank at a time%s).",
            self.host_cap,
            self.world_size,
            self.threads,
            self.block_cap,
            self.two_shot_min_bytes,
            self.dma_min_bytes,
            self.seg_bytes / (1 << 20),
            self.world_size,
            retried,
        )

    # -- setup ------------------------------------------------------------
    #
    # Setup is a fixed sequence of steps, and every rank runs every step's
    # collective whatever happened locally. Each step does its local work with
    # errors caught, then all ranks agree on the outcome (an all-gather of a
    # per-rank ok flag over the CPU group). A failure anywhere is therefore
    # seen by every rank at the same step, all of them stop there, and all of
    # them fall back to NCCL together. The earlier version let a rank whose
    # setup raised disable itself alone: it went on to the next collective
    # while its peers still waited in the setup barrier, and the boot hung.

    def _agree(self, ok: bool) -> bool:
        """All-gather `ok` over the group; True only if every rank is ok.

        Records which ranks failed. A failure of the collective itself (a peer
        died, a bogus group) counts as not ok -- it can never raise out of
        setup and can never wait forever, because gloo collectives fail when a
        peer's connection closes.
        """
        try:
            mine = torch.tensor([1 if ok else 0], dtype=torch.int32)
            flags = [torch.zeros(1, dtype=torch.int32) for _ in range(self.world_size)]
            dist.all_gather(flags, mine, group=self.group)
        except Exception as exc:
            self._failed_ranks = []
            if self._local_error is None:
                self._local_error = f"agreement failed: {type(exc).__name__}: {exc}"
            return False
        bad = [r for r, f in enumerate(flags) if int(f.item()) == 0]
        if bad:
            self._failed_ranks = bad
            return False
        return True

    def _step(self, fn) -> bool:
        """Run `fn` on this rank (None: nothing to do here), then agree."""
        ok = True
        if fn is not None:
            try:
                fn()
            except Exception as exc:
                ok = False
                self._local_error = f"{type(exc).__name__}: {exc}"
        return self._agree(ok)

    def _setup(self) -> bool:
        first = self.rank == 0
        try:
            # Rank 0 compiles first; the others reuse the cache rather than
            # racing the same ninja build directory.
            if not self._step(self._load if first else None):
                return False
            if not self._step(None if first else self._load):
                return False
            # A collective on every rank; it cannot fail on one rank alone
            # without the group's connections failing on all of them.
            try:
                self.name = self._segment_name()
            except Exception as exc:
                self._local_error = f"{type(exc).__name__}: {exc}"
                self.name = None
            if not self._agree(self.name is not None):
                return False
            if not self._step(self._create_segment if first else None):
                return False
            if not self._step(self._map_segment):
                return False
            # Register one rank at a time, rank 0 first. See _register.
            for r in range(self.world_size):
                if not self._step(self._register if self.rank == r else None):
                    return False
            return self._step(self._alloc_seq)
        finally:
            # Every rank has mapped the segment or given up, so the name can
            # go: mappings stay valid and nothing is left in /dev/shm, on
            # success, on failure, or if we are killed later.
            self._unlink_segment()

    def _load(self) -> None:
        self.mod = _load_module()

    def _segment_name(self) -> str:
        """A name every rank of this group agrees on, unique per group.

        Derived on rank 0 and broadcast over the CPU group, rather than from
        environment variables: two TP groups in one job, or a restart that
        reuses the same ports, must not collide on one segment.
        """
        token = None
        if self.rank == 0:
            token = hashlib.sha1(os.urandom(32)).hexdigest()[:16]
        box = [token]
        dist.broadcast_object_list(box, src=dist.get_global_rank(self.group, 0),
                                   group=self.group)
        return f"vllm_hostshm_ar_{box[0]}"

    def _path(self) -> str:
        return f"/dev/shm/{self.name}"

    def _create_segment(self) -> None:
        fd = os.open(self._path(), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        self._created = True
        try:
            os.ftruncate(fd, self.seg_bytes)
            # The flag words must start at zero, or the first generation
            # never matches.
            with mmap.mmap(fd, HEADER_BYTES) as hdr:
                hdr.write(b"\0" * HEADER_BYTES)
        finally:
            os.close(fd)

    def _map_segment(self) -> None:
        fd = os.open(self._path(), os.O_RDWR)
        try:
            self._mm = mmap.mmap(fd, self.seg_bytes)
        finally:
            os.close(fd)
        self._cbuf = ctypes.c_char.from_buffer(self._mm)
        self.host_ptr = ctypes.addressof(self._cbuf)
        assert self.host_ptr % PAGE == 0, "mmap base must be page aligned"

    def _register(self) -> None:
        """`cudaHostRegister` this rank's mapping of the shared segment.

        Called by one rank at a time, rank 0 first, never concurrently. The
        driver pins the pages with `pin_user_pages(FOLL_LONGTERM)`, and the
        kernel will not long-term pin a page that sits in a CMA pageblock (on
        Linux 6.16+ with Kexec HandOver on, its scratch areas are CMA
        pageblocks, tens of GiB of ordinary movable memory): it first migrates
        the page elsewhere. When several processes pin the same shared pages at
        once, each one's reference makes the others' migration fail, and the
        losers' `cudaHostRegister` returns `invalid argument` (the kernel log
        shows "Cannot map memory with base addr ... and size of ... pages").
        Registered alone, rank 0 migrates any such page once, and later ranks
        find every page already pinnable. A short retry covers any other
        transient refusal.
        """
        delays = REGISTER_RETRY_DELAYS_S
        for attempt, delay in enumerate((0.0,) + tuple(delays)):
            if delay:
                time.sleep(delay)
            self._register_attempts = attempt + 1
            try:
                self.dev_base = self.mod.host_register(self.host_ptr, self.seg_bytes)
            except Exception as exc:
                if attempt == len(delays):
                    raise
                logger.warning(
                    "Host-shm all-reduce: registering the segment failed on rank "
                    "%d (attempt %d, %s); retrying.",
                    self.rank,
                    attempt + 1,
                    exc,
                )
                continue
            self._registered = True
            return

    def _alloc_seq(self) -> None:
        self.seq = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=self.device)
        self._scratch: dict[tuple, torch.Tensor] = {}

    def _unlink_segment(self) -> None:
        if not self._created:
            return
        self._created = False
        try:
            os.unlink(self._path())
        except FileNotFoundError:
            pass

    # -- dispatch ---------------------------------------------------------
    def _blocks(self, nbytes: int) -> int:
        units = nbytes // 16
        return max(1, min(self.block_cap, (units + self.threads - 1) // self.threads))

    def should_host_ar(self, inp: torch.Tensor) -> bool:
        """Whether the host path serves this message.

        Depends only on dtype and byte count, both of which are identical on
        every rank of a TP group, so all ranks agree. See the class docstring
        for why that matters.
        """
        if self.disabled:
            return False
        if inp.dtype not in _SUPPORTED or not inp.is_cuda:
            return False
        nbytes = inp.numel() * inp.element_size()
        # A non-multiple of 16 would need a scalar tail path; NCCL can have it.
        return 0 < nbytes <= self.host_cap and nbytes % 16 == 0

    def _aligned(self, t: torch.Tensor) -> torch.Tensor:
        """The kernel needs a 16-byte-aligned source; stage if we were handed
        an odd offset into somebody's buffer."""
        if t.data_ptr() % 16 == 0:
            return t
        key = (t.dtype, t.numel())
        buf = self._scratch.get(key)
        if buf is None:
            buf = torch.empty(t.numel(), dtype=t.dtype, device=self.device)
            self._scratch[key] = buf
        buf.copy_(t.reshape(-1))
        return buf.view_as(t)

    def host_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Sum `inp` across the group. Input untouched; result is a new tensor.

        Only call this when `should_host_ar(inp)` is true. Safe to capture in a
        CUDA graph; issues on the current stream and performs no host sync.
        """
        if not inp.is_contiguous():
            inp = inp.contiguous()
        src = self._aligned(inp)
        out = torch.empty_like(inp)
        nbytes = inp.numel() * inp.element_size()
        blocks = self._blocks(nbytes)
        # Dispatch on nbytes only, so every rank picks the same kernel.
        if nbytes >= self.dma_min_bytes and blocks == 1:
            algo = 2
        elif nbytes >= self.two_shot_min_bytes:
            algo = 1
        else:
            algo = 0
        self.mod.all_reduce(
            src,
            out,
            self.dev_base,
            self.host_ptr,
            self.seq,
            self.rank,
            self.world_size,
            self.slot_bytes,
            blocks,
            self.threads,
            self.spin_cycles,
            algo,
        )
        self.calls += 1
        if self.calls % self.check_every == 0:
            self._check_error_flags()
        return out

    # -- diagnostics / teardown -------------------------------------------
    def error_flags(self) -> list[int]:
        """The per-rank error words from the shared header.

        Read straight from host memory with no synchronisation, so the value may
        lag the GPU by a few calls; a set flag is sticky, so it is still seen.
        """
        if self._mm is None:
            return []
        off = N_BARRIERS * 2 * MAX_BLOCKS * MAX_RANKS * FLAG_STRIDE
        return [
            int.from_bytes(self._mm[off + r * FLAG_STRIDE : off + r * FLAG_STRIDE + 4],
                           "little")
            for r in range(self.world_size)
        ]

    def _check_error_flags(self) -> None:
        flags = self.error_flags()
        if any(flags):
            self.disabled = True
            raise RuntimeError(
                "host-shm all-reduce: an in-kernel spin hit its "
                f"{self.spin_cycles} cycle bound (per-rank error flags {flags}). "
                "A peer rank stopped issuing collectives, or the ranks fell out "
                "of step. Results from the affected calls are invalid. Restart "
                "with VLLM_HOST_ALLREDUCE=0 to use NCCL only."
            )

    def close(self) -> None:
        self.disabled = True
        if self._registered:
            try:
                torch.cuda.synchronize()
                self.mod.host_unregister(self.host_ptr)
            except Exception as exc:
                logger.debug("host-shm all-reduce: unregister failed (%s)", exc)
            self._registered = False
        self._cbuf = None
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                pass  # a view is still alive; the process is going away anyway
            self._mm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
