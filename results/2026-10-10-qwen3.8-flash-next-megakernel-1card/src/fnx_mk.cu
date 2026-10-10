// fnx_mk.cu - Qwen3.8-Flash-Next (llama.cpp arch qwen4exp) decode megakernel for one CMP 170HX (sm_80).
//
// One persistent cooperative kernel runs a whole decode step (48 layers + head) as phases split by a
// software grid barrier, the design of L-Forster's open-jet megakernel for Qwen3.8-27B
// (https://github.com/L-Forster/open-jet/tree/master/megakernel, AGPL-3.0). The forward pass follows
// llama.cpp's src/models/qwen4exp.cpp (ggml-org/llama.cpp, MIT).
//
// Scope (v1): batch 1, greedy, no speculative decoding, context <= 2048 tokens. Up to 2048 cells the
// QSA indexer selects every cell, so dense attention is exact and the indexer is skipped.
// Weights: unsloth UD-Q3_K_XL GGUF. Dense matrices run as Q8_0 (Q6_K output re-quantised to Q8_0 at
// load), experts stay IQ3_XXS / IQ4_NL / Q8_0 (the IQ4_XS layer is converted to IQ4_NL losslessly in
// value, fp16-rounded in scale). The 26.8 GiB PLE n-gram table and the token embedding stay in the
// mmapped GGUF; the host gathers 16 PLE rows + 1 embedding row per token.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ggml.h"
#include "llama.h"
#include "fnx_tables.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

// ---------------------------------------------------------------- model constants
constexpr int NE = 2560, HC = 4, HCD = NE * HC, LR = 320, NL = 48;
constexpr int NEXP = 512, NUSED = 10, FF = 640;
constexpr int NH = 24, NKV = 2, HD = 256, NROT = 64;
constexpr int GK = 16, GV = 48, DS = 128, KDIM = GK * DS, VDIM = GV * DS, CONVCH = 2 * KDIM + VDIM;
constexpr int VOCAB = 248320;
constexpr int PLE_LAYER = 1, PLE_HEADS = 16, PLE_HD = 160, PLE_HIST = 10;  // ring >= (kernel-1)*dilation + 1
constexpr int MAXCTX = 2048;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 1e7f;
constexpr int EOS_TOK = 248044;

constexpr int NT = 512, NWARP = NT / 32;

// ---------------------------------------------------------------- device structs
struct Q8 { const int8_t * q; const __half * d; int rows, K; };
enum { ET_IQ3XXS = 0, ET_IQ4NL = 1, ET_Q8 = 2 };
struct EX { int type; const uint8_t * q; const __half * d; const uint8_t * s; int rows, K; size_t qe, de, se; };

struct Layer {
    int recr;
    Q8 hca_down, hca_up, hcf_down, hcf_up;
    const float *hca_norm, *hcf_norm, *hca_inj, *hcf_inj;
    Q8 qkv, z, ssm_out;
    const float *alpha, *beta, *dt, *a, *conv, *snorm;
    Q8 wq, wk, wv, wo;
    const float *qn, *kn;
    const float *router, *sh_gate_inp;
    Q8 sh_gate, sh_up, sh_down;
    EX eg, eu, ed;
    Q8 ple_k, ple_v;
    const float *ple_nk, *ple_nq, *ple_nc, *ple_conv;
    float * ssm_state;   // [GV][DS][DS], row j (value dim) contiguous over i (key dim)
    float * conv_ring;   // [4][CONVCH]
    __half *kc, *vc;     // [MAXCTX][NKV][HD]
};

struct Glob {
    Layer * L;
    Q8 head_down, head_up, out;
    const float * head_norm;
    float * res[2];
    float * blk_out;
    float * inj[2];    // [HC slices][HC] split-K partials
    float * lo;        // [HC][LR] split-K partials
    float * xn;        // HCD normalised residual of the current mix
    float * mixed;
    float * proj;      // projections of the token mixer, max 12288 + 1024
    float * core;      // mixer core output, VDIM
    float * rlog;      // router logits 512 + shared gate 1
    float * sh_h;      // 640
    float * exp_h;     // NUSED * FF
    int   * sel_id;    // NUSED
    float * sel_w;     // NUSED + 1 (last = shared gate sigmoid)
    float * ple_kv;    // HCD + NE
    float * ple_hist;  // PLE_HIST * HCD
    const float * emb;     // NE (device copy of the token embedding row)
    const float * ple_in;  // NE (device copy of the PLE rows)
    unsigned long long * best;
    int * out_tok;
    unsigned long long * prof;  // optional: [tag, t] pairs per barrier
    unsigned * bar_count;
    int dbg;  // micro-benchmark switches: bit0 skip staging, bit1 skip the GEMV part
    volatile unsigned * bar_gen;
};

// ---------------------------------------------------------------- small helpers
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }
__device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }
__device__ __forceinline__ int gwarp() { return warp_id() * gridDim.x + blockIdx.x; }
__device__ __forceinline__ int nwarps() { return NWARP * gridDim.x; }
__device__ __forceinline__ float ldcg(const float * p) { return __ldcg(p); }

// block-wide sum, every thread gets the result; red needs NWARP floats
__device__ float bsum(float v, float * red) {
    v = wsum(v);
    __syncthreads();
    if (lane_id() == 0) red[warp_id()] = v;
    __syncthreads();
    float t = lane_id() < NWARP ? red[lane_id()] : 0.0f;
    return wsum(t);
}

// software grid barrier (all blocks co-resident, cooperative launch): release-add on arrival, acquire-poll
// until the whole generation arrived; the counter is monotonic within a launch and zeroed before it.
// Same scheme as open-jet's gsync.
__device__ void gsync(const Glob & G, int & pk, int tag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned target = (unsigned) (pk + 1) * gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(G.bar_count) : "memory");
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(G.bar_count) : "memory");
        } while (v < target);
        if (G.prof && blockIdx.x == 0 && pk < 2046) {
            unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            G.prof[2 * (pk + 1)] = (unsigned long long) tag; G.prof[2 * (pk + 1) + 1] = t;
        }
    }
    ++pk;
    __syncthreads();
}

// ---------------------------------------------------------------- shared memory
struct SM {
    float * f;    // HCD floats
    float * f2;   // 6144 floats scratch
    int8_t * xq;  // HCD int8
    float * xd;   // HCD/32 scales
    float * red;  // 64
    uint32_t * grid;  // iq3xxs grid, 256
    int8_t * kv4;     // iq4nl values, 16
    uint64_t * sg64;  // ksigns64, 128: byte 0xff where the value is negative
};
__device__ SM smem_layout() {
    extern __shared__ __align__(16) uint8_t sm_raw[];
    SM s;
    s.f   = (float *) sm_raw;
    s.f2  = s.f + HCD;
    s.xq  = (int8_t *) (s.f2 + 6144);
    s.xd  = (float *) (s.xq + HCD);
    s.red = s.xd + HCD / 32;
    s.grid = (uint32_t *) (s.red + 64);
    s.kv4 = (int8_t *) (s.grid + 256);
    s.sg64 = (uint64_t *) (s.grid + 256 + 4);
    return s;
}
constexpr size_t SM_BYTES = HCD * 4 + 6144 * 4 + HCD + (HCD / 32) * 4 + 64 * 4 + 256 * 4 + 16 + 128 * 8;

// quantise n floats (n % 32 == 0) of smem src into xq / xd, one warp per 32-block
__device__ void quant_q8(const float * src, int n, int8_t * xq, float * xd) {
    for (int b = warp_id(); b < n / 32; b += NWARP) {
        const float v = src[b * 32 + lane_id()];
        const float amax = wmax(fabsf(v));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        xq[b * 32 + lane_id()] = (int8_t) __float2int_rn(v * id);
        if (lane_id() == 0) xd[b] = d;
    }
}

// stage N floats of global (written in this kernel): optional float copy in smem + q8 (xq/xd).
// 32-block b goes to warp b % NWARP; every load of a thread is issued before the first use.
template <int N>
__device__ void stage_q8(const float * src, float * fs, int8_t * xq, float * xd) {
    constexpr int NB = N / 32, U = (NB + NWARP - 1) / NWARP;
    float v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int b = warp_id() + u * NWARP;
        v[u] = b < NB ? ldcg(src + b * 32 + lane_id()) : 0.0f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int b = warp_id() + u * NWARP;
        if (b < NB) {
            const float amax = wmax(fabsf(v[u]));
            const float id = amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f;
            xq[b * 32 + lane_id()] = (int8_t) __float2int_rn(v[u] * id);
            if (fs) fs[b * 32 + lane_id()] = v[u];
            if (lane_id() == 0) xd[b] = amax * (1.0f / 127.0f);
        }
    }
}

// load n floats of global (written in this kernel) into smem
__device__ void load_vec(float * dst, const float * src, int n) {
    for (int i = threadIdx.x; i < n; i += NT) dst[i] = ldcg(src + i);
}

// ---------------------------------------------------------------- dot products (per lane partials)
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

__device__ __forceinline__ float dot_q8_lane(const Q8 & W, int row, const int8_t * xq, const float * xd) {
    const int nch = W.K / 16;
    const int8_t * wq = W.q + (size_t) row * W.K;
    const __half * wd = W.d + (size_t) row * (W.K / 32);
    float acc = 0.0f;
#pragma unroll 4
    for (int c = lane_id(); c < nch; c += 32) {
        const int4 w = __ldg((const int4 *) (wq + 16 * c));
        const int4 x = *(const int4 *) (xq + 16 * c);
        int s = dp4a(w.x, x.x, 0); s = dp4a(w.y, x.y, s); s = dp4a(w.z, x.z, s); s = dp4a(w.w, x.w, s);
        acc += (float) s * __half2float(wd[c >> 1]) * xd[c >> 1];
    }
    return acc;
}

__device__ __forceinline__ float dot_f32_lane(const float * w, int K, const float * x) {
    float acc = 0.0f;
    for (int c = lane_id(); c < K / 4; c += 32) {
        const float4 a = __ldg((const float4 *) (w + 4 * c));
        const float4 b = *(const float4 *) (x + 4 * c);
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    return acc;
}

// R rows of a Q8 matrix with a compile-time K against xq/xd; every weight load is issued before any math
template <int K, int R>
__device__ __forceinline__ void q8_dot_rows(const Q8 & W, const int (&row)[R], const int8_t * xq, const float * xd, float (&out)[R]) {
    constexpr int NC = K / 16, C = (NC + 31) / 32;
    int4 w[R][C];
    __half d[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int u = 0; u < C; ++u) {
            const int c = lane_id() + 32 * u;
            if (NC % 32 == 0 || c < NC) {
                w[r][u] = __ldg((const int4 *) (W.q + (size_t) row[r] * K + 16 * c));
                d[r][u] = __ldg(W.d + (size_t) row[r] * (K / 32) + (c >> 1));
            }
        }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float acc = 0.0f;
#pragma unroll
        for (int u = 0; u < C; ++u) {
            const int c = lane_id() + 32 * u;
            if (NC % 32 == 0 || c < NC) {
                const int4 x = *(const int4 *) (xq + 16 * c);
                int sm = dp4a(w[r][u].x, x.x, 0); sm = dp4a(w[r][u].y, x.y, sm); sm = dp4a(w[r][u].z, x.z, sm); sm = dp4a(w[r][u].w, x.w, sm);
                acc += (float) sm * __half2float(d[r][u]) * xd[c >> 1];
            }
        }
        out[r] = wsum(acc);
    }
}

// rows [0, nrows) of W, R rows per warp step; woff rotates the start warp so consecutive matrices of one
// phase spread over the grid. f(row, value) runs on lane 0. Returns the next woff.
template <int K, int R, class F>
__device__ __forceinline__ int gemv_q8(const Q8 & W, int nrows, int woff, const int8_t * xq, const float * xd, F f) {
    const int nw = nwarps();
    const int gw = (gwarp() + nw - woff % nw) % nw;
    for (int r0 = gw * R; r0 < nrows; r0 += nw * R) {
        int rows[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rows[r] = min(r0 + r, nrows - 1);
        float v[R];
        q8_dot_rows<K, R>(W, rows, xq, xd, v);
        if (lane_id() == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) if (r0 + r < nrows) f(r0 + r, v[r]);
        }
    }
    return woff + (nrows + R - 1) / R;
}

// f32 dot over NE elements, loads issued in two batches of 10 float4 per lane; result on every lane
__device__ __forceinline__ float f32_dot_ne(const float * w, const float * x) {
    constexpr int C = NE / 4 / 32, B = 10;
    float acc = 0.0f;
#pragma unroll
    for (int b0 = 0; b0 < C; b0 += B) {
        float4 a[B];
#pragma unroll
        for (int u = 0; u < B; ++u) a[u] = __ldg((const float4 *) (w + 4 * (lane_id() + 32 * (b0 + u))));
#pragma unroll
        for (int u = 0; u < B; ++u) {
            const float4 xx = *(const float4 *) (x + 4 * (lane_id() + 32 * (b0 + u)));
            acc += a[u].x * xx.x + a[u].y * xx.y + a[u].z * xx.z + a[u].w * xx.w;
        }
    }
    return wsum(acc);
}

// f32 rows with K = NE against a float vector in smem
template <class F>
__device__ __forceinline__ int gemv_f32(const float * W, int nrows, int ldw, int woff, const float * x, F f) {
    const int nw = nwarps();
    const int gw = (gwarp() + nw - woff % nw) % nw;
    for (int row = gw; row < nrows; row += nw) {
        const float acc = f32_dot_ne(W + (size_t) row * ldw, x);
        if (lane_id() == 0) f(row, acc);
    }
    return woff + nrows;
}

__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

__device__ __forceinline__ int2 tab16(int q4, const int8_t * table) {
    const uint32_t * t32 = (const uint32_t *) table;
    uint32_t tmp[2];
    const uint32_t sel = (0x32103210u | ((q4 & 0x88888888u) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t32[0], t32[1], q4 >> sh);
        const uint32_t hi = __byte_perm(t32[2], t32[3], q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

struct ExLd { int4 a, b; float d; };

template <int T> struct ExL;
template <> struct ExL<ET_IQ3XXS> { uint2 g; uint32_t aux; __half d; };
template <> struct ExL<ET_IQ4NL>  { int4 q; __half d; };
template <> struct ExL<ET_Q8>     { int4 a, b; __half d; };

template <int T>
__device__ __forceinline__ ExL<T> exl_load(const uint8_t * rq, const __half * rd, const uint8_t * rs, int sb) {
    ExL<T> l;
    if constexpr (T == ET_IQ3XXS) {
        const int B = sb >> 3, ib = sb & 7;
        l.g = __ldg((const uint2 *) (rq + B * 64 + ib * 8));
        l.aux = __ldg((const uint32_t *) (rs + B * 32 + ib * 4));
        l.d = __ldg(rd + B);
    } else if constexpr (T == ET_IQ4NL) {
        l.q = __ldg((const int4 *) (rq + sb * 16));
        l.d = __ldg(rd + sb);
    } else {
        l.a = __ldg((const int4 *) (rq + sb * 32));
        l.b = __ldg((const int4 *) (rq + sb * 32 + 16));
        l.d = __ldg(rd + sb);
    }
    return l;
}

// integer dot of one 32-block, times the block scale (not the activation scale)
template <int T>
__device__ __forceinline__ float exl_dot(const ExL<T> & l, const int8_t * xq, const uint32_t * grid, const int8_t * kv4,
                                         const uint64_t * sg64 = nullptr) {
    const int * x = (const int *) xq;
    int sumi = 0;
    if constexpr (T == ET_IQ3XXS) {
        const uint32_t g2[2] = { l.g.x, l.g.y };
        const uint8_t * q3 = (const uint8_t *) g2;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int gx = grid[q3[l0]], gy = grid[q3[l0 + 1]];
            const uint2 sg = *(const uint2 *) (sg64 + ((l.aux >> (7 * l0 / 2)) & 0x7f));
            const int s0 = (int) sg.x, s1 = (int) sg.y;
            sumi = dp4a(__vsub4(gx ^ s0, s0), x[l0], sumi);
            sumi = dp4a(__vsub4(gy ^ s1, s1), x[l0 + 1], sumi);
        }
        return __half2float(l.d) * ((float) (l.aux >> 28) + 0.5f) * 0.5f * (float) sumi;
    } else if constexpr (T == ET_IQ4NL) {
        const int qq[4] = { l.q.x, l.q.y, l.q.z, l.q.w };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int2 v = tab16(qq[k], kv4);
            sumi = dp4a(v.x, x[k], sumi);
            sumi = dp4a(v.y, x[k + 4], sumi);
        }
    } else {
        sumi = dp4a(l.a.x, x[0], 0); sumi = dp4a(l.a.y, x[1], sumi); sumi = dp4a(l.a.z, x[2], sumi); sumi = dp4a(l.a.w, x[3], sumi);
        sumi = dp4a(l.b.x, x[4], sumi); sumi = dp4a(l.b.y, x[5], sumi); sumi = dp4a(l.b.z, x[6], sumi); sumi = dp4a(l.b.w, x[7], sumi);
    }
    return __half2float(l.d) * (float) sumi;
}
__device__ __forceinline__ ExLd ex_load(int type, const uint8_t * rq, const __half * rd, const uint8_t * rs, int sb) {
    ExLd l;
    if (type == ET_IQ3XXS) {
        const int B = sb >> 3, ib = sb & 7;
        const uint2 g = __ldg((const uint2 *) (rq + B * 64 + ib * 8));
        l.a = make_int4((int) g.x, (int) g.y, (int) __ldg((const uint32_t *) (rs + B * 32 + ib * 4)), 0);
        l.d = __half2float(__ldg(rd + B));
    } else if (type == ET_IQ4NL) {
        l.a = __ldg((const int4 *) (rq + sb * 16));
        l.d = __half2float(__ldg(rd + sb));
    } else {
        l.a = __ldg((const int4 *) (rq + sb * 32));
        l.b = __ldg((const int4 *) (rq + sb * 32 + 16));
        l.d = __half2float(__ldg(rd + sb));
    }
    return l;
}
__device__ __forceinline__ float ex_dot(int type, const ExLd & l, const int8_t * xq, float xd, const uint32_t * grid, const int8_t * kv4) {
    const int * x = (const int *) xq;
    int sumi = 0;
    if (type == ET_IQ3XXS) {
        const uint32_t g2[2] = { (uint32_t) l.a.x, (uint32_t) l.a.y };
        const uint8_t * q3 = (const uint8_t *) g2;
        const uint32_t aux = (uint32_t) l.a.z;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int gx = grid[q3[l0]], gy = grid[q3[l0 + 1]];
            const uint32_t sg = unpack_ksigns((aux >> (7 * l0 / 2)) & 0x7f);
            const int s0 = __vcmpne4(sg & 0x08040201u, 0), s1 = __vcmpne4(sg & 0x80402010u, 0);
            sumi = dp4a(__vsub4(gx ^ s0, s0), x[l0], sumi);
            sumi = dp4a(__vsub4(gy ^ s1, s1), x[l0 + 1], sumi);
        }
        return l.d * ((float) (aux >> 28) + 0.5f) * 0.5f * xd * (float) sumi;
    } else if (type == ET_IQ4NL) {
        const int qq[4] = { l.a.x, l.a.y, l.a.z, l.a.w };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int2 v = tab16(qq[k], kv4);
            sumi = dp4a(v.x, x[k], sumi);
            sumi = dp4a(v.y, x[k + 4], sumi);
        }
    } else {
        sumi = dp4a(l.a.x, x[0], 0); sumi = dp4a(l.a.y, x[1], sumi); sumi = dp4a(l.a.z, x[2], sumi); sumi = dp4a(l.a.w, x[3], sumi);
        sumi = dp4a(l.b.x, x[4], sumi); sumi = dp4a(l.b.y, x[5], sumi); sumi = dp4a(l.b.z, x[6], sumi); sumi = dp4a(l.b.w, x[7], sumi);
    }
    return l.d * xd * (float) sumi;
}

// one 32-wide sub-block sb of expert row (already offset to the row) against xq/xd at block xb
__device__ __forceinline__ float ex_sub(const EX & E, const uint8_t * rq, const __half * rd, const uint8_t * rs, int sb,
                                        const int8_t * xq, float xd, const uint32_t * grid, const int8_t * kv4) {
    const int * x = (const int *) xq;
    if (E.type == ET_IQ3XXS) {
        const int B = sb >> 3, ib = sb & 7;
        const uint2 g = __ldg((const uint2 *) (rq + B * 64 + ib * 8));
        const uint32_t aux = __ldg((const uint32_t *) (rs + B * 32 + ib * 4));
        const uint8_t * q3 = (const uint8_t *) &g;
        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int gx = grid[q3[l0]], gy = grid[q3[l0 + 1]];
            const uint32_t sg = unpack_ksigns((aux >> (7 * l0 / 2)) & 0x7f);
            const int s0 = __vcmpne4(sg & 0x08040201u, 0), s1 = __vcmpne4(sg & 0x80402010u, 0);
            sumi = dp4a(__vsub4(gx ^ s0, s0), x[l0], sumi);
            sumi = dp4a(__vsub4(gy ^ s1, s1), x[l0 + 1], sumi);
        }
        const float db = __half2float(__ldg(rd + B)) * ((float) (aux >> 28) + 0.5f) * 0.5f;
        return db * xd * (float) sumi;
    } else if (E.type == ET_IQ4NL) {
        const int4 q = __ldg((const int4 *) (rq + sb * 16));
        const int qq[4] = { q.x, q.y, q.z, q.w };
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int2 v = tab16(qq[l], kv4);
            sumi = dp4a(v.x, x[l], sumi);
            sumi = dp4a(v.y, x[l + 4], sumi);
        }
        return __half2float(__ldg(rd + sb)) * xd * (float) sumi;
    } else {
        const int4 a = __ldg((const int4 *) (rq + sb * 32)), b = __ldg((const int4 *) (rq + sb * 32 + 16));
        int s = dp4a(a.x, x[0], 0); s = dp4a(a.y, x[1], s); s = dp4a(a.z, x[2], s); s = dp4a(a.w, x[3], s);
        s = dp4a(b.x, x[4], s); s = dp4a(b.y, x[5], s); s = dp4a(b.z, x[6], s); s = dp4a(b.w, x[7], s);
        return __half2float(__ldg(rd + sb)) * xd * (float) s;
    }
}

__device__ __forceinline__ void ex_row(const EX & E, int e, int row, const uint8_t *& rq, const __half *& rd, const uint8_t *& rs) {
    const size_t r = (size_t) row;
    if (E.type == ET_IQ3XXS) {
        rq = E.q + e * E.qe + r * (E.K / 4);
        rd = E.d + e * E.de + r * (E.K / 256);
        rs = E.s + e * E.se + r * (E.K / 8);
    } else if (E.type == ET_IQ4NL) {
        rq = E.q + e * E.qe + r * (E.K / 2);
        rd = E.d + e * E.de + r * (E.K / 32);
        rs = nullptr;
    } else {
        rq = E.q + e * E.qe + r * E.K;
        rd = E.d + e * E.de + r * (E.K / 32);
        rs = nullptr;
    }
}

// ---------------------------------------------------------------- phases
// the 16 split-K inject partials -> 4 combine weights; one load per block, smem broadcast (a grid-wide read
// of the same 64 bytes from every warp serialises on one L2 slice). Contains __syncthreads.
__device__ __forceinline__ void inj_weights(const float * inj, float * sm16, float (&w)[HC]) {
    if (threadIdx.x < HC * HC) sm16[threadIdx.x] = ldcg(inj + threadIdx.x);
    __syncthreads();
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) v += sm16[k * HC + s];
        w[s] = 2.0f * sigm(v * (1.0f / HC));
    }
}

// HC mix, first half. Makes the new residual (mode 0: as is, 1: + combine of the pending block output,
// 2: embedding x4), writes it to res[1-cur] when it changed, then lo = down(xn), inj = inject(xn).
// Returns the new cur.
__device__ __noinline__ int mix_pre(const Glob & G, const SM & S, int cur, int mode, int injp, const float * wn,
                       const Q8 & down, const float * winj, float * inj_out) {
    // block b works on hc stream k = b % HC only: the split-K slice of the down GEMV equals the stream, so the
    // block stages 2560 values, takes the stream rms locally and runs the rows of that slice.
    constexpr int U = NE / 32 / NWARP;  // 5 blocks of 32 per warp
    const int k = blockIdx.x % HC, kb = blockIdx.x / HC, nkb = (gridDim.x - k + HC - 1) / HC;  // blocks sharing stream k
    const int w = warp_id(), ln = lane_id();
    float x[U], wv[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int i = (w * U + u) * 32 + ln;
        wv[u] = __ldg(wn + k * NE + i);
        x[u] = mode == 2 ? ldcg(G.emb + i) : ldcg(G.res[cur] + k * NE + i);
    }
    if (mode == 1 && !(G.dbg & 4)) {
        float wi[HC];
        inj_weights(G.inj[injp], S.red + 32, wi);
        float bo[U];
#pragma unroll
        for (int u = 0; u < U; ++u) bo[u] = ldcg(G.blk_out + (w * U + u) * 32 + ln);
#pragma unroll
        for (int u = 0; u < U; ++u) x[u] += bo[u] * wi[k];
    }
    // ownership of the global writes among the nkb blocks of this stream: 32-block index % nkb
    const int ncur = mode != 0 ? 1 - cur : cur;
    float ss = 0.0f;
#pragma unroll
    for (int u = 0; u < U; ++u) {
        ss += x[u] * x[u];
        if (mode != 0 && !(G.dbg & 16) && (w * U + u) % nkb == kb) G.res[ncur][k * NE + (w * U + u) * 32 + ln] = x[u];
    }
    ss = wsum(ss);
    if (ln == 0) S.red[w] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int q = 0; q < NWARP; ++q) tot += S.red[q];
    const float inv = rsqrtf(tot / NE + EPS);
#pragma unroll
    for (int u = 0; u < U; ++u) {
        if (G.dbg & 8) break;
        const int i = (w * U + u) * 32 + ln;
        const float xn = x[u] * inv * wv[u];
        const float amax = wmax(fabsf(xn));
        const float id = amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f;
        S.xq[i] = (int8_t) __float2int_rn(xn * id);
        S.f[i] = xn;
        if (ln == 0) S.xd[w * U + u] = amax * (1.0f / 127.0f);
        if ((w * U + u) % nkb == kb) G.xn[k * NE + i] = xn;
    }
    __syncthreads();
    if (G.dbg & 2) return ncur;
    // rows of slice k: down rows (K = 2560 slice of each row) + the HC inject rows of this slice
    const int nw = nkb * NWARP, gw = w * nkb + kb;
    for (int r0 = gw * 2; r0 < down.rows; r0 += nw * 2) {
        const int rows[2] = { r0, min(r0 + 1, down.rows - 1) };
        constexpr int C = NE / 16 / 32;
        int4 wq[2][C]; __half wd[2][C];
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int u = 0; u < C; ++u) {
                const int c = ln + 32 * u;
                wq[r][u] = __ldg((const int4 *) (down.q + (size_t) rows[r] * HCD + k * NE + 16 * c));
                wd[r][u] = __ldg(down.d + (size_t) rows[r] * (HCD / 32) + k * (NE / 32) + (c >> 1));
            }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float acc = 0.0f;
#pragma unroll
            for (int u = 0; u < C; ++u) {
                const int c = ln + 32 * u;
                const int4 xv = *(const int4 *) (S.xq + 16 * c);
                int sm = dp4a(wq[r][u].x, xv.x, 0); sm = dp4a(wq[r][u].y, xv.y, sm); sm = dp4a(wq[r][u].z, xv.z, sm); sm = dp4a(wq[r][u].w, xv.w, sm);
                acc += (float) sm * __half2float(wd[r][u]) * S.xd[c >> 1];
            }
            acc = wsum(acc);
            if (ln == 0 && r0 + r < down.rows) G.lo[k * LR + r0 + r] = acc;
        }
    }
    if (winj) {
        // inject rows: the last warps of this stream's blocks
        for (int s = gw; s < HC; s += nw) {
            const float v = f32_dot_ne(winj + (size_t) s * HCD + k * NE, S.f);
            if (ln == 0) inj_out[k * HC + s] = v;
        }
    }
    return ncur;
}

// HC mix, second half: gate = up(silu(lo/hc)); mixed = mean_s(xn_s * sigmoid(gate_s))
__device__ __noinline__ void mix_post(const Glob & G, const SM & S, int cur, const float * wn, const Q8 & up) {
    float * lo = S.f2;
    for (int i = threadIdx.x; i < LR; i += NT) {
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) v += ldcg(G.lo + k * LR + i);
        lo[i] = silu(v * (1.0f / HC));
    }
    __syncthreads();
    quant_q8(lo, LR, S.xq, S.xd);
    __syncthreads();
    for (int i = gwarp(); i < NE; i += nwarps()) {
        const int rows[HC] = { i, NE + i, 2 * NE + i, 3 * NE + i };
        float g[HC];
        q8_dot_rows<LR, HC>(up, rows, S.xq, S.xd, g);
        float m = 0.0f;
#pragma unroll
        for (int s = 0; s < HC; ++s) m += ldcg(G.xn + s * NE + i) * sigm(g[s]);
        if (lane_id() == 0) G.mixed[i] = m * (1.0f / HC);
    }
}

// mixed -> smem float (S.f) + q8 (S.xq/S.xd)
__device__ void stage_mixed(const Glob & G, const SM & S) {
    stage_q8<NE>(G.mixed, S.f, S.xq, S.xd);
    __syncthreads();
}

// projections of a linear-attention layer: qkv | z | beta | alpha
__device__ __noinline__ void lin_proj(const Glob & G, const SM & S, const Layer & L) {
    stage_mixed(G, S);
    float * P = G.proj;
    int wo = 0;
    wo = gemv_q8<NE, 2>(L.qkv, CONVCH, wo, S.xq, S.xd, [&](int r, float v) { P[r] = v; });
    wo = gemv_q8<NE, 2>(L.z, VDIM, wo, S.xq, S.xd, [&](int r, float v) { P[CONVCH + r] = v; });
    wo = gemv_f32(L.beta, GV, NE, wo, S.f, [&](int r, float v) { P[CONVCH + VDIM + r] = v; });
    gemv_f32(L.alpha, GV, NE, wo, S.f, [&](int r, float v) { P[CONVCH + VDIM + GV + r] = v; });
}

// gated delta net for one value head per block
__device__ __noinline__ void lin_core(const Glob & G, const SM & S, const Layer & L, int n) {
    const int h = blockIdx.x;
    if (h >= GV) return;
    const int kh = h % GK;
    float * q = S.f2, * k = S.f2 + DS, * v = S.f2 + 2 * DS, * o = S.f2 + 3 * DS;
    const int slot = n & 3;
    // causal conv (kernel 4) + silu on the 384 channels this head reads
    for (int t = threadIdx.x; t < 3 * DS; t += NT) {
        const int part = t / DS, i = t % DS;
        const int ch = part == 0 ? kh * DS + i : part == 1 ? KDIM + kh * DS + i : 2 * KDIM + h * DS + i;
        const float xc = ldcg(G.proj + ch);
        float acc = L.conv[3 + 4 * ch] * xc;
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) {
            const int back = 3 - kk;  // tap kk reads t - back
            const float xp = n - back >= 0 ? ldcg(L.conv_ring + ((n - back) & 3) * CONVCH + ch) : 0.0f;
            acc += L.conv[kk + 4 * ch] * xp;
        }
        const float y = silu(acc);
        (part == 0 ? q : part == 1 ? k : v)[i] = y;
        const bool owner = part == 2 || h < GK;
        if (owner) L.conv_ring[slot * CONVCH + ch] = xc;
    }
    __syncthreads();
    // l2 norm of q and k: x / sqrt(sum x^2 + eps)
    {
        float sq = 0.0f, sk = 0.0f;
        for (int i = threadIdx.x; i < DS; i += NT) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
        sq = bsum(sq, S.red);
        sk = bsum(sk, S.red + 16 * 0 + 32);
        const float iq = rsqrtf(sq + EPS), ik = rsqrtf(sk + EPS);
        __syncthreads();
        for (int i = threadIdx.x; i < DS; i += NT) { q[i] *= iq; k[i] *= ik; }
    }
    __syncthreads();
    const float beta = sigm(ldcg(G.proj + CONVCH + VDIM + h));
    const float al = ldcg(G.proj + CONVCH + VDIM + GV + h) + L.dt[h];
    const float sp = al > 20.0f ? al : log1pf(__expf(al));
    const float decay = __expf(sp * L.a[h]);
    // state rows: 4 threads per value row j, 32 key dims each
    const int j = threadIdx.x >> 2, part = threadIdx.x & 3;
    float * st = L.ssm_state + (size_t) h * DS * DS + (size_t) j * DS + part * 32;
    float sreg[32];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const float4 t4 = __ldcg((const float4 *) (st + 4 * u));
        sreg[4 * u] = t4.x; sreg[4 * u + 1] = t4.y; sreg[4 * u + 2] = t4.z; sreg[4 * u + 3] = t4.w;
    }
    float kv = 0.0f;
#pragma unroll
    for (int u = 0; u < 32; ++u) { sreg[u] *= decay; kv += sreg[u] * k[part * 32 + u]; }
    kv += __shfl_xor_sync(0xffffffffu, kv, 1);
    kv += __shfl_xor_sync(0xffffffffu, kv, 2);
    const float delta = (v[j] - kv) * beta;
    float oj = 0.0f;
#pragma unroll
    for (int u = 0; u < 32; ++u) { sreg[u] += k[part * 32 + u] * delta; oj += sreg[u] * q[part * 32 + u]; }
    oj += __shfl_xor_sync(0xffffffffu, oj, 1);
    oj += __shfl_xor_sync(0xffffffffu, oj, 2);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        __stcg((float4 *) (st + 4 * u), make_float4(sreg[4 * u], sreg[4 * u + 1], sreg[4 * u + 2], sreg[4 * u + 3]));
    }
    oj *= rsqrtf((float) DS);
    if (part == 0) o[j] = oj;
    __syncthreads();
    // gated rms norm over the head, sigmoid gate z
    float ss = 0.0f;
    for (int i = threadIdx.x; i < DS; i += NT) ss += o[i] * o[i];
    ss = bsum(ss, S.red);
    const float inv = rsqrtf(ss / DS + EPS);
    for (int i = threadIdx.x; i < DS; i += NT) {
        const float zz = ldcg(G.proj + CONVCH + h * DS + i);
        G.core[h * DS + i] = o[i] * inv * L.snorm[i] * sigm(zz);
    }
}

// projections of a full-attention layer: q|gate (12288), k (512), v (512)
__device__ __noinline__ void att_proj(const Glob & G, const SM & S, const Layer & L) {
    stage_mixed(G, S);
    float * P = G.proj;
    constexpr int n0 = NH * HD * 2, n1 = n0 + NKV * HD;
    int wo = 0;
    wo = gemv_q8<NE, 2>(L.wq, n0, wo, S.xq, S.xd, [&](int r, float v) { P[r] = v; });
    wo = gemv_q8<NE, 2>(L.wk, NKV * HD, wo, S.xq, S.xd, [&](int r, float v) { P[n0 + r] = v; });
    gemv_q8<NE, 2>(L.wv, NKV * HD, wo, S.xq, S.xd, [&](int r, float v) { P[n1 + r] = v; });
}

// rms norm (head) + neox rope on the first NROT dims, 256 values in smem, called by a whole block
__device__ void norm_rope(float * x, const float * w, int pos, float * red) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HD; i += NT) ss += x[i] * x[i];
    ss = bsum(ss, red);
    const float inv = rsqrtf(ss / HD + EPS);
    __syncthreads();
    for (int i = threadIdx.x; i < HD; i += NT) x[i] = x[i] * inv * w[i];
    __syncthreads();
    if (threadIdx.x < NROT / 2) {
        const int i = threadIdx.x;
        const float theta = (float) pos * powf(ROPE_BASE, -2.0f * i / NROT);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const float x0 = x[i], x1 = x[i + NROT / 2];
        x[i] = x0 * cs - x1 * sn;
        x[i + NROT / 2] = x0 * sn + x1 * cs;
    }
    __syncthreads();
}

// one query head per block, dense causal attention over positions 0..n
__device__ __noinline__ void att_core(const Glob & G, const SM & S, const Layer & L, int n) {
    const int h = blockIdx.x;
    if (h >= NH) return;
    const int g = h / (NH / NKV);
    float * q = S.f2, * kx = S.f2 + HD, * vx = S.f2 + 2 * HD;
    float * mrg = S.f;  // NWARP x (HD + 2)
    for (int i = threadIdx.x; i < HD; i += NT) {
        q[i]  = ldcg(G.proj + h * 2 * HD + i);
        kx[i] = ldcg(G.proj + NH * HD * 2 + g * HD + i);
        vx[i] = ldcg(G.proj + NH * HD * 2 + NKV * HD + g * HD + i);
    }
    __syncthreads();
    norm_rope(q, L.qn, n, S.red);
    norm_rope(kx, L.kn, n, S.red);
    if (h % (NH / NKV) == 0) {
        for (int i = threadIdx.x; i < HD; i += NT) {
            L.kc[((size_t) n * NKV + g) * HD + i] = __float2half(kx[i]);
            L.vc[((size_t) n * NKV + g) * HD + i] = __float2half(vx[i]);
        }
    }
    const float scale = rsqrtf((float) HD);
    float qr[8], acc[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) { qr[u] = q[lane_id() * 8 + u] * scale; acc[u] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int p = warp_id(); p <= n; p += NWARP) {
        float kk[8], vv[8];
        if (p == n) {
#pragma unroll
            for (int u = 0; u < 8; ++u) { kk[u] = kx[lane_id() * 8 + u]; vv[u] = vx[lane_id() * 8 + u]; }
        } else {
            const uint4 kr = __ldcg((const uint4 *) (L.kc + ((size_t) p * NKV + g) * HD + lane_id() * 8));
            const uint4 vr = __ldcg((const uint4 *) (L.vc + ((size_t) p * NKV + g) * HD + lane_id() * 8));
            const __half2 * k2 = (const __half2 *) &kr, * v2 = (const __half2 *) &vr;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 a = __half22float2(k2[u]), b = __half22float2(v2[u]);
                kk[2 * u] = a.x; kk[2 * u + 1] = a.y; vv[2 * u] = b.x; vv[2 * u + 1] = b.y;
            }
        }
        float s = 0.0f;
#pragma unroll
        for (int u = 0; u < 8; ++u) s += qr[u] * kk[u];
        s = wsum(s);
        const float mn = fmaxf(m, s);
        const float c = __expf(m - mn), e = __expf(s - mn);
        l = l * c + e;
#pragma unroll
        for (int u = 0; u < 8; ++u) acc[u] = acc[u] * c + e * vv[u];
        m = mn;
    }
    float * my = mrg + warp_id() * (HD + 2);
#pragma unroll
    for (int u = 0; u < 8; ++u) my[lane_id() * 8 + u] = acc[u];
    if (lane_id() == 0) { my[HD] = m; my[HD + 1] = l; }
    __syncthreads();
    for (int i = threadIdx.x; i < HD; i += NT) {
        float M = -INFINITY;
        for (int w = 0; w < NWARP; ++w) M = fmaxf(M, mrg[w * (HD + 2) + HD]);
        float num = 0.0f, den = 0.0f;
        for (int w = 0; w < NWARP; ++w) {
            const float mw = mrg[w * (HD + 2) + HD];
            if (mw == -INFINITY) continue;
            const float c = __expf(mw - M);
            num += c * mrg[w * (HD + 2) + i];
            den += c * mrg[w * (HD + 2) + HD + 1];
        }
        const float gate = ldcg(G.proj + h * 2 * HD + HD + i);
        G.core[h * HD + i] = num / den * sigm(gate);
    }
}

// mixer output projection (K = 6144) -> blk_out
__device__ __noinline__ void mix_out(const Glob & G, const SM & S, const Q8 & W) {
    stage_q8<VDIM>(G.core, nullptr, S.xq, S.xd);
    __syncthreads();
    float * O = G.blk_out;
    gemv_q8<VDIM, 1>(W, NE, 0, S.xq, S.xd, [&](int r, float v) { O[r] = v; });
}

// router logits, shared expert gate, shared expert gate/up
__device__ __noinline__ void ffn_router(const Glob & G, const SM & S, const Layer & L) {
    stage_mixed(G, S);
    float * RL = G.rlog, * SH = G.sh_h;
    int wo = 0;
    wo = gemv_f32(L.router, NEXP, NE, wo, S.f, [&](int r, float v) { RL[r] = v; });
    wo = gemv_f32(L.sh_gate_inp, 1, NE, wo, S.f, [&](int, float v) { RL[NEXP] = v; });
    wo = gemv_q8<NE, 2>(L.sh_gate, FF, wo, S.xq, S.xd, [&](int r, float v) { SH[r] = v; });
    gemv_q8<NE, 2>(L.sh_up, FF, wo, S.xq, S.xd, [&](int r, float v) { SH[FF + r] = v; });
}

template <int T>
__device__ __forceinline__ void gu_body(const Glob & G, const SM & S, const Layer & L, const int * sid) {
    constexpr int NSB = NE / 32, J = (NSB + 31) / 32, R = 2;
    for (int task = gwarp(); task < NUSED * FF / R; task += nwarps()) {
        const int e = task / (FF / R), r0 = (task % (FF / R)) * R;
        ExL<T> lg[R][J], lu[R][J];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *gq, *gs, *uq, *us; const __half *gd, *ud;
            ex_row(L.eg, sid[e], r0 + r, gq, gd, gs);
            ex_row(L.eu, sid[e], r0 + r, uq, ud, us);
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const int sb = lane_id() + 32 * j;
                if (sb < NSB) { lg[r][j] = exl_load<T>(gq, gd, gs, sb); lu[r][j] = exl_load<T>(uq, ud, us, sb); }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float ga = 0.0f, ua = 0.0f;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const int sb = lane_id() + 32 * j;
                if (sb < NSB) {
                    const float xd = S.xd[sb];
                    ga += xd * exl_dot<T>(lg[r][j], S.xq + sb * 32, S.grid, S.kv4, S.sg64);
                    ua += xd * exl_dot<T>(lu[r][j], S.xq + sb * 32, S.grid, S.kv4, S.sg64);
                }
            }
            ga = wsum(ga); ua = wsum(ua);
            if (lane_id() == 0) G.exp_h[e * FF + r0 + r] = silu(ga) * ua;
        }
    }
}

// top-k (block 0 publishes), expert gate/up
__device__ __noinline__ void ffn_experts_gu(const Glob & G, const SM & S, const Layer & L) {
    float * lg = S.f2;
    int * sid = (int *) (S.f2 + NEXP);
    float * sw = S.f2 + NEXP + NUSED;
    load_vec(lg, G.rlog, NEXP);
    __syncthreads();
    if (warp_id() == 0) {
        float vals[NEXP / 32];
#pragma unroll
        for (int u = 0; u < NEXP / 32; ++u) vals[u] = lg[u * 32 + lane_id()];
        float top[NUSED];
        for (int k = 0; k < NUSED; ++k) {
            float bv = -INFINITY; int bi = 0x7fffffff;
#pragma unroll
            for (int u = 0; u < NEXP / 32; ++u) if (vals[u] > bv) { bv = vals[u]; bi = u * 32 + lane_id(); }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            top[k] = bv;
            if (lane_id() == 0) sid[k] = bi;
            if ((bi & 31) == lane_id()) vals[bi >> 5] = -INFINITY;
        }
        if (lane_id() == 0) {
            float sum = 0.0f;
            for (int k = 0; k < NUSED; ++k) { sw[k] = __expf(top[k] - top[0]); sum += sw[k]; }
            for (int k = 0; k < NUSED; ++k) sw[k] /= sum;
            if (blockIdx.x == 0) {
                for (int k = 0; k < NUSED; ++k) { G.sel_id[k] = sid[k]; G.sel_w[k] = sw[k]; }
                G.sel_w[NUSED] = sigm(ldcg(G.rlog + NEXP));
            }
        }
    }
    for (int r = blockIdx.x * NT + threadIdx.x; r < FF; r += gridDim.x * NT)
        G.exp_h[NUSED * FF + r] = silu(ldcg(G.sh_h + r)) * ldcg(G.sh_h + FF + r);
    stage_mixed(G, S);
    if (G.dbg & 2) return;
    if (L.eg.type == ET_IQ3XXS) gu_body<ET_IQ3XXS>(G, S, L, sid);
    else                        gu_body<ET_IQ4NL>(G, S, L, sid);
}

template <int T>
__device__ __forceinline__ void down_body(const Glob & G, const SM & S, const Layer & L, const int * sid, const float * sw) {
    constexpr int NB = FF / 32, NE_T = NUSED * NB, JE = (NE_T + 31) / 32;   // routed tasks: 200 -> 7 per lane
    constexpr int NS_T = NB;                                               // shared tasks: 20, lanes 0..19
    for (int row = gwarp(); row < NE; row += nwarps()) {
        ExL<T> le[JE];
        ExL<ET_Q8> ls;
#pragma unroll
        for (int j = 0; j < JE; ++j) {
            const int t = lane_id() + 32 * j;
            if (t < NE_T) {
                const int e = t / NB, b = t % NB;
                const uint8_t *rq, *rs; const __half * rd;
                ex_row(L.ed, sid[e], row, rq, rd, rs);
                le[j] = exl_load<T>(rq, rd, rs, b);
            }
        }
        if (lane_id() < NS_T)
            ls = exl_load<ET_Q8>((const uint8_t *) (L.sh_down.q + (size_t) row * FF), L.sh_down.d + (size_t) row * NB, nullptr, lane_id());
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < JE; ++j) {
            const int t = lane_id() + 32 * j;
            if (t < NE_T) acc += sw[t / NB] * S.xd[t] * exl_dot<T>(le[j], S.xq + t * 32, S.grid, S.kv4);
        }
        if (lane_id() < NS_T) {
            const int t = NE_T + lane_id();
            acc += sw[NUSED] * S.xd[t] * exl_dot<ET_Q8>(ls, S.xq + t * 32, S.grid, S.kv4);
        }
        acc = wsum(acc);
        if (lane_id() == 0) G.blk_out[row] = acc;
    }
}

// expert down + shared down, weighted -> blk_out
__device__ __noinline__ void ffn_down(const Glob & G, const SM & S, const Layer & L) {
    int * sid = (int *) S.f2;
    float * sw = S.f2 + 16;
    if (threadIdx.x < NUSED) { sid[threadIdx.x] = __ldcg(G.sel_id + threadIdx.x); sw[threadIdx.x] = ldcg(G.sel_w + threadIdx.x); }
    if (threadIdx.x == NUSED) sw[NUSED] = ldcg(G.sel_w + NUSED);
    stage_q8<(NUSED + 1) * FF>(G.exp_h, nullptr, S.xq, S.xd);
    __syncthreads();
    if (G.dbg & 2) return;
    if (L.ed.type == ET_IQ4NL) down_body<ET_IQ4NL>(G, S, L, sid, sw);
    else                       down_body<ET_Q8>(G, S, L, sid, sw);
}

// PLE, phase A: key (HCD) and value (NE) projections of the gathered n-gram rows
__device__ __noinline__ void ple_proj(const Glob & G, const SM & S, const Layer & L) {
    stage_q8<NE>(G.ple_in, nullptr, S.xq, S.xd);
    __syncthreads();
    float * KV = G.ple_kv;
    const int wo = gemv_q8<NE, 2>(L.ple_k, HCD, 0, S.xq, S.xd, [&](int r, float v) { KV[r] = v; });
    gemv_q8<NE, 2>(L.ple_v, NE, wo, S.xq, S.xd, [&](int r, float v) { KV[HCD + r] = v; });
}

// PLE, phase B: gated value + dilated depthwise conv, added to the residual. Returns the new cur.
__device__ __noinline__ int ple_apply(const Glob & G, const SM & S, const Layer & L, int cur, int n) {
    float * r = S.f;
    load_vec(r, G.res[cur], HCD);
    __syncthreads();
    float gate[HC], ginv[HC];
    float vv = 0.0f;
    for (int i = threadIdx.x; i < NE; i += NT) { const float v = ldcg(G.ple_kv + HCD + i); vv += v * v; }
    vv = bsum(vv, S.red);
    for (int s = 0; s < HC; ++s) {
        float kk = 0.0f, qq = 0.0f;
        for (int i = threadIdx.x; i < NE; i += NT) {
            const float kv = ldcg(G.ple_kv + s * NE + i), rv = r[s * NE + i];
            kk += kv * kv; qq += rv * rv;
        }
        kk = bsum(kk, S.red);
        qq = bsum(qq, S.red);
        const float ik = rsqrtf(kk / NE + EPS), iq = rsqrtf(qq / NE + EPS);
        float dt = 0.0f;
        for (int i = threadIdx.x; i < NE; i += NT) {
            dt += ldcg(G.ple_kv + s * NE + i) * ik * L.ple_nk[s * NE + i] * r[s * NE + i] * iq * L.ple_nq[s * NE + i];
        }
        dt = bsum(dt, S.red) * rsqrtf((float) NE);
        const float mag = sqrtf(fmaxf(fabsf(dt), 1e-6f));
        gate[s] = sigm(dt > 0.0f ? mag : (dt < 0.0f ? -mag : 0.0f));
        // rms of gated_s = |gate| * rms(value)
        ginv[s] = rsqrtf(gate[s] * gate[s] * vv / NE + EPS);
    }
    const int ncur = 1 - cur;
    const int per = (HCD + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * per, hi = min(HCD, lo + per);
    const int slot = n % PLE_HIST;
    for (int c = lo + threadIdx.x; c < hi; c += NT) {
        const int s = c / NE, i = c % NE;
        const float gated = ldcg(G.ple_kv + HCD + i) * gate[s];
        const float nrm = gated * ginv[s] * L.ple_nc[c];
        float acc = L.ple_conv[3 + 4 * c] * nrm;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int back = (3 - k) * 3;
            if (n - back >= 0) acc += L.ple_conv[k + 4 * c] * ldcg(G.ple_hist + ((n - back) % PLE_HIST) * HCD + c);
        }
        G.ple_hist[slot * HCD + c] = nrm;
        G.res[ncur][c] = r[c] + gated + silu(acc);
    }
    return ncur;
}

// lm head + argmax
__device__ __noinline__ void head_out(const Glob & G, const SM & S) {
    stage_mixed(G, S);
    float bv = -INFINITY; int bi = 0;
    for (int r0 = gwarp() * 2; r0 < VOCAB; r0 += nwarps() * 2) {
        const int rows[2] = { r0, min(r0 + 1, VOCAB - 1) };
        float v[2];
        q8_dot_rows<NE, 2>(G.out, rows, S.xq, S.xd, v);
        if (v[0] > bv) { bv = v[0]; bi = r0; }
        if (r0 + 1 < VOCAB && v[1] > bv) { bv = v[1]; bi = r0 + 1; }
    }
    if (lane_id() == 0 && bi >= 0) {
        unsigned u = __float_as_uint(bv);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        const unsigned long long key = ((unsigned long long) u << 32) | (unsigned) (0xffffffffu - (unsigned) bi);
        atomicMax(G.best, key);
    }
}

__global__ void __launch_bounds__(NT, 1) fnx_step(Glob G, int n) {
    const SM S = smem_layout();
    int pk = 0;
    for (int i = threadIdx.x; i < 256; i += NT) S.grid[i] = iq3xxs_grid[i];
    if (threadIdx.x < 16) S.kv4[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
    if (threadIdx.x < 128) S.sg64[threadIdx.x] = ksigns64[threadIdx.x];
    __syncthreads();
    if (n <= -1000000) {
        // phase micro-benchmark: -1000000 - which*1000 - iters
        const int which = (-n - 1000000) / 1000, iters = (-n - 1000000) % 1000;
        const Layer & L = G.L[0];
        int c = 0, ip = 0;
        for (int it = 0; it < iters; ++it) {
            switch (which) {
                case 0: c = mix_pre(G, S, c, 1, ip, L.hca_norm, L.hca_down, L.hca_inj, G.inj[ip ^ 1]); break;
                case 1: mix_post(G, S, c, L.hca_norm, L.hca_up); break;
                case 2: lin_proj(G, S, L); break;
                case 3: lin_core(G, S, L, 5); break;
                case 4: mix_out(G, S, L.ssm_out); break;
                case 5: ffn_router(G, S, L); break;
                case 6: ffn_experts_gu(G, S, L); break;
                case 7: ffn_down(G, S, L); break;
                case 8: head_out(G, S); break;
                default: break;
            }
            gsync(G, pk, 0);
        }
        return;
    }
    if (n < 0) {
        int q = 0;
        for (int k = 0; k < -n; ++k) gsync(G, q, 0);
        return;
    }
    if (G.prof && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); G.prof[0] = 0; G.prof[1] = t; }
    int cur = 0, injp = 0;
    int mode = 2;  // first mix builds the residual from the embedding
    for (int il = 0; il < NL; ++il) {
        const Layer & L = G.L[il];
        if (il == PLE_LAYER) {
            ple_proj(G, S, L);
            gsync(G, pk, __LINE__);
            // the pending ffn combine of layer il-1 goes first: do it as a mix_pre-free residual update
            {
                float * r = S.f;
                load_vec(r, G.res[cur], HCD);
                __syncthreads();
                float w[HC];
                inj_weights(G.inj[injp], S.red + 32, w);
                const int per = (HCD + gridDim.x - 1) / gridDim.x;
                const int lo = blockIdx.x * per, hi = min(HCD, lo + per);
                for (int i = lo + threadIdx.x; i < hi; i += NT) G.res[1 - cur][i] = r[i] + ldcg(G.blk_out + (i % NE)) * w[i / NE];
                cur = 1 - cur;
            }
            gsync(G, pk, __LINE__);
            cur = ple_apply(G, S, L, cur, n);
            gsync(G, pk, __LINE__);
            mode = 0;
        }
        // attention-side HC mix
        cur = mix_pre(G, S, cur, mode, injp, L.hca_norm, L.hca_down, L.hca_inj, G.inj[injp ^ 1]);
        injp ^= 1;
        gsync(G, pk, __LINE__);
        mix_post(G, S, cur, L.hca_norm, L.hca_up);
        gsync(G, pk, __LINE__);
        if (L.recr) {
            lin_proj(G, S, L);
            gsync(G, pk, __LINE__);
            lin_core(G, S, L, n);
            gsync(G, pk, __LINE__);
            mix_out(G, S, L.ssm_out);
        } else {
            att_proj(G, S, L);
            gsync(G, pk, __LINE__);
            att_core(G, S, L, n);
            gsync(G, pk, __LINE__);
            mix_out(G, S, L.wo);
        }
        gsync(G, pk, __LINE__);
        // ffn-side HC mix (combines the token mixer output first)
        cur = mix_pre(G, S, cur, 1, injp, L.hcf_norm, L.hcf_down, L.hcf_inj, G.inj[injp ^ 1]);
        injp ^= 1;
        gsync(G, pk, __LINE__);
        mix_post(G, S, cur, L.hcf_norm, L.hcf_up);
        gsync(G, pk, __LINE__);
        ffn_router(G, S, L);
        gsync(G, pk, __LINE__);
        ffn_experts_gu(G, S, L);
        gsync(G, pk, __LINE__);
        ffn_down(G, S, L);
        gsync(G, pk, __LINE__);
        mode = 1;
    }
    cur = mix_pre(G, S, cur, 1, injp, G.head_norm, G.head_down, nullptr, nullptr);
    gsync(G, pk, __LINE__);
    mix_post(G, S, cur, G.head_norm, G.head_up);
    gsync(G, pk, __LINE__);
    head_out(G, S);
    gsync(G, pk, __LINE__);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned long long k = *(volatile unsigned long long *) G.best;
        *G.out_tok = (int) (0xffffffffu - (unsigned) (k & 0xffffffffu));
    }
}

// ================================================================ host: GGUF
struct GT { std::string name; int type; int64_t ne[4]; int nd; const uint8_t * data; size_t nbytes; };

struct GGUF {
    std::map<std::string, GT> t;
    std::vector<std::pair<void *, size_t>> maps;

    static uint64_t rd64(const uint8_t *& p) { uint64_t v; memcpy(&v, p, 8); p += 8; return v; }
    static uint32_t rd32(const uint8_t *& p) { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
    static std::string rds(const uint8_t *& p) { uint64_t n = rd64(p); std::string s((const char *) p, n); p += n; return s; }
    static void skipv(const uint8_t *& p, uint32_t ty) {
        static const int sz[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
        if (ty == 8) { rds(p); return; }
        if (ty == 9) { uint32_t at = rd32(p); uint64_t n = rd64(p); for (uint64_t i = 0; i < n; ++i) skipv(p, at); return; }
        p += sz[ty];
    }
    void open(const std::string & path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { perror(path.c_str()); exit(1); }
        struct stat st; fstat(fd, &st);
        void * m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { perror("mmap"); exit(1); }
        ::close(fd);
        maps.push_back({ m, (size_t) st.st_size });
        const uint8_t * p = (const uint8_t *) m;
        if (memcmp(p, "GGUF", 4)) { fprintf(stderr, "%s: not gguf\n", path.c_str()); exit(1); }
        p += 4; rd32(p);
        const uint64_t nt = rd64(p), nkv = rd64(p);
        uint32_t align = 32;
        for (uint64_t i = 0; i < nkv; ++i) {
            std::string k = rds(p); uint32_t ty = rd32(p);
            if (k == "general.alignment") { align = rd32(p); continue; }
            skipv(p, ty);
        }
        std::vector<GT> loc;
        std::vector<uint64_t> offs;
        for (uint64_t i = 0; i < nt; ++i) {
            GT g; g.name = rds(p); g.nd = rd32(p);
            for (int d = 0; d < 4; ++d) g.ne[d] = 1;
            for (int d = 0; d < g.nd; ++d) g.ne[d] = (int64_t) rd64(p);
            g.type = rd32(p);
            offs.push_back(rd64(p));
            loc.push_back(g);
        }
        size_t base = (size_t) (p - (const uint8_t *) m);
        base = (base + align - 1) / align * align;
        for (size_t i = 0; i < loc.size(); ++i) {
            GT g = loc[i];
            g.data = (const uint8_t *) m + base + offs[i];
            const int64_t n = g.ne[0] * g.ne[1] * g.ne[2] * g.ne[3];
            g.nbytes = ggml_row_size((ggml_type) g.type, g.ne[0]) * (n / g.ne[0]);
            t[g.name] = g;
        }
    }
    const GT & get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) { fprintf(stderr, "missing tensor %s\n", n.c_str()); exit(1); }
        return it->second;
    }
    bool has(const std::string & n) const { return t.count(n) != 0; }
};

// ================================================================ host: upload
static size_t g_dev_bytes = 0;
static FILE * g_cw = nullptr, * g_cr = nullptr;  // repacked-weight cache: every dup() in load order
template <class T> static T * dalloc(size_t n) { void * p; CK(cudaMalloc(&p, n * sizeof(T))); g_dev_bytes += n * sizeof(T); return (T *) p; }
template <class T> static T * dup(const T * h, size_t n) {
    T * d = dalloc<T>(n);
    CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    if (g_cw && fwrite(h, sizeof(T), n, g_cw) != n) { fprintf(stderr, "cache write failed\n"); exit(1); }
    return d;
}
// cache read: the next n elements of the blob straight to a new device buffer
template <class T> static T * cread(size_t n) {
    static std::vector<uint8_t> stage;
    T * d = dalloc<T>(n);
    size_t left = n * sizeof(T), off = 0;
    const size_t CH = 256u << 20;
    if (stage.size() < CH) stage.resize(CH);
    while (left) {
        const size_t k = std::min(left, CH);
        if (fread(stage.data(), 1, k, g_cr) != k) { fprintf(stderr, "cache read failed\n"); exit(1); }
        CK(cudaMemcpy((uint8_t *) d + off, stage.data(), k, cudaMemcpyHostToDevice));
        left -= k; off += k;
    }
    return d;
}

static void to_f32(const GT & g, float * out, int64_t n) {
    if (g.type == GGML_TYPE_F32) { memcpy(out, g.data, n * 4); return; }
    const auto * tr = ggml_get_type_traits((ggml_type) g.type);
    if (!tr->to_float) { fprintf(stderr, "no to_float for %s type %d\n", g.name.c_str(), g.type); exit(1); }
    tr->to_float(g.data, out, n);
}

static const float * up_f32(const GGUF & M, const std::string & n) {
    const GT & g = M.get(n);
    const int64_t ne = g.ne[0] * g.ne[1] * g.ne[2] * g.ne[3];
    if (g_cr) return cread<float>(ne);
    std::vector<float> h(ne);
    to_f32(g, h.data(), ne);
    return dup(h.data(), ne);
}

static void quant_rows_q8(const float * x, int64_t rows, int K, int8_t * q, __half * d) {
    for (int64_t r = 0; r < rows; ++r) {
        for (int b = 0; b < K / 32; ++b) {
            const float * v = x + r * K + b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, fabsf(v[i]));
            const float dd = amax / 127.0f, id = dd > 0 ? 1.0f / dd : 0.0f;
            const __half dh = __float2half(dd);
            d[r * (K / 32) + b] = dh;
            for (int i = 0; i < 32; ++i) q[r * K + b * 32 + i] = (int8_t) lrintf(v[i] * id);
        }
    }
}

static Q8 up_q8(const GGUF & M, const std::string & n) {
    const GT & g = M.get(n);
    const int K = (int) g.ne[0];
    const int64_t rows = g.ne[1] * g.ne[2];
    if (g_cr) {
        Q8 m;
        m.q = cread<int8_t>((size_t) rows * K);
        m.d = cread<__half>((size_t) rows * K / 32);
        m.rows = (int) rows; m.K = K;
        return m;
    }
    std::vector<int8_t> q((size_t) rows * K);
    std::vector<__half> d((size_t) rows * K / 32);
    if (g.type == GGML_TYPE_Q8_0) {
        const uint8_t * p = g.data;
        for (int64_t i = 0; i < rows * K / 32; ++i, p += 34) {
            memcpy(&d[i], p, 2);
            memcpy(&q[i * 32], p + 2, 32);
        }
    } else {
        std::vector<float> f((size_t) rows * K);
        to_f32(g, f.data(), rows * K);
        quant_rows_q8(f.data(), rows, K, q.data(), d.data());
    }
    Q8 m;
    m.q = dup(q.data(), q.size());
    m.d = dup(d.data(), d.size());
    m.rows = (int) rows; m.K = K;
    return m;
}

// experts tensor [K, rows, NEXP]
static EX up_ex(const GGUF & M, const std::string & n) {
    const GT & g = M.get(n);
    const int K = (int) g.ne[0], rows = (int) g.ne[1], ne = (int) g.ne[2];
    const size_t nrow = (size_t) rows * ne;
    EX E{};
    E.K = K; E.rows = rows;
    if (g_cr) {
        if (g.type == GGML_TYPE_IQ3_XXS) {
            const int nb = K / 256;
            E.type = ET_IQ3XXS;
            E.q = cread<uint8_t>(nrow * nb * 64); E.s = cread<uint8_t>(nrow * nb * 32); E.d = cread<__half>(nrow * nb);
            E.qe = (size_t) rows * K / 4; E.se = (size_t) rows * K / 8; E.de = (size_t) rows * nb;
        } else if (g.type == GGML_TYPE_IQ4_NL || g.type == GGML_TYPE_IQ4_XS) {
            const size_t nblk = nrow * (K / 32);
            E.type = ET_IQ4NL;
            E.q = cread<uint8_t>(nblk * 16); E.d = cread<__half>(nblk);
            E.qe = (size_t) rows * K / 2; E.de = (size_t) rows * K / 32;
        } else {
            const size_t nblk = nrow * (K / 32);
            E.type = ET_Q8;
            E.q = cread<uint8_t>(nblk * 32); E.d = cread<__half>(nblk);
            E.qe = (size_t) rows * K; E.de = (size_t) rows * K / 32;
        }
        return E;
    }
    if (g.type == GGML_TYPE_IQ3_XXS) {
        const int nb = K / 256;
        std::vector<uint8_t> q(nrow * nb * 64), s(nrow * nb * 32);
        std::vector<__half> d(nrow * nb);
        const uint8_t * p = g.data;
        for (size_t i = 0; i < nrow * nb; ++i, p += 98) {
            memcpy(&d[i], p, 2);
            memcpy(&q[i * 64], p + 2, 64);
            memcpy(&s[i * 32], p + 66, 32);
        }
        E.type = ET_IQ3XXS;
        E.q = dup(q.data(), q.size()); E.s = dup(s.data(), s.size()); E.d = dup(d.data(), d.size());
        E.qe = (size_t) rows * K / 4; E.se = (size_t) rows * K / 8; E.de = (size_t) rows * nb;
    } else if (g.type == GGML_TYPE_IQ4_NL || g.type == GGML_TYPE_IQ4_XS) {
        const size_t nblk = nrow * (K / 32);
        std::vector<uint8_t> q(nblk * 16);
        std::vector<__half> d(nblk);
        if (g.type == GGML_TYPE_IQ4_NL) {
            const uint8_t * p = g.data;
            for (size_t i = 0; i < nblk; ++i, p += 18) { memcpy(&d[i], p, 2); memcpy(&q[i * 16], p + 2, 16); }
        } else {
            const uint8_t * p = g.data;
            for (size_t i = 0; i < nrow * (K / 256); ++i, p += 136) {
                __half dh; memcpy(&dh, p, 2);
                uint16_t sh; memcpy(&sh, p + 2, 2);
                const uint8_t * sl = p + 4, * qs = p + 8;
                const float dsup = __half2float(dh);
                for (int ib = 0; ib < 8; ++ib) {
                    const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
                    d[i * 8 + ib] = __float2half(dsup * (float) (ls - 32));
                    memcpy(&q[(i * 8 + ib) * 16], qs + 16 * ib, 16);
                }
            }
        }
        E.type = ET_IQ4NL;
        E.q = dup(q.data(), q.size()); E.d = dup(d.data(), d.size()); E.s = nullptr;
        E.qe = (size_t) rows * K / 2; E.de = (size_t) rows * K / 32; E.se = 0;
    } else if (g.type == GGML_TYPE_Q8_0) {
        const size_t nblk = nrow * (K / 32);
        std::vector<uint8_t> q(nblk * 32);
        std::vector<__half> d(nblk);
        const uint8_t * p = g.data;
        for (size_t i = 0; i < nblk; ++i, p += 34) { memcpy(&d[i], p, 2); memcpy(&q[i * 32], p + 2, 32); }
        E.type = ET_Q8;
        E.q = dup(q.data(), q.size()); E.d = dup(d.data(), d.size()); E.s = nullptr;
        E.qe = (size_t) rows * K; E.de = (size_t) rows * K / 32; E.se = 0;
    } else {
        fprintf(stderr, "%s: unsupported expert type %d\n", n.c_str(), g.type); exit(1);
    }
    return E;
}

// ================================================================ host: engine
struct Engine {
    GGUF M;
    Glob G{};
    std::vector<Layer> hL;
    int nsm = 0;
    float * h_emb = nullptr;   // pinned NE
    float * h_ple = nullptr;   // pinned NE
    float * d_emb = nullptr, * d_ple = nullptr;
    int * h_tok = nullptr;
    unsigned long long * h_prof = nullptr;
    std::map<int, double> prof_ns; std::map<int, int> prof_cnt; int prof_steps = 0;
    std::vector<int> seq;      // tokens fed so far
    double host_ple_s = 0.0;
    const GT * tok_embd = nullptr, * ple = nullptr;
    const uint8_t * ple_data = nullptr;  // the PLE table: in the GGUF mmap, or a local raw copy (--ple-file)
    std::vector<uint64_t> ple_mul, ple_off, ple_voc;
    std::vector<std::pair<void *, size_t>> state_bufs;

    void load(const std::vector<std::string> & shards, const std::string & cache, const std::string & ple_file) {
        for (auto & s : shards) M.open(s);
        if (!cache.empty()) {
            g_cr = fopen(cache.c_str(), "rb");
            if (!g_cr) { g_cw = fopen((cache + ".tmp").c_str(), "wb"); if (!g_cw) { perror(cache.c_str()); exit(1); } }
            fprintf(stderr, "weight cache %s: %s\n", cache.c_str(), g_cr ? "reading" : "writing");
        }
        tok_embd = &M.get("token_embd.weight");
        ple = &M.get("per_layer_token_embd.weight");
        for (const GT * g : { tok_embd, ple }) {
            const uintptr_t a = (uintptr_t) g->data & ~(uintptr_t) 4095;
            if (madvise((void *) a, (uintptr_t) g->data + g->nbytes - a, MADV_RANDOM)) perror("madvise");
        }
        ple_data = ple->data;
        if (!ple_file.empty()) {
            int fd = ::open(ple_file.c_str(), O_RDONLY);
            if (fd < 0) { perror(ple_file.c_str()); exit(1); }
            struct stat st; fstat(fd, &st);
            if ((size_t) st.st_size != ple->nbytes) { fprintf(stderr, "%s: %zu bytes, expected %zu\n", ple_file.c_str(), (size_t) st.st_size, ple->nbytes); exit(1); }
            void * m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            madvise(m, st.st_size, MADV_RANDOM);
            ple_data = (const uint8_t *) m;
            fprintf(stderr, "PLE table from %s\n", ple_file.c_str());
        }
        ple_mul = { 23703573157769ull, 20109073645365ull, 8052911324071ull };
        ple_off = { 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455,
                    200000548, 220000655, 240000802, 260000955, 280001114, 300001275 };
        ple_voc = { 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
                    20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 };
        auto t0 = std::chrono::steady_clock::now();
        hL.resize(NL);
        for (int il = 0; il < NL; ++il) {
            Layer & L = hL[il];
            auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
            L.recr = ((il + 1) % 4) != 0;
            L.hca_down = up_q8(M, b("hc_attn_down.weight")); L.hca_up = up_q8(M, b("hc_attn_up.weight"));
            L.hcf_down = up_q8(M, b("hc_ffn_down.weight"));  L.hcf_up = up_q8(M, b("hc_ffn_up.weight"));
            L.hca_norm = up_f32(M, b("hc_attn_norm.weight")); L.hcf_norm = up_f32(M, b("hc_ffn_norm.weight"));
            L.hca_inj = up_f32(M, b("hc_attn_inject.weight")); L.hcf_inj = up_f32(M, b("hc_ffn_inject.weight"));
            if (L.recr) {
                L.qkv = up_q8(M, b("attn_qkv.weight")); L.z = up_q8(M, b("attn_gate.weight"));
                L.ssm_out = up_q8(M, b("ssm_out.weight"));
                L.alpha = up_f32(M, b("ssm_alpha.weight")); L.beta = up_f32(M, b("ssm_beta.weight"));
                L.dt = up_f32(M, b("ssm_dt.bias")); L.a = up_f32(M, b("ssm_a"));
                L.conv = up_f32(M, b("ssm_conv1d.weight")); L.snorm = up_f32(M, b("ssm_norm.weight"));
                L.ssm_state = dalloc<float>((size_t) GV * DS * DS);
                L.conv_ring = dalloc<float>((size_t) 4 * CONVCH);
                state_bufs.push_back({ L.ssm_state, (size_t) GV * DS * DS * 4 });
                state_bufs.push_back({ L.conv_ring, (size_t) 4 * CONVCH * 4 });
            } else {
                L.wq = up_q8(M, b("attn_q.weight")); L.wk = up_q8(M, b("attn_k.weight"));
                L.wv = up_q8(M, b("attn_v.weight")); L.wo = up_q8(M, b("attn_output.weight"));
                L.qn = up_f32(M, b("attn_q_norm.weight")); L.kn = up_f32(M, b("attn_k_norm.weight"));
                L.kc = dalloc<__half>((size_t) MAXCTX * NKV * HD);
                L.vc = dalloc<__half>((size_t) MAXCTX * NKV * HD);
            }
            L.router = up_f32(M, b("ffn_gate_inp.weight"));
            L.sh_gate_inp = up_f32(M, b("ffn_gate_inp_shexp.weight"));
            L.sh_gate = up_q8(M, b("ffn_gate_shexp.weight")); L.sh_up = up_q8(M, b("ffn_up_shexp.weight"));
            L.sh_down = up_q8(M, b("ffn_down_shexp.weight"));
            L.eg = up_ex(M, b("ffn_gate_exps.weight")); L.eu = up_ex(M, b("ffn_up_exps.weight"));
            L.ed = up_ex(M, b("ffn_down_exps.weight"));
            if (il == PLE_LAYER) {
                L.ple_k = up_q8(M, b("ple_key.weight")); L.ple_v = up_q8(M, b("ple_value.weight"));
                L.ple_nk = up_f32(M, b("ple_norm_key.weight")); L.ple_nq = up_f32(M, b("ple_norm_query.weight"));
                L.ple_nc = up_f32(M, b("ple_norm_conv.weight")); L.ple_conv = up_f32(M, b("ple_conv1d.weight"));
            }
            fprintf(stderr, "\rload layer %2d/%d  %.1f GiB on device", il + 1, NL, g_dev_bytes / 1073741824.0);
        }
        G.L = dalloc<Layer>(hL.size());
        CK(cudaMemcpy(G.L, hL.data(), hL.size() * sizeof(Layer), cudaMemcpyHostToDevice));
        G.head_down = up_q8(M, "output_hc_down.weight");
        G.head_up = up_q8(M, "output_hc_up.weight");
        G.head_norm = up_f32(M, "output_hc_norm.weight");
        G.out = up_q8(M, "output.weight");
        if (g_cw) { fclose(g_cw); g_cw = nullptr; rename((cache + ".tmp").c_str(), cache.c_str()); }
        if (g_cr) { fclose(g_cr); g_cr = nullptr; }
        G.res[0] = dalloc<float>(HCD); G.res[1] = dalloc<float>(HCD);
        G.blk_out = dalloc<float>(NE);
        G.inj[0] = dalloc<float>(HC * HC); G.inj[1] = dalloc<float>(HC * HC);
        G.lo = dalloc<float>(HC * LR);
        G.xn = dalloc<float>(HCD);
        G.mixed = dalloc<float>(NE);
        G.proj = dalloc<float>(NH * HD * 2 + 2 * NKV * HD + CONVCH + VDIM + 2 * GV);
        G.core = dalloc<float>(VDIM);
        G.rlog = dalloc<float>(NEXP + 1);
        G.sh_h = dalloc<float>(2 * FF);
        G.exp_h = dalloc<float>((NUSED + 1) * FF);
        G.sel_id = dalloc<int>(NUSED);
        G.sel_w = dalloc<float>(NUSED + 1);
        G.ple_kv = dalloc<float>(HCD + NE);
        G.ple_hist = dalloc<float>((size_t) PLE_HIST * HCD);
        state_bufs.push_back({ G.ple_hist, (size_t) PLE_HIST * HCD * 4 });
        d_emb = dalloc<float>(NE); d_ple = dalloc<float>(NE);
        G.emb = d_emb; G.ple_in = d_ple;
        G.best = dalloc<unsigned long long>(1);
        G.out_tok = dalloc<int>(1);
        if (getenv("MK_PROFILE")) { G.prof = dalloc<unsigned long long>(4096); CK(cudaMallocHost(&h_prof, 4096 * 8)); }
        G.bar_count = dalloc<unsigned>(1);
        unsigned * gen = dalloc<unsigned>(1);
        G.bar_gen = gen;
        CK(cudaMemset(G.bar_count, 0, 4)); CK(cudaMemset(gen, 0, 4));
        CK(cudaMallocHost(&h_emb, NE * 4)); CK(cudaMallocHost(&h_ple, NE * 4)); CK(cudaMallocHost(&h_tok, 4));
        CK(cudaFuncSetAttribute(fnx_step, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SM_BYTES));
        cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
        int per = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fnx_step, NT, SM_BYTES));
        if (per < 1) { fprintf(stderr, "kernel does not fit one block per SM\n"); exit(1); }
        nsm = pr.multiProcessorCount;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "\nloaded in %.1f s, %.2f GiB on device, %d SMs (%s)\n", sec, g_dev_bytes / 1073741824.0, nsm, pr.name);
        reset();
    }

    void reset() {
        for (auto & b : state_bufs) CK(cudaMemset(b.first, 0, b.second));
        seq.clear();
    }

    // host-side n-gram hash of the PLE rows for the token at the end of seq, EOS resets the window
    void ple_rows(int32_t * idx) {
        const int p = (int) seq.size() - 1;
        int64_t ctx[3];
        ctx[0] = seq[p];
        bool cut = false;
        for (int s = 1; s < 3; ++s) {
            const int64_t t = (cut || p - s < 0) ? -1 : seq[p - s];
            cut = cut || t < 0 || t == EOS_TOK;
            ctx[s] = cut ? EOS_TOK : t;
        }
        for (int ng = 2; ng <= 3; ++ng) {
            uint64_t mixed = (uint64_t) ctx[0] * ple_mul[0];
            for (int j = 1; j < ng; ++j) mixed ^= (uint64_t) ctx[j] * ple_mul[j];
            for (int g = 0; g < 8; ++g) {
                const int h = (ng - 2) * 8 + g;
                idx[h] = (int32_t) (mixed % ple_voc[h] + ple_off[h]);
            }
        }
    }

    double bench_phase(int which, int iters, int dbg) {
        G.dbg = dbg;
        CK(cudaMemsetAsync(G.bar_count, 0, 4));
        int n = -1000000 - which * 1000 - iters;
        void * args[] = { &G, &n };
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        CK(cudaLaunchCooperativeKernel((void *) fnx_step, dim3(nsm), dim3(NT), args, SM_BYTES, 0));
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        CK(cudaGetLastError());
        float ms; cudaEventElapsedTime(&ms, a, b);
        G.dbg = 0;
        return 1e3 * ms / iters;
    }

    double bench_barrier(int k) {
        CK(cudaMemsetAsync(G.bar_count, 0, 4));
        int n = -k;
        void * args[] = { &G, &n };
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        CK(cudaLaunchCooperativeKernel((void *) fnx_step, dim3(nsm), dim3(NT), args, SM_BYTES, 0));
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b);
        return 1e3 * ms / k;
    }

    int step(int tok) {
        const int n = (int) seq.size();
        if (n >= MAXCTX) { fprintf(stderr, "context full (%d)\n", MAXCTX); exit(1); }
        seq.push_back(tok);
        const size_t er = ggml_row_size((ggml_type) tok_embd->type, NE);
        const auto * te = ggml_get_type_traits((ggml_type) tok_embd->type);
        te->to_float(tok_embd->data + (size_t) tok * er, h_emb, NE);
        int32_t idx[PLE_HEADS];
        ple_rows(idx);
        const size_t pr = ggml_row_size((ggml_type) ple->type, PLE_HD);
        const auto * tp = ggml_get_type_traits((ggml_type) ple->type);
        const auto th0 = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(PLE_HEADS) schedule(static, 1)
        for (int h = 0; h < PLE_HEADS; ++h) tp->to_float(ple_data + (size_t) idx[h] * pr, h_ple + h * PLE_HD, PLE_HD);
        host_ple_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - th0).count();
        CK(cudaMemcpyAsync(d_emb, h_emb, NE * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpyAsync(d_ple, h_ple, NE * 4, cudaMemcpyHostToDevice));
        CK(cudaMemsetAsync(G.best, 0, 8));
        CK(cudaMemsetAsync(G.bar_count, 0, 4));
        void * args[] = { &G, (void *) &n };
        CK(cudaLaunchCooperativeKernel((void *) fnx_step, dim3(nsm), dim3(NT), args, SM_BYTES, 0));
        CK(cudaMemcpyAsync(h_tok, G.out_tok, 4, cudaMemcpyDeviceToHost));
        CK(cudaStreamSynchronize(0));
        if (G.prof) {
            CK(cudaMemcpy(h_prof, G.prof, 4096 * 8, cudaMemcpyDeviceToHost));
            for (int k = 1; k < 2048 && h_prof[2 * k] != 0; ++k) {
                prof_ns[(int) h_prof[2 * k]] += (double) (h_prof[2 * k + 1] - h_prof[2 * k - 1]);
                prof_cnt[(int) h_prof[2 * k]]++;
            }
            CK(cudaMemset(G.prof, 0, 4096 * 8));
            ++prof_steps;
        }
        return *h_tok;
    }
};

// ================================================================ host: CLI
static std::vector<int> read_ids(const std::string & path) {
    std::vector<int> v;
    FILE * f = fopen(path.c_str(), "r");
    if (!f) { perror(path.c_str()); exit(1); }
    int x;
    while (fscanf(f, " %d ,", &x) == 1 || fscanf(f, " %d", &x) == 1) v.push_back(x);
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    std::vector<std::string> shards;
    std::string prompt = "Write a quicksort in C.", prompt_ids, score_ids, out_ids, cache, ple_file;
    int n_gen = 128, n_prompt_score = -1, reps = 1;
    std::string bench_file;
    bool raw = false, think = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "-m") shards.push_back(nx());
        else if (a == "-p") prompt = nx();
        else if (a == "-n") n_gen = atoi(nx().c_str());
        else if (a == "--raw") raw = true;
        else if (a == "--think") think = true;
        else if (a == "--prompt-ids") prompt_ids = nx();
        else if (a == "--score") score_ids = nx();
        else if (a == "--score-prompt") n_prompt_score = atoi(nx().c_str());
        else if (a == "--out-ids") out_ids = nx();
        else if (a == "--cache") cache = nx();
        else if (a == "--ple-file") ple_file = nx();
        else if (a == "--bench") bench_file = nx();
        else if (a == "--reps") reps = atoi(nx().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (shards.empty()) { fprintf(stderr, "usage: %s -m shard1.gguf -m shard2.gguf ... [-p text | --prompt-ids f] [-n N] [--score ids --score-prompt P]\n", argv[0]); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * vm = llama_model_load_from_file(shards[0].c_str(), mp);
    if (!vm) { fprintf(stderr, "tokenizer load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(vm);
    auto piece = [&](int t) { char buf[256]; int k = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, false); return std::string(buf, k > 0 ? k : 0); };

    Engine E;
    E.load(shards, cache, ple_file);

    if (getenv("MK_PHASES")) {
        static const char * nm[] = { "mix_pre", "mix_post", "lin_proj", "lin_core", "mix_out", "router", "experts_gu", "down", "head", "empty" };
        E.step(9707);  // one real step so routing / state buffers hold sane values
        for (int d : { 2, 2 | 4, 2 | 8, 2 | 16, 2 | 4 | 8 | 16 })
            fprintf(stderr, "mix_pre dbg %2d: %7.2f us\n", d, E.bench_phase(0, 500, d));
        for (int w = 0; w < 10; ++w) {
            const double t0 = E.bench_phase(w, 500, 0), t1 = E.bench_phase(w, 500, 2), t2 = E.bench_phase(w, 500, 0);
            fprintf(stderr, "phase %-10s full %7.2f us | no-gemv %7.2f us | full again %7.2f us\n", nm[w], t0, t1, t2);
        }
        return 0;
    }
    if (getenv("MK_BARRIER")) {
        for (int r = 0; r < 3; ++r) fprintf(stderr, "barrier: %.3f us each (10000 in one launch)\n", E.bench_barrier(10000));
    }
    if (!score_ids.empty()) {
        // teacher-forced agreement against a reference continuation
        const std::vector<int> ids = read_ids(score_ids);
        const int P = n_prompt_score;
        if (P <= 0 || P >= (int) ids.size()) { fprintf(stderr, "--score-prompt must be in (0, %zu)\n", ids.size()); return 1; }
        int agree = 0, tot = 0;
        std::vector<int> miss;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            const int pred = E.step(ids[i]);
            if ((int) i >= P - 1) { ++tot; if (pred == ids[i + 1]) ++agree; else miss.push_back((int) i + 1); }
        }
        printf("score: %d / %d positions agree (%.2f%%)\n", agree, tot, 100.0 * agree / tot);
        for (size_t k = 0; k < miss.size() && k < 20; ++k) printf("  miss at %d\n", miss[k]);
        return 0;
    }

    if (!bench_file.empty()) {
        // benchmark: every line "name<TAB>prompt" x reps, greedy, stops at EOG or n_gen; one JSON line per run
        FILE * f = fopen(bench_file.c_str(), "r");
        if (!f) { perror(bench_file.c_str()); return 1; }
        std::vector<std::pair<std::string, std::string>> items;
        char line[8192];
        while (fgets(line, sizeof line, f)) {
            std::string l(line);
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
            const size_t tab = l.find('\t');
            if (tab != std::string::npos) items.push_back({ l.substr(0, tab), l.substr(tab + 1) });
        }
        fclose(f);
        for (int r = 0; r < reps; ++r) {
            for (auto & it : items) {
                E.reset();
                const std::string text = "<|im_start|>user\n" + it.second + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
                std::vector<int> tk(text.size() + 16);
                tk.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), tk.data(), (int) tk.size(), false, true));
                auto a = std::chrono::steady_clock::now();
                int nx = 0;
                for (int t : tk) nx = E.step(t);
                auto b = std::chrono::steady_clock::now();
                std::vector<int> g;
                for (int i = 0; i < n_gen; ++i) {
                    g.push_back(nx);
                    if (llama_vocab_is_eog(vocab, nx)) break;
                    if (i + 1 < n_gen) nx = E.step(nx);
                }
                auto c = std::chrono::steady_clock::now();
                const double tp = std::chrono::duration<double>(b - a).count(), td = std::chrono::duration<double>(c - b).count();
                printf("{\"name\": \"%s\", \"rep\": %d, \"prompt_tokens\": %zu, \"gen_tokens\": %zu, \"prefill_s\": %.4f, \"decode_s\": %.4f, "
                       "\"decode_tok_s\": %.3f, \"ids\": [", it.first.c_str(), r, tk.size(), g.size(), tp, td, (g.size() - 1) / td);
                for (size_t k = 0; k < tk.size(); ++k) printf("%d,", tk[k]);
                for (size_t k = 0; k < g.size(); ++k) printf("%d%s", g[k], k + 1 < g.size() ? "," : "");
                printf("]}\n");
                fflush(stdout);
                fprintf(stderr, "%s rep %d: decode %.2f tok/s (%zu tok)\n", it.first.c_str(), r, (g.size() - 1) / td, g.size());
            }
        }
        return 0;
    }

    std::vector<int> toks;
    if (!prompt_ids.empty()) {
        toks = read_ids(prompt_ids);
    } else {
        std::string text = raw ? prompt : "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n" +
                                              (think ? "<think>\n" : "<think>\n\n</think>\n\n");
        toks.resize(text.size() + 16);
        const int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, true);
        toks.resize(nt);
    }
    fprintf(stderr, "prompt: %zu tokens\n", toks.size());
    auto t0 = std::chrono::steady_clock::now();
    int next = 0;
    for (int t : toks) next = E.step(t);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<int> gen;
    std::string text;
    for (int i = 0; i < n_gen; ++i) {
        gen.push_back(next);
        text += piece(next);
        if (llama_vocab_is_eog(vocab, next)) break;
        if (i + 1 < n_gen) next = E.step(next);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double tp = std::chrono::duration<double>(t1 - t0).count(), td = std::chrono::duration<double>(t2 - t1).count();
    printf("%s\n", text.c_str());
    fprintf(stderr, "prompt %zu tok in %.2f s (%.1f tok/s, token-by-token) | decode %zu tok in %.2f s = %.2f tok/s\n",
            toks.size(), tp, toks.size() / tp, gen.size(), td, (gen.size() - 1) / td);
    if (!out_ids.empty()) {
        FILE * f = fopen(out_ids.c_str(), "w");
        for (int t : toks) fprintf(f, "%d\n", t);
        for (int t : gen) fprintf(f, "%d\n", t);
        fclose(f);
    }
    fprintf(stderr, "host PLE gather: %.3f ms/step mean\n", 1e3 * E.host_ple_s / E.seq.size());
    if (!E.prof_ns.empty()) {
        double tot = 0; for (auto & kv : E.prof_ns) tot += kv.second;
        fprintf(stderr, "profile over %d steps: %.3f ms/step in barriers-delimited phases\n", E.prof_steps, tot / E.prof_steps / 1e6);
        for (auto & kv : E.prof_ns) fprintf(stderr, "  line %4d  x%-4d  %8.1f us/step  (%.2f us each)\n", kv.first, E.prof_cnt[kv.first] / E.prof_steps,
                                            kv.second / E.prof_steps / 1e3, kv.second / E.prof_cnt[kv.first] / 1e3);
    }
    llama_model_free(vm);
    return 0;
}
