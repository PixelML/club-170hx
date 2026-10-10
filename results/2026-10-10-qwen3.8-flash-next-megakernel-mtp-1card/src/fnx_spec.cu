// fnx_mk.cu - Qwen3.8-Flash-Next (llama.cpp arch qwen4exp) decode megakernel for one CMP 170HX (sm_80).
//
// One persistent cooperative kernel runs a whole decode step (48 layers + head) as phases split by a
// software grid barrier, the design of L-Forster's open-jet megakernel for Qwen3.8-27B
// (https://github.com/L-Forster/open-jet/tree/master/megakernel, AGPL-3.0). The forward pass follows
// llama.cpp's src/models/qwen4exp.cpp (ggml-org/llama.cpp, MIT).
//
// v2: T rows per pass (verify 1 + k drafts), GDN lazy replay, native MTP drafter (unsloth MTP GGUF, shared head).
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
constexpr int PLE_LAYER = 1, PLE_HEADS = 16, PLE_HD = 160;
constexpr int CONV_RING = 8, PLE_RING = 16;  // indexed by absolute position: drafts never overwrite live history
constexpr int MAXCTX = 2048;
constexpr int TMAX = 5;                      // rows per pass: 1 + up to 4 drafts, or catch-up rows
constexpr int KMAX = 8;                      // drafts per cycle
constexpr int PROJ = CONVCH + VDIM + 2 * GV; // 16480 >= NH*HD*2 + 2*NKV*HD = 13312
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
    float * ssm_state;   // [GV][DS][DS] committed state, row j (value dim) contiguous over i (key dim)
    float * conv_ring;   // [CONV_RING][CONVCH] raw qkv projections by position
    float * rk[2];       // replay buffers [TMAX][KDIM] normalised k, [TMAX][VDIM] v, [TMAX][GV] beta, decay
    float * rv[2];
    float * rb[2];
    float * rg[2];
    __half *kc, *vc;     // [MAXCTX][NKV][HD]
};

struct Glob {
    Layer * L;            // NL main layers
    Layer * M;            // the MTP layer
    Q8 head_down, head_up, out, embq;
    const float * head_norm;
    Q8 m_eh, m_hdown, m_hup;           // MTP eh_proj, head mixer
    const float *m_enorm, *m_hnorm, *m_hnorm_head;
    float * res[2];       // [TMAX][HCD]
    float * blk_out;      // [TMAX][NE]
    float * inj[2];       // [TMAX][HC slices][HC]
    float * lo;           // [TMAX][HC][LR]
    float * xn;           // [TMAX][HCD]
    float * mixed;        // [TMAX][NE]
    float * proj;         // [TMAX][PROJ]
    float * core;         // [TMAX][VDIM]
    float * rlog;         // [TMAX][NEXP + 1]
    float * sh_h;         // [TMAX][2 FF]
    float * exp_h;        // [TMAX][(NUSED + 1) FF]
    int   * sel_id;       // [TMAX][NUSED]
    float * sel_w;        // [TMAX][NUSED + 1]
    float * ple_kv;       // [TMAX][HCD + NE]
    float * ple_g;        // [TMAX][2 HC] gate, 1/rms
    float * ple_hist;     // [PLE_RING][HCD]
    float * ple_in;       // [TMAX][NE]
    int   * tok;          // [TMAX] main rows
    float * hmain;        // [TMAX][HCD] trunk residual before the final mixer (MTP input h)
    int   * mtp_tok;      // [TMAX]
    float * mtp_h;        // [TMAX][HCD]
    unsigned long long * best;  // [KMAX + TMAX]
    int * ret;            // [TMAX] targets of a main pass / [KMAX] drafts of an MTP pass
    unsigned long long * prof;
    unsigned * bar_count;
};

struct Step {
    int mode;     // 0 main, 1 MTP, 2 barrier bench
    int n;        // position of row 0
    int T;        // rows
    int replay;   // main: rows of the previous main pass to commit into the GDN state
    int rpar;     // main: replay buffer parity to write (reads 1 - rpar)
    int nchain;   // MTP: drafts to produce (0 = catch-up rows only)
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
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

// block-wide sum, every thread gets the result; red needs NWARP floats
__device__ float bsum(float v, float * red) {
    v = wsum(v);
    __syncthreads();
    if (lane_id() == 0) red[warp_id()] = v;
    __syncthreads();
    float t = lane_id() < NWARP ? red[lane_id()] : 0.0f;
    __syncthreads();
    return wsum(t);
}

// software grid barrier: release-add on arrival, acquire-poll (the open-jet scheme)
__device__ void gsync(const Glob & G, int & pk, int tag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned target = (unsigned) (pk + 1) * gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(G.bar_count) : "memory");
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(G.bar_count) : "memory");
        } while (v < target);
        if (G.prof && blockIdx.x == 0 && pk < 4094) {
            unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            G.prof[2 * (pk + 1)] = (unsigned long long) tag; G.prof[2 * (pk + 1) + 1] = t;
        }
    }
    ++pk;
    __syncthreads();
}

// ---------------------------------------------------------------- shared memory
constexpr int SF = TMAX * NE;                 // floats
constexpr int SF2 = 6144;                     // floats
constexpr int SXQ = 36864;                    // int8: max (NUSED+1)*FF*TMAX = 35200
struct SM {
    float * f;
    float * f2;
    int8_t * xq;
    float * xd;
    float * red;
    int * sid;        // [TMAX][16]
    float * sw;       // [TMAX][16]
    uint32_t * grid;
    int8_t * kv4;
    uint64_t * sg64;
};
__device__ SM smem_layout() {
    extern __shared__ __align__(16) uint8_t sm_raw[];
    SM s;
    s.f = (float *) sm_raw;
    s.f2 = s.f + SF;
    s.xq = (int8_t *) (s.f2 + SF2);
    s.xd = (float *) (s.xq + SXQ);
    s.red = s.xd + SXQ / 32;
    s.sid = (int *) (s.red + 64);
    s.sw = (float *) (s.sid + TMAX * 16);
    s.grid = (uint32_t *) (s.sw + TMAX * 16);
    s.kv4 = (int8_t *) (s.grid + 256);
    s.sg64 = (uint64_t *) (s.grid + 256 + 4);
    return s;
}
constexpr size_t SM_BYTES = SF * 4 + SF2 * 4 + SXQ + (SXQ / 32) * 4 + 64 * 4 + TMAX * 16 * 8 + 256 * 4 + 16 + 128 * 8;

// stage N floats of global row: optional float copy + q8 (xq/xd); all loads before use
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
// T rows of N floats, row stride N in src, packed at stride N in fs/xq, N/32 in xd
template <int N>
__device__ void stage_rows(const float * src, int T, float * fs, int8_t * xq, float * xd) {
    for (int t = 0; t < T; ++t) stage_q8<N>(src + (size_t) t * N, fs ? fs + t * N : nullptr, xq + t * N, xd + t * (N / 32));
    __syncthreads();
}
__device__ void quant_q8(const float * src, int n, int8_t * xq, float * xd) {
    for (int b = warp_id(); b < n / 32; b += NWARP) {
        const float v = src[b * 32 + lane_id()];
        const float amax = wmax(fabsf(v));
        const float id = amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f;
        xq[b * 32 + lane_id()] = (int8_t) __float2int_rn(v * id);
        if (lane_id() == 0) xd[b] = amax * (1.0f / 127.0f);
    }
}

// ---------------------------------------------------------------- GEMV over T rows of activations
// R weight rows x T activation rows; every weight load issued before the math; out[r][t] on all lanes
template <int K, int R>
__device__ __forceinline__ void q8_dot_rows(const Q8 & W, const int (&row)[R], const int8_t * xq, const float * xd, int T,
                                            float (&out)[R][TMAX]) {
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
    for (int t = 0; t < TMAX; ++t) {
        if (t >= T) break;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float acc = 0.0f;
#pragma unroll
            for (int u = 0; u < C; ++u) {
                const int c = lane_id() + 32 * u;
                if (NC % 32 == 0 || c < NC) {
                    const int4 x = *(const int4 *) (xq + t * K + 16 * c);
                    int sm = dp4a(w[r][u].x, x.x, 0); sm = dp4a(w[r][u].y, x.y, sm); sm = dp4a(w[r][u].z, x.z, sm); sm = dp4a(w[r][u].w, x.w, sm);
                    acc += (float) sm * __half2float(d[r][u]) * xd[t * (K / 32) + (c >> 1)];
                }
            }
            out[r][t] = wsum(acc);
        }
    }
}

// rows [0, nrows) of W against T activation rows; f(row, t, value) on lane 0. Returns the next woff.
template <int K, int R, class F>
__device__ __forceinline__ int gemv_q8(const Q8 & W, int nrows, int woff, const int8_t * xq, const float * xd, int T, F f) {
    const int nw = nwarps();
    const int gw = (gwarp() + nw - woff % nw) % nw;
    for (int r0 = gw * R; r0 < nrows; r0 += nw * R) {
        int rows[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rows[r] = min(r0 + r, nrows - 1);
        float v[R][TMAX];
        q8_dot_rows<K, R>(W, rows, xq, xd, T, v);
        if (lane_id() == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (r0 + r < nrows)
                    for (int t = 0; t < T; ++t) f(r0 + r, t, v[r][t]);
        }
    }
    return woff + (nrows + R - 1) / R;
}

// f32 row of NE against T float rows (stride NE); out[t] on all lanes
__device__ __forceinline__ void f32_dot_ne(const float * w, const float * x, int T, float (&out)[TMAX]) {
    constexpr int C = NE / 4 / 32, B = 10;
    float acc[TMAX];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) acc[t] = 0.0f;
#pragma unroll
    for (int b0 = 0; b0 < C; b0 += B) {
        float4 a[B];
#pragma unroll
        for (int u = 0; u < B; ++u) a[u] = __ldg((const float4 *) (w + 4 * (lane_id() + 32 * (b0 + u))));
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t >= T) break;
#pragma unroll
            for (int u = 0; u < B; ++u) {
                const float4 xx = *(const float4 *) (x + t * NE + 4 * (lane_id() + 32 * (b0 + u)));
                acc[t] += a[u].x * xx.x + a[u].y * xx.y + a[u].z * xx.z + a[u].w * xx.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TMAX; ++t) out[t] = wsum(acc[t]);
}

template <class F>
__device__ __forceinline__ int gemv_f32(const float * W, int nrows, int ldw, int woff, const float * x, int T, F f) {
    const int nw = nwarps();
    const int gw = (gwarp() + nw - woff % nw) % nw;
    for (int row = gw; row < nrows; row += nw) {
        float v[TMAX];
        f32_dot_ne(W + (size_t) row * ldw, x, T, v);
        if (lane_id() == 0) for (int t = 0; t < T; ++t) f(row, t, v[t]);
    }
    return woff + nrows;
}

// ---------------------------------------------------------------- expert blocks
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

// integer dot of one 32-block times the block scale (not the activation scale)
template <int T>
__device__ __forceinline__ float exl_dot(const ExL<T> & l, const int8_t * xq, const SM & S) {
    const int * x = (const int *) xq;
    int sumi = 0;
    if constexpr (T == ET_IQ3XXS) {
        const uint32_t g2[2] = { l.g.x, l.g.y };
        const uint8_t * q3 = (const uint8_t *) g2;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int gx = S.grid[q3[l0]], gy = S.grid[q3[l0 + 1]];
            const uint2 sg = *(const uint2 *) (S.sg64 + ((l.aux >> (7 * l0 / 2)) & 0x7f));
            const int s0 = (int) sg.x, s1 = (int) sg.y;
            sumi = dp4a(__vsub4(gx ^ s0, s0), x[l0], sumi);
            sumi = dp4a(__vsub4(gy ^ s1, s1), x[l0 + 1], sumi);
        }
        return __half2float(l.d) * ((float) (l.aux >> 28) + 0.5f) * 0.5f * (float) sumi;
    } else if constexpr (T == ET_IQ4NL) {
        const int qq[4] = { l.q.x, l.q.y, l.q.z, l.q.w };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int2 v = tab16(qq[k], S.kv4);
            sumi = dp4a(v.x, x[k], sumi);
            sumi = dp4a(v.y, x[k + 4], sumi);
        }
    } else {
        sumi = dp4a(l.a.x, x[0], 0); sumi = dp4a(l.a.y, x[1], sumi); sumi = dp4a(l.a.z, x[2], sumi); sumi = dp4a(l.a.w, x[3], sumi);
        sumi = dp4a(l.b.x, x[4], sumi); sumi = dp4a(l.b.y, x[5], sumi); sumi = dp4a(l.b.z, x[6], sumi); sumi = dp4a(l.b.w, x[7], sumi);
    }
    return __half2float(l.d) * (float) sumi;
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
// the 16 split-K inject partials of row t -> 4 combine weights (smem broadcast; contains __syncthreads)
__device__ __forceinline__ void inj_weights(const float * inj, float * sm16, float (&w)[HC]) {
    __syncthreads();
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

// HC mix, first half, T rows. mode 0: residual as is; 1: + combine of the pending block output;
// 2: residual = token embedding x4 (tokens from tok[]). Block b works on hc stream k = b % HC.
__device__ __noinline__ int mix_pre(const Glob & G, const SM & S, int T, int cur, int mode, int injp, const float * wn,
                                    const Q8 & down, const float * winj, float * inj_out, const int * tok) {
    constexpr int U = NE / 32 / NWARP;  // 5 blocks of 32 per warp
    const int k = blockIdx.x % HC, kb = blockIdx.x / HC, nkb = (gridDim.x - k + HC - 1) / HC;
    const int w = warp_id(), ln = lane_id();
    const int ncur = mode != 0 ? 1 - cur : cur;
    float wv[U];
#pragma unroll
    for (int u = 0; u < U; ++u) wv[u] = __ldg(wn + k * NE + (w * U + u) * 32 + ln);
    for (int t = 0; t < T; ++t) {
        float x[U];
        if (mode == 2) {
            const int tk = tok[t];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int i = (w * U + u) * 32 + ln;
                x[u] = (float) G.embq.q[(size_t) tk * NE + i] * __half2float(G.embq.d[(size_t) tk * (NE / 32) + (i >> 5)]);
            }
        } else {
#pragma unroll
            for (int u = 0; u < U; ++u) x[u] = ldcg(G.res[cur] + (size_t) t * HCD + k * NE + (w * U + u) * 32 + ln);
        }
        if (mode == 1) {
            float wi[HC];
            inj_weights(G.inj[injp] + t * HC * HC, S.red + 32, wi);
            float bo[U];
#pragma unroll
            for (int u = 0; u < U; ++u) bo[u] = ldcg(G.blk_out + (size_t) t * NE + (w * U + u) * 32 + ln);
#pragma unroll
            for (int u = 0; u < U; ++u) x[u] += bo[u] * wi[k];
        }
        float ss = 0.0f;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            ss += x[u] * x[u];
            if (mode != 0 && (w * U + u) % nkb == kb) G.res[ncur][(size_t) t * HCD + k * NE + (w * U + u) * 32 + ln] = x[u];
        }
        ss = wsum(ss);
        __syncthreads();
        if (ln == 0) S.red[w] = ss;
        __syncthreads();
        float tot = 0.0f;
#pragma unroll
        for (int q = 0; q < NWARP; ++q) tot += S.red[q];
        const float inv = rsqrtf(tot / NE + EPS);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = (w * U + u) * 32 + ln;
            const float xn = x[u] * inv * wv[u];
            const float amax = wmax(fabsf(xn));
            const float id = amax > 0.0f ? __fdividef(127.0f, amax) : 0.0f;
            S.xq[t * NE + i] = (int8_t) __float2int_rn(xn * id);
            S.f[t * NE + i] = xn;
            if (ln == 0) S.xd[t * (NE / 32) + w * U + u] = amax * (1.0f / 127.0f);
            if ((w * U + u) % nkb == kb) G.xn[(size_t) t * HCD + k * NE + i] = xn;
        }
    }
    __syncthreads();
    // rows of slice k: down rows (K = 2560 slice) + the HC inject rows of this slice
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
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float acc = 0.0f;
#pragma unroll
                for (int u = 0; u < C; ++u) {
                    const int c = ln + 32 * u;
                    const int4 xv = *(const int4 *) (S.xq + t * NE + 16 * c);
                    int sm = dp4a(wq[r][u].x, xv.x, 0); sm = dp4a(wq[r][u].y, xv.y, sm); sm = dp4a(wq[r][u].z, xv.z, sm); sm = dp4a(wq[r][u].w, xv.w, sm);
                    acc += (float) sm * __half2float(wd[r][u]) * S.xd[t * (NE / 32) + (c >> 1)];
                }
                acc = wsum(acc);
                if (ln == 0 && r0 + r < down.rows) G.lo[((size_t) t * HC + k) * LR + r0 + r] = acc;
            }
        }
    }
    if (winj) {
        for (int s = gw; s < HC; s += nw) {
            float v[TMAX];
            f32_dot_ne(winj + (size_t) s * HCD + k * NE, S.f, T, v);
            if (ln == 0) for (int t = 0; t < T; ++t) inj_out[t * HC * HC + k * HC + s] = v[t];
        }
    }
    return ncur;
}

// HC mix, second half: gate = up(silu(lo/hc)); mixed = mean_s(xn_s * sigmoid(gate_s)); optional copy of the
// residual rows (hsave: the MTP input h of a main pass, or the chained h of an MTP pass)
__device__ __noinline__ void mix_post(const Glob & G, const SM & S, int T, int cur, const Q8 & up, float * hsave, int hrow0) {
    for (int i = threadIdx.x; i < T * LR; i += NT) {
        const int t = i / LR, j = i % LR;
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) v += ldcg(G.lo + ((size_t) t * HC + k) * LR + j);
        S.f2[i] = silu(v * (1.0f / HC));
    }
    __syncthreads();
    quant_q8(S.f2, T * LR, S.xq, S.xd);
    __syncthreads();
    for (int i = gwarp(); i < NE; i += nwarps()) {
        const int rows[HC] = { i, NE + i, 2 * NE + i, 3 * NE + i };
        float g[HC][TMAX];
        q8_dot_rows<LR, HC>(up, rows, S.xq, S.xd, T, g);
        if (lane_id() == 0) {
            for (int t = 0; t < T; ++t) {
                float m = 0.0f;
#pragma unroll
                for (int s = 0; s < HC; ++s) m += ldcg(G.xn + (size_t) t * HCD + s * NE + i) * sigm(g[s][t]);
                G.mixed[(size_t) t * NE + i] = m * (1.0f / HC);
            }
        }
    }
    if (hsave) {
        const int nrow = T - hrow0;
        for (int i = blockIdx.x * NT + threadIdx.x; i < nrow * HCD; i += gridDim.x * NT)
            hsave[i] = ldcg(G.res[cur] + (size_t) hrow0 * HCD + i);
    }
}

__device__ __noinline__ void lin_proj(const Glob & G, const SM & S, int T, const Layer & L) {
    stage_rows<NE>(G.mixed, T, S.f, S.xq, S.xd);
    float * P = G.proj;
    int wo = 0;
    wo = gemv_q8<NE, 2>(L.qkv, CONVCH, wo, S.xq, S.xd, T, [&](int r, int t, float v) { P[t * PROJ + r] = v; });
    wo = gemv_q8<NE, 2>(L.z, VDIM, wo, S.xq, S.xd, T, [&](int r, int t, float v) { P[t * PROJ + CONVCH + r] = v; });
    wo = gemv_f32(L.beta, GV, NE, wo, S.f, T, [&](int r, int t, float v) { P[t * PROJ + CONVCH + VDIM + r] = v; });
    gemv_f32(L.alpha, GV, NE, wo, S.f, T, [&](int r, int t, float v) { P[t * PROJ + CONVCH + VDIM + GV + r] = v; });
}

// gated delta net, one value head per block: replay the accepted rows of the previous pass into the committed
// state, then run the T new rows on a register copy without committing them (lazy replay, as open-jet does)
__device__ __noinline__ void lin_core(const Glob & G, const SM & S, const Step & P, const Layer & L) {
    const int h = blockIdx.x;
    if (h >= GV) return;
    const int kh = h % GK, n = P.n, T = P.T;
    const int j = threadIdx.x >> 2, part = threadIdx.x & 3;
    float * st = L.ssm_state + (size_t) h * DS * DS + (size_t) j * DS + part * 32;
    float sreg[32];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const float4 t4 = __ldcg((const float4 *) (st + 4 * u));
        sreg[4 * u] = t4.x; sreg[4 * u + 1] = t4.y; sreg[4 * u + 2] = t4.z; sreg[4 * u + 3] = t4.w;
    }
    const int rp = 1 - P.rpar;
    for (int r = 0; r < P.replay; ++r) {
        const float * kr = L.rk[rp] + (size_t) r * KDIM + kh * DS + part * 32;
        const float vj = ldcg(L.rv[rp] + (size_t) r * VDIM + h * DS + j);
        const float beta = ldcg(L.rb[rp] + r * GV + h), decay = ldcg(L.rg[rp] + r * GV + h);
        float kk[32];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float4 t4 = __ldcg((const float4 *) (kr + 4 * u));
            kk[4 * u] = t4.x; kk[4 * u + 1] = t4.y; kk[4 * u + 2] = t4.z; kk[4 * u + 3] = t4.w;
        }
        float kv = 0.0f;
#pragma unroll
        for (int u = 0; u < 32; ++u) { sreg[u] *= decay; kv += sreg[u] * kk[u]; }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float delta = (vj - kv) * beta;
#pragma unroll
        for (int u = 0; u < 32; ++u) sreg[u] += kk[u] * delta;
    }
    if (P.replay > 0) {
#pragma unroll
        for (int u = 0; u < 8; ++u)
            __stcg((float4 *) (st + 4 * u), make_float4(sreg[4 * u], sreg[4 * u + 1], sreg[4 * u + 2], sreg[4 * u + 3]));
    }
    float * q = S.f2, * k = S.f2 + DS, * v = S.f2 + 2 * DS, * o = S.f2 + 3 * DS;
    for (int t = 0; t < T; ++t) {
        const int pos = n + t;
        // causal conv (kernel 4) + silu on the 384 channels this head reads; rows >= n come from this pass
        for (int c = threadIdx.x; c < 3 * DS; c += NT) {
            const int pt = c / DS, i = c % DS;
            const int ch = pt == 0 ? kh * DS + i : pt == 1 ? KDIM + kh * DS + i : 2 * KDIM + h * DS + i;
            float acc = L.conv[3 + 4 * ch] * ldcg(G.proj + (size_t) t * PROJ + ch);
#pragma unroll
            for (int kk = 0; kk < 3; ++kk) {
                const int p = pos - (3 - kk);
                float xp = 0.0f;
                if (p >= n) xp = ldcg(G.proj + (size_t) (p - n) * PROJ + ch);
                else if (p >= 0) xp = ldcg(L.conv_ring + (p & (CONV_RING - 1)) * CONVCH + ch);
                acc += L.conv[kk + 4 * ch] * xp;
            }
            (pt == 0 ? q : pt == 1 ? k : v)[i] = silu(acc);
        }
        __syncthreads();
        float sq = 0.0f, sk = 0.0f;
        for (int i = threadIdx.x; i < DS; i += NT) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
        sq = bsum(sq, S.red);
        sk = bsum(sk, S.red);
        const float iq = rsqrtf(sq + EPS), ik = rsqrtf(sk + EPS);
        for (int i = threadIdx.x; i < DS; i += NT) { q[i] *= iq; k[i] *= ik; }
        __syncthreads();
        const float beta = sigm(ldcg(G.proj + (size_t) t * PROJ + CONVCH + VDIM + h));
        const float al = ldcg(G.proj + (size_t) t * PROJ + CONVCH + VDIM + GV + h) + L.dt[h];
        const float sp = al > 20.0f ? al : log1pf(__expf(al));
        const float decay = __expf(sp * L.a[h]);
        // replay record of this row (k by the first block of its key head)
        if (h < GK) for (int i = threadIdx.x; i < DS; i += NT) L.rk[P.rpar][(size_t) t * KDIM + kh * DS + i] = k[i];
        for (int i = threadIdx.x; i < DS; i += NT) L.rv[P.rpar][(size_t) t * VDIM + h * DS + i] = v[i];
        if (threadIdx.x == 0) { L.rb[P.rpar][t * GV + h] = beta; L.rg[P.rpar][t * GV + h] = decay; }
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
        oj *= rsqrtf((float) DS);
        if (part == 0) o[j] = oj;
        __syncthreads();
        float ss = 0.0f;
        for (int i = threadIdx.x; i < DS; i += NT) ss += o[i] * o[i];
        ss = bsum(ss, S.red);
        const float inv = rsqrtf(ss / DS + EPS);
        for (int i = threadIdx.x; i < DS; i += NT) {
            const float zz = ldcg(G.proj + (size_t) t * PROJ + CONVCH + h * DS + i);
            G.core[(size_t) t * VDIM + h * DS + i] = o[i] * inv * L.snorm[i] * sigm(zz);
        }
        __syncthreads();
    }
    // raw projections of the new rows into the position ring (q/k channels by the first head of the group)
    for (int c = threadIdx.x; c < 3 * DS; c += NT) {
        const int pt = c / DS, i = c % DS;
        if (pt < 2 && h >= GK) continue;
        const int ch = pt == 0 ? kh * DS + i : pt == 1 ? KDIM + kh * DS + i : 2 * KDIM + h * DS + i;
        for (int t = 0; t < T; ++t) L.conv_ring[((n + t) & (CONV_RING - 1)) * CONVCH + ch] = ldcg(G.proj + (size_t) t * PROJ + ch);
    }
}

__device__ __noinline__ void att_proj(const Glob & G, const SM & S, int T, const Layer & L) {
    stage_rows<NE>(G.mixed, T, nullptr, S.xq, S.xd);
    float * P = G.proj;
    constexpr int n0 = NH * HD * 2, n1 = n0 + NKV * HD;
    int wo = 0;
    wo = gemv_q8<NE, 2>(L.wq, n0, wo, S.xq, S.xd, T, [&](int r, int t, float v) { P[t * PROJ + r] = v; });
    wo = gemv_q8<NE, 2>(L.wk, NKV * HD, wo, S.xq, S.xd, T, [&](int r, int t, float v) { P[t * PROJ + n0 + r] = v; });
    gemv_q8<NE, 2>(L.wv, NKV * HD, wo, S.xq, S.xd, T, [&](int r, int t, float v) { P[t * PROJ + n1 + r] = v; });
}

// rms norm (head) + neox rope on the first NROT dims, called by a whole block
__device__ void norm_rope(float * x, const float * w, int pos, float * red) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < HD; i += NT) ss += x[i] * x[i];
    ss = bsum(ss, red);
    const float inv = rsqrtf(ss / HD + EPS);
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

// one query head per block, dense causal attention; rows of this pass see each other causally
__device__ __noinline__ void att_core(const Glob & G, const SM & S, int n, int T, const Layer & L) {
    const int h = blockIdx.x;
    if (h >= NH) return;
    const int g = h / (NH / NKV);
    constexpr int n0 = NH * HD * 2;
    float * kl = S.f2, * vl = S.f2 + TMAX * HD, * q = S.f2 + 2 * TMAX * HD;
    float * mrg = S.f;  // NWARP x (HD + 2)
    for (int t = 0; t < T; ++t) {
        for (int i = threadIdx.x; i < HD; i += NT) {
            kl[t * HD + i] = ldcg(G.proj + (size_t) t * PROJ + n0 + g * HD + i);
            vl[t * HD + i] = ldcg(G.proj + (size_t) t * PROJ + n0 + NKV * HD + g * HD + i);
        }
        __syncthreads();
        norm_rope(kl + t * HD, L.kn, n + t, S.red);
        // round to the cache precision so a row reads the same keys whether they come from this pass or the cache
        for (int i = threadIdx.x; i < HD; i += NT) {
            kl[t * HD + i] = __half2float(__float2half(kl[t * HD + i]));
            vl[t * HD + i] = __half2float(__float2half(vl[t * HD + i]));
        }
        __syncthreads();
        if (h % (NH / NKV) == 0) {
            for (int i = threadIdx.x; i < HD; i += NT) {
                L.kc[((size_t) (n + t) * NKV + g) * HD + i] = __float2half(kl[t * HD + i]);
                L.vc[((size_t) (n + t) * NKV + g) * HD + i] = __float2half(vl[t * HD + i]);
            }
        }
    }
    const float scale = rsqrtf((float) HD);
    for (int t = 0; t < T; ++t) {
        for (int i = threadIdx.x; i < HD; i += NT) q[i] = ldcg(G.proj + (size_t) t * PROJ + h * 2 * HD + i);
        __syncthreads();
        norm_rope(q, L.qn, n + t, S.red);
        float qr[8], acc[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) { qr[u] = q[lane_id() * 8 + u] * scale; acc[u] = 0.0f; }
        float m = -INFINITY, l = 0.0f;
        for (int p = warp_id(); p <= n + t; p += NWARP) {
            float kk[8], vv[8];
            if (p >= n) {
#pragma unroll
                for (int u = 0; u < 8; ++u) { kk[u] = kl[(p - n) * HD + lane_id() * 8 + u]; vv[u] = vl[(p - n) * HD + lane_id() * 8 + u]; }
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
            const float gate = ldcg(G.proj + (size_t) t * PROJ + h * 2 * HD + HD + i);
            G.core[(size_t) t * VDIM + h * HD + i] = num / den * sigm(gate);
        }
        __syncthreads();
    }
}

// mixer output projection (K = 6144) -> blk_out
__device__ __noinline__ void mix_out(const Glob & G, const SM & S, int T, const Q8 & W) {
    stage_rows<VDIM>(G.core, T, nullptr, S.xq, S.xd);
    float * O = G.blk_out;
    gemv_q8<VDIM, 1>(W, NE, 0, S.xq, S.xd, T, [&](int r, int t, float v) { O[t * NE + r] = v; });
}

__device__ __noinline__ void ffn_router(const Glob & G, const SM & S, int T, const Layer & L) {
    stage_rows<NE>(G.mixed, T, S.f, S.xq, S.xd);
    float * RL = G.rlog, * SH = G.sh_h;
    int wo = 0;
    wo = gemv_f32(L.router, NEXP, NE, wo, S.f, T, [&](int r, int t, float v) { RL[t * (NEXP + 1) + r] = v; });
    wo = gemv_f32(L.sh_gate_inp, 1, NE, wo, S.f, T, [&](int, int t, float v) { RL[t * (NEXP + 1) + NEXP] = v; });
    wo = gemv_q8<NE, 2>(L.sh_gate, FF, wo, S.xq, S.xd, T, [&](int r, int t, float v) { SH[t * 2 * FF + r] = v; });
    gemv_q8<NE, 2>(L.sh_up, FF, wo, S.xq, S.xd, T, [&](int r, int t, float v) { SH[t * 2 * FF + FF + r] = v; });
}

template <int ET, int R>
__device__ __forceinline__ void gu_body(const Glob & G, const SM & S, int T, const Layer & L) {
    constexpr int NSB = NE / 32, J = (NSB + 31) / 32;
    const int per_t = NUSED * FF / R;
    for (int task = gwarp(); task < T * per_t; task += nwarps()) {
        const int t = task / per_t, rem = task % per_t;
        const int e = rem / (FF / R), r0 = (rem % (FF / R)) * R;
        const int eid = S.sid[t * 16 + e];
        ExL<ET> lg[R][J], lu[R][J];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *gq, *gs, *uq, *us; const __half *gd, *ud;
            ex_row(L.eg, eid, r0 + r, gq, gd, gs);
            ex_row(L.eu, eid, r0 + r, uq, ud, us);
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const int sb = lane_id() + 32 * j;
                if (sb < NSB) { lg[r][j] = exl_load<ET>(gq, gd, gs, sb); lu[r][j] = exl_load<ET>(uq, ud, us, sb); }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float ga = 0.0f, ua = 0.0f;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const int sb = lane_id() + 32 * j;
                if (sb < NSB) {
                    const float xd = S.xd[t * NSB + sb];
                    ga += xd * exl_dot<ET>(lg[r][j], S.xq + t * NE + sb * 32, S);
                    ua += xd * exl_dot<ET>(lu[r][j], S.xq + t * NE + sb * 32, S);
                }
            }
            ga = wsum(ga); ua = wsum(ua);
            if (lane_id() == 0) G.exp_h[(size_t) t * (NUSED + 1) * FF + e * FF + r0 + r] = silu(ga) * ua;
        }
    }
}

// top-k per row (warp t), shared expert activation, expert gate/up
__device__ __noinline__ void ffn_experts_gu(const Glob & G, const SM & S, int T, const Layer & L) {
    if (warp_id() < T) {
        const int t = warp_id();
        const float * lg = G.rlog + t * (NEXP + 1);
        float vals[NEXP / 32];
#pragma unroll
        for (int u = 0; u < NEXP / 32; ++u) vals[u] = ldcg(lg + u * 32 + lane_id());
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
            if (lane_id() == 0) S.sid[t * 16 + k] = bi;
            if ((bi & 31) == lane_id()) vals[bi >> 5] = -INFINITY;
        }
        if (lane_id() == 0) {
            float sum = 0.0f, w[NUSED];
            for (int k = 0; k < NUSED; ++k) { w[k] = __expf(top[k] - top[0]); sum += w[k]; }
            for (int k = 0; k < NUSED; ++k) S.sw[t * 16 + k] = w[k] / sum;
            S.sw[t * 16 + NUSED] = sigm(ldcg(lg + NEXP));
            if (blockIdx.x == 0) {
                for (int k = 0; k < NUSED; ++k) { G.sel_id[t * NUSED + k] = S.sid[t * 16 + k]; G.sel_w[t * (NUSED + 1) + k] = S.sw[t * 16 + k]; }
                G.sel_w[t * (NUSED + 1) + NUSED] = S.sw[t * 16 + NUSED];
            }
        }
    }
    for (int i = blockIdx.x * NT + threadIdx.x; i < T * FF; i += gridDim.x * NT) {
        const int t = i / FF, r = i % FF;
        G.exp_h[(size_t) t * (NUSED + 1) * FF + NUSED * FF + r] = silu(ldcg(G.sh_h + t * 2 * FF + r)) * ldcg(G.sh_h + t * 2 * FF + FF + r);
    }
    stage_rows<NE>(G.mixed, T, nullptr, S.xq, S.xd);
    if (L.eg.type == ET_IQ3XXS)    gu_body<ET_IQ3XXS, 2>(G, S, T, L);
    else if (L.eg.type == ET_Q8)   gu_body<ET_Q8, 1>(G, S, T, L);
    else                           gu_body<ET_IQ4NL, 2>(G, S, T, L);
}

template <int ET>
__device__ __forceinline__ void down_body(const Glob & G, const SM & S, int T, const Layer & L) {
    constexpr int NB = FF / 32, NE_T = NUSED * NB, JE = (NE_T + 31) / 32, ROW = (NUSED + 1) * FF;
    for (int row = gwarp(); row < NE; row += nwarps()) {
        ExL<ET_Q8> ls;
        if (lane_id() < NB)
            ls = exl_load<ET_Q8>((const uint8_t *) (L.sh_down.q + (size_t) row * FF), L.sh_down.d + (size_t) row * NB, nullptr, lane_id());
        for (int t = 0; t < T; ++t) {
            ExL<ET> le[JE];
#pragma unroll
            for (int j = 0; j < JE; ++j) {
                const int k = lane_id() + 32 * j;
                if (k < NE_T) {
                    const uint8_t *rq, *rs; const __half * rd;
                    ex_row(L.ed, S.sid[t * 16 + k / NB], row, rq, rd, rs);
                    le[j] = exl_load<ET>(rq, rd, rs, k % NB);
                }
            }
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < JE; ++j) {
                const int k = lane_id() + 32 * j;
                if (k < NE_T) acc += S.sw[t * 16 + k / NB] * S.xd[t * (ROW / 32) + k] * exl_dot<ET>(le[j], S.xq + t * ROW + k * 32, S);
            }
            if (lane_id() < NB) {
                const int k = NE_T + lane_id();
                acc += S.sw[t * 16 + NUSED] * S.xd[t * (ROW / 32) + k] * exl_dot<ET_Q8>(ls, S.xq + t * ROW + k * 32, S);
            }
            acc = wsum(acc);
            if (lane_id() == 0) G.blk_out[(size_t) t * NE + row] = acc;
        }
    }
}

__device__ __noinline__ void ffn_down(const Glob & G, const SM & S, int T, const Layer & L) {
    for (int i = threadIdx.x; i < T * NUSED; i += NT) S.sid[(i / NUSED) * 16 + i % NUSED] = __ldcg(G.sel_id + i);
    for (int i = threadIdx.x; i < T * (NUSED + 1); i += NT) S.sw[(i / (NUSED + 1)) * 16 + i % (NUSED + 1)] = ldcg(G.sel_w + i);
    stage_rows<(NUSED + 1) * FF>(G.exp_h, T, nullptr, S.xq, S.xd);
    if (L.ed.type == ET_IQ4NL) down_body<ET_IQ4NL>(G, S, T, L);
    else                       down_body<ET_Q8>(G, S, T, L);
}

// PLE A: key (HCD) and value (NE) projections of the gathered n-gram rows
__device__ __noinline__ void ple_proj(const Glob & G, const SM & S, int T, const Layer & L) {
    stage_rows<NE>(G.ple_in, T, nullptr, S.xq, S.xd);
    float * KV = G.ple_kv;
    const int wo = gemv_q8<NE, 2>(L.ple_k, HCD, 0, S.xq, S.xd, T, [&](int r, int t, float v) { KV[t * (HCD + NE) + r] = v; });
    gemv_q8<NE, 2>(L.ple_v, NE, wo, S.xq, S.xd, T, [&](int r, int t, float v) { KV[t * (HCD + NE) + HCD + r] = v; });
}

// PLE B: one block per (row, stream): the gate and the 1/rms of the gated value; the residual is the one after
// the pending ffn combine of layer 0, built on the fly
__device__ __noinline__ void ple_gate(const Glob & G, const SM & S, int T, const Layer & L, int cur, int injp) {
    for (int task = blockIdx.x; task < T * HC; task += gridDim.x) {
        const int t = task / HC, s = task % HC;
        float wi[HC];
        inj_weights(G.inj[injp] + t * HC * HC, S.red + 32, wi);
        const float * kv = G.ple_kv + (size_t) t * (HCD + NE);
        float kk = 0.0f, qq = 0.0f, vv = 0.0f, dt = 0.0f;
        for (int i = threadIdx.x; i < NE; i += NT) {
            const float r = ldcg(G.res[cur] + (size_t) t * HCD + s * NE + i) + ldcg(G.blk_out + (size_t) t * NE + i) * wi[s];
            const float kx = ldcg(kv + s * NE + i), vx = ldcg(kv + HCD + i);
            kk += kx * kx; qq += r * r; vv += vx * vx;
            dt += kx * L.ple_nk[s * NE + i] * r * L.ple_nq[s * NE + i];
        }
        kk = bsum(kk, S.red); qq = bsum(qq, S.red); vv = bsum(vv, S.red); dt = bsum(dt, S.red);
        if (threadIdx.x == 0) {
            const float d = dt * rsqrtf(kk / NE + EPS) * rsqrtf(qq / NE + EPS) * rsqrtf((float) NE);
            const float mag = sqrtf(fmaxf(fabsf(d), 1e-6f));
            const float gate = sigm(d > 0.0f ? mag : (d < 0.0f ? -mag : 0.0f));
            G.ple_g[t * 2 * HC + s] = gate;
            G.ple_g[t * 2 * HC + HC + s] = rsqrtf(gate * gate * vv / NE + EPS);
        }
    }
}

// PLE C: gated value + dilated depthwise conv, added to the combined residual. Returns the new cur.
__device__ __noinline__ int ple_apply(const Glob & G, const SM & S, int n, int T, const Layer & L, int cur, int injp) {
    float wi[TMAX][HC];
    for (int t = 0; t < T; ++t) inj_weights(G.inj[injp] + t * HC * HC, S.red + 32, wi[t]);
    const int ncur = 1 - cur;
    const int per = (HCD + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * per, hi = min(HCD, lo + per);
    for (int c = lo + threadIdx.x; c < hi; c += NT) {
        const int s = c / NE, i = c % NE;
        float nrm[TMAX];
        for (int t = 0; t < T; ++t) {
            const float r = ldcg(G.res[cur] + (size_t) t * HCD + c) + ldcg(G.blk_out + (size_t) t * NE + i) * wi[t][s];
            const float gated = ldcg(G.ple_kv + (size_t) t * (HCD + NE) + HCD + i) * ldcg(G.ple_g + t * 2 * HC + s);
            nrm[t] = gated * ldcg(G.ple_g + t * 2 * HC + HC + s) * L.ple_nc[c];
            float acc = L.ple_conv[3 + 4 * c] * nrm[t];
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int p = n + t - (3 - k) * 3;
                if (p >= n) acc += L.ple_conv[k + 4 * c] * nrm[p - n];
                else if (p >= 0) acc += L.ple_conv[k + 4 * c] * ldcg(G.ple_hist + (p & (PLE_RING - 1)) * HCD + c);
            }
            G.res[ncur][(size_t) t * HCD + c] = r + gated + silu(acc);
        }
        for (int t = 0; t < T; ++t) G.ple_hist[((n + t) & (PLE_RING - 1)) * HCD + c] = nrm[t];
    }
    return ncur;
}

// lm head + argmax for rows [r0, r0 + T) of mixed; best[slot + t]
__device__ __noinline__ void head_out(const Glob & G, const SM & S, int r0, int T, int slot) {
    stage_rows<NE>(G.mixed + (size_t) r0 * NE, T, nullptr, S.xq, S.xd);
    float bv[TMAX]; int bi[TMAX];
    for (int t = 0; t < TMAX; ++t) { bv[t] = -INFINITY; bi[t] = 0; }
    for (int q0 = gwarp() * 2; q0 < VOCAB; q0 += nwarps() * 2) {
        const int rows[2] = { q0, min(q0 + 1, VOCAB - 1) };
        float v[2][TMAX];
        q8_dot_rows<NE, 2>(G.out, rows, S.xq, S.xd, T, v);
#pragma unroll
        for (int t = 0; t < TMAX; ++t) {
            if (t >= T) break;
            if (v[0][t] > bv[t]) { bv[t] = v[0][t]; bi[t] = q0; }
            if (q0 + 1 < VOCAB && v[1][t] > bv[t]) { bv[t] = v[1][t]; bi[t] = q0 + 1; }
        }
    }
    if (lane_id() == 0) {
        for (int t = 0; t < T; ++t) {
            unsigned u = __float_as_uint(bv[t]);
            u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            atomicMax(G.best + slot + t, ((unsigned long long) u << 32) | (unsigned) (0xffffffffu - (unsigned) bi[t]));
        }
    }
}
__device__ __forceinline__ int best_tok(const Glob & G, int slot) {
    const unsigned long long k = __ldcg(G.best + slot);
    return (int) (0xffffffffu - (unsigned) (k & 0xffffffffu));
}

// MTP input: res[0][t][s] = eh_proj @ [enorm(embed(tok_t)) ; hnorm_s(h_t,s)]
__device__ __noinline__ void mtp_in(const Glob & G, const SM & S, int T, const int * toks) {
    // e part: en_t staged at xq[0..T*NE); h part per stream at xq[T*NE..2*T*NE)
    float * ef = S.f;
    for (int t = 0; t < T; ++t) {
        const int tk = toks[t];
        float ss = 0.0f;
        for (int i = threadIdx.x; i < NE; i += NT) {
            const float e = (float) G.embq.q[(size_t) tk * NE + i] * __half2float(G.embq.d[(size_t) tk * (NE / 32) + (i >> 5)]);
            ef[t * NE + i] = e; ss += e * e;
        }
        ss = bsum(ss, S.red);
        const float inv = rsqrtf(ss / NE + EPS);
        for (int i = threadIdx.x; i < NE; i += NT) ef[t * NE + i] *= inv * G.m_enorm[i];
    }
    __syncthreads();
    quant_q8(ef, T * NE, S.xq, S.xd);
    int8_t * hq = S.xq + T * NE;
    float * hd = S.xd + T * (NE / 32);
    for (int s = 0; s < HC; ++s) {
        __syncthreads();
        for (int t = 0; t < T; ++t) {
            float ss = 0.0f;
            for (int i = threadIdx.x; i < NE; i += NT) { const float v = ldcg(G.mtp_h + (size_t) t * HCD + s * NE + i); ef[t * NE + i] = v; ss += v * v; }
            ss = bsum(ss, S.red);
            const float inv = rsqrtf(ss / NE + EPS);
            for (int i = threadIdx.x; i < NE; i += NT) ef[t * NE + i] *= inv * G.m_hnorm[s * NE + i];
        }
        __syncthreads();
        quant_q8(ef, T * NE, hq, hd);
        __syncthreads();
        // rows of eh_proj: K = 2 NE, first half against e, second against h
        for (int r = gwarp(); r < NE; r += nwarps()) {
            constexpr int C = NE / 16 / 32;
            int4 we[C], wh[C]; __half de[C], dh[C];
#pragma unroll
            for (int u = 0; u < C; ++u) {
                const int c = lane_id() + 32 * u;
                we[u] = __ldg((const int4 *) (G.m_eh.q + (size_t) r * 2 * NE + 16 * c));
                wh[u] = __ldg((const int4 *) (G.m_eh.q + (size_t) r * 2 * NE + NE + 16 * c));
                de[u] = __ldg(G.m_eh.d + (size_t) r * (2 * NE / 32) + (c >> 1));
                dh[u] = __ldg(G.m_eh.d + (size_t) r * (2 * NE / 32) + NE / 32 + (c >> 1));
            }
            for (int t = 0; t < T; ++t) {
                float acc = 0.0f;
#pragma unroll
                for (int u = 0; u < C; ++u) {
                    const int c = lane_id() + 32 * u;
                    const int4 xe = *(const int4 *) (S.xq + t * NE + 16 * c);
                    const int4 xh = *(const int4 *) (hq + t * NE + 16 * c);
                    int a = dp4a(we[u].x, xe.x, 0); a = dp4a(we[u].y, xe.y, a); a = dp4a(we[u].z, xe.z, a); a = dp4a(we[u].w, xe.w, a);
                    int b = dp4a(wh[u].x, xh.x, 0); b = dp4a(wh[u].y, xh.y, b); b = dp4a(wh[u].z, xh.z, b); b = dp4a(wh[u].w, xh.w, b);
                    acc += (float) a * __half2float(de[u]) * S.xd[t * (NE / 32) + (c >> 1)] + (float) b * __half2float(dh[u]) * hd[t * (NE / 32) + (c >> 1)];
                }
                acc = wsum(acc);
                if (lane_id() == 0) G.res[0][(size_t) t * HCD + s * NE + r] = acc;
            }
        }
    }
}

// one full-attention + MoE layer over T rows (main attention layers and the MTP layer)
__device__ void run_attn_block(const Glob & G, const SM & S, int & pk, const Layer & L, int n, int T, int & cur, int & injp, int mode, const int * tok) {
    cur = mix_pre(G, S, T, cur, mode, injp, L.hca_norm, L.hca_down, L.hca_inj, G.inj[injp ^ 1], tok);
    injp ^= 1;
    gsync(G, pk, __LINE__);
    mix_post(G, S, T, cur, L.hca_up, nullptr, 0);
    gsync(G, pk, __LINE__);
    att_proj(G, S, T, L);
    gsync(G, pk, __LINE__);
    att_core(G, S, n, T, L);
    gsync(G, pk, __LINE__);
    mix_out(G, S, T, L.wo);
    gsync(G, pk, __LINE__);
}
__device__ void run_ffn(const Glob & G, const SM & S, int & pk, const Layer & L, int T, int & cur, int & injp) {
    cur = mix_pre(G, S, T, cur, 1, injp, L.hcf_norm, L.hcf_down, L.hcf_inj, G.inj[injp ^ 1], nullptr);
    injp ^= 1;
    gsync(G, pk, __LINE__);
    mix_post(G, S, T, cur, L.hcf_up, nullptr, 0);
    gsync(G, pk, __LINE__);
    ffn_router(G, S, T, L);
    gsync(G, pk, __LINE__);
    ffn_experts_gu(G, S, T, L);
    gsync(G, pk, __LINE__);
    ffn_down(G, S, T, L);
    gsync(G, pk, __LINE__);
}

__global__ void __launch_bounds__(NT, 1) fnx_kernel(Glob G, Step P) {
    const SM S = smem_layout();
    int pk = 0;
    for (int i = threadIdx.x; i < 256; i += NT) S.grid[i] = iq3xxs_grid[i];
    if (threadIdx.x < 16) S.kv4[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
    if (threadIdx.x < 128) S.sg64[threadIdx.x] = ksigns64[threadIdx.x];
    __syncthreads();
    if (P.mode == 2) {
        for (int k = 0; k < P.n; ++k) gsync(G, pk, 0);
        return;
    }
    if (G.prof && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); G.prof[0] = 0; G.prof[1] = t; }
    const int T = P.T;
    if (P.mode == 0) {
        int cur = 0, injp = 0, mode = 2;
        for (int il = 0; il < NL; ++il) {
            const Layer & L = G.L[il];
            if (il == PLE_LAYER) {
                ple_proj(G, S, T, L);
                gsync(G, pk, __LINE__);
                ple_gate(G, S, T, L, cur, injp);
                gsync(G, pk, __LINE__);
                cur = ple_apply(G, S, P.n, T, L, cur, injp);
                gsync(G, pk, __LINE__);
                mode = 0;
            }
            if (L.recr) {
                cur = mix_pre(G, S, T, cur, mode, injp, L.hca_norm, L.hca_down, L.hca_inj, G.inj[injp ^ 1], G.tok);
                injp ^= 1;
                gsync(G, pk, __LINE__);
                mix_post(G, S, T, cur, L.hca_up, nullptr, 0);
                gsync(G, pk, __LINE__);
                lin_proj(G, S, T, L);
                gsync(G, pk, __LINE__);
                lin_core(G, S, P, L);
                gsync(G, pk, __LINE__);
                mix_out(G, S, T, L.ssm_out);
                gsync(G, pk, __LINE__);
            } else {
                run_attn_block(G, S, pk, L, P.n, T, cur, injp, mode, G.tok);
            }
            run_ffn(G, S, pk, L, T, cur, injp);
            mode = 1;
        }
        cur = mix_pre(G, S, T, cur, 1, injp, G.head_norm, G.head_down, nullptr, nullptr, nullptr);
        gsync(G, pk, __LINE__);
        mix_post(G, S, T, cur, G.head_up, G.hmain, 0);
        gsync(G, pk, __LINE__);
        head_out(G, S, 0, T, 0);
        gsync(G, pk, __LINE__);
        if (blockIdx.x == 0 && threadIdx.x < T) G.ret[threadIdx.x] = best_tok(G, threadIdx.x);
        return;
    }
    // MTP: catch-up rows at positions n..n+T-1, then P.nchain drafts, each a one-row pass on the previous output
    int rows = T, pos = P.n;
    for (int step = 0; step < max(1, P.nchain); ++step) {
        const int * toks = G.mtp_tok;
        if (step > 0) {
            // previous draft token and h (copied into mtp_tok[0] / mtp_h[0] below)
            toks = G.mtp_tok;
        }
        mtp_in(G, S, rows, toks);
        gsync(G, pk, __LINE__);
        int cur = 0, injp = 0;
        run_attn_block(G, S, pk, *G.M, pos, rows, cur, injp, 0, nullptr);
        run_ffn(G, S, pk, *G.M, rows, cur, injp);
        if (P.nchain == 0) return;
        // head mixer; its residual row (the last one) is the h of the next draft step
        cur = mix_pre(G, S, rows, cur, 1, injp, G.m_hnorm_head, G.m_hdown, nullptr, nullptr, nullptr);
        gsync(G, pk, __LINE__);
        mix_post(G, S, rows, cur, G.m_hup, G.mtp_h, rows - 1);
        gsync(G, pk, __LINE__);
        head_out(G, S, rows - 1, 1, KMAX + step);
        gsync(G, pk, __LINE__);
        const int d = best_tok(G, KMAX + step);
        if (blockIdx.x == 0 && threadIdx.x == 0) { G.ret[step] = d; G.mtp_tok[0] = d; }
        gsync(G, pk, __LINE__);
        pos += rows;
        rows = 1;
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
static void open_cache(const std::string & path) {
    g_cr = fopen(path.c_str(), "rb");
    if (!g_cr) { g_cw = fopen((path + ".tmp").c_str(), "wb"); if (!g_cw) { perror(path.c_str()); exit(1); } }
    fprintf(stderr, "weight cache %s: %s\n", path.c_str(), g_cr ? "reading" : "writing");
}
static void close_cache(const std::string & path) {
    if (g_cw) { fclose(g_cw); g_cw = nullptr; rename((path + ".tmp").c_str(), path.c_str()); }
    if (g_cr) { fclose(g_cr); g_cr = nullptr; }
}

struct Engine {
    GGUF M;
    Glob G{};
    std::vector<Layer> hL;
    Layer hM{};
    int nsm = 0;
    const GT * ple = nullptr;
    const uint8_t * ple_data = nullptr;
    std::vector<uint64_t> ple_mul, ple_off, ple_voc;
    std::vector<std::pair<void *, size_t>> state_bufs;
    float * h_ple = nullptr;   // pinned [TMAX][NE]
    int * h_out = nullptr;     // pinned [KMAX]
    float * hzero = nullptr;   // device HCD zeros
    bool has_mtp = false;
    int rpar = 0;
    double host_ple_s = 0.0;
    std::map<int, double> prof_ns; std::map<int, int> prof_cnt; int prof_launches = 0;
    unsigned long long * h_prof = nullptr;

    // one layer, in the exact upload order of the v1 engine (the weight cache depends on it)
    void load_layer(Layer & L, int il, bool mtp) {
        auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
        L.recr = !mtp && ((il + 1) % 4) != 0;
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
            L.conv_ring = dalloc<float>((size_t) CONV_RING * CONVCH);
            state_bufs.push_back({ L.ssm_state, (size_t) GV * DS * DS * 4 });
            state_bufs.push_back({ L.conv_ring, (size_t) CONV_RING * CONVCH * 4 });
            for (int p = 0; p < 2; ++p) {
                L.rk[p] = dalloc<float>((size_t) TMAX * KDIM); L.rv[p] = dalloc<float>((size_t) TMAX * VDIM);
                L.rb[p] = dalloc<float>(TMAX * GV); L.rg[p] = dalloc<float>(TMAX * GV);
            }
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
        if (il == PLE_LAYER && !mtp) {
            L.ple_k = up_q8(M, b("ple_key.weight")); L.ple_v = up_q8(M, b("ple_value.weight"));
            L.ple_nk = up_f32(M, b("ple_norm_key.weight")); L.ple_nq = up_f32(M, b("ple_norm_query.weight"));
            L.ple_nc = up_f32(M, b("ple_norm_conv.weight")); L.ple_conv = up_f32(M, b("ple_conv1d.weight"));
        }
    }

    void load(const std::vector<std::string> & shards, const std::string & mtp_file, const std::string & cache,
              const std::string & cache2, const std::string & ple_file) {
        for (auto & s : shards) M.open(s);
        if (!mtp_file.empty()) { M.open(mtp_file); has_mtp = true; }
        ple = &M.get("per_layer_token_embd.weight");
        ple_data = ple->data;
        if (!ple_file.empty()) {
            int fd = ::open(ple_file.c_str(), O_RDONLY);
            if (fd < 0) { perror(ple_file.c_str()); exit(1); }
            struct stat st; fstat(fd, &st);
            if ((size_t) st.st_size != ple->nbytes) { fprintf(stderr, "%s: wrong size\n", ple_file.c_str()); exit(1); }
            void * m = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            madvise(m, st.st_size, MADV_RANDOM);
            ple_data = (const uint8_t *) m;
        }
        ple_mul = { 23703573157769ull, 20109073645365ull, 8052911324071ull };
        ple_off = { 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455,
                    200000548, 220000655, 240000802, 260000955, 280001114, 300001275 };
        ple_voc = { 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
                    20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 };
        auto t0 = std::chrono::steady_clock::now();
        if (!cache.empty()) open_cache(cache);
        hL.resize(NL);
        for (int il = 0; il < NL; ++il) {
            load_layer(hL[il], il, false);
            fprintf(stderr, "\rload layer %2d/%d  %.1f GiB on device", il + 1, NL, g_dev_bytes / 1073741824.0);
        }
        G.head_down = up_q8(M, "output_hc_down.weight");
        G.head_up = up_q8(M, "output_hc_up.weight");
        G.head_norm = up_f32(M, "output_hc_norm.weight");
        G.out = up_q8(M, "output.weight");
        if (!cache.empty()) close_cache(cache);
        // v2 additions: device token embedding, the MTP layer
        if (!cache2.empty()) open_cache(cache2);
        G.embq = up_q8(M, "token_embd.weight");
        if (has_mtp) {
            load_layer(hM, NL, true);
            G.m_eh = up_q8(M, "blk.48.nextn.eh_proj.weight");
            G.m_enorm = up_f32(M, "blk.48.nextn.enorm.weight");
            G.m_hnorm = up_f32(M, "blk.48.nextn.hnorm.weight");
            G.m_hnorm_head = up_f32(M, "blk.48.nextn.hc_head_norm.weight");
            G.m_hdown = up_q8(M, "blk.48.nextn.hc_head_down.weight");
            G.m_hup = up_q8(M, "blk.48.nextn.hc_head_up.weight");
        }
        if (!cache2.empty()) close_cache(cache2);
        G.L = dalloc<Layer>(NL);
        CK(cudaMemcpy(G.L, hL.data(), NL * sizeof(Layer), cudaMemcpyHostToDevice));
        if (has_mtp) {
            G.M = dalloc<Layer>(1);
            CK(cudaMemcpy(G.M, &hM, sizeof(Layer), cudaMemcpyHostToDevice));
        }
        G.res[0] = dalloc<float>((size_t) TMAX * HCD); G.res[1] = dalloc<float>((size_t) TMAX * HCD);
        G.blk_out = dalloc<float>(TMAX * NE);
        G.inj[0] = dalloc<float>(TMAX * HC * HC); G.inj[1] = dalloc<float>(TMAX * HC * HC);
        G.lo = dalloc<float>(TMAX * HC * LR);
        G.xn = dalloc<float>((size_t) TMAX * HCD);
        G.mixed = dalloc<float>(TMAX * NE);
        G.proj = dalloc<float>((size_t) TMAX * PROJ);
        G.core = dalloc<float>((size_t) TMAX * VDIM);
        G.rlog = dalloc<float>(TMAX * (NEXP + 1));
        G.sh_h = dalloc<float>(TMAX * 2 * FF);
        G.exp_h = dalloc<float>(TMAX * (NUSED + 1) * FF);
        G.sel_id = dalloc<int>(TMAX * NUSED);
        G.sel_w = dalloc<float>(TMAX * (NUSED + 1));
        G.ple_kv = dalloc<float>((size_t) TMAX * (HCD + NE));
        G.ple_g = dalloc<float>(TMAX * 2 * HC);
        G.ple_hist = dalloc<float>((size_t) PLE_RING * HCD);
        state_bufs.push_back({ G.ple_hist, (size_t) PLE_RING * HCD * 4 });
        G.ple_in = dalloc<float>(TMAX * NE);
        G.tok = dalloc<int>(TMAX);
        G.hmain = dalloc<float>((size_t) TMAX * HCD);
        G.mtp_tok = dalloc<int>(TMAX);
        G.mtp_h = dalloc<float>((size_t) TMAX * HCD);
        G.best = dalloc<unsigned long long>(KMAX + TMAX);
        G.ret = dalloc<int>(KMAX);
        G.bar_count = dalloc<unsigned>(1);
        hzero = dalloc<float>(HCD);
        CK(cudaMemset(hzero, 0, HCD * 4));
        if (getenv("MK_PROFILE")) { G.prof = dalloc<unsigned long long>(8192); CK(cudaMallocHost(&h_prof, 8192 * 8)); CK(cudaMemset(G.prof, 0, 8192 * 8)); }
        CK(cudaMallocHost(&h_ple, TMAX * NE * 4)); CK(cudaMallocHost(&h_out, KMAX * 4));
        CK(cudaFuncSetAttribute(fnx_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SM_BYTES));
        cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
        int per = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fnx_kernel, NT, SM_BYTES));
        if (per < 1) { fprintf(stderr, "kernel does not fit one block per SM\n"); exit(1); }
        nsm = pr.multiProcessorCount;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "\nloaded in %.1f s, %.2f GiB on device, %d SMs, smem %zu B, MTP %s\n", sec, g_dev_bytes / 1073741824.0,
                nsm, SM_BYTES, has_mtp ? "yes" : "no");
        reset();
    }

    void reset() {
        for (auto & b : state_bufs) CK(cudaMemset(b.first, 0, b.second));
        rpar = 0;
    }

    // PLE rows of the token at index p of ctx (EOS resets the window)
    void ple_rows(const std::vector<int> & ctx, int p, float * dst) {
        int64_t c[3];
        c[0] = ctx[p];
        bool cut = false;
        for (int s = 1; s < 3; ++s) {
            const int64_t t = (cut || p - s < 0) ? -1 : ctx[p - s];
            cut = cut || t < 0 || t == EOS_TOK;
            c[s] = cut ? EOS_TOK : t;
        }
        const size_t pr = ggml_row_size((ggml_type) ple->type, PLE_HD);
        const auto * tp = ggml_get_type_traits((ggml_type) ple->type);
        int32_t idx[PLE_HEADS];
        for (int ng = 2; ng <= 3; ++ng) {
            uint64_t mixed = (uint64_t) c[0] * ple_mul[0];
            for (int j = 1; j < ng; ++j) mixed ^= (uint64_t) c[j] * ple_mul[j];
            for (int g = 0; g < 8; ++g) { const int h = (ng - 2) * 8 + g; idx[h] = (int32_t) (mixed % ple_voc[h] + ple_off[h]); }
        }
#pragma omp parallel for num_threads(PLE_HEADS) schedule(static, 1)
        for (int h = 0; h < PLE_HEADS; ++h) tp->to_float(ple_data + (size_t) idx[h] * pr, dst + h * PLE_HD, PLE_HD);
    }

    void launch(Step P) {
        CK(cudaMemsetAsync(G.bar_count, 0, 4));
        CK(cudaMemsetAsync(G.best, 0, 8 * (KMAX + TMAX)));
        void * args[] = { &G, &P };
        CK(cudaLaunchCooperativeKernel((void *) fnx_kernel, dim3(nsm), dim3(NT), args, SM_BYTES, 0));
        if (G.prof) {
            CK(cudaStreamSynchronize(0));
            CK(cudaMemcpy(h_prof, G.prof, 8192 * 8, cudaMemcpyDeviceToHost));
            for (int k = 1; k < 4096 && h_prof[2 * k] != 0; ++k) {
                const int key = P.mode * 100000 + (int) h_prof[2 * k];
                prof_ns[key] += (double) (h_prof[2 * k + 1] - h_prof[2 * k - 1]);
                prof_cnt[key]++;
            }
            CK(cudaMemset(G.prof, 0, 8192 * 8));
            ++prof_launches;
        }
    }

    // main pass: rows ctx[n .. n+T) at positions n.., committing `replay` rows of the previous pass first
    void main_pass(const std::vector<int> & ctx, int n, int T, int replay, int * targets) {
        if (n + T > MAXCTX) { fprintf(stderr, "context full (%d)\n", MAXCTX); exit(1); }
        const auto a = std::chrono::steady_clock::now();
        for (int t = 0; t < T; ++t) ple_rows(ctx, n + t, h_ple + t * NE);
        host_ple_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        CK(cudaMemcpyAsync(G.ple_in, h_ple, T * NE * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpyAsync(G.tok, ctx.data() + n, T * 4, cudaMemcpyHostToDevice));
        Step P{ 0, n, T, replay, rpar, 0 };
        launch(P);
        rpar ^= 1;
        CK(cudaMemcpyAsync(h_out, G.ret, T * 4, cudaMemcpyDeviceToHost));
        CK(cudaStreamSynchronize(0));
        for (int t = 0; t < T; ++t) targets[t] = h_out[t];
    }

    // MTP pass: rows (tok[r], h[r]) at positions pos.., then nchain drafts
    void mtp_pass(const int * tok, const float * const * h, int rows, int pos, int nchain, int * drafts) {
        CK(cudaMemcpyAsync(G.mtp_tok, tok, rows * 4, cudaMemcpyHostToDevice));
        for (int r = 0; r < rows; ++r) CK(cudaMemcpyAsync(G.mtp_h + (size_t) r * HCD, h[r], HCD * 4, cudaMemcpyDeviceToDevice));
        Step P{ 1, pos, rows, 0, 0, nchain };
        launch(P);
        if (nchain > 0) {
            CK(cudaMemcpyAsync(h_out, G.ret, nchain * 4, cudaMemcpyDeviceToHost));
            CK(cudaStreamSynchronize(0));
            for (int k = 0; k < nchain; ++k) drafts[k] = h_out[k];
        }
    }

    double bench_barrier(int k) {
        Step P{ 2, k, 1, 0, 0, 0 };
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        CK(cudaMemsetAsync(G.bar_count, 0, 4));
        cudaEventRecord(a);
        void * args[] = { &G, &P };
        CK(cudaLaunchCooperativeKernel((void *) fnx_kernel, dim3(nsm), dim3(NT), args, SM_BYTES, 0));
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b);
        return 1e3 * ms / k;
    }

    struct GenStats { int cycles = 0, accepted = 0, drafted = 0; double prefill_s = 0, decode_s = 0; std::vector<int> acc_hist; };

    // greedy generation with k MTP drafts per cycle (k = 0: plain decode). Returns generated tokens.
    std::vector<int> generate(const std::vector<int> & prompt, int n_gen, int k, const llama_vocab * vocab, GenStats & st) {
        reset();
        const bool adaptive = k < 0;
        k = std::abs(k);
        k = std::min(k, TMAX - 1);
        if (k > 0 && !has_mtp) { fprintf(stderr, "drafts need --mtp\n"); exit(1); }
        st = GenStats{};
        st.acc_hist.assign(k + 1, 0);
        std::vector<int> ctx(prompt);
        const int P = (int) prompt.size();
        int tg[TMAX], drafts[KMAX];
        int kc = std::min(k, TMAX - 1);
        const auto t0 = std::chrono::steady_clock::now();
        if (k > 0) { const float * hz[1] = { hzero }; mtp_pass(&ctx[0], hz, 1, 0, 0, nullptr); }
        int replay = 0;
        for (int p = 0; p < P; ++p) {
            main_pass(ctx, p, 1, replay, tg);
            replay = 1;
            if (k > 0) {
                const float * hm[1] = { G.hmain };
                if (p < P - 1) mtp_pass(&ctx[p + 1], hm, 1, p + 1, 0, nullptr);
                else           mtp_pass(&tg[0], hm, 1, P, k, drafts);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::vector<int> gen;
        int g = tg[0], n = P;
        gen.push_back(g);
        while ((int) gen.size() < n_gen && !llama_vocab_is_eog(vocab, g)) {
            ctx.resize(n);
            ctx.push_back(g);
            if (k == 0) {
                main_pass(ctx, n, 1, replay, tg);
                replay = 1; n += 1; g = tg[0]; gen.push_back(g);
                ++st.cycles; st.acc_hist[0]++;
                continue;
            }
            for (int j = 0; j < kc; ++j) ctx.push_back(drafts[j]);
            main_pass(ctx, n, kc + 1, replay, tg);
            int a = 0;
            while (a < kc && ctx[n + 1 + a] == tg[a]) ++a;
            ++st.cycles; st.accepted += a; st.drafted += kc; st.acc_hist[a]++;
            bool stop = false;
            for (int j = 0; j < a && !stop; ++j) { gen.push_back(ctx[n + 1 + j]); stop = llama_vocab_is_eog(vocab, ctx[n + 1 + j]) || (int) gen.size() >= n_gen; }
            if (stop) break;
            gen.push_back(tg[a]);
            replay = a + 1;
            // MTP catch-up rows n+1 .. n+a+1 (the accepted drafts, then the bonus token), then k drafts
            int mt[TMAX]; const float * mh[TMAX];
            for (int j = 0; j < a; ++j) mt[j] = ctx[n + 1 + j];
            mt[a] = tg[a];
            for (int j = 0; j <= a; ++j) mh[j] = G.hmain + (size_t) j * HCD;
            ctx.resize(n + 1 + a);
            g = tg[a];
            n += a + 1;
            if ((int) gen.size() >= n_gen || llama_vocab_is_eog(vocab, g)) break;
            // adaptive depth: one draft after a full miss, the maximum after a full hit, else one past the hits
            if (adaptive) kc = a == 0 ? 1 : (a == kc ? k : std::min(k, a + 1));
            mtp_pass(mt, mh, a + 1, n - a, kc, drafts);
        }
        const auto t2 = std::chrono::steady_clock::now();
        st.prefill_s = std::chrono::duration<double>(t1 - t0).count();
        st.decode_s = std::chrono::duration<double>(t2 - t1).count();
        if ((int) gen.size() > n_gen) gen.resize(n_gen);
        return gen;
    }
};

// ================================================================ host: CLI
static std::vector<int> read_ids(const std::string & path) {
    std::vector<int> v;
    FILE * f = fopen(path.c_str(), "r");
    if (!f) { perror(path.c_str()); exit(1); }
    int x;
    while (fscanf(f, " %d", &x) == 1) v.push_back(x);
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    std::vector<std::string> shards;
    std::string prompt = "Write a quicksort in C.", score_ids, cache, cache2, ple_file, mtp_file, bench_file, dlist = "0";
    int n_gen = 128, n_prompt_score = -1, reps = 1, k = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "-m") shards.push_back(nx());
        else if (a == "-p") prompt = nx();
        else if (a == "-n") n_gen = atoi(nx().c_str());
        else if (a == "-d") { dlist = nx(); k = atoi(dlist.c_str()); }
        else if (a == "--mtp") mtp_file = nx();
        else if (a == "--score") score_ids = nx();
        else if (a == "--score-prompt") n_prompt_score = atoi(nx().c_str());
        else if (a == "--cache") cache = nx();
        else if (a == "--cache2") cache2 = nx();
        else if (a == "--ple-file") ple_file = nx();
        else if (a == "--bench") bench_file = nx();
        else if (a == "--reps") reps = atoi(nx().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (shards.empty()) { fprintf(stderr, "usage: %s -m shard ... [--mtp mtp.gguf] [-d k | -d 0,2,3] [-p text] [-n N] [--bench prompts.tsv]\n", argv[0]); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * vm = llama_model_load_from_file(shards[0].c_str(), mp);
    if (!vm) { fprintf(stderr, "tokenizer load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(vm);
    auto piece = [&](int t) { char buf[256]; int kk = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, false); return std::string(buf, kk > 0 ? kk : 0); };
    auto chat = [&](const std::string & p) {
        const std::string text = "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        std::vector<int> tk(text.size() + 16);
        tk.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), tk.data(), (int) tk.size(), false, true));
        return tk;
    };

    Engine E;
    E.load(shards, mtp_file, cache, cache2, ple_file);

    if (getenv("MK_BARRIER")) for (int r = 0; r < 3; ++r) fprintf(stderr, "barrier: %.3f us each (10000 in one launch)\n", E.bench_barrier(10000));

    if (!score_ids.empty()) {
        const std::vector<int> ids = read_ids(score_ids);
        const int P = n_prompt_score;
        int agree = 0, tot = 0, tg[TMAX];
        std::vector<int> miss;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            E.main_pass(ids, (int) i, 1, i > 0 ? 1 : 0, tg);
            if ((int) i >= P - 1) { ++tot; if (tg[0] == ids[i + 1]) ++agree; else miss.push_back((int) i + 1); }
        }
        printf("score: %d / %d positions agree (%.2f%%)\n", agree, tot, 100.0 * agree / tot);
        for (size_t q = 0; q < miss.size() && q < 20; ++q) printf("  miss at %d\n", miss[q]);
        return 0;
    }

    std::vector<int> ks;
    for (size_t s = 0; s <= dlist.size();) { size_t e = dlist.find(',', s); if (e == std::string::npos) e = dlist.size(); ks.push_back(atoi(dlist.substr(s, e - s).c_str())); s = e + 1; }

    std::vector<std::pair<std::string, std::string>> items;
    if (!bench_file.empty()) {
        FILE * f = fopen(bench_file.c_str(), "r");
        if (!f) { perror(bench_file.c_str()); return 1; }
        char line[8192];
        while (fgets(line, sizeof line, f)) {
            std::string l(line);
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
            const size_t tab = l.find('\t');
            if (tab != std::string::npos) items.push_back({ l.substr(0, tab), l.substr(tab + 1) });
        }
        fclose(f);
    } else {
        items.push_back({ "cli", prompt });
    }
    for (int r = 0; r < reps; ++r) {
        for (auto & it : items) {
            for (int kk : ks) {
                if (getenv("MK_COOL")) {
                    // wait for the card to cool before each generation (repository rule: stop at 80 C)
                    const int rc = system("until [ $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -le 60 ]; do sleep 2; done");
                    (void) rc;
                }
                const std::vector<int> tk = chat(it.second);
                Engine::GenStats st;
                const std::vector<int> g = E.generate(tk, n_gen, kk, vocab, st);
                const double tps = (g.size() - 1) / st.decode_s;
                std::string hist;
                for (size_t q = 0; q < st.acc_hist.size(); ++q) hist += (q ? "," : "") + std::to_string(st.acc_hist[q]);
                printf("{\"name\": \"%s\", \"rep\": %d, \"d\": %d, \"prompt_tokens\": %zu, \"gen_tokens\": %zu, \"prefill_s\": %.4f, "
                       "\"decode_s\": %.4f, \"decode_tok_s\": %.3f, \"cycles\": %d, \"accepted\": %d, \"drafted\": %d, \"acc_hist\": [%s], \"ids\": [",
                       it.first.c_str(), r, kk, tk.size(), g.size(), st.prefill_s, st.decode_s, tps, st.cycles, st.accepted, st.drafted, hist.c_str());
                for (size_t q = 0; q < g.size(); ++q) printf("%d%s", g[q], q + 1 < g.size() ? "," : "");
                printf("]}\n");
                fflush(stdout);
                fprintf(stderr, "%s rep %d d=%d: decode %.2f tok/s (%zu tok, %d cycles, %.2f tok/cycle, accept %.1f%%)\n", it.first.c_str(), r, kk, tps,
                        g.size(), st.cycles, (double) (g.size() - 1) / std::max(1, st.cycles),
                        st.drafted ? 100.0 * st.accepted / st.drafted : 0.0);
                if (bench_file.empty()) { std::string s; for (int t : g) s += piece(t); fprintf(stderr, "%s\n", s.c_str()); }
            }
        }
    }
    fprintf(stderr, "host PLE gather: %.3f s total\n", E.host_ple_s);
    if (!E.prof_ns.empty()) {
        double tot = 0; for (auto & kv : E.prof_ns) tot += kv.second;
        fprintf(stderr, "profile: %d launches, %.3f ms in phases\n", E.prof_launches, tot / 1e6);
        for (auto & kv : E.prof_ns) fprintf(stderr, "  mode %d line %4d  x%-6d %10.1f us total (%.2f us each)\n", kv.first / 100000, kv.first % 100000,
                                            E.prof_cnt[kv.first], kv.second / 1e3, kv.second / E.prof_cnt[kv.first] / 1e3);
    }
    llama_model_free(vm);
    return 0;
}
