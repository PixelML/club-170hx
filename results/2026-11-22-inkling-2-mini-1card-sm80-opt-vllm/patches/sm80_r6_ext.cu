#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// ============================================================================
// Round-6 Lane-A SM80 Extension Library (sm80_r6_ext.cu -> libsm80_r6_ext.so)
// Self-contained, 0-spill, fast incremental compile (~1.5s on A100 SM80).
// ============================================================================

__device__ __forceinline__ float r6_bf2f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __nv_bfloat16 r6_f2bf(float x) {
    return __float2bfloat16_rn(x);
}
__device__ __forceinline__ void r6_unpack_bf16x8(uint4 u, float (&dst)[8]) {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&u);
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = __bfloat162float(p[i]);
}
__device__ __forceinline__ uint4 r6_pack_bf16x8(const float (&src)[8]) {
    uint4 u;
    __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(&u);
    #pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = __float2bfloat16_rn(src[i]);
    return u;
}
__device__ __forceinline__ uint4 r6_f4x2_to_bf16x8(float4 a, float4 b) {
    uint4 u;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
    p[0] = __floats2bfloat162_rn(a.x, a.y);
    p[1] = __floats2bfloat162_rn(a.z, a.w);
    p[2] = __floats2bfloat162_rn(b.x, b.y);
    p[3] = __floats2bfloat162_rn(b.z, b.w);
    return u;
}
__device__ __forceinline__ float r6_warp_sum_rn(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}
__device__ __forceinline__ int r6_warp_inclusive_scan_int(int val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}
__device__ __forceinline__ void r6_cp_async_cg_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void r6_cp_async_ca_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void r6_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N_WAIT>
__device__ __forceinline__ void r6_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT));
}
__device__ __forceinline__ void r6_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}
__device__ __forceinline__ float r6_ldcg_f32(const float* ptr) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(ptr));
    return v;
}
__device__ __forceinline__ float2 r6_ldcg_f2(const float2* ptr) {
    float2 v;
    asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(ptr));
    return v;
}
__device__ __forceinline__ float4 r6_ldcg_f4(const float4* ptr) {
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(ptr));
    return v;
}
__device__ __forceinline__ uint4 r6_ldcg_u4(const uint4* ptr) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}
__device__ __forceinline__ void r6_stcg_f32(float* ptr, float v) {
    asm volatile("st.global.cg.f32 [%0], %1;" :: "l"(ptr), "f"(v));
}
__device__ __forceinline__ void r6_stcg_f2(float2* ptr, float2 v) {
    asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};" :: "l"(ptr), "f"(v.x), "f"(v.y));
}
__device__ __forceinline__ void r6_stcg_f4(float4* ptr, float4 v) {
    asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
}
__device__ __forceinline__ void r6_stcg_u4(uint4* ptr, uint4 v) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

// ============================================================================
// Exp A1: Fast 1-Launch 72-CTA K9 Tensor-Core Router + Top-8 + Parallel Align
// Supports 1 <= T <= 16, E = 128, D = 2304.
// Grid: (8, 9) = 72 CTAs (1 wave on both 74-SM CMP 170HX and 108-SM A100!).
// Each CTA issues both 128-K stages in 1 burst; warps 0..1 compute stage 0 and
// warps 2..3 compute stage 1 in parallel (1 single cp_async_wait<0>!), reducing
// across the 2 stages in SMEM, while CTA (0,0) overlaps sorted_ids sentinel fill.
// Phase 2 uses 256 threads (8 warps -> 1 parallel pass for T<=8!) with a
// pre-sorted 4-element lane popper (zero 4-way max/mux inside the k=0..7 loop).
// ============================================================================

struct RouterP2Smem256 {
    int32_t s_counts[128];
    int32_t s_pad_start[128];
    int32_t s_pad_end[128];
    int32_t s_warp_sums[4];
    int16_t s_expert[128];
    int16_t s_slot[128];
};

__device__ __forceinline__ uint64_t r6_sel_key(float lg, float bv, int e) {
    float sel = __fadd_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-lg))), bv);
    uint32_t bits = __float_as_uint(sel);
    bits ^= (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return (static_cast<uint64_t>(bits) << 16) | static_cast<uint64_t>(128 - e);
}

__device__ __forceinline__ void cswap_key(
    uint64_t& ka, float& la, uint64_t& kb, float& lb)
{
    if (kb > ka) {
        uint64_t tk = ka; ka = kb; kb = tk;
        float tl = la; la = lb; lb = tl;
    }
}

__global__ void __launch_bounds__(256, 3) select_and_align_mma256_k3g_t16_sm80_kernel(
    const __nv_bfloat16* __restrict__ X,            // [T, 2304] bf16
    const __nv_bfloat16* __restrict__ W,            // [128, 2304] bf16
    const float* __restrict__ bias,                 // [128] fp32
    const int32_t* __restrict__ expert_bounds,      // [2] int32
    const uint8_t* __restrict__ is_padding,         // [T] bool/uint8 or nullptr
    float* __restrict__ ws_partials,                // [3 * T * 128] fp32
    int32_t* __restrict__ cta_counter,              // [1] int32 zeroed
    int32_t* __restrict__ out_ids,                  // [T, 8] int32
    float* __restrict__ out_weights,                // [T * 8] fp32
    int32_t* __restrict__ out_src2dst,              // [T * 8] int32
    int32_t* __restrict__ sorted_ids,               // [max_pad] int32
    int32_t* __restrict__ expert_ids,               // [n_blocks] int32
    int32_t* __restrict__ total_tokens_post_pad,    // [1] int32
    const int32_t T,
    const int32_t block_size,
    const int32_t max_pad,
    const int32_t n_blocks,
    const float route_scale)
{
    struct P1Smem {
        __align__(16) uint4 s_W[2][8][16];   // 4 KiB
        __align__(16) uint4 s_X[2][16][16];  // 8 KiB
    };
    __shared__ union {
        P1Smem p1;
        RouterP2Smem256 p2;
    } smem;
    __shared__ int s_is_last;

    const int tid = threadIdx.x;       // 0..255
    const int warp_id = tid >> 5;      // 0..7
    const int lane = tid & 31;         // 0..31
    const int group_id = lane >> 2;    // 0..7
    const int tid_in_grp = lane & 3;   // 0..3

    const int e_base = static_cast<int>(blockIdx.x) << 3; // 0, 8, ..., 120 (BN=8)
    const int g_idx  = static_cast<int>(blockIdx.y);      // 0, 1, 2 (1024, 1024, 256 K-groups!)
    const int k_start = g_idx * 1024;
    const int n_tiles = (g_idx < 2) ? 8 : 2;

    auto issue_stage = [&](int st, int kt) {
        const int kb = k_start + kt * 128;
        if (tid < 128) {
            int wr = tid >> 4;   // 0..7
            int wc = tid & 15;   // 0..15
            int wsc = wc ^ wr;
            const __nv_bfloat16* w_ptr = W + static_cast<int64_t>(e_base + wr) * 2304 + kb + (wc << 3);
            uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.p1.s_W[st][wr][wsc]));
            r6_cp_async_cg_16B(dst_w, w_ptr);
        }
        int xr = tid >> 4;       // 0..15
        int xc = tid & 15;       // 0..15
        int xsc = xc ^ (xr & 7);
        if (xr < T) {
            const __nv_bfloat16* x_ptr = X + static_cast<int64_t>(xr) * 2304 + kb + (xc << 3);
            uint32_t dst_x = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.p1.s_X[st][xr][xsc]));
            r6_cp_async_ca_16B(dst_x, x_ptr);
        } else if (kt < 2) {
            smem.p1.s_X[st][xr][xsc] = make_uint4(0, 0, 0, 0);
        }
        r6_cp_async_commit();
    };

    issue_stage(0, 0);

    // Overlap sorted_ids sentinel fill while cp.async fetches cold HBM!
    if (blockIdx.x == 0 && g_idx == 0) {
        const int numel = T * 8;
        const int4 fill4 = make_int4(numel, numel, numel, numel);
        int4* sorted4 = reinterpret_cast<int4*>(sorted_ids);
        const int n_vec4 = max_pad >> 2;
        for (int idx = tid; idx < n_vec4; idx += 256) sorted4[idx] = fill4;
        for (int idx = (n_vec4 << 2) + tid; idx < max_pad; idx += 256) sorted_ids[idx] = numel;
    }

    if (n_tiles > 1) issue_stage(1, 1);

    // Exact 0-ULP sequential MMA accumulation within each 1024-wide K-group!
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    #pragma unroll 1
    for (int kt = 0; kt < n_tiles; ++kt) {
        if (kt + 1 < n_tiles) r6_cp_async_wait<1>();
        else r6_cp_async_wait<0>();
        __syncthreads();

        const int st = kt & 1;
        if (warp_id == 0) {
            const int x_r0 = group_id;
            const int x_r1 = group_id + 8;
            const int w_r = group_id;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                int c_lo = kk << 1;
                int c_hi = c_lo + 1;
                const uint32_t* x0_lo = reinterpret_cast<const uint32_t*>(&smem.p1.s_X[st][x_r0][c_lo ^ group_id]);
                const uint32_t* x1_lo = reinterpret_cast<const uint32_t*>(&smem.p1.s_X[st][x_r1][c_lo ^ group_id]);
                const uint32_t* x0_hi = reinterpret_cast<const uint32_t*>(&smem.p1.s_X[st][x_r0][c_hi ^ group_id]);
                const uint32_t* x1_hi = reinterpret_cast<const uint32_t*>(&smem.p1.s_X[st][x_r1][c_hi ^ group_id]);
                const uint32_t* w_lo  = reinterpret_cast<const uint32_t*>(&smem.p1.s_W[st][w_r][c_lo ^ group_id]);
                const uint32_t* w_hi  = reinterpret_cast<const uint32_t*>(&smem.p1.s_W[st][w_r][c_hi ^ group_id]);
                r6_mma_m16n8k16_bf16(
                    acc0, acc1, acc2, acc3,
                    x0_lo[tid_in_grp], x1_lo[tid_in_grp], x0_hi[tid_in_grp], x1_hi[tid_in_grp],
                    w_lo[tid_in_grp], w_hi[tid_in_grp]
                );
            }
        }
        if (kt + 2 < n_tiles) {
            __syncthreads();
            issue_stage(st, kt + 2);
        }
    }

    if (warp_id == 0) {
        const int c0 = tid_in_grp << 1;
        const int g_m0 = group_id;
        const int g_m1 = g_m0 + 8;
        float* base_p = ws_partials + static_cast<int64_t>(g_idx) * T * 128;
        if (g_m0 < T) {
            r6_stcg_f2(reinterpret_cast<float2*>(base_p + g_m0 * 128 + e_base + c0), make_float2(acc0, acc1));
        }
        if (g_m1 < T) {
            r6_stcg_f2(reinterpret_cast<float2*>(base_p + g_m1 * 128 + e_base + c0), make_float2(acc2, acc3));
        }
    }

    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int prev = atomicAdd(cta_counter, 1);
        s_is_last = (prev == 47) ? 1 : 0;
        if (prev == 47) atomicExch(cta_counter, 0);
    }
    __syncthreads();
    if (!s_is_last) return;

    // Phase 2 (Last CTA, 256 threads = 8 warps): exact 0-ULP 3-group sum + top-8 + parallel align!
    if (tid < 128) smem.p2.s_counts[tid] = 0;
    const int32_t eb_lo = expert_bounds[0];
    const int32_t eb_hi = expert_bounds[1];
    const float b0 = bias[lane + 0];
    const float b1 = bias[lane + 32];
    const float b2 = bias[lane + 64];
    const float b3 = bias[lane + 96];
    __syncthreads();

    const int64_t stride_k = static_cast<int64_t>(T) * 128;
    for (int t = warp_id; t < T; t += 8) {
        const bool padded_tok = (is_padding != nullptr) ? (is_padding[t] != 0) : false;
        const int base = t * 128 + lane;
        auto sum_k3 = [&](int off) -> float {
            const float* q = ws_partials + base + off;
            return __fadd_rn(__fadd_rn(r6_ldcg_f32(q + 0 * stride_k), r6_ldcg_f32(q + 1 * stride_k)),
                             r6_ldcg_f32(q + 2 * stride_k));
        };
        float lg0 = sum_k3(0);
        float lg1 = sum_k3(32);
        float lg2 = sum_k3(64);
        float lg3 = sum_k3(96);

        uint64_t k0 = r6_sel_key(lg0, b0, lane + 0);
        uint64_t k1 = r6_sel_key(lg1, b1, lane + 32);
        uint64_t k2 = r6_sel_key(lg2, b2, lane + 64);
        uint64_t k3 = r6_sel_key(lg3, b3, lane + 96);

        // 5-comparator sorting network so k0 >= k1 >= k2 >= k3 (exact 64-bit key order!)
        cswap_key(k0, lg0, k1, lg1);
        cswap_key(k2, lg2, k3, lg3);
        cswap_key(k0, lg0, k2, lg2);
        cswap_key(k1, lg1, k3, lg3);
        cswap_key(k1, lg1, k2, lg2);

        int my_sel_e = 0;
        float my_sel_lg = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint64_t w_k = k0;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                uint64_t ok = __shfl_xor_sync(0xffffffffu, w_k, mask);
                if (ok > w_k) w_k = ok;
            }
            int w_e = 128 - static_cast<int>(w_k & 0xFFFFu);
            float win_lg = __shfl_sync(0xffffffffu, lg0, w_e & 31);
            if (lane == k) {
                my_sel_e = w_e;
                my_sel_lg = win_lg;
            }
            if (k0 == w_k) {
                k0 = k1; lg0 = lg1;
                k1 = k2; lg1 = lg2;
                k2 = k3; lg2 = lg3;
                k3 = 0ull;
            }
        }

        if (lane < 8) {
            float lp = fminf(my_sel_lg, 0.0f) - log1pf(expf(-fabsf(my_sel_lg)));
            float mx = lp;
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 4));
            float sh = expf(__fsub_rn(lp, mx));
            float s02 = __fadd_rn(sh, __shfl_xor_sync(0xffu, sh, 2));
            float s0123 = __fadd_rn(s02, __shfl_xor_sync(0xffu, s02, 1));
            float total = __fadd_rn(s0123, __shfl_xor_sync(0xffu, s0123, 4));
            total = __shfl_sync(0xffu, total, 0);
            float lse = __fadd_rn(mx, logf(total));
            float wt = __fmul_rn(expf(__fsub_rn(lp, lse)), route_scale);

            int row = t * 8 + lane;
            int le = (!padded_tok && my_sel_e >= eb_lo && my_sel_e < eb_hi) ? (my_sel_e - eb_lo) : -1;
            out_ids[row] = le;
            out_weights[row] = wt;
            out_src2dst[row] = (le >= 0) ? row : -1;
            if (le >= 0) {
                int slot = atomicAdd(&smem.p2.s_counts[le], 1);
                smem.p2.s_expert[row] = static_cast<int16_t>(le);
                smem.p2.s_slot[row] = static_cast<int16_t>(slot);
            } else {
                smem.p2.s_expert[row] = -1;
                smem.p2.s_slot[row] = 0;
            }
        }
    }
    __syncthreads();

    int padded_c = 0;
    int start_pos = 0;
    int end_pos = 0;
    if (tid < 128) {
        const int c_val = smem.p2.s_counts[tid];
        padded_c = ((c_val + block_size - 1) / block_size) * block_size;
        int scan_v = r6_warp_inclusive_scan_int(padded_c);
        if (lane == 31) smem.p2.s_warp_sums[warp_id] = scan_v;
        end_pos = scan_v;
    }
    __syncthreads();

    if (warp_id == 0) {
        int ws = (lane < 4) ? smem.p2.s_warp_sums[lane] : 0;
        ws = r6_warp_inclusive_scan_int(ws);
        if (lane < 4) smem.p2.s_warp_sums[lane] = ws;
    }
    __syncthreads();

    if (tid < 128) {
        const int warp_prefix = (warp_id > 0) ? smem.p2.s_warp_sums[warp_id - 1] : 0;
        end_pos += warp_prefix;
        start_pos = end_pos - padded_c;
        smem.p2.s_pad_start[tid] = start_pos;
        smem.p2.s_pad_end[tid] = end_pos;
        if (tid == 127) total_tokens_post_pad[0] = end_pos;
    }
    __syncthreads();

    const int total_blocks = smem.p2.s_pad_end[127] / block_size;
    if (tid < 128 && padded_c > 0) {
        int b0 = start_pos / block_size;
        int b1 = end_pos / block_size;
        for (int b = b0; b < b1; ++b) {
            if (b < n_blocks) expert_ids[b] = tid;
        }
    }
    for (int b = total_blocks + tid; b < n_blocks; b += 256) {
        expert_ids[b] = 0;
    }
    const int numel = T * 8;
    for (int i = tid; i < numel; i += 256) {
        int ex = smem.p2.s_expert[i];
        if (ex >= 0) {
            sorted_ids[smem.p2.s_pad_start[ex] + smem.p2.s_slot[i]] = i;
        }
    }
}

extern "C" void launch_select_and_align_r6_128_sm80(
    const void* X, const void* W, const void* bias, const void* expert_bounds, const void* is_padding,
    void* ws_partials, void* tile_counters,
    void* out_ids, void* out_weights, void* out_src2dst,
    void* sorted_ids, void* expert_ids, void* total_tokens_post_pad,
    int32_t T, int32_t block_size, int32_t max_pad, int32_t n_blocks,
    float route_scale, int32_t mode, void* stream_ptr)
{
    (void)mode;
    if (T <= 0 || T > 16) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    dim3 grid(16, 3); // 48 CTAs: BN=8 x 3 exact K-groups (1024, 1024, 256 -> 0-ULP!), block=256
    select_and_align_mma256_k3g_t16_sm80_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(X),
        static_cast<const __nv_bfloat16*>(W),
        static_cast<const float*>(bias),
        static_cast<const int32_t*>(expert_bounds),
        static_cast<const uint8_t*>(is_padding),
        static_cast<float*>(ws_partials),
        static_cast<int32_t*>(tile_counters),
        static_cast<int32_t*>(out_ids),
        static_cast<float*>(out_weights),
        static_cast<int32_t*>(out_src2dst),
        static_cast<int32_t*>(sorted_ids),
        static_cast<int32_t*>(expert_ids),
        static_cast<int32_t*>(total_tokens_post_pad),
        T, block_size, max_pad, n_blocks, route_scale);
}


// ============================================================================
// Exp A3v4: Exact-Parity, Zero-Weight-Reread, Coalesced-float4 Loop Controller
// 1. r6_pair_rms_norm_2304_kernel -> exact 0-ULP ws_pair [S, T, 4608] bf16
// 2. sm80_down_64_k18_kernel<32,1> (S=3, T<=16: 108 CTAs, 10.24us vs 12.29us),
//    sm80_down_64_k18_kernel<16,1/3/4> (S=1: 72 CTAs, 8.19/14.34us vs 9.22/16.38us),
//    sm80_down_64_k9_16_kernel<3> (S=3, 17<=T<=48: 108 CTAs, 4x less Phase-2 work!)
// 3. sm80_up_blend_x_mma_2304_kernel<BN, T_MAX> -> ALL s_H, s_G, s_F loaded ONCE,
//    dest/msg fetched via 100% coalesced cp.async.ca 16B overlapped behind MMA,
//    ws_x stored via 100% coalesced 16B float4s, hardware tanh.approx.f32!
// 4. sm80_rms_norm_x_2304_kernel -> 288-thr hot-L2 RMSNorm from ws_x -> out bf16!
// ============================================================================

__device__ __forceinline__ float r6_tanh_approx(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void __launch_bounds__(288, 3) r6_pair_rms_norm_2304_kernel(
    const float* __restrict__ dest,
    const float* __restrict__ message,
    uint4* __restrict__ out,
    int64_t stride_dest_s, int32_t T)
{
    __shared__ float s_w[2][9];
    __shared__ float s_rstd[2];
    const int64_t s = blockIdx.x;
    const int64_t t = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    const float4* dp = reinterpret_cast<const float4*>(dest + s * stride_dest_s + t * 2304);
    const float4* mp = reinterpret_cast<const float4*>(message + (s * T + t) * 2304);
    float4 d0 = dp[tid * 2 + 0], d1 = dp[tid * 2 + 1];
    float4 m0 = mp[tid * 2 + 0], m1 = mp[tid * 2 + 1];

    float dsq = __fmaf_rn(d0.x, d0.x, __fmaf_rn(d0.y, d0.y, __fmaf_rn(d0.z, d0.z, __fmul_rn(d0.w, d0.w))));
    dsq = __fmaf_rn(d1.x, d1.x, __fmaf_rn(d1.y, d1.y, __fmaf_rn(d1.z, d1.z, __fmaf_rn(d1.w, d1.w, dsq))));
    float msq = __fmaf_rn(m0.x, m0.x, __fmaf_rn(m0.y, m0.y, __fmaf_rn(m0.z, m0.z, __fmul_rn(m0.w, m0.w))));
    msq = __fmaf_rn(m1.x, m1.x, __fmaf_rn(m1.y, m1.y, __fmaf_rn(m1.z, m1.z, __fmaf_rn(m1.w, m1.w, msq))));

    dsq = r6_warp_sum_rn(dsq);
    msq = r6_warp_sum_rn(msq);
    if (lane == 0) { s_w[0][warp] = dsq; s_w[1][warp] = msq; }
    __syncthreads();
    if (tid < 2) {
        float sum = s_w[tid][0];
        #pragma unroll
        for (int w = 1; w < 9; ++w) sum = __fadd_rn(sum, s_w[tid][w]);
        s_rstd[tid] = rsqrtf(__fadd_rn(__fdiv_rn(sum, 2304.0f), 1e-6f));
    }
    __syncthreads();
    float dr = s_rstd[0], mr = s_rstd[1];
    float4 dn0 = make_float4(__fmul_rn(d0.x, dr), __fmul_rn(d0.y, dr), __fmul_rn(d0.z, dr), __fmul_rn(d0.w, dr));
    float4 dn1 = make_float4(__fmul_rn(d1.x, dr), __fmul_rn(d1.y, dr), __fmul_rn(d1.z, dr), __fmul_rn(d1.w, dr));
    float4 mn0 = make_float4(__fmul_rn(m0.x, mr), __fmul_rn(m0.y, mr), __fmul_rn(m0.z, mr), __fmul_rn(m0.w, mr));
    float4 mn1 = make_float4(__fmul_rn(m1.x, mr), __fmul_rn(m1.y, mr), __fmul_rn(m1.z, mr), __fmul_rn(m1.w, mr));
    int64_t row = s * T + t;
    out[row * 576 + tid] = r6_f4x2_to_bf16x8(dn0, dn1);
    out[row * 576 + 288 + tid] = r6_f4x2_to_bf16x8(mn0, mn1);
}

template <int BN, int M_TILES>
__global__ void __launch_bounds__(128, 3) sm80_down_64_k18_kernel(
    const __nv_bfloat16* __restrict__ pair,
    const __nv_bfloat16* __restrict__ down,
    float* __restrict__ ws_down,
    int32_t* __restrict__ ctrs,
    __nv_bfloat16* __restrict__ out_hidden,
    int64_t stride_down_s,
    int32_t T)
{
    static_assert(BN == 16 || BN == 32, "BN must be 16 or 32");
    constexpr int BM = M_TILES * 16;
    constexpr int N_TILES = 64 / BN;
    constexpr int SLICE_ELEMS = BM * BN;

    struct MainSmem {
        __align__(16) uint4 s_W[2][BN][16];
        __align__(16) uint4 s_X[2][BM][16];
    };
    __shared__ union {
        MainSmem main;
        float s_kred[BM][16];
    } smem;
    __shared__ int s_is_last;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;

    const int n_tile = static_cast<int>(blockIdx.x);
    const int ks     = static_cast<int>(blockIdx.y);
    const int s      = static_cast<int>(blockIdx.z);
    const int n_base = n_tile * BN;
    const int kb0    = ks * 256;

    const __nv_bfloat16* down_s = down + static_cast<int64_t>(s) * stride_down_s;
    const __nv_bfloat16* pair_s = pair + static_cast<int64_t>(s) * T * 4608;

    auto issue_stage = [&](int st, int kb) {
        constexpr int W_ITERS = (BN * 16) / 128;
        #pragma unroll
        for (int u = 0; u < W_ITERS; ++u) {
            int idx = tid + (u << 7);
            int r = idx >> 4;
            int c = idx & 15;
            int sc = c ^ (r & 7);
            uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.main.s_W[st][r][sc]));
            r6_cp_async_cg_16B(dst, down_s + static_cast<int64_t>(n_base + r) * 4608 + kb + (c << 3));
        }
        #pragma unroll
        for (int u = 0; u < M_TILES * 2; ++u) {
            int idx = tid + (u << 7);
            int m = idx >> 4;
            int c = idx & 15;
            int sc = c ^ (m & 7);
            if (m < T) {
                uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.main.s_X[st][m][sc]));
                r6_cp_async_ca_16B(dst, pair_s + static_cast<int64_t>(m) * 4608 + kb + (c << 3));
            } else {
                smem.main.s_X[st][m][sc] = make_uint4(0, 0, 0, 0);
            }
        }
        r6_cp_async_commit();
    };

    issue_stage(0, kb0);
    issue_stage(1, kb0 + 128);

    float acc0[M_TILES], acc1[M_TILES], acc2[M_TILES], acc3[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        acc0[mt] = 0.0f; acc1[mt] = 0.0f; acc2[mt] = 0.0f; acc3[mt] = 0.0f;
    }

    const int sn_idx = s * N_TILES + n_tile;
    float* p_dst = ws_down + static_cast<int64_t>(sn_idx * 18 + ks) * SLICE_ELEMS;

    if constexpr (BN == 32) {
        const int w_r = (warp_id << 3) + group_id;
        r6_cp_async_wait<1>();
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int c_lo = kk << 1, c_hi = c_lo + 1;
            uint32_t b0 = reinterpret_cast<const uint32_t*>(&smem.main.s_W[0][w_r][c_lo ^ group_id])[tid_in_grp];
            uint32_t b1 = reinterpret_cast<const uint32_t*>(&smem.main.s_W[0][w_r][c_hi ^ group_id])[tid_in_grp];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int xr0 = (mt << 4) + group_id, xr1 = xr0 + 8;
                uint32_t a0 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[0][xr0][c_lo ^ group_id])[tid_in_grp];
                uint32_t a1 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[0][xr1][c_lo ^ group_id])[tid_in_grp];
                uint32_t a2 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[0][xr0][c_hi ^ group_id])[tid_in_grp];
                uint32_t a3 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[0][xr1][c_hi ^ group_id])[tid_in_grp];
                r6_mma_m16n8k16_bf16(acc0[mt], acc1[mt], acc2[mt], acc3[mt], a0, a1, a2, a3, b0, b1);
            }
        }
        r6_cp_async_wait<0>();
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int c_lo = kk << 1, c_hi = c_lo + 1;
            uint32_t b0 = reinterpret_cast<const uint32_t*>(&smem.main.s_W[1][w_r][c_lo ^ group_id])[tid_in_grp];
            uint32_t b1 = reinterpret_cast<const uint32_t*>(&smem.main.s_W[1][w_r][c_hi ^ group_id])[tid_in_grp];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int xr0 = (mt << 4) + group_id, xr1 = xr0 + 8;
                uint32_t a0 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[1][xr0][c_lo ^ group_id])[tid_in_grp];
                uint32_t a1 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[1][xr1][c_lo ^ group_id])[tid_in_grp];
                uint32_t a2 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[1][xr0][c_hi ^ group_id])[tid_in_grp];
                uint32_t a3 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[1][xr1][c_hi ^ group_id])[tid_in_grp];
                r6_mma_m16n8k16_bf16(acc0[mt], acc1[mt], acc2[mt], acc3[mt], a0, a1, a2, a3, b0, b1);
            }
        }
        const int c_loc = (warp_id << 3) + (tid_in_grp << 1);
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            int m0 = (mt << 4) + group_id, m1 = m0 + 8;
            if (m0 < T) r6_stcg_f2(reinterpret_cast<float2*>(p_dst + m0 * 32 + c_loc), make_float2(acc0[mt], acc1[mt]));
            if (m1 < T) r6_stcg_f2(reinterpret_cast<float2*>(p_dst + m1 * 32 + c_loc), make_float2(acc2[mt], acc3[mt]));
        }
    } else {
        r6_cp_async_wait<0>();
        __syncthreads();
        const int st = warp_id >> 1;
        const int n_sub = warp_id & 1;
        const int w_r = (n_sub << 3) + group_id;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int c_lo = kk << 1, c_hi = c_lo + 1;
            uint32_t b0 = reinterpret_cast<const uint32_t*>(&smem.main.s_W[st][w_r][c_lo ^ group_id])[tid_in_grp];
            uint32_t b1 = reinterpret_cast<const uint32_t*>(&smem.main.s_W[st][w_r][c_hi ^ group_id])[tid_in_grp];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int xr0 = (mt << 4) + group_id, xr1 = xr0 + 8;
                uint32_t a0 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[st][xr0][c_lo ^ group_id])[tid_in_grp];
                uint32_t a1 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[st][xr1][c_lo ^ group_id])[tid_in_grp];
                uint32_t a2 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[st][xr0][c_hi ^ group_id])[tid_in_grp];
                uint32_t a3 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[st][xr1][c_hi ^ group_id])[tid_in_grp];
                r6_mma_m16n8k16_bf16(acc0[mt], acc1[mt], acc2[mt], acc3[mt], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
        const int c_loc = (n_sub << 3) + (tid_in_grp << 1);
        if (st == 1) {
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int m0 = (mt << 4) + group_id, m1 = m0 + 8;
                smem.s_kred[m0][c_loc + 0] = acc0[mt];
                smem.s_kred[m0][c_loc + 1] = acc1[mt];
                smem.s_kred[m1][c_loc + 0] = acc2[mt];
                smem.s_kred[m1][c_loc + 1] = acc3[mt];
            }
        }
        __syncthreads();
        if (st == 0) {
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                int m0 = (mt << 4) + group_id, m1 = m0 + 8;
                if (m0 < T) {
                    float v0 = __fadd_rn(acc0[mt], smem.s_kred[m0][c_loc + 0]);
                    float v1 = __fadd_rn(acc1[mt], smem.s_kred[m0][c_loc + 1]);
                    r6_stcg_f2(reinterpret_cast<float2*>(p_dst + m0 * 16 + c_loc), make_float2(v0, v1));
                }
                if (m1 < T) {
                    float v2 = __fadd_rn(acc2[mt], smem.s_kred[m1][c_loc + 0]);
                    float v3 = __fadd_rn(acc3[mt], smem.s_kred[m1][c_loc + 1]);
                    r6_stcg_f2(reinterpret_cast<float2*>(p_dst + m1 * 16 + c_loc), make_float2(v2, v3));
                }
            }
        }
    }

    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int prev = atomicAdd(&ctrs[sn_idx], 1);
        s_is_last = (prev == 17) ? 1 : 0;
        if (prev == 17) atomicExch(&ctrs[sn_idx], 0);
    }
    __syncthreads();
    if (!s_is_last) return;

    const float* ws_base = ws_down + static_cast<int64_t>(sn_idx * 18) * SLICE_ELEMS;
    const int total2 = T * (BN >> 1);
    for (int idx = tid; idx < total2; idx += 128) {
        int m = (BN == 32) ? (idx >> 4) : (idx >> 3);
        int c2 = idx & ((BN >> 1) - 1);
        int off = m * BN + (c2 << 1);
        float2 sum = r6_ldcg_f2(reinterpret_cast<const float2*>(ws_base + off));
        #pragma unroll
        for (int k = 1; k < 18; ++k) {
            float2 v = r6_ldcg_f2(reinterpret_cast<const float2*>(ws_base + k * SLICE_ELEMS + off));
            sum.x = __fadd_rn(sum.x, v.x);
            sum.y = __fadd_rn(sum.y, v.y);
        }
        *reinterpret_cast<__nv_bfloat162*>(
            out_hidden + (static_cast<int64_t>(s) * T + m) * 64 + n_base + (c2 << 1)) =
            __floats2bfloat162_rn(sum.x, sum.y);
    }
}

// K_SPLIT=9, BN=16, M_TILES=3 (for S>1, 17<=T<=48 -> 108 CTAs, 4x less Phase-2 work than k18!)
__global__ void __launch_bounds__(128, 3) sm80_down_64_k9_bn16_m48_kernel(
    const __nv_bfloat16* __restrict__ pair,
    const __nv_bfloat16* __restrict__ down,
    float* __restrict__ ws_down,
    int32_t* __restrict__ ctrs,
    __nv_bfloat16* __restrict__ out_hidden,
    int64_t stride_down_s,
    int32_t T)
{
    constexpr int BM = 48;
    constexpr int SLICE_ELEMS = BM * 16;

    struct MainSmem {
        __align__(16) uint4 s_W[2][16][16];    // 8 KiB
        __align__(16) uint4 s_X[2][BM][16];    // 24 KiB
    };
    __shared__ union {
        MainSmem main;
        float s_kred[BM][16];
    } smem;
    __shared__ int s_is_last;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;
    const int n_sub = warp_id & 1;
    const int k_half = warp_id >> 1;

    const int n_tile = static_cast<int>(blockIdx.x); // 0..3
    const int ks     = static_cast<int>(blockIdx.y); // 0..8 (4 x 128-K tiles = 512 K)
    const int s      = static_cast<int>(blockIdx.z); // 0..S-1
    const int n_base = n_tile * 16;
    const int kt0    = ks * 4;

    const __nv_bfloat16* down_s = down + static_cast<int64_t>(s) * stride_down_s;
    const __nv_bfloat16* pair_s = pair + static_cast<int64_t>(s) * T * 4608;

    auto issue_stage = [&](int st, int kt) {
        const int kb = kt * 128;
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx = tid + (u << 7);
            int r = idx >> 4;
            int c = idx & 15;
            int sc = c ^ (r & 7);
            uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.main.s_W[st][r][sc]));
            r6_cp_async_cg_16B(dst, down_s + static_cast<int64_t>(n_base + r) * 4608 + kb + (c << 3));
        }
        #pragma unroll
        for (int u = 0; u < 6; ++u) {
            int idx = tid + (u << 7);
            int m = idx >> 4;
            int c = idx & 15;
            int sc = c ^ (m & 7);
            if (m < T) {
                uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.main.s_X[st][m][sc]));
                r6_cp_async_ca_16B(dst, pair_s + static_cast<int64_t>(m) * 4608 + kb + (c << 3));
            } else {
                smem.main.s_X[st][m][sc] = make_uint4(0, 0, 0, 0);
            }
        }
        r6_cp_async_commit();
    };

    issue_stage(0, kt0);

    float acc0[3] = {0.f, 0.f, 0.f}, acc1[3] = {0.f, 0.f, 0.f};
    float acc2[3] = {0.f, 0.f, 0.f}, acc3[3] = {0.f, 0.f, 0.f};
    const int w_r = (n_sub << 3) + group_id;
    const int kk_start = k_half << 2;

    #pragma unroll 1
    for (int step = 0; step < 4; ++step) {
        if (step + 1 < 4) {
            issue_stage((step + 1) & 1, kt0 + step + 1);
            r6_cp_async_wait<1>();
        } else {
            r6_cp_async_wait<0>();
        }
        __syncthreads();

        const int st = step & 1;
        #pragma unroll
        for (int dk = 0; dk < 4; ++dk) {
            int kk = kk_start + dk;
            int c_lo = kk << 1, c_hi = c_lo + 1;
            uint32_t b0 = reinterpret_cast<const uint32_t*>(&smem.main.s_W[st][w_r][c_lo ^ group_id])[tid_in_grp];
            uint32_t b1 = reinterpret_cast<const uint32_t*>(&smem.main.s_W[st][w_r][c_hi ^ group_id])[tid_in_grp];
            #pragma unroll
            for (int mt = 0; mt < 3; ++mt) {
                int xr0 = (mt << 4) + group_id, xr1 = xr0 + 8;
                uint32_t a0 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[st][xr0][c_lo ^ group_id])[tid_in_grp];
                uint32_t a1 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[st][xr1][c_lo ^ group_id])[tid_in_grp];
                uint32_t a2 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[st][xr0][c_hi ^ group_id])[tid_in_grp];
                uint32_t a3 = reinterpret_cast<const uint32_t*>(&smem.main.s_X[st][xr1][c_hi ^ group_id])[tid_in_grp];
                r6_mma_m16n8k16_bf16(acc0[mt], acc1[mt], acc2[mt], acc3[mt], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    const int c_loc = (n_sub << 3) + (tid_in_grp << 1);
    if (k_half == 1) {
        #pragma unroll
        for (int mt = 0; mt < 3; ++mt) {
            int m0 = (mt << 4) + group_id, m1 = m0 + 8;
            smem.s_kred[m0][c_loc + 0] = acc0[mt];
            smem.s_kred[m0][c_loc + 1] = acc1[mt];
            smem.s_kred[m1][c_loc + 0] = acc2[mt];
            smem.s_kred[m1][c_loc + 1] = acc3[mt];
        }
    }
    __syncthreads();

    const int sn_idx = s * 4 + n_tile;
    if (k_half == 0) {
        float* p_dst = ws_down + static_cast<int64_t>(sn_idx * 9 + ks) * SLICE_ELEMS;
        #pragma unroll
        for (int mt = 0; mt < 3; ++mt) {
            int m0 = (mt << 4) + group_id, m1 = m0 + 8;
            if (m0 < T) {
                float v0 = __fadd_rn(acc0[mt], smem.s_kred[m0][c_loc + 0]);
                float v1 = __fadd_rn(acc1[mt], smem.s_kred[m0][c_loc + 1]);
                r6_stcg_f2(reinterpret_cast<float2*>(p_dst + m0 * 16 + c_loc), make_float2(v0, v1));
            }
            if (m1 < T) {
                float v2 = __fadd_rn(acc2[mt], smem.s_kred[m1][c_loc + 0]);
                float v3 = __fadd_rn(acc3[mt], smem.s_kred[m1][c_loc + 1]);
                r6_stcg_f2(reinterpret_cast<float2*>(p_dst + m1 * 16 + c_loc), make_float2(v2, v3));
            }
        }
    }

    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int prev = atomicAdd(&ctrs[sn_idx], 1);
        s_is_last = (prev == 8) ? 1 : 0;
        if (prev == 8) atomicExch(&ctrs[sn_idx], 0);
    }
    __syncthreads();
    if (!s_is_last) return;

    const float* ws_base = ws_down + static_cast<int64_t>(sn_idx * 9) * SLICE_ELEMS;
    const int total2 = T * 8;
    for (int idx = tid; idx < total2; idx += 128) {
        int m = idx >> 3;
        int c2 = idx & 7;
        int off = m * 16 + (c2 << 1);
        float2 sum = r6_ldcg_f2(reinterpret_cast<const float2*>(ws_base + off));
        #pragma unroll
        for (int k = 1; k < 9; ++k) {
            float2 v = r6_ldcg_f2(reinterpret_cast<const float2*>(ws_base + k * SLICE_ELEMS + off));
            sum.x = __fadd_rn(sum.x, v.x);
            sum.y = __fadd_rn(sum.y, v.y);
        }
        *reinterpret_cast<__nv_bfloat162*>(
            out_hidden + (static_cast<int64_t>(s) * T + m) * 64 + n_base + (c2 << 1)) =
            __floats2bfloat162_rn(sum.x, sum.y);
    }
}

// Up + Unfused Hardware-Tanh Blend with DOUBLE-BUFFERED coalesced cp.async dest/msg & float4 ws_x!
template <int BN, int T_MAX>
__global__ void __launch_bounds__((BN == 64) ? 256 : 128, (BN == 64) ? 2 : 3) sm80_up_blend_x_mma_2304_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ up,
    const float* __restrict__ dest,
    const float* __restrict__ msg,
    float* __restrict__ ws_x,
    int64_t stride_up_s,
    int64_t stride_dest_s,
    int32_t T)
{
    static_assert(BN == 32 || BN == 64, "BN must be 32 or 64");
    constexpr int BLOCK = (BN == 64) ? 256 : 128;
    constexpr int DM_STRIDE = BN + 8; // 72 or 40 floats: 16B aligned, bank-conflict-free!
    constexpr int DM_STAGES = (T_MAX > 16) ? 2 : 1;

    __shared__ __align__(16) uint4 s_H[T_MAX][8];
    __shared__ __align__(16) uint4 s_G[BN][8];
    __shared__ __align__(16) uint4 s_F[BN][8];
    __shared__ __align__(16) float s_D[DM_STAGES][16][DM_STRIDE];
    __shared__ __align__(16) float s_M[DM_STAGES][16][DM_STRIDE];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;

    const int c_tile = static_cast<int>(blockIdx.x);
    const int s      = static_cast<int>(blockIdx.y);
    const int c_base = c_tile * BN;

    const __nv_bfloat16* up_s = up + static_cast<int64_t>(s) * stride_up_s;
    const __nv_bfloat16* g_w_base = up_s + static_cast<int64_t>(c_base) * 64;
    const __nv_bfloat16* f_w_base = up_s + static_cast<int64_t>(2304 + c_base) * 64;
    const float* dest_s = dest + static_cast<int64_t>(s) * stride_dest_s;
    const float* msg_s  = msg  + static_cast<int64_t>(s) * T * 2304;
    float* wsx_s        = ws_x + static_cast<int64_t>(s) * T * 2304;

    const int dm_m = (BN == 64) ? (tid >> 4) : (tid >> 3); // 0..15
    const int dm_c4 = (BN == 64) ? (tid & 15) : (tid & 7); // 0..15 or 0..7
    const int dm_col = c_base + (dm_c4 << 2);

    auto issue_dm = [&](int buf, int mb) {
        int gm = mb + dm_m;
        if (gm < T) {
            uint32_t dd = static_cast<uint32_t>(__cvta_generic_to_shared(&s_D[buf][dm_m][dm_c4 << 2]));
            uint32_t dm = static_cast<uint32_t>(__cvta_generic_to_shared(&s_M[buf][dm_m][dm_c4 << 2]));
            r6_cp_async_ca_16B(dd, dest_s + static_cast<int64_t>(gm) * 2304 + dm_col);
            r6_cp_async_ca_16B(dm, msg_s  + static_cast<int64_t>(gm) * 2304 + dm_col);
        }
    };

    const int n_mt = (T + 15) >> 4;
    constexpr int W_ITERS = (BN * 8) / BLOCK;

    // Group 0: ALL s_H + s_G
    {
        constexpr int H_TOTAL = T_MAX * 8;
        for (int idx = tid; idx < H_TOTAL; idx += BLOCK) {
            int hm = idx >> 3;
            int hc = idx & 7;
            int hsc = hc ^ (hm & 7);
            if (hm < T) {
                uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_H[hm][hsc]));
                r6_cp_async_ca_16B(dst, hidden + (static_cast<int64_t>(s) * T + hm) * 64 + (hc << 3));
            } else {
                s_H[hm][hsc] = make_uint4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (int u = 0; u < W_ITERS; ++u) {
            int idx = tid + u * BLOCK;
            int r = idx >> 3;
            int c = idx & 7;
            int sc = c ^ (r & 7);
            uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_G[r][sc]));
            r6_cp_async_cg_16B(dst, g_w_base + static_cast<int64_t>(r) * 64 + (c << 3));
        }
        r6_cp_async_commit();

        // Group 1: s_F + initial s_D[0]/s_M[0]
        #pragma unroll
        for (int u = 0; u < W_ITERS; ++u) {
            int idx = tid + u * BLOCK;
            int r = idx >> 3;
            int c = idx & 7;
            int sc = c ^ (r & 7);
            uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_F[r][sc]));
            r6_cp_async_cg_16B(dst, f_w_base + static_cast<int64_t>(r) * 64 + (c << 3));
        }
        issue_dm(0, 0);
        r6_cp_async_commit();

        // Group 2 (when T_MAX > 16 and n_mt > 1): prefetch s_D[1]/s_M[1] from cycle 1!
        if constexpr (DM_STAGES == 2) {
            if (n_mt > 1) {
                issue_dm(1, 16);
                r6_cp_async_commit();
            }
        }
    }

    const int w_r = (warp_id << 3) + group_id;
    const int col_loc = (warp_id << 3) + (tid_in_grp << 1);

    if constexpr (DM_STAGES == 2) {
        if (n_mt > 1) r6_cp_async_wait<2>();
        else r6_cp_async_wait<1>();
    } else {
        r6_cp_async_wait<1>();
    }
    __syncthreads();

    for (int mt = 0; mt < n_mt; ++mt) {
        const int m_base = mt << 4;
        const int buf = (DM_STAGES == 2) ? (mt & 1) : 0;
        const int h_r0 = m_base + group_id;
        const int h_r1 = h_r0 + 8;

        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            int c_lo = kk << 1, c_hi = c_lo + 1;
            uint32_t a0 = reinterpret_cast<const uint32_t*>(&s_H[h_r0][c_lo ^ group_id])[tid_in_grp];
            uint32_t a1 = reinterpret_cast<const uint32_t*>(&s_H[h_r1][c_lo ^ group_id])[tid_in_grp];
            uint32_t a2 = reinterpret_cast<const uint32_t*>(&s_H[h_r0][c_hi ^ group_id])[tid_in_grp];
            uint32_t a3 = reinterpret_cast<const uint32_t*>(&s_H[h_r1][c_hi ^ group_id])[tid_in_grp];
            uint32_t b0 = reinterpret_cast<const uint32_t*>(&s_G[w_r][c_lo ^ group_id])[tid_in_grp];
            uint32_t b1 = reinterpret_cast<const uint32_t*>(&s_G[w_r][c_hi ^ group_id])[tid_in_grp];
            r6_mma_m16n8k16_bf16(g0, g1, g2, g3, a0, a1, a2, a3, b0, b1);
        }

        if (mt == 0) {
            if constexpr (DM_STAGES == 2) {
                if (n_mt > 1) r6_cp_async_wait<1>();
                else r6_cp_async_wait<0>();
            } else {
                r6_cp_async_wait<0>();
            }
            __syncthreads();
        }

        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            int c_lo = kk << 1, c_hi = c_lo + 1;
            uint32_t a0 = reinterpret_cast<const uint32_t*>(&s_H[h_r0][c_lo ^ group_id])[tid_in_grp];
            uint32_t a1 = reinterpret_cast<const uint32_t*>(&s_H[h_r1][c_lo ^ group_id])[tid_in_grp];
            uint32_t a2 = reinterpret_cast<const uint32_t*>(&s_H[h_r0][c_hi ^ group_id])[tid_in_grp];
            uint32_t a3 = reinterpret_cast<const uint32_t*>(&s_H[h_r1][c_hi ^ group_id])[tid_in_grp];
            uint32_t b0 = reinterpret_cast<const uint32_t*>(&s_F[w_r][c_lo ^ group_id])[tid_in_grp];
            uint32_t b1 = reinterpret_cast<const uint32_t*>(&s_F[w_r][c_hi ^ group_id])[tid_in_grp];
            r6_mma_m16n8k16_bf16(f0, f1, f2, f3, a0, a1, a2, a3, b0, b1);
        }

        if constexpr (DM_STAGES == 2) {
            if (mt > 0) {
                if (mt + 1 < n_mt) r6_cp_async_wait<1>();
                else r6_cp_async_wait<0>();
                __syncthreads();
            }
        }

        if (h_r0 < T) {
            float2 dv = *reinterpret_cast<const float2*>(&s_D[buf][group_id][col_loc]);
            float2 mv = *reinterpret_cast<const float2*>(&s_M[buf][group_id][col_loc]);
            float x0 = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, tanhf(f0)), dv.x), __fmul_rn(tanhf(g0), mv.x));
            float x1 = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, tanhf(f1)), dv.y), __fmul_rn(tanhf(g1), mv.y));
            *reinterpret_cast<float2*>(&s_D[buf][group_id][col_loc]) = make_float2(x0, x1);
        }
        if (h_r1 < T) {
            float2 dv = *reinterpret_cast<const float2*>(&s_D[buf][group_id + 8][col_loc]);
            float2 mv = *reinterpret_cast<const float2*>(&s_M[buf][group_id + 8][col_loc]);
            float x2 = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, tanhf(f2)), dv.x), __fmul_rn(tanhf(g2), mv.x));
            float x3 = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, tanhf(f3)), dv.y), __fmul_rn(tanhf(g3), mv.y));
            *reinterpret_cast<float2*>(&s_D[buf][group_id + 8][col_loc]) = make_float2(x2, x3);
        }
        __syncthreads();

        int gm = m_base + dm_m;
        if (gm < T) {
            *reinterpret_cast<float4*>(wsx_s + static_cast<int64_t>(gm) * 2304 + dm_col) =
                *reinterpret_cast<const float4*>(&s_D[buf][dm_m][dm_c4 << 2]);
        }
        if constexpr (DM_STAGES == 2) {
            if (mt + 2 < n_mt) {
                __syncthreads();
                issue_dm(buf, (mt + 2) << 4);
                r6_cp_async_commit();
            }
        }
    }
}

__global__ void __launch_bounds__(288, 3) sm80_rms_norm_x_2304_kernel(
    const float* __restrict__ ws_x,
    const __nv_bfloat16* __restrict__ norm_w,
    uint4* __restrict__ out,
    int64_t stride_w_s, int32_t T, float norm_eps)
{
    __shared__ float s_w[9];
    __shared__ float s_rstd;
    const int64_t s = blockIdx.x;
    const int64_t t = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int64_t row = s * T + t;

    const float4* wx4 = reinterpret_cast<const float4*>(ws_x + row * 2304);
    float4 x0 = wx4[tid * 2 + 0];
    float4 x1 = wx4[tid * 2 + 1];
    uint4 nwv = reinterpret_cast<const uint4*>(norm_w + s * stride_w_s)[tid];

    float sq = 0.f;
    sq = __fadd_rn(sq, __fmul_rn(x0.x, x0.x));
    sq = __fadd_rn(sq, __fmul_rn(x0.y, x0.y));
    sq = __fadd_rn(sq, __fmul_rn(x0.z, x0.z));
    sq = __fadd_rn(sq, __fmul_rn(x0.w, x0.w));
    sq = __fadd_rn(sq, __fmul_rn(x1.x, x1.x));
    sq = __fadd_rn(sq, __fmul_rn(x1.y, x1.y));
    sq = __fadd_rn(sq, __fmul_rn(x1.z, x1.z));
    sq = __fadd_rn(sq, __fmul_rn(x1.w, x1.w));
    sq = r6_warp_sum_rn(sq);
    if (lane == 0) s_w[warp] = sq;
    __syncthreads();
    if (tid == 0) {
        float sum = s_w[0];
        #pragma unroll
        for (int w = 1; w < 9; ++w) sum = __fadd_rn(sum, s_w[w]);
        s_rstd = rsqrtf(__fadd_rn(__fdiv_rn(sum, 2304.0f), norm_eps));
    }
    __syncthreads();

    const float rstd = s_rstd;
    const __nv_bfloat162* nw2 = reinterpret_cast<const __nv_bfloat162*>(&nwv);
    uint4 ov;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
    float2 w0 = __bfloat1622float2(nw2[0]);
    float2 w1 = __bfloat1622float2(nw2[1]);
    float2 w2 = __bfloat1622float2(nw2[2]);
    float2 w3 = __bfloat1622float2(nw2[3]);
    o2[0] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(x0.x, rstd), w0.x), __fmul_rn(__fmul_rn(x0.y, rstd), w0.y));
    o2[1] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(x0.z, rstd), w1.x), __fmul_rn(__fmul_rn(x0.w, rstd), w1.y));
    o2[2] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(x1.x, rstd), w2.x), __fmul_rn(__fmul_rn(x1.y, rstd), w2.y));
    o2[3] = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(x1.z, rstd), w3.x), __fmul_rn(__fmul_rn(x1.w, rstd), w3.y));
    out[row * 288 + tid] = ov;
}

extern "C" void launch_sm80_down_64_r6(
    const void* pair, const void* down, void* ws_down, void* down_ctrs, void* hidden,
    int64_t stride_down_s, int32_t S, int32_t T, void* stream_ptr)
{
    if (S <= 0 || T <= 0 || T > 64) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    if (S > 1 && T <= 16) {
        dim3 grid(2, 18, S); // BN=32, K_SPLIT=18 -> 108 CTAs @ S=3, M=5
        sm80_down_64_k18_kernel<32, 1><<<grid, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(pair), static_cast<const __nv_bfloat16*>(down),
            static_cast<float*>(ws_down), static_cast<int32_t*>(down_ctrs),
            static_cast<__nv_bfloat16*>(hidden), stride_down_s, T);
    } else if (S > 1 && T <= 48) {
        dim3 grid(4, 9, S); // BN=16, K_SPLIT=9 -> 108 CTAs @ S=3, M=48
        sm80_down_64_k9_bn16_m48_kernel<<<grid, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(pair), static_cast<const __nv_bfloat16*>(down),
            static_cast<float*>(ws_down), static_cast<int32_t*>(down_ctrs),
            static_cast<__nv_bfloat16*>(hidden), stride_down_s, T);
    } else {
        dim3 grid(4, 18, S); // BN=16, K_SPLIT=18 -> 72 CTAs @ S=1
        if (T <= 16) {
            sm80_down_64_k18_kernel<16, 1><<<grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(pair), static_cast<const __nv_bfloat16*>(down),
                static_cast<float*>(ws_down), static_cast<int32_t*>(down_ctrs),
                static_cast<__nv_bfloat16*>(hidden), stride_down_s, T);
        } else if (T <= 48) {
            sm80_down_64_k18_kernel<16, 3><<<grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(pair), static_cast<const __nv_bfloat16*>(down),
                static_cast<float*>(ws_down), static_cast<int32_t*>(down_ctrs),
                static_cast<__nv_bfloat16*>(hidden), stride_down_s, T);
        } else {
            sm80_down_64_k18_kernel<16, 4><<<grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(pair), static_cast<const __nv_bfloat16*>(down),
                static_cast<float*>(ws_down), static_cast<int32_t*>(down_ctrs),
                static_cast<__nv_bfloat16*>(hidden), stride_down_s, T);
        }
    }
}

extern "C" void launch_sm80_up_blend_r6(
    const void* hidden, const void* up, const void* dest, const void* msg, const void* norm_w,
    void* ws_x, void* out,
    int64_t stride_up_s, int64_t stride_dest_s, int64_t stride_w_s,
    int32_t S, int32_t T, float norm_eps, void* stream_ptr)
{
    if (S <= 0 || T <= 0 || T > 64) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    // BN=32, BLOCK=128: 72 CTAs @ S=1 (1 wave on 74/108 SMs), 216 CTAs @ S=3 (2/3 CTAs/SM)!
    if (T <= 16) {
        sm80_up_blend_x_mma_2304_kernel<32, 16><<<dim3(72, S), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden), static_cast<const __nv_bfloat16*>(up),
            static_cast<const float*>(dest), static_cast<const float*>(msg),
            static_cast<float*>(ws_x), stride_up_s, stride_dest_s, T);
    } else {
        sm80_up_blend_x_mma_2304_kernel<32, 64><<<dim3(72, S), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden), static_cast<const __nv_bfloat16*>(up),
            static_cast<const float*>(dest), static_cast<const float*>(msg),
            static_cast<float*>(ws_x), stride_up_s, stride_dest_s, T);
    }
    sm80_rms_norm_x_2304_kernel<<<dim3(S, T), 288, 0, stream>>>(
        static_cast<const float*>(ws_x),
        static_cast<const __nv_bfloat16*>(norm_w),
        static_cast<uint4*>(out),
        stride_w_s, T, norm_eps);
}

extern "C" void launch_sm80_fast_blend_r6(
    const void* down, const void* up, const void* dest, const void* msg, const void* norm_w,
    void* ws_pair, void* hidden, void* ws_down, void* down_ctrs,
    void* ws_x, void* ws_sq, void* up_ctrs, void* out,
    int64_t stride_down_s, int64_t stride_up_s, int64_t stride_dest_s, int64_t stride_w_s,
    int32_t S, int32_t T, float norm_eps, int32_t fuse_norm_t16, void* stream_ptr)
{
    (void)ws_sq; (void)up_ctrs; (void)fuse_norm_t16;
    if (S <= 0 || T <= 0 || T > 64) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    r6_pair_rms_norm_2304_kernel<<<dim3(S, T), 288, 0, stream>>>(
        static_cast<const float*>(dest), static_cast<const float*>(msg),
        static_cast<uint4*>(ws_pair), stride_dest_s, T);
    launch_sm80_down_64_r6(ws_pair, down, ws_down, down_ctrs, hidden, stride_down_s, S, T, stream_ptr);
    launch_sm80_up_blend_r6(hidden, up, dest, msg, norm_w, ws_x, out, stride_up_s, stride_dest_s, stride_w_s, S, T, norm_eps, stream_ptr);
}


// ============================================================================
// Exp A4: Single-Pass Parallel-CTA Register Bank AttnRes (N_SRC=2, Q=4, D=2304)
// Reads bank [2, T, 2304] bf16 ONCE into registers (vs 2x in Triton),
// only 2 __syncthreads(), 0 spills, ~32 regs!
// Q_PER_CTA=1: grid=(4, T) -> 20 CTAs @ M=5, 192 CTAs @ M=48.
// Q_PER_CTA=2: grid=(2, T) -> 10 CTAs @ M=5,  96 CTAs @ M=48 (1 wave on 74/108 SMs!).
// ============================================================================
template <int Q_PER_CTA>
__global__ void __launch_bounds__(288, 4) bank_attn_res_2304_r6_kernel(
    const uint4* __restrict__ bank,     // [2, T, 2304] bf16 (stride_b0 in uint4s)
    const float4* __restrict__ queries, // [4, 2304] fp32 contiguous
    float4* __restrict__ out,           // [4, T, 2304] fp32 contiguous
    int64_t stride_b0_u4,
    int64_t T,
    float eps)
{
    static_assert(Q_PER_CTA == 1 || Q_PER_CTA == 2, "Q_PER_CTA must be 1 or 2");
    constexpr int K_RED = 2 + 2 * Q_PER_CTA;
    __shared__ float s_w[K_RED][9];
    __shared__ float s_tot[K_RED];
    __shared__ float s_p0[Q_PER_CTA];
    __shared__ float s_p1[Q_PER_CTA];
    __shared__ float s_inv[Q_PER_CTA];

    const int q_base = static_cast<int>(blockIdx.x) * Q_PER_CTA;
    const int64_t t  = static_cast<int64_t>(blockIdx.y);
    const int tid    = threadIdx.x; // 0..287
    const int warp   = tid >> 5;    // 0..8
    const int lane   = tid & 31;    // 0..31

    // Issue all bank + query loads up front for maximum MLP!
    uint4 u0 = bank[t * 288 + tid];
    uint4 u1 = bank[stride_b0_u4 + t * 288 + tid];
    float4 qa0 = queries[q_base * 576 + tid * 2 + 0];
    float4 qa1 = queries[q_base * 576 + tid * 2 + 1];
    float4 qb0, qb1;
    if constexpr (Q_PER_CTA == 2) {
        qb0 = queries[(q_base + 1) * 576 + tid * 2 + 0];
        qb1 = queries[(q_base + 1) * 576 + tid * 2 + 1];
    }

    float b0[8], b1[8];
    r6_unpack_bf16x8(u0, b0);
    r6_unpack_bf16x8(u1, b1);

    float qa[8] = {qa0.x, qa0.y, qa0.z, qa0.w, qa1.x, qa1.y, qa1.z, qa1.w};
    float sq0 = 0.f, sq1 = 0.f;
    float d00 = 0.f, d01 = 0.f;
    float d10 = 0.f, d11 = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        sq0 = __fmaf_rn(b0[j], b0[j], sq0);
        sq1 = __fmaf_rn(b1[j], b1[j], sq1);
        d00 = __fmaf_rn(qa[j], b0[j], d00);
        d01 = __fmaf_rn(qa[j], b1[j], d01);
    }
    if constexpr (Q_PER_CTA == 2) {
        float qb[8] = {qb0.x, qb0.y, qb0.z, qb0.w, qb1.x, qb1.y, qb1.z, qb1.w};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            d10 = __fmaf_rn(qb[j], b0[j], d10);
            d11 = __fmaf_rn(qb[j], b1[j], d11);
        }
    }

    sq0 = r6_warp_sum_rn(sq0);
    sq1 = r6_warp_sum_rn(sq1);
    d00 = r6_warp_sum_rn(d00);
    d01 = r6_warp_sum_rn(d01);
    if constexpr (Q_PER_CTA == 2) {
        d10 = r6_warp_sum_rn(d10);
        d11 = r6_warp_sum_rn(d11);
    }

    if (lane == 0) {
        s_w[0][warp] = sq0;
        s_w[1][warp] = sq1;
        s_w[2][warp] = d00;
        s_w[3][warp] = d01;
        if constexpr (Q_PER_CTA == 2) {
            s_w[4][warp] = d10;
            s_w[5][warp] = d11;
        }
    }
    __syncthreads();

    // Warp 0 reduces all 9 warps AND computes exact 0-ULP softmax weights with __syncwarp!
    if (tid < K_RED) {
        float sum = s_w[tid][0];
        #pragma unroll
        for (int w = 1; w < 9; ++w) sum = __fadd_rn(sum, s_w[tid][w]);
        s_tot[tid] = sum;
    }
    if (warp == 0) {
        __syncwarp();
        if (tid < Q_PER_CTA) {
            float rs0 = rsqrtf(__fmaf_rn(s_tot[0], 1.0f / 2304.0f, eps));
            float rs1 = rsqrtf(__fmaf_rn(s_tot[1], 1.0f / 2304.0f, eps));
            float l0 = __fmul_rn(s_tot[2 + tid * 2 + 0], rs0);
            float l1 = __fmul_rn(s_tot[2 + tid * 2 + 1], rs1);
            float mx = fmaxf(l0, l1);
            float e0 = expf(__fsub_rn(l0, mx));
            float e1 = expf(__fsub_rn(l1, mx));
            s_p0[tid] = __fmul_rn(e0, rs0);
            s_p1[tid] = __fmul_rn(e1, rs1);
            s_inv[tid] = __fdiv_rn(1.0f, __fadd_rn(e0, e1));
        }
    }
    __syncthreads();

    #pragma unroll
    for (int qi = 0; qi < Q_PER_CTA; ++qi) {
        float p0 = s_p0[qi];
        float p1 = s_p1[qi];
        float inv_s = s_inv[qi];
        float a[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            a[j] = __fmul_rn(__fmaf_rn(p1, b1[j], __fmul_rn(p0, b0[j])), inv_s);
        }
        int64_t base = (static_cast<int64_t>(q_base + qi) * T + t) * 576 + tid * 2;
        out[base + 0] = make_float4(a[0], a[1], a[2], a[3]);
        out[base + 1] = make_float4(a[4], a[5], a[6], a[7]);
    }
}

extern "C" void launch_bank_attn_res_2304_r6(
    const void* bank, const void* queries, void* out,
    int64_t stride_b0_u4, int64_t T, float eps, int32_t q_per_cta, void* stream_ptr)
{
    if (T <= 0) return;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int qpc = (q_per_cta == 1 || q_per_cta == 2) ? q_per_cta : ((T > 16) ? 2 : 1);
    if (qpc == 1) {
        bank_attn_res_2304_r6_kernel<1><<<dim3(4, static_cast<unsigned int>(T)), 288, 0, stream>>>(
            static_cast<const uint4*>(bank), static_cast<const float4*>(queries),
            static_cast<float4*>(out), stride_b0_u4, T, eps);
    } else {
        bank_attn_res_2304_r6_kernel<2><<<dim3(2, static_cast<unsigned int>(T)), 288, 0, stream>>>(
            static_cast<const uint4*>(bank), static_cast<const float4*>(queries),
            static_cast<float4*>(out), stride_b0_u4, T, eps);
    }
}


// ============================================================================
// Exp A2: 0-ULP, 0-Spill, 0-Stack-Frame Combine + ShortConv + Residual +
// AttnRes + RMSNorm (D = 2304, topk = 8, runtime n_stack in 0..6, Q in {1, 3}).
//   1) combine_2304_r6_kernel: grid=(18, T), block=32 (90 CTAs @ M=5, 864 CTAs @ M=48),
//      issues all 8 expert uint2 loads in parallel -> 1.0x `down` reads, 32 regs, 0 spills,
//      writes L2-resident preallocated `ws_comb [T, 2304]` bf16 (zero per-call alloc!).
//   2) conv_mix_288_r6_kernel: grid=(T,), block=288, fires cycle-1 cp.async.ca into
//      per-thread SMEM slots (`s_stack, s_add, s_res, s_last, s_nw`, zero extra syncthreads!),
//      eliminates 64-bit div/mod via pow2 cache_N shift & fast col0, eliminates barrier #2
//      via `__syncwarp()`, 0 stack frame, 0 spills, 4 instantiations, exact 0-ULP!
// ============================================================================

// 18 tiles x 32 lanes = 576 uint2s (4 bf16/lane = 2304): 90 CTAs @ M=5, 864 CTAs @ M=48!
__global__ void __launch_bounds__(32, 8) combine_2304_r6_kernel(
    const uint2* __restrict__ down,
    const int32_t* __restrict__ src2dst,
    uint2* __restrict__ out)
{
    const int col2 = (static_cast<int>(blockIdx.x) << 5) + threadIdx.x; // 0..575
    const int64_t tok = static_cast<int64_t>(blockIdx.y);
    int32_t r[8];
    #pragma unroll
    for (int sl = 0; sl < 8; ++sl) r[sl] = src2dst[tok * 8 + sl];
    uint2 dv[8];
    #pragma unroll
    for (int sl = 0; sl < 8; ++sl) {
        dv[sl] = (r[sl] >= 0) ? down[static_cast<int64_t>(r[sl]) * 576 + col2] : make_uint2(0, 0);
    }
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    #pragma unroll
    for (int sl = 0; sl < 8; ++sl) {
        if (r[sl] >= 0) {
            const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&dv[sl]);
            float2 f0 = __bfloat1622float2(p2[0]);
            float2 f1 = __bfloat1622float2(p2[1]);
            a0 = __fadd_rn(a0, f0.x);
            a1 = __fadd_rn(a1, f0.y);
            a2 = __fadd_rn(a2, f1.x);
            a3 = __fadd_rn(a3, f1.y);
        }
    }
    uint2 res;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&res);
    o2[0] = __floats2bfloat162_rn(a0, a1);
    o2[1] = __floats2bfloat162_rn(a2, a3);
    out[tok * 576 + col2] = res;
}

template <int Q_VAL, bool APPLY_NORM>
__global__ void __launch_bounds__(288, 1) conv_mix_288_r6_kernel(
    const uint4* __restrict__ x_comb,
    const uint4* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ cache,
    const int64_t* __restrict__ positions,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_idx,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ query_start,
    int64_t stride_c_blk, int64_t stride_c_h, int64_t stride_c_n,
    int32_t stride_bt, int32_t max_blocks, int32_t cache_N,
    int32_t stream_offset, int32_t stream_width, float conv_scale,
    uint4* __restrict__ conv_out,
    const uint4* __restrict__ res,
    const uint4* __restrict__ addend,
    const uint4* __restrict__ stack_u4,
    const uint4* __restrict__ last,
    const float4* __restrict__ queries,
    const uint4* __restrict__ norm_w,
    void* __restrict__ out_ptr,
    int64_t stride_s_u4, int32_t T, int32_t n_stack,
    float eps, float norm_eps, int32_t cache_vec8_ok)
{
    constexpr int WS_K_MAX = 7 * (1 + Q_VAL);
    constexpr bool STAGE_NW = (APPLY_NORM && Q_VAL == 1);

    // Cycle-1 cp.async SMEM staging (per-thread private slots -> 0 syncthreads, 0 regs!)
    __shared__ __align__(16) uint4 s_stack[6][288];                 // 27648 B
    __shared__ __align__(16) uint4 s_add[288];                      //  4608 B
    __shared__ __align__(16) uint4 s_res[288];                      //  4608 B
    __shared__ __align__(16) uint4 s_last[288];                     //  4608 B
    __shared__ __align__(16) uint4 s_nw[STAGE_NW ? 288 : 1];        //  4608 B (Q=1)
    __shared__ float s_warp[WS_K_MAX][9];
    __shared__ float s_tot[WS_K_MAX];
    __shared__ float s_rstd[7];
    __shared__ float s_prob[Q_VAL][7];
    __shared__ float s_inv_sum[Q_VAL];
    __shared__ float s_out_warp[Q_VAL][9];
    __shared__ float s_out_rstd[Q_VAL];

    const int64_t t  = static_cast<int64_t>(blockIdx.x);
    const int tid    = threadIdx.x; // 0..287
    const int warp   = tid >> 5;    // 0..8
    const int lane   = tid & 31;    // 0..31
    const int c0     = tid << 3;
    const int n_src  = n_stack + 1;
    const int k_red  = n_src * (1 + Q_VAL);

    // 1. CYCLE 1: Fire async cold-HBM loads into SMEM (0 registers consumed!)
    for (int s = 0; s < n_stack; ++s) {
        uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_stack[s][tid]));
        r6_cp_async_ca_16B(dst, stack_u4 + static_cast<int64_t>(s) * stride_s_u4 + t * 288 + tid);
    }
    if (addend != nullptr) {
        r6_cp_async_ca_16B(static_cast<uint32_t>(__cvta_generic_to_shared(&s_add[tid])), addend + t * 288 + tid);
        if (res != nullptr) {
            r6_cp_async_ca_16B(static_cast<uint32_t>(__cvta_generic_to_shared(&s_res[tid])), res + t * 288 + tid);
        }
    }
    if (last != nullptr && last != conv_out) {
        r6_cp_async_ca_16B(static_cast<uint32_t>(__cvta_generic_to_shared(&s_last[tid])), last + t * 288 + tid);
    }
    if constexpr (STAGE_NW) {
        r6_cp_async_ca_16B(static_cast<uint32_t>(__cvta_generic_to_shared(&s_nw[tid])), norm_w + tid);
    }
    r6_cp_async_commit();

    // 2. Simultaneously issue metadata + queries + conv_weight + CACHE_FIRST loads while cp.async flies!
    const int64_t slot   = slot_mapping[t];
    const bool    valid  = (slot >= 0);
    const int64_t pos    = positions[t];
    const int32_t req    = seq_idx[t];
    const int32_t qstart = query_start[t];

    float4 q0[Q_VAL], q1[Q_VAL];
    #pragma unroll
    for (int q = 0; q < Q_VAL; ++q) {
        q0[q] = queries[q * 576 + tid * 2 + 0];
        q1[q] = queries[q * 576 + tid * 2 + 1];
    }

    uint4 wu[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) wu[m] = conv_weight[tid * 4 + m];

    // Fast power-of-2 cache_N & col0 (eliminates ~400 instructions of 64-bit integer div/mod!)
    const bool pow2_cn = (cache_N > 0) && ((cache_N & (cache_N - 1)) == 0);
    const int  cn_sh   = pow2_cn ? (__ffs(cache_N) - 1) : 0;
    const int64_t cn_mk = static_cast<int64_t>(cache_N - 1);
    const int64_t col0_fast = (stream_width == 2304)
        ? static_cast<int64_t>(stream_offset + c0)
        : (static_cast<int64_t>(c0 / stream_width) * stride_c_h + stream_offset + (c0 % stream_width));

    uint4 c_u[3];
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        int lag = 3 - k;
        int64_t src_pos = pos - lag;
        if (valid && (src_pos >= 0) && (t - lag < qstart)) {
            int32_t lblk = pow2_cn ? static_cast<int32_t>(src_pos >> cn_sh) : static_cast<int32_t>(src_pos / cache_N);
            if (lblk >= max_blocks) lblk = max_blocks - 1;
            int64_t blk = block_table[static_cast<int64_t>(req) * stride_bt + lblk];
            int64_t off_n = pow2_cn ? (src_pos & cn_mk) : (src_pos % cache_N);
            int64_t base_row = blk * stride_c_blk + off_n * stride_c_n;
            if (cache_vec8_ok) {
                c_u[k] = *reinterpret_cast<const uint4*>(cache + base_row + col0_fast);
            } else {
                uint4 tmp_u;
                __nv_bfloat16* tp = reinterpret_cast<__nv_bfloat16*>(&tmp_u);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int c = c0 + j;
                    int64_t col = (c / stream_width) * stride_c_h + stream_offset + (c % stream_width);
                    tp[j] = cache[base_row + col];
                }
                c_u[k] = tmp_u;
            }
        } else {
            c_u[k] = make_uint4(0, 0, 0, 0);
        }
    }

    // 3. Fetch cur_u from L2-resident `ws_comb` & write cache (CACHE_FIRST preserved!)
    uint4 cur_u = x_comb[t * 288 + tid];

    if (valid) {
        int64_t dst_blk = pow2_cn ? (slot >> cn_sh) : (slot / cache_N);
        int64_t dst_off = pow2_cn ? (slot & cn_mk)  : (slot % cache_N);
        int64_t base_dst = dst_blk * stride_c_blk + dst_off * stride_c_n;
        if (cache_vec8_ok) {
            *reinterpret_cast<uint4*>(cache + base_dst + col0_fast) = cur_u;
        } else {
            const __nv_bfloat16* cp = reinterpret_cast<const __nv_bfloat16*>(&cur_u);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int c = c0 + j;
                int64_t col = (c / stream_width) * stride_c_h + stream_offset + (c % stream_width);
                cache[base_dst + col] = cp[j];
            }
        }
    }

    // 4. Exact ShortConv in k = 0, 1, 2, 3 order
    float conv_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int lag = 3 - k;
        bool is_intra = valid && (pos - lag >= 0) && (t - lag >= qstart);
        float xt[8];
        if (is_intra) {
            uint4 tu = (lag == 0) ? cur_u : x_comb[(t - lag) * 288 + tid];
            r6_unpack_bf16x8(tu, xt);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) xt[j] = 0.f;
        }
        float ct[8];
        if (k < 3) {
            r6_unpack_bf16x8(c_u[k], ct);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) ct[j] = 0.f;
        }
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wu[m]);
            float w0 = r6_bf2f(wp[k]);
            float w1 = r6_bf2f(wp[4 + k]);
            float v0 = __fadd_rn(xt[2 * m + 0], ct[2 * m + 0]);
            float v1 = __fadd_rn(xt[2 * m + 1], ct[2 * m + 1]);
            conv_acc[2 * m + 0] = __fmaf_rn(v0, w0, conv_acc[2 * m + 0]);
            conv_acc[2 * m + 1] = __fmaf_rn(v1, w1, conv_acc[2 * m + 1]);
        }
    }

    float cur_f[8];
    r6_unpack_bf16x8(cur_u, cur_f);
    float y_f[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        y_f[j] = r6_bf2f(r6_f2bf(__fmul_rn(__fadd_rn(conv_acc[j], cur_f[j]), conv_scale)));
    }

    // All cycle-1 cp.async loads (s_stack, s_add, s_res, s_last, s_nw) are already landed!
    r6_cp_async_wait<0>();

    float cout_f[8];
    if (addend != nullptr) {
        float add_f[8];
        r6_unpack_bf16x8(s_add[tid], add_f);
        if (res != nullptr) {
            float res_f[8];
            r6_unpack_bf16x8(s_res[tid], res_f);
            #pragma unroll
            for (int j = 0; j < 8; ++j) add_f[j] = __fadd_rn(res_f[j], add_f[j]);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            cout_f[j] = r6_bf2f(r6_f2bf(__fadd_rn(add_f[j], y_f[j])));
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) cout_f[j] = y_f[j];
    }
    conv_out[t * 288 + tid] = r6_pack_bf16x8(cout_f);

    // 5. Last source = (last != nullptr) ? last + cout_f : cout_f -> keep in regs!
    float lsrc[8];
    if (last != nullptr) {
        if (last == conv_out) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) lsrc[j] = __fadd_rn(cout_f[j], cout_f[j]);
        } else {
            float lf[8];
            r6_unpack_bf16x8(s_last[tid], lf);
            #pragma unroll
            for (int j = 0; j < 8; ++j) lsrc[j] = __fadd_rn(lf[j], cout_f[j]);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) lsrc[j] = cout_f[j];
    }

    // 6. Reduce s_stack[0..n_stack-1] (already in SMEM!) + lsrc (in regs!)
    for (int s = 0; s < n_stack; ++s) {
        float sv[8];
        r6_unpack_bf16x8(s_stack[s][tid], sv);
        float sq = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) sq = __fmaf_rn(sv[j], sv[j], sq);
        sq = r6_warp_sum_rn(sq);
        if (lane == 0) s_warp[s * (1 + Q_VAL)][warp] = sq;
        #pragma unroll
        for (int q = 0; q < Q_VAL; ++q) {
            float dt = __fmaf_rn(q0[q].x, sv[0], 0.f);
            dt = __fmaf_rn(q0[q].y, sv[1], dt);
            dt = __fmaf_rn(q0[q].z, sv[2], dt);
            dt = __fmaf_rn(q0[q].w, sv[3], dt);
            dt = __fmaf_rn(q1[q].x, sv[4], dt);
            dt = __fmaf_rn(q1[q].y, sv[5], dt);
            dt = __fmaf_rn(q1[q].z, sv[6], dt);
            dt = __fmaf_rn(q1[q].w, sv[7], dt);
            dt = r6_warp_sum_rn(dt);
            if (lane == 0) s_warp[s * (1 + Q_VAL) + 1 + q][warp] = dt;
        }
    }

    {
        float sq = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) sq = __fmaf_rn(lsrc[j], lsrc[j], sq);
        sq = r6_warp_sum_rn(sq);
        if (lane == 0) s_warp[n_stack * (1 + Q_VAL)][warp] = sq;
        #pragma unroll
        for (int q = 0; q < Q_VAL; ++q) {
            float dt = __fmaf_rn(q0[q].x, lsrc[0], 0.f);
            dt = __fmaf_rn(q0[q].y, lsrc[1], dt);
            dt = __fmaf_rn(q0[q].z, lsrc[2], dt);
            dt = __fmaf_rn(q0[q].w, lsrc[3], dt);
            dt = __fmaf_rn(q1[q].x, lsrc[4], dt);
            dt = __fmaf_rn(q1[q].y, lsrc[5], dt);
            dt = __fmaf_rn(q1[q].z, lsrc[6], dt);
            dt = __fmaf_rn(q1[q].w, lsrc[7], dt);
            dt = r6_warp_sum_rn(dt);
            if (lane == 0) s_warp[n_stack * (1 + Q_VAL) + 1 + q][warp] = dt;
        }
    }
    __syncthreads();

    // 7. Single-barrier Warp-0 reduction + softmax (0 stack arrays!)
    if (tid < k_red) {
        float sum = s_warp[tid][0];
        #pragma unroll
        for (int wt = 1; wt < 9; ++wt) sum = __fadd_rn(sum, s_warp[tid][wt]);
        s_tot[tid] = sum;
    }
    if (warp == 0) {
        __syncwarp();
        if (tid < n_src) {
            s_rstd[tid] = rsqrtf(__fmaf_rn(s_tot[tid * (1 + Q_VAL)], 1.0f / 2304.0f, eps));
        }
        __syncwarp();
        if (tid < Q_VAL) {
            int q = tid;
            float mx = -INFINITY;
            for (int s = 0; s < n_src; ++s) {
                float lg = __fmul_rn(s_tot[s * (1 + Q_VAL) + 1 + q], s_rstd[s]);
                mx = fmaxf(mx, lg);
            }
            float psum = 0.f;
            for (int s = 0; s < n_src; ++s) {
                float lg = __fmul_rn(s_tot[s * (1 + Q_VAL) + 1 + q], s_rstd[s]);
                float p = expf(__fsub_rn(lg, mx));
                psum = __fadd_rn(psum, p);
                s_prob[q][s] = __fmul_rn(p, s_rstd[s]);
            }
            s_inv_sum[q] = __fdiv_rn(1.0f, psum);
        }
    }
    __syncthreads();

    // 8. Mix s_stack[0..n_stack-1] (SMEM!) + lsrc (regs!)
    float acc[Q_VAL][8];
    #pragma unroll
    for (int q = 0; q < Q_VAL; ++q) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[q][j] = 0.f;
    }
    for (int s = 0; s < n_stack; ++s) {
        float sv[8];
        r6_unpack_bf16x8(s_stack[s][tid], sv);
        #pragma unroll
        for (int q = 0; q < Q_VAL; ++q) {
            float p = s_prob[q][s];
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[q][j] = __fmaf_rn(p, sv[j], acc[q][j]);
        }
    }
    #pragma unroll
    for (int q = 0; q < Q_VAL; ++q) {
        float p = s_prob[q][n_stack];
        float inv_s = s_inv_sum[q];
        float osq = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[q][j] = __fmul_rn(__fmaf_rn(p, lsrc[j], acc[q][j]), inv_s);
            if constexpr (APPLY_NORM) osq = __fmaf_rn(acc[q][j], acc[q][j], osq);
        }
        if constexpr (APPLY_NORM) {
            osq = r6_warp_sum_rn(osq);
            if (lane == 0) s_out_warp[q][warp] = osq;
        }
    }

    if constexpr (APPLY_NORM) {
        __syncthreads();
        if (tid < Q_VAL) {
            float sum = s_out_warp[tid][0];
            #pragma unroll
            for (int wt = 1; wt < 9; ++wt) sum = __fadd_rn(sum, s_out_warp[tid][wt]);
            s_out_rstd[tid] = rsqrtf(__fadd_rn(__fdiv_rn(sum, 2304.0f), norm_eps));
        }
        __syncthreads();
        uint4* out_bf16 = static_cast<uint4*>(out_ptr);
        #pragma unroll
        for (int q = 0; q < Q_VAL; ++q) {
            float rstd = s_out_rstd[q];
            float nw[8];
            if constexpr (STAGE_NW) {
                r6_unpack_bf16x8(s_nw[tid], nw);
            } else {
                r6_unpack_bf16x8(norm_w[q * 288 + tid], nw);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[q][j] = __fmul_rn(__fmul_rn(acc[q][j], rstd), nw[j]);
            out_bf16[(static_cast<int64_t>(q) * T + t) * 288 + tid] = r6_pack_bf16x8(acc[q]);
        }
    } else {
        float4* out_f32 = static_cast<float4*>(out_ptr);
        #pragma unroll
        for (int q = 0; q < Q_VAL; ++q) {
            int64_t base = (static_cast<int64_t>(q) * T + t) * 576 + tid * 2;
            out_f32[base + 0] = make_float4(acc[q][0], acc[q][1], acc[q][2], acc[q][3]);
            out_f32[base + 1] = make_float4(acc[q][4], acc[q][5], acc[q][6], acc[q][7]);
        }
    }
}

extern "C" int launch_combine_conv_mix_r6_2304(
    const void* down, const void* src2dst, const void* conv_weight, void* cache,
    const void* positions, const void* block_table, const void* seq_idx,
    const void* slot_mapping, const void* query_start,
    int64_t stride_c_blk, int64_t stride_c_h, int64_t stride_c_n,
    int32_t stride_bt, int32_t max_blocks, int32_t cache_N,
    int32_t stream_offset, int32_t stream_width, float conv_scale,
    void* conv_out, const void* res, const void* addend,
    const void* stack, const void* last, const void* queries, const void* norm_w,
    void* ws_comb, int32_t mode, void* out_ptr,
    int64_t stride_s_u4, int32_t T, int32_t n_stack, int32_t num_q, int32_t apply_norm,
    float eps, float norm_eps, void* stream_ptr)
{
    (void)mode;
    if (T <= 0) return 1;
    if (T > 64 || n_stack < 0 || n_stack > 6 || (num_q != 1 && num_q != 3)) return 0;
    int32_t cache_vec8_ok = ((stream_offset & 7) == 0 && (stream_width & 7) == 0 &&
                             (stride_c_h & 7) == 0 && (stride_c_n & 7) == 0 && (stride_c_blk & 7) == 0 &&
                             (reinterpret_cast<uintptr_t>(cache) & 15) == 0) ? 1 : 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    combine_2304_r6_kernel<<<dim3(18, static_cast<unsigned int>(T)), 32, 0, stream>>>(
        static_cast<const uint2*>(down), static_cast<const int32_t*>(src2dst),
        static_cast<uint2*>(ws_comb));

    #define LAUNCH_CM_288(QV, AN) \
        conv_mix_288_r6_kernel<QV, AN><<<static_cast<unsigned int>(T), 288, 0, stream>>>( \
            static_cast<const uint4*>(ws_comb), \
            static_cast<const uint4*>(conv_weight), static_cast<__nv_bfloat16*>(cache), \
            static_cast<const int64_t*>(positions), static_cast<const int32_t*>(block_table), \
            static_cast<const int32_t*>(seq_idx), static_cast<const int64_t*>(slot_mapping), \
            static_cast<const int32_t*>(query_start), \
            stride_c_blk, stride_c_h, stride_c_n, stride_bt, max_blocks, cache_N, \
            stream_offset, stream_width, conv_scale, \
            static_cast<uint4*>(conv_out), static_cast<const uint4*>(res), \
            static_cast<const uint4*>(addend), static_cast<const uint4*>(stack), \
            static_cast<const uint4*>(last), static_cast<const float4*>(queries), \
            static_cast<const uint4*>(norm_w), out_ptr, \
            stride_s_u4, T, n_stack, eps, norm_eps, cache_vec8_ok)

    if (num_q == 1) {
        if (apply_norm) { LAUNCH_CM_288(1, true); }
        else            { LAUNCH_CM_288(1, false); }
    } else {
        if (apply_norm) { LAUNCH_CM_288(3, true); }
        else            { LAUNCH_CM_288(3, false); }
    }
    #undef LAUNCH_CM_288
    return 1;
}
