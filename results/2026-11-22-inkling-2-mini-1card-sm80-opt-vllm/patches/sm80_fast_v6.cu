#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// ============================================================================
// Helpers (SM80 native bf16 + IEEE _rn intrinsics)
// ============================================================================
__device__ __forceinline__ float bf2f(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __nv_bfloat16 f2bf(float x) {
    return __float2bfloat16_rn(x);
}
__device__ __forceinline__ float round_bf16_f32(float x) {
    uint32_t bits = __float_as_uint(x);
    bits = (bits + 0x7FFFU + ((bits >> 16) & 1U)) & 0xFFFF0000U;
    return __uint_as_float(bits);
}
__device__ __forceinline__ void unpack_bf16x8(uint4 u, float (&dst)[8]) {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&u);
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = __bfloat162float(p[i]);
}
__device__ __forceinline__ uint4 pack_bf16x8(const float (&src)[8]) {
    uint4 u;
    __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(&u);
    #pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = __float2bfloat16_rn(src[i]);
    return u;
}
__device__ __forceinline__ float warp_sum_rn(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// ============================================================================
// 1. Round-1 MoE Align Block Size (preserved)
// ============================================================================
__global__ void __launch_bounds__(256) moe_align_small_sm80_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t numel,
    const int32_t block_size,
    const int32_t num_experts,
    const int32_t max_pad,
    const int32_t n_blocks,
    const int32_t ignore_invalid)
{
    __shared__ int32_t s_counts[256];
    __shared__ int32_t s_pad_start[256];
    __shared__ int32_t s_pad_end[256];
    __shared__ int32_t s_write_off[256];
    __shared__ int16_t s_ids[512];

    const int tid = threadIdx.x;
    s_counts[tid] = 0;
    __syncthreads();

    for (int i = tid; i < numel; i += blockDim.x) {
        int32_t eid = topk_ids[i];
        bool valid = ignore_invalid ? (eid >= 0 && eid < num_experts) : true;
        if (valid) {
            s_ids[i] = static_cast<int16_t>(eid);
            atomicAdd(&s_counts[eid], 1);
        } else {
            s_ids[i] = static_cast<int16_t>(-1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        int32_t running = 0;
        for (int e = 0; e < num_experts; ++e) {
            int32_t c = s_counts[e];
            int32_t padded = ((c + block_size - 1) / block_size) * block_size;
            s_pad_start[e] = running;
            s_write_off[e] = running;
            running += padded;
            s_pad_end[e] = running;
        }
        *total_tokens_post_pad = running;
    }
    __syncthreads();

    for (int i = tid; i < max_pad; i += blockDim.x) {
        sorted_ids[i] = numel;
    }
    __syncthreads();

    if (tid < num_experts && s_counts[tid] > 0) {
        int32_t out_pos = s_write_off[tid];
        for (int i = 0; i < numel; ++i) {
            if (s_ids[i] == tid) {
                sorted_ids[out_pos++] = i;
            }
        }
        int32_t b_start = s_pad_start[tid] / block_size;
        int32_t b_end = s_pad_end[tid] / block_size;
        for (int32_t b = b_start; b < b_end; ++b) {
            if (b < n_blocks) expert_ids[b] = tid;
        }
    }
    const int32_t total_blocks = (num_experts > 0) ? (s_pad_end[num_experts - 1] / block_size) : 0;
    for (int32_t b = total_blocks + tid; b < n_blocks; b += blockDim.x) {
        expert_ids[b] = 0;
    }
}

__global__ void __launch_bounds__(256) moe_align_large_part1_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ cta_counts,
    const int32_t numel,
    const int32_t num_experts,
    const int32_t max_pad,
    const int32_t ignore_invalid,
    const int32_t items_per_cta)
{
    __shared__ int32_t s_counts[256];
    const int tid = threadIdx.x;
    const int cid = blockIdx.x;
    const int gtid = cid * blockDim.x + tid;
    const int gstride = gridDim.x * blockDim.x;

    for (int i = gtid; i < max_pad; i += gstride) sorted_ids[i] = numel;
    s_counts[tid] = 0;
    __syncthreads();

    const int start = cid * items_per_cta;
    int end = start + items_per_cta;
    if (end > numel) end = numel;
    for (int i = start + tid; i < end; i += blockDim.x) {
        int32_t eid = topk_ids[i];
        bool valid = ignore_invalid ? (eid >= 0 && eid < num_experts) : true;
        if (valid) atomicAdd(&s_counts[eid], 1);
    }
    __syncthreads();
    if (tid < num_experts) cta_counts[cid * 256 + tid] = s_counts[tid];
}

__global__ void __launch_bounds__(256) moe_align_large_part2_kernel(
    const int32_t* __restrict__ cta_counts,
    int32_t* __restrict__ cta_offsets,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t num_ctas,
    const int32_t block_size,
    const int32_t num_experts,
    const int32_t n_blocks)
{
    __shared__ int32_t s_tot[256];
    __shared__ int32_t s_pad_start[256];
    __shared__ int32_t s_pad_end[256];
    const int tid = threadIdx.x;
    int sum = 0;
    if (tid < num_experts) {
        for (int c = 0; c < num_ctas; ++c) sum += cta_counts[c * 256 + tid];
    }
    s_tot[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        int running = 0;
        for (int e = 0; e < num_experts; ++e) {
            int c = s_tot[e];
            int padded = ((c + block_size - 1) / block_size) * block_size;
            s_pad_start[e] = running;
            running += padded;
            s_pad_end[e] = running;
        }
        *total_tokens_post_pad = running;
    }
    __syncthreads();
    if (tid < num_experts) {
        int off = s_pad_start[tid];
        for (int c = 0; c < num_ctas; ++c) {
            cta_offsets[c * 256 + tid] = off;
            off += cta_counts[c * 256 + tid];
        }
        int b_start = s_pad_start[tid] / block_size;
        int b_end = s_pad_end[tid] / block_size;
        for (int b = b_start; b < b_end; ++b) {
            if (b < n_blocks) expert_ids[b] = tid;
        }
    }
    const int total_blocks = s_pad_end[num_experts - 1] / block_size;
    for (int b = total_blocks + tid; b < n_blocks; b += blockDim.x) expert_ids[b] = 0;
}

__global__ void __launch_bounds__(256) moe_align_large_part3_kernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ cta_counts,
    const int32_t* __restrict__ cta_offsets,
    int32_t* __restrict__ sorted_ids,
    const int32_t numel,
    const int32_t num_experts,
    const int32_t ignore_invalid,
    const int32_t items_per_cta)
{
    __shared__ int16_t s_tile[2048];
    const int tid = threadIdx.x;
    const int cid = blockIdx.x;
    const int start = cid * items_per_cta;
    if (start >= numel) return;
    int end = start + items_per_cta;
    if (end > numel) end = numel;
    const bool active = (tid < num_experts) && (cta_counts[cid * 256 + tid] > 0);
    int my_out = (tid < num_experts) ? cta_offsets[cid * 256 + tid] : 0;
    for (int base = start; base < end; base += 2048) {
        int chunk = end - base;
        if (chunk > 2048) chunk = 2048;
        for (int i = tid; i < chunk; i += blockDim.x) {
            int32_t v = topk_ids[base + i];
            bool valid = ignore_invalid ? (v >= 0 && v < num_experts) : true;
            s_tile[i] = valid ? static_cast<int16_t>(v) : static_cast<int16_t>(-1);
        }
        __syncthreads();
        if (active) {
            #pragma unroll 8
            for (int i = 0; i < chunk; ++i) {
                if (s_tile[i] == tid) sorted_ids[my_out++] = base + i;
            }
        }
        __syncthreads();
    }
}

extern "C" void launch_moe_align_block_size_sm80(
    const void* topk_ids, void* sorted_ids, void* expert_ids, void* total_tokens_post_pad,
    void* workspace, int32_t numel, int32_t block_size, int32_t num_experts,
    int32_t max_pad, int32_t n_blocks, int32_t ignore_invalid, void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    if (numel <= 512) {
        moe_align_small_sm80_kernel<<<1, 256, 0, stream>>>(
            static_cast<const int32_t*>(topk_ids), static_cast<int32_t*>(sorted_ids),
            static_cast<int32_t*>(expert_ids), static_cast<int32_t*>(total_tokens_post_pad),
            numel, block_size, num_experts, max_pad, n_blocks, ignore_invalid);
    } else {
        int32_t num_ctas = (numel + 255) / 256;
        if (num_ctas > 64) num_ctas = 64;
        int32_t items_per_cta = (numel + num_ctas - 1) / num_ctas;
        int32_t* cta_counts = static_cast<int32_t*>(workspace);
        int32_t* cta_offsets = cta_counts + 64 * 256;
        moe_align_large_part1_kernel<<<num_ctas, 256, 0, stream>>>(
            static_cast<const int32_t*>(topk_ids), static_cast<int32_t*>(sorted_ids),
            cta_counts, numel, num_experts, max_pad, ignore_invalid, items_per_cta);
        moe_align_large_part2_kernel<<<1, 256, 0, stream>>>(
            cta_counts, cta_offsets, static_cast<int32_t*>(expert_ids),
            static_cast<int32_t*>(total_tokens_post_pad), num_ctas, block_size, num_experts, n_blocks);
        moe_align_large_part3_kernel<<<num_ctas, 256, 0, stream>>>(
            static_cast<const int32_t*>(topk_ids), cta_counts, cta_offsets,
            static_cast<int32_t*>(sorted_ids), numel, num_experts, ignore_invalid, items_per_cta);
    }
}

// ============================================================================
// 2. Fused Select Experts + MoE Align (1 CTA for T <= 256, E=128, TOPK=8)
// ============================================================================
__device__ __forceinline__ void align_from_s_ids_128(
    const int16_t* s_ids, const int32_t* s_counts,
    int32_t* s_pad_start, int32_t* s_pad_end,
    int32_t* sorted_ids, int32_t* expert_ids, int32_t* total_tokens_post_pad,
    int32_t numel, int32_t block_size, int32_t num_local_experts,
    int32_t max_pad, int32_t n_blocks)
{
    const int tid = threadIdx.x;
    if (tid == 0) {
        int32_t running = 0;
        for (int e = 0; e < num_local_experts; ++e) {
            int32_t c = s_counts[e];
            int32_t padded = ((c + block_size - 1) / block_size) * block_size;
            s_pad_start[e] = running;
            running += padded;
            s_pad_end[e] = running;
        }
        *total_tokens_post_pad = running;
    }
    for (int i = tid; i < max_pad; i += blockDim.x) sorted_ids[i] = numel;
    __syncthreads();

    if (tid < num_local_experts && s_counts[tid] > 0) {
        int32_t out_pos = s_pad_start[tid];
        for (int i = 0; i < numel; ++i) {
            if (s_ids[i] == tid) sorted_ids[out_pos++] = i;
        }
        int32_t b_start = s_pad_start[tid] / block_size;
        int32_t b_end = s_pad_end[tid] / block_size;
        for (int32_t b = b_start; b < b_end; ++b) {
            if (b < n_blocks) expert_ids[b] = tid;
        }
    }
    const int32_t total_blocks = (num_local_experts > 0) ? (s_pad_end[num_local_experts - 1] / block_size) : 0;
    for (int32_t b = total_blocks + tid; b < n_blocks; b += blockDim.x) expert_ids[b] = 0;
}

template <int GROUPS>
__global__ void __launch_bounds__(256) select_and_align_128_sm80_kernel(
    const float* __restrict__ partials,       // [GROUPS, T, 128]
    const float* __restrict__ bias,           // [128]
    const int32_t* __restrict__ expert_bounds,// [2]
    const uint8_t* __restrict__ is_padding,   // [T] or nullptr
    int32_t* __restrict__ ids_out,            // [T * 8]
    float* __restrict__ weights_out,          // [T * 8]
    int32_t* __restrict__ src2dst_out,        // [T * 8]
    int32_t* __restrict__ sorted_1, int32_t* __restrict__ exp_1, int32_t* __restrict__ pad_1,
    int32_t bm1, int32_t max_pad1, int32_t n_blk1,
    int32_t* __restrict__ sorted_2, int32_t* __restrict__ exp_2, int32_t* __restrict__ pad_2,
    int32_t bm2, int32_t max_pad2, int32_t n_blk2,
    int32_t T, float route_scale, int32_t num_local_experts)
{
    __shared__ int32_t s_counts[128];
    __shared__ int32_t s_pad_start[128];
    __shared__ int32_t s_pad_end[128];
    __shared__ int16_t s_ids[2048];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (tid < 128) s_counts[tid] = 0;
    __syncthreads();

    const int32_t lo_exp = expert_bounds[0];
    const int32_t hi_exp = expert_bounds[1];
    float b_reg[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) b_reg[i] = bias[lane * 4 + i];

    for (int t = warp; t < T; t += 8) {
        float logits[4];
        uint64_t keys[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int e = lane * 4 + i;
            float sum = partials[t * 128 + e];
            #pragma unroll
            for (int g = 1; g < GROUPS; ++g) {
                sum = __fadd_rn(sum, partials[(g * T + t) * 128 + e]);
            }
            logits[i] = sum;
            float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-sum)));
            float sel = __fadd_rn(sig, b_reg[i]);
            uint32_t bits = __float_as_uint(sel);
            bits ^= (bits & 0x80000000U) ? 0xFFFFFFFFU : 0x80000000U;
            keys[i] = (static_cast<uint64_t>(bits) << 16) | static_cast<uint64_t>(128 - e);
        }

        int my_chosen_id = 0;
        float my_chosen_logit = 0.f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint64_t best = keys[0];
            #pragma unroll
            for (int i = 1; i < 4; ++i) if (keys[i] > best) best = keys[i];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                uint32_t lo = __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(best), o);
                uint32_t hi = __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(best >> 32), o);
                uint64_t other = (static_cast<uint64_t>(hi) << 32) | lo;
                if (other > best) best = other;
            }
            int win_e = 128 - static_cast<int>(best & 0xFFFFU);
            int owner_lane = win_e >> 2;
            int sub = win_e & 3;
            float sub_logit = (sub == 0) ? logits[0] : ((sub == 1) ? logits[1] : ((sub == 2) ? logits[2] : logits[3]));
            float win_logit = __shfl_sync(0xffffffffu, sub_logit, owner_lane);
            if (lane == owner_lane) {
                if (sub == 0) keys[0] = 0;
                else if (sub == 1) keys[1] = 0;
                else if (sub == 2) keys[2] = 0;
                else keys[3] = 0;
            }
            if (lane == k) {
                my_chosen_id = win_e;
                my_chosen_logit = win_logit;
            }
        }

        float lp = (lane < 8) ? (fminf(my_chosen_logit, 0.0f) - log1pf(expf(-fabsf(my_chosen_logit)))) : -INFINITY;
        float max_lp = lp;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            max_lp = fmaxf(max_lp, __shfl_xor_sync(0xffffffffu, max_lp, o));
        }
        max_lp = __shfl_sync(0xffffffffu, max_lp, 0);
        float sh = (lane < 8) ? expf(lp - max_lp) : 0.f;
        float s0 = __shfl_sync(0xffffffffu, sh, 0);
        float s1 = __shfl_sync(0xffffffffu, sh, 1);
        float s2 = __shfl_sync(0xffffffffu, sh, 2);
        float s3 = __shfl_sync(0xffffffffu, sh, 3);
        float s4 = __shfl_sync(0xffffffffu, sh, 4);
        float s5 = __shfl_sync(0xffffffffu, sh, 5);
        float s6 = __shfl_sync(0xffffffffu, sh, 6);
        float s7 = __shfl_sync(0xffffffffu, sh, 7);
        float tot = __fadd_rn(__fadd_rn(__fadd_rn(s0, s2), __fadd_rn(s1, s3)),
                              __fadd_rn(__fadd_rn(s4, s6), __fadd_rn(s5, s7)));
        float lse = __fadd_rn(max_lp, logf(tot));

        bool pad = (is_padding != nullptr) ? (is_padding[t] != 0) : false;
        if (lane < 8) {
            int row = t * 8 + lane;
            float wt = __fmul_rn(expf(lp - lse), route_scale);
            int32_t loc = (!pad && my_chosen_id >= lo_exp && my_chosen_id < hi_exp) ? (my_chosen_id - lo_exp) : -1;
            ids_out[row] = loc;
            weights_out[row] = wt;
            src2dst_out[row] = (loc >= 0) ? row : -1;
            if (loc >= 0 && loc < num_local_experts) {
                s_ids[row] = static_cast<int16_t>(loc);
                atomicAdd(&s_counts[loc], 1);
            } else {
                s_ids[row] = static_cast<int16_t>(-1);
            }
        }
    }
    __syncthreads();

    const int32_t numel = T * 8;
    align_from_s_ids_128(s_ids, s_counts, s_pad_start, s_pad_end,
                         sorted_1, exp_1, pad_1, numel, bm1, num_local_experts, max_pad1, n_blk1);
    if (bm2 > 0) {
        __syncthreads();
        align_from_s_ids_128(s_ids, s_counts, s_pad_start, s_pad_end,
                             sorted_2, exp_2, pad_2, numel, bm2, num_local_experts, max_pad2, n_blk2);
    }
}

extern "C" void launch_select_and_align_128_sm80(
    const void* partials, const void* bias, const void* expert_bounds, const void* is_padding,
    void* ids_out, void* weights_out, void* src2dst_out,
    void* sorted_1, void* exp_1, void* pad_1, int32_t bm1, int32_t max_pad1, int32_t n_blk1,
    void* sorted_2, void* exp_2, void* pad_2, int32_t bm2, int32_t max_pad2, int32_t n_blk2,
    int32_t T, int32_t groups, float route_scale, int32_t num_local_experts, void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    if (groups == 3) {
        select_and_align_128_sm80_kernel<3><<<1, 256, 0, stream>>>(
            static_cast<const float*>(partials), static_cast<const float*>(bias),
            static_cast<const int32_t*>(expert_bounds), static_cast<const uint8_t*>(is_padding),
            static_cast<int32_t*>(ids_out), static_cast<float*>(weights_out), static_cast<int32_t*>(src2dst_out),
            static_cast<int32_t*>(sorted_1), static_cast<int32_t*>(exp_1), static_cast<int32_t*>(pad_1), bm1, max_pad1, n_blk1,
            static_cast<int32_t*>(sorted_2), static_cast<int32_t*>(exp_2), static_cast<int32_t*>(pad_2), bm2, max_pad2, n_blk2,
            T, route_scale, num_local_experts);
    } else if (groups == 1) {
        select_and_align_128_sm80_kernel<1><<<1, 256, 0, stream>>>(
            static_cast<const float*>(partials), static_cast<const float*>(bias),
            static_cast<const int32_t*>(expert_bounds), static_cast<const uint8_t*>(is_padding),
            static_cast<int32_t*>(ids_out), static_cast<float*>(weights_out), static_cast<int32_t*>(src2dst_out),
            static_cast<int32_t*>(sorted_1), static_cast<int32_t*>(exp_1), static_cast<int32_t*>(pad_1), bm1, max_pad1, n_blk1,
            static_cast<int32_t*>(sorted_2), static_cast<int32_t*>(exp_2), static_cast<int32_t*>(pad_2), bm2, max_pad2, n_blk2,
            T, route_scale, num_local_experts);
    }
}

// ============================================================================
// 3. Bitwise-exact Softcapped SwiGLU (N=1152) & Coalesced MoE Combine (H=2304)
// ============================================================================
__device__ __forceinline__ float sigmoid_approx_ptx(float x) {
    float out;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }" : "=f"(out) : "f"(x));
    return out;
}
__device__ __forceinline__ float tanh_approx_ptx(float x) {
    float out;
    asm("tanh.approx.f32 %0, %1;" : "=f"(out) : "f"(x));
    return out;
}
__device__ __forceinline__ float div_rn_25(float x) {
    constexpr float inv25 = 1.0f / 25.0f;
    float q = __fmul_rn(x, inv25);
    float res = fmaxf(__fmaf_rn(q, 25.0f, -x), -3.402823466e+38f);
    return __fmaf_rn(-res, inv25, q);
}

__global__ void __launch_bounds__(288) swiglu_1152_sm80_kernel(
    const uint4* __restrict__ gate_up,
    const float* __restrict__ weights,
    const int32_t* __restrict__ src2dst,
    uint2* __restrict__ out)
{
    const int row = blockIdx.x;
    if (src2dst[row] < 0) return;
    const int tid = threadIdx.x;
    const float gamma = weights[row];
    uint4 u = gate_up[static_cast<int64_t>(row) * 288 + tid];
    uint32_t pairs[4] = {u.x, u.y, u.z, u.w};
    uint2 res;
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(&res);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float g = __uint_as_float(pairs[i] << 16);
        float up = __uint_as_float(pairs[i] & 0xFFFF0000U);
        float sig = sigmoid_approx_ptx(g);
        float gc = __fmul_rn(4.0f, tanh_approx_ptx(__fmul_rn(g, 0.25f)));
        float uc = __fmul_rn(25.0f, tanh_approx_ptx(div_rn_25(up)));
        float val = __fmul_rn(__fmul_rn(__fmul_rn(gc, sig), uc), gamma);
        dst[i] = __float2bfloat16_rn(val);
    }
    out[static_cast<int64_t>(row) * 288 + tid] = res;
}

extern "C" void launch_swiglu_1152_sm80(
    const void* gate_up, const void* weights, const void* src2dst, void* out,
    int32_t rows, void* stream_ptr)
{
    if (rows <= 0) return;
    swiglu_1152_sm80_kernel<<<rows, 288, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const uint4*>(gate_up), static_cast<const float*>(weights),
        static_cast<const int32_t*>(src2dst), static_cast<uint2*>(out));
}

__global__ void __launch_bounds__(96) combine_2304_bf16_sm80_kernel(
    const uint4* __restrict__ down,
    const int32_t* __restrict__ src2dst,
    uint4* __restrict__ out)
{
    const int64_t tok = blockIdx.x;
    const int col_vec = blockIdx.y * 96 + threadIdx.x;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        int32_t r = src2dst[tok * 8 + slot];
        if (r >= 0) {
            float v[8];
            unpack_bf16x8(down[static_cast<int64_t>(r) * 288 + col_vec], v);
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] = __fadd_rn(acc[j], v[j]);
        }
    }
    out[tok * 288 + col_vec] = pack_bf16x8(acc);
}

extern "C" void launch_combine_2304_bf16_sm80(
    const void* down, const void* src2dst, void* out, int32_t tokens, void* stream_ptr)
{
    if (tokens <= 0) return;
    dim3 grid(tokens, 3);
    combine_2304_bf16_sm80_kernel<<<grid, 96, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const uint4*>(down), static_cast<const int32_t*>(src2dst),
        static_cast<uint4*>(out));
}

// ============================================================================
// 4. 1-Pass Register Multi-Query AttnRes + Fused ShortConv+AttnRes (D=2304)
// ============================================================================
template <int N_STACK, int Q, bool APPLY_NORM>
__device__ __forceinline__ void attn_res_epilogue_2304(
    float (&src)[N_STACK + 1][8],
    const float* __restrict__ queries,
    const uint4* __restrict__ norm_w,
    void* __restrict__ out_ptr,
    int64_t t, int64_t T, float eps, float norm_eps, bool normalize_values)
{
    constexpr int N_SRC = N_STACK + 1;
    constexpr int K_RED = N_SRC * (1 + Q);
    __shared__ float s_warp[K_RED][9];
    __shared__ float s_tot[K_RED];
    __shared__ float s_prob[Q][N_SRC];
    __shared__ float s_inv_sum[Q];
    __shared__ float s_out_warp[Q][9];
    __shared__ float s_out_rstd[Q];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float q_reg[Q][8];
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        float4 f0 = reinterpret_cast<const float4*>(queries + q * 2304)[tid * 2 + 0];
        float4 f1 = reinterpret_cast<const float4*>(queries + q * 2304)[tid * 2 + 1];
        q_reg[q][0] = f0.x; q_reg[q][1] = f0.y; q_reg[q][2] = f0.z; q_reg[q][3] = f0.w;
        q_reg[q][4] = f1.x; q_reg[q][5] = f1.y; q_reg[q][6] = f1.z; q_reg[q][7] = f1.w;
    }

    float vals[K_RED];
    #pragma unroll
    for (int s = 0; s < N_SRC; ++s) {
        float sq = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) sq = __fmaf_rn(src[s][j], src[s][j], sq);
        vals[s * (1 + Q)] = warp_sum_rn(sq);
        #pragma unroll
        for (int q = 0; q < Q; ++q) {
            float dt = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) dt = __fmaf_rn(q_reg[q][j], src[s][j], dt);
            vals[s * (1 + Q) + 1 + q] = warp_sum_rn(dt);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < K_RED; ++k) s_warp[k][warp] = vals[k];
    }
    __syncthreads();

    if (tid < K_RED) {
        float sum = s_warp[tid][0];
        #pragma unroll
        for (int w = 1; w < 9; ++w) sum = __fadd_rn(sum, s_warp[tid][w]);
        s_tot[tid] = sum;
    }
    __syncthreads();

    if (tid < Q) {
        int q = tid;
        float rstd[N_SRC];
        float logits[N_SRC];
        float mx = -INFINITY;
        #pragma unroll
        for (int s = 0; s < N_SRC; ++s) {
            float rs = rsqrtf(__fmaf_rn(s_tot[s * (1 + Q)], 1.0f / 2304.0f, eps));
            rstd[s] = rs;
            float lg = __fmul_rn(s_tot[s * (1 + Q) + 1 + q], rs);
            logits[s] = lg;
            mx = fmaxf(mx, lg);
        }
        float psum = 0.f;
        #pragma unroll
        for (int s = 0; s < N_SRC; ++s) {
            float p = expf(__fsub_rn(logits[s], mx));
            psum = __fadd_rn(psum, p);
            s_prob[q][s] = normalize_values ? __fmul_rn(p, rstd[s]) : p;
        }
        s_inv_sum[q] = __fdiv_rn(1.0f, psum);
    }
    __syncthreads();

    float acc[Q][8];
    float out_sq_w[Q];
    #pragma unroll
    for (int q = 0; q < Q; ++q) {
        float inv_s = s_inv_sum[q];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[q][j] = 0.f;
        #pragma unroll
        for (int s = 0; s < N_SRC; ++s) {
            float p = s_prob[q][s];
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[q][j] = __fmaf_rn(p, src[s][j], acc[q][j]);
        }
        float osq = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[q][j] = __fmul_rn(acc[q][j], inv_s);
            if constexpr (APPLY_NORM) osq = __fmaf_rn(acc[q][j], acc[q][j], osq);
        }
        if constexpr (APPLY_NORM) out_sq_w[q] = warp_sum_rn(osq);
    }

    if constexpr (APPLY_NORM) {
        if (lane == 0) {
            #pragma unroll
            for (int q = 0; q < Q; ++q) s_out_warp[q][warp] = out_sq_w[q];
        }
        __syncthreads();
        if (tid < Q) {
            float sum = s_out_warp[tid][0];
            #pragma unroll
            for (int w = 1; w < 9; ++w) sum = __fadd_rn(sum, s_out_warp[tid][w]);
            s_out_rstd[tid] = rsqrtf(__fadd_rn(__fdiv_rn(sum, 2304.0f), norm_eps));
        }
        __syncthreads();

        uint4* out_bf16 = static_cast<uint4*>(out_ptr);
        #pragma unroll
        for (int q = 0; q < Q; ++q) {
            float rstd = s_out_rstd[q];
            float nw[8];
            unpack_bf16x8(norm_w[q * 288 + tid], nw);
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[q][j] = __fmul_rn(__fmul_rn(acc[q][j], rstd), nw[j]);
            out_bf16[(q * T + t) * 288 + tid] = pack_bf16x8(acc[q]);
        }
    } else {
        float4* out_f32 = static_cast<float4*>(out_ptr);
        #pragma unroll
        for (int q = 0; q < Q; ++q) {
            int64_t base = (q * T + t) * 576 + tid * 2;
            out_f32[base + 0] = make_float4(acc[q][0], acc[q][1], acc[q][2], acc[q][3]);
            out_f32[base + 1] = make_float4(acc[q][4], acc[q][5], acc[q][6], acc[q][7]);
        }
    }
}

template <int N_STACK, int Q, bool HAS_ADDEND, bool APPLY_NORM>
__global__ void __launch_bounds__(288) attn_res_2304_sm80_kernel(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ last,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ queries,
    const uint4* __restrict__ norm_w,
    void* __restrict__ out_ptr,
    int64_t stride_s, int64_t T, float eps, float norm_eps, bool normalize_values)
{
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    float src[N_STACK + 1][8];
    #pragma unroll
    for (int s = 0; s < N_STACK; ++s) {
        const uint4* sp = reinterpret_cast<const uint4*>(stack + s * stride_s + t * 2304);
        unpack_bf16x8(sp[tid], src[s]);
    }
    unpack_bf16x8(reinterpret_cast<const uint4*>(last)[t * 288 + tid], src[N_STACK]);
    if constexpr (HAS_ADDEND) {
        float a[8];
        unpack_bf16x8(reinterpret_cast<const uint4*>(addend)[t * 288 + tid], a);
        #pragma unroll
        for (int j = 0; j < 8; ++j) src[N_STACK][j] = __fadd_rn(src[N_STACK][j], a[j]);
    }
    attn_res_epilogue_2304<N_STACK, Q, APPLY_NORM>(
        src, queries, norm_w, out_ptr, t, T, eps, norm_eps, normalize_values);
}

template <int N_STACK, int Q, bool APPLY_NORM>
__global__ void __launch_bounds__(288) conv_mix_2304_sm80_kernel(
    const uint4* __restrict__ x,
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
    const __nv_bfloat16* __restrict__ stack,
    const uint4* __restrict__ last,
    const float* __restrict__ queries,
    const uint4* __restrict__ norm_w,
    void* __restrict__ out_ptr,
    int64_t stride_s, int64_t T, float eps, float norm_eps, bool cache_vec8_ok)
{
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    const int c0 = tid * 8;

    const int64_t slot = slot_mapping[t];
    const bool valid = (slot >= 0);
    const int64_t pos = positions[t];
    const int32_t req = seq_idx[t];
    const int32_t qstart = query_start[t];

    // Preload earlier steps taps from cache BEFORE writing cur to cache (CACHE_FIRST order)
    float cached_tap[4][8];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int lag = 3 - k;
        int64_t src_pos = pos - lag;
        bool is_cached = valid && (src_pos >= 0) && (t - lag < qstart);
        if (is_cached) {
            int64_t safe_src = src_pos;
            int32_t lblk = static_cast<int32_t>(safe_src / cache_N);
            if (lblk >= max_blocks) lblk = max_blocks - 1;
            int64_t blk = block_table[req * stride_bt + lblk];
            int64_t off_n = safe_src % cache_N;
            int64_t base_row = blk * stride_c_blk + off_n * stride_c_n;
            if (cache_vec8_ok) {
                int64_t col0 = (c0 / stream_width) * stride_c_h + stream_offset + (c0 % stream_width);
                uint4 cu = *reinterpret_cast<const uint4*>(cache + base_row + col0);
                unpack_bf16x8(cu, cached_tap[k]);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int c = c0 + j;
                    int64_t col = (c / stream_width) * stride_c_h + stream_offset + (c % stream_width);
                    cached_tap[k][j] = bf2f(cache[base_row + col]);
                }
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) cached_tap[k][j] = 0.f;
        }
    }

    uint4 cur_u = x[t * 288 + tid];
    float cur_f[8];
    unpack_bf16x8(cur_u, cur_f);
    if (valid) {
        int64_t base_dst = (slot / cache_N) * stride_c_blk + (slot % cache_N) * stride_c_n;
        if (cache_vec8_ok) {
            int64_t col0 = (c0 / stream_width) * stride_c_h + stream_offset + (c0 % stream_width);
            *reinterpret_cast<uint4*>(cache + base_dst + col0) = cur_u;
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

    float w_f[8][4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        float tmp[8];
        unpack_bf16x8(conv_weight[tid * 4 + m], tmp);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            w_f[2 * m + 0][k] = tmp[k];
            w_f[2 * m + 1][k] = tmp[4 + k];
        }
    }

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int lag = 3 - k;
        bool is_intra = valid && (pos - lag >= 0) && (t - lag >= qstart);
        float xt[8];
        if (is_intra) {
            unpack_bf16x8(x[(t - lag) * 288 + tid], xt);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) xt[j] = 0.f;
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float val = __fadd_rn(xt[j], cached_tap[k][j]);
            acc[j] = __fmaf_rn(val, w_f[j][k], acc[j]);
        }
    }

    float y_f[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        __nv_bfloat16 y_bf = f2bf(__fmul_rn(__fadd_rn(acc[j], cur_f[j]), conv_scale));
        y_f[j] = bf2f(y_bf);
    }

    float cout_f[8];
    if (addend != nullptr) {
        float add_f[8];
        unpack_bf16x8(addend[t * 288 + tid], add_f);
        if (res != nullptr) {
            float res_f[8];
            unpack_bf16x8(res[t * 288 + tid], res_f);
            #pragma unroll
            for (int j = 0; j < 8; ++j) add_f[j] = __fadd_rn(res_f[j], add_f[j]);
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_bfloat16 c_bf = f2bf(__fadd_rn(add_f[j], y_f[j]));
            cout_f[j] = bf2f(c_bf);
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) cout_f[j] = y_f[j];
    }
    conv_out[t * 288 + tid] = pack_bf16x8(cout_f);

    float src[N_STACK + 1][8];
    #pragma unroll
    for (int s = 0; s < N_STACK; ++s) {
        const uint4* sp = reinterpret_cast<const uint4*>(stack + s * stride_s + t * 2304);
        unpack_bf16x8(sp[tid], src[s]);
    }
    if (last != nullptr) {
        float lf[8];
        unpack_bf16x8(last[t * 288 + tid], lf);
        #pragma unroll
        for (int j = 0; j < 8; ++j) src[N_STACK][j] = __fadd_rn(lf[j], cout_f[j]);
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) src[N_STACK][j] = cout_f[j];
    }

    attn_res_epilogue_2304<N_STACK, Q, APPLY_NORM>(
        src, queries, norm_w, out_ptr, t, T, eps, norm_eps, true);
}

#define DISPATCH_ATTN_RES_Q(NS, QVAL) \
    if (has_addend) { \
        if (apply_norm) attn_res_2304_sm80_kernel<NS, QVAL, true, true><<<T, 288, 0, stream>>>(s_ptr, l_ptr, a_ptr, q_ptr, nw_ptr, out_ptr, stride_s, T, eps, norm_eps, norm_val); \
        else attn_res_2304_sm80_kernel<NS, QVAL, true, false><<<T, 288, 0, stream>>>(s_ptr, l_ptr, a_ptr, q_ptr, nw_ptr, out_ptr, stride_s, T, eps, norm_eps, norm_val); \
    } else { \
        if (apply_norm) attn_res_2304_sm80_kernel<NS, QVAL, false, true><<<T, 288, 0, stream>>>(s_ptr, l_ptr, a_ptr, q_ptr, nw_ptr, out_ptr, stride_s, T, eps, norm_eps, norm_val); \
        else attn_res_2304_sm80_kernel<NS, QVAL, false, false><<<T, 288, 0, stream>>>(s_ptr, l_ptr, a_ptr, q_ptr, nw_ptr, out_ptr, stride_s, T, eps, norm_eps, norm_val); \
    }

#define DISPATCH_ATTN_RES_NS(NS) \
    case NS: \
        if (num_q == 1) { DISPATCH_ATTN_RES_Q(NS, 1); } \
        else if (num_q == 3) { DISPATCH_ATTN_RES_Q(NS, 3); } \
        else if (num_q == 4) { DISPATCH_ATTN_RES_Q(NS, 4); } \
        else if (num_q == 2) { DISPATCH_ATTN_RES_Q(NS, 2); } \
        break;

extern "C" int launch_attn_res_2304_sm80(
    const void* stack, const void* last, const void* addend, const void* queries,
    const void* norm_w, void* out_ptr, int64_t stride_s, int64_t T,
    int32_t n_stack, int32_t num_q, int32_t has_addend, int32_t apply_norm, int32_t norm_val,
    float eps, float norm_eps, void* stream_ptr)
{
    if (T <= 0) return 1;
    if (n_stack < 0 || n_stack > 6 || num_q < 1 || num_q > 4) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    auto s_ptr = static_cast<const __nv_bfloat16*>(stack);
    auto l_ptr = static_cast<const __nv_bfloat16*>(last);
    auto a_ptr = static_cast<const __nv_bfloat16*>(addend);
    auto q_ptr = static_cast<const float*>(queries);
    auto nw_ptr = static_cast<const uint4*>(norm_w);
    switch (n_stack) {
        DISPATCH_ATTN_RES_NS(0)
        DISPATCH_ATTN_RES_NS(1)
        DISPATCH_ATTN_RES_NS(2)
        DISPATCH_ATTN_RES_NS(3)
        DISPATCH_ATTN_RES_NS(4)
        DISPATCH_ATTN_RES_NS(5)
        DISPATCH_ATTN_RES_NS(6)
    }
    return 1;
}

#define DISPATCH_CONV_MIX_Q(NS, QVAL) \
    if (apply_norm) conv_mix_2304_sm80_kernel<NS, QVAL, true><<<T, 288, 0, stream>>>( \
        static_cast<const uint4*>(x), static_cast<const uint4*>(conv_weight), static_cast<__nv_bfloat16*>(cache), \
        static_cast<const int64_t*>(positions), static_cast<const int32_t*>(block_table), static_cast<const int32_t*>(seq_idx), \
        static_cast<const int64_t*>(slot_mapping), static_cast<const int32_t*>(query_start), \
        stride_c_blk, stride_c_h, stride_c_n, stride_bt, max_blocks, cache_N, stream_offset, stream_width, conv_scale, \
        static_cast<uint4*>(conv_out), static_cast<const uint4*>(res), static_cast<const uint4*>(addend), \
        static_cast<const __nv_bfloat16*>(stack), static_cast<const uint4*>(last), static_cast<const float*>(queries), \
        static_cast<const uint4*>(norm_w), out_ptr, stride_s, T, eps, norm_eps, cache_vec8_ok); \
    else conv_mix_2304_sm80_kernel<NS, QVAL, false><<<T, 288, 0, stream>>>( \
        static_cast<const uint4*>(x), static_cast<const uint4*>(conv_weight), static_cast<__nv_bfloat16*>(cache), \
        static_cast<const int64_t*>(positions), static_cast<const int32_t*>(block_table), static_cast<const int32_t*>(seq_idx), \
        static_cast<const int64_t*>(slot_mapping), static_cast<const int32_t*>(query_start), \
        stride_c_blk, stride_c_h, stride_c_n, stride_bt, max_blocks, cache_N, stream_offset, stream_width, conv_scale, \
        static_cast<uint4*>(conv_out), static_cast<const uint4*>(res), static_cast<const uint4*>(addend), \
        static_cast<const __nv_bfloat16*>(stack), static_cast<const uint4*>(last), static_cast<const float*>(queries), \
        static_cast<const uint4*>(norm_w), out_ptr, stride_s, T, eps, norm_eps, cache_vec8_ok);

#define DISPATCH_CONV_MIX_NS(NS) \
    case NS: \
        if (num_q == 1) { DISPATCH_CONV_MIX_Q(NS, 1); } \
        else if (num_q == 3) { DISPATCH_CONV_MIX_Q(NS, 3); } \
        else if (num_q == 2) { DISPATCH_CONV_MIX_Q(NS, 2); } \
        else if (num_q == 4) { DISPATCH_CONV_MIX_Q(NS, 4); } \
        break;

extern "C" int launch_conv_mix_2304_sm80(
    const void* x, const void* conv_weight, void* cache,
    const void* positions, const void* block_table, const void* seq_idx,
    const void* slot_mapping, const void* query_start,
    int64_t stride_c_blk, int64_t stride_c_h, int64_t stride_c_n,
    int32_t stride_bt, int32_t max_blocks, int32_t cache_N,
    int32_t stream_offset, int32_t stream_width, float conv_scale,
    void* conv_out, const void* res, const void* addend,
    const void* stack, const void* last, const void* queries,
    const void* norm_w, void* out_ptr, int64_t stride_s, int64_t T,
    int32_t n_stack, int32_t num_q, int32_t apply_norm, float eps, float norm_eps, void* stream_ptr)
{
    if (T <= 0) return 1;
    if (n_stack < 0 || n_stack > 6 || num_q < 1 || num_q > 4) return 0;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    bool cache_vec8_ok = ((stream_width & 7) == 0) && ((stream_offset & 7) == 0) &&
                         ((stride_c_h & 7) == 0) && ((stride_c_n & 7) == 0) && ((stride_c_blk & 7) == 0) &&
                         ((reinterpret_cast<uintptr_t>(cache) & 15) == 0);
    switch (n_stack) {
        DISPATCH_CONV_MIX_NS(0)
        DISPATCH_CONV_MIX_NS(1)
        DISPATCH_CONV_MIX_NS(2)
        DISPATCH_CONV_MIX_NS(3)
        DISPATCH_CONV_MIX_NS(4)
        DISPATCH_CONV_MIX_NS(5)
        DISPATCH_CONV_MIX_NS(6)
    }
    return 1;
}

// ============================================================================
// 5. Loop Message Controller Norms (pair_rms_norm & blend_rms_norm, D=2304)
// ============================================================================
__global__ void __launch_bounds__(288) pair_rms_norm_2304_sm80_kernel(
    const float* __restrict__ dest,
    const float* __restrict__ message,
    uint4* __restrict__ out,
    int64_t stride_dest_s, int64_t T, float eps)
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
    float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
    float mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};

    float dsq = 0.f, msq = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        dsq = __fmaf_rn(dv[j], dv[j], dsq);
        msq = __fmaf_rn(mv[j], mv[j], msq);
    }
    dsq = warp_sum_rn(dsq);
    msq = warp_sum_rn(msq);
    if (lane == 0) { s_w[0][warp] = dsq; s_w[1][warp] = msq; }
    __syncthreads();
    if (tid < 2) {
        float sum = s_w[tid][0];
        #pragma unroll
        for (int w = 1; w < 9; ++w) sum = __fadd_rn(sum, s_w[tid][w]);
        s_rstd[tid] = rsqrtf(__fadd_rn(__fdiv_rn(sum, 2304.0f), eps));
    }
    __syncthreads();
    float dr = s_rstd[0], mr = s_rstd[1];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        dv[j] = __fmul_rn(dv[j], dr);
        mv[j] = __fmul_rn(mv[j], mr);
    }
    int64_t row = s * T + t;
    out[row * 576 + tid] = pack_bf16x8(dv);
    out[row * 576 + 288 + tid] = pack_bf16x8(mv);
}

__global__ void __launch_bounds__(288) blend_rms_norm_2304_sm80_kernel(
    const float* __restrict__ dest,
    const float* __restrict__ message,
    const float* __restrict__ gain_forget,
    const __nv_bfloat16* __restrict__ norm_w,
    uint4* __restrict__ out,
    int64_t stride_dest_s, int64_t stride_w_s, int64_t T, float norm_eps)
{
    __shared__ float s_w[9];
    __shared__ float s_rstd;
    const int64_t s = blockIdx.x;
    const int64_t t = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int64_t row = s * T + t;

    const float4* dp = reinterpret_cast<const float4*>(dest + s * stride_dest_s + t * 2304);
    const float4* mp = reinterpret_cast<const float4*>(message + row * 2304);
    const float4* gp = reinterpret_cast<const float4*>(gain_forget + row * 4608);
    const float4* fp = reinterpret_cast<const float4*>(gain_forget + row * 4608 + 2304);
    float4 d0 = dp[tid * 2 + 0], d1 = dp[tid * 2 + 1];
    float4 m0 = mp[tid * 2 + 0], m1 = mp[tid * 2 + 1];
    float4 g0 = gp[tid * 2 + 0], g1 = gp[tid * 2 + 1];
    float4 f0 = fp[tid * 2 + 0], f1 = fp[tid * 2 + 1];
    float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
    float mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
    float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    float fv[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};

    float xv[8];
    float xsq = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float gn = tanhf(gv[j]);
        float fg = tanhf(fv[j]);
        xv[j] = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, fg), dv[j]), __fmul_rn(gn, mv[j]));
        xsq = __fadd_rn(xsq, __fmul_rn(xv[j], xv[j]));
    }
    xsq = warp_sum_rn(xsq);
    if (lane == 0) s_w[warp] = xsq;
    __syncthreads();
    if (tid == 0) {
        float sum = s_w[0];
        #pragma unroll
        for (int w = 1; w < 9; ++w) sum = __fadd_rn(sum, s_w[w]);
        s_rstd = rsqrtf(__fadd_rn(__fdiv_rn(sum, 2304.0f), norm_eps));
    }
    __syncthreads();
    float rstd = s_rstd;
    float nw[8];
    unpack_bf16x8(reinterpret_cast<const uint4*>(norm_w + s * stride_w_s)[tid], nw);
    #pragma unroll
    for (int j = 0; j < 8; ++j) xv[j] = __fmul_rn(__fmul_rn(xv[j], rstd), nw[j]);
    out[row * 288 + tid] = pack_bf16x8(xv);
}

extern "C" void launch_pair_rms_norm_2304_sm80(
    const void* dest, const void* message, void* out,
    int64_t stride_dest_s, int64_t S, int64_t T, float eps, void* stream_ptr)
{
    if (S <= 0 || T <= 0) return;
    dim3 grid(S, T);
    pair_rms_norm_2304_sm80_kernel<<<grid, 288, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const float*>(dest), static_cast<const float*>(message),
        static_cast<uint4*>(out), stride_dest_s, T, eps);
}

extern "C" void launch_blend_rms_norm_2304_sm80(
    const void* dest, const void* message, const void* gain_forget, const void* norm_w, void* out,
    int64_t stride_dest_s, int64_t stride_w_s, int64_t S, int64_t T, float norm_eps, void* stream_ptr)
{
    if (S <= 0 || T <= 0) return;
    dim3 grid(S, T);
    blend_rms_norm_2304_sm80_kernel<<<grid, 288, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const float*>(dest), static_cast<const float*>(message),
        static_cast<const float*>(gain_forget), static_cast<const __nv_bfloat16*>(norm_w),
        static_cast<uint4*>(out), stride_dest_s, stride_w_s, T, norm_eps);
}

// ============================================================================
// 6. DSpark Fused Markov Step Kernels (RANK = 256)
// ============================================================================
__global__ void __launch_bounds__(256) markov_rnn_gate_256_sm80_kernel(
    const uint4* __restrict__ state_in,
    const void* __restrict__ prev_tokens,
    const uint4* __restrict__ markov_w1,
    const float* __restrict__ context,
    const uint4* __restrict__ rnn_w_step,
    __nv_bfloat16* __restrict__ state_out,
    __nv_bfloat16* __restrict__ latent_out,
    int32_t prev_is_i64)
{
    const int req = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 8 + warp;

    int64_t tok = prev_is_i64
        ? static_cast<const int64_t*>(prev_tokens)[req]
        : static_cast<int64_t>(static_cast<const int32_t*>(prev_tokens)[req]);

    float in0[8], in1[8];
    unpack_bf16x8(state_in[req * 32 + lane], in0);
    unpack_bf16x8(markov_w1[tok * 32 + lane], in1);

    float raw3[3];
    #pragma unroll
    for (int part = 0; part < 3; ++part) {
        int row = part * 256 + c;
        const uint4* wr = rnn_w_step + row * 64;
        float w0[8], w1[8];
        unpack_bf16x8(wr[lane], w0);
        unpack_bf16x8(wr[32 + lane], w1);
        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) s = __fmaf_rn(in0[j], w0[j], s);
        #pragma unroll
        for (int j = 0; j < 8; ++j) s = __fmaf_rn(in1[j], w1[j], s);
        raw3[part] = warp_sum_rn(s);
    }

    if (lane == 0) {
        float g_raw = round_bf16_f32(__fadd_rn(context[req * 768 + c], raw3[0]));
        float c_raw = round_bf16_f32(__fadd_rn(context[req * 768 + 256 + c], raw3[1]));
        float o_raw = round_bf16_f32(__fadd_rn(context[req * 768 + 512 + c], raw3[2]));
        float st = bf2f(reinterpret_cast<const __nv_bfloat16*>(state_in)[req * 256 + c]);

        float gate = round_bf16_f32(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g_raw))));
        float cand = round_bf16_f32(tanhf(c_raw));
        float keep = round_bf16_f32(__fmul_rn(gate, st));
        float upd = round_bf16_f32(__fmul_rn(round_bf16_f32(__fsub_rn(1.0f, gate)), cand));
        state_out[req * 256 + c] = f2bf(__fadd_rn(keep, upd));
        latent_out[req * 256 + c] = f2bf(tanhf(o_raw));
    }
}

extern "C" void launch_markov_rnn_gate_256_sm80(
    const void* state_in, const void* prev_tokens, const void* markov_w1,
    const void* context, const void* rnn_w_step, void* state_out, void* latent_out,
    int32_t num_reqs, int32_t prev_is_i64, void* stream_ptr)
{
    if (num_reqs <= 0) return;
    dim3 grid(32, num_reqs);
    markov_rnn_gate_256_sm80_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const uint4*>(state_in), prev_tokens, static_cast<const uint4*>(markov_w1),
        static_cast<const float*>(context), static_cast<const uint4*>(rnn_w_step),
        static_cast<__nv_bfloat16*>(state_out), static_cast<__nv_bfloat16*>(latent_out), prev_is_i64);
}

__global__ void __launch_bounds__(256) markov_fold_softcap_r1_sm80_kernel(
    const uint4* __restrict__ latent,
    const uint4* __restrict__ fold_w,
    const uint4* __restrict__ fold_b,
    const uint4* __restrict__ base_logits,
    uint4* __restrict__ out_logits,
    int32_t V, float softcap)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 64 + warp * 8;
    if (row0 >= V) return;

    float lat[8];
    unpack_bf16x8(latent[lane], lat);

    float dots[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        float w[8];
        unpack_bf16x8(fold_w[(row0 + r) * 32 + lane], w);
        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) s = __fmaf_rn(lat[j], w[j], s);
        dots[r] = warp_sum_rn(s);
    }

    if (lane == 0) {
        int vec_idx = (row0 >> 3);
        float fb[8], bl[8], out_f[8];
        unpack_bf16x8(fold_b[vec_idx], fb);
        unpack_bf16x8(base_logits[vec_idx], bl);
        float inv_sc = 1.0f / softcap;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float bias_bf = bf2f(f2bf(__fadd_rn(fb[r], dots[r])));
            float x_bf = bf2f(f2bf(__fadd_rn(bl[r], bias_bf)));
            out_f[r] = __fmul_rn(softcap, tanhf(__fmul_rn(x_bf, inv_sc)));
        }
        out_logits[vec_idx] = pack_bf16x8(out_f);
    }
}

extern "C" void launch_markov_fold_softcap_r1_sm80(
    const void* latent, const void* fold_w, const void* fold_b,
    const void* base_logits, void* out_logits, int32_t V, float softcap, void* stream_ptr)
{
    int blocks = (V + 63) / 64;
    markov_fold_softcap_r1_sm80_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const uint4*>(latent), static_cast<const uint4*>(fold_w),
        static_cast<const uint4*>(fold_b), static_cast<const uint4*>(base_logits),
        static_cast<uint4*>(out_logits), V, softcap);
}

// ============================================================================
// ROUND 3 SM80 BREAKTHROUGH KERNELS (sm80_fast_v3.cu)
// 1. moe_align_v3_sm80_kernel: O(1) shared-atomic + 8-warp shfl_up_sync scan
//    (eliminates 128-step serial tid==0 stall & 3-launch fallback up to 1024!)
// 2. select_and_align_32cta_128_sm80_kernel: 32-CTA warp-per-expert BF16 gate
//    GEMV + exact selection keys + fused parallel moe_align in ONE launch!
// 3. sm80_skinny_gemm_streamk_m8_kernel: 148-CTA (2 CTAs x 74 SMs) Stream-K
//    cp.async.cg + mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 W @ X^T
//    with 3-input QKV support (zero wave quantization on 74 SMs!).
// 4. fused_up_blend_rms_2304_sm80_kernel: fuses controller_gain_forget up-proj
//    + tanh gate/forget blend + RMSNorm for c=16 / large S*T batches!
// ============================================================================

#include <cuda_bf16.h>
#include <mma.h>

__device__ __forceinline__ int warp_inclusive_scan_int(int val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}

// Fast 1-CTA parallel moe_align_block_size for numel <= 1024, num_experts <= 256.
// Preserves exact per-expert counts and padded block boundaries; intra-expert slots
// assigned via shared-memory atomicAdd (valid bijection for Humming indexed GEMM).
__global__ void __launch_bounds__(256) moe_align_v3_small_sm80_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t numel,
    const int32_t block_size,
    const int32_t num_experts,
    const int32_t max_pad,
    const int32_t n_blocks,
    const int32_t ignore_invalid)
{
    __shared__ int32_t s_counts[256];
    __shared__ int32_t s_pad_start[256];
    __shared__ int32_t s_pad_end[256];
    __shared__ int32_t s_warp_sums[8];
    __shared__ int16_t s_expert[1024];
    __shared__ int16_t s_slot[1024];

    const int tid = threadIdx.x;
    s_counts[tid] = 0;

    // Vectorized int4 fill of sorted_ids with sentinel `numel`
    const int4 fill4 = make_int4(numel, numel, numel, numel);
    int4* sorted4 = reinterpret_cast<int4*>(sorted_ids);
    const int n_vec4 = max_pad >> 2;
    for (int idx = tid; idx < n_vec4; idx += 256) {
        sorted4[idx] = fill4;
    }
    for (int idx = (n_vec4 << 2) + tid; idx < max_pad; idx += 256) {
        sorted_ids[idx] = numel;
    }
    __syncthreads();

    for (int i = tid; i < numel; i += 256) {
        int32_t v = topk_ids[i];
        bool valid = ignore_invalid ? ((uint32_t)v < (uint32_t)num_experts) : true;
        if (valid) {
            int slot = atomicAdd(&s_counts[v], 1);
            s_expert[i] = static_cast<int16_t>(v);
            s_slot[i] = static_cast<int16_t>(slot);
        } else {
            s_expert[i] = static_cast<int16_t>(-1);
            s_slot[i] = static_cast<int16_t>(0);
        }
    }
    __syncthreads();

    const int cnt = (tid < num_experts) ? s_counts[tid] : 0;
    const int padded = ((cnt + block_size - 1) / block_size) * block_size;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    int scan = warp_inclusive_scan_int(padded);
    if (lane == 31) s_warp_sums[warp_id] = scan;
    __syncthreads();

    if (warp_id == 0) {
        int wval = (lane < 8) ? s_warp_sums[lane] : 0;
        wval = warp_inclusive_scan_int(wval);
        if (lane < 8) s_warp_sums[lane] = wval;
    }
    __syncthreads();

    const int warp_prefix = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
    const int end = scan + warp_prefix;
    const int start = end - padded;
    s_pad_start[tid] = start;
    s_pad_end[tid] = end;
    if (tid == num_experts - 1) {
        total_tokens_post_pad[0] = end;
    }
    __syncthreads();

    const int total_blocks = s_pad_end[num_experts - 1] / block_size;
    if (tid < num_experts && padded > 0) {
        int b_start = start / block_size;
        int b_end = end / block_size;
        for (int b = b_start; b < b_end; ++b) {
            if (b < n_blocks) expert_ids[b] = tid;
        }
    }
    for (int b = total_blocks + tid; b < n_blocks; b += 256) {
        expert_ids[b] = 0;
    }

    for (int i = tid; i < numel; i += 256) {
        int e = s_expert[i];
        if (e >= 0) {
            sorted_ids[s_pad_start[e] + s_slot[i]] = i;
        }
    }
}

extern "C" void launch_moe_align_v3_sm80(
    const void* topk_ids,
    void* sorted_ids,
    void* expert_ids,
    void* total_tokens_post_pad,
    void* workspace,
    int32_t numel,
    int32_t block_size,
    int32_t num_experts,
    int32_t max_pad,
    int32_t n_blocks,
    int32_t ignore_invalid,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    if (numel <= 1024) {
        moe_align_v3_small_sm80_kernel<<<1, 256, 0, stream>>>(
            static_cast<const int32_t*>(topk_ids),
            static_cast<int32_t*>(sorted_ids),
            static_cast<int32_t*>(expert_ids),
            static_cast<int32_t*>(total_tokens_post_pad),
            numel, block_size, num_experts, max_pad, n_blocks, ignore_invalid
        );
    } else {
        launch_moe_align_block_size_sm80(
            topk_ids, sorted_ids, expert_ids, total_tokens_post_pad, workspace,
            numel, block_size, num_experts, max_pad, n_blocks, ignore_invalid, stream_ptr
        );
    }
}

// ============================================================================
// 2. Fused 32-CTA Gate GEMV (BF16 x [T, 2304], BF16 gate_w [128, 2304])
//    + Exact Inkling2 Top-8 Sigmoid+Bias Selection + Parallel moe_align!
//    Supports 1 <= T <= 16, E = 128, TOPK = 8, D = 2304.
// ============================================================================
__device__ __forceinline__ float warp_sum_f32(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

__global__ void __launch_bounds__(128, 4) select_and_align_32cta_128_sm80_kernel(
    const __nv_bfloat16* __restrict__ x,          // [T, 2304] bf16
    const __nv_bfloat16* __restrict__ gate_w,     // [128, 2304] bf16
    const float* __restrict__ bias,               // [128] fp32
    const int32_t* __restrict__ expert_bounds,    // [2] int32
    const uint8_t* __restrict__ is_padding,       // [T] bool or nullptr
    float* __restrict__ ws_logits,                // [16 * 128] fp32 scratch
    int32_t* __restrict__ cta_counter,            // [1] int32 zero-initialized counter
    int32_t* __restrict__ out_ids,                // [T * 8] int32
    float* __restrict__ out_weights,              // [T * 8] fp32
    int32_t* __restrict__ out_src2dst,            // [T * 8] int32
    int32_t* __restrict__ sorted_ids,             // [max_pad] int32
    int32_t* __restrict__ expert_ids,             // [n_blocks] int32
    int32_t* __restrict__ total_tokens_post_pad,  // [1] int32
    const int32_t T,
    const int32_t block_size,
    const int32_t max_pad,
    const int32_t n_blocks,
    const float route_scale)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;   // 0..3
    const int lane = tid & 31;      // 0..31
    const int expert = (blockIdx.x << 2) + warp_id; // 0..127

    // Phase 1: Each warp computes exact 1024-group dot products for 1 expert across T<=8 or T<=16 tokens!
    // Each lane loads 8 bf16s (uint4) per iter; 32 lanes * 8 = 256 bf16s/iter; 2304 / 256 = 9 iters!
    // Groups: iters 0..3 (k=0..1023), iters 4..7 (k=1024..2047), iter 8 (k=2048..2303).
    const uint4* w_vec = reinterpret_cast<const uint4*>(gate_w + expert * 2304);
    uint4 w_reg[9];
    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        w_reg[it] = w_vec[it * 32 + lane];
    }

    for (int t = 0; t < T; ++t) {
        const uint4* x_vec = reinterpret_cast<const uint4*>(x + t * 2304);
        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;

        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            uint4 xv = x_vec[it * 32 + lane];
            uint4 wv = w_reg[it];
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float2 xf = __bfloat1622float2(x2[p]);
                float2 wf = __bfloat1622float2(w2[p]);
                g0 += xf.x * wf.x + xf.y * wf.y;
            }
        }
        g0 = warp_sum_f32(g0);

        #pragma unroll
        for (int it = 4; it < 8; ++it) {
            uint4 xv = x_vec[it * 32 + lane];
            uint4 wv = w_reg[it];
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float2 xf = __bfloat1622float2(x2[p]);
                float2 wf = __bfloat1622float2(w2[p]);
                g1 += xf.x * wf.x + xf.y * wf.y;
            }
        }
        g1 = warp_sum_f32(g1);

        {
            uint4 xv = x_vec[8 * 32 + lane];
            uint4 wv = w_reg[8];
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float2 xf = __bfloat1622float2(x2[p]);
                float2 wf = __bfloat1622float2(w2[p]);
                g2 += xf.x * wf.x + xf.y * wf.y;
            }
        }
        g2 = warp_sum_f32(g2);

        if (lane == 0) {
            ws_logits[t * 128 + expert] = (g0 + g1) + g2;
        }
    }

    __shared__ int s_is_last;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int prev = atomicAdd(cta_counter, 1);
        s_is_last = (prev == 31) ? 1 : 0;
        if (prev == 31) {
            atomicExch(cta_counter, 0);
        }
    }
    __syncthreads();
    if (!s_is_last) return;

    // Phase 2 (Last CTA only, 128 threads = 4 warps):
    // Fill sorted_ids with sentinel numel = T * 8 while loading bias & bounds
    const int numel = T * 8;
    const int4 fill4 = make_int4(numel, numel, numel, numel);
    int4* sorted4 = reinterpret_cast<int4*>(sorted_ids);
    const int n_vec4 = max_pad >> 2;
    for (int idx = tid; idx < n_vec4; idx += 128) {
        sorted4[idx] = fill4;
    }
    for (int idx = (n_vec4 << 2) + tid; idx < max_pad; idx += 128) {
        sorted_ids[idx] = numel;
    }

    __shared__ int32_t s_counts[128];
    __shared__ int32_t s_pad_start[128];
    __shared__ int32_t s_pad_end[128];
    __shared__ int32_t s_warp_sums[4];
    __shared__ int16_t s_expert[128];
    __shared__ int16_t s_slot[128];

    s_counts[tid] = 0;
    const int32_t eb_lo = expert_bounds[0];
    const int32_t eb_hi = expert_bounds[1];
    float b_val[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        b_val[m] = bias[lane + (m << 5)];
    }
    __syncthreads();

    // Each warp handles tokens t = warp_id, warp_id + 4, ...
    for (int t = warp_id; t < T; t += 4) {
        const bool padded_tok = (is_padding != nullptr) ? (is_padding[t] != 0) : false;
        float my_logit[4];
        uint64_t my_key[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int e = lane + (m << 5);
            float lg = ws_logits[t * 128 + e];
            my_logit[m] = lg;
            float sig = __fdividef(1.0f, 1.0f + expf(-lg));
            float sel = sig + b_val[m];
            uint32_t bits = __float_as_uint(sel);
            bits ^= (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
            my_key[m] = (static_cast<uint64_t>(bits) << 16) | static_cast<uint64_t>(128 - e);
        }

        int sel_e[8];
        float sel_lp[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint64_t best_k = my_key[0];
            int best_m = 0;
            #pragma unroll
            for (int m = 1; m < 4; ++m) {
                if (my_key[m] > best_k) {
                    best_k = my_key[m];
                    best_m = m;
                }
            }
            uint64_t w_best = best_k;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                uint64_t other = __shfl_xor_sync(0xffffffffu, w_best, mask);
                if (other > w_best) w_best = other;
            }
            int win_e = 128 - static_cast<int>(w_best & 0xFFFFu);
            if (best_k == w_best) {
                my_key[best_m] = 0ULL;
            }
            // Broadcast winning logit from owner lane (win_e & 31)
            int owner_lane = win_e & 31;
            int owner_m = win_e >> 5;
            float cand_lg = (owner_m == 0) ? my_logit[0] : (owner_m == 1) ? my_logit[1] : (owner_m == 2) ? my_logit[2] : my_logit[3];
            float win_lg = __shfl_sync(0xffffffffu, cand_lg, owner_lane);
            sel_e[k] = win_e;
            sel_lp[k] = fminf(win_lg, 0.0f) - log1pf(expf(-fabsf(win_lg)));
        }

        if (lane < 8) {
            float lp = sel_lp[lane];
            // max over lanes 0..7
            float mx = lp;
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 4));
            float sh = expf(lp - mx);
            // _sum8 order: ((x0 + x2) + (x1 + x3)) + ((x4 + x6) + (x5 + x7))
            float s02 = sh + __shfl_xor_sync(0xffu, sh, 2);
            float s0123 = s02 + __shfl_xor_sync(0xffu, s02, 1);
            float total = s0123 + __shfl_xor_sync(0xffu, s0123, 4);
            float lse = mx + logf(total);
            float w_out = expf(lp - lse) * route_scale;

            int raw_e = sel_e[lane];
            int local_e = (!padded_tok && raw_e >= eb_lo && raw_e < eb_hi) ? (raw_e - eb_lo) : -1;
            int row = t * 8 + lane;
            out_ids[row] = local_e;
            out_weights[row] = w_out;
            out_src2dst[row] = (local_e >= 0) ? row : -1;
            if (local_e >= 0) {
                int slot = atomicAdd(&s_counts[local_e], 1);
                s_expert[row] = static_cast<int16_t>(local_e);
                s_slot[row] = static_cast<int16_t>(slot);
            } else {
                s_expert[row] = static_cast<int16_t>(-1);
                s_slot[row] = static_cast<int16_t>(0);
            }
        }
    }
    __syncthreads();

    // 4-warp parallel prefix scan over 128 experts (tid = 0..127)
    const int cnt = s_counts[tid];
    const int padded = ((cnt + block_size - 1) / block_size) * block_size;
    int scan = warp_inclusive_scan_int(padded);
    if (lane == 31) s_warp_sums[warp_id] = scan;
    __syncthreads();

    if (warp_id == 0) {
        int wval = (lane < 4) ? s_warp_sums[lane] : 0;
        wval = warp_inclusive_scan_int(wval);
        if (lane < 4) s_warp_sums[lane] = wval;
    }
    __syncthreads();

    const int warp_prefix = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
    const int end = scan + warp_prefix;
    const int start = end - padded;
    s_pad_start[tid] = start;
    s_pad_end[tid] = end;
    if (tid == 127) {
        total_tokens_post_pad[0] = end;
    }
    __syncthreads();

    const int total_blocks = s_pad_end[127] / block_size;
    if (padded > 0) {
        int b_start = start / block_size;
        int b_end = end / block_size;
        for (int b = b_start; b < b_end; ++b) {
            if (b < n_blocks) expert_ids[b] = tid;
        }
    }
    for (int b = total_blocks + tid; b < n_blocks; b += 128) {
        expert_ids[b] = 0;
    }

    if (tid < numel) {
        int e = s_expert[tid];
        if (e >= 0) {
            sorted_ids[s_pad_start[e] + s_slot[tid]] = tid;
        }
    }
}

extern "C" void launch_select_and_align_32cta_128_sm80(
    const void* x,
    const void* gate_w,
    const void* bias,
    const void* expert_bounds,
    const void* is_padding,
    void* ws_logits,
    void* cta_counter,
    void* out_ids,
    void* out_weights,
    void* out_src2dst,
    void* sorted_ids,
    void* expert_ids,
    void* total_tokens_post_pad,
    int32_t T,
    int32_t block_size,
    int32_t max_pad,
    int32_t n_blocks,
    float route_scale,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    select_and_align_32cta_128_sm80_kernel<<<32, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(gate_w),
        static_cast<const float*>(bias),
        static_cast<const int32_t*>(expert_bounds),
        static_cast<const uint8_t*>(is_padding),
        static_cast<float*>(ws_logits),
        static_cast<int32_t*>(cta_counter),
        static_cast<int32_t*>(out_ids),
        static_cast<float*>(out_weights),
        static_cast<int32_t*>(out_src2dst),
        static_cast<int32_t*>(sorted_ids),
        static_cast<int32_t*>(expert_ids),
        static_cast<int32_t*>(total_tokens_post_pad),
        T, block_size, max_pad, n_blocks, route_scale
    );
}

// ============================================================================
// 3. Fused Up-Projection + Tanh Blend + RMSNorm (for S*T >= 12, e.g. c=16)
//    controller_gf_t is pre-transposed [S, 2, 64, 2304] bf16 so threads c=0..2303
//    load coalesced uint4s across D=2304!
//    Grid: (S, T), Block: 288 threads (each thread owns 8 contiguous channels!).
// ============================================================================
__global__ void __launch_bounds__(288, 3) fused_up_blend_rms_2304_sm80_kernel(
    const float* __restrict__ dest,              // [S, T, 2304] fp32 (stride_dest_s may be 0)
    const float* __restrict__ message,           // [S, T, 2304] fp32 contiguous
    const __nv_bfloat16* __restrict__ hidden,    // [S, T, 64] bf16 contiguous
    const __nv_bfloat16* __restrict__ gf_t,      // [S, 2, 64, 2304] bf16 contiguous
    const __nv_bfloat16* __restrict__ norm_w,    // [S, 2304] bf16 (stride_w_s may be 0)
    __nv_bfloat16* __restrict__ out,             // [S, T, 2304] bf16
    const int64_t stride_dest_s,
    const int64_t stride_w_s,
    const int32_t T,
    const float norm_eps)
{
    __shared__ float s_h[64];
    __shared__ float s_sum[9];

    const int s = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x; // 0..287 -> col8 = tid * 8 .. tid * 8 + 7

    if (tid < 64) {
        s_h[tid] = __bfloat162float(hidden[(s * T + t) * 64 + tid]);
    }
    __syncthreads();

    const int col_vec = tid; // 288 * 8 = 2304!
    const __nv_bfloat16* g_base = gf_t + (static_cast<int64_t>(s) * 2 + 0) * (64 * 2304);
    const __nv_bfloat16* f_base = gf_t + (static_cast<int64_t>(s) * 2 + 1) * (64 * 2304);

    float gain_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float forg_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll 8
    for (int r = 0; r < 64; ++r) {
        float hr = s_h[r];
        uint4 gv = reinterpret_cast<const uint4*>(g_base + r * 2304)[col_vec];
        uint4 fv = reinterpret_cast<const uint4*>(f_base + r * 2304)[col_vec];
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
        const __nv_bfloat162* f2 = reinterpret_cast<const __nv_bfloat162*>(&fv);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 gf = __bfloat1622float2(g2[p]);
            float2 ff = __bfloat1622float2(f2[p]);
            gain_acc[2 * p + 0] += hr * gf.x;
            gain_acc[2 * p + 1] += hr * gf.y;
            forg_acc[2 * p + 0] += hr * ff.x;
            forg_acc[2 * p + 1] += hr * ff.y;
        }
    }

    const int64_t row_st = static_cast<int64_t>(s) * T + t;
    const float4* dest4 = reinterpret_cast<const float4*>(dest + static_cast<int64_t>(s) * stride_dest_s + static_cast<int64_t>(t) * 2304);
    const float4* msg4  = reinterpret_cast<const float4*>(message + row_st * 2304);
    float4 d_lo = dest4[2 * col_vec + 0];
    float4 d_hi = dest4[2 * col_vec + 1];
    float4 m_lo = msg4[2 * col_vec + 0];
    float4 m_hi = msg4[2 * col_vec + 1];

    float d_arr[8] = {d_lo.x, d_lo.y, d_lo.z, d_lo.w, d_hi.x, d_hi.y, d_hi.z, d_hi.w};
    float m_arr[8] = {m_lo.x, m_lo.y, m_lo.z, m_lo.w, m_hi.x, m_hi.y, m_hi.z, m_hi.w};
    float x_arr[8];
    float sq_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // Unfused blend matching enable_fp_fusion=False
        float one_m_f = __fsub_rn(1.0f, tanhf(forg_acc[i]));
        float g_val = tanhf(gain_acc[i]);
        float xv = __fadd_rn(__fmul_rn(one_m_f, d_arr[i]), __fmul_rn(g_val, m_arr[i]));
        x_arr[i] = xv;
        sq_sum += xv * xv;
    }

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    sq_sum = warp_sum_f32(sq_sum);
    if (lane == 0) s_sum[warp_id] = sq_sum;
    __syncthreads();

    if (tid < 32) {
        float v = (tid < 9) ? s_sum[tid] : 0.0f;
        v = warp_sum_f32(v);
        if (tid == 0) s_sum[0] = rsqrtf(__fdiv_rn(v, 2304.0f) + norm_eps);
    }
    __syncthreads();

    const float rstd = s_sum[0];
    uint4 nw_vec = reinterpret_cast<const uint4*>(norm_w + static_cast<int64_t>(s) * stride_w_s)[col_vec];
    const __nv_bfloat162* nw2 = reinterpret_cast<const __nv_bfloat162*>(&nw_vec);
    uint4 out_vec;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&out_vec);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        float2 wf = __bfloat1622float2(nw2[p]);
        float y0 = (x_arr[2 * p + 0] * rstd) * wf.x;
        float y1 = (x_arr[2 * p + 1] * rstd) * wf.y;
        o2[p] = __floats2bfloat162_rn(y0, y1);
    }
    reinterpret_cast<uint4*>(out + row_st * 2304)[col_vec] = out_vec;
}

extern "C" void launch_fused_up_blend_rms_2304_sm80(
    const void* dest,
    const void* message,
    const void* hidden,
    const void* gf_t,
    const void* norm_w,
    void* out,
    int64_t stride_dest_s,
    int64_t stride_w_s,
    int32_t S,
    int32_t T,
    float norm_eps,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    dim3 grid(S, T);
    fused_up_blend_rms_2304_sm80_kernel<<<grid, 288, 0, stream>>>(
        static_cast<const float*>(dest),
        static_cast<const float*>(message),
        static_cast<const __nv_bfloat16*>(hidden),
        static_cast<const __nv_bfloat16*>(gf_t),
        static_cast<const __nv_bfloat16*>(norm_w),
        static_cast<__nv_bfloat16*>(out),
        stride_dest_s, stride_w_s, T, norm_eps
    );
}

// ============================================================================
// 4. Native SM80 cp.async.cg + mma.sync.aligned.m16n8k16 Stream-K Skinny GEMM
//    Computes Y [M, N] = X [M, K] @ W [N, K]^T (+ optional Bias [N]) for M <= 8
//    using 148 persistent CTAs (2 CTAs/SM x 74 SMs) so ZERO wave quantization!
//    Supports both BF16 output (out_fp32=0) and FP32 output (out_fp32=1).
// ============================================================================
__device__ __forceinline__ void cp_async_cg_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_ca_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N_WAIT>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
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

// BN = 64 (4 warps x 16 rows of W), BK = 64 (4 x k16), STAGES = 4 (36 KB smem -> 2 or 3 CTAs/SM!)
__global__ void __launch_bounds__(128, 3) sm80_skinny_gemm_streamk_m8_kernel(
    const __nv_bfloat16* __restrict__ X,     // [M, K] bf16
    const __nv_bfloat16* __restrict__ W,     // [N, K] bf16
    const __nv_bfloat16* __restrict__ Bias,  // [N] bf16 or nullptr
    void* __restrict__ Y,                    // [M, N] bf16 or fp32
    float* __restrict__ P,                   // [MAX_N_TILES * 8 * 64] fp32 partial buffer
    int32_t* __restrict__ Locks,             // [MAX_N_TILES] int32 zeroed counters
    const int32_t M,
    const int32_t N,
    const int32_t K,
    const int32_t out_fp32)
{
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int STAGES = 4;
    // W_smem: [STAGES][64 rows][8 uint4s (64 bf16s)] = 4 * 8192B = 32768B
    // X_smem: [STAGES][8 rows][8 uint4s (64 bf16s)]  = 4 * 1024B = 4096B
    __shared__ __align__(16) uint4 s_W[STAGES][64][8];
    __shared__ __align__(16) uint4 s_X[STAGES][8][8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;       // 0..3
    const int lane = tid & 31;          // 0..31
    const int group_id = lane >> 2;     // 0..7
    const int tid_in_grp = lane & 3;    // 0..3

    const int n_tiles = (N + BN - 1) >> 6;
    const int k_tiles = K >> 6; // K is always a multiple of 64 (2304, 3072, 4096, 4608, 9216)
    const int total_iters = n_tiles * k_tiles;
    const int num_ctas = gridDim.x;
    const int cid = blockIdx.x;

    const int iter_start = static_cast<int>((static_cast<int64_t>(total_iters) * cid) / num_ctas);
    const int iter_end   = static_cast<int>((static_cast<int64_t>(total_iters) * (cid + 1)) / num_ctas);
    if (iter_start >= iter_end) return;

    int cur_iter = iter_start;
    while (cur_iter < iter_end) {
        const int n_tile = cur_iter / k_tiles;
        const int k_start = cur_iter - n_tile * k_tiles;
        const int tile_iter_end = (n_tile + 1) * k_tiles;
        const int run_end = (iter_end < tile_iter_end) ? iter_end : tile_iter_end;
        const int k_end = run_end - n_tile * k_tiles;
        const int num_k = k_end - k_start;

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        // Helper lambda to issue async loads for stage `st` at k_idx `ki`
        auto issue_stage = [&](int st, int ki) {
            const int k_base = ki * BK;
            // 128 threads load 64 * 8 = 512 uint4s -> 4 uint4s per thread!
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int idx = tid + (u << 7); // 0..511
                int r = idx >> 3;         // 0..63
                int c = idx & 7;          // 0..7
                int sc = c ^ (r & 7);     // bank-conflict-free XOR swizzle!
                int g_n = (n_tile << 6) + r;
                uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_W[st][r][sc]));
                if (g_n < N) {
                    cp_async_cg_16B(dst, W + static_cast<int64_t>(g_n) * K + k_base + (c << 3));
                } else {
                    s_W[st][r][sc] = make_uint4(0, 0, 0, 0);
                }
            }
            // 64 threads load 8 * 8 = 64 uint4s of X
            if (tid < 64) {
                int m_row = tid >> 3; // 0..7
                int c = tid & 7;      // 0..7
                int sc = c ^ m_row;
                uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_X[st][m_row][sc]));
                if (m_row < M) {
                    cp_async_ca_16B(dst, X + static_cast<int64_t>(m_row) * K + k_base + (c << 3));
                } else {
                    s_X[st][m_row][sc] = make_uint4(0, 0, 0, 0);
                }
            }
            cp_async_commit();
        };

        const int preload = (num_k < (STAGES - 1)) ? num_k : (STAGES - 1);
        for (int p = 0; p < preload; ++p) {
            issue_stage(p, k_start + p);
        }

        for (int step = 0; step < num_k; ++step) {
            int next_p = step + (STAGES - 1);
            if (next_p < num_k) {
                issue_stage(next_p % STAGES, k_start + next_p);
                cp_async_wait<STAGES - 1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();

            const int st = step % STAGES;
            const int w_row0 = (warp_id << 4) + group_id;
            const int w_row1 = w_row0 + 8;

            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                // kk=0..3 corresponds to 16-bf16 subblocks (2 uint4s: c0 = 2*kk, c1 = 2*kk + 1)
                // c_lo/c_hi used directly // uint4 index for a0/a1 (tid_in_grp 0,1 -> 2*kk; 2,3 -> 2*kk)
                // Wait: tid_in_grp = 0..3 needs 2 bf16s = 1 uint32_t each inside a 16-bf16 (2-uint4 = 8-uint32) span!
                // Specifically:
                //   a0/a1: k_sub = 2*tid_in_grp (0..7 bf16s -> uint4 c_lo = 2*kk, uint32 word = tid_in_grp!)
                //   a2/a3: k_sub = 2*tid_in_grp + 8 (8..15 bf16s -> uint4 c_hi = 2*kk + 1, uint32 word = tid_in_grp!)
                int c_lo = (kk << 1);
                int c_hi = c_lo + 1;

                const uint32_t* w0_lo = reinterpret_cast<const uint32_t*>(&s_W[st][w_row0][c_lo ^ (w_row0 & 7)]);
                const uint32_t* w1_lo = reinterpret_cast<const uint32_t*>(&s_W[st][w_row1][c_lo ^ (w_row1 & 7)]);
                const uint32_t* w0_hi = reinterpret_cast<const uint32_t*>(&s_W[st][w_row0][c_hi ^ (w_row0 & 7)]);
                const uint32_t* w1_hi = reinterpret_cast<const uint32_t*>(&s_W[st][w_row1][c_hi ^ (w_row1 & 7)]);

                const uint32_t* x_lo  = reinterpret_cast<const uint32_t*>(&s_X[st][group_id][c_lo ^ group_id]);
                const uint32_t* x_hi  = reinterpret_cast<const uint32_t*>(&s_X[st][group_id][c_hi ^ group_id]);

                mma_m16n8k16_bf16(
                    acc0, acc1, acc2, acc3,
                    w0_lo[tid_in_grp], w1_lo[tid_in_grp], w0_hi[tid_in_grp], w1_hi[tid_in_grp],
                    x_lo[tid_in_grp], x_hi[tid_in_grp]
                );
            }
            __syncthreads();
        }

        // Write output or combine Stream-K split partial!
        // Calculate how many CTAs touch this `n_tile`:
        const int first_iter_of_tile = n_tile * k_tiles;
        const int last_iter_of_tile  = tile_iter_end - 1;
        // c_first is smallest cid with iter_end > first_iter_of_tile:
        // iter_end(c) = ((c+1)*total_iters)/num_ctas > first_iter_of_tile
        const int c_first = static_cast<int>((static_cast<int64_t>(first_iter_of_tile + 1) * num_ctas + total_iters - 1) / total_iters - 1);
        const int c_last  = static_cast<int>((static_cast<int64_t>(last_iter_of_tile + 1) * num_ctas + total_iters - 1) / total_iters - 1);
        const int splits_for_tile = c_last - c_first + 1;

        const int out_n0 = (n_tile << 6) + (warp_id << 4) + group_id;
        const int out_n1 = out_n0 + 8;
        const int m0 = (tid_in_grp << 1);
        const int m1 = m0 + 1;

        if (splits_for_tile == 1) {
            float b0 = (Bias != nullptr && out_n0 < N) ? __bfloat162float(Bias[out_n0]) : 0.0f;
            float b1 = (Bias != nullptr && out_n1 < N) ? __bfloat162float(Bias[out_n1]) : 0.0f;
            if (out_fp32) {
                float* Yf = static_cast<float*>(Y);
                if (out_n0 < N) {
                    if (m0 < M) Yf[static_cast<int64_t>(m0) * N + out_n0] = acc0 + b0;
                    if (m1 < M) Yf[static_cast<int64_t>(m1) * N + out_n0] = acc1 + b0;
                }
                if (out_n1 < N) {
                    if (m0 < M) Yf[static_cast<int64_t>(m0) * N + out_n1] = acc2 + b1;
                    if (m1 < M) Yf[static_cast<int64_t>(m1) * N + out_n1] = acc3 + b1;
                }
            } else {
                __nv_bfloat16* Yb = static_cast<__nv_bfloat16*>(Y);
                if (out_n0 < N) {
                    if (m0 < M) Yb[static_cast<int64_t>(m0) * N + out_n0] = __float2bfloat16_rn(acc0 + b0);
                    if (m1 < M) Yb[static_cast<int64_t>(m1) * N + out_n0] = __float2bfloat16_rn(acc1 + b0);
                }
                if (out_n1 < N) {
                    if (m0 < M) Yb[static_cast<int64_t>(m0) * N + out_n1] = __float2bfloat16_rn(acc2 + b1);
                    if (m1 < M) Yb[static_cast<int64_t>(m1) * N + out_n1] = __float2bfloat16_rn(acc3 + b1);
                }
            }
        } else {
            // Deterministic split order: slot = cid - c_first (at most 2 or 3 splits per boundary tile!)
            const int my_slot = cid - c_first;
            // P layout: [n_tile * 4 + my_slot][8][64] (assuming <= 4 splits per boundary tile)
            float* p_base = P + (static_cast<int64_t>(n_tile) * 8 + my_slot) * 512;
            const int local_n0 = (warp_id << 4) + group_id;
            const int local_n1 = local_n0 + 8;
            p_base[m0 * 64 + local_n0] = acc0;
            p_base[m1 * 64 + local_n0] = acc1;
            p_base[m0 * 64 + local_n1] = acc2;
            p_base[m1 * 64 + local_n1] = acc3;

            __syncthreads();
            __shared__ int s_tile_done;
            if (tid == 0) {
                __threadfence();
                int prev = atomicAdd(&Locks[n_tile], 1);
                s_tile_done = (prev == splits_for_tile - 1) ? 1 : 0;
                if (prev == splits_for_tile - 1) {
                    atomicExch(&Locks[n_tile], 0);
                }
            }
            __syncthreads();

            if (s_tile_done) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int sp = 0; sp < splits_for_tile; ++sp) {
                    const float* ps = P + (static_cast<int64_t>(n_tile) * 8 + sp) * 512;
                    s0 += ps[m0 * 64 + local_n0];
                    s1 += ps[m1 * 64 + local_n0];
                    s2 += ps[m0 * 64 + local_n1];
                    s3 += ps[m1 * 64 + local_n1];
                }
                float b0 = (Bias != nullptr && out_n0 < N) ? __bfloat162float(Bias[out_n0]) : 0.0f;
                float b1 = (Bias != nullptr && out_n1 < N) ? __bfloat162float(Bias[out_n1]) : 0.0f;
                if (out_fp32) {
                    float* Yf = static_cast<float*>(Y);
                    if (out_n0 < N) {
                        if (m0 < M) Yf[static_cast<int64_t>(m0) * N + out_n0] = s0 + b0;
                        if (m1 < M) Yf[static_cast<int64_t>(m1) * N + out_n0] = s1 + b0;
                    }
                    if (out_n1 < N) {
                        if (m0 < M) Yf[static_cast<int64_t>(m0) * N + out_n1] = s2 + b1;
                        if (m1 < M) Yf[static_cast<int64_t>(m1) * N + out_n1] = s3 + b1;
                    }
                } else {
                    __nv_bfloat16* Yb = static_cast<__nv_bfloat16*>(Y);
                    if (out_n0 < N) {
                        if (m0 < M) Yb[static_cast<int64_t>(m0) * N + out_n0] = __float2bfloat16_rn(s0 + b0);
                        if (m1 < M) Yb[static_cast<int64_t>(m1) * N + out_n0] = __float2bfloat16_rn(s1 + b0);
                    }
                    if (out_n1 < N) {
                        if (m0 < M) Yb[static_cast<int64_t>(m0) * N + out_n1] = __float2bfloat16_rn(s2 + b1);
                        if (m1 < M) Yb[static_cast<int64_t>(m1) * N + out_n1] = __float2bfloat16_rn(s3 + b1);
                    }
                }
            }
        }

        cur_iter = run_end;
    }
}

extern "C" void launch_sm80_skinny_gemm_streamk_m8(
    const void* X,
    const void* W,
    const void* Bias,
    void* Y,
    void* P,
    void* Locks,
    int32_t M,
    int32_t N,
    int32_t K,
    int32_t out_fp32,
    int32_t num_ctas,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int32_t n_tiles = (N + 63) >> 6;
    int32_t k_tiles = K >> 6;
    int32_t max_ctas = n_tiles * 6;
    if (max_ctas > n_tiles * k_tiles) max_ctas = n_tiles * k_tiles;
    if (num_ctas > max_ctas) num_ctas = max_ctas;
    if (num_ctas < 1) num_ctas = 1;
    sm80_skinny_gemm_streamk_m8_kernel<<<num_ctas, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(X),
        static_cast<const __nv_bfloat16*>(W),
        static_cast<const __nv_bfloat16*>(Bias),
        Y,
        static_cast<float*>(P),
        static_cast<int32_t*>(Locks),
        M, N, K, out_fp32
    );
}

// ============================================================================
// 5. Fused Select-from-Partials + Parallel moe_align_v3 (1 CTA of 256 threads = 8 warps)
//    Takes Tensor-Core `k_group_partials` [3, T, 128] fp32 for 1 <= T <= 128,
//    computes exact Inkling2 top-8 routing + parallel warp-scan moe_align in ONE launch!
//    Deletes 48 kernel launches/step at ALL batch sizes (c=1 M=5 AND c=16 M=48/80)!
// ============================================================================
__global__ void __launch_bounds__(256) select_partials_and_align_128_sm80_kernel(
    const float* __restrict__ partials,           // [GROUPS, T, 128] fp32
    const float* __restrict__ bias,               // [128] fp32
    const int32_t* __restrict__ expert_bounds,    // [2] int32
    const uint8_t* __restrict__ is_padding,       // [T] bool or nullptr
    int32_t* __restrict__ out_ids,                // [T * 8] int32
    float* __restrict__ out_weights,              // [T * 8] fp32
    int32_t* __restrict__ out_src2dst,            // [T * 8] int32
    int32_t* __restrict__ sorted_ids,             // [max_pad] int32
    int32_t* __restrict__ expert_ids,             // [n_blocks] int32
    int32_t* __restrict__ total_tokens_post_pad,  // [1] int32
    const int32_t T,
    const int32_t groups,
    const int32_t block_size,
    const int32_t max_pad,
    const int32_t n_blocks,
    const float route_scale)
{
    __shared__ int32_t s_counts[256];
    __shared__ int32_t s_pad_start[256];
    __shared__ int32_t s_pad_end[256];
    __shared__ int32_t s_warp_sums[8];
    __shared__ int16_t s_expert[1024];
    __shared__ int16_t s_slot[1024];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;  // 0..7
    const int lane = tid & 31;     // 0..31

    s_counts[tid] = 0;

    const int numel = T * 8;
    const int4 fill4 = make_int4(numel, numel, numel, numel);
    int4* sorted4 = reinterpret_cast<int4*>(sorted_ids);
    const int n_vec4 = max_pad >> 2;
    for (int idx = tid; idx < n_vec4; idx += 256) {
        sorted4[idx] = fill4;
    }
    for (int idx = (n_vec4 << 2) + tid; idx < max_pad; idx += 256) {
        sorted_ids[idx] = numel;
    }

    const int32_t eb_lo = expert_bounds[0];
    const int32_t eb_hi = expert_bounds[1];
    float b_val[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        b_val[m] = bias[lane + (m << 5)];
    }
    __syncthreads();

    // 8 warps process tokens t = warp_id, warp_id + 8, ...
    for (int t = warp_id; t < T; t += 8) {
        const bool padded_tok = (is_padding != nullptr) ? (is_padding[t] != 0) : false;
        float my_logit[4];
        uint64_t my_key[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int e = lane + (m << 5);
            float lg = partials[t * 128 + e];
            for (int g = 1; g < groups; ++g) {
                lg += partials[(g * T + t) * 128 + e];
            }
            my_logit[m] = lg;
            float sig = __fdividef(1.0f, 1.0f + expf(-lg));
            float sel = sig + b_val[m];
            uint32_t bits = __float_as_uint(sel);
            bits ^= (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
            my_key[m] = (static_cast<uint64_t>(bits) << 16) | static_cast<uint64_t>(128 - e);
        }

        int sel_e[8];
        float sel_lp[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint64_t best_k = my_key[0];
            int best_m = 0;
            #pragma unroll
            for (int m = 1; m < 4; ++m) {
                if (my_key[m] > best_k) {
                    best_k = my_key[m];
                    best_m = m;
                }
            }
            uint64_t w_best = best_k;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                uint64_t other = __shfl_xor_sync(0xffffffffu, w_best, mask);
                if (other > w_best) w_best = other;
            }
            int win_e = 128 - static_cast<int>(w_best & 0xFFFFu);
            if (best_k == w_best) {
                my_key[best_m] = 0ULL;
            }
            int owner_lane = win_e & 31;
            int owner_m = win_e >> 5;
            float cand_lg = (owner_m == 0) ? my_logit[0] : (owner_m == 1) ? my_logit[1] : (owner_m == 2) ? my_logit[2] : my_logit[3];
            float win_lg = __shfl_sync(0xffffffffu, cand_lg, owner_lane);
            sel_e[k] = win_e;
            sel_lp[k] = fminf(win_lg, 0.0f) - log1pf(expf(-fabsf(win_lg)));
        }

        if (lane < 8) {
            float lp = sel_lp[lane];
            float mx = lp;
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 4));
            float sh = expf(lp - mx);
            float s02 = sh + __shfl_xor_sync(0xffu, sh, 2);
            float s0123 = s02 + __shfl_xor_sync(0xffu, s02, 1);
            float total = s0123 + __shfl_xor_sync(0xffu, s0123, 4);
            float lse = mx + logf(total);
            float w_out = expf(lp - lse) * route_scale;

            int raw_e = sel_e[lane];
            int local_e = (!padded_tok && raw_e >= eb_lo && raw_e < eb_hi) ? (raw_e - eb_lo) : -1;
            int row = t * 8 + lane;
            out_ids[row] = local_e;
            out_weights[row] = w_out;
            out_src2dst[row] = (local_e >= 0) ? row : -1;
            if (local_e >= 0) {
                int slot = atomicAdd(&s_counts[local_e], 1);
                s_expert[row] = static_cast<int16_t>(local_e);
                s_slot[row] = static_cast<int16_t>(slot);
            } else {
                s_expert[row] = static_cast<int16_t>(-1);
                s_slot[row] = static_cast<int16_t>(0);
            }
        }
    }
    __syncthreads();

    const int cnt = (tid < 128) ? s_counts[tid] : 0;
    const int padded = ((cnt + block_size - 1) / block_size) * block_size;
    int scan = warp_inclusive_scan_int(padded);
    if (lane == 31) s_warp_sums[warp_id] = scan;
    __syncthreads();

    if (warp_id == 0) {
        int wval = (lane < 8) ? s_warp_sums[lane] : 0;
        wval = warp_inclusive_scan_int(wval);
        if (lane < 8) s_warp_sums[lane] = wval;
    }
    __syncthreads();

    const int warp_prefix = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
    const int end = scan + warp_prefix;
    const int start = end - padded;
    s_pad_start[tid] = start;
    s_pad_end[tid] = end;
    if (tid == 127) {
        total_tokens_post_pad[0] = end;
    }
    __syncthreads();

    const int total_blocks = s_pad_end[127] / block_size;
    if (tid < 128 && padded > 0) {
        int b_start = start / block_size;
        int b_end = end / block_size;
        for (int b = b_start; b < b_end; ++b) {
            if (b < n_blocks) expert_ids[b] = tid;
        }
    }
    for (int b = total_blocks + tid; b < n_blocks; b += 256) {
        expert_ids[b] = 0;
    }

    for (int i = tid; i < numel; i += 256) {
        int e = s_expert[i];
        if (e >= 0) {
            sorted_ids[s_pad_start[e] + s_slot[i]] = i;
        }
    }
}

extern "C" void launch_select_partials_and_align_128_sm80(
    const void* partials,
    const void* bias,
    const void* expert_bounds,
    const void* is_padding,
    void* out_ids,
    void* out_weights,
    void* out_src2dst,
    void* sorted_ids,
    void* expert_ids,
    void* total_tokens_post_pad,
    int32_t T,
    int32_t groups,
    int32_t block_size,
    int32_t max_pad,
    int32_t n_blocks,
    float route_scale,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    select_partials_and_align_128_sm80_kernel<<<1, 256, 0, stream>>>(
        static_cast<const float*>(partials),
        static_cast<const float*>(bias),
        static_cast<const int32_t*>(expert_bounds),
        static_cast<const uint8_t*>(is_padding),
        static_cast<int32_t*>(out_ids),
        static_cast<float*>(out_weights),
        static_cast<int32_t*>(out_src2dst),
        static_cast<int32_t*>(sorted_ids),
        static_cast<int32_t*>(expert_ids),
        static_cast<int32_t*>(total_tokens_post_pad),
        T, groups, block_size, max_pad, n_blocks, route_scale
    );
}

// ============================================================================
// 6. Bit-Exact Fused 1-Launch SM80 CUDA Markov Fold + Softcap + Sparse Cache
//    Write + Exact Murmur3 Gumbel-Max Argmax (for c=1, R=1)!
// ============================================================================
__device__ __forceinline__ uint32_t murmur3_rotl32_cu(uint32_t v, int shift) {
    return (v << shift) | (v >> (32 - shift));
}
__device__ __forceinline__ uint32_t murmur3_mix_cu(uint32_t h, uint32_t k) {
    k *= 0xCC9E2D51u;
    k = murmur3_rotl32_cu(k, 15);
    k *= 0x1B873593u;
    h ^= k;
    h = murmur3_rotl32_cu(h, 13);
    return h * 5u + 0xE6546B64u;
}
__device__ __forceinline__ uint32_t murmur3_fmix32_cu(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    return h ^ (h >> 16);
}
__device__ __forceinline__ float murmur3_gumbel_noise_cu(int64_t seed, int64_t pos, uint32_t offset) {
    uint32_t h = 0u;
    h = murmur3_mix_cu(h, static_cast<uint32_t>(seed & 0xFFFFFFFFLL));
    h = murmur3_mix_cu(h, static_cast<uint32_t>((static_cast<uint64_t>(seed) >> 32) & 0xFFFFFFFFULL));
    h = murmur3_mix_cu(h, static_cast<uint32_t>(pos & 0xFFFFFFFFLL));
    h = murmur3_mix_cu(h, offset);
    uint32_t r32 = murmur3_fmix32_cu(h ^ 16u);
    int32_t hi16 = static_cast<int32_t>(r32 >> 16);
    int32_t lo16 = static_cast<int32_t>(r32 & 0xFFFFu);
    float u = static_cast<float>(hi16) * 1.52587890625e-05f
            + (static_cast<float>(lo16) + 0.5f) * 2.3283064365386963e-10f;
    float poly = 1.0f / 8.0f;
    poly = 1.0f / 7.0f + u * poly;
    poly = 1.0f / 6.0f + u * poly;
    poly = 1.0f / 5.0f + u * poly;
    poly = 1.0f / 4.0f + u * poly;
    poly = 1.0f / 3.0f + u * poly;
    poly = 1.0f / 2.0f + u * poly;
    poly = 1.0f + u * poly;
    float series = -u * poly;
    float direct = logf(fmaxf(1.0f - u, 5.960464477539063e-08f));
    float l1p = (u < 0.25f) ? series : direct;
    return -logf(-l1p);
}

// Grid: (K + 255) / 256 CTAs (192 for K=49152, 128 for K=32768), Block: 256 threads (8 warps).
// Each warp computes 32 rows (4 groups of 8 rows) with identical warp_sum_rn & bf16 rounding
// to markov_fold_softcap_r1_sm80_kernel!
__global__ void __launch_bounds__(256, 3) markov_fold_gumbel_cache_r1_sm80_kernel(
    const uint4* __restrict__ latent,               // [1, 256] bf16 (32 uint4s)
    const uint4* __restrict__ fold_w,               // [K, 256] bf16 (K * 32 uint4s)
    const uint4* __restrict__ fold_b,               // [K] bf16 (K / 8 uint4s)
    const uint4* __restrict__ base_logits,          // [K] bf16 (K / 8 uint4s)
    const int64_t* __restrict__ keep_ids,           // [K] int64
    const int64_t* __restrict__ idx_map,            // [1] int64
    const float* __restrict__ temp_ptr,             // [max_reqs] fp32
    const int64_t* __restrict__ seeds_ptr,          // [max_reqs] int64
    const int64_t* __restrict__ sample_pos_ptr,     // [1] int64
    const int32_t* __restrict__ row_mapping_ptr,    // [max_reqs] int32 or nullptr
    __nv_bfloat16* __restrict__ logits_cache,       // [max_rows, n_spec, V_target] bf16
    float* __restrict__ ws_max,                     // [256] fp32 scratch
    int64_t* __restrict__ ws_id,                    // [256] int64 scratch
    int32_t* __restrict__ cta_counter,              // [1] int32 zeroed counter
    int64_t* __restrict__ out_sampled,              // [1] int64
    int64_t* __restrict__ out_draft_col,            // [1] int64 (&draft_tokens[0, step])
    const int64_t cache_stride_0,
    const int64_t cache_stride_1,
    const int32_t step_col,
    const int32_t K_vocab,
    const float softcap)
{
    __shared__ float s_wmax[8];
    __shared__ int64_t s_wid[8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;   // 0..7
    const int lane = tid & 31;   // 0..31

    float lat[8];
    unpack_bf16x8(latent[lane], lat);

    const int64_t req_state_idx = idx_map[0];
    const bool valid_req = (req_state_idx >= 0);
    const float temp = valid_req ? temp_ptr[req_state_idx] : 0.0f;
    int64_t cache_row = req_state_idx;
    if (row_mapping_ptr != nullptr) {
        cache_row = valid_req ? static_cast<int64_t>(row_mapping_ptr[req_state_idx]) : -1;
    }
    const int64_t raw_seed = valid_req ? seeds_ptr[req_state_idx] : 0LL;
    const int64_t seed = raw_seed ^ (static_cast<int64_t>(step_col) << 32);
    const int64_t pos_salted = (sample_pos_ptr[0] - 1LL) + (1LL << 30);
    const float inv_sc = 1.0f / softcap;

    float best_val = -INFINITY;
    int64_t best_id = (1LL << 60);

    // Each warp handles 4 chunks of 8 rows = 32 rows (8 warps * 32 = 256 rows/CTA)
    const int warp_base = blockIdx.x * 256 + warp * 32;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        int row0 = warp_base + g * 8;
        if (row0 < K_vocab) {
            float dots[8];
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                float w[8];
                unpack_bf16x8(fold_w[(row0 + r) * 32 + lane], w);
                float s = 0.f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) s = __fmaf_rn(lat[j], w[j], s);
                dots[r] = warp_sum_rn(s);
            }
            // Broadcast dots[0..7] so lane r in 0..7 processes row (row0 + r) in parallel!
            float my_dot = 0.f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                float d_r = __shfl_sync(0xffffffffu, dots[r], 0);
                if (lane == r) my_dot = d_r;
            }
            if (lane < 8) {
                int col = row0 + lane;
                const __nv_bfloat16* fb_ptr = reinterpret_cast<const __nv_bfloat16*>(fold_b);
                const __nv_bfloat16* bl_ptr = reinterpret_cast<const __nv_bfloat16*>(base_logits);
                float fb_v = bf2f(fb_ptr[col]);
                float bl_v = bf2f(bl_ptr[col]);
                float bias_bf = bf2f(f2bf(__fadd_rn(fb_v, my_dot)));
                float x_bf = bf2f(f2bf(__fadd_rn(bl_v, bias_bf)));
                float out_f = __fmul_rn(softcap, tanhf(__fmul_rn(x_bf, inv_sc)));
                __nv_bfloat16 logit_bf = f2bf(out_f);
                float logit_f32 = bf2f(logit_bf);
                int64_t target_id = keep_ids[col];
                if (cache_row >= 0) {
                    logits_cache[cache_row * cache_stride_0 + static_cast<int64_t>(step_col) * cache_stride_1 + target_id] = logit_bf;
                }
                float score = logit_f32;
                if (temp != 0.0f) {
                    score = (score / temp) + murmur3_gumbel_noise_cu(seed, pos_salted, static_cast<uint32_t>(target_id));
                }
                if (score > best_val || (score == best_val && target_id < best_id)) {
                    best_val = score;
                    best_id = target_id;
                }
            }
        }
    }

    // Reduce across lanes 0..7 in each warp
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffu, best_val, offset);
        int64_t oid = __shfl_down_sync(0xffu, best_id, offset);
        if (ov > best_val || (ov == best_val && oid < best_id)) {
            best_val = ov;
            best_id = oid;
        }
    }
    if (lane == 0) {
        s_wmax[warp] = best_val;
        s_wid[warp] = best_id;
    }
    __syncthreads();

    if (tid < 8) {
        float v = s_wmax[tid];
        int64_t id = s_wid[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffu, v, offset);
            int64_t oid = __shfl_down_sync(0xffu, id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
        }
        if (tid == 0) {
            ws_max[blockIdx.x] = v;
            ws_id[blockIdx.x] = id;
        }
    }

    __shared__ int s_is_last;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int prev = atomicAdd(cta_counter, 1);
        int target = gridDim.x - 1;
        s_is_last = (prev == target) ? 1 : 0;
        if (prev == target) {
            atomicExch(cta_counter, 0);
        }
    }
    __syncthreads();
    if (!s_is_last) return;

    float cv = (tid < gridDim.x) ? ws_max[tid] : -INFINITY;
    int64_t cid_tok = (tid < gridDim.x) ? ws_id[tid] : (1LL << 60);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, cv, offset);
        int64_t oid = __shfl_down_sync(0xffffffffu, cid_tok, offset);
        if (ov > cv || (ov == cv && oid < cid_tok)) {
            cv = ov;
            cid_tok = oid;
        }
    }
    if (lane == 0) {
        s_wmax[warp] = cv;
        s_wid[warp] = cid_tok;
    }
    __syncthreads();

    if (tid < 8) {
        float v = s_wmax[tid];
        int64_t id = s_wid[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffu, v, offset);
            int64_t oid = __shfl_down_sync(0xffu, id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
        }
        if (tid == 0) {
            out_sampled[0] = id;
            out_draft_col[0] = id;
        }
    }
}

extern "C" void launch_markov_fold_gumbel_cache_r1_sm80(
    const void* latent,
    const void* fold_w,
    const void* fold_bias,
    const void* base_logits,
    const void* keep_ids,
    const void* idx_map,
    const void* temp_ptr,
    const void* seeds_ptr,
    const void* sample_pos_ptr,
    const void* row_mapping_ptr,
    void* logits_cache,
    void* ws_max,
    void* ws_id,
    void* cta_counter,
    void* out_sampled,
    void* out_draft_col,
    int64_t cache_stride_0,
    int64_t cache_stride_1,
    int32_t step_col,
    int32_t K_vocab,
    float softcap,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int32_t n_ctas = (K_vocab + 255) >> 8;
    markov_fold_gumbel_cache_r1_sm80_kernel<<<n_ctas, 256, 0, stream>>>(
        static_cast<const uint4*>(latent),
        static_cast<const uint4*>(fold_w),
        static_cast<const uint4*>(fold_bias),
        static_cast<const uint4*>(base_logits),
        static_cast<const int64_t*>(keep_ids),
        static_cast<const int64_t*>(idx_map),
        static_cast<const float*>(temp_ptr),
        static_cast<const int64_t*>(seeds_ptr),
        static_cast<const int64_t*>(sample_pos_ptr),
        static_cast<const int32_t*>(row_mapping_ptr),
        static_cast<__nv_bfloat16*>(logits_cache),
        static_cast<float*>(ws_max),
        static_cast<int64_t*>(ws_id),
        static_cast<int32_t*>(cta_counter),
        static_cast<int64_t*>(out_sampled),
        static_cast<int64_t*>(out_draft_col),
        cache_stride_0, cache_stride_1, step_col, K_vocab, softcap
    );
}

// ============================================================================
// ROUND 3 v5: 100%-BITWISE-EXACT Tensor-Core (mma.sync.m16n8k16) 1-Launch
// Fused Gate Partials + Top-8 Selection + Parallel moe_align (T <= 8, E=128)
// Grid: (2, 3) = 6 CTAs? No: (4, 3) = 12 CTAs of 128 threads (4 warps x 16 experts = 64 -> 2 N-tiles x 3 K-groups = 6 CTAs?
// Better: BN=32 (2 warps compute 16 experts each, all 4 warps load cp.async!) -> 4 N-tiles x 3 K-groups = 12 CTAs!
// Or BN=16 (warp 0 computes 16 experts, 4 warps load cp.async) -> 8 N-tiles x 3 K-groups = 24 CTAs across 24 SMs!
// ============================================================================
__global__ void __launch_bounds__(128, 4) select_and_align_24cta_mma_m8_128_sm80_kernel(
    const __nv_bfloat16* __restrict__ X,            // [T, 2304] bf16 (T <= 8)
    const __nv_bfloat16* __restrict__ W,            // [128, 2304] bf16
    const float* __restrict__ bias,                 // [128] fp32
    const int32_t* __restrict__ expert_bounds,      // [2] int32
    const uint8_t* __restrict__ is_padding,         // [T] bool/uint8 or nullptr
    float* __restrict__ ws_partials,                // [3 * 8 * 128] fp32
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
    constexpr int STAGES = 3;
    // BK = 128 bf16s = 16 uint4s!
    // s_W: [STAGES][16 experts][16 uint4s] = 3 * 4096B = 12 KB
    // s_X: [STAGES][8 tokens][16 uint4s]   = 3 * 2048B = 6 KB
    __shared__ __align__(16) uint4 s_W[STAGES][16][16];
    __shared__ __align__(16) uint4 s_X[STAGES][8][16];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;    // 0..7
    const int tid_in_grp = lane & 3;   // 0..3

    const int n_tile = blockIdx.x;     // 0..7 -> experts [n_tile*16 .. n_tile*16 + 15]
    const int k_grp  = blockIdx.y;     // 0..2 -> K [0..1024), [1024..2048), [2048..2304)
    const int k_tile_start = k_grp * 8; // 128-wide tiles: 0, 8, 16
    const int num_k = (k_grp < 2) ? 8 : 2;

    auto issue_stage = [&](int st, int kt) {
        int k_base = kt * 128;
        // 16 * 16 = 256 uint4s of W across 128 threads (2 uint4s/thread)
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx = tid + (u << 7); // 0..255
            int r = idx >> 4;         // 0..15
            int c = idx & 15;         // 0..15
            int sc = c ^ (r & 7);
            int g_e = (n_tile << 4) + r;
            uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_W[st][r][sc]));
            cp_async_cg_16B(dst, W + static_cast<int64_t>(g_e) * 2304 + k_base + (c << 3));
        }
        // 8 * 16 = 128 uint4s of X across 128 threads (1 uint4/thread)
        {
            int m_row = tid >> 4; // 0..7
            int c = tid & 15;     // 0..15
            int sc = c ^ m_row;
            uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_X[st][m_row][sc]));
            if (m_row < T) {
                cp_async_ca_16B(dst, X + static_cast<int64_t>(m_row) * 2304 + k_base + (c << 3));
            } else {
                s_X[st][m_row][sc] = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
    };

    const int preload = (num_k < (STAGES - 1)) ? num_k : (STAGES - 1);
    for (int p = 0; p < preload; ++p) {
        issue_stage(p, k_tile_start + p);
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int step = 0; step < num_k; ++step) {
        int next_p = step + (STAGES - 1);
        if (next_p < num_k) {
            issue_stage(next_p % STAGES, k_tile_start + next_p);
            cp_async_wait<STAGES - 1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        if (warp_id == 0) {
            const int st = step % STAGES;
            const int w_row0 = group_id;
            const int w_row1 = group_id + 8;
            // 8 k16 subblocks across BK = 128!
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                int c_lo = (kk << 1);
                int c_hi = c_lo + 1;
                const uint32_t* w0_lo = reinterpret_cast<const uint32_t*>(&s_W[st][w_row0][c_lo ^ (w_row0 & 7)]);
                const uint32_t* w1_lo = reinterpret_cast<const uint32_t*>(&s_W[st][w_row1][c_lo ^ (w_row1 & 7)]);
                const uint32_t* w0_hi = reinterpret_cast<const uint32_t*>(&s_W[st][w_row0][c_hi ^ (w_row0 & 7)]);
                const uint32_t* w1_hi = reinterpret_cast<const uint32_t*>(&s_W[st][w_row1][c_hi ^ (w_row1 & 7)]);
                const uint32_t* x_lo  = reinterpret_cast<const uint32_t*>(&s_X[st][group_id][c_lo ^ group_id]);
                const uint32_t* x_hi  = reinterpret_cast<const uint32_t*>(&s_X[st][group_id][c_hi ^ group_id]);
                mma_m16n8k16_bf16(
                    acc0, acc1, acc2, acc3,
                    w0_lo[tid_in_grp], w1_lo[tid_in_grp], w0_hi[tid_in_grp], w1_hi[tid_in_grp],
                    x_lo[tid_in_grp], x_hi[tid_in_grp]
                );
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        const int e0 = (n_tile << 4) + group_id;
        const int e1 = e0 + 8;
        const int m0 = (tid_in_grp << 1);
        const int m1 = m0 + 1;
        float* base_p = ws_partials + k_grp * (T * 128);
        if (m0 < T) {
            base_p[m0 * 128 + e0] = acc0;
            base_p[m0 * 128 + e1] = acc2;
        }
        if (m1 < T) {
            base_p[m1 * 128 + e0] = acc1;
            base_p[m1 * 128 + e1] = acc3;
        }
    }

    __shared__ int s_is_last;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int prev = atomicAdd(cta_counter, 1);
        s_is_last = (prev == 23) ? 1 : 0;
        if (prev == 23) {
            atomicExch(cta_counter, 0);
        }
    }
    __syncthreads();
    if (!s_is_last) return;

    // Phase 2 (Last CTA, 128 threads = 4 warps): exact top-8 + parallel moe_align!
    const int numel = T * 8;
    const int4 fill4 = make_int4(numel, numel, numel, numel);
    int4* sorted4 = reinterpret_cast<int4*>(sorted_ids);
    const int n_vec4 = max_pad >> 2;
    for (int idx = tid; idx < n_vec4; idx += 128) sorted4[idx] = fill4;
    for (int idx = (n_vec4 << 2) + tid; idx < max_pad; idx += 128) sorted_ids[idx] = numel;

    __shared__ int32_t s_counts[128];
    __shared__ int32_t s_pad_start[128];
    __shared__ int32_t s_pad_end[128];
    __shared__ int32_t s_warp_sums[4];
    __shared__ int16_t s_expert[64];
    __shared__ int16_t s_slot[64];

    s_counts[tid] = 0;
    const int32_t eb_lo = expert_bounds[0];
    const int32_t eb_hi = expert_bounds[1];
    float b_val[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) b_val[m] = bias[lane + (m << 5)];
    __syncthreads();

    const float* p0 = ws_partials;
    const float* p1 = ws_partials + T * 128;
    const float* p2 = ws_partials + 2 * T * 128;

    for (int t = warp_id; t < T; t += 4) {
        const bool padded_tok = (is_padding != nullptr) ? (is_padding[t] != 0) : false;
        float my_logit[4];
        uint64_t my_key[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int e = lane + (m << 5);
            float lg = __fadd_rn(__fadd_rn(p0[t * 128 + e], p1[t * 128 + e]), p2[t * 128 + e]);
            my_logit[m] = lg;
            float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-lg)));
            float sel = __fadd_rn(sig, b_val[m]);
            uint32_t bits = __float_as_uint(sel);
            bits ^= (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
            my_key[m] = (static_cast<uint64_t>(bits) << 16) | static_cast<uint64_t>(128 - e);
        }

        int sel_e[8];
        float sel_lp[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint64_t best_k = my_key[0];
            int best_m = 0;
            #pragma unroll
            for (int m = 1; m < 4; ++m) {
                if (my_key[m] > best_k) { best_k = my_key[m]; best_m = m; }
            }
            uint64_t w_best = best_k;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                uint64_t other = __shfl_xor_sync(0xffffffffu, w_best, mask);
                if (other > w_best) w_best = other;
            }
            int win_e = 128 - static_cast<int>(w_best & 0xFFFFu);
            if (best_k == w_best) my_key[best_m] = 0ULL;
            int owner_lane = win_e & 31;
            int owner_m = win_e >> 5;
            float cand_lg = (owner_m == 0) ? my_logit[0] : (owner_m == 1) ? my_logit[1] : (owner_m == 2) ? my_logit[2] : my_logit[3];
            float win_lg = __shfl_sync(0xffffffffu, cand_lg, owner_lane);
            sel_e[k] = win_e;
            sel_lp[k] = fminf(win_lg, 0.0f) - log1pf(expf(-fabsf(win_lg)));
        }

        if (lane < 8) {
            float lp = sel_lp[lane];
            float mx = lp;
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 4));
            float sh = expf(__fsub_rn(lp, mx));
            float s02 = __fadd_rn(sh, __shfl_xor_sync(0xffu, sh, 2));
            float s0123 = __fadd_rn(s02, __shfl_xor_sync(0xffu, s02, 1));
            float total = __fadd_rn(s0123, __shfl_xor_sync(0xffu, s0123, 4));
            float lse = __fadd_rn(mx, logf(total));
            float wt = __fmul_rn(expf(__fsub_rn(lp, lse)), route_scale);

            int row = t * 8 + lane;
            int ge = sel_e[lane];
            int le = (!padded_tok && ge >= eb_lo && ge < eb_hi) ? (ge - eb_lo) : -1;
            out_ids[row] = le;
            out_weights[row] = wt;
            out_src2dst[row] = (le >= 0) ? row : -1;
            if (le >= 0) {
                int slot = atomicAdd(&s_counts[le], 1);
                s_expert[row] = static_cast<int16_t>(le);
                s_slot[row] = static_cast<int16_t>(slot);
            } else {
                s_expert[row] = -1;
                s_slot[row] = 0;
            }
        }
    }
    __syncthreads();

    // Parallel prefix scan across 128 experts using 4 warps (32 experts/warp)
    int c_val = s_counts[tid];
    int padded_c = ((c_val + block_size - 1) / block_size) * block_size;
    int scan_v = warp_inclusive_scan_int(padded_c);
    if (lane == 31) s_warp_sums[warp_id] = scan_v;
    __syncthreads();

    if (warp_id == 0 && lane < 4) {
        int ws = s_warp_sums[lane];
        #pragma unroll
        for (int offset = 1; offset < 4; offset <<= 1) {
            int n = __shfl_up_sync(0xfu, ws, offset);
            if (lane >= offset) ws += n;
        }
        s_warp_sums[lane] = ws;
    }
    __syncthreads();

    int warp_prefix = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
    int end_pos = warp_prefix + scan_v;
    int start_pos = end_pos - padded_c;
    s_pad_start[tid] = start_pos;
    s_pad_end[tid] = end_pos;
    if (tid == 127) {
        total_tokens_post_pad[0] = end_pos;
    }
    __syncthreads();

    for (int idx = tid; idx < numel; idx += 128) {
        int ex = s_expert[idx];
        if (ex >= 0) {
            sorted_ids[s_pad_start[ex] + s_slot[idx]] = idx;
        }
    }
    int total_blocks = s_pad_end[127] / block_size;
    for (int e = warp_id; e < 128; e += 4) {
        int b0 = s_pad_start[e] / block_size;
        int b1 = s_pad_end[e] / block_size;
        for (int b = b0 + lane; b < b1; b += 32) {
            expert_ids[b] = e;
        }
    }
    for (int b = total_blocks + tid; b < n_blocks; b += 128) {
        expert_ids[b] = -1;
    }
}

extern "C" void launch_select_and_align_24cta_mma_m8_128_sm80(
    const void* X, const void* W, const void* bias, const void* expert_bounds, const void* is_padding,
    void* ws_partials, void* cta_counter,
    void* out_ids, void* out_weights, void* out_src2dst,
    void* sorted_ids, void* expert_ids, void* total_tokens_post_pad,
    int32_t T, int32_t block_size, int32_t max_pad, int32_t n_blocks, float route_scale, void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    dim3 grid(8, 3);
    select_and_align_24cta_mma_m8_128_sm80_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(X),
        static_cast<const __nv_bfloat16*>(W),
        static_cast<const float*>(bias),
        static_cast<const int32_t*>(expert_bounds),
        static_cast<const uint8_t*>(is_padding),
        static_cast<float*>(ws_partials),
        static_cast<int32_t*>(cta_counter),
        static_cast<int32_t*>(out_ids),
        static_cast<float*>(out_weights),
        static_cast<int32_t*>(out_src2dst),
        static_cast<int32_t*>(sorted_ids),
        static_cast<int32_t*>(expert_ids),
        static_cast<int32_t*>(total_tokens_post_pad),
        T, block_size, max_pad, n_blocks, route_scale
    );
}


// ============================================================================
// ROUND 4 ADDITIONS (sm80_fast_v5.cu -> libsm80_fast_v5.so)
// 1. Fused OnlineAcceptanceEstimator + Markov Fold + Softcap + Sparse 49K Cache
//    + Exact Murmur3 Gumbel-Max Argmax in ONE launch (0 extra launches for R4-2!)
// 2. SM80 Hardware L2 Persisting Residency Window control (cudaAccessPolicyWindow)
// ============================================================================

__device__ __forceinline__ void merge_max_excl_sumexp(
    float& m1, float& e1, float m2, float e2)
{
    if (m2 == -INFINITY) return;
    if (m1 == -INFINITY) {
        m1 = m2;
        e1 = e2;
        return;
    }
    float hi = fmaxf(m1, m2);
    float lo = fminf(m1, m2);
    float s_hi = (m1 >= m2) ? e1 : e2;
    float s_lo = (m1 >= m2) ? e2 : e1;
    m1 = hi;
    e1 = s_hi + (s_lo + 1.0f) * expf(lo - hi);
}

__global__ void __launch_bounds__(256, 3) markov_fold_gumbel_est_cache_r1_sm80_kernel(
    const uint4* __restrict__ latent,               // [1, 256] bf16
    const uint4* __restrict__ fold_w,               // [K, 256] bf16
    const uint4* __restrict__ fold_b,               // [K] bf16
    const uint4* __restrict__ base_logits,          // [K] bf16
    const int64_t* __restrict__ keep_ids,           // [K] int64
    const int64_t* __restrict__ idx_map,            // [1] int64
    const float* __restrict__ temp_ptr,             // [max_reqs] fp32
    const int64_t* __restrict__ seeds_ptr,          // [max_reqs] int64
    const int64_t* __restrict__ sample_pos_ptr,     // [1] int64
    const int32_t* __restrict__ row_mapping_ptr,    // [max_reqs] int32 or nullptr
    __nv_bfloat16* __restrict__ logits_cache,       // [max_rows, n_spec, V_target] bf16
    float* __restrict__ ws_max,                     // [256] fp32 scratch (gumbel)
    int64_t* __restrict__ ws_id,                    // [256] int64 scratch (gumbel)
    float* __restrict__ ws_est_m,                   // [256] fp32 scratch (est max)
    float* __restrict__ ws_est_e,                   // [256] fp32 scratch (est excl sumexp)
    int32_t* __restrict__ cta_counter,              // [1] int32 zeroed counter
    int64_t* __restrict__ out_sampled,              // [1] int64
    int64_t* __restrict__ out_draft_col,            // [1] int64
    float* __restrict__ est_features,               // [max_reqs, n_spec] fp32
    float* __restrict__ est_preds,                  // [max_reqs, n_spec] fp32
    float* __restrict__ est_conf,                   // [max_reqs, n_spec] fp32
    const float* __restrict__ est_slope,            // [1] fp32
    const float* __restrict__ est_intercepts,       // [n_spec] fp32
    const int64_t cache_stride_0,
    const int64_t cache_stride_1,
    const int64_t est_stride,
    const int32_t step_col,
    const int32_t K_vocab,
    const float softcap)
{
    __shared__ float s_wmax[8];
    __shared__ int64_t s_wid[8];
    __shared__ float s_em[8];
    __shared__ float s_ee[8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float lat[8];
    unpack_bf16x8(latent[lane], lat);

    const int64_t req_state_idx = idx_map[0];
    const bool valid_req = (req_state_idx >= 0);
    const float temp = valid_req ? temp_ptr[req_state_idx] : 0.0f;
    const float inv_est_temp = (temp > 0.0f) ? (1.0f / temp) : 1.0f;
    int64_t cache_row = req_state_idx;
    if (row_mapping_ptr != nullptr) {
        cache_row = valid_req ? static_cast<int64_t>(row_mapping_ptr[req_state_idx]) : -1;
    }
    const int64_t raw_seed = valid_req ? seeds_ptr[req_state_idx] : 0LL;
    const int64_t seed = raw_seed ^ (static_cast<int64_t>(step_col) << 32);
    const int64_t pos_salted = (sample_pos_ptr[0] - 1LL) + (1LL << 30);
    const float inv_sc = 1.0f / softcap;

    float best_val = -INFINITY;
    int64_t best_id = (1LL << 60);
    float est_m = -INFINITY;
    float est_e = 0.0f;

    const int warp_base = blockIdx.x * 256 + warp * 32;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        int row0 = warp_base + g * 8;
        if (row0 < K_vocab) {
            float dots[8];
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                float w[8];
                unpack_bf16x8(fold_w[(row0 + r) * 32 + lane], w);
                float s = 0.f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) s = __fmaf_rn(lat[j], w[j], s);
                dots[r] = warp_sum_rn(s);
            }
            float my_dot = 0.f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                float d_r = __shfl_sync(0xffffffffu, dots[r], 0);
                if (lane == r) my_dot = d_r;
            }
            if (lane < 8) {
                int col = row0 + lane;
                const __nv_bfloat16* fb_ptr = reinterpret_cast<const __nv_bfloat16*>(fold_b);
                const __nv_bfloat16* bl_ptr = reinterpret_cast<const __nv_bfloat16*>(base_logits);
                float fb_v = bf2f(fb_ptr[col]);
                float bl_v = bf2f(bl_ptr[col]);
                float bias_bf = bf2f(f2bf(__fadd_rn(fb_v, my_dot)));
                float x_bf = bf2f(f2bf(__fadd_rn(bl_v, bias_bf)));
                float out_f = __fmul_rn(softcap, tanhf(__fmul_rn(x_bf, inv_sc)));
                __nv_bfloat16 logit_bf = f2bf(out_f);
                float logit_f32 = bf2f(logit_bf);
                int64_t target_id = keep_ids[col];
                if (cache_row >= 0) {
                    logits_cache[cache_row * cache_stride_0 + static_cast<int64_t>(step_col) * cache_stride_1 + target_id] = logit_bf;
                }
                float scaled_lg = logit_f32 * inv_est_temp;
                merge_max_excl_sumexp(est_m, est_e, scaled_lg, 0.0f);

                float score = logit_f32;
                if (temp != 0.0f) {
                    score = scaled_lg + murmur3_gumbel_noise_cu(seed, pos_salted, static_cast<uint32_t>(target_id));
                }
                if (score > best_val || (score == best_val && target_id < best_id)) {
                    best_val = score;
                    best_id = target_id;
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffu, best_val, offset);
        int64_t oid = __shfl_down_sync(0xffu, best_id, offset);
        if (ov > best_val || (ov == best_val && oid < best_id)) {
            best_val = ov;
            best_id = oid;
        }
        float om = __shfl_down_sync(0xffu, est_m, offset);
        float oe = __shfl_down_sync(0xffu, est_e, offset);
        merge_max_excl_sumexp(est_m, est_e, om, oe);
    }
    if (lane == 0) {
        s_wmax[warp] = best_val;
        s_wid[warp] = best_id;
        s_em[warp] = est_m;
        s_ee[warp] = est_e;
    }
    __syncthreads();

    if (tid < 8) {
        float v = s_wmax[tid];
        int64_t id = s_wid[tid];
        float em = s_em[tid];
        float ee = s_ee[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffu, v, offset);
            int64_t oid = __shfl_down_sync(0xffu, id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
            float om = __shfl_down_sync(0xffu, em, offset);
            float oe = __shfl_down_sync(0xffu, ee, offset);
            merge_max_excl_sumexp(em, ee, om, oe);
        }
        if (tid == 0) {
            ws_max[blockIdx.x] = v;
            ws_id[blockIdx.x] = id;
            ws_est_m[blockIdx.x] = em;
            ws_est_e[blockIdx.x] = ee;
        }
    }

    __shared__ int s_is_last;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int prev = atomicAdd(cta_counter, 1);
        int target = gridDim.x - 1;
        s_is_last = (prev == target) ? 1 : 0;
        if (prev == target) {
            atomicExch(cta_counter, 0);
        }
    }
    __syncthreads();
    if (!s_is_last) return;

    float cv = (tid < gridDim.x) ? ws_max[tid] : -INFINITY;
    int64_t cid_tok = (tid < gridDim.x) ? ws_id[tid] : (1LL << 60);
    float cem = (tid < gridDim.x) ? ws_est_m[tid] : -INFINITY;
    float cee = (tid < gridDim.x) ? ws_est_e[tid] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, cv, offset);
        int64_t oid = __shfl_down_sync(0xffffffffu, cid_tok, offset);
        if (ov > cv || (ov == cv && oid < cid_tok)) {
            cv = ov;
            cid_tok = oid;
        }
        float om = __shfl_down_sync(0xffffffffu, cem, offset);
        float oe = __shfl_down_sync(0xffffffffu, cee, offset);
        merge_max_excl_sumexp(cem, cee, om, oe);
    }
    if (lane == 0) {
        s_wmax[warp] = cv;
        s_wid[warp] = cid_tok;
        s_em[warp] = cem;
        s_ee[warp] = cee;
    }
    __syncthreads();

    if (tid < 8) {
        float v = s_wmax[tid];
        int64_t id = s_wid[tid];
        float em = s_em[tid];
        float ee = s_ee[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffu, v, offset);
            int64_t oid = __shfl_down_sync(0xffu, id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
            float om = __shfl_down_sync(0xffu, em, offset);
            float oe = __shfl_down_sync(0xffu, ee, offset);
            merge_max_excl_sumexp(em, ee, om, oe);
        }
        if (tid == 0) {
            out_sampled[0] = id;
            out_draft_col[0] = id;
            if (valid_req && est_features != nullptr) {
                float feat = -logf(ee);
                feat = fminf(fmaxf(feat, -40.0f), 40.0f);
                est_features[req_state_idx * est_stride + step_col] = feat;
                float prob = 1.0f / (1.0f + expf(-(est_slope[0] * feat + est_intercepts[step_col])));
                est_preds[req_state_idx * est_stride + step_col] = prob;
                est_conf[step_col] = prob;
            }
        }
    }
}

extern "C" void launch_markov_fold_gumbel_est_cache_r1_sm80(
    const void* latent,
    const void* fold_w,
    const void* fold_bias,
    const void* base_logits,
    const void* keep_ids,
    const void* idx_map,
    const void* temp_ptr,
    const void* seeds_ptr,
    const void* sample_pos_ptr,
    const void* row_mapping_ptr,
    void* logits_cache,
    void* ws_max,
    void* ws_id,
    void* ws_est_m,
    void* ws_est_e,
    void* cta_counter,
    void* out_sampled,
    void* out_draft_col,
    void* est_features,
    void* est_preds,
    void* est_conf,
    const void* est_slope,
    const void* est_intercepts,
    int64_t cache_stride_0,
    int64_t cache_stride_1,
    int64_t est_stride,
    int32_t step_col,
    int32_t K_vocab,
    float softcap,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int32_t n_ctas = (K_vocab + 255) >> 8;
    markov_fold_gumbel_est_cache_r1_sm80_kernel<<<n_ctas, 256, 0, stream>>>(
        static_cast<const uint4*>(latent),
        static_cast<const uint4*>(fold_w),
        static_cast<const uint4*>(fold_bias),
        static_cast<const uint4*>(base_logits),
        static_cast<const int64_t*>(keep_ids),
        static_cast<const int64_t*>(idx_map),
        static_cast<const float*>(temp_ptr),
        static_cast<const int64_t*>(seeds_ptr),
        static_cast<const int64_t*>(sample_pos_ptr),
        static_cast<const int32_t*>(row_mapping_ptr),
        static_cast<__nv_bfloat16*>(logits_cache),
        static_cast<float*>(ws_max),
        static_cast<int64_t*>(ws_id),
        static_cast<float*>(ws_est_m),
        static_cast<float*>(ws_est_e),
        static_cast<int32_t*>(cta_counter),
        static_cast<int64_t*>(out_sampled),
        static_cast<int64_t*>(out_draft_col),
        static_cast<float*>(est_features),
        static_cast<float*>(est_preds),
        static_cast<float*>(est_conf),
        static_cast<const float*>(est_slope),
        static_cast<const float*>(est_intercepts),
        cache_stride_0, cache_stride_1, est_stride, step_col, K_vocab, softcap
    );
}

extern "C" int set_l2_persisting_window_sm80(
    const void* base_ptr,
    uint64_t num_bytes,
    uint64_t limit_bytes,
    float hit_ratio,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    if (limit_bytes > 0) {
        cudaError_t e0 = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, static_cast<size_t>(limit_bytes));
        if (e0 != cudaSuccess) return static_cast<int>(e0);
    }
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(base_ptr);
    attr.accessPolicyWindow.num_bytes = static_cast<size_t>(num_bytes);
    attr.accessPolicyWindow.hitRatio = hit_ratio;
    attr.accessPolicyWindow.hitProp = (num_bytes > 0) ? cudaAccessPropertyPersisting : cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaError_t e1 = cudaStreamSetAttribute(
        stream,
        static_cast<cudaStreamAttrID>(cudaLaunchAttributeAccessPolicyWindow),
        &attr
    );
    if (num_bytes == 0) {
        cudaCtxResetPersistingL2Cache();
    }
    return static_cast<int>(e1);
}

// ============================================================================
// ROUND 5 SM80 BREAKTHROUGH KERNELS (sm80_fast_v6.cu)
// 1. select_and_worklist_64cta_128_sm80_kernel:
//    64-CTA (2x SM coverage) BF16 gate GEMV + exact Inkling2 top-8 selection
//    + parallel moe_align + compact active-expert worklist (U <= 64) for T <= 16.
// 2. sm80_moe_w13_swiglu_kernel<IS_NVFP4> & sm80_moe_w2_kernel<IS_NVFP4>:
//    0-extra-VRAM persistent BN=64 BK=64 cp.async + mma.sync.aligned.m16n8k16
//    kernels reading Humming repacked NVFP4/MXFP8 expert weights directly,
//    eliminating the 2.13-wave tail bubble (315 -> 1,260 tiles across 222/296
//    CTAs = 5.68 waves!) and fusing bit-exact SwiGLU in registers on w13!
// 3. suffix_logits_pointmass_sm80_kernel:
//    Ultra-fast GPU-resident DraftLogitsCache sparse -inf hygiene + 0.0
//    point-mass writer for Hybrid SuffixDecoding + DSpark (100% lossless!).
// ============================================================================

__device__ __forceinline__ __nv_bfloat16 swiglu_bf16_exact(float g, float up, float gamma) {
    float sig = sigmoid_approx_ptx(g);
    float gc = __fmul_rn(4.0f, tanh_approx_ptx(__fmul_rn(g, 0.25f)));
    float uc = __fmul_rn(25.0f, tanh_approx_ptx(div_rn_25(up)));
    float val = __fmul_rn(__fmul_rn(__fmul_rn(gc, sig), uc), gamma);
    return __float2bfloat16_rn(val);
}

__global__ void __launch_bounds__(128, 4) select_and_worklist_64cta_128_sm80_kernel(
    const __nv_bfloat16* __restrict__ x,          // [T, 2304] bf16
    const __nv_bfloat16* __restrict__ gate_w,     // [128, 2304] bf16
    const float* __restrict__ bias,               // [128] fp32
    const int32_t* __restrict__ expert_bounds,    // [2] int32
    const uint8_t* __restrict__ is_padding,       // [T] bool or nullptr
    float* __restrict__ ws_logits,                // [16 * 128] fp32 scratch
    int32_t* __restrict__ cta_counter,            // [1] int32 zero-initialized counter
    int32_t* __restrict__ out_ids,                // [T * 8] int32
    float* __restrict__ out_weights,              // [T * 8] fp32
    int32_t* __restrict__ out_src2dst,            // [T * 8] int32
    int32_t* __restrict__ sorted_ids,             // [max_pad] int32 or nullptr
    int32_t* __restrict__ expert_ids,             // [n_blocks] int32 or nullptr
    int32_t* __restrict__ total_tokens_post_pad,  // [1] int32 or nullptr
    int32_t* __restrict__ out_num_active,         // [1] int32 or nullptr (for T <= 8)
    int32_t* __restrict__ out_active_experts,     // [64] int32 or nullptr
    int32_t* __restrict__ out_active_slots,       // [64 * 8] int32 or nullptr
    const int32_t T,
    const int32_t block_size,
    const int32_t max_pad,
    const int32_t n_blocks,
    const float route_scale)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;   // 0..3
    const int lane = tid & 31;      // 0..31
    const int bx = blockIdx.x;
    const int e_grp = bx & 31;      // 0..31
    const int t_start = bx >> 5;    // 0 or 1
    const int t_step = (gridDim.x == 64) ? 2 : 1;
    const int expert = (e_grp << 2) + warp_id; // 0..127

    if (t_start < T) {
        const uint4* w_vec = reinterpret_cast<const uint4*>(gate_w + expert * 2304);
        uint4 w_reg[9];
        #pragma unroll
        for (int it = 0; it < 9; ++it) {
            w_reg[it] = w_vec[it * 32 + lane];
        }

        for (int t = t_start; t < T; t += t_step) {
            const uint4* x_vec = reinterpret_cast<const uint4*>(x + t * 2304);
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;

            #pragma unroll
            for (int it = 0; it < 4; ++it) {
                uint4 xv = x_vec[it * 32 + lane];
                uint4 wv = w_reg[it];
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
                const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    float2 xf = __bfloat1622float2(x2[p]);
                    float2 wf = __bfloat1622float2(w2[p]);
                    g0 += xf.x * wf.x + xf.y * wf.y;
                }
            }
            g0 = warp_sum_f32(g0);

            #pragma unroll
            for (int it = 4; it < 8; ++it) {
                uint4 xv = x_vec[it * 32 + lane];
                uint4 wv = w_reg[it];
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
                const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    float2 xf = __bfloat1622float2(x2[p]);
                    float2 wf = __bfloat1622float2(w2[p]);
                    g1 += xf.x * wf.x + xf.y * wf.y;
                }
            }
            g1 = warp_sum_f32(g1);

            {
                uint4 xv = x_vec[8 * 32 + lane];
                uint4 wv = w_reg[8];
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
                const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    float2 xf = __bfloat1622float2(x2[p]);
                    float2 wf = __bfloat1622float2(w2[p]);
                    g2 += xf.x * wf.x + xf.y * wf.y;
                }
            }
            g2 = warp_sum_f32(g2);

            if (lane == 0) {
                ws_logits[t * 128 + expert] = (g0 + g1) + g2;
            }
        }
    }

    __shared__ int s_is_last;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int target_prev = static_cast<int>(gridDim.x) - 1;
        int prev = atomicAdd(cta_counter, 1);
        s_is_last = (prev == target_prev) ? 1 : 0;
        if (prev == target_prev) {
            atomicExch(cta_counter, 0);
        }
    }
    __syncthreads();
    if (!s_is_last) return;

    const int numel = T * 8;
    if (sorted_ids != nullptr) {
        const int4 fill4 = make_int4(numel, numel, numel, numel);
        int4* sorted4 = reinterpret_cast<int4*>(sorted_ids);
        const int n_vec4 = max_pad >> 2;
        for (int idx = tid; idx < n_vec4; idx += 128) {
            sorted4[idx] = fill4;
        }
        for (int idx = (n_vec4 << 2) + tid; idx < max_pad; idx += 128) {
            sorted_ids[idx] = numel;
        }
    }
    if (out_active_slots != nullptr) {
        // 128 threads x int4 = 512 int32s = 64 experts x 8 slots!
        reinterpret_cast<int4*>(out_active_slots)[tid] = make_int4(-1, -1, -1, -1);
    }

    __shared__ int32_t s_counts[128];
    __shared__ int32_t s_pad_start[128];
    __shared__ int32_t s_pad_end[128];
    __shared__ int32_t s_u_idx[128];
    __shared__ int32_t s_warp_sums[4];
    __shared__ int16_t s_expert[128];
    __shared__ int16_t s_slot[128];

    s_counts[tid] = 0;
    const int32_t eb_lo = expert_bounds[0];
    const int32_t eb_hi = expert_bounds[1];
    float b_val[4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        b_val[m] = bias[lane + (m << 5)];
    }
    __syncthreads();

    for (int t = warp_id; t < T; t += 4) {
        const bool padded_tok = (is_padding != nullptr) ? (is_padding[t] != 0) : false;
        float my_logit[4];
        uint64_t my_key[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int e = lane + (m << 5);
            float lg = ws_logits[t * 128 + e];
            my_logit[m] = lg;
            float sig = __fdividef(1.0f, 1.0f + expf(-lg));
            float sel = sig + b_val[m];
            uint32_t bits = __float_as_uint(sel);
            bits ^= (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
            my_key[m] = (static_cast<uint64_t>(bits) << 16) | static_cast<uint64_t>(128 - e);
        }

        int my_sel_e = 0;
        float my_sel_lp = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint64_t best_k = my_key[0];
            int best_m = 0;
            #pragma unroll
            for (int m = 1; m < 4; ++m) {
                if (my_key[m] > best_k) {
                    best_k = my_key[m];
                    best_m = m;
                }
            }
            uint64_t w_best = best_k;
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                uint64_t other = __shfl_xor_sync(0xffffffffu, w_best, mask);
                if (other > w_best) w_best = other;
            }
            int win_e = 128 - static_cast<int>(w_best & 0xFFFFu);
            if (best_k == w_best) {
                my_key[best_m] = 0ULL;
            }
            int owner_lane = win_e & 31;
            int owner_m = win_e >> 5;
            float cand_lg = (owner_m == 0) ? my_logit[0] : (owner_m == 1) ? my_logit[1] : (owner_m == 2) ? my_logit[2] : my_logit[3];
            float win_lg = __shfl_sync(0xffffffffu, cand_lg, owner_lane);
            float win_lp = fminf(win_lg, 0.0f) - log1pf(expf(-fabsf(win_lg)));
            if (lane == k) {
                my_sel_e = win_e;
                my_sel_lp = win_lp;
            }
        }

        if (lane < 8) {
            float lp = my_sel_lp;
            float mx = lp;
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffu, mx, 4));
            float sh = expf(lp - mx);
            float s02 = sh + __shfl_xor_sync(0xffu, sh, 2);
            float s0123 = s02 + __shfl_xor_sync(0xffu, s02, 1);
            float total = s0123 + __shfl_xor_sync(0xffu, s0123, 4);
            float lse = mx + logf(total);
            float w_out = expf(lp - lse) * route_scale;

            int raw_e = my_sel_e;
            int local_e = (!padded_tok && raw_e >= eb_lo && raw_e < eb_hi) ? (raw_e - eb_lo) : -1;
            int row = t * 8 + lane;
            out_ids[row] = local_e;
            out_weights[row] = w_out;
            out_src2dst[row] = (local_e >= 0) ? row : -1;
            if (local_e >= 0) {
                int slot = atomicAdd(&s_counts[local_e], 1);
                s_expert[row] = static_cast<int16_t>(local_e);
                s_slot[row] = static_cast<int16_t>(slot);
            } else {
                s_expert[row] = static_cast<int16_t>(-1);
                s_slot[row] = static_cast<int16_t>(0);
            }
        }
    }
    __syncthreads();

    const int cnt = s_counts[tid];
    if (sorted_ids != nullptr) {
        const int padded = ((cnt + block_size - 1) / block_size) * block_size;
        int scan = warp_inclusive_scan_int(padded);
        if (lane == 31) s_warp_sums[warp_id] = scan;
        __syncthreads();

        if (warp_id == 0) {
            int wval = (lane < 4) ? s_warp_sums[lane] : 0;
            wval = warp_inclusive_scan_int(wval);
            if (lane < 4) s_warp_sums[lane] = wval;
        }
        __syncthreads();

        const int warp_prefix = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
        const int end = scan + warp_prefix;
        const int start = end - padded;
        s_pad_start[tid] = start;
        s_pad_end[tid] = end;
        if (tid == 127) {
            total_tokens_post_pad[0] = end;
        }
        __syncthreads();

        const int total_blocks = s_pad_end[127] / block_size;
        if (padded > 0) {
            int b_start = start / block_size;
            int b_end = end / block_size;
            for (int b = b_start; b < b_end; ++b) {
                if (b < n_blocks) expert_ids[b] = tid;
            }
        }
        for (int b = total_blocks + tid; b < n_blocks; b += 128) {
            expert_ids[b] = 0;
        }

        if (tid < numel) {
            int e = s_expert[tid];
            if (e >= 0) {
                sorted_ids[s_pad_start[e] + s_slot[tid]] = tid;
            }
        }
    }

    if (out_num_active != nullptr) {
        if (sorted_ids != nullptr) __syncthreads();
        const int is_act = (cnt > 0) ? 1 : 0;
        int ascan = warp_inclusive_scan_int(is_act);
        if (lane == 31) s_warp_sums[warp_id] = ascan;
        __syncthreads();

        if (warp_id == 0) {
            int wval = (lane < 4) ? s_warp_sums[lane] : 0;
            wval = warp_inclusive_scan_int(wval);
            if (lane < 4) s_warp_sums[lane] = wval;
        }
        __syncthreads();

        const int aprefix = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
        const int u_end = ascan + aprefix;
        if (tid == 127) {
            out_num_active[0] = u_end;
        }
        if (is_act) {
            const int u = u_end - 1;
            out_active_experts[u] = tid;
            s_u_idx[tid] = u;
        }
        __syncthreads();

        if (tid < numel) {
            int e = s_expert[tid];
            if (e >= 0) {
                out_active_slots[s_u_idx[e] * 8 + s_slot[tid]] = tid;
            }
        }
    }
}

extern "C" void launch_select_and_worklist_64cta_128_sm80(
    const void* x,
    const void* gate_w,
    const void* bias,
    const void* expert_bounds,
    const void* is_padding,
    void* ws_logits,
    void* cta_counter,
    void* out_ids,
    void* out_weights,
    void* out_src2dst,
    void* sorted_ids,
    void* expert_ids,
    void* total_tokens_post_pad,
    void* out_num_active,
    void* out_active_experts,
    void* out_active_slots,
    int32_t T,
    int32_t block_size,
    int32_t max_pad,
    int32_t n_blocks,
    float route_scale,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int grid = (T >= 2) ? 64 : 32;
    select_and_worklist_64cta_128_sm80_kernel<<<grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(gate_w),
        static_cast<const float*>(bias),
        static_cast<const int32_t*>(expert_bounds),
        static_cast<const uint8_t*>(is_padding),
        static_cast<float*>(ws_logits),
        static_cast<int32_t*>(cta_counter),
        static_cast<int32_t*>(out_ids),
        static_cast<float*>(out_weights),
        static_cast<int32_t*>(out_src2dst),
        static_cast<int32_t*>(sorted_ids),
        static_cast<int32_t*>(expert_ids),
        static_cast<int32_t*>(total_tokens_post_pad),
        static_cast<int32_t*>(out_num_active),
        static_cast<int32_t*>(out_active_experts),
        static_cast<int32_t*>(out_active_slots),
        T, block_size, max_pad, n_blocks, route_scale
    );
}

// Bit-exact Humming software dequant helpers for SM80 (Float4E2M1+E4M3 & Float8E4M3+E8M0)
__device__ __forceinline__ void dequant_nvfp4_mma16x16_sm80(
    uint32_t qb_word,
    uint16_t spair,
    uint32_t (&res)[4])
{
    // Extract 8x E2M1 nibbles -> 4x bf162 in exact Humming dequant_b1248 order
    uint32_t q0 = qb_word << 12;
    uint32_t q1 = qb_word << 8;
    uint32_t q2 = qb_word << 4;
    uint32_t q3 = qb_word;

    res[0] = (q0 & 0x80008000u) | ((q0 & 0x70007000u) >> 6);
    res[1] = (q1 & 0x80008000u) | ((q1 & 0x70007000u) >> 6);
    res[2] = (q2 & 0x80008000u) | ((q2 & 0x70007000u) >> 6);
    res[3] = (q3 & 0x80008000u) | ((q3 & 0x70007000u) >> 6);

    // Group scale E4M3 -> BF16 * 2^127 * 2^112 (matching Humming mainloop_arith.cuh:134-149)
    uint32_t s_lo = ((spair & 0x0080u) << 8) | ((spair & 0x007Fu) << 4);
    uint32_t s_hi = ((spair & 0x8000u)     ) | ((spair & 0x7F00u) >> 4);
    uint32_t s_pk = (s_hi << 16) | s_lo;
    const uint32_t kFac127 = 0x7F007F00u;
    const uint32_t kFac112 = 0x77807780u;
    __nv_bfloat162 s01 = *reinterpret_cast<const __nv_bfloat162*>(&s_pk);
    s01 = __hmul2(s01, *reinterpret_cast<const __nv_bfloat162*>(&kFac127));
    s01 = __hmul2(s01, *reinterpret_cast<const __nv_bfloat162*>(&kFac112));
    __nv_bfloat162 bs0 = __bfloat162bfloat162(__low2bfloat16(s01));
    __nv_bfloat162 bs1 = __bfloat162bfloat162(__high2bfloat16(s01));

    __nv_bfloat162* b2 = reinterpret_cast<__nv_bfloat162*>(res);
    b2[0] = __hmul2(b2[0], bs0);
    b2[1] = __hmul2(b2[1], bs0);
    b2[2] = __hmul2(b2[2], bs1);
    b2[3] = __hmul2(b2[3], bs1);
}

__device__ __forceinline__ void dequant_mxfp8_mma16x16_sm80(
    uint2 qb2,
    uint16_t spair,
    uint32_t (&res)[4])
{
    uint32_t q0 = qb2.x << 8;
    uint32_t q1 = qb2.x;
    uint32_t q2 = qb2.y << 8;
    uint32_t q3 = qb2.y;

    res[0] = (q0 & 0x80008000u) | ((q0 & 0x7F007F00u) >> 4);
    res[1] = (q1 & 0x80008000u) | ((q1 & 0x7F007F00u) >> 4);
    res[2] = (q2 & 0x80008000u) | ((q2 & 0x7F007F00u) >> 4);
    res[3] = (q3 & 0x80008000u) | ((q3 & 0x7F007F00u) >> 4);

    const uint32_t kFac120 = 0x7B807B80u;
    const __nv_bfloat162 f120 = *reinterpret_cast<const __nv_bfloat162*>(&kFac120);
    uint32_t s_lo = static_cast<uint32_t>(spair & 0x00FFu) << 7;
    uint32_t s_hi = static_cast<uint32_t>(spair >> 8) << 7;
    uint32_t bs0_u = (s_lo << 16) | s_lo;
    uint32_t bs1_u = (s_hi << 16) | s_hi;
    const __nv_bfloat162 bs0 = *reinterpret_cast<const __nv_bfloat162*>(&bs0_u);
    const __nv_bfloat162 bs1 = *reinterpret_cast<const __nv_bfloat162*>(&bs1_u);

    __nv_bfloat162* b2 = reinterpret_cast<__nv_bfloat162*>(res);
    b2[0] = __hmul2(__hmul2(b2[0], f120), bs0);
    b2[1] = __hmul2(__hmul2(b2[1], f120), bs0);
    b2[2] = __hmul2(__hmul2(b2[2], f120), bs1);
    b2[3] = __hmul2(__hmul2(b2[3], f120), bs1);
}

template <bool IS_NVFP4, int CTA_PER_SM>
__global__ void __launch_bounds__(128, CTA_PER_SM) sm80_moe_w13_swiglu_kernel(
    const __nv_bfloat16* __restrict__ X,          // [T, 2304] bf16
    const uint4* __restrict__ W_u4,               // Humming tensors["w13"]["weight"]
    const uint4* __restrict__ BS_u4,              // Humming tensors["w13"]["weight_scale"]
    const float* __restrict__ BS2,                // [128] fp32 or nullptr
    const int32_t* __restrict__ num_active_ptr,   // [1] int32
    const int32_t* __restrict__ active_experts,   // [64] int32
    const int32_t* __restrict__ active_slots,     // [64 * 8] int32
    const float* __restrict__ route_weights,      // [T * 8] fp32
    __nv_bfloat16* __restrict__ OutAct)           // [T * 8, 1152] bf16
{
    const int U = num_active_ptr[0];
    if (U <= 0) return;

    constexpr int STAGES = 4;
    constexpr int NUM_K_ITERS = 36; // 2304 / 64 = 36
    constexpr int QB_U4_PER_SLAB = IS_NVFP4 ? 32 : 64;
    constexpr int BS_GROUPS_PER_STAGE = IS_NVFP4 ? 4 : 2;

    __shared__ __align__(16) uint4 s_QB[STAGES][4][QB_U4_PER_SLAB];
    __shared__ __align__(16) uint4 s_BS[STAGES][BS_GROUPS_PER_STAGE][4];
    __shared__ __align__(16) uint4 s_X[STAGES][8][8];
    __shared__ int32_t s_expert_id;
    __shared__ int32_t s_slot_m[8];
    __shared__ int32_t s_tok_m[8];
    __shared__ float   s_gamma_m[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;
    const int total_tiles = U * 36;

    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        const int u = tile_idx / 36;
        const int nb = tile_idx - u * 36;

        if (tid == 0) {
            s_expert_id = active_experts[u];
        }
        if (tid < 8) {
            int sl = active_slots[u * 8 + tid];
            s_slot_m[tid] = sl;
            s_tok_m[tid] = (sl >= 0) ? (sl >> 3) : -1;
            s_gamma_m[tid] = (sl >= 0) ? route_weights[sl] : 0.0f;
        }
        __syncthreads();

        const int expert = s_expert_id;
        const float gs_f32 = IS_NVFP4 ? (BS2[expert] * 128.0f) : 1.0f;

        auto issue_stage = [&](int st, int ki) {
            const int k_base = ki * 64;
            if (tid < 64) {
                int m_row = tid >> 3;
                int c = tid & 7;
                int sc = c ^ m_row;
                int tok = s_tok_m[m_row];
                uint32_t dst_x = static_cast<uint32_t>(__cvta_generic_to_shared(&s_X[st][m_row][sc]));
                if (tok >= 0) {
                    cp_async_ca_16B(dst_x, X + static_cast<int64_t>(tok) * 2304 + k_base + (c << 3));
                } else {
                    s_X[st][m_row][sc] = make_uint4(0, 0, 0, 0);
                }
            }
            if constexpr (IS_NVFP4) {
                int kk_ld = tid >> 5;
                int lane_ld = tid & 31;
                uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(&s_QB[st][kk_ld][lane_ld]));
                int64_t w_off = ((static_cast<int64_t>(expert) * 144 + (ki * 4 + kk_ld)) * 36 + nb) * 32 + lane_ld;
                cp_async_cg_16B(dst_w, W_u4 + w_off);

                if (tid < 16) {
                    int kk_bs = tid >> 2;
                    int c_bs = tid & 3;
                    uint32_t dst_bs = static_cast<uint32_t>(__cvta_generic_to_shared(&s_BS[st][kk_bs][c_bs]));
                    int64_t bs_off = ((static_cast<int64_t>(expert) * 144 + (ki * 4 + kk_bs)) * 36 + nb) * 4 + c_bs;
                    cp_async_ca_16B(dst_bs, BS_u4 + bs_off);
                }
            } else {
                #pragma unroll
                for (int u_ld = 0; u_ld < 2; ++u_ld) {
                    int idx = tid + (u_ld << 7);
                    int kk_ld = idx >> 6;
                    int col_u4 = idx & 63;
                    uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(&s_QB[st][kk_ld][col_u4]));
                    int64_t w_off = ((static_cast<int64_t>(expert) * 144 + (ki * 4 + kk_ld)) * 36 + nb) * 64 + col_u4;
                    cp_async_cg_16B(dst_w, W_u4 + w_off);
                }
                if (tid < 8) {
                    int g_bs = tid >> 2;
                    int c_bs = tid & 3;
                    uint32_t dst_bs = static_cast<uint32_t>(__cvta_generic_to_shared(&s_BS[st][g_bs][c_bs]));
                    int64_t bs_off = ((static_cast<int64_t>(expert) * 72 + (ki * 2 + g_bs)) * 36 + nb) * 4 + c_bs;
                    cp_async_ca_16B(dst_bs, BS_u4 + bs_off);
                }
            }
            cp_async_commit();
        };

        #pragma unroll
        for (int p = 0; p < STAGES - 1; ++p) {
            issue_stage(p, p);
        }

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (int step = 0; step < NUM_K_ITERS; ++step) {
            int next_p = step + (STAGES - 1);
            if (next_p < NUM_K_ITERS) {
                issue_stage(next_p % STAGES, next_p);
                cp_async_wait<STAGES - 1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();

            const int st = step % STAGES;
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t res[4];
                if constexpr (IS_NVFP4) {
                    uint32_t qb_word = reinterpret_cast<const uint32_t*>(&s_QB[st][kk][lane])[warp_id];
                    uint16_t spair = reinterpret_cast<const uint16_t*>(s_BS[st][kk])[group_id * 4 + warp_id];
                    dequant_nvfp4_mma16x16_sm80(qb_word, spair, res);
                } else {
                    uint2 qb2 = reinterpret_cast<const uint2*>(s_QB[st][kk])[lane * 4 + warp_id];
                    uint16_t spair = reinterpret_cast<const uint16_t*>(s_BS[st][kk >> 1])[group_id * 4 + warp_id];
                    dequant_mxfp8_mma16x16_sm80(qb2, spair, res);
                }

                int c_lo = (kk << 1);
                int c_hi = c_lo + 1;
                const uint32_t* x_lo = reinterpret_cast<const uint32_t*>(&s_X[st][group_id][c_lo ^ group_id]);
                const uint32_t* x_hi = reinterpret_cast<const uint32_t*>(&s_X[st][group_id][c_hi ^ group_id]);

                mma_m16n8k16_bf16(
                    acc0, acc1, acc2, acc3,
                    res[0], res[2], res[1], res[3],
                    x_lo[tid_in_grp], x_hi[tid_in_grp]
                );
            }
            __syncthreads();
        }

        // Fused bit-exact SwiGLU epilogue (reusing s_X[0] = 1024B as [8][32] bf16 staging!)
        float v0 = bf2f(f2bf(acc0 * gs_f32));
        float v1 = bf2f(f2bf(acc1 * gs_f32));
        float v2 = bf2f(f2bf(acc2 * gs_f32));
        float v3 = bf2f(f2bf(acc3 * gs_f32));

        float p0 = __shfl_xor_sync(0xffffffffu, v0, 4);
        float p1 = __shfl_xor_sync(0xffffffffu, v1, 4);
        float p2 = __shfl_xor_sync(0xffffffffu, v2, 4);
        float p3 = __shfl_xor_sync(0xffffffffu, v3, 4);

        __nv_bfloat16 (*s_epi)[32] = reinterpret_cast<__nv_bfloat16 (*)[32]>(&s_X[0][0][0]);
        const int m0 = (tid_in_grp << 1);
        const int m1 = m0 + 1;
        if ((lane & 4) == 0) {
            int pair_idx = group_id >> 1; // 0..3
            int c0 = warp_id * 8 + pair_idx;
            int c1 = c0 + 4;
            float gm0 = s_gamma_m[m0];
            float gm1 = s_gamma_m[m1];
            s_epi[m0][c0] = swiglu_bf16_exact(v0, p0, gm0);
            s_epi[m1][c0] = swiglu_bf16_exact(v1, p1, gm1);
            s_epi[m0][c1] = swiglu_bf16_exact(v2, p2, gm0);
            s_epi[m1][c1] = swiglu_bf16_exact(v3, p3, gm1);
        }
        __syncthreads();

        if (tid < 32) {
            int m_row = tid >> 2;
            int vec = tid & 3;
            int sl = s_slot_m[m_row];
            if (sl >= 0) {
                reinterpret_cast<uint4*>(OutAct + static_cast<int64_t>(sl) * 1152 + nb * 32)[vec] =
                    reinterpret_cast<const uint4*>(s_epi[m_row])[vec];
            }
        }
        __syncthreads();
    }
}

template <bool IS_NVFP4, int CTA_PER_SM>
__global__ void __launch_bounds__(128, CTA_PER_SM) sm80_moe_w2_kernel(
    const __nv_bfloat16* __restrict__ InAct,      // [T * 8, 1152] bf16
    const uint4* __restrict__ W_u4,               // Humming tensors["w2"]["weight"]
    const uint4* __restrict__ BS_u4,              // Humming tensors["w2"]["weight_scale"]
    const float* __restrict__ BS2,                // [128] fp32 or nullptr
    const int32_t* __restrict__ num_active_ptr,   // [1] int32
    const int32_t* __restrict__ active_experts,   // [64] int32
    const int32_t* __restrict__ active_slots,     // [64 * 8] int32
    __nv_bfloat16* __restrict__ OutDown)          // [T * 8, 2304] bf16
{
    const int U = num_active_ptr[0];
    if (U <= 0) return;

    constexpr int STAGES = 4;
    constexpr int NUM_K_ITERS = 18; // 1152 / 64 = 18
    constexpr int QB_U4_PER_SLAB = IS_NVFP4 ? 32 : 64;
    constexpr int BS_GROUPS_PER_STAGE = IS_NVFP4 ? 4 : 2;

    __shared__ __align__(16) uint4 s_QB[STAGES][4][QB_U4_PER_SLAB];
    __shared__ __align__(16) uint4 s_BS[STAGES][BS_GROUPS_PER_STAGE][4];
    __shared__ __align__(16) uint4 s_X[STAGES][8][8];
    __shared__ int32_t s_expert_id;
    __shared__ int32_t s_slot_m[8];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int tid_in_grp = lane & 3;
    const int total_tiles = U * 36;

    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        const int u = tile_idx / 36;
        const int nb = tile_idx - u * 36;

        if (tid == 0) {
            s_expert_id = active_experts[u];
        }
        if (tid < 8) {
            s_slot_m[tid] = active_slots[u * 8 + tid];
        }
        __syncthreads();

        const int expert = s_expert_id;
        const float gs_f32 = IS_NVFP4 ? (BS2[expert] * 128.0f) : 1.0f;

        auto issue_stage = [&](int st, int ki) {
            const int k_base = ki * 64;
            if (tid < 64) {
                int m_row = tid >> 3;
                int c = tid & 7;
                int sc = c ^ m_row;
                int sl = s_slot_m[m_row];
                uint32_t dst_x = static_cast<uint32_t>(__cvta_generic_to_shared(&s_X[st][m_row][sc]));
                if (sl >= 0) {
                    cp_async_ca_16B(dst_x, InAct + static_cast<int64_t>(sl) * 1152 + k_base + (c << 3));
                } else {
                    s_X[st][m_row][sc] = make_uint4(0, 0, 0, 0);
                }
            }
            if constexpr (IS_NVFP4) {
                int kk_ld = tid >> 5;
                int lane_ld = tid & 31;
                uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(&s_QB[st][kk_ld][lane_ld]));
                int64_t w_off = ((static_cast<int64_t>(expert) * 72 + (ki * 4 + kk_ld)) * 36 + nb) * 32 + lane_ld;
                cp_async_cg_16B(dst_w, W_u4 + w_off);

                if (tid < 16) {
                    int kk_bs = tid >> 2;
                    int c_bs = tid & 3;
                    uint32_t dst_bs = static_cast<uint32_t>(__cvta_generic_to_shared(&s_BS[st][kk_bs][c_bs]));
                    int64_t bs_off = ((static_cast<int64_t>(expert) * 72 + (ki * 4 + kk_bs)) * 36 + nb) * 4 + c_bs;
                    cp_async_ca_16B(dst_bs, BS_u4 + bs_off);
                }
            } else {
                #pragma unroll
                for (int u_ld = 0; u_ld < 2; ++u_ld) {
                    int idx = tid + (u_ld << 7);
                    int kk_ld = idx >> 6;
                    int col_u4 = idx & 63;
                    uint32_t dst_w = static_cast<uint32_t>(__cvta_generic_to_shared(&s_QB[st][kk_ld][col_u4]));
                    int64_t w_off = ((static_cast<int64_t>(expert) * 72 + (ki * 4 + kk_ld)) * 36 + nb) * 64 + col_u4;
                    cp_async_cg_16B(dst_w, W_u4 + w_off);
                }
                if (tid < 8) {
                    int g_bs = tid >> 2;
                    int c_bs = tid & 3;
                    uint32_t dst_bs = static_cast<uint32_t>(__cvta_generic_to_shared(&s_BS[st][g_bs][c_bs]));
                    int64_t bs_off = ((static_cast<int64_t>(expert) * 36 + (ki * 2 + g_bs)) * 36 + nb) * 4 + c_bs;
                    cp_async_ca_16B(dst_bs, BS_u4 + bs_off);
                }
            }
            cp_async_commit();
        };

        #pragma unroll
        for (int p = 0; p < STAGES - 1; ++p) {
            issue_stage(p, p);
        }

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (int step = 0; step < NUM_K_ITERS; ++step) {
            int next_p = step + (STAGES - 1);
            if (next_p < NUM_K_ITERS) {
                issue_stage(next_p % STAGES, next_p);
                cp_async_wait<STAGES - 1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();

            const int st = step % STAGES;
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t res[4];
                if constexpr (IS_NVFP4) {
                    uint32_t qb_word = reinterpret_cast<const uint32_t*>(&s_QB[st][kk][lane])[warp_id];
                    uint16_t spair = reinterpret_cast<const uint16_t*>(s_BS[st][kk])[group_id * 4 + warp_id];
                    dequant_nvfp4_mma16x16_sm80(qb_word, spair, res);
                } else {
                    uint2 qb2 = reinterpret_cast<const uint2*>(s_QB[st][kk])[lane * 4 + warp_id];
                    uint16_t spair = reinterpret_cast<const uint16_t*>(s_BS[st][kk >> 1])[group_id * 4 + warp_id];
                    dequant_mxfp8_mma16x16_sm80(qb2, spair, res);
                }

                int c_lo = (kk << 1);
                int c_hi = c_lo + 1;
                const uint32_t* x_lo = reinterpret_cast<const uint32_t*>(&s_X[st][group_id][c_lo ^ group_id]);
                const uint32_t* x_hi = reinterpret_cast<const uint32_t*>(&s_X[st][group_id][c_hi ^ group_id]);

                mma_m16n8k16_bf16(
                    acc0, acc1, acc2, acc3,
                    res[0], res[2], res[1], res[3],
                    x_lo[tid_in_grp], x_hi[tid_in_grp]
                );
            }
            __syncthreads();
        }

        // Coalesced 16B uint4 epilogue (reusing s_X[0] = 1024B as [8][64] bf16 staging!)
        __nv_bfloat16 (*s_epi)[64] = reinterpret_cast<__nv_bfloat16 (*)[64]>(&s_X[0][0][0]);
        const int m0 = (tid_in_grp << 1);
        const int m1 = m0 + 1;
        const int col0 = (warp_id << 4) + group_id;
        const int col1 = col0 + 8;

        s_epi[m0][col0] = f2bf(acc0 * gs_f32);
        s_epi[m1][col0] = f2bf(acc1 * gs_f32);
        s_epi[m0][col1] = f2bf(acc2 * gs_f32);
        s_epi[m1][col1] = f2bf(acc3 * gs_f32);
        __syncthreads();

        if (tid < 64) {
            int m_row = tid >> 3;
            int vec = tid & 7;
            int sl = s_slot_m[m_row];
            if (sl >= 0) {
                reinterpret_cast<uint4*>(OutDown + static_cast<int64_t>(sl) * 2304 + nb * 64)[vec] =
                    reinterpret_cast<const uint4*>(s_epi[m_row])[vec];
            }
        }
        __syncthreads();
    }
}

extern "C" void launch_sm80_moe_w13_swiglu(
    const void* X,
    const void* W_u4,
    const void* BS_u4,
    const void* BS2,
    const void* num_active_ptr,
    const void* active_experts,
    const void* active_slots,
    const void* route_weights,
    void* OutAct,
    int32_t is_nvfp4,
    int32_t num_ctas,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    if (is_nvfp4) {
        if (num_ctas <= 222) {
            sm80_moe_w13_swiglu_kernel<true, 3><<<num_ctas, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const uint4*>(W_u4),
                static_cast<const uint4*>(BS_u4),
                static_cast<const float*>(BS2),
                static_cast<const int32_t*>(num_active_ptr),
                static_cast<const int32_t*>(active_experts),
                static_cast<const int32_t*>(active_slots),
                static_cast<const float*>(route_weights),
                static_cast<__nv_bfloat16*>(OutAct));
        } else {
            sm80_moe_w13_swiglu_kernel<true, 4><<<num_ctas, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const uint4*>(W_u4),
                static_cast<const uint4*>(BS_u4),
                static_cast<const float*>(BS2),
                static_cast<const int32_t*>(num_active_ptr),
                static_cast<const int32_t*>(active_experts),
                static_cast<const int32_t*>(active_slots),
                static_cast<const float*>(route_weights),
                static_cast<__nv_bfloat16*>(OutAct));
        }
    } else {
        if (num_ctas <= 222) {
            sm80_moe_w13_swiglu_kernel<false, 3><<<num_ctas, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const uint4*>(W_u4),
                static_cast<const uint4*>(BS_u4),
                nullptr,
                static_cast<const int32_t*>(num_active_ptr),
                static_cast<const int32_t*>(active_experts),
                static_cast<const int32_t*>(active_slots),
                static_cast<const float*>(route_weights),
                static_cast<__nv_bfloat16*>(OutAct));
        } else {
            sm80_moe_w13_swiglu_kernel<false, 4><<<num_ctas, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(X),
                static_cast<const uint4*>(W_u4),
                static_cast<const uint4*>(BS_u4),
                nullptr,
                static_cast<const int32_t*>(num_active_ptr),
                static_cast<const int32_t*>(active_experts),
                static_cast<const int32_t*>(active_slots),
                static_cast<const float*>(route_weights),
                static_cast<__nv_bfloat16*>(OutAct));
        }
    }
}

extern "C" void launch_sm80_moe_w2(
    const void* InAct,
    const void* W_u4,
    const void* BS_u4,
    const void* BS2,
    const void* num_active_ptr,
    const void* active_experts,
    const void* active_slots,
    void* OutDown,
    int32_t is_nvfp4,
    int32_t num_ctas,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    if (is_nvfp4) {
        if (num_ctas <= 222) {
            sm80_moe_w2_kernel<true, 3><<<num_ctas, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(InAct),
                static_cast<const uint4*>(W_u4),
                static_cast<const uint4*>(BS_u4),
                static_cast<const float*>(BS2),
                static_cast<const int32_t*>(num_active_ptr),
                static_cast<const int32_t*>(active_experts),
                static_cast<const int32_t*>(active_slots),
                static_cast<__nv_bfloat16*>(OutDown));
        } else {
            sm80_moe_w2_kernel<true, 4><<<num_ctas, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(InAct),
                static_cast<const uint4*>(W_u4),
                static_cast<const uint4*>(BS_u4),
                static_cast<const float*>(BS2),
                static_cast<const int32_t*>(num_active_ptr),
                static_cast<const int32_t*>(active_experts),
                static_cast<const int32_t*>(active_slots),
                static_cast<__nv_bfloat16*>(OutDown));
        }
    } else {
        if (num_ctas <= 222) {
            sm80_moe_w2_kernel<false, 3><<<num_ctas, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(InAct),
                static_cast<const uint4*>(W_u4),
                static_cast<const uint4*>(BS_u4),
                nullptr,
                static_cast<const int32_t*>(num_active_ptr),
                static_cast<const int32_t*>(active_experts),
                static_cast<const int32_t*>(active_slots),
                static_cast<__nv_bfloat16*>(OutDown));
        } else {
            sm80_moe_w2_kernel<false, 4><<<num_ctas, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(InAct),
                static_cast<const uint4*>(W_u4),
                static_cast<const uint4*>(BS_u4),
                nullptr,
                static_cast<const int32_t*>(num_active_ptr),
                static_cast<const int32_t*>(active_experts),
                static_cast<const int32_t*>(active_slots),
                static_cast<__nv_bfloat16*>(OutDown));
        }
    }
}

// ============================================================================
// 3. Zero-Host-Sync GPU-Resident Hybrid SuffixDecoding + DSpark Point-Mass &
//    Markov Early-Exit Kernels (100% CUDA-Graph Safe, Race-Free Cache Hygiene)
// ============================================================================
__global__ void __launch_bounds__(256) markov_rnn_gate_256_suf_sm80_kernel(
    const uint4* __restrict__ state_in,
    const void* __restrict__ prev_tokens,
    const uint4* __restrict__ markov_w1,
    const float* __restrict__ context,
    const uint4* __restrict__ rnn_w_step,
    const int32_t* __restrict__ skip_flag_ptr,
    __nv_bfloat16* __restrict__ state_out,
    __nv_bfloat16* __restrict__ latent_out,
    int32_t prev_is_i64)
{
    if (skip_flag_ptr != nullptr && skip_flag_ptr[0] != 0) return;

    const int req = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 8 + warp;

    int64_t tok = prev_is_i64
        ? static_cast<const int64_t*>(prev_tokens)[req]
        : static_cast<int64_t>(static_cast<const int32_t*>(prev_tokens)[req]);

    float in0[8], in1[8];
    unpack_bf16x8(state_in[req * 32 + lane], in0);
    unpack_bf16x8(markov_w1[tok * 32 + lane], in1);

    float raw3[3];
    #pragma unroll
    for (int part = 0; part < 3; ++part) {
        int row = part * 256 + c;
        const uint4* wr = rnn_w_step + row * 64;
        float w0[8], w1[8];
        unpack_bf16x8(wr[lane], w0);
        unpack_bf16x8(wr[32 + lane], w1);
        float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) s = __fmaf_rn(in0[j], w0[j], s);
        #pragma unroll
        for (int j = 0; j < 8; ++j) s = __fmaf_rn(in1[j], w1[j], s);
        raw3[part] = warp_sum_rn(s);
    }

    if (lane == 0) {
        float g_raw = round_bf16_f32(__fadd_rn(context[req * 768 + c], raw3[0]));
        float c_raw = round_bf16_f32(__fadd_rn(context[req * 768 + 256 + c], raw3[1]));
        float o_raw = round_bf16_f32(__fadd_rn(context[req * 768 + 512 + c], raw3[2]));
        float st = bf2f(reinterpret_cast<const __nv_bfloat16*>(state_in)[req * 256 + c]);

        float gate = round_bf16_f32(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g_raw))));
        float cand = round_bf16_f32(tanhf(c_raw));
        float outg = round_bf16_f32(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-o_raw))));

        float new_st = round_bf16_f32(__fadd_rn(
            round_bf16_f32(__fmul_rn(__fsub_rn(1.0f, gate), st)),
            round_bf16_f32(__fmul_rn(gate, cand))
        ));
        float lat = round_bf16_f32(__fmul_rn(outg, round_bf16_f32(tanhf(new_st))));
        state_out[req * 256 + c] = f2bf(new_st);
        latent_out[req * 256 + c] = f2bf(lat);
    }
}

extern "C" void launch_markov_rnn_gate_256_suf_sm80(
    const void* state_in, const void* prev_tokens, const void* markov_w1,
    const void* context, const void* rnn_w_step, const void* skip_flag_ptr,
    void* state_out, void* latent_out,
    int32_t num_reqs, int32_t prev_is_i64, void* stream_ptr)
{
    if (num_reqs <= 0) return;
    dim3 grid(32, num_reqs);
    markov_rnn_gate_256_suf_sm80_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const uint4*>(state_in), prev_tokens, static_cast<const uint4*>(markov_w1),
        static_cast<const float*>(context), static_cast<const uint4*>(rnn_w_step),
        static_cast<const int32_t*>(skip_flag_ptr),
        static_cast<__nv_bfloat16*>(state_out), static_cast<__nv_bfloat16*>(latent_out), prev_is_i64);
}

__global__ void __launch_bounds__(256) markov_fold_gumbel_cache_r1_suf_sm80_kernel(
    const uint4* __restrict__ latent,
    const uint4* __restrict__ fold_w,
    const uint4* __restrict__ fold_b,
    const uint4* __restrict__ base_logits,
    const int64_t* __restrict__ keep_ids,
    const int64_t* __restrict__ idx_map,
    const float* __restrict__ temp_ptr,
    const int64_t* __restrict__ seeds_ptr,
    const int64_t* __restrict__ sample_pos_ptr,
    const int32_t* __restrict__ row_mapping_ptr,
    const int32_t* __restrict__ skip_flag_ptr,
    int32_t* __restrict__ keep_dirty_ptr,
    __nv_bfloat16* __restrict__ logits_cache,
    float* __restrict__ ws_max,
    int64_t* __restrict__ ws_id,
    int32_t* __restrict__ cta_counter,
    int64_t* __restrict__ out_sampled,
    int64_t* __restrict__ out_draft_col,
    int64_t cache_stride_0,
    int64_t cache_stride_1,
    int32_t step_col,
    int32_t K_vocab,
    float softcap)
{
    if (skip_flag_ptr != nullptr && skip_flag_ptr[0] != 0) return;

    __shared__ float s_wmax[8];
    __shared__ int64_t s_wid[8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float lat[8];
    unpack_bf16x8(latent[lane], lat);

    const int64_t req_state_idx = idx_map[0];
    const bool valid_req = (req_state_idx >= 0);
    const float temp = valid_req ? temp_ptr[req_state_idx] : 0.0f;
    int64_t cache_row = req_state_idx;
    if (row_mapping_ptr != nullptr) {
        cache_row = valid_req ? static_cast<int64_t>(row_mapping_ptr[req_state_idx]) : -1;
    }
    const int64_t raw_seed = valid_req ? seeds_ptr[req_state_idx] : 0LL;
    const int64_t seed = raw_seed ^ (static_cast<int64_t>(step_col) << 32);
    const int64_t pos_salted = (sample_pos_ptr[0] - 1LL) + (1LL << 30);
    const float inv_sc = 1.0f / softcap;

    float best_val = -INFINITY;
    int64_t best_id = (1LL << 60);

    const int warp_base = blockIdx.x * 256 + warp * 32;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        int row0 = warp_base + g * 8;
        if (row0 < K_vocab) {
            float dots[8];
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                float w[8];
                unpack_bf16x8(fold_w[(row0 + r) * 32 + lane], w);
                float s = 0.f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) s = __fmaf_rn(lat[j], w[j], s);
                dots[r] = warp_sum_rn(s);
            }
            float my_dot = 0.f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                float d_r = __shfl_sync(0xffffffffu, dots[r], 0);
                if (lane == r) my_dot = d_r;
            }
            if (lane < 8) {
                int col = row0 + lane;
                const __nv_bfloat16* fb_ptr = reinterpret_cast<const __nv_bfloat16*>(fold_b);
                const __nv_bfloat16* bl_ptr = reinterpret_cast<const __nv_bfloat16*>(base_logits);
                float fb_v = bf2f(fb_ptr[col]);
                float bl_v = bf2f(bl_ptr[col]);
                float bias_bf = bf2f(f2bf(__fadd_rn(fb_v, my_dot)));
                float x_bf = bf2f(f2bf(__fadd_rn(bl_v, bias_bf)));
                float out_f = __fmul_rn(softcap, tanhf(__fmul_rn(x_bf, inv_sc)));
                __nv_bfloat16 logit_bf = f2bf(out_f);
                float logit_f32 = bf2f(logit_bf);
                int64_t target_id = keep_ids[col];
                if (cache_row >= 0) {
                    logits_cache[cache_row * cache_stride_0 + static_cast<int64_t>(step_col) * cache_stride_1 + target_id] = logit_bf;
                }
                float score = logit_f32;
                if (temp != 0.0f) {
                    score = (score / temp) + murmur3_gumbel_noise_cu(seed, pos_salted, static_cast<uint32_t>(target_id));
                }
                if (score > best_val || (score == best_val && target_id < best_id)) {
                    best_val = score;
                    best_id = target_id;
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffu, best_val, offset);
        int64_t oid = __shfl_down_sync(0xffu, best_id, offset);
        if (ov > best_val || (ov == best_val && oid < best_id)) {
            best_val = ov;
            best_id = oid;
        }
    }
    if (lane == 0) {
        s_wmax[warp] = best_val;
        s_wid[warp] = best_id;
    }
    __syncthreads();

    if (tid < 8) {
        float v = s_wmax[tid];
        int64_t id = s_wid[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffu, v, offset);
            int64_t oid = __shfl_down_sync(0xffu, id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
        }
        if (tid == 0) {
            ws_max[blockIdx.x] = v;
            ws_id[blockIdx.x] = id;
        }
    }

    __shared__ int s_is_last;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        int prev = atomicAdd(cta_counter, 1);
        int target = gridDim.x - 1;
        s_is_last = (prev == target) ? 1 : 0;
        if (prev == target) {
            atomicExch(cta_counter, 0);
        }
    }
    __syncthreads();
    if (!s_is_last) return;

    float cv = (tid < gridDim.x) ? ws_max[tid] : -INFINITY;
    int64_t cid_tok = (tid < gridDim.x) ? ws_id[tid] : (1LL << 60);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, cv, offset);
        int64_t oid = __shfl_down_sync(0xffffffffu, cid_tok, offset);
        if (ov > cv || (ov == cv && oid < cid_tok)) {
            cv = ov;
            cid_tok = oid;
        }
    }
    if (lane == 0) {
        s_wmax[warp] = cv;
        s_wid[warp] = cid_tok;
    }
    __syncthreads();

    if (tid < 8) {
        float v = s_wmax[tid];
        int64_t id = s_wid[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(0xffu, v, offset);
            int64_t oid = __shfl_down_sync(0xffu, id, offset);
            if (ov > v || (ov == v && oid < id)) {
                v = ov;
                id = oid;
            }
        }
        if (tid == 0) {
            out_sampled[0] = id;
            out_draft_col[0] = id;
            if (keep_dirty_ptr != nullptr && cache_row >= 0) {
                keep_dirty_ptr[cache_row] = 1;
            }
        }
    }
}

extern "C" void launch_markov_fold_gumbel_cache_r1_suf_sm80(
    const void* latent,
    const void* fold_w,
    const void* fold_bias,
    const void* base_logits,
    const void* keep_ids,
    const void* idx_map,
    const void* temp_ptr,
    const void* seeds_ptr,
    const void* sample_pos_ptr,
    const void* row_mapping_ptr,
    const void* skip_flag_ptr,
    void* keep_dirty_ptr,
    void* logits_cache,
    void* ws_max,
    void* ws_id,
    void* cta_counter,
    void* out_sampled,
    void* out_draft_col,
    int64_t cache_stride_0,
    int64_t cache_stride_1,
    int32_t step_col,
    int32_t K_vocab,
    float softcap,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    int32_t n_ctas = (K_vocab + 255) >> 8;
    markov_fold_gumbel_cache_r1_suf_sm80_kernel<<<n_ctas, 256, 0, stream>>>(
        static_cast<const uint4*>(latent),
        static_cast<const uint4*>(fold_w),
        static_cast<const uint4*>(fold_bias),
        static_cast<const uint4*>(base_logits),
        static_cast<const int64_t*>(keep_ids),
        static_cast<const int64_t*>(idx_map),
        static_cast<const float*>(temp_ptr),
        static_cast<const int64_t*>(seeds_ptr),
        static_cast<const int64_t*>(sample_pos_ptr),
        static_cast<const int32_t*>(row_mapping_ptr),
        static_cast<const int32_t*>(skip_flag_ptr),
        static_cast<int32_t*>(keep_dirty_ptr),
        static_cast<__nv_bfloat16*>(logits_cache),
        static_cast<float*>(ws_max),
        static_cast<int64_t*>(ws_id),
        static_cast<int32_t*>(cta_counter),
        static_cast<int64_t*>(out_sampled),
        static_cast<int64_t*>(out_draft_col),
        cache_stride_0, cache_stride_1, step_col, K_vocab, softcap
    );
}

__global__ void __launch_bounds__(256) suffix_cleanup_prev_sm80_kernel(
    __nv_bfloat16* __restrict__ logits_cache,
    const int32_t* __restrict__ idx_mapping,      // [B] int32
    const int32_t* __restrict__ row_mapping,      // [max_reqs] int32 or nullptr
    int64_t* __restrict__ prev_suffix_tok,        // [max_rows, max_spec] int64
    int32_t* __restrict__ keep_dirty,             // [max_rows] int32 or nullptr
    int32_t* __restrict__ just_scrubbed_all,      // [max_rows] int32 or nullptr
    int64_t cache_stride_0,
    int64_t cache_stride_1,
    int32_t num_reqs,
    int32_t max_spec)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_reqs * max_spec) return;
    const int b = idx / max_spec;
    const int s = idx - b * max_spec;
    const int req = idx_mapping[b];
    if (req < 0) return;
    const int row = (row_mapping != nullptr) ? row_mapping[req] : req;
    if (row < 0) return;
    const int64_t slot = static_cast<int64_t>(row) * max_spec + s;
    const int64_t old_tok = prev_suffix_tok[slot];
    if (old_tok >= 0) {
        const uint16_t neg_inf_bf16 = 0xFF80u;
        logits_cache[row * cache_stride_0 + s * cache_stride_1 + old_tok] =
            *reinterpret_cast<const __nv_bfloat16*>(&neg_inf_bf16);
        prev_suffix_tok[slot] = -1LL;
    }
    if (s == 0 && keep_dirty != nullptr && just_scrubbed_all != nullptr) {
        if (just_scrubbed_all[row] != 0) {
            keep_dirty[row] = 0;
            just_scrubbed_all[row] = 0;
        }
    }
}

__global__ void __launch_bounds__(256) suffix_apply_pointmass_sm80_kernel(
    __nv_bfloat16* __restrict__ logits_cache,
    const int64_t* __restrict__ keep_ids,         // [K] int64
    const int32_t* __restrict__ idx_mapping,      // [B] int32
    const int32_t* __restrict__ row_mapping,      // [max_reqs] int32 or nullptr
    const int64_t* __restrict__ suffix_drafts,    // [B, max_spec] int64
    const int32_t* __restrict__ suffix_num_valid, // [B] int32
    const int64_t* __restrict__ suffix_match_len, // [B] int64
    int64_t* __restrict__ prev_suffix_tok,        // [max_rows, max_spec] int64
    const int32_t* __restrict__ keep_dirty,       // [max_rows] int32 read-only!
    int32_t* __restrict__ just_scrubbed_all,      // [max_rows] int32
    int64_t* __restrict__ out_draft_tokens,       // [max_reqs, max_spec] int64
    int64_t cache_stride_0,
    int64_t cache_stride_1,
    int32_t K_vocab,
    int32_t n_spec,
    int32_t max_spec,
    int32_t min_match_len)
{
    const int b = blockIdx.y / n_spec;
    const int s = blockIdx.y - b * n_spec;
    const int n_valid = suffix_num_valid[b];
    const int64_t m_len = suffix_match_len[b];
    // Partial Suffix + DSpark Grafting: apply SuffixDecoding on positions 0..min(n_valid,n_spec)-1!
    if (s >= n_valid || m_len < static_cast<int64_t>(min_match_len)) return;

    const int req = idx_mapping[b];
    if (req < 0) return;
    const int row = (row_mapping != nullptr) ? row_mapping[req] : req;
    const int64_t s_tok = suffix_drafts[static_cast<int64_t>(b) * max_spec + s];
    const uint16_t neg_inf_bf16 = 0xFF80u;
    const uint16_t zero_bf16 = 0x0000u;
    const __nv_bfloat16 v_ninf = *reinterpret_cast<const __nv_bfloat16*>(&neg_inf_bf16);
    const __nv_bfloat16 v_zero = *reinterpret_cast<const __nv_bfloat16*>(&zero_bf16);

    if (row >= 0 && logits_cache != nullptr) {
        __nv_bfloat16* row_ptr = logits_cache + row * cache_stride_0 + s * cache_stride_1;
        const int32_t is_dirty = (keep_dirty != nullptr) ? keep_dirty[row] : 1;
        if (is_dirty != 0) {
            for (int col = blockIdx.x * 256 + threadIdx.x; col < K_vocab; col += gridDim.x * 256) {
                int64_t tid_v = keep_ids[col];
                if (tid_v != s_tok) {
                    row_ptr[tid_v] = v_ninf;
                }
            }
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            const int64_t slot = static_cast<int64_t>(row) * max_spec + s;
            const int64_t old_tok = prev_suffix_tok[slot];
            if (old_tok >= 0 && old_tok != s_tok) {
                row_ptr[old_tok] = v_ninf;
            }
            row_ptr[s_tok] = v_zero;
            prev_suffix_tok[slot] = s_tok;
            out_draft_tokens[static_cast<int64_t>(b) * max_spec + s] = s_tok;
            if (s == 0 && n_valid >= max_spec && n_spec >= max_spec && just_scrubbed_all != nullptr) {
                just_scrubbed_all[row] = 1;
            }
        }
    } else if (blockIdx.x == 0 && threadIdx.x == 0) {
        out_draft_tokens[static_cast<int64_t>(b) * max_spec + s] = s_tok;
    }
}

extern "C" void launch_suffix_cleanup_prev_sm80(
    void* logits_cache,
    const void* idx_mapping,
    const void* row_mapping,
    void* prev_suffix_tok,
    void* keep_dirty,
    void* just_scrubbed_all,
    int64_t cache_stride_0,
    int64_t cache_stride_1,
    int32_t num_reqs,
    int32_t max_spec,
    void* stream_ptr)
{
    if (num_reqs <= 0 || max_spec <= 0 || logits_cache == nullptr) return;
    int blocks = (num_reqs * max_spec + 255) / 256;
    suffix_cleanup_prev_sm80_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<__nv_bfloat16*>(logits_cache),
        static_cast<const int32_t*>(idx_mapping),
        static_cast<const int32_t*>(row_mapping),
        static_cast<int64_t*>(prev_suffix_tok),
        static_cast<int32_t*>(keep_dirty),
        static_cast<int32_t*>(just_scrubbed_all),
        cache_stride_0, cache_stride_1, num_reqs, max_spec);
}

extern "C" void launch_suffix_apply_pointmass_sm80(
    void* logits_cache,
    const void* keep_ids,
    const void* idx_mapping,
    const void* row_mapping,
    const void* suffix_drafts,
    const void* suffix_num_valid,
    const void* suffix_match_len,
    void* prev_suffix_tok,
    const void* keep_dirty,
    void* just_scrubbed_all,
    void* out_draft_tokens,
    int64_t cache_stride_0,
    int64_t cache_stride_1,
    int32_t num_reqs,
    int32_t K_vocab,
    int32_t n_spec,
    int32_t max_spec,
    int32_t min_match_len,
    void* stream_ptr)
{
    if (num_reqs <= 0 || n_spec <= 0) return;
    int gx = (logits_cache != nullptr) ? ((K_vocab + 255) / 256) : 1;
    dim3 grid(gx, num_reqs * n_spec);
    suffix_apply_pointmass_sm80_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<__nv_bfloat16*>(logits_cache),
        static_cast<const int64_t*>(keep_ids),
        static_cast<const int32_t*>(idx_mapping),
        static_cast<const int32_t*>(row_mapping),
        static_cast<const int64_t*>(suffix_drafts),
        static_cast<const int32_t*>(suffix_num_valid),
        static_cast<const int64_t*>(suffix_match_len),
        static_cast<int64_t*>(prev_suffix_tok),
        static_cast<const int32_t*>(keep_dirty),
        static_cast<int32_t*>(just_scrubbed_all),
        static_cast<int64_t*>(out_draft_tokens),
        cache_stride_0, cache_stride_1, K_vocab, n_spec, max_spec, min_match_len);
}

// ============================================================================
// 4. Zero-Barrier Loop Controller Down-Projection (r = 64, 2D = 4608)
//    Grid: (64, S), Block: 128 threads (4 warps).
//    Loads down[s, r, 0..4607] ONCE per CTA and computes exact 1024-group dot
//    for all T <= 16 tokens in registers (0 split-K workspace, 0 debug_barrier!).
// ============================================================================
__global__ void __launch_bounds__(128, 4) sm80_loop_down_64_kernel(
    const __nv_bfloat16* __restrict__ pair,       // [S, T, 4608] bf16
    const __nv_bfloat16* __restrict__ down,       // [S, 64, 4608] bf16 (stride_down_s)
    __nv_bfloat16* __restrict__ out_hidden,       // [S, T, 64] bf16
    int64_t stride_down_s,
    int32_t T)
{
    const int r = blockIdx.x;       // 0..63
    const int s = blockIdx.y;       // 0..S-1
    const int tid = threadIdx.x;    // 0..127 (576 uint4s across 128 threads -> 4 or 5 per thread)
    const int warp_id = tid >> 5;   // 0..3
    const int lane = tid & 31;      // 0..31

    // Match linear_k_groups (1024-element groups: 0..1023, 1024..2047, 2048..3071, 3072..4095, 4096..4607)
    // Each 1024-bf16 group = 128 uint4s = 1 uint4 per thread! Last 512-bf16 group = 64 uint4s (tid < 64)!
    const uint4* w_row = reinterpret_cast<const uint4*>(down + static_cast<int64_t>(s) * stride_down_s + r * 4608);
    uint4 w_reg[5];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        w_reg[g] = w_row[g * 128 + tid];
    }
    if (tid < 64) {
        w_reg[4] = w_row[4 * 128 + tid];
    } else {
        w_reg[4] = make_uint4(0, 0, 0, 0);
    }

    __shared__ float s_wsum[5][4];

    for (int t = 0; t < T; ++t) {
        const uint4* x_row = reinterpret_cast<const uint4*>(pair + (static_cast<int64_t>(s) * T + t) * 4608);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uint4 xv = x_row[g * 128 + tid];
            uint4 wv = w_reg[g];
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            float acc = 0.0f;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float2 xf = __bfloat1622float2(x2[p]);
                float2 wf = __bfloat1622float2(w2[p]);
                acc += xf.x * wf.x + xf.y * wf.y;
            }
            acc = warp_sum_f32(acc);
            if (lane == 0) s_wsum[g][warp_id] = acc;
        }
        {
            float acc = 0.0f;
            if (tid < 64) {
                uint4 xv = x_row[4 * 128 + tid];
                uint4 wv = w_reg[4];
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
                const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    float2 xf = __bfloat1622float2(x2[p]);
                    float2 wf = __bfloat1622float2(w2[p]);
                    acc += xf.x * wf.x + xf.y * wf.y;
                }
            }
            acc = warp_sum_f32(acc);
            if (lane == 0) s_wsum[4][warp_id] = acc;
        }
        __syncthreads();
        if (tid == 0) {
            float g0 = (s_wsum[0][0] + s_wsum[0][1]) + (s_wsum[0][2] + s_wsum[0][3]);
            float g1 = (s_wsum[1][0] + s_wsum[1][1]) + (s_wsum[1][2] + s_wsum[1][3]);
            float g2 = (s_wsum[2][0] + s_wsum[2][1]) + (s_wsum[2][2] + s_wsum[2][3]);
            float g3 = (s_wsum[3][0] + s_wsum[3][1]) + (s_wsum[3][2] + s_wsum[3][3]);
            float g4 = s_wsum[4][0] + s_wsum[4][1];
            float tot = (((g0 + g1) + g2) + g3) + g4;
            out_hidden[(static_cast<int64_t>(s) * T + t) * 64 + r] = __float2bfloat16_rn(tot);
        }
        __syncthreads();
    }
}

extern "C" void launch_sm80_loop_down_64(
    const void* pair,
    const void* down,
    void* out_hidden,
    int64_t stride_down_s,
    int32_t S,
    int32_t T,
    void* stream_ptr)
{
    if (S <= 0 || T <= 0) return;
    dim3 grid(64, S);
    sm80_loop_down_64_kernel<<<grid, 128, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const __nv_bfloat16*>(pair),
        static_cast<const __nv_bfloat16*>(down),
        static_cast<__nv_bfloat16*>(out_hidden),
        stride_down_s, T);
}

// Multi-CTA Fused Up-Projection + Tanh Blend + RMSNorm (splits D=2304 into 9 tiles of 256
// channels = 32 threads x 8 bf16s so S*T*9 CTAs saturate all 74 SMs even at S=1, T=5!)
__global__ void __launch_bounds__(128, 4) fused_up_blend_9tile_2304_sm80_kernel(
    const float* __restrict__ dest,              // [S, T, 2304] fp32
    const float* __restrict__ message,           // [S, T, 2304] fp32
    const __nv_bfloat16* __restrict__ hidden,    // [S, T, 64] bf16
    const __nv_bfloat16* __restrict__ gf_t,      // [S, 2, 64, 2304] bf16 contiguous (stride_gf_s)
    const __nv_bfloat16* __restrict__ norm_w,    // [S, 2304] bf16
    float* __restrict__ ws_x,                    // [24 * 2304] fp32 scratch
    float* __restrict__ ws_sq,                   // [24 * 9] fp32 scratch
    int32_t* __restrict__ tile_counters,         // [24] int32 zeroed counters
    __nv_bfloat16* __restrict__ out,             // [S, T, 2304] bf16
    int64_t stride_dest_s,
    int64_t stride_gf_s,
    int64_t stride_w_s,
    int32_t T,
    float norm_eps)
{
    // Grid: (9, S * T), Block: 128 threads (4 warps split r=0..63 into 4 chunks of 16, 32 lanes = 32 uint4s = 256 cols!)
    const int c_tile = blockIdx.x;  // 0..8
    const int row_st = blockIdx.y;  // 0..S*T-1
    const int s = row_st / T;
    const int t = row_st - s * T;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;   // 0..3 (each warp sums 16 r-rows: warp_id*16 .. warp_id*16+15)
    const int lane = tid & 31;      // 0..31 (each lane owns 1 uint4 = 8 channels -> 32*8 = 256 channels!)
    const int col_vec = c_tile * 32 + lane; // 0..287 (288 * 8 = 2304!)

    __shared__ float s_h[64];
    __shared__ float s_gain[4][32][8];
    __shared__ float s_forg[4][32][8];

    if (tid < 64) {
        s_h[tid] = __bfloat162float(hidden[row_st * 64 + tid]);
    }
    __syncthreads();

    const __nv_bfloat16* g_base = gf_t + static_cast<int64_t>(s) * stride_gf_s;
    const __nv_bfloat16* f_base = g_base + 64 * 2304;

    float g_acc[8] = {0,0,0,0,0,0,0,0};
    float f_acc[8] = {0,0,0,0,0,0,0,0};
    const int r_start = warp_id * 16;
    #pragma unroll
    for (int dr = 0; dr < 16; ++dr) {
        int r = r_start + dr;
        float hr = s_h[r];
        uint4 gv = reinterpret_cast<const uint4*>(g_base + r * 2304)[col_vec];
        uint4 fv = reinterpret_cast<const uint4*>(f_base + r * 2304)[col_vec];
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
        const __nv_bfloat162* f2 = reinterpret_cast<const __nv_bfloat162*>(&fv);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 gf = __bfloat1622float2(g2[p]);
            float2 ff = __bfloat1622float2(f2[p]);
            g_acc[2 * p + 0] += hr * gf.x;
            g_acc[2 * p + 1] += hr * gf.y;
            f_acc[2 * p + 0] += hr * ff.x;
            f_acc[2 * p + 1] += hr * ff.y;
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        s_gain[warp_id][lane][i] = g_acc[i];
        s_forg[warp_id][lane][i] = f_acc[i];
    }
    __syncthreads();

    // All 128 threads reduce the 256 channels (tid = 0..127 -> 2 channels each!)
    // Actually warp 0 (tid < 32) handles col_vec = c_tile*32 + lane (8 channels each):
    __shared__ int s_done;
    if (warp_id == 0) {
        const float4* dest4 = reinterpret_cast<const float4*>(dest + static_cast<int64_t>(s) * stride_dest_s + static_cast<int64_t>(t) * 2304);
        const float4* msg4  = reinterpret_cast<const float4*>(message + static_cast<int64_t>(row_st) * 2304);
        float4 d_lo = dest4[2 * col_vec + 0];
        float4 d_hi = dest4[2 * col_vec + 1];
        float4 m_lo = msg4[2 * col_vec + 0];
        float4 m_hi = msg4[2 * col_vec + 1];
        float d_arr[8] = {d_lo.x, d_lo.y, d_lo.z, d_lo.w, d_hi.x, d_hi.y, d_hi.z, d_hi.w};
        float m_arr[8] = {m_lo.x, m_lo.y, m_lo.z, m_lo.w, m_hi.x, m_hi.y, m_hi.z, m_hi.w};
        float x_arr[8];
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float gn = (s_gain[0][lane][i] + s_gain[1][lane][i]) + (s_gain[2][lane][i] + s_gain[3][lane][i]);
            float fg = (s_forg[0][lane][i] + s_forg[1][lane][i]) + (s_forg[2][lane][i] + s_forg[3][lane][i]);
            float one_m_f = __fsub_rn(1.0f, tanhf(fg));
            float g_val = tanhf(gn);
            float xv = __fadd_rn(__fmul_rn(one_m_f, d_arr[i]), __fmul_rn(g_val, m_arr[i]));
            x_arr[i] = xv;
            sq += xv * xv;
        }
        float4* wx4 = reinterpret_cast<float4*>(ws_x + static_cast<int64_t>(row_st) * 2304);
        wx4[2 * col_vec + 0] = make_float4(x_arr[0], x_arr[1], x_arr[2], x_arr[3]);
        wx4[2 * col_vec + 1] = make_float4(x_arr[4], x_arr[5], x_arr[6], x_arr[7]);
        sq = warp_sum_f32(sq);
        if (lane == 0) {
            ws_sq[row_st * 9 + c_tile] = sq;
            __threadfence();
            int prev = atomicAdd(&tile_counters[row_st], 1);
            s_done = (prev == 8) ? 1 : 0;
            if (prev == 8) {
                atomicExch(&tile_counters[row_st], 0);
            }
        }
    }
    __syncthreads();
    if (!s_done) return;

    // Last tile of row_st normalizes all 2304 channels in 3 iterations of 128 threads (or 288 uint4s)!
    __shared__ float s_rstd_row;
    if (tid < 32) {
        float v = (tid < 9) ? ws_sq[row_st * 9 + tid] : 0.0f;
        v = warp_sum_f32(v);
        if (tid == 0) {
            s_rstd_row = rsqrtf(__fdiv_rn(v, 2304.0f) + norm_eps);
        }
    }
    __syncthreads();

    const float rstd = s_rstd_row;
    const float4* wx4 = reinterpret_cast<const float4*>(ws_x + static_cast<int64_t>(row_st) * 2304);
    const uint4* nw_u4 = reinterpret_cast<const uint4*>(norm_w + static_cast<int64_t>(s) * stride_w_s);
    uint4* out_u4 = reinterpret_cast<uint4*>(out + static_cast<int64_t>(row_st) * 2304);
    for (int cv = tid; cv < 288; cv += 128) {
        float4 x0 = wx4[2 * cv + 0];
        float4 x1 = wx4[2 * cv + 1];
        uint4 nwv = nw_u4[cv];
        const __nv_bfloat162* nw2 = reinterpret_cast<const __nv_bfloat162*>(&nwv);
        float xa[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        uint4 ov;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 wf = __bfloat1622float2(nw2[p]);
            o2[p] = __floats2bfloat162_rn((xa[2 * p + 0] * rstd) * wf.x, (xa[2 * p + 1] * rstd) * wf.y);
        }
        out_u4[cv] = ov;
    }
}

extern "C" void launch_fused_up_blend_9tile_2304_sm80(
    const void* dest,
    const void* message,
    const void* hidden,
    const void* gf_t,
    const void* norm_w,
    void* ws_x,
    void* ws_sq,
    void* tile_counters,
    void* out,
    int64_t stride_dest_s,
    int64_t stride_gf_s,
    int64_t stride_w_s,
    int32_t S,
    int32_t T,
    float norm_eps,
    void* stream_ptr)
{
    if (S <= 0 || T <= 0) return;
    dim3 grid(9, S * T);
    fused_up_blend_9tile_2304_sm80_kernel<<<grid, 128, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
        static_cast<const float*>(dest),
        static_cast<const float*>(message),
        static_cast<const __nv_bfloat16*>(hidden),
        static_cast<const __nv_bfloat16*>(gf_t),
        static_cast<const __nv_bfloat16*>(norm_w),
        static_cast<float*>(ws_x),
        static_cast<float*>(ws_sq),
        static_cast<int32_t*>(tile_counters),
        static_cast<__nv_bfloat16*>(out),
        stride_dest_s, stride_gf_s, stride_w_s, T, norm_eps);
}
