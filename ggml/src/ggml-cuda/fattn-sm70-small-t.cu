#include "fattn-sm70-small-t.cuh"
#include "fattn-common.cuh"

#include <math_constants.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(FLASH_ATTN_AVAILABLE)

namespace ggml_sm70_small_t {

constexpr int D            = 256;
constexpr int Br           = 32;
constexpr int Bc           = 64;
constexpr int Warps        = 8;
constexpr int Threads      = Warps * 32; // 256
constexpr int DChunks      = D / 8;      // 32
constexpr int DSlice       = D / Warps;  // 32 PV cols per warp
constexpr int DChunksLocal = DSlice / 8; // 4
constexpr int KeyGroups    = Bc / 8;     // 8 (one 8-key QK^T group per warp)
constexpr int SmemPad      = 8;          // 16-byte aligned row pad to avoid 32-way bank conflicts
constexpr int Stride       = D + SmemPad;  // 264 halves
constexpr int PStride      = Bc + SmemPad; // 72 halves

constexpr size_t SmemBytes =
    sizeof(half) * (Br * Stride          // q_s:  16,896 B
                  + Bc * Stride          // k_s:  33,792 B
                  + Bc * Stride          // v_s:  33,792 B
                  + Br * PStride)        // p_s:   4,608 B
    + sizeof(float) * Warps * 32         // red_s: 1,024 B
    + sizeof(int);                       // tile_active: 4 B  -> total 90,116 B <= 98,304 B (96 KB Volta opt-in)

static_assert(SmemBytes <= 96 * 1024, "SM70 small-T shared memory exceeds Volta 96 KB opt-in limit");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

__device__ __forceinline__ int volta_qp_get_i() { return threadIdx.x & 31; }

__device__ __forceinline__ int volta_k_get_i() {
    const int lane = threadIdx.x & 31;
    return ((lane >> 4) << 2) + (lane & 3);
}

__device__ __forceinline__ int volta_d_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}

__device__ __forceinline__ int volta_d_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & 5);
}

__device__ __forceinline__ int volta_v_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return ((l >> 1) << 2) + (lane & 3);
}

__device__ __forceinline__ int volta_v_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return ((lane >> 4) << 1) + (l & 1);
}

__device__ __forceinline__ void volta_load_qp(half2 (&dst)[4], const half2 * __restrict__ base, int stride_h2) {
    const int row = volta_qp_get_i();
    *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(base + row * stride_h2);
}

__device__ __forceinline__ void volta_load_k(half2 (&dst)[4], const half2 * __restrict__ base, int stride_h2) {
    const int row = volta_k_get_i();
    *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(base + row * stride_h2);
}

__device__ __forceinline__ void volta_load_v(half2 (&dst)[4], const half2 * __restrict__ base, int stride_h2) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        dst[l] = base[volta_v_get_i(l) * stride_h2 + volta_v_get_j(l)];
    }
}

__device__ __forceinline__ void volta_mma_qk_f32(float (&d)[8], const half2 (&q)[4], const half2 (&k)[4]) {
    const int * Axi = reinterpret_cast<const int *>(q);
    const int * Bxi = reinterpret_cast<const int *>(k);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]), "r"(Bxi[1]));
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]), "r"(Bxi[3]));
}

__device__ __forceinline__ void volta_mma_pv_f32(float (&d)[8], const half2 (&p)[4], const half2 (&v)[4]) {
    const int * Pxi = reinterpret_cast<const int *>(p);
    const int * Vxi = reinterpret_cast<const int *>(v);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(Pxi[0]), "r"(Pxi[1]), "r"(Vxi[0]), "r"(Vxi[1]));
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(Pxi[2]), "r"(Pxi[3]), "r"(Vxi[2]), "r"(Vxi[3]));
}

// Fast Volta zero-I2F int8x8 -> half2x4 dequantizer via __byte_perm + fp16 1024 magic bias.
__device__ __forceinline__ int4 dequant_q8_0_8x_f16(uint2 raw, half sc) {
    const half2 scale2 = __half2half2(sc);
    const unsigned bias_bits = 0x64806480u; // 1152.0h packed in half2
    const half2 bias2 = *reinterpret_cast<const half2 *>(&bias_bits);
    const unsigned lo = raw.x ^ 0x80808080u;
    const unsigned hi = raw.y ^ 0x80808080u;
    unsigned h[4];
    h[0] = __byte_perm(lo, 0x64646464u, 0x5150);
    h[1] = __byte_perm(lo, 0x64646464u, 0x5352);
    h[2] = __byte_perm(hi, 0x64646464u, 0x5150);
    h[3] = __byte_perm(hi, 0x64646464u, 0x5352);
    half2 out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        out[j] = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&h[j]), bias2), scale2);
    }
    return *reinterpret_cast<const int4 *>(out);
}

// Fast Volta zero-I2F Q4_0 8-nibble -> half2x4 dequantizer via __byte_perm + fp16 1024 magic bias.
__device__ __forceinline__ int4 dequant_q4_0_8x_f16(uint2 raw, int shift, half sc) {
    const half2 scale2 = __half2half2(sc);
    const unsigned bias_bits = 0x64086408u; // 1032.0h packed in half2 (1024 + 8)
    const half2 bias2 = *reinterpret_cast<const half2 *>(&bias_bits);
    const unsigned lo = (raw.x >> shift) & 0x0F0F0F0Fu;
    const unsigned hi = (raw.y >> shift) & 0x0F0F0F0Fu;
    unsigned h[4];
    h[0] = __byte_perm(lo, 0x64646464u, 0x5150);
    h[1] = __byte_perm(lo, 0x64646464u, 0x5352);
    h[2] = __byte_perm(hi, 0x64646464u, 0x5150);
    h[3] = __byte_perm(hi, 0x64646464u, 0x5352);
    half2 out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        out[j] = __hmul2(__hsub2(*reinterpret_cast<const half2 *>(&h[j]), bias2), scale2);
    }
    return *reinterpret_cast<const int4 *>(out);
}

template <ggml_type type_KV>
struct kv_stage_reg {
    int4  f16_vec;
    uint2 q_raw;
    half  q_scale;
};

template <ggml_type type_KV>
__device__ __forceinline__ void load_kv_chunk_reg(
        const char * __restrict__ row_ptr, int lane, kv_stage_reg<type_KV> & reg) {
    if constexpr (type_KV == GGML_TYPE_F16) {
        reg.f16_vec = *reinterpret_cast<const int4 *>(row_ptr + lane * 16);
    } else if constexpr (type_KV == GGML_TYPE_Q8_0) {
        const block_q8_0 * blk = reinterpret_cast<const block_q8_0 *>(row_ptr) + (lane >> 2);
        const int iqs = (lane & 3) * 8;
        ggml_cuda_memcpy_1<8, 2>(&reg.q_raw, blk->qs + iqs);
        reg.q_scale = blk->d;
    } else if constexpr (type_KV == GGML_TYPE_Q4_0) {
        const block_q4_0 * blk = reinterpret_cast<const block_q4_0 *>(row_ptr) + (lane >> 2);
        const int iqs = (lane & 1) * 8;
        ggml_cuda_memcpy_1<8, 2>(&reg.q_raw, blk->qs + iqs);
        reg.q_scale = blk->d;
    }
}

template <ggml_type type_KV>
__device__ __forceinline__ void zero_kv_chunk_reg(kv_stage_reg<type_KV> & reg) {
    if constexpr (type_KV == GGML_TYPE_F16) {
        reg.f16_vec = make_int4(0, 0, 0, 0);
    } else {
        reg.q_raw   = make_uint2(0u, 0u);
        reg.q_scale = __float2half(0.0f);
    }
}

template <ggml_type type_KV>
__device__ __forceinline__ int4 decode_kv_chunk_reg(const kv_stage_reg<type_KV> & reg, int lane) {
    if constexpr (type_KV == GGML_TYPE_F16) {
        return reg.f16_vec;
    } else if constexpr (type_KV == GGML_TYPE_Q8_0) {
        return dequant_q8_0_8x_f16(reg.q_raw, reg.q_scale);
    } else if constexpr (type_KV == GGML_TYPE_Q4_0) {
        const int shift = ((lane >> 1) & 1) * 4;
        return dequant_q4_0_8x_f16(reg.q_raw, shift, reg.q_scale);
    } else {
        return make_int4(0, 0, 0, 0);
    }
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

template <ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(Threads, 1)
static __global__ void flash_attn_sm70_small_t_kernel(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        const char * __restrict__ sinks_ptr,
        float      * __restrict__ dst_ptr,
        float2     * __restrict__ dst_meta_ptr,
        float2     * __restrict__ dst_final_meta_ptr,
        const float scale,
        const float logit_softcap,
        const int32_t ne01, const int32_t ne02, const int32_t ne03,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int32_t ne11, const int32_t ne12,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne33,
        const int64_t nb31, const int64_t nb33,
        const int32_t gqa_ratio,
        const int32_t ntiles_q) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int ChunksPerThread = Bc * (D / 8) / Threads; // 8

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half  * q_s    = reinterpret_cast<half *>(smem_raw);
    half  * k_s    = q_s + Br * Stride;
    half  * v_s    = k_s + Bc * Stride;
    half  * p_s    = v_s + Bc * Stride;
    float * red_s  = reinterpret_cast<float *>(p_s + Br * PStride);
    int   * active_flag = reinterpret_cast<int *>(red_s + Warps * 32);

    const int q_tile      = blockIdx.x;
    const int split       = blockIdx.y;
    const int split_count = gridDim.y;
    const int z           = blockIdx.z;
    const int sequence    = z / ne12;
    const int kv_head     = z - sequence * ne12;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int total_rows_kv = ne01 * gqa_ratio;
    const int row_base      = q_tile * Br;
    const int row_count     = min(Br, total_rows_kv - row_base);
    if (row_count <= 0) {
        return;
    }

    const int head_base = kv_head * gqa_ratio;
    const int tok_min   = row_base / gqa_ratio;
    const int tok_max   = (row_base + row_count - 1) / gqa_ratio;

    // Stage Q tile [Br, D] into shared memory as FP16.
    for (int r = warp; r < Br; r += Warps) {
        const int gr = row_base + r;
        half * q_dst = q_s + r * Stride + lane * 8;
        if (r < row_count) {
            const int tok = gr / gqa_ratio;
            const int h   = gr - tok * gqa_ratio;
            const float2 * q_src = reinterpret_cast<const float2 *>(
                Q_ptr + sequence * nb03 + (head_base + h) * nb02 + tok * nb01 + lane * 8 * sizeof(float));
            half2 qh[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 fv = q_src[i];
                qh[i] = __floats2half2_rn(fv.x, fv.y);
            }
            *reinterpret_cast<int4 *>(q_dst) = *reinterpret_cast<const int4 *>(qh);
        } else {
            *reinterpret_cast<int4 *>(q_dst) = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const char * K_head = K_ptr + sequence * nb13 + kv_head * nb12;
    const char * V_head = V_ptr + sequence * nb23 + kv_head * nb22;
    const half * mask_seq = mask_ptr ? reinterpret_cast<const half *>(mask_ptr + (sequence % ne33) * nb33) : nullptr;
    const int64_t mask_s31 = mask_ptr ? (nb31 / int64_t(sizeof(half))) : 0;

    // Split KV range in Bc=64-aligned blocks across split_count CTAs.
    const int total_tiles = (ne11 + Bc - 1) / Bc;
    const int tiles_per_split = (total_tiles + split_count - 1) / split_count;
    const int tile_begin = split * tiles_per_split;
    const int tile_end   = min(total_tiles, tile_begin + tiles_per_split);

    const int r_lo  = volta_d_get_i(0) & ~2;
    const int r_hi  = volta_d_get_i(0) | 2;
    const int gr_lo = row_base + r_lo;
    const int gr_hi = row_base + r_hi;
    const int tok_lo = (r_lo < row_count) ? (gr_lo / gqa_ratio) : -1;
    const int tok_hi = (r_hi < row_count) ? (gr_hi / gqa_ratio) : -1;

    float acc_f[DChunksLocal][8];
#pragma unroll
    for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc_f[c][i] = 0.0f;
        }
    }

    float m_lo = -CUDART_INF_F;
    float m_hi = -CUDART_INF_F;
    float l_lo = 0.0f;
    float l_hi = 0.0f;

    auto check_tile_active = [&](int kb_tile) -> bool {
        if (!mask_seq) {
            return true;
        }
        const int k0 = kb_tile * Bc;
        int any_finite = 0;
        if (warp == 0) {
            for (int t = tok_min; t <= tok_max; ++t) {
                const half * mrow = mask_seq + int64_t(t) * mask_s31;
                const int k_a = k0 + lane;
                const int k_b = k0 + lane + 32;
                if ((k_a < ne11 && __half2float(mrow[k_a]) > -INFINITY) ||
                    (k_b < ne11 && __half2float(mrow[k_b]) > -INFINITY)) {
                    any_finite = 1;
                }
            }
            any_finite = __any_sync(FullMask, any_finite);
            if (lane == 0) {
                *active_flag = any_finite;
            }
        }
        __syncthreads();
        return (*active_flag) != 0;
    };

    kv_stage_reg<type_K> k_reg[ChunksPerThread];
    kv_stage_reg<type_V> v_reg[ChunksPerThread];

    auto load_tile_to_regs = [&](int kb_tile) {
        const int k0 = kb_tile * Bc;
        if (k0 + Bc <= ne11) {
            const char * k_base = K_head + int64_t(k0 + warp) * nb11;
            const char * v_base = V_head + int64_t(k0 + warp) * nb21;
#pragma unroll
            for (int i = 0; i < ChunksPerThread; ++i) {
                load_kv_chunk_reg<type_K>(k_base + int64_t(i * 8) * nb11, lane, k_reg[i]);
                load_kv_chunk_reg<type_V>(v_base + int64_t(i * 8) * nb21, lane, v_reg[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < ChunksPerThread; ++i) {
                const int key = k0 + warp + i * 8;
                if (key < ne11) {
                    load_kv_chunk_reg<type_K>(K_head + int64_t(key) * nb11, lane, k_reg[i]);
                    load_kv_chunk_reg<type_V>(V_head + int64_t(key) * nb21, lane, v_reg[i]);
                } else {
                    zero_kv_chunk_reg<type_K>(k_reg[i]);
                    zero_kv_chunk_reg<type_V>(v_reg[i]);
                }
            }
        }
    };

    // Find first active tile in [tile_begin, tile_end).
    int kb_cur = tile_begin;
    while (kb_cur < tile_end && !check_tile_active(kb_cur)) {
        ++kb_cur;
    }

    if (kb_cur < tile_end) {
        load_tile_to_regs(kb_cur);
    }

    while (kb_cur < tile_end) {
        const int k0 = kb_cur * Bc;

        // Decode prefetched K/V tile into shared memory.
#pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            const int key_l = warp + i * 8;
            const int d     = lane * 8;
            *reinterpret_cast<int4 *>(&k_s[key_l * Stride + d]) = decode_kv_chunk_reg<type_K>(k_reg[i], lane);
            *reinterpret_cast<int4 *>(&v_s[key_l * Stride + d]) = decode_kv_chunk_reg<type_V>(v_reg[i], lane);
        }

        // Find next active tile and overlap its HBM loads with current tile compute.
        int kb_next = kb_cur + 1;
        while (kb_next < tile_end && !check_tile_active(kb_next)) {
            ++kb_next;
        }
        if (kb_next < tile_end) {
            load_tile_to_regs(kb_next);
        }
        __syncthreads(); // (1) K/V tile visible in shared memory

        // ---- QK^T: warp w computes only its own 8 keys [k0 + w*8, k0 + w*8 + 8) ----
        const int sub_k0 = k0 + warp * 8;
        float d_score[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int c = 0; c < DChunks; ++c) {
            half2 qf[4];
            volta_load_qp(qf, reinterpret_cast<const half2 *>(&q_s[c * 8]), Stride / 2);
            half2 kf[4];
            volta_load_k(kf, reinterpret_cast<const half2 *>(&k_s[warp * 8 * Stride + c * 8]), Stride / 2);
            volta_mma_qk_f32(d_score, qf, kf);
        }

#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int r   = volta_d_get_i(l);
            const int key = sub_k0 + volta_d_get_j(l);
            const int tok = ((l & 2) == 0) ? tok_lo : tok_hi;
            if (r < row_count && key < ne11) {
                float s = d_score[l] * scale;
                if constexpr (use_logit_softcap) {
                    s = logit_softcap * tanhf(s);
                }
                if (mask_seq) {
                    const float mv = __half2float(mask_seq[int64_t(tok) * mask_s31 + key]);
                    s = (mv == -INFINITY) ? -CUDART_INF_F : (s + mv);
                }
                d_score[l] = s;
            } else {
                d_score[l] = -CUDART_INF_F;
            }
        }

        // Per-warp row maxima for the 8 keys owned by this warp.
        float bm_lo = fmaxf(fmaxf(d_score[0], d_score[1]), fmaxf(d_score[4], d_score[5]));
        float bm_hi = fmaxf(fmaxf(d_score[2], d_score[3]), fmaxf(d_score[6], d_score[7]));
        bm_lo = fmaxf(bm_lo, __shfl_xor_sync(FullMask, bm_lo, 2, 32));
        bm_hi = fmaxf(bm_hi, __shfl_xor_sync(FullMask, bm_hi, 2, 32));

        // Identity: (((lane & 2) == 0) ? r_lo : r_hi) == lane -> each row written once.
        red_s[warp * 32 + lane] = ((lane & 2) == 0) ? bm_lo : bm_hi;
        __syncthreads(); // (2) per-warp maxima visible

        float tm_lo = -CUDART_INF_F;
        float tm_hi = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            tm_lo = fmaxf(tm_lo, red_s[w * 32 + r_lo]);
            tm_hi = fmaxf(tm_hi, red_s[w * 32 + r_hi]);
        }

        const float new_m_lo = fmaxf(m_lo, tm_lo);
        const float new_m_hi = fmaxf(m_hi, tm_hi);
        const float alpha_lo = (m_lo > -CUDART_INF_F) ? exp2f((m_lo - new_m_lo) * LOG2E) : 0.0f;
        const float alpha_hi = (m_hi > -CUDART_INF_F) ? exp2f((m_hi - new_m_hi) * LOG2E) : 0.0f;

#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const float nm = ((l & 2) == 0) ? new_m_lo : new_m_hi;
            d_score[l] = (nm > -CUDART_INF_F && d_score[l] > -CUDART_INF_F)
                ? exp2f((d_score[l] - nm) * LOG2E)
                : 0.0f;
        }

        float bl_lo = d_score[0] + d_score[1] + d_score[4] + d_score[5];
        float bl_hi = d_score[2] + d_score[3] + d_score[6] + d_score[7];
        bl_lo += __shfl_xor_sync(FullMask, bl_lo, 2, 32);
        bl_hi += __shfl_xor_sync(FullMask, bl_hi, 2, 32);

        l_lo = l_lo * alpha_lo + bl_lo;
        l_hi = l_hi * alpha_hi + bl_hi;
        m_lo = new_m_lo;
        m_hi = new_m_hi;

#pragma unroll
        for (int l = 0; l < 8; ++l) {
            p_s[volta_d_get_i(l) * PStride + warp * 8 + volta_d_get_j(l)] = __float2half(d_score[l]);
        }
        __syncthreads(); // (3) full 32x64 P tile visible

        // ---- PV: each warp computes its own D/8 = 32 output columns across all 64 keys ----
#pragma unroll
        for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc_f[c][i] *= ((i & 2) == 0) ? alpha_lo : alpha_hi;
            }
        }
#pragma unroll
        for (int g = 0; g < KeyGroups; ++g) {
            half2 pf[4];
            *reinterpret_cast<int4 *>(pf) = *reinterpret_cast<const int4 *>(&p_s[lane * PStride + g * 8]);
#pragma unroll
            for (int c = 0; c < DChunksLocal; ++c) {
                half2 vf[4];
                volta_load_v(vf, reinterpret_cast<const half2 *>(&v_s[g * 8 * Stride + warp * DSlice + c * 8]), Stride / 2);
                volta_mma_pv_f32(acc_f[c], pf, vf);
            }
        }
        __syncthreads(); // (4) safe to overwrite k_s/v_s/p_s on next tile

        kb_cur = kb_next;
    }

    // Combine per-warp row sums across the 8 warps.
    red_s[warp * 32 + lane] = ((lane & 2) == 0) ? l_lo : l_hi;
    __syncthreads();

    float sum_lo = 0.0f;
    float sum_hi = 0.0f;
#pragma unroll
    for (int w = 0; w < Warps; ++w) {
        sum_lo += red_s[w * 32 + r_lo];
        sum_hi += red_s[w * 32 + r_hi];
    }

    // Fold optional attention sinks on split 0 only (matches flash_attn_ext_vec_gqa).
    if (sinks_ptr != nullptr && split == 0) {
        const float * sinks = reinterpret_cast<const float *>(sinks_ptr);
        if (r_lo < row_count) {
            const int h_lo    = gr_lo - tok_lo * gqa_ratio;
            const float sink  = sinks[head_base + h_lo];
            const float new_m = fmaxf(sink, m_lo);
            const float ms    = (m_lo > -CUDART_INF_F) ? expf(m_lo - new_m) : 0.0f;
            m_lo   = new_m;
            sum_lo = sum_lo * ms + expf(sink - new_m);
#pragma unroll
            for (int c = 0; c < DChunksLocal; ++c) {
                acc_f[c][0] *= ms; acc_f[c][1] *= ms;
                acc_f[c][4] *= ms; acc_f[c][5] *= ms;
            }
        }
        if (r_hi < row_count) {
            const int h_hi    = gr_hi - tok_hi * gqa_ratio;
            const float sink  = sinks[head_base + h_hi];
            const float new_m = fmaxf(sink, m_hi);
            const float ms    = (m_hi > -CUDART_INF_F) ? expf(m_hi - new_m) : 0.0f;
            m_hi   = new_m;
            sum_hi = sum_hi * ms + expf(sink - new_m);
#pragma unroll
            for (int c = 0; c < DChunksLocal; ++c) {
                acc_f[c][2] *= ms; acc_f[c][3] *= ms;
                acc_f[c][6] *= ms; acc_f[c][7] *= ms;
            }
        }
    }

    // Write metadata (natural-log space, matching flash_attn_combine_results).
    if (warp == 0 && lane < row_count) {
        const int gr   = row_base + lane;
        const int tok  = gr / gqa_ratio;
        const int head = head_base + (gr - tok * gqa_ratio);
        const int64_t j_dst = (int64_t(sequence) * ne01 + tok) * ne02 + head;
        const float own_m = ((lane & 2) == 0) ? m_lo : m_hi;
        const float own_l = ((lane & 2) == 0) ? sum_lo : sum_hi;
        const float safe_m = (own_l > 0.0f && own_m > -CUDART_INF_F) ? own_m : (-FLT_MAX / 2.0f);
        if (split_count == 1) {
            if (dst_final_meta_ptr != nullptr) {
                dst_final_meta_ptr[j_dst] = make_float2(safe_m, own_l);
            }
        } else {
            dst_meta_ptr[j_dst * split_count + split] = make_float2(safe_m, own_l);
        }
    }

    // Write output / split partials.
    const float inv_l_lo = (split_count == 1 && sum_lo > 0.0f) ? (1.0f / sum_lo) : 1.0f;
    const float inv_l_hi = (split_count == 1 && sum_hi > 0.0f) ? (1.0f / sum_hi) : 1.0f;

#pragma unroll
    for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = volta_d_get_i(i);
            if (r >= row_count) {
                continue;
            }
            const int gr   = row_base + r;
            const int tok  = ((i & 2) == 0) ? tok_lo : tok_hi;
            const int head = head_base + (gr - tok * gqa_ratio);
            const int d    = warp * DSlice + c * 8 + volta_d_get_j(i);
            const int64_t j_dst = (int64_t(sequence) * ne01 + tok) * ne02 + head;
            float val = acc_f[c][i];
            if (split_count == 1) {
                const float inv_l = ((i & 2) == 0) ? inv_l_lo : inv_l_hi;
                const float sum_r = ((i & 2) == 0) ? sum_lo   : sum_hi;
                dst_ptr[j_dst * D + d] = (sum_r > 0.0f) ? (val * inv_l) : 0.0f;
            } else {
                dst_ptr[(j_dst * split_count + split) * D + d] = val;
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr, dst_meta_ptr, dst_final_meta_ptr,
        scale, logit_softcap, ne01, ne02, ne03, nb01, nb02, nb03, ne11, ne12, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne33, nb31, nb33, gqa_ratio, ntiles_q);
    NO_DEVICE_CODE;
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
}

template <ggml_type type_K, ggml_type type_V>
static void launch_sm70_small_t_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q         = dst->src[0];
    const ggml_tensor * K         = dst->src[1];
    const ggml_tensor * V         = dst->src[2];
    const ggml_tensor * mask      = dst->src[3];
    const ggml_tensor * sinks     = dst->src[4];
    const ggml_tensor * body_meta = dst->src[8];

    float scale         = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const bool use_softcap = logit_softcap != 0.0f;
    if (use_softcap) {
        scale /= logit_softcap;
    }

    const int32_t ne01 = int32_t(Q->ne[1]);
    const int32_t ne02 = int32_t(Q->ne[2]);
    const int32_t ne03 = int32_t(Q->ne[3]);
    const int32_t ne11 = int32_t(K->ne[1]);
    const int32_t ne12 = int32_t(K->ne[2]);
    const int32_t gqa_ratio = ne02 / ne12;
    const int32_t ntiles_q  = (ne01 * gqa_ratio + Br - 1) / Br;

    const int device = ggml_cuda_get_device();
    const int nsm    = ggml_cuda_info().devices[device].nsm;
    const int grid_xz = ntiles_q * ne12 * ne03;
    const int total_kv_tiles = std::max(1, (ne11 + Bc - 1) / Bc);

    // Choose split count to saturate all 80 Volta SMs while avoiding fragmented combine overhead.
    static const int env_split_keys = []() {
        const char * e = std::getenv("GGML_CUDA_SM70_SMALL_T_SPLIT_KEYS");
        return e ? std::atoi(e) : 0;
    }();
    int splits = 1;
    if (total_kv_tiles > 1 && grid_xz > 0) {
        const int target_wave_splits = std::max(1, nsm / grid_xz);
        if (env_split_keys > 0) {
            const int by_keys = (ne11 + env_split_keys - 1) / env_split_keys;
            splits = std::max(1, std::min({ total_kv_tiles, by_keys, std::max(target_wave_splits, 64) }));
        } else if (ne11 >= 32768) {
            // Deep context: 1 full wave (80 CTAs) or up to 2 waves when keys/split >= 1920.
            const int max_two_waves = std::min(64, target_wave_splits * 2);
            const int by_1920       = std::max(target_wave_splits, ne11 / 1920);
            splits = std::max(1, std::min({ total_kv_tiles, by_1920, max_two_waves }));
        } else {
            // Short/medium context: keep >= 256 keys per split, fill up to 1 SM wave.
            const int by_min_keys = std::max(1, ne11 / 256);
            splits = std::max(1, std::min({ total_kv_tiles, target_wave_splits, by_min_keys }));
        }
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    if (splits > 1) {
        dst_tmp.alloc(size_t(splits) * ggml_nelements(dst));
        dst_tmp_meta.alloc(size_t(splits) * ggml_nrows(dst));
    }

    auto kernel = use_softcap
        ? flash_attn_sm70_small_t_kernel<type_K, type_V, true>
        : flash_attn_sm70_small_t_kernel<type_K, type_V, false>;

    static bool smem_configured[GGML_CUDA_MAX_DEVICES][2] = {};
    const int cfg_idx = use_softcap ? 1 : 0;
    if (!smem_configured[device][cfg_idx]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SmemBytes)));
        smem_configured[device][cfg_idx] = true;
    }

    const dim3 blocks_num(ntiles_q, splits, ne12 * ne03);
    const dim3 block_dim(Threads, 1, 1);

    kernel<<<blocks_num, block_dim, SmemBytes, stream>>>(
        (const char *) Q->data,
        (const char *) K->data,
        (const char *) V->data,
        mask ? (const char *) mask->data : nullptr,
        sinks ? (const char *) sinks->data : nullptr,
        splits > 1 ? dst_tmp.ptr : (float *) dst->data,
        dst_tmp_meta.ptr,
        body_meta ? (float2 *) body_meta->data : nullptr,
        scale, logit_softcap,
        ne01, ne02, ne03,
        Q->nb[1], Q->nb[2], Q->nb[3],
        ne11, ne12,
        K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3],
        mask ? int32_t(mask->ne[1]) : 0,
        mask ? int32_t(mask->ne[3]) : 0,
        mask ? mask->nb[1] : 0,
        mask ? mask->nb[3] : 0,
        gqa_ratio,
        ntiles_q);
    CUDA_CHECK(cudaGetLastError());

    if (splits > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine(ne01, ne02, ne03);
        const size_t nbytes_shared_combine = size_t(splits) * sizeof(float2);
        const ggml_cuda_kernel_launch_params launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<D>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) dst->data,
            body_meta ? (float2 *) body_meta->data : nullptr, splits);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ggml_sm70_small_t

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(FLASH_ATTN_AVAILABLE)

bool ggml_cuda_flash_attn_ext_sm70_small_t_supported(int device, const ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(FLASH_ATTN_AVAILABLE)
    static const bool enabled = []() {
        const char * e = std::getenv("GGML_CUDA_SM70_SMALL_T_FA");
        return !(e && std::strcmp(e, "0") == 0);
    }();
    if (!enabled) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!volta_mma_available(cc)) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    // Small-T verify & multi-seq decode window (single-token decode stays on flash_attn_ext_vec_gqa).
    if (Q->ne[1] < 2 || Q->ne[1] > 16) {
        return false;
    }
    if (K->ne[1] <= 0 || K->ne[2] <= 0 || Q->ne[2] <= 0 || K->ne[2] != V->ne[2]) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio < 2) {
        return false;
    }
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }

    auto kv_type_ok = [](ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
    };
    if (!kv_type_ok(K->type) || !kv_type_ok(V->type)) {
        return false;
    }

    // Supported pairs: homogeneous (F16/F16, Q8_0/Q8_0, Q4_0/Q4_0) and common mixed (Q8_0/Q4_0).
    const bool pair_ok = (K->type == V->type) ||
                         (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0);
    if (!pair_ok) {
        return false;
    }

    // For pure F16/F16 KV, fattn-mma-f16 already reads F16 directly without materialization;
    // route F16/F16 here only at deep context (>= 40K) when all Q rows fit in one Br=32 tile
    // (e.g. MTP T=2..5 on GQA-6), where our 8-warp key-split kernel beats MMA (+9% at 64K).
    // Quantized KV (Q8_0/Q4_0) avoids full-KV F16 materialization and wins across all depths & T=2..16.
    if (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 &&
            (Q->ne[1] * gqa_ratio > 32 || K->ne[1] < 40960)) {
        return false;
    }

    // Check alignment constraints for coalesced loads.
    if (Q->nb[0] != sizeof(float) || reinterpret_cast<uintptr_t>(Q->data) % 8 != 0 ||
            Q->nb[1] % 8 != 0 || Q->nb[2] % 8 != 0 || Q->nb[3] % 8 != 0) {
        return false;
    }
    const size_t k_align = K->type == GGML_TYPE_F16 ? 16 : 2;
    const size_t v_align = V->type == GGML_TYPE_F16 ? 16 : 2;
    if (K->nb[0] != ggml_type_size(K->type) || reinterpret_cast<uintptr_t>(K->data) % k_align != 0 ||
            K->nb[1] % k_align != 0 || K->nb[2] % k_align != 0 || K->nb[3] % k_align != 0) {
        return false;
    }
    if (V->nb[0] != ggml_type_size(V->type) || reinterpret_cast<uintptr_t>(V->data) % v_align != 0 ||
            V->nb[1] % v_align != 0 || V->nb[2] % v_align != 0 || V->nb[3] % v_align != 0) {
        return false;
    }

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 4) != 0) { // sparse n_kv_max
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[0] != sizeof(half))) {
        return false;
    }
    if (dst->src[4] != nullptr && dst->src[4]->type != GGML_TYPE_F32) {
        return false;
    }

    return true;
#else
    GGML_UNUSED_VARS(device, dst);
    return false;
#endif
}

void ggml_cuda_flash_attn_ext_sm70_small_t(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(FLASH_ATTN_AVAILABLE)
    const ggml_type type_K = dst->src[1]->type;
    const ggml_type type_V = dst->src[2]->type;

    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
        ggml_sm70_small_t::launch_sm70_small_t_case<GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst);
    } else if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
        ggml_sm70_small_t::launch_sm70_small_t_case<GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
    } else if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
        ggml_sm70_small_t::launch_sm70_small_t_case<GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>(ctx, dst);
    } else if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
        ggml_sm70_small_t::launch_sm70_small_t_case<GGML_TYPE_Q8_0, GGML_TYPE_Q4_0>(ctx, dst);
    } else {
        GGML_ABORT("unsupported K/V type pair in ggml_cuda_flash_attn_ext_sm70_small_t");
    }
#else
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("ggml_cuda_flash_attn_ext_sm70_small_t unavailable");
#endif
}
