#include "gated-delta-net-volta.cuh"
#include "mma.cuh"

using namespace ggml_cuda_mma;

// Volta sm_70 FP16 Tensor-Core (mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32)
// chunked WY/UT Gated Delta Net kernel for S_v = 128, !KDA, BT = 16.
//
// Each threadblock owns a [32_v, 128_k] slice of the transposed recurrent state M = S^T
// (4 blocks per (head, seq) cover all 128 V-rows -> 192 blocks for H=48 on 80-SM V100).
// Within each block of 4 warps (128 threads), warp w (0..3) keeps M[col0..col0+31, 32*w..32*w+31]
// permanently in 4 FP32 Volta MMA accumulator tiles (32 registers/thread) across all chunks.
//
// Chunk math (BT = 16, G_t = sum_{r=0}^t g_r):
//   L_{t,j} = beta_t * exp(G_t - G_j) * (k_t^T k_j)  (unit lower triangular)
//   A_{t,j} = exp(G_t - G_j) * (q_t^T k_j)           (lower triangular)
//   R[c,t]  = beta_t * (v_t[c] - exp(G_t) * (M_{-1} k_t)[c])
//   Solve L * Delta^T = R^T in FP32
//   O[c,t]  = scale * (exp(G_t) * (M_{-1} q_t)[c] + sum_{j<=t} A_{t,j} * Delta[c,j])
//   M_new   = exp(G_15) * M_{-1} + sum_{t=0}^{15} (exp(G_15 - G_t) * Delta[:,t]) * k_t^T
//
// Unscaled k_t and q_t (L2-normalized O(1)) are passed into FP16 MMA; exp(G_t) is applied
// in FP32 after MMA so large negative log-gates (e.g. g = -20) never underflow FP16 inputs.

template <bool keep_rs_t>
__global__ void __launch_bounds__(128, 3)
gated_delta_net_volta_chunked_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ curr_state,
        float * __restrict__       dst,
        float * __restrict__       state,
        int64_t                    H,
        int64_t                    n_tokens,
        int64_t                    n_seqs,
        int64_t                    sq1,
        int64_t                    sq2,
        int64_t                    sq3,
        int64_t                    sv1,
        int64_t                    sv2,
        int64_t                    sv3,
        int64_t                    sb1,
        int64_t                    sb2,
        int64_t                    sb3,
        const uint3                neqk1_magic,
        const uint3                rq3_magic,
        float                      scale,
        int64_t                    state_slot_stride,
        int                        K) {
#if defined(VOLTA_MMA_AVAILABLE)
    constexpr int S_v         = 128;
    constexpr int BT          = 16;
    constexpr int V_TILE      = 32;
    constexpr int KQ_STRIDE_H = 136; // 128 + 8 halfs padding -> 272 B (68 banks -> 0 bank conflicts on LDS.128)
    constexpr int KT_STRIDE_H = 24;  // 16 + 8 halfs padding  -> 48 B  (12 banks -> 0 bank conflicts on LDS.128)

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      col0     = blockIdx.z * V_TILE;

    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..3
    const int tid  = warp * 32 + lane;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const int64_t gb_base = sequence * sb3 + h_idx * sb1;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    curr_state += state_offset;
    state      += state_offset;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    // Shared memory layout (~31.7 KB/block -> 3 blocks/SM on V100's 96 KB SMEM)
    __shared__ __align__(16) half  smem_KQ_h[32][KQ_STRIDE_H];   // rows 0..15 = K_c, rows 16..31 = Q_c
    __shared__ __align__(16) half  smem_KT_h[128][KT_STRIDE_H];  // K_c^T [128_k, 16_t]
    __shared__ __align__(16) half  smem_Delta_h[32][KT_STRIDE_H];// Delta_decay [32_v, 16_t]
    __shared__ float               smem_part[4][32][17];         // 4-warp reduction buffer
    __shared__ float               smem_Delta_f32[32][17];       // R / Delta [32_v, 16_t]
    __shared__ float               smem_V[32][17];               // V_c then MQ [32_v, 16_t]
    __shared__ float               smem_L[16][17];               // unit lower triangular L
    __shared__ float               smem_A[16][17];               // lower triangular A_qk
    __shared__ float               smem_g_cs[16];
    __shared__ float               smem_beta[16];

    // Load initial state M_blk[col0..col0+31, 32*warp..32*warp+31] into FP32 Volta MMA tiles
    tile<32, 8, float> M_reg[4];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int i_v = M_reg[m].get_i(l);
            const int j_k = 32 * warp + 8 * m + M_reg[m].get_j(l);
            M_reg[m].x[l] = curr_state[(col0 + i_v) * S_v + j_k];
        }
    }

    // Reserve at least K tokens for the exact scalar tail when keep_rs_t is enabled
    const int64_t n_chunkable = keep_rs_t ? (n_tokens - K) : n_tokens;
    const int64_t n_chunks    = n_chunkable > 0 ? (n_chunkable / BT) : 0;

    for (int64_t chunk = 0; chunk < n_chunks; ++chunk) {
        const int64_t t0 = chunk * BT;

        // ------------------------------------------------------------------
        // Step 1: Load K_c, Q_c, V_c, g, beta for BT = 16 tokens
        // ------------------------------------------------------------------
#pragma unroll
        for (int t = 0; t < BT; ++t) {
            const float kv = k_base[(t0 + t) * sq2 + tid];
            const float qv = q_base[(t0 + t) * sq2 + tid];
            const half  kh = __float2half(kv);
            smem_KQ_h[t][tid]      = kh;
            smem_KQ_h[BT + t][tid] = __float2half(qv);
            smem_KT_h[tid][t]      = kh;
        }

        // Load V_c [32_v, 16_t] (512 floats -> 4 floats/thread, coalesced across 32 threads)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = r * 128 + tid;
            const int t_v = idx >> 5;  // 0..15
            const int c_v = idx & 31;  // 0..31
            smem_V[c_v][t_v] = v_base[(t0 + t_v) * sv2 + (col0 + c_v)];
        }

        if (tid < BT) {
            const int64_t gb_off = gb_base + (t0 + tid) * sb2;
            smem_beta[tid] = beta[gb_off];
            smem_g_cs[tid] = g[gb_off];
        }
        __syncthreads();

        // Prefix sum of log-gates G_t = sum_{r=0}^t g_r (16 elements in 1 thread, ~16 cycles)
        if (tid == 0) {
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                acc += smem_g_cs[t];
                smem_g_cs[t] = acc;
            }
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // Step 2: Intra-chunk Gram [K_c @ K_c^T ; Q_c @ K_c^T] via Volta MMA
        // ------------------------------------------------------------------
        tile<32, 8, float> gram[2]; // default-initialized to 0
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int k_off = 32 * warp + 8 * m;
            tile<32, 4, half2> A_kq;
            tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> B_k0, B_k1;
            load_ldmatrix(A_kq, (const half2 *) &smem_KQ_h[0][k_off], KQ_STRIDE_H / 2);
            load_ldmatrix(B_k0, (const half2 *) &smem_KQ_h[0][k_off], KQ_STRIDE_H / 2);
            load_ldmatrix(B_k1, (const half2 *) &smem_KQ_h[8][k_off], KQ_STRIDE_H / 2);
            mma(gram[0], A_kq, B_k0);
            mma(gram[1], A_kq, B_k1);
        }

#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int i = gram[0].get_i(l);
            const int j = gram[0].get_j(l);
            smem_part[warp][i][j]     = gram[0].x[l];
            smem_part[warp][i][j + 8] = gram[1].x[l];
        }
        __syncthreads();

        // Reduce gram across 4 warps and materialize L (rows 0..15) and A (rows 16..31)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = r * 128 + tid;
            const int i   = idx >> 4; // 0..31
            const int j   = idx & 15; // 0..15
            const float val = smem_part[0][i][j] + smem_part[1][i][j] +
                              smem_part[2][i][j] + smem_part[3][i][j];
            if (i < BT) {
                if (j < i) {
                    smem_L[i][j] = smem_beta[i] * expf(smem_g_cs[i] - smem_g_cs[j]) * val;
                } else {
                    smem_L[i][j] = (j == i) ? 1.0f : 0.0f;
                }
            } else {
                const int ti = i - BT;
                if (j <= ti) {
                    smem_A[ti][j] = expf(smem_g_cs[ti] - smem_g_cs[j]) * val;
                } else {
                    smem_A[ti][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // Step 3: State projections MK = M_blk @ K_c^T, MQ = M_blk @ Q_c^T
        // ------------------------------------------------------------------
        tile<32, 8, float> MK_acc[2];
        tile<32, 8, float> MQ_acc[2];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int k_off = 32 * warp + 8 * m;
            const tile<32, 4, half2> A_m = get_half2(M_reg[m]);

            tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> B_t;
            load_ldmatrix(B_t, (const half2 *) &smem_KQ_h[0][k_off], KQ_STRIDE_H / 2);
            mma(MK_acc[0], A_m, B_t);
            load_ldmatrix(B_t, (const half2 *) &smem_KQ_h[8][k_off], KQ_STRIDE_H / 2);
            mma(MK_acc[1], A_m, B_t);

            load_ldmatrix(B_t, (const half2 *) &smem_KQ_h[16][k_off], KQ_STRIDE_H / 2);
            mma(MQ_acc[0], A_m, B_t);
            load_ldmatrix(B_t, (const half2 *) &smem_KQ_h[24][k_off], KQ_STRIDE_H / 2);
            mma(MQ_acc[1], A_m, B_t);
        }

        // Reduce MK across 4 warps -> R[c][t] = beta[t] * (V[c][t] - exp(G[t]) * MK[c][t])
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int i = MK_acc[0].get_i(l);
            const int j = MK_acc[0].get_j(l);
            smem_part[warp][i][j]     = MK_acc[0].x[l];
            smem_part[warp][i][j + 8] = MK_acc[1].x[l];
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = r * 128 + tid;
            const int c   = idx >> 4; // 0..31
            const int t   = idx & 15; // 0..15
            const float mk = smem_part[0][c][t] + smem_part[1][c][t] +
                             smem_part[2][c][t] + smem_part[3][c][t];
            smem_Delta_f32[c][t] = smem_beta[t] * (smem_V[c][t] - expf(smem_g_cs[t]) * mk);
        }
        __syncthreads();

        // Reduce MQ across 4 warps -> smem_V[c][t] = exp(G[t]) * MQ[c][t]
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int i = MQ_acc[0].get_i(l);
            const int j = MQ_acc[0].get_j(l);
            smem_part[warp][i][j]     = MQ_acc[0].x[l];
            smem_part[warp][i][j + 8] = MQ_acc[1].x[l];
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = r * 128 + tid;
            const int c   = idx >> 4; // 0..31
            const int t   = idx & 15; // 0..15
            const float mq = smem_part[0][c][t] + smem_part[1][c][t] +
                             smem_part[2][c][t] + smem_part[3][c][t];
            smem_V[c][t] = expf(smem_g_cs[t]) * mq;
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // Step 4: Forward substitution L @ Delta = R, output O, pack Delta_decay
        // ------------------------------------------------------------------
        if (warp == 0) {
            const int c = lane; // 0..31 -> 1 full warp, 100% coalesced output stores
            float d[BT];
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                d[t] = smem_Delta_f32[c][t];
            }

            // Unit lower-triangular solve in FP32
#pragma unroll
            for (int t = 1; t < BT; ++t) {
                float sum = d[t];
#pragma unroll
                for (int j = 0; j < t; ++j) {
                    sum -= smem_L[t][j] * d[j];
                }
                d[t] = sum;
            }

            const float g_last = smem_g_cs[BT - 1];
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                float o_t = smem_V[c][t];
#pragma unroll
                for (int j = 0; j <= t; ++j) {
                    o_t += smem_A[t][j] * d[j];
                }
                attn_data[(t0 + t) * S_v * H + (col0 + c)] = o_t * scale;
                smem_Delta_h[c][t] = __float2half(d[t] * expf(g_last - smem_g_cs[t]));
            }
        }
        __syncthreads();

        // ------------------------------------------------------------------
        // Step 5: State update M_blk = exp(G_15) * M_blk + Delta_decay @ K_c
        // ------------------------------------------------------------------
        const float g_decay = expf(smem_g_cs[BT - 1]);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                M_reg[m].x[l] *= g_decay;
            }
        }

        tile<32, 4, half2> D0, D1;
        load_ldmatrix(D0, (const half2 *) &smem_Delta_h[0][0], KT_STRIDE_H / 2);
        load_ldmatrix(D1, (const half2 *) &smem_Delta_h[0][8], KT_STRIDE_H / 2);

#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int k_off = 32 * warp + 8 * m;
            tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> KT0, KT1;
            load_ldmatrix(KT0, (const half2 *) &smem_KT_h[k_off][0], KT_STRIDE_H / 2);
            load_ldmatrix(KT1, (const half2 *) &smem_KT_h[k_off][8], KT_STRIDE_H / 2);
            mma(M_reg[m], D0, KT0);
            mma(M_reg[m], D1, KT1);
        }
        __syncthreads();
    }

    // ----------------------------------------------------------------------
    // Exact scalar tail for remaining tokens (including all K rollback slots)
    // Reuses smem_KQ_h (as float[128]*2), smem_V[0..31][0], smem_Delta_f32[0..31][0], smem_part
    // ----------------------------------------------------------------------
    const int64_t t_tail_start = n_chunks * BT;
    if (t_tail_start < n_tokens) {
        float * smem_k_tail = (float *) &smem_KQ_h[0][0];   // 128 floats = 512 B
        float * smem_q_tail = smem_k_tail + S_v;            // 128 floats = 512 B

        const int i_v0 = (lane & ~2);
        const int i_v1 = i_v0 + 2;

        for (int64_t t = t_tail_start; t < n_tokens; ++t) {
            smem_k_tail[tid] = k_base[t * sq2 + tid];
            smem_q_tail[tid] = q_base[t * sq2 + tid];
            if (tid < V_TILE) {
                smem_V[tid][0] = v_base[t * sv2 + (col0 + tid)];
            }
            if (tid == 0) {
                const int64_t gb_off = gb_base + t * sb2;
                smem_g_cs[0] = expf(g[gb_off]);
                smem_beta[0] = beta[gb_off];
            }
            __syncthreads();

            const float g_val    = smem_g_cs[0];
            const float beta_val = smem_beta[0];

            // kv = M_blk @ k_t
            float kv0 = 0.0f;
            float kv1 = 0.0f;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int j_base = 32 * warp + 8 * m + (lane & 2);
                kv0 += M_reg[m].x[0] * smem_k_tail[j_base + 0];
                kv0 += M_reg[m].x[1] * smem_k_tail[j_base + 1];
                kv0 += M_reg[m].x[4] * smem_k_tail[j_base + 4];
                kv0 += M_reg[m].x[5] * smem_k_tail[j_base + 5];

                kv1 += M_reg[m].x[2] * smem_k_tail[j_base + 0];
                kv1 += M_reg[m].x[3] * smem_k_tail[j_base + 1];
                kv1 += M_reg[m].x[6] * smem_k_tail[j_base + 4];
                kv1 += M_reg[m].x[7] * smem_k_tail[j_base + 5];
            }
            kv0 += __shfl_xor_sync(0xFFFFFFFFU, kv0, 2);
            kv1 += __shfl_xor_sync(0xFFFFFFFFU, kv1, 2);
            if ((lane & 2) == 0) {
                smem_part[warp][i_v0][0] = kv0;
                smem_part[warp][i_v1][0] = kv1;
            }
            __syncthreads();

            if (tid < V_TILE) {
                const float kv_c = smem_part[0][tid][0] + smem_part[1][tid][0] +
                                   smem_part[2][tid][0] + smem_part[3][tid][0];
                smem_Delta_f32[tid][0] = (smem_V[tid][0] - g_val * kv_c) * beta_val;
            }
            __syncthreads();

            // Update M_reg and compute attn = M_blk @ q_t
            const float d0 = smem_Delta_f32[i_v0][0];
            const float d1 = smem_Delta_f32[i_v1][0];
            float attn0 = 0.0f;
            float attn1 = 0.0f;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int j_base = 32 * warp + 8 * m + (lane & 2);
                const float k0 = smem_k_tail[j_base + 0];
                const float k1 = smem_k_tail[j_base + 1];
                const float k4 = smem_k_tail[j_base + 4];
                const float k5 = smem_k_tail[j_base + 5];
                const float q0 = smem_q_tail[j_base + 0];
                const float q1 = smem_q_tail[j_base + 1];
                const float q4 = smem_q_tail[j_base + 4];
                const float q5 = smem_q_tail[j_base + 5];

                M_reg[m].x[0] = g_val * M_reg[m].x[0] + k0 * d0;
                M_reg[m].x[1] = g_val * M_reg[m].x[1] + k1 * d0;
                M_reg[m].x[4] = g_val * M_reg[m].x[4] + k4 * d0;
                M_reg[m].x[5] = g_val * M_reg[m].x[5] + k5 * d0;
                attn0 += M_reg[m].x[0] * q0 + M_reg[m].x[1] * q1 + M_reg[m].x[4] * q4 + M_reg[m].x[5] * q5;

                M_reg[m].x[2] = g_val * M_reg[m].x[2] + k0 * d1;
                M_reg[m].x[3] = g_val * M_reg[m].x[3] + k1 * d1;
                M_reg[m].x[6] = g_val * M_reg[m].x[6] + k4 * d1;
                M_reg[m].x[7] = g_val * M_reg[m].x[7] + k5 * d1;
                attn1 += M_reg[m].x[2] * q0 + M_reg[m].x[3] * q1 + M_reg[m].x[6] * q4 + M_reg[m].x[7] * q5;
            }
            attn0 += __shfl_xor_sync(0xFFFFFFFFU, attn0, 2);
            attn1 += __shfl_xor_sync(0xFFFFFFFFU, attn1, 2);
            if ((lane & 2) == 0) {
                smem_part[warp][i_v0][0] = attn0;
                smem_part[warp][i_v1][0] = attn1;
            }
            __syncthreads();

            if (tid < V_TILE) {
                const float attn_c = smem_part[0][tid][0] + smem_part[1][tid][0] +
                                     smem_part[2][tid][0] + smem_part[3][tid][0];
                attn_data[t * S_v * H + (col0 + tid)] = attn_c * scale;
            }

            if constexpr (keep_rs_t) {
                const int target_slot = (int) (n_tokens - 1 - t);
                if (target_slot >= 0 && target_slot < K) {
                    float * slot_state = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
#pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            const int i_v = M_reg[m].get_i(l);
                            const int j_k = 32 * warp + 8 * m + M_reg[m].get_j(l);
                            slot_state[(col0 + i_v) * S_v + j_k] = M_reg[m].x[l];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int i_v = M_reg[m].get_i(l);
                const int j_k = 32 * warp + 8 * m + M_reg[m].get_j(l);
                state[(col0 + i_v) * S_v + j_k] = M_reg[m].x[l];
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, curr_state, dst, state, H, n_tokens, n_seqs,
                     sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic,
                     scale, state_slot_stride, K);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

template <bool keep_rs_t>
void launch_gated_delta_net_volta_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    dim3 grid_dims(H, n_seqs, 4); // 128 / 32 = 4 blocks per (head, seq)
    dim3 block_dims(32, 4, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_volta_chunked_kernel<keep_rs_t>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
}

template void launch_gated_delta_net_volta_chunked<false>(
        const float *, const float *, const float *,
        const float *, const float *, const float *,
        float *, float *,
        int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t,
        int64_t, int64_t,
        float, int64_t, int, cudaStream_t);

template void launch_gated_delta_net_volta_chunked<true>(
        const float *, const float *, const float *,
        const float *, const float *, const float *,
        float *, float *,
        int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t,
        int64_t, int64_t,
        float, int64_t, int, cudaStream_t);
