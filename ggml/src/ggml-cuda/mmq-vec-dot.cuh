#pragma once

#include "vecdotq.cuh"
#include "mma.cuh"

using namespace ggml_cuda_mma;

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_0*VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_0_Q8_1_MMQ];

                constexpr int max_cpy = ggml_cuda_get_max_cpy_bytes();
                constexpr int mcpy_int = max_cpy / sizeof(int);
                static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

                int tmp0[4], tmp1[4];

                #pragma unroll
                for (int l0 = 0; l0 < 4 / mcpy_int; ++l0) {
                    ggml_cuda_memcpy_1<max_cpy>(tmp0 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + l0 * mcpy_int]  );
                    ggml_cuda_memcpy_1<max_cpy>(tmp1 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_0 + l0 * mcpy_int]);
                }

                u[0]=tmp0[0]; u[2]=tmp0[1]; u[4]=tmp0[2]; u[6]=tmp0[3];
                u[1]=tmp1[0]; u[3]=tmp1[1]; u[5]=tmp1[2]; u[7]=tmp1[3];

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                    (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_0], u,
                     x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + k0/(QR4_0*QI4_0)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_1_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_1*VDR_Q4_1_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_1_Q8_1_MMQ];

                constexpr int max_cpy = ggml_cuda_get_max_cpy_bytes();
                constexpr int mcpy_int = max_cpy / sizeof(int);
                static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

                int tmp0[4], tmp1[4];

                #pragma unroll
                for (int l0 = 0; l0 < 4 / mcpy_int; ++l0) {
                    ggml_cuda_memcpy_1<max_cpy>(tmp0 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + l0 * mcpy_int]  );
                    ggml_cuda_memcpy_1<max_cpy>(tmp1 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_1 + l0 * mcpy_int]);
                }

                u[0]=tmp0[0]; u[2]=tmp0[1]; u[4]=tmp0[2]; u[6]=tmp0[3];
                u[1]=tmp1[0]; u[3]=tmp1[1]; u[5]=tmp1[2]; u[7]=tmp1[3];

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                    (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_1], u,
                     x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + k0/(QR4_1*QI4_1)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K],
                     x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            float dB;
            const int j = j0 + tile_C::get_j(0);
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            } else {
                dB = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    const float dA = x_df[i*sram_stride + k0/QI8_0];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA*dB;
                }
            }
        }
    }
#else
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
    float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*sram_stride + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] =             y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_0], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}


template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

        // Row scale pair (d, m) is invariant over the j0 loop; load it once per element.
        float2 dmA_reg[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                dmA_reg[n][l] = __half22float2(x_dm[i*sram_stride + k0/QI8_1]);
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA_reg[n][l].x*dsB.x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA_reg[n][l].y*dsB.y;
                }
            }
        }
    }
#else
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*sram_stride + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

// Used for NVFP4, Q3_K, IQ2_S, and IQ2_XS
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0],
                    &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + k0/(QI8_0/2)],
                    y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

// Used for Q3_K, IQ2_S, and IQ2_XS:
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, prec_src1);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * x_df[i*sram_stride + k0/4] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, prec_src1);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/8], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*sram_stride + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    float2 y_df[J/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        y_df[j0/nwarps] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
    }

#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K/2; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                constexpr int ns = 2;
                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }
        }
    }

    // Some compilers fail to unroll the loop over k01 if there is a conditional statement for ns in the inner loop.
    // As a workaround 2 separate loops are used instead.
#pragma unroll
    for (int k01 = MMQ_TILE_NE_K/2; k01 < MMQ_TILE_NE_K; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                constexpr int ns = 1;
                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = (k01 < MMQ_TILE_NE_K/2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K]).x : __half22float2(y_ds[j*MMQ_TILE_Y_K]).y;
            const float sB = (k01 >= MMQ_TILE_NE_K * 3/4) ? 0
                                              : (((k01/4)%2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).y
                                                             : __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).x);

            tile_C Cm;
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
#pragma unroll
                for (int l = 0; l < tile_A::ne; ++l) {
                    A1.x[l] = 0x01010101;
                }
                mma(Cm, A1, B);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd;
                mma(Cd, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const float2 dm = __half22float2(x_dm[i*sram_stride + k0/4]);
                    float tmp = Cd.x[l]*dm.x;
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm.x[l]*dm.y;
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*dB;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= dm.y*sB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];
    float  mA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i*sram_stride + k0/(QI8_1/2)]);

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float2 dB[tile_C::ne/2];

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int j = j0 + tile_C::get_j(l);

            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B B[2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

            tile_C Cm[2];
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                mma(Cm[0], A1, B[0]);
                mma(Cm[1], A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd[2];

                mma(Cd[0], A[n][k01/4 + 0], B[0]);
                mma(Cd[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*(k01 < MMQ_TILE_NE_K/2 ? dB[l%2].x : dB[l%2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K * 3/4; k01 += QI8_1) {
            float2 sB[tile_C::ne/2];

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 0]*sB[l%2].x;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 1]*sB[l%2].y;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q3_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const int8_t * scales = ((const int8_t *) (x_sc + i*(MMQ_TILE_NE_K/8) + i/8)) + k0/4;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q3_K_q8_1_impl_mmq(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], scales,
                    x_df[i], y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

#if defined(VOLTA_MMA_AVAILABLE)
// Volta sm_70 in-register 4-bit nibble -> FP16 half2 unpacker using prmt.b32 + lop3.b32
// Unpacks 4 nibbles (at bit shift 0 or 4) from a 32-bit word into 2 half2 registers in {0.0h .. 15.0h} scaled by sc.
static __device__ __forceinline__ void ggml_cuda_volta_unpack_q4_4_scaled(
        const int v, const int shift, const half2 sc_h2, const half2 neg_1024_sc_h2, half2 & out0, half2 & out1) {
    const uint32_t q = (uint32_t)(v >> shift);
    const uint32_t magic = 0x64006400u; // IEEE-754 FP16 half2(1024.0f, 1024.0f)
    uint32_t p0 = __byte_perm(q, magic, 0x7170);
    uint32_t p1 = __byte_perm(q, magic, 0x7372);
    asm("lop3.b32 %0, %0, 0x000f000f, 0x64006400, 0xea;" : "+r"(p0));
    asm("lop3.b32 %0, %0, 0x000f000f, 0x64006400, 0xea;" : "+r"(p1));
    const half2 neg_1024 = __float2half2_rn(-1024.0f);
    // Exact integer nibble in {0.0h .. 15.0h}, then exact integer product nibble * sc in {0.0h .. 945.0h}
    out0 = __hmul2(__hadd2(reinterpret_cast<const half2 &>(p0), neg_1024), sc_h2);
    out1 = __hmul2(__hadd2(reinterpret_cast<const half2 &>(p1), neg_1024), sc_h2);
    GGML_UNUSED(neg_1024_sc_h2);
}

// Volta sm_70 in-register int8 -> FP16 half2 unpacker using prmt.b32 + XOR sign flip
static __device__ __forceinline__ void ggml_cuda_volta_unpack_q8_4(
        const int u, half2 & out0, half2 & out1) {
    const uint32_t x = (uint32_t)u ^ 0x80808080u;
    const uint32_t magic = 0x64006400u; // 1024.0h
    const uint32_t p0 = __byte_perm(x, magic, 0x7170);
    const uint32_t p1 = __byte_perm(x, magic, 0x7372);
    const half2 neg_1152 = __float2half2_rn(-1152.0f); // -(1024 + 128)
    out0 = __hadd2(reinterpret_cast<const half2 &>(p0), neg_1152);
    out1 = __hadd2(reinterpret_cast<const half2 &>(p1), neg_1152);
}
#endif // defined(VOLTA_MMA_AVAILABLE)

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

#if defined(VOLTA_MMA_AVAILABLE)
    // Volta sm_70 Tensor Core (mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32) Q4_K x Q8_1 kernel:
    // Each warp (threadIdx.y in 0..7) owns 16 rows: i_base = threadIdx.y * 16.
    // Within a 32-thread warp, the 4 hardware m8n8k4 quadpairs (qp = (threadIdx.x % 16) / 4) are split:
    //   - qp_row = qp & 1 (0: rows 0..7, 1: rows 8..15)
    //   - qp_k   = qp >> 1 (0: first 16 K-elements of 32-element sub-block, 1: second 16 K-elements)
    // This keeps 100% of Volta's 4 quadpair Tensor Cores active for every J (8, 16, 24, 32, ...).
    const int i_base  = threadIdx.y * 16;
    const int qp      = (threadIdx.x % 16) / 4;
    const int qp_row  = qp & 1;
    const int qp_k    = qp >> 1;
    const int half_w  = threadIdx.x / 16;
    const int tid_qp  = threadIdx.x % 4;

    // Row index in tile_x loaded by this thread for matrix A (m8n8k4 row-major):
    const int i_a = i_base + qp_row * 8 + half_w * 4 + tid_qp;
    // Column offset within an 8-column block j0 loaded by this thread for matrix B (m8n8k4 col-major):
    const int c_b = half_w * 4 + tid_qp;

    const float2 dm_a = __half22float2(x_dm[i_a]);

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += 8) {
        const int j_b = j0 + c_b;

        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
            const int k0 = k00 + k01;
            const uint8_t * sc_a = (const uint8_t *) &x_sc[i_a * (MMQ_TILE_NE_K/8) + i_a/8 + k0/32] + 2*(k01/16);
            const int * v_a = &x_qs[i_a * (MMQ_TILE_NE_K + 1) + k0/2];

            float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int shift = 4 * sub;
                const half2 sc_h2 = __float2half2_rn((float)sc_a[sub]);
                const half2 dummy = make_half2(0.0f, 0.0f);

                // Each sub-block has 32 elements = 8 ints.
                // qp_k == 0 processes ints 0..3 (16 elements = 4 m8n8k4 steps of k=4);
                // qp_k == 1 processes ints 4..7 (16 elements = 4 m8n8k4 steps of k=4).
                const int k_int_base = qp_k * 4;
                const int * u_b = &y_qs[j_b * MMQ_TILE_Y_K + k01 + 8 * sub + k_int_base];

                float c_sub[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    half2 a0, a1, b0, b1;
                    ggml_cuda_volta_unpack_q4_4_scaled(v_a[k_int_base + s], shift, sc_h2, dummy, a0, a1);
                    ggml_cuda_volta_unpack_q8_4(u_b[s], b0, b1);

                    const int & a0_i = reinterpret_cast<const int &>(a0);
                    const int & a1_i = reinterpret_cast<const int &>(a1);
                    const int & b0_i = reinterpret_cast<const int &>(b0);
                    const int & b1_i = reinterpret_cast<const int &>(b1);

                    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                        "{%0, %1, %2, %3, %4, %5, %6, %7};"
                        : "+f"(c_sub[0]), "+f"(c_sub[1]), "+f"(c_sub[2]), "+f"(c_sub[3]),
                          "+f"(c_sub[4]), "+f"(c_sub[5]), "+f"(c_sub[6]), "+f"(c_sub[7])
                        : "r"(a0_i), "r"(a1_i), "r"(b0_i), "r"(b1_i));
                }

                // Scale each column c_d(l) = (tid_qp & 2) + (l & 5) by d_B(j0 + c_d, sub):
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int c_d = (tid_qp & 2) + (l & 5);
                    const float d_b = __low2float(y_ds[(j0 + c_d) * MMQ_TILE_Y_K + k01/QI8_1 + sub]);
                    acc[l] += c_sub[l] * d_b;
                }
            }

            // Reduce across qp_k = 0 and qp_k = 1 (partner lane = threadIdx.x ^ 8):
            // Threads with qp_k == 0 keep columns 0..3 (l = 0..3);
            // Threads with qp_k == 1 keep columns 4..7 (l = 4..7).
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float mine  = (qp_k == 0) ? acc[l]     : acc[l + 4];
                const float other = (qp_k == 0) ? acc[l + 4] : acc[l];
                const float recv  = __shfl_xor_sync(0xFFFFFFFF, other, 8, 32);
                const float dot_sum = mine + recv;

                const int i_out = i_base + qp_row * 8 + half_w * 4 + (l & 2) + (tid_qp & 1);
                const int j_out = j0 + qp_k * 4 + (tid_qp & 2) + (l & 1);

                const float2 dm_out = __half22float2(x_dm[i_out]);
                const uint8_t * sc_out = (const uint8_t *) &x_sc[i_out * (MMQ_TILE_NE_K/8) + i_out/8 + k0/32] + 2*(k01/16);
                const half2 ds0 = y_ds[j_out * MMQ_TILE_Y_K + k01/QI8_1 + 0];
                const half2 ds1 = y_ds[j_out * MMQ_TILE_Y_K + k01/QI8_1 + 1];
                const float min_term = dm_out.y * (sc_out[8] * __high2float(ds0) + sc_out[9] * __high2float(ds1));

                sum[(j0 / 8) * 4 + l] += dm_out.x * dot_sum - min_term;
            }
        }
    }
    GGML_UNUSED(dm_a);
#else
// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = (const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/32] + 2*(k01/16);

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i*(MMQ_TILE_NE_K + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
#endif // defined(VOLTA_MMA_AVAILABLE)
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q5_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR5_K*VDR_Q5_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = ((const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k00/32]) + 2*(k01/16);

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q5_K_q8_1_impl_mmq(
                    &x_qs[i*(QR5_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR6_K*VDR_Q6_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const int8_t * sc = ((const int8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/16]);

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q6_K_q8_1_impl_mmq(
                    &x_qs[i*(QR6_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc,
                    x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    // Row base scales are invariant over the k01 and j0 loops; load them once.
    // Each thread owns fixed elements of the C tile, so one value per element suffices.
    float x_df_reg[ntx][tile_C::ne];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(l);
            x_df_reg[n][l] = x_df[i*sram_stride];
        }
    }

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

        // Sub-scales for this k01 chunk; invariant over the j0 loop.
        int8_t x_sc_reg[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                x_sc_reg[n][l] = ((const int8_t *) (x_sc + i*sram_stride + k00/16))[k01/4];
            }
        }

        // Fold the sub-scale and the row base scale into one f32 per element;
        // saves one int-multiply and one convert per element in the j0 loop.
        float x_s2_reg[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                x_s2_reg[n][l] = (float) x_sc_reg[n][l] * x_df_reg[n][l];
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += (float) C.x[l] * x_s2_reg[n][l] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A   A[ntx][8];
    int    scA[ntx][tile_C::ne/2][8];
    float   dA[ntx][tile_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/4 + 0], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + 0),         sram_stride);
            load_ldmatrix(A[n][k01/4 + 1], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + tile_A::J), sram_stride);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

                const int      sc_packed = x_sc[i*sram_stride + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

            dA[n][l] = x_df[i*sram_stride];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float tmp[ntx][tile_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0         + k01, MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------

// Shared MMA kernel for MXFP4 and NVFP4 on Blackwell.
// Both quantizations encode values as e2m1 (FP4) and produce one uint32 scale per
// m16n8k64 MMA call; only the PTX kind (scale_vec::2X ue8m0 vs scale_vec::4X ue4m3)
// and the per-type stride constant differ.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_fp4_fp4_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {

    typedef tile<16, 8, int>   tile_A;
    typedef tile<8,  8, int>   tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q4);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs = (const int *) x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int *      y_qs = (const int *) y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;

    // 2 threads per quad supply the packed scale register to the block_scale MMA,
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0     = (threadIdx.y / ntx) * rows_per_warp;

    tile_A   A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * sram_stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scaleB[nfrags];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            // accumulate in place into the output sum array
            tile_C & C = *reinterpret_cast<tile_C *>(sum + (j0 / tile_C::J + n) * tile_C::ne);
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                mma_block_scaled_fp4<type>(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
            }
        }
    }
}
