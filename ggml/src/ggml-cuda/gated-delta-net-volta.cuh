#pragma once

#include "common.cuh"

// Volta (sm_70) FP16 Tensor-Core (m8n8k4) chunked WY/UT Gated Delta Net kernel.
// Supports S_v == 128, !KDA, arbitrary n_tokens (chunk size BT = 16 with exact scalar tail),
// and both keep_rs == false (K <= 1) and keep_rs == true (K > 1 rollback snapshots).
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
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream);
