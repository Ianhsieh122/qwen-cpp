#pragma once

#include <cstdint>
#include <cstddef>

namespace qwen38_27b {
    // Model architecture geometry
    constexpr int64_t N_EMBD            = 5120;
    constexpr int64_t N_LAYER           = 64;
    constexpr int64_t N_FF              = 17408;
    constexpr int64_t N_CTX_TRAIN       = 262144;
    constexpr int64_t N_VOCAB           = 152064;

    // Macro-block structure: 16 blocks of 3x Gated DeltaNet + 1x Gated Attention
    constexpr int64_t FULL_ATTN_INTERVAL = 4;
    constexpr int64_t N_MACRO_BLOCKS     = 16;
    constexpr int64_t N_RECR_PER_BLOCK   = 3;
    constexpr int64_t N_ATTN_PER_BLOCK   = 1;

    // Compile-time check for layer type (true = linear/recurrent Gated DeltaNet, false = full Gated Attention)
    constexpr bool is_recurrent_layer(int il) {
        return (il >= 0 && il < N_LAYER) && ((il + 1) % FULL_ATTN_INTERVAL != 0);
    }

    // Gated Attention parameters
    constexpr int64_t ATTN_N_HEAD       = 24;      // Query heads
    constexpr int64_t ATTN_N_HEAD_KV    = 4;       // KV heads (GQA 6:1)
    constexpr int64_t ATTN_HEAD_DIM     = 256;     // Head dimension
    constexpr int64_t ATTN_ROPE_DIM     = 64;      // RoPE dimension
    constexpr int64_t ATTN_Q_DIM        = ATTN_N_HEAD * ATTN_HEAD_DIM;       // 6144
    constexpr int64_t ATTN_KV_DIM       = ATTN_N_HEAD_KV * ATTN_HEAD_DIM;    // 1024

    // Gated DeltaNet (Linear Attention) parameters
    constexpr int64_t DELTA_NUM_V_HEADS = 48;      // Heads for V
    constexpr int64_t DELTA_NUM_K_HEADS = 16;      // Heads for QK
    constexpr int64_t DELTA_HEAD_DIM    = 128;     // Head dimension
    constexpr int64_t DELTA_D_CONV      = 4;       // Conv kernel size
    constexpr int64_t DELTA_KEY_DIM     = DELTA_NUM_K_HEADS * DELTA_HEAD_DIM;  // 2048
    constexpr int64_t DELTA_VALUE_DIM   = DELTA_NUM_V_HEADS * DELTA_HEAD_DIM;  // 6144
    constexpr int64_t DELTA_CONV_DIM    = DELTA_KEY_DIM * 2 + DELTA_VALUE_DIM; // 10240
    constexpr int64_t DELTA_D_INNER     = DELTA_VALUE_DIM;                    // 6144
    constexpr int64_t DELTA_DT_RANK     = DELTA_NUM_V_HEADS;                  // 48
    constexpr int64_t DELTA_N_GROUP     = DELTA_NUM_K_HEADS;                  // 16

    // Pipeline Parallelism split boundaries (2x 16GB Tesla V100)
    constexpr int PP_SPLIT_LAYER        = 32;      // GPU 0: Layers 0..31, GPU 1: Layers 32..63 + Head
    constexpr int GPU0_NUM_LAYERS       = 32;
    constexpr int GPU1_NUM_LAYERS       = 32;

    // Multi-Token Prediction (MTP) Speculative parameters
    constexpr int64_t MTP_K             = 4;       // Speculative draft window
    constexpr size_t  MTP_TOKEN_BYTES   = MTP_K * sizeof(int32_t);             // 16 bytes
    constexpr size_t  MTP_ACT_BYTES     = MTP_K * N_EMBD * sizeof(uint16_t);   // 40,960 bytes (~40 KB)
}
