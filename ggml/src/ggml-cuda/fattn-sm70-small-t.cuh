#pragma once

#include "common.cuh"

// Volta (sm_70) native 8-warp key-split small-T FlashAttention for D=256.
// Consumes F16, Q8_0, and Q4_0 K/V caches directly (zero full-KV F16 materialization)
// via Volta mma.sync.aligned.m8n8k4 tensor cores with PRMT int8/int4 -> fp16 dequantization.
bool ggml_cuda_flash_attn_ext_sm70_small_t_supported(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_sm70_small_t(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
