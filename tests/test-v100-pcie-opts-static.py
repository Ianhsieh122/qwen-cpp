#!/usr/bin/env python3

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(haystack: str, needle: str, message: str) -> None:
    if needle not in haystack:
        raise AssertionError(message)


def main() -> None:
    mmvq = (ROOT / "ggml/src/ggml-cuda/mmvq.cu").read_text(encoding="utf-8")
    volta_block = mmvq.split("if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA)", 1)[1].split("if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_TURING)", 1)[0]
    for qtype, limit in (
        ("GGML_TYPE_Q2_K", 4),
        ("GGML_TYPE_Q3_K", 6),
        ("GGML_TYPE_Q4_K", 5),
        ("GGML_TYPE_Q5_K", 6),
        ("GGML_TYPE_Q6_K", 7),
    ):
        if not re.search(rf"case\s+{qtype}:\s*return\s+ne11\s*<=\s*{limit};", volta_block):
            raise AssertionError(f"mmvq.cu Volta threshold missing {qtype} <= {limit}")
    require(
        mmvq,
        "rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_x",
        "mmvq.cu mul_mat_vec_q lacks nrows_x bounds guard for multi-row Volta decode",
    )
    require(
        mmvq,
        "c_rows_per_block == 1 || uint32_t(row0 + i) < nrows_x",
        "mmvq.cu mul_mat_vec_q_moe lacks nrows_x bounds guard",
    )

    convert = (ROOT / "ggml/src/ggml-cuda/convert.cu").read_text(encoding="utf-8")
    for needle in (
        "load_uint32_b2",
        "dequantize_block_q2_K_f16",
        "dequantize_block_q3_K_f16",
    ):
        require(convert, needle, f"convert.cu lacks Volta FP16 coalesced dequantize helper {needle}")

    fattn_common = (ROOT / "ggml/src/ggml-cuda/fattn-common.cuh").read_text(encoding="utf-8")
    q8_body = fattn_common.split("static __device__ __forceinline__ void dequantize_V_q8_0", 1)[1].split("\n}", 1)[0]
    require(
        q8_body,
        "const half2 neg_1152 = __float2half2_rn(-1152.0f);",
        "fattn-common.cuh dequantize_V_q8_0<half, 4> must pre-subtract 1152.0f before multiplying by d",
    )
    require(
        q8_body,
        "__hmul2(__hadd2(reinterpret_cast<const half2 &>(p0), neg_1152), d)",
        "fattn-common.cuh dequantize_V_q8_0<half, 4> lost pre-multiply subtraction order",
    )

    fattn_mma = (ROOT / "ggml/src/ggml-cuda/fattn-mma-f16.cuh").read_text(encoding="utf-8")
    volta_mma = fattn_mma.split("ggml_cuda_fattn_mma_get_config_volta(", 1)[1].split("ggml_cuda_fattn_mma_get_config_rdna(", 1)[0]
    for ncols in (8, 16, 32, 128):
        if not re.search(rf"GGML_CUDA_FATTN_MMA_CONFIG_CASE\(\s*256,\s*256,\s*{ncols},\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*1,\s*true\)", volta_mma):
            raise AssertionError(f"Volta MMA (256, 256, {ncols}) must set nstages_target = 1")
    if not re.search(r"GGML_CUDA_FATTN_MMA_CONFIG_CASE\(\s*256,\s*256,\s*64,\s*256,\s*1,\s*64,\s*64,\s*64,\s*64,\s*1,\s*false\)", volta_mma):
        raise AssertionError("Volta MMA (256, 256, 64) must use tuned 256-thread 64-tile config")

    fattn = (ROOT / "ggml/src/ggml-cuda/fattn.cu").read_text(encoding="utf-8")
    require(fattn, "gqa_ratio_eff=2", "fattn.cu Volta GQA routing comment should match effective 2-col tile")
    require(fattn, "volta_mma_gqa_eff < 32", "fattn.cu Volta m8n8k4 MMA gate must enforce >= 32 cols")
    require(
        fattn,
        "DKQ == 256 && DV == 256 && gqa_ratio > 4 && Q->ne[1] <= 16",
        "fattn.cu Volta MMA switch must enable padded ncols2=8 for D=256 GQA>4 small batches",
    )

    fattn_vec = (ROOT / "ggml/src/ggml-cuda/fattn-vec.cuh").read_text(encoding="utf-8")
    require(fattn_vec, "flash_attn_ext_vec_gqa", "fattn-vec.cuh missing GQA-sharing Split-K decode kernel")
    require(fattn_vec, "launch_fattn<D, 1, 6>", "fattn-vec.cuh missing D=256 GQA-6 launch_fattn dispatch")

    gdn_cu = (ROOT / "ggml/src/ggml-cuda/gated_delta_net.cu").read_text(encoding="utf-8")
    require(gdn_cu, "const int prefix_end = n_tokens > K", "gated_delta_net.cu missing prefix/tail snapshot loop split")
    require(gdn_cu, "gated_delta_net_cuda<128, false, keep_rs_t, true>", "gated_delta_net.cu missing S_v=128 float4 vectorization")

    delta_base = (ROOT / "src/models/delta-net-base.cpp").read_text(encoding="utf-8")
    require(delta_base, "!cparams.fused_gdn_ch && n_seq_tokens > K", "delta-net-base.cpp missing chunked-prefix fallback under rollback")

    cuda_cu = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    require(cuda_cu, "static bool ggml_cuda_p2p_requested()", "ggml-cuda.cu must provide ggml_cuda_p2p_requested()")
    if cuda_cu.count("ggml_cuda_p2p_requested()") < 3:
        raise AssertionError("ggml-cuda.cu must gate both peer-access enable and P2P copy checks via ggml_cuda_p2p_requested()")
    p2p_block = cuda_cu.split("cudaDeviceEnablePeerAccess(id_other, 0);", 1)[1].split("return info;", 1)[0]
    require(p2p_block, "GGML_LOG_WARN(", "cudaDeviceEnablePeerAccess failures must warn rather than abort")
    require(p2p_block, "(void) cudaGetLastError();", "cudaDeviceEnablePeerAccess failure must clear CUDA last error")

    # --split-act-type plumbing and backend pinning
    require((ROOT / "include/llama.h").read_text(encoding="utf-8"), "enum ggml_type split_act_type;", "llama.h missing split_act_type")
    require((ROOT / "src/llama-cparams.h").read_text(encoding="utf-8"), "ggml_type split_act_type = GGML_TYPE_F32;", "llama-cparams.h missing split_act_type")
    graph_h = (ROOT / "src/llama-graph.h").read_text(encoding="utf-8")
    if not re.search(r"cparams\.split_act_type\s+==\s+other\.cparams\.split_act_type", graph_h):
        raise AssertionError("llama-graph.h allow_reuse must check split_act_type")
    require((ROOT / "common/arg.cpp").read_text(encoding="utf-8"), '"--split-act-type"', "arg.cpp missing --split-act-type")

    ctx_cpp = (ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
    for needle in (
        "ggml_backend_supports_op",
        "ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get())",
        "model.dev_layer(il)",
        "model.dev_output()",
        "split_compress",
        "split_decompress",
    ):
        require(ctx_cpp, needle, f"llama-context.cpp missing split_act_type wiring: {needle}")
    if not re.search(r"/\*\.split_act_type\s+=\*/\s+GGML_TYPE_F32,", ctx_cpp):
        raise AssertionError("llama-context.cpp missing split_act_type default initializer")

    qwen35 = (ROOT / "src/models/qwen35.cpp").read_text(encoding="utf-8")
    for needle in ("maybe_compress_boundary", "split_cont", "split_compress", "split_decompress", "dev_output()"):
        require(qwen35, needle, f"qwen35.cpp missing layer-boundary compression wiring: {needle}")

    # Multi-V100 scripts & docs
    for rel, needles in (
        ("scripts/test_speculative_27b.py", ("--split-act-type", "--split-mode", "--gpus", "--backend-sampling", "--p2p")),
        ("scripts/benchmark_qwen38_27b_q3kl.py", ("--split-act-type", "--split-mode", "--gpus", "--backend-sampling", "--p2p")),
        ("scripts/test_prompt_cache_and_restore.py", ("--split-act-type", "--gpus", "--backend-sampling", "--p2p")),
        ("docs/beellama-args.md", ("--split-act-type", "GGML_CUDA_P2P", "--backend-sampling")),
        ("docs/beellama-features.md", ("--split-act-type", "GGML_CUDA_P2P", "PCIe 2.0 x1")),
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for needle in needles:
            require(text, needle, f"{rel} missing expected multi-V100 option/documentation: {needle}")

    # Phase 2: 4-col/warp GDN + Volta sm_70 FP16 Tensor-Core chunked WY/UT GDN
    gdn_cu = (ROOT / "ggml/src/ggml-cuda/gated_delta_net.cu").read_text(encoding="utf-8")
    for needle in ("K_COLS_PER_WARP", "warp_reduce_sum<sub_warp>", "launch_gated_delta_net_volta_chunked", "GGML_CUDA_GDN_VOLTA_MMA"):
        require(gdn_cu, needle, f"gated_delta_net.cu missing Phase 2 optimization: {needle}")
    gdn_volta = (ROOT / "ggml/src/ggml-cuda/gated-delta-net-volta.cu").read_text(encoding="utf-8")
    for needle in ("VOLTA_MMA_AVAILABLE", "get_half2(M_reg[m])", "gated_delta_net_volta_chunked_kernel"):
        require(gdn_volta, needle, f"gated-delta-net-volta.cu missing expected Volta MMA chunked GDN code: {needle}")

    # Phase 3: MTP FP16 h_nextn staging, on-device multi-ubatch staging, sched_reserve PP priming & capped verify_h copy
    spec_cpp = (ROOT / "common/speculative.cpp").read_text(encoding="utf-8")
    for needle in ("llama_get_embeddings_nextn_f16", "llama_set_embeddings_nextn_inp_f16", "keep_rows", "use_h_f16"):
        require(spec_cpp, needle, f"speculative.cpp missing Phase 3 MTP optimization: {needle}")
    for needle in ("LLAMA_MTP_H_FP16", "h_nextn_f16", "h_nextn_dev_alloc", "n_outputs_pp_final"):
        require(ctx_cpp, needle, f"llama-context.cpp missing Phase 3 MTP optimization: {needle}")
    require(graph_h, "embeddings_nextn_fp16", "llama-graph.h allow_reuse missing embeddings_nextn_fp16 check")

    # Phase 4: Volta prmt/lop3 Q4_K f16 dequant, cuBLAS src1 F16 cache, chunked CUDA graphs, hoisted row-outer MMVQ
    require(convert, "lop3.b32 %0, %0, 0x000f000f, 0x64006400, 0xea;", "convert.cu dequantize_block_q4_K_f16 missing Volta lop3 unpacker")
    require(cuda_cu, "GGML_CUDA_CUBLAS_CACHE_SRC1", "ggml-cuda.cu missing cuBLAS src1 F16 cache")
    require(cuda_cu, "GGML_CUDA_GRAPH_CHUNK_SIZE", "ggml-cuda.cu missing chunked CUDA graph support")
    require(mmvq, "kbx_row_offsets[rows_per_cuda_block]", "mmvq.cu missing hoisted branchless row offsets for multi-col CSE")

    # Batch A: dynamic 2-copy pipeline_parallel + hazard guards, Q8_0/Q4_0 GQA-6 D=256 FA vec decode
    backend_cpp = (ROOT / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
    for needle in (
        "n_copies_parallel = std::max(2, std::min(n_non_cpu_backends, GGML_SCHED_MAX_COPIES))",
        "ggml_backend_event_wait(input_backend, sched->events[split_backend_id][sched->cur_copy])",
        "pipeline-parallel graph reallocation forced backend sync",
        "for (int i = 0; i < sched->n_graph_inputs; i++)",
    ):
        require(backend_cpp, needle, f"ggml-backend.cpp missing Batch A pipeline_parallel optimization/guard: {needle}")
    require(ctx_cpp, "pipeline_parallel = %s", "llama-context.cpp missing pipeline_parallel status in sched_reserve log")
    for needle in (
        "flash_attn_vec_gqa_load_row",
        "type_KV == GGML_TYPE_Q8_0",
        "type_KV == GGML_TYPE_Q4_0",
        "need_f16_K, need_f16_V",
    ):
        require(fattn_vec, needle, f"fattn-vec.cuh missing Batch A Q8_0/Q4_0 GQA-6 decode support: {needle}")
    require(fattn, "gqa6_vec_kv_supported", "fattn.cu missing Q8_0/Q4_0 GQA-6 vector routing")

    # v0.1.1: P0 + P1 Qwen 3.8 27B PP/TG/long-context optimizations
    require(qwen35, "t->ne[1] < 16", "qwen35.cpp must gate split_act_type on n_tokens >= 16")
    require(convert, "dequant_iq4nl_pair_f16_volta", "convert.cu missing Volta IQ4_XS prmt F16 unpacker")
    require(fattn_vec, "half2  Q_reg[GQA][h2_per_lane]", "fattn-vec.cuh missing Volta half2 Q_reg GQA decode path")
    require(fattn_vec, "exp2f", "fattn-vec.cuh missing exp2f log2-space softmax")
    require(mmvq, "vec_dot_q5_K_q8_1_multi_vdr2", "mmvq.cu missing multi-col Q5_K unpack-once kernel")
    require(mmvq, "vec_dot_iq4_xs_q8_1_multi_vdr4", "mmvq.cu missing multi-col IQ4_XS unpack-once kernel")
    require(gdn_volta, "smem_exp_g", "gated-delta-net-volta.cu missing precomputed chunk exp(G) table")
    require(gdn_volta, "smem_decay", "gated-delta-net-volta.cu missing precomputed decay table")
    require(cuda_cu, "32 * 1024 * 1024", "ggml-cuda.cu missing 32 MiB cuBLAS src1 F16 cache")
    require(cuda_cu, "ggml_backend_fill_causal_mask_f16_async", "ggml-cuda.cu missing GPU causal mask fill proc")
    ggml_h = (ROOT / "ggml/include/ggml.h").read_text(encoding="utf-8")
    kv_cache_cpp = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
    require(ggml_h, "GGML_CAUSAL_MASK_MAGIC", "ggml.h missing GGML_CAUSAL_MASK_MAGIC")
    require(kv_cache_cpp, "GGML_CAUSAL_MASK_MAGIC", "llama-kv-cache.cpp missing deferred causal mask path")
    require(backend_cpp, "GGML_CAUSAL_MASK_MAGIC", "ggml-backend.cpp missing GPU causal mask dispatch")

    # DFlash2 slow-PCIe + async dual-GPU peer stage + Volta MMVQ ncols<=8
    require(cuda_cu, "GGML_CUDA_ASYNC_PEER_STAGE", "ggml-cuda.cu missing GGML_CUDA_ASYNC_PEER_STAGE")
    require(cuda_cu, "peer_copy_stream", "ggml-cuda.cu missing async peer_copy_stream")
    require(mmvq, "GGML_CUDA_VOLTA_MMVQ_MAX_BATCH", "mmvq.cu missing GGML_CUDA_VOLTA_MMVQ_MAX_BATCH")
    require(ctx_cpp, "embeddings_layer_inp_fp16", "llama-context.cpp missing embeddings_layer_inp_fp16")
    require(ctx_cpp, "embeddings_layer_inp_tail", "llama-context.cpp missing embeddings_layer_inp_tail")
    require(spec_cpp, "LLAMA_LAYER_INP_FP16", "speculative.cpp missing LLAMA_LAYER_INP_FP16")
    require(spec_cpp, "LLAMA_DFLASH_SWA_SKIP", "speculative.cpp missing LLAMA_DFLASH_SWA_SKIP")
    dflash_cpp = (ROOT / "src/models/dflash.cpp").read_text(encoding="utf-8")
    require(dflash_cpp, "embeddings_dflash_inp_f16", "dflash.cpp missing embeddings_dflash_inp_f16")

    # Runtime pipeline-parallel full-n_kv re-prime & server unified-KV prompt-cache guards
    require(backend_cpp, "ggml_backend_sched_get_n_reallocs", "ggml-backend.cpp missing ggml_backend_sched_get_n_reallocs")
    require(ctx_cpp, "ggml_backend_sched_get_n_reallocs", "llama-context.cpp missing full-n_kv pipeline re-prime")
    srv_ctx_cpp = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
    require(srv_ctx_cpp, "prompt_cache.can_admit", "server-context.cpp missing prompt_save pre-copy admission check")
    require(srv_ctx_cpp, "try_clear_idle_slots(ret->id)", "server-context.cpp missing unified-KV prompt_load idle-slot retry")


if __name__ == "__main__":
    main()
