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
    for ncols in (8, 16, 32, 64, 128):
        if not re.search(rf"GGML_CUDA_FATTN_MMA_CONFIG_CASE\(\s*256,\s*256,\s*{ncols},\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+,\s*1,\s*true\)", volta_mma):
            raise AssertionError(f"Volta MMA (256, 256, {ncols}) must set nstages_target = 1")

    fattn = (ROOT / "ggml/src/ggml-cuda/fattn.cu").read_text(encoding="utf-8")
    require(fattn, "gqa_ratio_eff=2", "fattn.cu Volta GQA routing comment should match effective 2-col tile")
    require(fattn, "Q->ne[1] * gqa_ratio_eff < 32", "fattn.cu Volta m8n8k4 MMA gate must enforce >= 32 cols")

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


if __name__ == "__main__":
    main()
