#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool silent_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static void init_dummy_tensors(struct ggml_tensor * tensor, void * /*userdata*/) {
    const int64_t ne = ggml_nelements(tensor);
    const bool is_ssm_a = std::strstr(tensor->name, "ssm_a") != nullptr;
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne, is_ssm_a ? -0.1f : 0.01f);
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne, ggml_fp32_to_fp16(is_ssm_a ? -0.1f : 0.01f));
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected synthetic tensor type");
    }
}

static gguf_context_ptr make_qwen35_gguf() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN35, ret.get());

    const uint32_t n_vocab     = 128;
    const uint32_t n_ctx       = 128;
    const uint32_t n_embd      = 256;
    const uint32_t n_head      = 2;
    const uint32_t n_head_kv   = 2;
    const uint32_t n_embd_head = n_embd / n_head;
    const uint32_t n_ff        = 384;
    const uint32_t n_layer     = 2;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,          llm_arch_name(LLM_ARCH_QWEN35));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                    n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,                n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,              n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                   n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,           n_ff);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,       uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,          n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,       n_head_kv);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,   1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,       std::vector<uint32_t>({n_embd_head / 4, n_embd_head / 4, n_embd_head / 4, n_embd_head / 4}));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,                uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,               uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,                uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,            n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,               uint32_t(2));

    std::vector<std::string> tokenizer_list(n_vocab);
    std::vector<float>       tokenizer_scores(n_vocab, 0.0f);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "test");
    for (uint32_t i = 0; i < n_vocab; ++i) {
        tokenizer_list[i] = "tok_" + std::to_string(i);
    }
    ms.add_kv(LLM_KV_TOKENIZER_LIST, tokenizer_list);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);

    return ret;
}

static bool parse_args(const std::vector<std::string> & args, common_params & out) {
    std::vector<std::string> copy = args;
    std::vector<char *> argv_ptrs;
    argv_ptrs.reserve(copy.size());
    for (std::string & s : copy) {
        argv_ptrs.push_back(s.data());
    }
    return common_params_parse((int) argv_ptrs.size(), argv_ptrs.data(), out, LLAMA_EXAMPLE_COMMON);
}

static void test_cli_parsing_and_defaults() {
    const llama_context_params cparams_default = llama_context_default_params();
    assert(cparams_default.split_act_type == GGML_TYPE_F32);

    common_params defaults;
    assert(defaults.split_act_type == GGML_TYPE_F32);
    assert(common_context_params_to_llama(defaults).split_act_type == GGML_TYPE_F32);

    struct valid_case {
        const char * flag_val;
        ggml_type    expected;
    };
    for (const valid_case & tc : {
            valid_case{ "none", GGML_TYPE_F32 },
            valid_case{ "f32",  GGML_TYPE_F32 },
            valid_case{ "F32",  GGML_TYPE_F32 },
            valid_case{ "f16",  GGML_TYPE_F16 },
            valid_case{ "F16",  GGML_TYPE_F16 },
            valid_case{ "bf16", GGML_TYPE_BF16 },
            valid_case{ "BF16", GGML_TYPE_BF16 },
            valid_case{ "q8_0", GGML_TYPE_Q8_0 },
            valid_case{ "Q8_0", GGML_TYPE_Q8_0 },
        }) {
        common_params parsed;
        const bool ok = parse_args({ "test-split-act", "-m", "dummy.gguf", "--split-act-type", tc.flag_val }, parsed);
        assert(ok);
        assert(parsed.split_act_type == tc.expected);
        const llama_context_params lparams = common_context_params_to_llama(parsed);
        assert(lparams.split_act_type == tc.expected);
    }

    for (const char * bad_val : { "q4_0", "q4_k", "invalid" }) {
        common_params parsed;
        const bool ok = parse_args({ "test-split-act", "-m", "dummy.gguf", "--split-act-type", bad_val }, parsed);
        assert(!ok);
    }
}

static void test_context_validation_and_decode() {
    gguf_context_ptr gguf_ctx = make_qwen35_gguf();
    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = silent_progress;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), init_dummy_tensors, nullptr, mparams));
    assert(model != nullptr);

    // Invalid split_act_type must be rejected cleanly by llama_init_from_model.
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx          = 64;
        cparams.n_batch        = 16;
        cparams.n_ubatch       = 16;
        cparams.n_threads      = 2;
        cparams.offload_kqv    = false;
        cparams.split_act_type = GGML_TYPE_Q4_0;
        llama_context_ptr bad_ctx(llama_init_from_model(model.get(), cparams));
        assert(bad_ctx == nullptr);
    }

    // All valid split_act_type modes must initialize and decode cleanly on CPU.
    for (const ggml_type valid_type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0 }) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = 64;
        cparams.n_batch         = 16;
        cparams.n_ubatch        = 16;
        cparams.n_threads       = 2;
        cparams.n_threads_batch = 2;
        cparams.offload_kqv     = false;
        cparams.split_act_type  = valid_type;

        llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
        assert(ctx != nullptr);

        common_batch batch(ctx.get());
        batch.add(1, 0, 0, true);
        batch.add(2, 1, 0, true);
        const int32_t rc = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
        assert(rc == 0);
    }
}

int main() {
    test_cli_parsing_and_defaults();
    test_context_validation_and_decode();
    std::printf("test-split-act: ok\n");
    return 0;
}
