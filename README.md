# qwen-cpp

**qwen-cpp** is an independent, performance-focused GGUF inference engine optimized for **Qwen 3.5 / 3.8** hybrid architectures (Gated Delta Net + GQA-6 Full Attention), **NVIDIA Volta (`sm_70` / Tesla V100)** multi-GPU setups over slow PCIe links, and **MTP / DFlash speculative decoding** on heterogeneous GPUs.

`qwen-cpp` is built on top of [Anbeeld's BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) and [ggml-org's llama.cpp](https://github.com/ggml-org/llama.cpp).

---

## Key Additions in qwen-cpp

- **Volta (`sm_70`) Tensor-Core Gated Delta Net (`gated-delta-net-volta`)**: Chunked WY/UT FP16 `m8n8k4` Tensor-Core prefill kernel (`S_v = 128`) plus 4-column/warp GDN decode & tail kernels (`GGML_CUDA_GDN_VOLTA_MMA`), keeping long-context Qwen 3.5/3.8 prefill fast even under rollback snapshots.
- **Volta Long-Context GQA-6 Attention & Dequantization**:
  - Dedicated `D=256` GQA-6 split-K FP16 FlashAttention vector decode kernel and Volta `m8n8k4` MMA routing for small-batch verification (`Q->ne[1]` in `4..16`).
  - Fast `prmt`/`lop3` `Q4_K`/`Q3_K`/`Q2_K` → `F16` dequantization kernels for cuBLAS prefill.
  - Multi-column unpack-once `Q4_K`/`Q6_K` MMVQ (`ncols_dst = 2..4`), hoisted row-outer offsets, `cuBLAS` `src1` F16 cache (`GGML_CUDA_CUBLAS_CACHE_SRC1`), and chunked CUDA graphs (`GGML_CUDA_GRAPH_CHUNK_SIZE`).
- **PCIe-Constrained Multi-GPU Splitting**:
  - `--split-act-type`: Compresses cross-GPU residual activations (e.g. `q8_0`) between pipeline layer splits to cut transfer bottlenecks on PCIe 2.0/3.0 links.
  - Opt-in CUDA P2P via `GGML_CUDA_P2P` (prevents unexpected hangs/slowdowns across non-NVLink topologies) and `--backend-sampling`.
- **Heterogeneous MTP Speculative Decoding Optimizations**:
  - FP16 `h_nextn` hidden-state staging and capped trailing-window `verify_h` host copies for lower prompt-prefill overhead when pairing multi-GPU targets (e.g., 2× V100) with a dedicated draft GPU (e.g., RTX 3070).
  - Independent draft micro-batch sizing (`--spec-draft-ubatch-size` / `-ubd`) and `--spec-draft-backend-sampling`.

## Inherited Engine Capabilities (via BeeLlama.cpp & llama.cpp)

- **Variance-normalized KV-cache quantization (KVarN)**: `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, `kvarn8` for target caches and supported owned draft caches (`--spec-draft-type-k/v`).
- **KV cache precision tail (`--kv-tail-tokens`)**: Keeps most of the KV cache quantized while preserving recent tokens in F16/BF16, merged via a single FlashAttention softmax.
- **Standard low-bit KV cache formats**: `q2_0`, `q2_1`, `q3_0`, `q3_1`, `q6_0`, `q6_1` alongside upstream `q4`/`q5`/`q8`.
- **Speculative decoding & loop protection**: `draft-mtp`, `draft-dflash` (with profit-only `--spec-dm-controller profit`), `draft-eagle3`, n-gram modes, reasoning-loop guard, and transactional RAM prompt caching.

For the full inherited argument and feature reference, see [docs/beellama-features.md](docs/beellama-features.md) and [docs/beellama-args.md](docs/beellama-args.md).

---

## Building from Source

### Windows (CUDA 12.9 — Volta `sm_70` + Ampere `sm_86`)

```powershell
powershell -File scripts/build-win-cuda-v100.ps1 -AllTests
```

Pass `-CudaArch "70;86"` (default) or customize `-CudaArch` for your GPUs. Built binaries are placed in `build-win-cuda-v100\bin\`.

### Generic CMake Build

```bash
# Linux (CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Windows (MSVC + CUDA)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES="70;86" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# macOS (Metal)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Optional CUDA FlashAttention build flags:
- `-DGGML_CUDA_FA_ALL_QUANTS=ON`: Compiles all 169 standard and 36 KVarN FlashAttention quant pairs (longer compile time).
- `-DGGML_CUDA_KVARN=OFF`: Builds without KVarN kernels.

---

## Usage Examples

### Multi-V100 Server with PCIe Compression + Dedicated Draft GPU (MTP)

```bash
llama-server -m Qwen3.8-27B-Uncensored-Q4_K_M.gguf \
  --device CUDA0,CUDA1 --split-mode layer --main-gpu 0 --tensor-split 1.05,1 \
  --split-act-type q8_0 --backend-sampling \
  -c 163840 -b 8192 -ub 1024 -ctk f16 -ctv f16 -fa on \
  --model-draft Qwen3.8-27B-Uncensored-draft-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-device CUDA2 --spec-draft-ngl 99 \
  --spec-draft-n-max 3 --spec-draft-ubatch-size 1024 --spec-draft-backend-sampling \
  --host 0.0.0.0 --port 8888
```

### KVarN Quantized Target Cache

```bash
llama-server -m model.gguf --flash-attn on \
  --cache-type-k kvarn5 --cache-type-v kvarn4 \
  --kv-tail-tokens 1024
```

### Local CLI

```bash
llama-cli -m model.gguf
llama-server -m model.gguf --port 8080
```

---

## Documentation

- [Inherited engine features](docs/beellama-features.md)
- [Argument reference](docs/beellama-args.md)
- [Build guide](docs/build.md)
- [Server documentation](tools/server/README.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)

---

## Acknowledgments

`qwen-cpp` builds directly on the work of:
- **[Anbeeld / BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp)** — KVarN KV-cache quantization, KV precision tail, low-bit KV formats, reasoning-loop guard, adaptive speculative decoding, and transactional prompt caching.
- **[ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp)** — Core `ggml` tensor library, GGUF format, backend infrastructure, model implementations, and server architecture.

## License

Licensed under the **MIT License** — see [LICENSE](LICENSE). Copyright notices from the `ggml` authors and Anbeeld are preserved in full.
