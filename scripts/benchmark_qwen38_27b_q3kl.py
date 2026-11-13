import argparse
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import subprocess
import time
import urllib.error
import urllib.request

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    r"C:\Users\ianhs\OneDrive\30BB~1\lammacpp\models\Qwen3.8-27B-UD-Q3_K_XL.gguf",
)
CACHE_PATH = "./tmp_cache"
LOG_PATH = "./server_27b.log"
DEFAULT_PORT = 8080


def resolve_server_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_bin = os.environ.get("LLAMA_SERVER_BIN")
    if env_bin:
        return env_bin
    candidates = [
        "build-win-cuda-v100/bin/llama-server.exe",
        "build-win-cuda-v100/bin/Release/llama-server.exe",
        "build/bin/llama-server.exe",
        "build/bin/llama-server",
        "build-cpu/bin/llama-server",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if sys.platform == "win32" else "build/bin/llama-server"


def make_request(base_url: str, method: str, endpoint: str, data=None):
    url = f"{base_url}{endpoint}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode("utf-8") if data is not None else None
    try:
        with urllib.request.urlopen(req, data=body, timeout=180) as resp:
            resp_body = resp.read().decode("utf-8")
            return resp.status, json.loads(resp_body) if resp_body else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, {"error": err_body}
    except Exception as e:
        return 0, {"error": str(e)}


def wait_for_server(base_url: str, timeout: int = 180) -> bool:
    start = time.time()
    print("[BENCH] Waiting for Qwen 3.8 27B server to load and become healthy...")
    while time.time() - start < timeout:
        status, _ = make_request(base_url, "GET", "/health")
        if status == 200:
            print("[BENCH] Server is healthy and ready!")
            return True
        time.sleep(2)
    return False


def get_gpu_vram(cuda_visible_devices: str) -> list[tuple[str, int, int]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            universal_newlines=True,
        )
        wanted = {g.strip() for g in cuda_visible_devices.split(",") if g.strip()}
        stats: list[tuple[str, int, int]] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            idx, used_s, total_s = parts
            if not wanted or idx in wanted:
                stats.append((idx, int(used_s), int(total_s)))
        return stats
    except Exception:
        return []


def print_vram_summary(prefix: str, cuda_visible_devices: str) -> None:
    stats = get_gpu_vram(cuda_visible_devices)
    if not stats:
        print(f"\n[{prefix}] nvidia-smi unavailable\n")
        return
    print(f"\n[{prefix}] GPU Memory:")
    total_used = 0
    total_cap = 0
    for idx, used, cap in stats:
        total_used += used
        total_cap += cap
        pct = (used / cap * 100.0) if cap > 0 else 0.0
        print(f"  GPU {idx}: {used} MiB / {cap} MiB ({pct:.1f}% used, {cap - used} MiB free)")
    if len(stats) > 1 and total_cap > 0:
        print(f"  Total: {total_used} MiB / {total_cap} MiB ({total_used / total_cap * 100.0:.1f}% used)")
    print()


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    multi_gpu = len(gpu_list) > 1
    cmd = [
        resolve_server_bin(args.server_bin),
        "-m", args.model,
        "-ngl", str(args.ngl),
        "-c", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v,
        "--flash-attn", "on",
        "--slot-save-path", CACHE_PATH,
        "--verbose",
        "--port", str(args.port),
    ]
    if args.backend_sampling:
        cmd.append("--backend-sampling")
    if multi_gpu:
        cmd.extend(["--split-mode", "layer"])
        if args.tensor_split:
            cmd.extend(["--tensor-split", args.tensor_split])
        if args.split_act_type and args.split_act_type != "none":
            cmd.extend(["--split-act-type", args.split_act_type])
    elif args.split_act_type and args.split_act_type not in ("none", "f32"):
        cmd.extend(["--split-act-type", args.split_act_type])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen 3.8 27B Q3_K_XL on single or multi-V100.")
    parser.add_argument("--server-bin", default=None, help="Path to llama-server binary.")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, help="Target GGUF model path.")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
                        help="Comma-separated CUDA device IDs (default: 0,1).")
    parser.add_argument("-ngl", type=int, default=99, help="Target GPU layers (default: 99).")
    parser.add_argument("-ts", "--tensor-split", default="1,1",
                        help="Tensor split ratios when multiple GPUs are selected (default: 1,1).")
    parser.add_argument("--split-act-type", default="q8_0",
                        choices=["none", "f32", "f16", "bf16", "q8_0"],
                        help="Cross-GPU boundary activation compression for --split-mode layer (default: q8_0).")
    parser.add_argument("--backend-sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep sampling on the output GPU to avoid vocab logits PCIe transfer (default: on).")
    parser.add_argument("--p2p", action="store_true",
                        help="Opt in to GGML_CUDA_P2P=1 (unsafe on some IOMMU/ACS BIOS setups).")
    parser.add_argument("-c", "--ctx-size", type=int, default=8192)
    parser.add_argument("-b", "--batch-size", type=int, default=2048)
    parser.add_argument("-ub", "--ubatch-size", type=int, default=1024)
    parser.add_argument("--cache-type-k", default="kvarn4")
    parser.add_argument("--cache-type-v", default="kvarn4")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--keep-alive", action="store_true", help="Leave llama-server running after benchmark.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(CACHE_PATH, exist_ok=True)
    if os.path.exists(LOG_PATH):
        try:
            os.remove(LOG_PATH)
        except Exception:
            pass

    cmd = build_server_cmd(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.p2p:
        env["GGML_CUDA_P2P"] = "1"
    else:
        env.pop("GGML_CUDA_P2P", None)

    base_url = f"http://127.0.0.1:{args.port}"
    print("=" * 65)
    print(f"LAUNCHING QWEN 3.8 27B (Q3_K_XL) ON GPUS [{args.gpus}]")
    print("=" * 65)
    print(f"Model:   {args.model}")
    print(f"P2P:     {'GGML_CUDA_P2P=1 (opt-in)' if args.p2p else 'off (host-staged)'}")
    print(f"Command: {' '.join(cmd)}\n")

    log_file = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        universal_newlines=True,
    )

    try:
        if not wait_for_server(base_url, 180):
            print("[FAIL] Server failed to start within timeout.")
            log_file.flush()
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                print(f.read()[-3000:])
            return

        print_vram_summary("VRAM CHECK", args.gpus)

        # Benchmark 1: Standard Prompt (500 tokens) + 64 decode tokens
        base_prompt_text = (
            "Analyze the fundamental tradeoffs between pipeline parallelism and tensor parallelism "
            "in low-bandwidth interconnected GPU clusters. Specifically examine the activation volume "
            "at stage boundaries compared to intermediate layer-wise All-Reduce communications, and explain "
            "why low-bandwidth interconnects like PCIe 2.0 x1 favor pipeline parallelism over tensor parallelism. "
        ) * 12

        prompt_1 = f"<|im_start|>system\nYou are an expert systems engineer.<|im_end|>\n<|im_start|>user\n{base_prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        print("[BENCHMARK 1] Sending 500-token prompt + 64 generation tokens...")
        t0 = time.time()
        status_1, resp_1 = make_request(base_url, "POST", "/completion", {
            "prompt": prompt_1,
            "n_predict": 64,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        t1 = time.time()
        assert status_1 == 200, f"Request 1 failed: {resp_1}"
        timings_1 = resp_1.get("timings", {})
        prompt_n_1 = timings_1.get("prompt_n", -1)
        prompt_ms_1 = timings_1.get("prompt_ms", -1.0)
        prompt_per_sec_1 = timings_1.get("prompt_per_second", -1.0)
        gen_n_1 = timings_1.get("predicted_n", -1)
        gen_ms_1 = timings_1.get("predicted_ms", -1.0)
        gen_per_sec_1 = timings_1.get("predicted_per_second", -1.0)
        content_1 = resp_1.get("content", "").strip()

        print("\n" + "-" * 65)
        print("BENCHMARK 1 RESULTS (500 Tokens):")
        print(f"  * Prompt Tokens (Prefill):     {prompt_n_1} tokens in {prompt_ms_1:.2f} ms")
        print(f"  * Prefill Throughput:          {prompt_per_sec_1:.2f} tok/s")
        print(f"  * Generation Tokens (Decode):  {gen_n_1} tokens in {gen_ms_1:.2f} ms")
        print(f"  * Decode Throughput:           {gen_per_sec_1:.2f} tok/s ({(gen_ms_1 / gen_n_1 if gen_n_1 > 0 else 0):.2f} ms/token)")
        print(f"  * Total Wall Clock Time:       {(t1 - t0) * 1000:.2f} ms")
        print(f"  * Response Sample:             {content_1[:120]}...")
        print("-" * 65)

        # Benchmark 2: Larger Prompt (1500 tokens) for High-Occupancy Prefill
        long_prompt_text = base_prompt_text * 3
        prompt_2 = f"<|im_start|>system\nYou are an expert systems engineer.<|im_end|>\n<|im_start|>user\n{long_prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        print("\n[BENCHMARK 2] Sending Large 1,500-token prompt for maximum prefill saturation...")
        status_2, resp_2 = make_request(base_url, "POST", "/completion", {
            "prompt": prompt_2,
            "n_predict": 32,
            "id_slot": 0,
            "cache_prompt": False,
            "temperature": 0.0,
        })
        assert status_2 == 200, f"Request 2 failed: {resp_2}"
        timings_2 = resp_2.get("timings", {})
        prompt_n_2 = timings_2.get("prompt_n", -1)
        prompt_ms_2 = timings_2.get("prompt_ms", -1.0)
        prompt_per_sec_2 = timings_2.get("prompt_per_second", -1.0)
        gen_n_2 = timings_2.get("predicted_n", -1)
        gen_ms_2 = timings_2.get("predicted_ms", -1.0)
        gen_per_sec_2 = timings_2.get("predicted_per_second", -1.0)

        print("\n" + "-" * 65)
        print("BENCHMARK 2 RESULTS (1,500 Tokens):")
        print(f"  * Prompt Tokens (Prefill):     {prompt_n_2} tokens in {prompt_ms_2:.2f} ms")
        print(f"  * Peak Prefill Throughput:     {prompt_per_sec_2:.2f} tok/s")
        print(f"  * Generation Tokens (Decode):  {gen_n_2} tokens in {gen_ms_2:.2f} ms")
        print(f"  * Decode Throughput:           {gen_per_sec_2:.2f} tok/s")
        print("-" * 65)

        print_vram_summary("FINAL VRAM STATE", args.gpus)

        print("=" * 65)
        if args.keep_alive:
            print(f"QWEN 3.8 27B BENCHMARK COMPLETE - SERVER WILL REMAIN LIVE ON PORT {args.port}")
        else:
            print("QWEN 3.8 27B BENCHMARK COMPLETE")
        print("=" * 65)
    finally:
        if not args.keep_alive:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        log_file.close()


if __name__ == "__main__":
    main()
