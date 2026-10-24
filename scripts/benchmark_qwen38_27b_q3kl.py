import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
import json
import urllib.request
import urllib.error
import subprocess

SERVER_BIN = "build-win-cuda-v100/bin/llama-server.exe"
MODEL_PATH = "C:\\Users\\ianhs\\OneDrive\\30BB~1\\lammacpp\\models\\Qwen3.8-27B-UD-Q3_K_XL.gguf"
CACHE_PATH = "./tmp_cache"
LOG_PATH = "./server_27b.log"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"

def make_request(method, endpoint, data=None):
    url = f"{BASE_URL}{endpoint}"
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

def wait_for_server(timeout=120):
    start = time.time()
    print("[BENCH] Waiting for Qwen 3.8 27B server to load and become healthy...")
    while time.time() - start < timeout:
        status, body = make_request("GET", "/health")
        if status == 200:
            print("[BENCH] Server is healthy and ready!")
            return True
        time.sleep(2)
    return False

def get_v100_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            universal_newlines=True
        )
        lines = out.strip().splitlines()
        gpu0 = lines[0].split(",")
        used = int(gpu0[0].strip())
        total = int(gpu0[1].strip())
        return used, total
    except Exception as e:
        return -1, -1

def main():
    os.makedirs(CACHE_PATH, exist_ok=True)
    if os.path.exists(LOG_PATH):
        try:
            os.remove(LOG_PATH)
        except Exception:
            pass

    cmd = [
        SERVER_BIN,
        "-m", MODEL_PATH,
        "-ngl", "99",
        "-c", "8192",
        "-b", "2048",
        "-ub", "1024",
        "--cache-type-k", "kvarn4",
        "--cache-type-v", "kvarn4",
        "--flash-attn", "on",
        "--slot-save-path", CACHE_PATH,
        "--verbose",
        "--port", str(PORT),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print("="*65)
    print("LAUNCHING QWEN 3.8 27B (Q3_K_XL) ON SINGLE TESLA V100 (CUDA0)")
    print("="*65)
    print(f"Model: {MODEL_PATH}")
    print(f"Command: {' '.join(cmd)}\n")

    log_file = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        universal_newlines=True
    )

    try:
        if not wait_for_server(180):
            print("[FAIL] Server failed to start within timeout.")
            log_file.flush()
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                print(f.read()[-3000:])
            return

        used_mb, total_mb = get_v100_vram()
        print(f"\n[VRAM CHECK] V100 GPU 0 Memory: {used_mb} MiB / {total_mb} MiB ({used_mb/total_mb*100:.1f}% used, {total_mb-used_mb} MiB free)\n")

        # Benchmark 1: Standard Prompt (500 tokens) + 64 decode tokens
        base_prompt_text = (
            "Analyze the fundamental tradeoffs between pipeline parallelism and tensor parallelism "
            "in low-bandwidth interconnected GPU clusters. Specifically examine the activation volume "
            "at stage boundaries compared to intermediate layer-wise All-Reduce communications, and explain "
            "why low-bandwidth interconnects like PCIe 2.0 x1 favor pipeline parallelism over tensor parallelism. "
        ) * 12 # ~500+ tokens

        prompt_1 = f"<|im_start|>system\nYou are an expert systems engineer.<|im_end|>\n<|im_start|>user\n{base_prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        print(f"[BENCHMARK 1] Sending 500-token prompt + 64 generation tokens...")
        t0 = time.time()
        status_1, resp_1 = make_request("POST", "/completion", {
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

        print("\n" + "-"*65)
        print("BENCHMARK 1 RESULTS (500 Tokens):")
        print(f"  * Prompt Tokens (Prefill):     {prompt_n_1} tokens in {prompt_ms_1:.2f} ms")
        print(f"  * Prefill Throughput:          {prompt_per_sec_1:.2f} tok/s")
        print(f"  * Generation Tokens (Decode):  {gen_n_1} tokens in {gen_ms_1:.2f} ms")
        print(f"  * Decode Throughput:           {gen_per_sec_1:.2f} tok/s ({(gen_ms_1/gen_n_1 if gen_n_1>0 else 0):.2f} ms/token)")
        print(f"  * Total Wall Clock Time:       {(t1-t0)*1000:.2f} ms")
        print(f"  * Response Sample:             {content_1[:120]}...")
        print("-"*65)

        # Benchmark 2: Larger Prompt (1500 tokens) for High-Occupancy Prefill
        long_prompt_text = base_prompt_text * 3
        prompt_2 = f"<|im_start|>system\nYou are an expert systems engineer.<|im_end|>\n<|im_start|>user\n{long_prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        print(f"\n[BENCHMARK 2] Sending Large 1,500-token prompt for maximum prefill saturation...")
        status_2, resp_2 = make_request("POST", "/completion", {
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

        print("\n" + "-"*65)
        print("BENCHMARK 2 RESULTS (1,500 Tokens):")
        print(f"  * Prompt Tokens (Prefill):     {prompt_n_2} tokens in {prompt_ms_2:.2f} ms")
        print(f"  * Peak Prefill Throughput:     {prompt_per_sec_2:.2f} tok/s")
        print(f"  * Generation Tokens (Decode):  {gen_n_2} tokens in {gen_ms_2:.2f} ms")
        print(f"  * Decode Throughput:           {gen_per_sec_2:.2f} tok/s")
        print("-"*65)

        used_mb_after, _ = get_v100_vram()
        print(f"\n[FINAL VRAM STATE] V100 GPU 0 Memory: {used_mb_after} MiB / {total_mb} MiB ({total_mb - used_mb_after} MiB headroom)")

        print("\n" + "="*65)
        print("QWEN 3.8 27B BENCHMARK COMPLETE - SERVER WILL REMAIN LIVE ON PORT 8080")
        print("="*65)

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        proc.terminate()
        log_file.close()
        raise

if __name__ == "__main__":
    main()
