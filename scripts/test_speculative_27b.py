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
TARGET_MODEL = "C:\\Users\\ianhs\\OneDrive\\30BB~1\\lammacpp\\models\\Qwen3.8-27B-UD-Q3_K_XL.gguf"
DRAFT_MODEL = "C:\\Users\\ianhs\\OneDrive\\30BB~1\\lammacpp\\models\\mtp-Qwen3.8-27B-Q4_0.gguf"
CACHE_PATH = "./tmp_cache"
LOG_PATH = "./server_27b_spec.log"
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
    print("[SPEC-TEST] Waiting for server with Target 27B + Draft to initialize...")
    while time.time() - start < timeout:
        status, body = make_request("GET", "/health")
        if status == 200:
            print("[SPEC-TEST] Server is healthy and ready!")
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
    except Exception:
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
        "-m", TARGET_MODEL,
        "-md", DRAFT_MODEL,
        "--spec-type", "draft-mtp",
        "-ngl", "99",
        "-ngld", "0",
        "-c", "4096",
        "-b", "1024",
        "-ub", "512",
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
    print("LAUNCHING QWEN 3.8 27B + DRAFT SPECULATIVE ON SINGLE V100")
    print("="*65)
    print(f"Target: {TARGET_MODEL}")
    print(f"Draft:  {DRAFT_MODEL}")
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
            print("[FAIL] Server failed to start with draft model.")
            log_file.flush()
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                print(f.read()[-3000:])
            proc.terminate()
            return

        used_mb, total_mb = get_v100_vram()
        print(f"\n[VRAM CHECK] V100 GPU 0 Memory: {used_mb} MiB / {total_mb} MiB ({used_mb/total_mb*100:.1f}% used, {total_mb-used_mb} MiB free)\n")

        # Benchmark with speculation
        prompt_text = (
            "Write a Python implementation of an asynchronous priority task queue with worker pools, "
            "exponential backoff retries, and cancellation tokens. Include unit test examples. "
        )
        prompt = f"<|im_start|>system\nYou are an expert Python engineer.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

        print(f"[BENCHMARK SPECULATIVE] Generating 64 tokens with draft model...")
        status, resp = make_request("POST", "/completion", {
            "prompt": prompt,
            "n_predict": 64,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        assert status == 200, f"Request failed: {resp}"
        timings = resp.get("timings", {})
        prompt_n = timings.get("prompt_n", -1)
        prompt_ms = timings.get("prompt_ms", -1.0)
        prompt_per_sec = timings.get("prompt_per_second", -1.0)
        gen_n = timings.get("predicted_n", -1)
        gen_ms = timings.get("predicted_ms", -1.0)
        gen_per_sec = timings.get("predicted_per_second", -1.0)
        content = resp.get("content", "").strip()

        print("\n" + "-"*65)
        print("SPECULATIVE DECODING RESULTS:")
        print(f"  * Prompt Tokens (Prefill):     {prompt_n} tokens in {prompt_ms:.2f} ms ({prompt_per_sec:.2f} tok/s)")
        print(f"  * Generation Tokens (Decode):  {gen_n} tokens in {gen_ms:.2f} ms")
        print(f"  * Decode Throughput:           {gen_per_sec:.2f} tok/s ({(gen_ms/gen_n if gen_n>0 else 0):.2f} ms/token)")
        print(f"  * Response Sample:             {content[:120]}...")
        print("-"*65)

    except Exception as e:
        print(f"[ERROR]: {e}")
        proc.terminate()
        log_file.close()
        raise

if __name__ == "__main__":
    main()
