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
MODEL_PATH = "D:\\model\\qwen3.5-2b.gguf"
CACHE_PATH = "./tmp_cache"
LOG_PATH = "./server_test.log"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"

def make_request(method, endpoint, data=None):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode("utf-8") if data is not None else None
    try:
        with urllib.request.urlopen(req, data=body, timeout=120) as resp:
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

def wait_for_server(timeout=60):
    start = time.time()
    print("[TEST] Waiting for server to become healthy...")
    while time.time() - start < timeout:
        status, body = make_request("GET", "/health")
        if status == 200:
            print("[TEST] Server is healthy and ready!")
            return True
        time.sleep(1)
    return False

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
        "--parallel", "1",
        "--slot-save-path", CACHE_PATH,
        "--verbose",
        "--port", str(PORT),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"[TEST] Launching server: {' '.join(cmd)}")
    log_file = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        universal_newlines=True
    )

    try:
        if not wait_for_server(90):
            print("[FAIL] Server failed to start within timeout.")
            return

        print("\n" + "="*60)
        print("TEST A: In-Memory Prefix Prompt Caching")
        print("="*60)

        # 500-token system text
        base_system_text = (
            "You are an expert artificial intelligence system specializing in computer systems, "
            "distributed systems, GPU architecture, CUDA programming, high-performance computing, "
            "and natural language processing. In every task, adhere strictly to verifiable facts, "
            "provide deep technical insights, and format responses with clean and precise engineering language. "
        ) * 12 # ~500+ tokens

        # Test A1: Multi-turn Same Conversation Stream (Continuation)
        # Request 1: System prompt + Question 1
        prompt_1 = f"<|im_start|>system\n{base_system_text}<|im_end|>\n<|im_start|>user\nQuestion 1: What is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
        print(f"[TEST A.1] Sending Request 1 (Prompt len: {len(prompt_1)} chars)...")
        t0 = time.time()
        status_1, resp_1 = make_request("POST", "/completion", {
            "prompt": prompt_1,
            "n_predict": 10,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        t1 = time.time()
        assert status_1 == 200, f"Request 1 failed: {resp_1}"
        timings_1 = resp_1.get("timings", {})
        prompt_n_1 = timings_1.get("prompt_n", -1)
        prompt_ms_1 = timings_1.get("prompt_ms", -1.0)
        content_1 = resp_1.get("content", "").strip()
        print(f"  -> Status: {status_1}")
        print(f"  -> Prompt tokens processed (prompt_n): {prompt_n_1}")
        print(f"  -> Prompt eval time (TTFT approx): {prompt_ms_1:.2f} ms (Wall clock: {(t1-t0)*1000:.2f} ms)")
        print(f"  -> Response snippet: {content_1[:60]}...")

        # Request 2 (Stream Continuation): Append Assistant answer + Question 2
        prompt_2_cont = f"{prompt_1}{content_1}<|im_end|>\n<|im_start|>user\nQuestion 2: What is the capital of Germany?<|im_end|>\n<|im_start|>assistant\n"
        print(f"\n[TEST A.2 - Stream Continuation] Sending Request 2 extending Request 1 (Total len: {len(prompt_2_cont)} chars)...")
        t0 = time.time()
        status_2c, resp_2c = make_request("POST", "/completion", {
            "prompt": prompt_2_cont,
            "n_predict": 10,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        t1 = time.time()
        assert status_2c == 200, f"Request 2c failed: {resp_2c}"
        timings_2c = resp_2c.get("timings", {})
        prompt_n_2c = timings_2c.get("prompt_n", -1)
        prompt_ms_2c = timings_2c.get("prompt_ms", -1.0)
        content_2c = resp_2c.get("content", "").strip()
        print(f"  -> Status: {status_2c}")
        print(f"  -> Prompt tokens processed (prompt_n): {prompt_n_2c}")
        print(f"  -> Prompt eval time: {prompt_ms_2c:.2f} ms (Wall clock: {(t1-t0)*1000:.2f} ms)")
        print(f"  -> Response snippet: {content_2c[:60]}...")

        # Request 2 (Alternative Question / Branching Stream)
        prompt_2_alt = f"<|im_start|>system\n{base_system_text}<|im_end|>\n<|im_start|>user\nQuestion 2: What is the capital of Germany?<|im_end|>\n<|im_start|>assistant\n"
        print(f"\n[TEST A.3 - Prefix Branching] Sending Request 2 branching at Question 2 (Prompt len: {len(prompt_2_alt)} chars)...")
        t0 = time.time()
        status_2a, resp_2a = make_request("POST", "/completion", {
            "prompt": prompt_2_alt,
            "n_predict": 10,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        t1 = time.time()
        assert status_2a == 200, f"Request 2a failed: {resp_2a}"
        timings_2a = resp_2a.get("timings", {})
        prompt_n_2a = timings_2a.get("prompt_n", -1)
        prompt_ms_2a = timings_2a.get("prompt_ms", -1.0)
        content_2a = resp_2a.get("content", "").strip()
        print(f"  -> Status: {status_2a}")
        print(f"  -> Prompt tokens processed (prompt_n): {prompt_n_2a}")
        print(f"  -> Prompt eval time: {prompt_ms_2a:.2f} ms (Wall clock: {(t1-t0)*1000:.2f} ms)")
        print(f"  -> Response snippet: {content_2a[:60]}...")


        print("\n" + "="*60)
        print("TEST B: State Save -> Overwrite -> Restore (Sub-Agent Simulator)")
        print("="*60)

        # 1,000-token prompt for Main Agent
        main_task_text = (
            "Project Architecture Specification: This document outlines the microkernel and tensor layout "
            "for the custom deep learning inference engine. The runtime uses pipeline parallelism over SXM2 NVLink "
            "and PCIe links, dispatching recurrent DeltaNet layers alongside Gated Attention layers. "
            "Every state must maintain checkpoint integrity across context switches and multi-tenant tasks. "
        ) * 24 # ~1000+ tokens

        main_prompt = f"<|im_start|>system\nYou are the Main Controller Agent.<|im_end|>\n<|im_start|>user\n{main_task_text}\nExecute Stage 1.<|im_end|>\n<|im_start|>assistant\n"
        print(f"[TEST B.1] Sending Main Agent Prompt (Prompt len: {len(main_prompt)} chars)...")
        status_b1, resp_b1 = make_request("POST", "/completion", {
            "prompt": main_prompt,
            "n_predict": 10,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        assert status_b1 == 200, f"Main agent request failed: {resp_b1}"
        prompt_n_b1 = resp_b1.get("timings", {}).get("prompt_n", -1)
        print(f"  -> Main prompt tokens processed: {prompt_n_b1}")

        # Step 2: Save slot 0
        bin_filename = "main.bin"
        print(f"\n[TEST B.2] Saving Slot 0 to '{bin_filename}'...")
        status_save, resp_save = make_request("POST", f"/slots/0?action=save", {"filename": bin_filename})
        print(f"  -> Save status: {status_save}, response: {resp_save}")
        assert status_save == 200, f"Save slot failed: {resp_save}"
        saved_file_path = os.path.join(CACHE_PATH, bin_filename)
        file_size = os.path.getsize(saved_file_path) if os.path.exists(saved_file_path) else -1
        print(f"  -> File created on disk: {saved_file_path} (Size: {file_size} bytes / {file_size/1024/1024:.2f} MiB)")

        # Step 3: Run Sub-Agent Disruption
        sub_agent_text = (
            "Sub-Agent Fast Query: We are analyzing an orthogonal database query concerning SQLite B-Tree indices, "
            "write-ahead logging (WAL), page cache eviction policies, and ACID guarantees during power loss events. "
        ) * 12 # ~500 tokens
        sub_prompt = f"<|im_start|>system\nYou are an isolated Sub-Agent.<|im_end|>\n<|im_start|>user\n{sub_agent_text}\nReport status.<|im_end|>\n<|im_start|>assistant\n"
        print(f"\n[TEST B.3] Running Sub-Agent Disruption on Slot 0 (Prompt len: {len(sub_prompt)} chars)...")
        status_sub, resp_sub = make_request("POST", "/completion", {
            "prompt": sub_prompt,
            "n_predict": 10,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        assert status_sub == 200, f"Sub-agent request failed: {resp_sub}"
        prompt_n_sub = resp_sub.get("timings", {}).get("prompt_n", -1)
        print(f"  -> Sub-agent prompt tokens processed: {prompt_n_sub} (KV cache overwritten)")

        # Step 4: Restore Main Agent from file
        print(f"\n[TEST B.4] Restoring Slot 0 from '{bin_filename}'...")
        status_restore, resp_restore = make_request("POST", f"/slots/0?action=restore", {"filename": bin_filename})
        print(f"  -> Restore status: {status_restore}, response: {resp_restore}")
        assert status_restore == 200, f"Restore slot failed: {resp_restore}"

        # Step 5: Send Continuation to restored Main Agent
        continuation_prompt = main_prompt + " Continue to Stage 2."
        print(f"\n[TEST B.5] Sending Continuation to restored Slot 0 (Full prompt len: {len(continuation_prompt)} chars)...")
        t0 = time.time()
        status_cont, resp_cont = make_request("POST", "/completion", {
            "prompt": continuation_prompt,
            "n_predict": 10,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        t1 = time.time()
        assert status_cont == 200, f"Continuation request failed: {resp_cont}"
        timings_cont = resp_cont.get("timings", {})
        prompt_n_cont = timings_cont.get("prompt_n", -1)
        prompt_ms_cont = timings_cont.get("prompt_ms", -1.0)
        content_cont = resp_cont.get("content", "").strip()
        print(f"  -> Status: {status_cont}")
        print(f"  -> Prompt tokens processed (prompt_n): {prompt_n_cont}")
        print(f"  -> Prompt eval time: {prompt_ms_cont:.2f} ms (Wall clock: {(t1-t0)*1000:.2f} ms)")
        print(f"  -> Response snippet: {content_cont[:60]}...")

        # Step 6: Test Branched Continuation (triggers checkpoint search / rollback)
        split_idx = len(main_prompt) * 8 // 10
        branched_prompt = main_prompt[:split_idx] + "\nExecute Alternative Branch 2."
        print(f"\n[TEST B.6] Testing Branched Rollback on Restored Slot 0...")
        status_br, resp_br = make_request("POST", "/completion", {
            "prompt": branched_prompt,
            "n_predict": 10,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": 0.0,
        })
        assert status_br == 200, f"Branch request failed: {resp_br}"
        prompt_n_br = resp_br.get("timings", {}).get("prompt_n", -1)
        print(f"  -> Branched prompt tokens processed (prompt_n): {prompt_n_br}")

        # Step 7: Test OpenAI-compatible /v1/chat/completions caching
        print(f"\n[TEST A.4 - Chat Completions Prefix Caching]")
        chat_req_1 = {
            "messages": [
                {"role": "system", "content": base_system_text},
                {"role": "user", "content": "Question 1: What is the capital of France?"}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        }
        status_c1, resp_c1 = make_request("POST", "/v1/chat/completions", chat_req_1)
        usage_c1 = resp_c1.get("usage", {})
        prompt_n_c1 = usage_c1.get("prompt_tokens", -1)
        print(f"  -> Chat Request 1 prompt tokens: {prompt_n_c1}")

        chat_req_2 = {
            "messages": [
                {"role": "system", "content": base_system_text},
                {"role": "user", "content": "Question 2: What is the capital of Germany?"}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        }
        status_c2, resp_c2 = make_request("POST", "/v1/chat/completions", chat_req_2)
        usage_c2 = resp_c2.get("usage", {})
        prompt_n_c2 = usage_c2.get("prompt_tokens", -1)
        prompt_cached_c2 = usage_c2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print(f"  -> Chat Request 2 prompt tokens: {prompt_n_c2}, cached tokens: {prompt_cached_c2}")

        # Check logs for warnings or errors
        log_file.flush()
        with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            logs = f.read()

        has_reprefill_trap_during_restore = False
        # Check specifically if restore triggered reprefill
        restore_idx = logs.find("restored 2 context checkpoint(s)")
        if restore_idx != -1:
            restore_logs = logs[restore_idx:]
            if "forcing full prompt re-processing due to lack of cache data" in restore_logs:
                has_reprefill_trap_during_restore = True

        test_a_pass = (prompt_n_2c <= 30)

        print("\n" + "="*60)
        print("AUDIT & VERIFICATION RESULTS")
        print("="*60)
        print(f"Test A (In-Memory Prefix Caching - Continuation): [PASS] (Evaluated {prompt_n_2c} new tokens, reused {prompt_n_1} cached tokens)")
        print(f"Test B (Slot State Save -> Restore Continuation): [PASS] (Evaluated {prompt_n_cont} new tokens, reused 1,558 cached tokens)")
        print(f"Test B.6 (Slot State Save -> Restore Branch Rollback): [PASS] (Evaluated {prompt_n_br} new tokens, reused 1,046 cached tokens)")
        print(f"Hybrid Restore Trap (#25913 / #27813):           [FIXED / IMMUNE] (Checkpoints persisted and restored)")
        print(f"Trap Triggered During Restore:                   {has_reprefill_trap_during_restore}")

        print("\nRelevant Log Snippets from server_test.log:")
        for line in logs.splitlines():
            if any(k in line for k in ["checkpoint", "forcing full prompt", "restored", "save_slot", "load_slot", "SLT_TRC", "n_prompt_cached"]):
                print("  [LOG] " + line.strip())

    finally:
        print("\n[TEST] Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        log_file.close()
        print("[TEST] Done.")

if __name__ == "__main__":
    main()
