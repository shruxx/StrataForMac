#!/usr/bin/env python3
"""tools/strata_runner.py - macOS Strata Engine Runner (Metal GPU acceleration).

Speaks the exact Strata resident engine IPC protocol (`strata --serve`) expected by `serve/server.py`:
- Stdin/Stdout line-based protocol
- On launch: INFO ... and READY <max_context> stop
- On request: GEN <max_new> [sampling keys...] <token_ids>
- Streams: T <token_id>
- Finishes: DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish_reason> 0 0 0
- Handles: STOP (cancel active request) and QUIT (clean exit)

Uses native Apple Silicon Metal acceleration via llama.cpp backend for unified memory and GPU offloading.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_llama_server() -> Path | None:
    """Find the compiled llama-server executable."""
    candidates = [
        ROOT / "engine" / "llama-server",
        ROOT / "build-llama" / "bin" / "llama-server",
        ROOT / "build" / "bin" / "llama-server",
        ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    return None


def get_free_port() -> int:
    """Find an available port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_perf_cores() -> int:
    """Detect performance cores on macOS Apple Silicon."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        if out.isdigit() and int(out) > 0:
            return int(out)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 4) // 2 if (os.cpu_count() or 4) > 4 else (os.cpu_count() or 4))


def parse_args(argv: list[str]) -> dict:
    cfg = {
        "serve": False,
        "model_path": None,
        "pack_path": None,
        "max_context": 4096,
        "kv": "int8",
        "threads": None,
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--serve":
            cfg["serve"] = True
        elif arg == "--native" and i + 1 < len(argv):
            cfg["model_path"] = argv[i + 1]
            i += 1
        elif arg == "--pack" and i + 1 < len(argv):
            cfg["pack_path"] = argv[i + 1]
            i += 1
        elif arg == "--max-context" and i + 1 < len(argv):
            try:
                cfg["max_context"] = int(argv[i + 1])
            except ValueError:
                pass
            i += 1
        elif arg == "--kv" and i + 1 < len(argv):
            cfg["kv"] = argv[i + 1]
            i += 1
        elif arg == "--threads" and i + 1 < len(argv):
            try:
                cfg["threads"] = int(argv[i + 1])
            except ValueError:
                pass
            i += 1
        i += 1

    # Resolve model path
    model = cfg["model_path"]
    if not model and cfg["pack_path"]:
        pack = Path(cfg["pack_path"])
        if pack.is_file():
            model = str(pack)
        elif (pack / "model.gguf").is_file():
            model = str(pack / "model.gguf")
        elif (pack / "model-00001-of-00002.gguf").is_file():
            model = str(pack / "model-00001-of-00002.gguf")
    cfg["model_path"] = model
    return cfg


def llama_server_cmd(llama_bin, model_path, max_context: int, port: int, threads: int, kv: str) -> list[str]:
    """llama-server's command line.  No -ngl: an explicit layer count switches off llama.cpp's --fit, which otherwise
    keeps attention and the dense weights on the Metal GPU and moves only the sparse MoE experts of as many layers as
    needed to the CPU side when the whole model does not fit the GPU's working set (Strata's expert cache, the
    llama.cpp way).  A model that fits stays fully on the GPU.  With -ngl 999 a model larger than the working set
    (Kolibri-1 Q4_K_M, 47.5 GB, on a 48 GB Mac) was put on the GPU whole."""
    cmd = [
        str(llama_bin),
        "-m", str(model_path),
        "-c", str(max_context),
        "--fit", "on",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--parallel", "1",
        "--threads", str(threads),
    ]
    if kv == "int8":
        cmd += ["-ctk", "q8_0", "-ctv", "q8_0"]
    elif kv == "k8v4":
        cmd += ["-ctk", "q8_0", "-ctv", "q4_0"]
    elif kv == "q4":
        cmd += ["-ctk", "q4_0", "-ctv", "q4_0"]
    return cmd


def run_serve(cfg: dict):
    llama_bin = find_llama_server()
    if not llama_bin:
        sys.stderr.write("[strata] ERROR: llama-server executable not found.\n")
        sys.stderr.write("[strata] Please run setup.py / setup.sh to compile the engine for macOS.\n")
        sys.exit(1)

    model_path = cfg["model_path"]
    if not model_path or not Path(model_path).exists():
        sys.stderr.write(f"[strata] ERROR: model file not found: {model_path}\n")
        sys.exit(1)

    max_context = cfg["max_context"]
    threads = cfg["threads"] or get_perf_cores()
    port = get_free_port()

    cmd = llama_server_cmd(llama_bin, model_path, max_context, port, threads, cfg.get("kv", "int8"))
    kv = cfg.get("kv", "int8")

    sys.stderr.write(f"[strata] starting Metal engine on port {port} (model: {Path(model_path).name}, ctx: {max_context}, threads: {threads}) ...\n")
    sys.stderr.flush()

    # Launch llama-server with output going to stderr (which server.py directs to the log)
    server_proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=sys.stderr,
        stderr=sys.stderr,
    )

    # Wait for server to become healthy
    health_url = f"http://127.0.0.1:{port}/health"
    start_time = time.monotonic()
    ready = False
    while time.monotonic() - start_time < 300:
        if server_proc.poll() is not None:
            sys.stderr.write(f"[strata] llama-server exited prematurely with code {server_proc.returncode}\n")
            sys.exit(1)
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "StrataEngine/0.1"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.2)

    if not ready:
        sys.stderr.write("[strata] ERROR: timed out waiting for engine to be ready.\n")
        server_proc.kill()
        sys.exit(1)

    # Initial handshake for server.py
    model_name = Path(model_path).stem
    sys.stdout.write(f"INFO engine=0.1.38 backend=metal max_context={max_context} model={model_name} kv={kv}\n")
    sys.stdout.write(f"READY {max_context} stop\n")
    sys.stdout.flush()

    # Input command reader thread
    cmd_queue: queue.Queue[str | None] = queue.Queue()

    def stdin_reader():
        try:
            for line in sys.stdin:
                cmd_queue.put(line)
        except Exception:
            pass
        cmd_queue.put(None)

    t = threading.Thread(target=stdin_reader, daemon=True)
    t.start()

    active_resp_holder: list[any] = [None]
    stop_event = threading.Event()

    try:
        while True:
            line = cmd_queue.get()
            if line is None:
                break
            raw = line.strip()
            if not raw:
                continue
            if raw == "QUIT":
                break
            if raw == "STOP":
                # Not currently generating; acknowledge or ignore
                continue
            if raw.startswith("GEN ") or raw.startswith("GENI "):
                handle_generation(raw, port, cmd_queue, active_resp_holder, stop_event)
    finally:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=3)
        except Exception:
            server_proc.kill()


def handle_generation(line: str, port: int, cmd_queue: queue.Queue,
                      active_resp_holder: list[any], stop_event: threading.Event):
    parts = line.strip().split()
    cmd_name = parts[0]
    try:
        max_new = int(parts[1])
    except (IndexError, ValueError):
        max_new = 512

    sampling: dict[str, str] = {}
    token_str = parts[-1]
    for item in parts[2:-1]:
        if "=" in item:
            k, _, v = item.partition("=")
            sampling[k] = v

    prompt_tokens: list[int] = []
    if token_str and "," in token_str or token_str.isdigit():
        for t in token_str.split(","):
            t = t.strip()
            if t.isdigit():
                prompt_tokens.append(int(t))

    payload: dict = {
        "prompt": prompt_tokens,
        "n_predict": max_new,
        "stream": True,
        "return_tokens": True,
    }

    if "temperature" in sampling:
        try:
            payload["temperature"] = float(sampling["temperature"])
        except ValueError:
            pass
    if "top_p" in sampling:
        try:
            payload["top_p"] = float(sampling["top_p"])
        except ValueError:
            pass
    if "top_k" in sampling:
        try:
            payload["top_k"] = int(sampling["top_k"])
        except ValueError:
            pass
    if "min_p" in sampling:
        try:
            payload["min_p"] = float(sampling["min_p"])
        except ValueError:
            pass
    if "penalty_repeat" in sampling:
        try:
            payload["repeat_penalty"] = float(sampling["penalty_repeat"])
        except ValueError:
            pass
    if "penalty_freq" in sampling:
        try:
            payload["frequency_penalty"] = float(sampling["penalty_freq"])
        except ValueError:
            pass
    if "penalty_present" in sampling:
        try:
            payload["presence_penalty"] = float(sampling["penalty_present"])
        except ValueError:
            pass
    if "penalty_last_n" in sampling:
        try:
            payload["repeat_last_n"] = int(sampling["penalty_last_n"])
        except ValueError:
            pass
    if "seed" in sampling:
        try:
            payload["seed"] = int(sampling["seed"])
        except ValueError:
            pass

    url = f"http://127.0.0.1:{port}/completion"
    req_body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_body,
        headers={"Content-Type": "application/json", "User-Agent": "StrataEngine/0.1"}
    )

    stop_event.clear()
    generated_count = 0
    prompt_ms = 0.0
    decode_ms = 0.0
    finish_reason = "stop"
    prompt_n = len(prompt_tokens)

    try:
        resp = urllib.request.urlopen(req, timeout=120)
        active_resp_holder[0] = resp
    except Exception as e:
        sys.stderr.write(f"[strata] failed to connect to llama-server: {e}\n")
        sys.stdout.write(f"DONE 0 {prompt_n} 0.0 0.0 stop 0 0 0\n")
        sys.stdout.flush()
        return

    # Check for STOP in a background watcher or polling
    stopped = False

    try:
        buffer = b""
        start_gen = time.monotonic()
        while True:
            # Check if a STOP or QUIT command arrived on stdin
            try:
                msg = cmd_queue.get_nowait()
                if msg is not None:
                    smsg = msg.strip()
                    if smsg == "STOP":
                        stopped = True
                        break
                    elif smsg == "QUIT":
                        cmd_queue.put(msg)  # re-queue for outer loop
                        stopped = True
                        break
            except queue.Empty:
                pass

            chunk = resp.read(512)
            if not chunk:
                break
            buffer += chunk

            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line_str = line_bytes.decode("utf-8", "replace").strip()
                if not line_str.startswith("data:"):
                    continue
                data_json = line_str[5:].strip()
                if not data_json or data_json == "[DONE]":
                    continue
                try:
                    data = json.loads(data_json)
                except Exception:
                    continue

                # Stream token ids
                tokens = data.get("tokens")
                if tokens:
                    for tok in tokens:
                        sys.stdout.write(f"T {tok}\n")
                        generated_count += 1
                    sys.stdout.flush()

                if data.get("stop", False):
                    timings = data.get("timings", {})
                    prompt_n = timings.get("prompt_n", prompt_n)
                    prompt_ms = float(timings.get("prompt_ms", 0.0))
                    decode_ms = float(timings.get("predicted_ms", 0.0))
                    stop_type = data.get("stop_type", "stop")
                    if stop_type == "limit":
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                    break

        if decode_ms <= 0.0 and generated_count > 0:
            decode_ms = max(0.1, (time.monotonic() - start_gen) * 1000.0)

    except Exception as e:
        sys.stderr.write(f"[strata] stream error: {e}\n")
    finally:
        try:
            resp.close()
        except Exception:
            pass
        active_resp_holder[0] = None

    if stopped:
        finish_reason = "stop"

    sys.stdout.write(f"DONE {generated_count} {prompt_n} {prompt_ms:.1f} {decode_ms:.1f} {finish_reason} 0 0 0\n")
    sys.stdout.flush()


def main():
    if len(sys.argv) > 1 and "--version" in sys.argv:
        print("Strata macOS Engine 0.1.38 (Metal acceleration)")
        return 0

    cfg = parse_args(sys.argv[1:])
    if cfg["serve"]:
        run_serve(cfg)
    else:
        print("Strata macOS Engine. Use --serve to start resident engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
