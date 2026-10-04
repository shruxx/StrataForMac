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
import re
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
    """The threads for llama-server: the cores of the Mac's fastest performance level (hw.perflevel0).  Not the
    other levels: on an M5 Pro (5 + 10 cores, the second level not called "Efficiency") Kolibri-1 wrote ~15 tok/s
    with 15 threads (9 GPU layers); with 5 threads it had written ~24-35 (other GPU splits).  Every step waits for its
    slowest thread."""
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


GIB = 1024 ** 3
MIB = 1024 ** 2
# what macOS, serve/server.py and the browser with the chat keep for themselves out of the shared memory
OS_RESERVE = 8 * GIB
# KV cache and compute buffers on top of the weights, for the "fits whole" check (--fit itself measures them)
CTX_ALLOWANCE = 4 * GIB
# when the model does not fit whole: the GPU's share of what is left after the reserve; the rest stays free as the
# page cache the CPU-side experts are read through (they come from the SSD when it is too small)
GPU_SHARE = 0.6


def physical_memory() -> int:
    """The Mac's unified memory in bytes (hw.memsize), 0 when unknown."""
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL).strip())
    except Exception:
        return 0


def metal_working_set(llama_bin) -> int:
    """The Metal device's working set in bytes (recommendedMaxWorkingSetSize, or the iogpu.wired_limit_mb a user
    set), as llama-server's --list-devices reports it ("MTL0: Apple M5 Pro (36864 MiB, 36863 MiB free)"); 0 when
    unknown."""
    try:
        out = subprocess.run([str(llama_bin), "--list-devices"], capture_output=True, text=True, timeout=30)
        m = re.search(r"MTL\d+:.*?\((\d+) MiB,", out.stdout + out.stderr)
        return int(m.group(1)) * MIB if m else 0
    except Exception:
        return 0


def model_bytes(model_path) -> int:
    """The model's size on disk: all shards of a split GGUF (name-00001-of-0000N.gguf)."""
    return sum(f.stat().st_size for f in model_shards(model_path))


def pages_from_ssd(ram: int, model: int) -> bool:
    """True when the model, its KV cache and buffers do not fit the memory left after OS_RESERVE: its CPU-side
    experts are then read through the page cache from the SSD."""
    return model + CTX_ALLOWANCE > max(ram - OS_RESERVE, 0)


def gpu_budget(ram: int, working_set: int, model: int) -> int:
    """How many bytes of the model, KV cache and compute buffers may live on the Metal GPU.

    On Apple Silicon the GPU and the CPU share one memory.  llama.cpp's --fit plans against the GPU's working set
    alone (~75% of the memory) and takes the CPU side as unlimited: with Kolibri-1 Q4_K_M (47.5 GB) on a 48 GB Mac
    it put ~35 GB on the GPU, the experts left on the CPU side and macOS did not fit next to it, and the first
    request failed with kIOGPUCommandBufferCallbackErrorOutOfMemory.  So: a model that fits the memory left after
    OS_RESERVE goes on the GPU whole (up to its working set); a larger one gets GPU_SHARE of that memory on the GPU
    and leaves the rest as page cache for its CPU-side experts."""
    usable = max(ram - OS_RESERVE, 0)
    if not pages_from_ssd(ram, model):
        return min(working_set, usable)
    return min(working_set, int(usable * GPU_SHARE))


def model_shards(model_path) -> list[Path]:
    """The model's files: all shards of a split GGUF (name-00001-of-0000N.gguf), else the one file."""
    p = Path(model_path)
    m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", p.name)
    return sorted(p.parent.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf")) if m else [p]


def layer_sizes(model_path) -> tuple[list[int], int] | None:
    """The bytes of each transformer block (blk.N.*) and of the output head (output*, which llama.cpp offloads as
    one more layer), from the GGUF tensor tables; None when the files cannot be read."""
    try:
        from gguf_reader import GGUFFile
        blocks: dict[int, int] = {}
        head = 0
        for f in model_shards(model_path):
            g = GGUFFile(f)
            end = f.stat().st_size - g.data_start
            ts = sorted(g.tensors, key=lambda t: t.offset)
            for t, nxt in zip(ts, ts[1:] + [None]):
                size = (nxt.offset if nxt else end) - t.offset
                m = re.match(r"blk\.(\d+)\.", t.name)
                if m:
                    blocks[int(m.group(1))] = blocks.get(int(m.group(1)), 0) + size
                elif t.name.startswith("output"):
                    head += size
        return ([blocks[i] for i in sorted(blocks)], head) if blocks else None
    except Exception:
        return None


def gpu_layer_count(blocks: list[int], head: int, weight_budget: int) -> int:
    """llama.cpp's -ngl N puts the output head and the last N-1 blocks on the GPU: the largest N whose weights fit."""
    if head > weight_budget:
        return 0
    used, n = head, 1
    for b in reversed(blocks):
        if used + b > weight_budget:
            break
        used, n = used + b, n + 1
    return n


# a model larger than the memory: where the layer tuner starts, as a share of the memory left after OS_RESERVE for
# the GPU's layers.  Every GPU layer holds all its experts and takes that memory from the page cache the CPU-side
# experts are read through: Kolibri-1 Q4_K_M on a 32 GB M1 Max, 0 / 6 / 12 GPU layers (0 / 4.9 / 10.2 GiB):
# 25.7 / 26.1 / 8.2 tok/s, 0.5 / 9.3 / 81.9 MB per token read from the SSD (bench/results/2026-10-04-macos-split)
START_GPU_SHARE = 0.2
TUNE_FILE = ROOT / "engine" / "metal-tune.json"
TUNE_MIN_TOKENS = 256          # decode tokens measured at one layer count before the tuner decides
TUNE_IDLE_S = 20.0             # quiet seconds before llama-server restarts with another layer count
TUNE_MAX_RESTARTS = 6          # per start of the engine
TUNE_GAIN = 1.03               # a layer count must be this much faster to count as better


def process_pageins(pid: int) -> int | None:
    """The pages a process read from disk so far (proc_pid_rusage), None when unknown."""
    try:
        import ctypes
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")

        class RusageInfoV2(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
                "user", "system", "pkg_idle", "interrupt", "pageins", "wired", "resident", "phys", "start", "exit",
                "child_user", "child_system", "child_pkg_idle", "child_interrupt", "child_pageins", "child_elapsed",
                "diskio_read", "diskio_written")]
        r = RusageInfoV2()
        return int(r.pageins) if lib.proc_pid_rusage(pid, 2, ctypes.byref(r)) == 0 else None
    except Exception:
        return None


class LayerTuner:
    """Finds the GPU layer count that writes fastest on this Mac, from the replies themselves.

    Each layer count is measured over TUNE_MIN_TOKENS decode tokens (the first reply after a start is the warm-up and
    not counted).  From the start count it tries one step towards more GPU layers when the SSD is quiet (< 2 MB read
    per token), else towards fewer; it keeps going while that is faster, then goes back to the fastest and stops.
    Results are kept in TUNE_FILE per model and memory size, so the next start begins at the fastest count."""

    def __init__(self, key: str, n_layers: int, start: int, path: Path | None = None):
        self.key, self.n, self.path = key, n_layers, path or TUNE_FILE
        self.step = max(2, n_layers // 12)
        self.results: dict[int, tuple[float, float]] = {}
        self.done, self.direction, self.flipped = False, 0, False
        saved = self._load().get(key) or {}
        for k, v in (saved.get("results") or {}).items():
            self.results[int(k)] = (float(v[0]), float(v[1]))
        self.done = bool(saved.get("done"))
        self.current = int(saved["best"]) if "best" in saved else max(0, min(start, n_layers))
        self.restarts = 0
        self._reset()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self):
        data = self._load()
        best = self.best()
        data[self.key] = {"best": best if best is not None else self.current, "done": self.done,
                          "results": {str(k): [round(v[0], 2), round(v[1], 2)] for k, v in sorted(self.results.items())}}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            pass

    def _reset(self):
        self.tokens, self.ms, self.pages, self.warm = 0, 0.0, 0, False

    def best(self) -> int | None:
        if not self.results:
            return None
        top = max(v[0] for v in self.results.values())
        # the fewest GPU layers within TUNE_GAIN of the fastest: more memory stays for everything else
        return min(k for k, v in self.results.items() if v[0] * TUNE_GAIN >= top)

    def add(self, tokens: int, ms: float, pages: int | None):
        """One finished reply at the current layer count."""
        if self.done or tokens < 16 or ms <= 0:
            return
        if not self.warm:                           # the first reply after a start fills the caches
            self.warm = True
            return
        self.tokens, self.ms, self.pages = self.tokens + tokens, self.ms + ms, self.pages + (pages or 0)

    def next_layers(self) -> int | None:
        """The layer count to restart with, when the current one is measured and another is worth trying."""
        if self.done or self.tokens < TUNE_MIN_TOKENS or self.restarts >= TUNE_MAX_RESTARTS:
            return None
        tok_s, mb = self.tokens / (self.ms / 1000.0), self.pages * 16384 / 1e6 / self.tokens
        self.results[self.current] = (tok_s, mb)
        self._reset()
        best = self.best()
        if self.direction == 0:                     # first decision: towards the GPU while the SSD is quiet
            self.direction = 1 if mb < 2.0 and self.current < self.n else -1
        elif best != self.current:                  # that step was not faster: the other way once, else done
            if self.flipped:
                self.done = True
            else:
                self.flipped, self.direction = True, -self.direction
        nxt = None if self.done else (best if best is not None else self.current) + self.direction * self.step
        if nxt is not None and (nxt < 0 or nxt > self.n or nxt in self.results):
            nxt = None
            self.done = True
        if self.done:
            nxt = best if best != self.current else None
        self._save()
        if nxt is not None:
            self.restarts += 1
            self.current = nxt
        return nxt


def fit_margin_mib(working_set: int, budget: int) -> int:
    """--fit-target: the MiB --fit leaves free of the working set so that the GPU holds at most `budget` (1024,
    llama.cpp's default, when that is less)."""
    return max((working_set - budget) // MIB, 1024)


def llama_server_cmd(llama_bin, model_path, max_context: int, port: int, threads: int, kv: str,
                     fit_target_mib: int | None = None, no_repack: bool = False,
                     gpu_layers: int | None = None) -> list[str]:
    """llama-server's command line.  No -ngl: an explicit layer count switches off llama.cpp's --fit, which otherwise
    keeps attention and the dense weights on the Metal GPU and moves only the sparse MoE experts of as many layers as
    needed to the CPU side when the whole model does not fit the GPU's working set (Strata's expert cache, the
    llama.cpp way).  A model that fits stays fully on the GPU.  With -ngl 999 a model larger than the working set
    (Kolibri-1 Q4_K_M, 47.5 GB, on a 48 GB Mac) was put on the GPU whole."""
    cmd = [
        str(llama_bin),
        "-m", str(model_path),
        "-c", str(max_context),
        # whole layers when the model does not fit the GPU (gpu_layers), else --fit with the budget's margin
        *(["--fit", "off", "-ngl", str(gpu_layers)] if gpu_layers is not None else
          ["--fit", "on", *(["--fit-target", str(fit_target_mib)] if fit_target_mib else [])]),
        # repacking copies the CPU-side experts into memory macOS can only swap, not drop and read again from the file
        *(["--no-repack"] if no_repack else []),
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
    kv = cfg.get("kv", "int8")

    # the GPU's share of the shared memory (gpu_budget); a margin in the config's "env" (LLAMA_ARG_FIT_TARGET) or
    # an explicit budget (STRATA_GPU_BUDGET_GB) wins
    ram, working_set, model = physical_memory(), metal_working_set(llama_bin), model_bytes(model_path)
    budget = None
    if os.environ.get("STRATA_GPU_BUDGET_GB"):
        budget = int(float(os.environ["STRATA_GPU_BUDGET_GB"]) * GIB)
    elif not os.environ.get("LLAMA_ARG_FIT_TARGET") and ram and working_set:
        budget = gpu_budget(ram, working_set, model)
    no_repack = bool(ram) and pages_from_ssd(ram, model)
    # a model larger than the budget: whole layers on the GPU, the first ones on the CPU.  --fit instead keeps
    # every layer's attention on the GPU and moves the experts of many layers to the CPU, and the switches between
    # them cost more than the work: OLMoE-1B-7B Q4_K_M on an M1 Max, 1.3 GiB of weights on the GPU, 55.8 tok/s with
    # --fit, 92.8 with -ngl 5 (bench/results/2026-10-04-macos-split).  How many is found by LayerTuner.
    fit_target, gpu_layers, sizes, tuner = None, None, None, None
    if os.environ.get("STRATA_GPU_LAYERS", "").strip().isdigit():     # set by hand: no tuning
        gpu_layers, sizes = int(os.environ["STRATA_GPU_LAYERS"]), layer_sizes(model_path)
    elif budget is not None and model + CTX_ALLOWANCE > budget:
        sizes = layer_sizes(model_path)
        if sizes:
            weights = (START_GPU_SHARE * max(ram - OS_RESERVE, 0) if no_repack else budget - CTX_ALLOWANCE)
            gpu_layers = gpu_layer_count(sizes[0], sizes[1], int(weights))
            if os.environ.get("STRATA_TUNE", "1") != "0":
                tuner = LayerTuner(f"{Path(model_path).name}:{model}:{ram // GIB}", len(sizes[0]) + 1, gpu_layers)
                gpu_layers = tuner.current
    if gpu_layers is None and budget is not None and working_set:
        fit_target = fit_margin_mib(working_set, budget)
    if ram and working_set:
        how = (f"{gpu_layers} of {len(sizes[0]) + 1 if sizes else '?'} layers on the GPU" if gpu_layers is not None else
               f"GPU budget {budget / GIB:.1f} GiB" if budget is not None else "llama.cpp's own fit")
        if tuner:
            how += " (tuned)" if tuner.done else " (tuning: measures the replies, tries other counts while idle)"
        sys.stderr.write(f"[strata] memory: {ram / GIB:.0f} GiB shared, GPU working set {working_set / GIB:.1f} GiB, "
                         f"model {model / GIB:.1f} GiB -> {how}\n")

    def start(layers):
        """llama-server with `layers` GPU layers (None: --fit), once its /health answers; exits the runner if not."""
        port = get_free_port()
        cmd = llama_server_cmd(llama_bin, model_path, max_context, port, threads, kv, fit_target, no_repack, layers)
        sys.stderr.write(f"[strata] starting Metal engine on port {port} (model: {Path(model_path).name}, "
                         f"ctx: {max_context}, threads: {threads}) ...\n")
        sys.stderr.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=sys.stderr, stderr=sys.stderr)
        start_time = time.monotonic()
        while time.monotonic() - start_time < 300:
            if proc.poll() is not None:
                sys.stderr.write(f"[strata] llama-server exited prematurely with code {proc.returncode}\n")
                sys.exit(1)
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"User-Agent": "StrataEngine/0.1"})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        return proc, port
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.2)
        sys.stderr.write("[strata] ERROR: timed out waiting for engine to be ready.\n")
        proc.kill()
        sys.exit(1)

    server_proc, port = start(gpu_layers)

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
    pending, last_done = None, time.monotonic()

    try:
        while True:
            try:
                line = cmd_queue.get(timeout=1.0)
            except queue.Empty:
                if pending is not None and time.monotonic() - last_done >= TUNE_IDLE_S:
                    # idle: restart llama-server with the next layer count (a request now waits for it)
                    server_proc.terminate()
                    try:
                        server_proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        server_proc.kill()
                        server_proc.wait()
                    server_proc, port = start(pending)
                    pending = None
                continue
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
                before = process_pageins(server_proc.pid) if tuner else None
                result = handle_generation(raw, port, cmd_queue, active_resp_holder, stop_event)
                last_done = time.monotonic()
                if tuner and result:
                    after = process_pageins(server_proc.pid)
                    was, measured = tuner.current, dict(tuner.results)
                    tuner.add(result[0], result[1], after - before if before is not None and after is not None else None)
                    nxt = tuner.next_layers()
                    if tuner.results != measured:            # a layer count was measured just now
                        tok_s, mb = tuner.results[was]
                        sys.stderr.write(f"[strata] layers: {was} on the GPU: {tok_s:.1f} tok/s, {mb:.1f} MB per token "
                                         f"from the SSD" + (f" -> trying {nxt} when idle" if nxt is not None and
                                                            not tuner.done else
                                                            f" -> back to {nxt}, the fastest" if nxt is not None else
                                                            " -> kept" if tuner.done else "") + "\n")
                        sys.stderr.flush()
                    pending = nxt
    finally:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=3)
        except Exception:
            server_proc.kill()


def report_error(message: str):
    """A failed request: the reason to the log and as `ERR` to serve/server.py, which ends the request with it."""
    message = " ".join(message.split())
    sys.stderr.write(f"[strata] {message}\n")
    sys.stderr.flush()
    sys.stdout.write(f"ERR {message}\n")
    sys.stdout.flush()


def handle_generation(line: str, port: int, cmd_queue: queue.Queue,
                      active_resp_holder: list[any], stop_event: threading.Event) -> tuple[int, float] | None:
    """One GEN: streams T lines and ends with DONE (or ERR).  Returns the reply's (decode tokens, decode ms) when it
    finished on its own, for the layer tuner; None after STOP or an error."""
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
        # ERR, not an empty DONE: an empty DONE reads as a reply that ended without text ("still thinking")
        detail = str(e)
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail += ": " + e.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
        report_error(f"llama-server did not take the request: {detail}")
        return None

    # Check for STOP in a background watcher or polling
    stopped = False
    finished = False
    failure = None

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
                if data.get("error"):                    # llama-server's own reason (an SSE error event)
                    err = data["error"]
                    raise RuntimeError(err.get("message", err) if isinstance(err, dict) else err)

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
                    finished = True
                    break

        if decode_ms <= 0.0 and generated_count > 0:
            decode_ms = max(0.1, (time.monotonic() - start_gen) * 1000.0)

    except Exception as e:
        failure = f"llama-server stopped the reply: {e}"
    finally:
        try:
            resp.close()
        except Exception:
            pass
        active_resp_holder[0] = None

    if stopped:
        finish_reason = "stop"
    elif failure or not finished:
        # without its final event the reply did not end normally (llama-server crashed or closed the stream)
        report_error(failure or f"llama-server ended the reply without finishing it ({generated_count} tokens)")
        return None

    sys.stdout.write(f"DONE {generated_count} {prompt_n} {prompt_ms:.1f} {decode_ms:.1f} {finish_reason} 0 0 0\n")
    sys.stdout.flush()
    return None if stopped else (generated_count, decode_ms)


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
