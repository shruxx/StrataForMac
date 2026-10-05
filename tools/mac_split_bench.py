#!/usr/bin/env python3
"""tools/mac_split_bench.py - how many layers of a model belong on the GPU of this Mac (macOS, Apple Silicon).

Starts engine/llama-server once per GPU layer count, generates 128 tokens three times after a warm-up and prints the
median output speed.  The runner (tools/strata_runner.py) picks the count from the memory budget; the fastest count
here can be set for every start with "env": {"STRATA_GPU_LAYERS": "<n>"} in strata-<model>.json.

    python3 tools/mac_split_bench.py <model.gguf>                 # the runner's count and a few around it
    python3 tools/mac_split_bench.py <model.gguf> 0 16 24 28      # these counts

Stop Strata first: the model is loaded once per count (a minute or two each for a large model).

**This measures llama-server alone, and a high layer count reads too fast here for two reasons.**  Nothing else is
resident, while in normal use serve/server.py and the runner are; and one prompt at temperature 0 picks nearly the
same experts every reply, so they stay in the page cache.  Both matter most where the GPU's layers have taken the
memory the CPU-side experts are read through.  Measured with Kolibri-1 Q4_K_M on a 48 GB M5 Pro, -c 131072: at 9
GPU layers this tool and the running server agree (28.3 against 26.5-28.3 tok/s), at 30 they do not (43.7 against
15.7-22.5), and the server is fastest at 1-5 layers, where this tool is slowest
(bench/results/2026-10-05-macos-m5pro-split).  So: good for ranking low counts, not for picking one.  The runner's
own tuner measures the real replies and is the authority on the count.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strata_runner as R  # noqa: E402

PORT = 51123
PROMPT = "Write a long story about a dragon who learns to cook."


def tok_s(llama_bin, model, args) -> float | None:
    # --threads like the runner's (get_perf_cores), not llama.cpp's own default: on an M5 Pro the thread count is
    # worth about as much as the layer split (15 threads ~15 tok/s against 5 threads ~27 at the same 9 GPU layers,
    # docs/MACOS.md), so a row measured here must not differ from the runner in it
    p = subprocess.Popen([str(llama_bin), "-m", str(model), "-c", "4096", "--port", str(PORT),
                          "--threads", str(R.get_perf_cores()), *args],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            if p.poll() is not None:
                return None
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1).status == 200:
                    break
            except Exception:
                time.sleep(1)
        body = json.dumps({"prompt": PROMPT, "n_predict": 128, "temperature": 0, "ignore_eos": True,
                           "cache_prompt": False}).encode()
        speeds = []
        for i in range(4):
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", body,
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            if i:
                speeds.append(r["timings"]["predicted_per_second"])
        return statistics.median(speeds)
    except Exception:
        return None
    finally:
        p.terminate()
        p.wait()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    model = Path(sys.argv[1])
    llama_bin = R.find_llama_server()
    sizes = R.layer_sizes(model)
    if not llama_bin or not sizes:
        print("llama-server or the model's layer table not found")
        return 1
    ram, ws, size = R.physical_memory(), R.metal_working_set(llama_bin), R.model_bytes(model)
    size -= R.lazy_bytes(R.tensor_sizes(model) or [])
    budget = R.gpu_budget(ram, ws, size)
    # the runner's start count, both of its cases: a model read from the SSD starts at START_GPU_SHARE of the
    # memory left after the reserve (the rest stays page cache), one that fits gets the GPU budget
    from_ssd = R.pages_from_ssd(ram, size)
    weights = R.START_GPU_SHARE * max(ram - R.OS_RESERVE, 0) if from_ssd else budget - R.CTX_ALLOWANCE
    auto = R.gpu_layer_count(sizes[0], sizes[1], int(weights))
    n_all = len(sizes[0]) + 1
    counts = [int(x) for x in sys.argv[2:]] or sorted({0, max(auto - 6, 0), auto, min(auto + 4, n_all)})
    no_repack = ["--no-repack"] if from_ssd else []
    print(f"{ram / R.GIB:.0f} GiB shared, GPU working set {ws / R.GIB:.1f} GiB, model {size / R.GIB:.1f} GiB, "
          f"{n_all} layers; the runner's count: {auto}")
    for n in counts:
        gib = (sizes[1] + sum(sizes[0][len(sizes[0]) - n + 1:])) / R.GIB if n else 0.0
        v = tok_s(llama_bin, model, ["--fit", "off", "-ngl", str(n), *no_repack])
        print(f"  -ngl {n:3d} ({gib:5.1f} GiB of weights on the GPU): "
              + (f"{v:6.1f} tok/s" if v else "failed (out of memory?)"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
