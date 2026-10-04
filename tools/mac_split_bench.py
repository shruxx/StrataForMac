#!/usr/bin/env python3
"""tools/mac_split_bench.py - how many layers of a model belong on the GPU of this Mac (macOS, Apple Silicon).

Starts engine/llama-server once per GPU layer count, generates 128 tokens three times after a warm-up and prints the
median output speed.  The runner (tools/strata_runner.py) picks the count from the memory budget; the fastest count
here can be set for every start with "env": {"STRATA_GPU_LAYERS": "<n>"} in strata-<model>.json.

    python3 tools/mac_split_bench.py <model.gguf>                 # the runner's count and a few around it
    python3 tools/mac_split_bench.py <model.gguf> 0 16 24 28      # these counts

Stop Strata first: the model is loaded once per count (a minute or two each for a large model).
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
    p = subprocess.Popen([str(llama_bin), "-m", str(model), "-c", "4096", "--port", str(PORT), *args],
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
    budget = R.gpu_budget(ram, ws, size)
    auto = R.gpu_layer_count(sizes[0], sizes[1], budget - R.CTX_ALLOWANCE)
    n_all = len(sizes[0]) + 1
    counts = [int(x) for x in sys.argv[2:]] or sorted({0, max(auto - 6, 0), auto, min(auto + 4, n_all)})
    no_repack = ["--no-repack"] if R.pages_from_ssd(ram, size) else []
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
