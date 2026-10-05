# Handover: the macOS port, state of 2026-10-05

A note for the next Claude Code session on the M5 Pro (and for the user). The previous session ran on a different
Mac (MacBook Pro M1 Max, 32 GB). The user writes in German: answer in German.

## The user's Macs and models

- **Target:** M5 Pro, 48 GB, 15 cores (`hw.perflevel0` = 5, a second fast-ish level of 10), Metal working set
  37.4 GiB (from the log), 1 TB SSD. Folder `~/Documents/StrataForMac-main` - most likely a **ZIP download, not a git
  clone**, so `git pull` did not bring updates (setup kept failing with errors already fixed on GitHub). First step:
  `git log --oneline -1` there; if it is not a repository, clone (see the end).
- **Models on it:** Kolibri-1 Q4_K_M, Qwen3.8-Flash-Next IQ2_XS, maybe the Coder IQ1_M.
- **Reference:** the user's PC with an RX 7900 XT writes ~60 tok/s with Qwen on Strata's native (HIP) engine.

## How Strata runs on macOS

The native Strata engine (CUDA/HIP) does not run on a Mac. `engine/strata` is a shell script that starts
`tools/strata_runner.py`, which starts llama.cpp's `llama-server` (Metal) and translates Strata's engine protocol
(GEN / T / DONE / STOP / QUIT) to its HTTP API. `serve/server.py`, the web app, setup and the MCP server are Strata's.
Details: [MACOS.md](MACOS.md).

What this fork adds on top of upstream Strata (`git log` from f3b6f18 on; upstream remote `Niko1221/Strata`):

| Piece | Where | Why |
|---|---|---|
| Kolibri-1 in llama.cpp | `third_party/patches/kolibri1-llama.cpp.patch` | llama.cpp has no `kolibri1` architecture (upstream issue #29922) |
| GPU buffer only for GPU weights | `third_party/patches/metal-split-mmap.patch` | with a GPU/CPU split, llama.cpp mapped the whole file into the Metal buffer: `kIOGPUCommandBufferCallbackErrorOutOfMemory` |
| Patches applied by setup | `setup.py` `patch_llama_macos` (`patch -p1`, not `git apply`) | `git apply` inside this repo's tree can skip files silently |
| Memory budget, whole-layer split, `--no-repack` | `tools/strata_runner.py` | GPU and CPU share memory; llama.cpp's `--fit` takes the CPU side as unlimited |
| LayerTuner | `tools/strata_runner.py`, results in `engine/metal-tune.json` | the best GPU layer count depends on the Mac and the model |
| Lazy table not counted | `lazy_bytes` in the runner | Qwen's 28.8 GB `per_layer_token_embd` stays in the file (llama.cpp lazy read) |
| Threads = `hw.perflevel0` | `get_perf_cores` | 15 threads on the M5 Pro were slower (~15 tok/s vs ~18-35 with 5) |
| `ERR` instead of silence | runner | an empty `DONE` showed as an empty reply; an unknown command (`VRAM`) made the server wait 120 s |
| Model switch in the web app | `serve/server.py` (`/models-configured`, `/switch-model`, `--switched-from`), `serve/web/app.js` | restarts the server with another `strata-*.json`; falls back if the new one fails |
| No low-RAM mode on a Mac | `setup.py` | its `experts.bin` (45.7 GB for Kolibri) is never read by the runner |
| Merge of upstream v0.1.39 | merge commit 6d7085d | arm64 stubs for new x86 code, Mach affinity beside new Windows/Linux code |

## Measured (M1 Max, 32 GB, Kolibri-1 Q4_K_M, 44 GiB to hold)

From [bench/results/2026-10-04-macos-split](../bench/results/2026-10-04-macos-split/README.md):

| How | tok/s | MB per token from the SSD |
|---|---:|---:|
| llama.cpp `--fit` (experts split) | 5.15 | 123 |
| 12 whole GPU layers | 8.2 | 82 |
| 0 GPU layers | 25.7 | 0.5 |
| 6 GPU layers | 26.1 | 9.3 |
| the tuner's choice: 1 GPU layer | 31.7 | 0.4 |
| an `madvise(MADV_WILLNEED)` prefetch of CPU experts (dropped) | 6.86 | 114 |

Every GPU layer holds all 384 experts of its layer and takes that memory from the page cache the CPU-side experts
are read through. With a model larger than memory, few GPU layers win.

## Seen on the M5 Pro (from the user's logs and screenshots)

- Kolibri Q4_K_M: ~35 tok/s with the first working version (`--fit` split, 24 GiB GPU budget, 5 threads); 23.8 with 23
  whole GPU layers (disk read ~248 MB/s, CPU 34%, memory 47.7 of 48 GB); ~15 with 15 threads and 9 GPU layers; ~18
  with 5 threads, tuning just started. No `[strata] layers:` lines were seen yet - unclear whether the tuner ran with
  the current code (ZIP folder).
- Qwen: at most ~10 tok/s. Model and runner version unknown.
- The model switch did not appear in the header - most likely the old code (ZIP folder).

## Open, in this order

1. **Get the current code onto the M5 Pro** (clone, below), run `./setup.sh`, and delete the unused
   `~/Documents/Strata-data/packs/kolibri-q4_k_m/experts.bin` (~43 GB, written by the old low-RAM mode).
2. **Qwen at ~10 tok/s:** find where the time per token goes. Start with the log's `[strata] memory:` and
   `[strata] layers:` lines. Suspects, none measured: the lazy reads of the 28.8 GB per-layer embedding table (llama.cpp
   reads rows on demand - per token and layer?), llama.cpp's Metal kernels for IQ2_XS / Q2_0, the IQ2_XS sitting just
   above the 37.4 GiB Metal limit (39.2 GB to hold), and no MTP (the runner ignores `--mtp`; upstream says MTP gives
   1.6-1.8x). `python3 tools/mac_split_bench.py <gguf> 0 8 16 24 32` measures GPU layer counts (stop Strata first).
   The Coder IQ1_M (29.6 GB to hold) now runs fully on the GPU on 48 GB - compare it.
3. **Kolibri-1 Q3_K_S** (`Eliasfpv28/Kolibri-1-Q3_K_S-GGUF`, 33.9 GB, unverified third-party quant): would fit 48 GB
   whole. Add it to setup (`MODELS` / `FAMILIES` in `setup.py` and `tools/strata_mcp.py`, keep both lists equal - a
   test checks), check that it loads and answers, compare its answers with Q4_K_M, measure speed on the M5 Pro.
4. **IQ2_XS near the Metal limit:** `sudo sysctl iogpu.wired_limit_mb=41984` (until reboot) plus
   `"env": {"STRATA_GPU_LAYERS": "49"}` would put it fully on the GPU, leaving ~6 GB for macOS. Unmeasured. A runner
   hint in the log when a model is just above the limit was offered, not built.
5. **Native engine bug (PCs, not the Mac):** `include/strata/artifact/gguf_reader.hpp` (~line 591) and
   `include/strata/core/layout.hpp` (`kolibri1()`) expect Kolibri's hidden size 6144 and expert FFN 1536; the real
   model (`config.json`) has 2560 and 512. The native engine would likely refuse the real GGUF. Not fixed.
6. **MLX:** MLX conversions of Kolibri exist (2-8 bit, e.g. `velaia/Kolibri-1-MLX-3bit`, 34.9 GB), but `mlx-lm` has no
   `kolibri1` yet (PR ml-explore/mlx-lm#1945, open). LM Studio and Ollama (MLX backend since 0.19) need that first.
   Their speed claims over llama.cpp are not measured here.

## Working on it

- Tests that run on a Mac: `python3 tools/test_strata_runner.py`, `python3 tools/test_setup_<name>.py` (all 18),
  `python3 tools/test_strata_mcp.py`, `python -m pytest serve/` (needs Python 3.10+ with jinja2, regex, pyyaml: the
  `.venv`). Known failures on a Mac, the same on plain upstream: `AmdTelemetry` (2) and `test_responses`
  `test_json_schema_text_format`.
- C++ on a Mac: `cmake -B build -DSTRATA_BUILD_TESTS=ON`, then `direct_file_async_test`, `platform_memory_test`,
  `expert_multi_test`, `pool_stress 5` (`pool_test` needs a pack; `pool_affinity_test` is Windows/Linux only).
- After a change to `third_party/llama.cpp`: reconfigure, not only rebuild (`cmake -B build-llama -S third_party/llama.cpp
  -DGGML_METAL=ON -DBUILD_SHARED_LIBS=OFF`) - new source files are globbed at configure time - then copy
  `build-llama/bin/llama-server` to `engine/`.
- Runner settings in a model's `"env"`: `STRATA_GPU_LAYERS` (fixed count, no tuning), `STRATA_TUNE=0`,
  `STRATA_GPU_BUDGET_GB`, `LLAMA_ARG_FIT_TARGET`.
- Docs style (AGENTS.md): plain words, every number with what it was measured on, no claims without a measurement.
- The user decides commits and pushes; ask before each push.

## Moving to a clone (on the M5 Pro)

```bash
cd ~/Documents
git clone https://github.com/shruxx/StrataForMac.git StrataForMac-git
cp StrataForMac-main/strata-*.json StrataForMac-git/
cd StrataForMac-git
./setup.sh
```

The models stay in `~/Documents/Strata-data` and are found again. From then on `git pull` updates.
