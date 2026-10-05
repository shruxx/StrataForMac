# Strata on macOS (Apple Silicon)

Strata runs on Apple Silicon Macs (M1, M2, M3, M4, M5 — base, Pro, Max, Ultra) under macOS.

Apple Silicon features Unified Memory Architecture (UMA), meaning CPU and GPU share the same physical memory pool with high bandwidth (e.g. 400 GB/s on M1 Max). This fits Strata's hybrid MoE design naturally: expert activations can be computed with ARM NEON and Apple Accelerate BLAS without PCIe transfer bottlenecks.

---

## What you need

- **Mac:** Any Apple Silicon Mac (M1/M2/M3/M4/M5 family).
- **RAM (Unified Memory):**
  - **32 GB:** Runs the Qwen3.8-Flash-Next Coder (`IQ1_M`, ~23 GB model arena).
  - **48 GB – 64 GB:** Runs `IQ2_XS` and `IQ3_XXS`.
  - **96 GB+:** Runs `IQ3_S`.
- **macOS:** macOS 12 (Monterey) or newer (macOS 13+ recommended).
- **Disk:** ~70–80 GB free SSD space for the model.
- **Tools:** Xcode Command Line Tools (`xcode-select --install`).

---

## One-click Setup

From the Strata folder in Terminal:

```bash
./setup.sh --check
```

This checks your Mac's hardware: CPU, GPU cores, Unified Memory, and tells you which models fit.

To install and run (e.g. with recommended defaults):

```bash
./setup.sh --yes
```

On a 32 GB Mac, setup automatically chooses the **Coder** model (`IQ1_M`), compiles the native macOS engine, prepares the expert layout, and starts the API server at `http://127.0.0.1:8080`.

---

## Platform Implementation Details

The macOS port replaces Linux- and Windows-specific APIs with native Darwin and Mach equivalents:

### 1. Direct File I/O (`src/platform/direct_file.cpp`)
- Linux uses `open(..., O_DIRECT)`. On macOS, `O_DIRECT` does not exist.
- Strata on macOS uses `fcntl(fd, F_NOCACHE, 1)` to disable the OS page cache for direct unbuffered disk reads.

### 2. Memory Detection (`src/platform/memory.cpp`, `src/core/expert_source.cpp`)
- Physical memory is detected via `sysctl(CTL_HW, HW_MEMSIZE)`.
- Available system memory is detected via Mach host statistics (`host_statistics64(HOST_VM_INFO64)`: free + inactive + purgeable pages).

### 3. Thread Affinity & Pause (`src/kernels/cpu/pool.cpp`)
- Linux uses `pthread_setaffinity_np` and `cpu_set_t`.
- Darwin does not expose core pinning; it uses Mach `thread_policy_set(..., THREAD_AFFINITY_POLICY, ...)` with affinity tags to group workers on performance cores.
- Topology is detected via `sysctlbyname("hw.perflevel0.physicalcpu", ...)`.
- Spin-wait pauses use the ARM instruction `isb` (`__asm__ __volatile__("isb")`) instead of x86 `_mm_pause()`.

### 4. SIMD & Vector Math (`src/kernels/cpu/`)
- x86 AVX2 / AVX-512 instructions are conditionally compiled only on `x86_64`.
- On ARM64 (`arm64`), portable dot products and Apple Accelerate BLAS (`cblas_sdot`) / ARM NEON instructions are used for MoE activation.

### 5. Multimodal Vision Encoder (`tools/vision/`)
- `strata-vision` compiles natively with Apple's `Metal.framework` and `Accelerate.framework`.
- Metal compute shaders accelerate image encoding directly on Apple Silicon GPUs.

### 6. Metal GPU Engine (`tools/strata_runner.py`, `engine/strata`)
- On Apple Silicon Macs, inference is accelerated natively via Apple Metal compute shaders with Unified Memory Architecture (UMA).
- `engine/strata` implements the resident Strata Engine IPC protocol (`--serve`, `GEN`, `T`, `DONE`, `STOP`, `QUIT`) interfacing with `serve/server.py`.
- Supports full offloading of active weights and KV cache to Apple Silicon Metal GPU cores for high token throughput across both Qwen and Kolibri-1 MoE models.
- A model that fits the GPU budget below runs fully on it (`--fit on`). A larger one runs as whole layers: the output and the last layers on the GPU, the first ones on the CPU (`--fit off -ngl N`). llama.cpp's `--fit` would keep every layer's attention on the GPU and move experts to the CPU, and the switches cost more than the work: on an M1 Max with OLMoE-1B-7B, 55.8 tok/s against 92.8 with whole layers at the same GPU memory ([bench/results/2026-10-04-macos-split](../bench/results/2026-10-04-macos-split/README.md)).
- A model larger than the memory (Kolibri-1 Q4_K_M on a 32 or 48 GB Mac) reads the experts it does not hold through the page cache from the SSD, and every GPU layer holds all of its experts, rarely used ones too, and takes that memory from the page cache. Kolibri-1 on a 32 GB M1 Max with 0 / 6 / 12 GPU layers: 25.7 / 26.1 / 8.2 tok/s, 0.5 / 9.3 / 81.9 MB per token read from the SSD. So the runner starts with 20% of the memory left after the reserve for GPU layers and **tunes the count from the replies**: each count is measured over 256 output tokens, then, 20 s after the last reply, llama-server restarts with one step more (when the SSD is quiet) or fewer GPU layers, as long as that is faster; it ends at the fastest count (the fewest GPU layers within 3% of it) and keeps it in `engine/metal-tune.json` per model and memory size, so the next start begins there. A request during such a restart waits for it (20-45 s). The log says each step (`[strata] layers: ...`). On the 32 GB M1 Max it went 5 -> 9 -> 1 and kept 1 GPU layer: 31.7 tok/s, 0.4 MB per token from the SSD. `"env": {"STRATA_GPU_LAYERS": "<n>"}` sets the count by hand (no tuning), `"STRATA_TUNE": "0"` keeps the start count; delete the file's entry to tune again. `python3 tools/mac_split_bench.py <model.gguf> 0 4 8` measures counts by hand.
- **Pick the count from the running server, not from `llama-server` alone.** `tools/mac_split_bench.py` measures llama-server by itself, with nothing else resident and one prompt at temperature 0; both make a high layer count look better than it is, because what a GPU layer costs is the page cache the CPU-side experts are read through, and `serve/server.py` and the runner need memory too. Kolibri-1 Q4_K_M on a 48 GB M5 Pro, `-c 131072`: at 9 GPU layers the tool and the server agree (28.3 against 26.5-28.3 tok/s), at 30 they do not (43.7 against 15.7-22.5, alternating as paging comes and goes), and the server is fastest at the 1-5 layers the tool rates slowest. 30 layers wires 28.7 GiB, which with the ~15.5 GiB the CPU side reads through the cache is 44.2 of 48 GiB - enough standalone, not enough with the server beside it ([2026-10-05-macos-m5pro-split](../bench/results/2026-10-05-macos-m5pro-split/README.md)). So the tuner's `[strata] layers:` lines are the trustworthy source, and the tool is for ranking low counts.
- The size that counts is what has to be in memory: Qwen3.8-Flash-Next's per-layer embedding table (`per_layer_token_embd`, 28.8 GB of the Coder IQ1_M's 58.4 GB) stays in the file, and llama.cpp reads its rows while it answers (lazy read, tensors over 4 GiB). The Coder needs 29.6 GB of memory: on a 48 GB Mac it runs fully on the GPU; on 32-36 GB part of its experts come from the SSD (the PC's "~32 GB of RAM" is RAM beside a graphics card's own memory). The log's `[strata] memory:` line shows both parts.
- **On a 48 GB Mac, Kolibri-1 is worth quantizing down yourself.** Setup installs Q4_K_M (47.5 GB, 44.2 GiB in memory), which cannot go on the GPU whole at any Metal limit, so it runs as a split at ~32 tok/s. A Q3_K_M made locally from the official Q8_0 holds 34.9 GiB, fits the stock working set with `-c 65536`, and writes **56.7 - 60.2 tok/s** - nearly twice as fast, and not distinguishable from Q4_K_M on a handful of German tasks (`llama-quantize`'s own note: +0.66 ppl at Llama-3-8B, against Q3_K_S's +1.63). Build the tool with `cmake --build build-llama --target llama-quantize`; a split input needs no merging, because the loader finds the other shards from `split.count`. Numbers and the exact command: [2026-10-05-macos-kolibri-3bit](../bench/results/2026-10-05-macos-kolibri-3bit/README.md).
- **A model that nearly fits wants every layer on the GPU, not a split.** There is a threshold where almost every tensor in the file belongs to the GPU's buffer and llama.cpp maps the file into it instead of copying the GPU's weights out of it (`metal-split-mmap.patch` does the copying below that point). Above the threshold the weights stay file-backed and stop competing for dirty memory; below it each GPU layer costs a copy. Qwen3.8-Flash-Next Q2_0 on a 48 GB M5 Pro, `-c 32768`: 44 layers give 23.8 tok/s with a 32.2 GiB footprint and a 31.7 s load, 47 give 29.2 at 0.9 GiB and 11.8 s, and 49 give 33.4 ([2026-10-05-macos-qwen-q2_0](../bench/results/2026-10-05-macos-qwen-q2_0/README.md)). So for such a model set `STRATA_GPU_LAYERS` to all of them and make the context fit, rather than letting the layer tuner work: 24 layers at `-c 131072` gave 11.8 tok/s, 49 at `-c 65536` give 34.0.
- Qwen3.8-Flash-Next is the model this matters for, because the difference between its quantizations is almost all in what has to be in memory, not in speed: `Q2_0` holds 35.0 GiB and `IQ2_XS` 36.5 GiB, and at the same layer count they are within 10% of each other (24 layers: 11.8 against 10.6 tok/s). The 1.5 GiB is what lets all 49 layers onto the GPU at the stock Metal limit. `Q2_0`'s Metal kernels are much cheaper per weight - a scale and per-bit adds against IQ2_XS's codebook lookups - and that turns out not to be what the model waits for.
- `-c 131072` at 49 layers does not fit the stock 37.44 GiB working set, `-c 65536` does, and the two run at the same speed (33.8 against 34.0). `sudo sysctl iogpu.wired_limit_mb=41984` (41 GiB, ~7 GiB left for macOS, gone after a reboot) makes the full context fit. KV quantization is not a lever here: q4 and int8 measure the same, so use int8.
- Threads: the cores of the fastest performance level (`hw.perflevel0`; on an M5 Pro that is the 5 cores macOS calls "Super", beside 10 it calls "Performance"). Every step waits for its slowest thread: on an M5 Pro Kolibri-1 wrote ~15 tok/s with 15 threads (9 GPU layers), and ~27 with 5 threads at the same 9 layers in the server ([2026-10-05-macos-m5pro-split](../bench/results/2026-10-05-macos-m5pro-split/README.md)).
- GPU and CPU share one memory, so the runner gives the GPU a budget (`--fit-target`) from the Mac's memory: 8 GiB stay for macOS, the server and the browser; a model that fits the rest (plus 4 GiB for KV cache and buffers) goes on the GPU whole, up to its working set (a larger one runs as whole layers, below; if its layer table cannot be read, 60% of the rest goes on the GPU through `--fit`). llama.cpp's own plan used the GPU's whole working set (~36 GiB) and the first request failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. The log's `[strata] memory:` line shows the numbers; `STRATA_GPU_BUDGET_GB` (or `LLAMA_ARG_FIT_TARGET`) in the config's `"env"` sets them by hand. The 8 GiB / 4 GiB are starting values, not yet measured on many Macs. When the model does not fit, the runner also passes `--no-repack`: repacking copies the CPU-side experts into memory macOS can only swap, where the mmap lets it drop them and read them again from the file.
- With the experts split, stock llama.cpp still mapped the model file into the GPU's buffer from its first tensor to its last, the CPU-side experts included, and Metal keeps that resident (OLMoE-1B-7B Q4_K_M on an M1 Max, --fit-target 25600: a 3961 MiB GPU buffer for a plan of 1420 MiB). Setup applies `third_party/patches/metal-split-mmap.patch`, which copies the GPU's own weights into its buffer in that case (1420 MiB) and leaves the rest mapped.
- llama.cpp has no `kolibri1` architecture yet: setup applies `third_party/patches/kolibri1-llama.cpp.patch` (from the [Kolibri-1 GGUF](https://huggingface.co/Hob-forge/Kolibri-1-GGUF), without its Python converter) before building `llama-server`. Without it the engine stops with "unknown model architecture: 'kolibri1'".

### 7. Upstream 0.1.39 on macOS
- Work as on a PC (server and web app): the OpenAI Responses API (`/v1/responses`, Codex CLI), the thinking levels, the Settings view, the Unsloth UD-IQ4_XS model in setup's menu.
- Not on the Metal engine, which is llama-server behind `tools/strata_runner.py`: several requests at once (`"parallel"`; the runner reports no batch slots, so requests run one at a time as before), the elastic expert cache (`VRAM`, answered with `ERR`), `"effort_position": "end"` (needs the native engine's `--tail-role-token`; the server says so and keeps the default), and the decode / long-prompt / multi-GPU speedups of the native CUDA and HIP engines.
- `src/kernels/cpu/kq_avx1.cpp` (new in 0.1.39, the AVX1 router dot for older x86 CPUs) and the new CPU probes `cpu_avx1_ok` / `cpu_sse42_ok` compile to stubs on arm64; the CPU thread pool keeps its Mach affinity tags beside 0.1.39's new Windows CPU Sets and Linux affinity code.

---

## Building by Hand

To build the macOS binaries manually:

```bash
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
cmake -B build -DSTRATA_BUILD_TESTS=ON
cmake --build build --target strata-gguf strata-plan strata-dequant strata_kernels_cpu direct_file_async_test platform_memory_test expert_multi_test pool_test pool_stress
```

To run the unit tests:

```bash
./build/direct_file_async_test
./build/platform_memory_test
./build/expert_multi_test
./build/pool_test --selftest
./build/pool_stress 5
```
