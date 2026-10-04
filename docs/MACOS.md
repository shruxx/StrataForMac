# Strata on macOS (Apple Silicon)

Strata runs on Apple Silicon Macs (M1, M2, M3, M4 — base, Pro, Max, Ultra) under macOS.

Apple Silicon features Unified Memory Architecture (UMA), meaning CPU and GPU share the same physical memory pool with high bandwidth (e.g. 400 GB/s on M1 Max). This fits Strata's hybrid MoE design naturally: expert activations can be computed with ARM NEON and Apple Accelerate BLAS without PCIe transfer bottlenecks.

---

## What you need

- **Mac:** Any Apple Silicon Mac (M1/M2/M3/M4 family).
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
