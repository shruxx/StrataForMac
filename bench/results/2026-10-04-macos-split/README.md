# GPU/CPU split on macOS (llama-server, Metal)

MacBook Pro M1 Max, 32 GB, macOS 26.4 (`iogpu.wired_limit_mb` 27648). OLMoE-1B-7B-0924-Instruct Q4_K_M (3.9 GiB,
16 layers, 64 experts, 8 active), llama.cpp 3cf0325 with `third_party/patches`, `-c 4096`, greedy, 128 tokens after
a short prompt, median of three runs after a warm-up (`tools/mac_split_bench.py`, and by hand for the `--fit` rows).
Numbers are output tokens/s.

## Experts split vs whole layers, same GPU memory

| How the model is split | GPU weights | tok/s |
|---|---:|---:|
| `--fit on --fit-target 25600 --no-repack` (attention of every layer on the GPU, experts of 12 layers on the CPU) | 1315 MiB | 55.8 |
| `--fit off -ngl 5 --no-repack` (the output and 4 whole layers on the GPU) | 1301 MiB | 92.8 |
| `--fit off -ngl 6 --no-repack` | 1311 MiB | 96.2 |
| `--fit off -ngl 7 --no-repack` | 1570 MiB | 97.8 |
| `--fit off -ngl 9 --no-repack` | 2022 MiB | 96.6 |
| `--fit off -ngl 9` (CPU weights repacked) | | 86.3 |
| `--fit on --fit-target 25600` (CPU weights repacked) | | 54.2 |
| `--fit off -ngl 0 --no-repack` (all on the CPU) | 0 | 104.4 - 115.6 |
| whole model on the GPU | 3961 MiB | 179.9 - 180.1 |

With `--fit`'s split every token goes from the GPU to the CPU and back in every layer whose experts are on the CPU;
whole layers switch twice per token.  So the runner (`tools/strata_runner.py`) offloads whole layers when a model
does not fit its GPU budget.  On this small model a part on the GPU is slower than none: the GPU's share of the work
is too small to pay for the switch.  For a large model that is a question for each Mac:
`tools/mac_split_bench.py <model.gguf>` measures it, `STRATA_GPU_LAYERS` sets the result.

## The GPU buffer with a split (metal-split-mmap.patch)

`--fit-target 25600` (fit planned 1420 MiB for the GPU): stock llama.cpp mapped 3961 MiB into the GPU's buffer (the
file from the GPU's first tensor to its last, the CPU's experts included), with the patch 1420 MiB.

## Kolibri-1 Q4_K_M, a model larger than the memory

Same Mac (M1 Max, 32 GB), Kolibri-1 Q4_K_M (47.5 GB, 50 layers + output), `-c 4096`, `--no-repack`, 8 threads,
96 tokens after a 9-token prompt, median of three runs after a warm-up.  "SSD" is the pages read per output token
(`vm_stat` Pageins, 16 KiB each) during the measured runs.

| How | GPU weights | tok/s | MB per token from the SSD |
|---|---:|---:|---:|
| `-ngl 0` | 0 | 25.73 | 0.5 |
| `-ngl 6` | 4.9 GiB | 26.06 | 9.3 |
| `-ngl 12` | 10.2 GiB | 7.95 - 8.23 | 75.5 - 81.9 |
| `--fit on --fit-target 12902` (14.4 GiB budget, experts split) | 13.5 GiB | 5.15 | 123.1 |
| `-ngl 12` with an `madvise(MADV_WILLNEED)` hint for the experts of each CPU MoE op | 10.2 GiB | 6.86 | 114.4 |
| `--fit` split with that hint | 13.5 GiB | 4.93 | 131.9 |

Every GPU layer holds all 384 experts of its layer and takes that memory from the page cache the CPU's experts are
read through; with few GPU layers the experts the replies use stay in the page cache.  The `MADV_WILLNEED` hint made
the kernel read more than the op used and was dropped.

The runner's layer tuner (`tools/strata_runner.py`, LayerTuner), 96-token requests 5 s apart, minimum 128 tokens and 3 s
idle for this run (256 and 20 s by default):

| GPU layers | tok/s | MB per token from the SSD | |
|---:|---:|---:|---|
| 5 (the start: 20% of 24 GiB) | 30.95 | 1.05 | SSD quiet: one step up |
| 9 | 30.49 | 0.98 | not faster: the other way |
| 1 | 31.65 | 0.39 | within 3% of the fastest with the fewest GPU layers: kept |
