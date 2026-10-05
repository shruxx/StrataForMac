# GPU/CPU split on an M5 Pro, 48 GB: how the measurement changes the answer

MacBook Pro M5 Pro (Mac17,9), 48 GB, 15 CPU cores (5 "Super" = `hw.perflevel0`, 10 "Performance" =
`hw.perflevel1`), 16 GPU cores, macOS 27.0.1, Metal working set 37.44 GiB (`iogpu.wired_limit_mb` at its default
0). llama.cpp build 782 / commit 6d7085d with `third_party/patches`, `--fit off -ngl N --no-repack`, 5 threads
(the runner's `get_perf_cores`), `-c 131072` (the context setup configures) unless a row says otherwise.
Numbers are output tokens/s. The previous session's 32 GB M1 Max results are in
[2026-10-04-macos-split](../2026-10-04-macos-split/README.md); layer counts do not carry over between Macs.

**The headline is a warning, not a speed.** Measuring `llama-server` on its own says Kolibri-1 wants 30 GPU layers
and runs at 43-52 tok/s. In the server the user actually runs, 30 layers gives 16-22 and the best count is 1-5, at
~32. The two methods agree at low counts and diverge by a factor of two at high ones, so a count picked from the
standalone tool is the wrong count.

## The A/B that settles it: the same server, the same prompts, two layer counts

`serve/server.py --engine strata` with `"env": {"STRATA_GPU_LAYERS": N}`, five 320-token replies to five varied
German prompts at temperature 0.7, the first discarded as the warm-up. Decode speed as llama.cpp reports it
(`eval time`), and wall-clock through the HTTP API beside it.

| GPU layers | decode tok/s (4 replies) | wall tok/s |
|---:|---|---|
| 9 | 28.1, 26.7, 28.3, 26.5 | 17.8 - 23.6 |
| 30 | 21.5, 15.7, 22.5, 15.9 | 11.9 - 15.2 |

Nine layers wins, and at 30 the replies alternate between ~16 and ~22 tok/s - the signature of paging that comes
and goes. The runner's own tuner, measuring real replies in the same server earlier the same day, puts the top
lower still:

| GPU layers | tok/s | MB per token from the SSD | |
|---:|---:|---:|---|
| 9 (its start count) | 20.3 | 36.4 | SSD busy: a step down |
| 5 | 32.4 | 2.7 | faster: another step down |
| 1 | 32.2 | 1.1 | within 3% of the fastest, fewest layers: kept |

So in the server the curve falls as GPU layers rise: ~32 tok/s at 1-5, ~27 at 9, ~18 at 30. **The tuner's answer of
1 layer is right**, and the heuristic that produced it - a busy SSD at the start count means fewer layers - read
this Mac correctly.

### Why the standalone tool disagrees

| GPU layers | `llama-server` alone | in the server | wired for Metal |
|---:|---:|---:|---:|
| 9 | 28.3, and 26.1 when repeated last | 26.5 - 28.3 | 10.7 GiB |
| 30 | 43.7 (47.9 in an earlier run) | 15.7 - 22.5 | 28.7 GiB |

At 9 layers the two agree to within the noise. At 30 they do not, and the difference is memory: 28.7 GiB wired for
Metal plus the ~15.5 GiB of weights the CPU side reads through the page cache is 44.2 of 48 GiB. Standalone, the
few hundred MB left over are enough; with `serve/server.py` and `tools/strata_runner.py` also resident they are
not, and the CPU-side experts start coming off the SSD again. Wired GPU memory cannot be paged out, so everything
else gives way.

The standalone figures are reproducible - 9 layers measured first and last in the same run gave 28.3 and 26.1, and
30 gave 43.7 against 47.9 in an earlier run - so this is not drift. It is a benchmark that leaves out part of the
system it is predicting.

## Kolibri-1 Q4_K_M, 47.5 GB on disk (44.2 GiB in memory), 50 layers + output: the standalone curve

Kept for the record, with the caveat above. Eight varied prompts at temperature 0.7, three warm-up and five
measured 320-token replies per count. "SSD" is `proc_pid_rusage` `ri_pageins` x 16 KiB per output token
(`ri_diskio_bytesread` agreed to one decimal in every row); "footprint" is `ri_phys_footprint` at the end.

| GPU layers | tok/s (median) | min | max | MB per token from the SSD | footprint |
|---:|---:|---:|---:|---:|---:|
| 1 | 32.0 (32.5 repeated last) | 27.2 | 33.2 | 3.2 | 3.1 GiB |
| 9 | 28.3 (26.1 repeated last) | 0.0 | 36.8 | 10.7 | 10.7 GiB |
| 24 | 43.6 | 21.0 | 46.9 | 3.5 | 23.6 GiB |
| 30 | 47.9 / 43.7 | 16.1 | 50.5 | 4.6 | 28.7 GiB |
| 33 | 46.1 | 12.7 | 49.5 | 5.9 | 31.3 GiB |
| 36 | 40.6 | 14.1 | 47.9 | 10.2 | 33.8 GiB |

Only the 1-layer row predicts the server (32.0 against the tuner's 32.2). Everything above 9 layers is optimistic,
the more so the higher the count.

With **one** prompt at temperature 0 and `-c 4096` (what `tools/mac_split_bench.py` does) the same model reads
0.0 MB per token at every count and gives 38.5 (0 layers), 40.4 (12), 47.1 (24), 52.0 (30), 14.0 (36), 14.7 (42).
A fixed greedy reply picks nearly the same 6 of each layer's 384 experts every time, so they never leave the page
cache. The collapse at 36 there does not reproduce under varied prompts (40.6).

## Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, 68.0 GB on disk, 48 layers + output

36.5 GiB has to be in memory; the other 28.8 GB are the `per_layer_token_embd` table, which llama.cpp leaves in the
file (lazy read, tensors over 4 GiB). Standalone, varied prompts, `-c 131072`:

| GPU layers | tok/s (median) | MB per token from the SSD | footprint |
|---:|---:|---:|---:|
| 1 | 6.5 | 5.6 | 5.8 GiB |
| 24 | 9.5 | 14.0 | 23.6 GiB |
| 30 | 7.3 | 49.4 | 28.2 GiB |

Not measured in the server, so read the 24-layer row as an upper bound like Kolibri's.

**The disk is not what holds this model back, and neither is the layer split.** The comparison that carries is at
1 GPU layer, the count where the standalone tool and the server agree: Qwen reads 5.6 MB per token and writes 6.5
tok/s, Kolibri-1 reads a comparable 3.2 MB per token and writes 32.0 - five times as fast, with *more* active
parameters per token (3.46B against ~2.4B). At comparable disk load the gap is still fivefold, so what is left
between them is the quantization: Q4_K_M scales blocks, IQ2_XS looks codebook entries up. The ~10 tok/s seen in
normal use is this file's ceiling on this Mac, not a misconfiguration.

Not measured: a k-quant of the same model (`Q2_0`, 66.4 GB) side by side, which is the test that would settle it,
and MTP, which the runner does not pass on (`--mtp` is ignored; upstream reports 1.6-1.8x).

`-ngl 49` fails to start although its 36.2 GiB of weights are below the 37.44 GiB working set: the KV cache and the
compute buffers come on top. Raising `iogpu.wired_limit_mb` to fit it is unmeasured, and on this evidence - every
count above ~9 layers losing to the page cache it takes - a win is unlikely.

## What this changed in the code

- `tools/mac_split_bench.py`: its docstring now says it measures `llama-server` alone and overestimates high layer
  counts, with the numbers above. Two real bugs beside that: it printed a "runner's count" computed differently
  from the runner (26 against the runner's 10 for Qwen, because it ignored the `START_GPU_SHARE` path a model read
  from the SSD takes), and it left `--threads` to llama.cpp instead of passing the runner's count.
- `LayerTuner` was **not** changed. A change to make it always probe upwards first was written, tested and then
  reverted: it was justified by the standalone curve's peak at 30 layers, which the A/B above shows the server
  never sees. The existing rule - a busy SSD at the start count means walk towards fewer layers - is the one that
  gets this Mac right.
- The per-token SSD figure the tuner decides on (`process_pageins`) reads correctly on macOS 27: 0.0 where the
  working set fits the page cache, 53.9 where it does not. Its struct offsets are right - `ri_phys_footprint`, a
  field behind the ones read, tracks the wired GPU memory to the GiB in every row above.

## For the next measurement on a Mac

Measure through `serve/server.py`, not `llama-server` alone, whenever the answer depends on how much memory is
left over - which is every model that does not fit the Mac whole. The runner's tuner already does this, and its
`[strata] layers:` lines are the cheapest trustworthy source of a layer count there is.
