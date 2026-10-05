# Kolibri-1 at 3 bits on a 48 GB M5 Pro: 32 -> 59 tok/s

MacBook Pro M5 Pro (Mac17,9), 48 GB, 5 `hw.perflevel0` cores, 16 GPU cores, macOS 27.0.1, Metal working set
37.44 GiB (stock) unless a row says otherwise. llama.cpp build 782 / commit 6d7085d with `third_party/patches`.
Measured through `serve/server.py --engine strata` with `"env": {"STRATA_GPU_LAYERS": N, "STRATA_TUNE": "0"}`,
five varied German prompts at temperature 0.7, 256 tokens each, the first discarded. Speed is llama.cpp's own
`eval time`. "SSD" is `proc_pid_rusage` `ri_pageins` x 16 KiB per output token, "footprint" is `ri_phys_footprint`.

## The result

| | on disk | in memory | GPU layers | tok/s | SSD/token | footprint |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M (what setup installs) | 47.5 GB | 44.2 GiB | 1 (the tuner's pick) | ~32 | 1.1 MB | |
| Q4_K_M | | | 9 | 26.5 - 28.3 | | 10.7 GiB |
| **Q3_K_M** | 37.5 GB | 34.9 GiB | **51 (all)** | **56.7 - 60.2** | 0.0 | 0.9 GiB |
| Q3_K_S | 33.9 GB | 31.5 GiB | 51 (all) | 62.4 - 63.1 | 0.0 | 0.9 GiB |

`-c 65536`, KV int8. Q3_K_M measured at the stock 37.44 GiB working set; Q3_K_S and a second Q3_K_M run were taken
at `iogpu.wired_limit_mb=41984` and agree with it (59.4 - 59.8), so the limit does not change the speed, only
whether the model fits.

The Q4_K_M figure of ~32 tok/s is the layer tuner's own measurement (two replies at 1 GPU layer) rather than a
five-reply A/B like the rows below it; the 9-layer row is the A/B. Either way the 3-bit files are close to twice
as fast.

**Why:** the same threshold as
[2026-10-05-macos-qwen-q2_0](../2026-10-05-macos-qwen-q2_0/README.md). At 44.2 GiB Kolibri-1 Q4_K_M cannot go on
the GPU whole on a 48 GB Mac at any limit, so it runs as a split, reads experts through the page cache, and
`metal-split-mmap.patch` copies the GPU's weights out of the file. Under ~35 GiB every tensor belongs to the GPU
buffer, llama.cpp maps the file instead, and the footprint drops to 0.9 GiB for a 35 GiB model while the SSD goes
quiet. Kolibri-1 was always the faster model per token - at **1** GPU layer, with the CPU doing nearly all of it,
it already matched Qwen Q2_0 running entirely on the GPU - it was only ever held back by not fitting.

## How the files were made

The official [Hob-forge/Kolibri-1-GGUF](https://huggingface.co/Hob-forge/Kolibri-1-GGUF) has Q4_K_M and Q8_0 and
nothing smaller, so both 3-bit files were quantized locally from its Q8_0 (two shards, 83.1 GB):

```
./engine/llama-quantize --allow-requantize \
    .../kolibri-Q8_0/Kolibri-1-Q8_0-00001-of-00002.gguf \
    .../kolibri-Q3_K_M/Kolibri-1-Q3_K_M.gguf  Q3_K_M  10
```

- `llama-quantize` is not in `engine/` after a normal setup; build it with
  `cmake --build build-llama --target llama-quantize`.
- **No merge is needed for a split input.** `llama_model_quantize_impl` passes an empty `splits` list, and
  `llama_model_loader` then discovers the other shards from `split.count`
  (`src/llama-model-loader.cpp`, the `splits.empty()` branch). That keeps the peak disk need at 83 GB + the output
  instead of 83 GB + 83 GB + the output.
- `--allow-requantize` is required: Q8_0 is already quantized. It is the best basis available short of converting
  Aleph Alpha's safetensors, and at 8 bits it is close to lossless, but this is still requantization and the
  quality notes below should be read with that in mind.
- 143 s for Q3_K_M, 71 s for Q3_K_S, with 10 threads.

Q3_K_S came out at 33.9 GB - exactly the size of the third-party `Eliasfpv28/Kolibri-1-Q3_K_S-GGUF` quant that a
previous session had noted and the user had declined, which suggests that file was made the same way. Making it
locally removes the question of trusting it.

## Answer quality, six German tasks at temperature 0

Compared against Q4_K_M. Equal on all three tasks that completed within 500 tokens: interest on a loan (all three
derive 315 EUR correctly), a business-letter rewrite (all three idiomatic), and a Python list comprehension (all
three correct). Three tasks were cut off mid-reasoning and re-run with 1400 tokens:

- **Physics terms** (work vs power): Q4_K_M and Q3_K_M both give "Kraft über einen Weg" with Joule and Watt;
  Q3_K_S omits the force-over-distance part and reads thinner.
- **Idiom** ("jemandem einen Bären aufbinden"): all three hedge that the etymology is unsettled, which is correct.
  Q3_K_M's reading ("aufbinden meint hier anbinden") is the cleanest; Q3_K_S offers a card-game origin and
  Q4_K_M a circus bear-riding one, both doubtful. German idiom etymology is genuinely contested, so this
  separates the three less than it looks.
- **Regional facts** (rivers in Baden-Württemberg): only Q4_K_M finished (Neckar/Heidelberg, Donau/Ulm,
  Rhein/Karlsruhe, all correct). Both 3-bit files were still reasoning at 1400 tokens, so this one is not compared.

So: **Q3_K_M is not distinguishable from Q4_K_M on what could be compared, and Q3_K_S is consistently correct but
terser.** That matches `llama-quantize`'s own perplexity notes (+0.66 against +1.63 ppl at Llama-3-8B). Six prompts
are not a benchmark - this rules out obvious damage to a German model and no more. Q3_K_M at 5% below Q3_K_S's
speed is the better trade.

## What this means for setup

Setup installs Q4_K_M for the `kolibri` family and there is no smaller official file, so a Mac with 48 GB gets the
one size that cannot go on its GPU. Worth considering: offering a locally quantized 3-bit step for Apple Silicon,
or at least saying in `docs/MACOS.md` that it is worth making. Not changed in `setup.py` here - it would mean
shipping a quantization step, and the measurements above come from one Mac.
