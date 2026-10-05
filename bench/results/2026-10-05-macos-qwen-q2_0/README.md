# Qwen3.8-Flash-Next on a 48 GB M5 Pro: from 10.6 to 34 tok/s

MacBook Pro M5 Pro (Mac17,9), 48 GB, 5 `hw.perflevel0` cores, 16 GPU cores, macOS 27.0.1, Metal working set
37.44 GiB unless a row says otherwise. llama.cpp build 782 / commit 6d7085d with `third_party/patches`.

Measured **through `serve/server.py`** (`--engine strata`), which is the only way these numbers mean anything here
(see [2026-10-05-macos-m5pro-split](../2026-10-05-macos-m5pro-split/README.md): `llama-server` on its own reads up
to twice too fast at high GPU layer counts). Per run: `"env": {"STRATA_GPU_LAYERS": N, "STRATA_TUNE": "0"}`, five
varied German prompts at temperature 0.7, 256 tokens each, the first discarded as the warm-up. Speed is llama.cpp's
own `eval time` (decode only). "SSD" is `proc_pid_rusage` `ri_pageins` x 16 KiB per output token, "footprint" is
`ri_phys_footprint`.

## The answer

| | GPU layers | context | KV | tok/s |
|---|---:|---:|---|---:|
| Before: IQ2_XS as the runner planned it | 24 | 131072 | int8 | 10.6 |
| Q2_0, same settings | 24 | 131072 | int8 | 11.8 |
| **Q2_0, all layers on the GPU** | **49** | **65536** | **int8** | **34.0** |
| the same with `iogpu.wired_limit_mb=41984` | 49 | 131072 | int8 | 33.8 |

Of the 3.2x, the quantization is worth ~10% and the layer count is worth the rest. The 131072 row needs the raised
`iogpu.wired_limit_mb` (41 GiB, ~7 GiB left for macOS, gone after a reboot); 65536 does not, and is not slower.

## Q2_0 against IQ2_XS at the same settings

`-c 131072`, KV int8. Decode tok/s over the four measured replies.

| GPU layers | IQ2_XS | Q2_0 |
|---:|---|---|
| 9 | 7.2, 7.5, 7.8, 8.0 | 7.5, 7.9, 8.5, 8.6 |
| 24 | 10.3, 10.6, 10.7, 10.8 | 11.1, 11.6, 11.9, 12.0 |
| 47 | out of memory on load | `kIOGPUCommandBufferCallbackErrorOutOfMemory` on the first request |

**~10%, not the several-fold the Metal kernels suggested.** `mul_mv.metal` computes the Q2_0 dot product as
`d * (sum_lo(y) + 2*sum_hi(y) - sumy)` over per-bit conditional adds - one scale and a few bit tests - while IQ2_XS
looks codebook entries up per weight group. That difference is real and nearly irrelevant to the result: the
dequantization is not what this model is waiting for.

What Q2_0 *is* needed for: **1.5 GiB less to hold** (35.0 GiB against 36.5 GiB, both plus a 26.8 GiB table left in
the file). That is what lets all 49 layers onto the GPU at all.

## Why the layer count is worth 3x: llama.cpp maps the file instead of copying it

Q2_0, `-c 32768`, KV q4:

| GPU layers | tok/s | MB per token from the SSD | footprint | load time |
|---:|---|---:|---:|---:|
| 44 | 23.0, 23.7, 23.8, 24.3 | 1.2 - 8.5 | 32.2 - 33.1 GiB | 31.7 s |
| 47 | 27.5, 29.2, 29.7, 29.9 | 0.2 - 0.3 | 0.9 - 1.8 GiB | 11.8 s |
| 48 | 31.3, 31.4, 31.5, 32.0 | 0.2 | 0.9 - 1.8 GiB | |
| 49 | 33.0, 33.1, 33.4, 34.2 | 0.2 - 0.3 | 0.7 - 1.6 GiB | |

Between 44 and 47 layers the footprint falls by 31 GiB and the load gets 20 s shorter. That is
`third_party/patches/metal-split-mmap.patch` switching over: with some layers' experts on the CPU it copies the
GPU's weights into a Metal buffer (31.7 s of copying, 32 GiB of anonymous memory), and once nearly every tensor in
the file belongs to the GPU buffer it maps the file instead (11.8 s, and the pages stay file-backed, which is why
the footprint reads ~1 GiB for a 35 GiB model). Mapped, the weights also stop competing with everything else for
dirty memory.

So the lever on a Mac is not "how much of the model on the GPU" by the GiB - it is **getting over the threshold
where the whole file maps**. Below it, more GPU layers cost a copy; above it, they are free.

## The 26.8 GiB table is not the bottleneck either

Qwen3.8-Flash-Next ships `per_layer_token_embd.weight`, shape `(160, 320001536)` - 320 million rows of 160 values,
28.8 GB on disk - which llama.cpp leaves in the file and reads from (lazy read, tensors over 4 GiB). Strata's native
engine has purpose-built kernels for it (`src/kernels/ngram.cpp`, `PLE_TABLE_ROWS`) and the config passes
`--ple-gguf`; the macOS runner ignores that flag because llama.cpp has nothing equivalent.

It still costs almost nothing here: **0.2 - 0.3 MB per token** at 47 - 49 GPU layers. Setup even hard-links the
file, so IQ2_XS and Q2_0 read the *same inode* - which is also why their speeds track each other so closely, and
why deleting IQ2_XS afterwards freed only its own 37 GB shard 1 and left Q2_0 intact.

## Answer quality

Five tasks at temperature 0, both quantizations, `-c 65536`: average speed from a word problem (both 90 km/h, with
different but valid derivations), three sentences on Alexander von Humboldt (both accurate), a palindrome function
in Python (both correct), a German relative-pronoun correction (Q2_0 produced the right "das du mir ..."; both hit
the token limit mid-answer), and splitting 17 apples three unequal ways (both 4 + 5 + 8). No visible difference.
Five prompts is not a benchmark - this rules out obvious damage, nothing more.

## What did not help, and what was not tried

- **KV quantization is not a lever**: at 49 layers and `-c 32768`, q4 gives 33.0 - 34.2 and int8 33.4 - 34.6.
  Use int8.
- **Context is only a memory question**: 65536 and 131072 run at the same speed once they fit (34.0 against 33.8).
  At the stock Metal limit 131072 does not fit at 49 layers, 65536 does.
- The runner's `CTX_ALLOWANCE` of 4 GiB is what it plans the layer count against, and at `-c 131072` on this model
  it is too little: the runner picks 47 layers, the weights fit the working set, and the first request dies with
  `kIOGPUCommandBufferCallbackErrorOutOfMemory`. A model whose weights land this close to the working set needs
  either a smaller context or the raised limit, and the runner cannot currently tell the user which.
- Not tried: MTP (the runner ignores `--mtp`; upstream reports 1.6-1.8x), llama.cpp's own draft-model speculative
  decoding, and the Coder IQ1_M, which carries the same 28.8 GB table but has half the experts.
