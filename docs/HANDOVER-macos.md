# Handover: the macOS port, state of 2026-10-05 (evening)

A note for the next Claude Code session and for the user. The user writes in German: answer in German. The session
before this one ran on a MacBook Pro M1 Max, 32 GB; this one ran on the target Mac.

## The Mac this now runs on

MacBook Pro M5 Pro (Mac17,9), 48 GB, 15 CPU cores (5 "Super" = `hw.perflevel0`, 10 "Performance" =
`hw.perflevel1`), 16 GPU cores, macOS 27.0.1, Metal working set 37.44 GiB by default, ~137 GB free on the SSD.
Working folder: `~/Documents/Projekte/StrataFormacOS`, a git clone of `shruxx/StrataForMac` (remote `upstream` =
`Niko1221/Strata`). Models in `~/Documents/Strata-data`: Kolibri-1 Q4_K_M and Qwen3.8-Flash-Next **Q2_0**.
IQ2_XS was deleted on the user's say-so once Q2_0 had been measured against it (38 GB freed; its shard 2 was a hard
link shared with Q2_0's, so only its own shard 1 and pack actually went). Q2_0 was verified to still load and answer
afterwards. Re-downloading IQ2_XS is `./setup.sh --setup --family qwen --model IQ2_XS` and ~10 minutes.

One leftover, not in the way: `~/Documents/StrataForMac-main` is the old ZIP download (1.3 GB, no `.git`) and holds
nothing this clone needs.

### What had to be repaired first

The clone was a **copy of the M1 Max's folder**, made under the user name `admin`, so everything machine-specific
in it pointed at a Mac that is not this one:

- `.venv` pointed at `/Users/admin/.local/share/uv/python/...`. Repaired by pointing `.venv/bin/python3.13` at
  Homebrew's 3.13 and running `python3.13 -m venv .venv` over it, which keeps the site-packages. The console
  scripts kept the old shebang too, so `cmake` and `ninja` were reinstalled at their pinned versions.
- `strata-kolibri-q4_k_m.json` pointed at `/Users/admin/...` and at a `--native` GGUF inside a deleted scratchpad.
  Rewritten for this Mac. `strata-iq2_xs.json` did not exist here at all (only in the ZIP folder) and was written.
  Both are gitignored, so they are per-machine and no commit carries them.
- `engine/` and `third_party/llama.cpp` came over intact: both patches are applied in the tree and
  `engine/llama-server` is build 782 / commit 6d7085d, which matches. Nothing had to be rebuilt.

So "run `./setup.sh`" from the old handover was **not** needed in the end, and was not run: a full setup would have
re-walked downloads and conversions for 108 GB that are already correct. `./setup.sh --check` (read-only) passes and
sees the Mac correctly.

### Tests on this Mac

All green: `tools/test_strata_runner.py` (14), `tools/test_strata_mcp.py` (23), all 18 `tools/test_setup_*.py`,
and `python -m pytest serve/` (259 passed). The only failures are the three the old handover already named and that
fail the same way on plain upstream: `AmdTelemetry` (2) and `test_responses::test_json_schema_text_format`.
`pytest` is not in the pinned list and was installed by hand.

## What was measured, and the mistake worth not repeating

Everything is in [bench/results/2026-10-05-macos-m5pro-split](../bench/results/2026-10-05-macos-m5pro-split/README.md).
**Read its first two sections before trusting any Mac layer-count number, this session's included.**

This session spent hours measuring GPU layer counts with `llama-server` on its own, concluded Kolibri-1 wants 30
layers at ~48 tok/s, set `STRATA_GPU_LAYERS: 30`, "fixed" the tuner for not finding it - and then found all of
that wrong when the same counts were measured **through `serve/server.py`**:

| GPU layers | `llama-server` alone | in the server |
|---:|---:|---:|
| 1 | 32.0 | 32.2 (the tuner's own number) |
| 5 | not measured | 32.4 (the tuner's) |
| 9 | 28.3, 26.1 repeated | 26.5 - 28.3 |
| 30 | 43.7 - 47.9 | 15.7 - 22.5, alternating |

The two agree at low counts and diverge by a factor of two at high ones. A GPU layer costs the page cache the
CPU-side experts are read through: 30 layers wires 28.7 GiB, which with the ~15.5 GiB read through the cache is
44.2 of 48 GiB - enough with nothing else resident, not enough with the server and the runner beside it. Wired
GPU memory cannot be paged out, so everything else gives way, and the replies alternate between ~16 and ~22 tok/s
as paging comes and goes. The standalone numbers are reproducible (9 layers measured first and last in one run:
28.3 and 26.1), so this is not drift - the benchmark simply leaves out part of the system it is predicting.

**So in the server the curve falls as GPU layers rise: ~32 tok/s at 1-5, ~27 at 9, ~18 at 30, and the tuner's
answer of 1 layer was right all along.** Its heuristic - a busy SSD at the start count means walk towards fewer
layers - reads this Mac correctly. `engine/metal-tune.json` holds that result (`best: 1`) and
`strata-kolibri-q4_k_m.json` has **no** `STRATA_GPU_LAYERS`: nothing to undo, the walk-back is complete. The
`LayerTuner` change was reverted and `tools/strata_runner.py` is identical to HEAD.

Next time: measure through the server whenever the answer depends on how much memory is left over, which is every
model that does not fit the Mac whole. `tools/mac_split_bench.py`'s docstring now says so.

### Kolibri: 32 -> 59 tok/s, by quantizing it down to fit

Full numbers in [2026-10-05-macos-kolibri-3bit](../bench/results/2026-10-05-macos-kolibri-3bit/README.md). Same
mechanism as the Qwen work below: Q4_K_M's 44.2 GiB cannot go on a 48 GB Mac's GPU at any Metal limit, so it runs
as a split and copies rather than maps. The official repo has nothing smaller than Q4_K_M, so **Q3_K_M and Q3_K_S
were quantized locally from the official Q8_0** (83.1 GB, downloaded for this; `llama-quantize` built from
`build-llama`, `--allow-requantize`, no merge needed for the split input).

| | in memory | GPU layers | tok/s |
|---|---:|---:|---:|
| Q4_K_M (what setup installs) | 44.2 GiB | 1 | ~32 |
| **Q3_K_M - now the default** | 34.9 GiB | 51 (all) | **56.7 - 60.2** |
| Q3_K_S | 31.5 GiB | 51 (all) | 62.4 - 63.1 |

`strata-kolibri-q3_k_m.json` is the config to use, at `-c 65536`, and it was verified at the **stock** Metal limit
(the log says "GPU working set 37.4 GiB"), so no `sysctl` and nothing to redo after a reboot. Q3_K_S is 5% faster
and reads consistently terser on German tasks; Q3_K_M is not distinguishable from Q4_K_M on what could be compared,
which is why it won. All three configs and matching `run-*.sh` scripts are in place, so the web app can switch
between them.

`run-kolibri-q4_k_m.sh` was another `/Users/admin` leftover from the M1 Max copy and would have failed on a double
click; all `run-*.sh` were rewritten for this Mac.

### Qwen: solved, 10.6 -> 34 tok/s

Full numbers in [2026-10-05-macos-qwen-q2_0](../bench/results/2026-10-05-macos-qwen-q2_0/README.md). The user asked
for this after the Kolibri work, and it came out well:

| | GPU layers | context | KV | tok/s |
|---|---:|---:|---|---:|
| Before: IQ2_XS as the runner planned it | 24 | 131072 | int8 | 10.6 |
| Q2_0, same settings | 24 | 131072 | int8 | 11.8 |
| **Now, in `strata-q2_0.json`** | **49** | **65536** | **int8** | **34.0** |
| with `iogpu.wired_limit_mb=41984` | 49 | 131072 | int8 | 33.8 |

`Q2_0` was downloaded this session (`./setup.sh --setup --family qwen --model Q2_0 --vision no --yes --no-start`);
only 37.6 GB came down, because setup hard-links shard 2 (the 26.8 GiB table) with IQ2_XS's - the same inode.

**Two hypotheses this session got wrong before the right one.** First the quantization: IQ2_XS's codebook lookups
against Q2_0's per-bit adds looked like the answer from the Metal kernels, and it is worth ~10%. Then the 26.8 GiB
`per_layer_token_embd` table: it reads 0.2 - 0.3 MB per token once the model is on the GPU, so it is not the cost
either. What it actually was: the threshold where llama.cpp **maps** the model file into the Metal buffer instead
of **copying** the GPU's weights out of it (`metal-split-mmap.patch` does the copying below it). Q2_0 at `-c 32768`:
44 layers 23.8 tok/s, 32.2 GiB footprint, 31.7 s to load; 47 layers 29.2 tok/s, 0.9 GiB, 11.8 s. Q2_0 is still
needed, because its 35.0 GiB (against IQ2_XS's 36.5) is what lets all 49 layers on at the stock limit.

So for a model that nearly fits a Mac: **set `STRATA_GPU_LAYERS` to every layer and make the context fit**, rather
than letting the tuner search a split. That is the opposite of the right answer for Kolibri-1, which does not fit
at all. Quality was checked at five tasks, temperature 0, both quantizations: no visible difference.

Left alone deliberately: the context is at 65536 because 131072 needs the raised `iogpu.wired_limit_mb`, which is
gone after a reboot - and with 131072 written into the config, Strata would die on the first prompt after a reboot
with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. A LaunchDaemon setting the limit at boot was offered to the
user and not built. The two contexts run at the same speed, so 65536 costs only context length.

### Qwen IQ2_XS, standalone, ctx 131072 - superseded, kept for the method note

| GPU layers | tok/s | MB per token from the SSD |
|---:|---:|---:|
| 1 | 6.5 | 5.6 |
| 24 | 9.5 | 14.0 |
| 30 | 7.3 | 49.4 |

Standalone, so high counts read too fast - the server gave 10.6 at 24 layers. Two readings were built on this table
and both were wrong: that the 28.8 GB table was the cost (it is 0.2 - 0.3 MB per token once the model is on the
GPU) and that the quantization was (worth ~10%). Neither survived being measured against Q2_0 and against a layer
count high enough to map the file. The table is above.

## Changed in the code this session

| What | Where | Why |
|---|---|---|
| Kolibri's real dimensions | `include/strata/core/layout.hpp` (`n_embd` 6144 -> 2560, `n_ff` 1536 -> 512), `include/strata/artifact/gguf_reader.hpp` (`exp_hidden`) | the native engine refused the real GGUF; see below |
| The bench tool matches the runner, and says what it cannot see | `tools/mac_split_bench.py` | it printed a start count computed differently (26 against the runner's 10, ignoring the `START_GPU_SHARE` path), left `--threads` to llama.cpp, and its docstring now carries the standalone-vs-server warning above |
| `engine/metal-tune.json`, `strata-*.json`, `.venv`, `build-m5/` | not in git (gitignored or new) | per-machine, see above |

`tools/strata_runner.py` and `tools/test_strata_runner.py` are **identical to HEAD**: a `LayerTuner` change was
written and reverted, see above.

### The native engine's Kolibri geometry (the old handover's point 5): fixed and verified

`check_architecture` refused the real file with `kolibri1.embedding_length = 2560, expected 6144`. Verified against
the GGUF itself: `embedding_length` is 2560 and `expert_feed_forward_length` is 512. The 6144 was `n_head *
head_dim` mistaken for `n_embd` - `attn_q.weight` is `[2560, 6144]`. `check_one` in `src/core/layout.cpp` already
writes its shape checks symbolically, and with 2560 / 512 **all ten of them match the real tensors exactly**
(`ffn_gate_exps` is `[2560, 512, 384]`, `attn_k` is `[2560, 512]`, ...). With the old values every one failed.

Proven by compiling `check_architecture` against the real file both ways: with 6144 it prints the refusal above,
with 2560 it accepts. `src/core/layout.cpp` compiles clean. What could **not** be tested here: the kernels
themselves - `strata_core`, which builds `layout.cpp`, is only built with CUDA or HIP, so this needs a PC with an
NVIDIA or AMD card to run end to end. Treat it as "the guard and the shapes are right, the graph is untried".

## Could these models run under Ollama instead? Measured: no, both ways

Worth knowing, because once a model fits the GPU whole this fork adds nothing to its *speed*: the layer tuner is
overridden by `STRATA_GPU_LAYERS`, the budget planning is moot, and `metal-split-mmap.patch` only applies when
weights are split with the CPU. Kolibri Q3_K_M at 51 of 51 layers is plain llama.cpp with Metal, so any
llama.cpp-based runtime should match its ~59 tok/s. Tried with Ollama 0.35.1 on this Mac:

- **Kolibri Q3_K_M**: `ollama create` *succeeds* - it copies the 35 GB, parses the GGUF and writes a manifest, so
  the import validates nothing. The first request then fails with
  `error loading model: unknown model architecture: 'kolibri1'`. Upstream llama.cpp has only an open feature
  request for the model (ggml-org/llama.cpp#29922); the architecture lives in this fork's
  `kolibri1-llama.cpp.patch` (18 mentions over 485 lines) and nowhere else.
- **Qwen Q2_0**: `ollama create` fails earlier, with
  `unsupported tensor "blk.0.ffn_down_exps.weight" size overflows` - Ollama's own GGUF parser, before any
  architecture check. The likely cause is `GGML_TYPE_Q2_0 = 42` with `GGML_TYPE_COUNT = 43`, the newest ggml type
  there is and so probably newer than Ollama's vendored llama.cpp: an unknown type cannot be sized. The split
  layout may contribute; this was **not** isolated. `qwen4exp` itself is upstream (it appears in
  `third_party/llama.cpp/src/llama-kv-cache.cpp` and in none of our patches), so for Qwen it is the quantization
  type, not the architecture, that is too new - but whether Ollama's llama.cpp knows `qwen4exp` is still untested.
  An IQ2_XS quant (ggml type 17, long-standing) would get past the parser and answer that.

Clean-up note: `ollama rm` removed the manifest but left the 35 GB blob behind, unreferenced
(`~/.ollama/models/blobs`, 0 hits when grepping `manifests/` for its digest). It was deleted by hand and
`~/.ollama` is back at its original 44 GB with the user's five own models intact.

**The unlock for Kolibri is upstreaming the patch**, which would serve Ollama, LM Studio and llama.cpp at once and
let this fork drop it.

## Open, in this order

1. **Decide what to keep on disk.** `~/Documents/Strata-data/models` now holds Kolibri Q8_0 (83.1 GB, only needed
   to quantize from), Q4_K_M (47.5 GB, the quality reference and fallback), Q3_K_M (37.5 GB, in use), Q3_K_S
   (33.9 GB, the runner-up) and Qwen Q2_0 (66.4 GB). The user was asked and had not answered when this was
   written. Q8_0 is the obvious candidate to go, unless another quantization mix is wanted - re-downloading it is
   ~20 minutes.
2. **Prefill is still unmeasured.** Every number in both bench results is decode. The user works with code, so
   pasted files make prompt processing the wait that is felt, and the runner never sets `-b` (default 2048) or
   `-ub` (512) - untuned knobs on Metal. This is the cheapest open lever: no download, no system change.
3. **Flash attention** is left at llama.cpp's `auto` and the runner never passes `-fa`. Probably already on, so
   verifying is five minutes and likely finds nothing - but it has not been checked.
4. **Speculative decoding is closed on macOS, for the record.** `llama-server` has the whole `--spec-draft-*`
   family but needs a draft model sharing the 248320-token vocabulary, and none exists.
   `~/Documents/Strata-data/mtp/mtp-q2_0.gguf` is not one: it declares `general.architecture = qwen4exp-mtp` with
   `mtp.*` tensors, a prediction head for Strata's own engine, not a standalone model. llama.cpp would need a
   patch like `kolibri1-llama.cpp.patch` for it. Upstream reports 1.6-1.8x for MTP, so it is the largest prize
   left and the most work.
5. **A runner hint when the weights land just under the working set.** The runner plans the layer count against
   `CTX_ALLOWANCE` (4 GiB), and for Qwen Q2_0 at `-c 131072` that is too little: it picks 47 layers, the weights
   fit, and the first request dies with `kIOGPUCommandBufferCallbackErrorOutOfMemory` - a failure that says
   nothing about the cause. Either plan the real KV size per architecture instead of a flat 4 GiB, or say in the
   log that the context does not fit beside the weights and name the two ways out (smaller context, or
   `iogpu.wired_limit_mb`). Offered twice now across sessions and still not built.
6. **Whether the user wants 131072 context for Qwen permanently.** It needs `iogpu.wired_limit_mb=41984`, which a
   reboot undoes, and the config would then fail on the first prompt. A LaunchDaemon that sets the limit at boot
   was offered and not built; the config is at 65536, which needs nothing and is not slower. Ask before building
   it - it is a system-level change, and the only thing it buys is context length. Kolibri Q3_K_M does **not**
   need it (verified at the stock limit).
7. **The Coder IQ1_M for code, tools and images** (58.4 GB download). The user named exactly those three as what
   they want Qwen for. It needs 29.6 GB in memory against Q2_0's 35.0, so it has more room to map the whole file.
   But the session's own numbers argue it will not be *faster*: pruning experts does not reduce the active count
   per token, and its experts are stored "like IQ3_S" (3.5 bits, codebook lookups) against Q2_0's 2 bits and
   per-bit adds - more bytes and more work per token once both are GPU-resident. Its real advantages are the
   specialized experts and that 131072 would fit without any `sysctl`. Also worth knowing first: does `--vision`
   work on this Mac at all? `engine/BUILD.json` says `"vision": "none"` and this session's setup run passed
   `--vision no`.
8. **A 3-bit step for Apple Silicon in setup.** Setup installs Kolibri Q4_K_M, the one size a 48 GB Mac cannot put
   on its GPU, and the official repo has nothing smaller. Either offer a local quantization step for Macs or at
   least point at it in the docs (done in `docs/MACOS.md`). Not changed in `setup.py`: it would mean shipping a
   quantization step, and the measurements come from one Mac. The third-party `Eliasfpv28/Kolibri-1-Q3_K_S-GGUF`
   is no longer the way in - a local Q3_K_S came out at exactly its 33.9 GB, so it can be made rather than
   trusted.
9. **MLX**: `mlx-lm` still has no `kolibri1`. PR ml-explore/mlx-lm#1945 (author velaia, "add
   support for the (German language) Kolibri 1 model by Aleph Alpha") was open and last touched 2026-10-04 when
   checked on 2026-10-05. LM Studio and Ollama's MLX backend need it merged first. `gh` is not logged in here;
   the PR was read through `https://api.github.com/repos/ml-explore/mlx-lm/pulls/1945`.

Three commits from this session are on `origin/main`: `e5338fb` (the Kolibri geometry fix), `4f33f10` (the
standalone-vs-server lesson) and `bfd340f` (Qwen at 34 tok/s). Git had no identity on this Mac; it is now set
**locally for this repo only** to `shruxx <24369531+shruxx@users.noreply.github.com>`, the GitHub account the
remote belongs to - deliberately not the user's work address, which would otherwise sit in a public history.

## Working on it

- Tests that run on a Mac: see "Tests on this Mac" above. `pytest` has to be installed into `.venv` by hand.
- C++ on a Mac: `cmake -B build-m5 -DSTRATA_BUILD_TESTS=ON` (this session's own directory; `build/` is setup's).
  `DEVELOPER_DIR` must be `/Library/Developer/CommandLineTools` and `cmake`/`ninja` come from `.venv/bin`. The
  `build/` copied from the M1 Max had `/Users/admin` paths in `CMakeCache.txt` **and** in the nested
  `_deps/*-subbuild` caches, which made every setup run fail at step 4; it was deleted and setup rebuilt it.
- After a change to `third_party/llama.cpp`: reconfigure, not only rebuild (`cmake -B build-llama -S
  third_party/llama.cpp -DGGML_METAL=ON -DBUILD_SHARED_LIBS=OFF`) - new source files are globbed at configure
  time - then copy `build-llama/bin/llama-server` to `engine/`.
- Runner settings in a model's `"env"`: `STRATA_GPU_LAYERS` (fixed count, no tuning), `STRATA_TUNE=0`,
  `STRATA_GPU_BUDGET_GB`, `LLAMA_ARG_FIT_TARGET`.
- Measuring layer counts: start `serve/server.py --engine strata --config <cfg> --port 8081` with
  `"env": {"STRATA_GPU_LAYERS": N}` per count, send five or more long replies to *varied* prompts at a real
  temperature, drop the first, and read the decode speed from the log's `eval time` lines - not the wall-clock,
  which carries the prompt too. `tools/mac_split_bench.py` and anything else driving `llama-server` directly
  reads high at high layer counts, by a factor of two on this Mac. Record the MB per token beside every speed:
  the two together say whether a count is losing to the page cache, and a speed alternating between two levels
  across replies is paging.
- `strata-*.json` and `*.log` are gitignored and per-machine. The log is appended to, so its first lines can be
  from another Mac entirely; `grep '^\[strata\]' <log> | tail` shows the current run.
- Docs style (AGENTS.md): plain words, every number with what it was measured on, no claims without a measurement.
- The user decides commits and pushes; ask before each push.
