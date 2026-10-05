# Handover: the macOS port, state of 2026-10-05 (evening)

A note for the next Claude Code session and for the user. The user writes in German: answer in German. The session
before this one ran on a MacBook Pro M1 Max, 32 GB; this one ran on the target Mac.

## The Mac this now runs on

MacBook Pro M5 Pro (Mac17,9), 48 GB, 15 CPU cores (5 "Super" = `hw.perflevel0`, 10 "Performance" =
`hw.perflevel1`), 16 GPU cores, macOS 27.0.1, Metal working set 37.44 GiB, 178 GB free on the SSD.
Working folder: `~/Documents/Projekte/StrataFormacOS`, a git clone of `shruxx/StrataForMac` (remote `upstream` =
`Niko1221/Strata`). Models in `~/Documents/Strata-data` (108 GB): Kolibri-1 Q4_K_M and Qwen3.8-Flash-Next IQ2_XS.

Two leftovers, neither in the way: `~/Documents/StrataForMac-main` is the old ZIP download (1.3 GB, no `.git`) and
holds nothing this clone needs; `~/Documents/Strata-data/models/Q2_0` is an empty directory.

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

### Qwen3.8-Flash-Next IQ2_XS (36.5 GiB in memory), standalone, ctx 131072

| GPU layers | tok/s | MB per token from the SSD |
|---:|---:|---:|
| 1 | 6.5 | 5.6 |
| 24 | 9.5 | 14.0 |
| 30 | 7.3 | 49.4 |

Not measured in the server, so read everything above 9 layers as an upper bound, as with Kolibri. No
`STRATA_GPU_LAYERS` is set for Qwen either.

**The old handover's prime suspect for Qwen is still wrong, and this conclusion survives the walk-back** because it
rests on the 1-layer row, the count where both methods agree. The 28.8 GB `per_layer_token_embd` table being
streamed per token is not what costs the speed: at 1 GPU layer Qwen reads 5.6 MB per token and writes 6.5 tok/s,
while Kolibri-1 at 1 GPU layer reads a comparable 3.2 MB per token and writes 32.0 - five times as fast, with
*more* active parameters per token (3.46B against ~2.4B). At comparable disk load the gap is still fivefold, so
what is left is the quantization: Q4_K_M scales blocks, IQ2_XS looks codebook entries up. ~10 tok/s is this file's
ceiling here, and the ~10 the user saw was already about the best of the layer counts, not a misconfiguration.

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

## Open, in this order

1. **Measure 1 against 5 GPU layers in the server for Kolibri.** The tuner put them at 32.2 and 32.4 tok/s and
   kept 1, being the fewer within 3%. Those two numbers come from one walk, two replies each; an A/B of the kind
   in the bench results (five replies per count, same prompts) would say whether 5 is really the better of the two
   and whether anything between 5 and 9 beats both. This is the remaining speed on the table for the user's main
   model, and it is small - not the 50% this session briefly thought it had found.
2. **`Q2_0` for Qwen, the measurement that settles the quantization question** (66.4 GB download, and the family's
   files already come from the repo setup knows). If a k-quant of the same model is several times faster than
   IQ2_XS on this Mac, that is the recommendation for Mac users and belongs in `docs/MACOS.md`.
3. **MTP.** The runner ignores `--mtp`; upstream reports 1.6-1.8x. llama.cpp has its own draft-model path
   (`--model-draft`), which is not the same thing as Strata's MTP but is the lever that exists here. Unmeasured.
4. **IQ2_XS near the Metal limit** (the old point 4, still unmeasured): `-ngl 49` fails although its 36.2 GiB of
   weights are under the 37.44 GiB working set, because the KV cache and compute buffers come on top.
   `sudo sysctl iogpu.wired_limit_mb=41984` (until reboot) plus `"env": {"STRATA_GPU_LAYERS": "49"}` would make it
   fit. The user was asked and chose to measure without `sudo` first, which is what the tables above are. On this
   session's evidence a win is unlikely: every count above ~9 layers loses more to the page cache it takes than it
   gains on the GPU, and 49 layers would take nearly all of it. Needs the user's password, so ask.
5. **Kolibri-1 Q3_K_S** (the old point 3): **dropped on the user's decision**, 2026-10-05. The 33.9 GB quant is a
   third-party file (`Eliasfpv28/Kolibri-1-Q3_K_S-GGUF`), not from the official `Hob-forge/Kolibri-1-GGUF` that
   setup's `kolibri` family points at, and setup builds its download URL from the family alone
   (`setup.py` ~line 4395, `fam["hf"].format(q=model) + s.name`), so it would need a per-model repo override first.
   The user did not want an unverified third-party file in the installer. Do not re-add it without asking.
6. **MLX** (the old point 6): `mlx-lm` still has no `kolibri1`. PR ml-explore/mlx-lm#1945 (author velaia, "add
   support for the (German language) Kolibri 1 model by Aleph Alpha") was open and last touched 2026-10-04 when
   checked on 2026-10-05. LM Studio and Ollama's MLX backend need it merged first. `gh` is not logged in here;
   the PR was read through `https://api.github.com/repos/ml-explore/mlx-lm/pulls/1945`.

Nothing is committed. `git status` shows the working tree with all of the above.

## Working on it

- Tests that run on a Mac: see "Tests on this Mac" above. `pytest` has to be installed into `.venv` by hand.
- C++ on a Mac: `cmake -B build-m5 -DSTRATA_BUILD_TESTS=ON` (this session's directory; the old `build/` holds the
  M1 Max's `CMakeCache.txt` with `/Users/admin` paths and should not be reused). `DEVELOPER_DIR` must be
  `/Library/Developer/CommandLineTools` and `cmake`/`ninja` come from `.venv/bin`.
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
