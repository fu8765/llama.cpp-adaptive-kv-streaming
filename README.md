# llama.cpp adaptive KV streaming - fork notes

This repository is a fork of
[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming).

Upstream adds an experimental, block-granular KV cache streaming path to the CUDA
`llama-server`. With `--kv-stream-arena-mib N` the authoritative KV tensors live
in pinned host memory while one bounded CUDA arena is shared between
phase-specific compute buffers, resident KV pages, and the transfer ring. The
upstream README (linked below) covers that design, its build, and its benchmarks.

This fork keeps that implementation and adds speculative-decoding support and
memory management for the MTP draft context. Everything below is experimental.

## TLDR

If you have a 16GB CUDA GPU - just do the following:

* Clone this repo, build with these parameters

```sh
cmake -B build -DGGML_NATIVE=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA=ON
cmake --build build --config Release -j
```

* Download this version of Qwen 3.8 27B

[bsaleh03's ASCII condensed UD-IQ4_XS](https://huggingface.co/bsaleh03/Qwen3.8-27B-ASCII-Condensed)

* Run this script to extract MTP to a separate file

```sh
python3 gguf-py/gguf/scripts/gguf_extract_mtp.py \
    Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf \
    Qwen3.8-27B-ASCII-Condensed-MTP.gguf
```

* Use the following parameters (models-preset.ini format) when launching llama-server

```
m = Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf
md = Qwen3.8-27B-ASCII-Condensed-MTP.gguf
device-draft = CUDA0
n-gpu-layers-draft = all
ctx-size = 160000
n-gpu-layers = 99
batch-size = 256
ubatch-size = 256
# lower this value if you don't have the full 16GB available for the model
kv-stream-arena-mib = 3264
cache-type-k = q8_0
cache-type-v = q4_0
spec-type = draft-mtp
spec-draft-n-max = 3
kv-stream-mtp-dynamic = on
kv-stream-mtp-keep-pages = 380
kv-stream-mtp-reenable-pages = 8
kv-stream-mtp-stable-decodes = 4
fit = off
parallel = 1
temp = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
repeat-penalty = 1.0
reasoning = on
reasoning-preserve = on
# (slow) CPU only multimodal is better than none
no-mmproj-offload = on
mmproj = Qwen3.8-mmproj-BF16.gguf
load-mode = none
flash-attn = on
```

## Results

Measured on an RTX 5060 Ti 16 GB with Qwen3.8-27B (Q8_0 K cache, Q4_0 V
cache), `-ngl 99`, `--flash-attn on`, and `--parallel 1`. The model used is
[bsaleh03's ASCII condensed version of Unsloth UD-IQ4_XS](https://huggingface.co/bsaleh03/Qwen3.8-27B-ASCII-Condensed).

### Upstream vs MTP (head to head)

Raymond's phase-arena branch (no speculative decoding) against this fork with
MTP, each at the largest `--kv-stream-arena-mib` that loads on the 16 GB card
(3072 MiB upstream, 3264 MiB fork[^arena-pin]). Both run `--ctx-size 160000`; the prompt is a
source tree followed by a review instruction, with 256 tokens generated at
temperature 0.

| prompt tokens | upstream TG t/s | fork TG t/s | TG gain | upstream PP t/s | fork PP t/s | PP loss | MTP     |
|--------------:|----------------:|------------:|--------:|----------------:|------------:|--------:|:-------:|
|        20,000 |            25.8 |        54.8 |  2.12x  |             947 |         807 |  14.8%  | active  |
|        40,000 |            23.3 |        57.7 |  2.47x  |             862 |         738 |  14.5%  | active  |
|        80,000 |            19.6 |        19.6 |  1.00x  |             730 |         675 |   7.5%  | ejected |
|       120,000 |            16.9 |        16.8 |  1.00x  |             633 |         603 |   4.6%  | ejected |
|       160,000 |            12.9 |        13.1 |  1.01x  |             544 |         527 |   3.2%  | ejected |

MTP roughly doubles generation while its working set fits, then the dynamic
controller ejects it and generation tracks upstream exactly. Prompt processing
is slower with MTP because the draft model adds work to each batch; the loss
shrinks from 14.8% to 3.2% as the context grows and the decode-side share of
the total drops. Reproduce with `benchmarks/benchmark_upstream_vs_mtp.py`; the
sweep writes `results.jsonl`/`results.csv` plus a four-panel PNG/SVG under
`benchmarks/results/`.

[^arena-pin]: The fork's MTP work adds a pinned region inside the phase arena:
on this hybrid SSM model the recurrent-state cache (~150 MiB), the MTP draft
weights, and the MTP KV cache all come from it, while upstream allocates the
recurrent-state cache as a separate `cudaMalloc` on top of the arena. At a given
arena size the fork's total VRAM is therefore lower, which lets it run a larger
arena on the same 16 GB card. The cost is internal: the pin takes KV-window
budget, which the dynamic eject controller manages.

### MTP KV quantization

The MTP draft KV defaults to F16 and does not inherit the target `-ctk`/`-ctv`.
Pass `-ctkd`/`-ctvd` to quantize it. The automatic pin now sizes the pinned MTP
KV from the draft types, so the saved bytes become decode window:

| MTP KV type | MTP KV pin | decode window |
|---|---|---|
| F16 | 164 pages / 41984 tokens | 164 pages |
| q8_0 K / q4_0 V | 178 pages / 45568 tokens | 178 pages |
| q4_0 K / q4_0 V | 182 pages / 46592 tokens | 181 pages |

Measured at arena 3072, ctx 160000, ub 256, auto pin. The window grows by 14 to
18 pages, which moves the MTP crossover from about 39K to about 42K tokens.
Prefill loses 1.5 to 4 percent versus F16. Quantizing shrinks the pin, so the
shared arena compute region grows and the init peak rises: at arena 3264 the MTP
context can fail to allocate its compute buffer. Details and the throughput
table are in
[docs/next-steps/01-mtp-kv-quantization.md](docs/next-steps/01-mtp-kv-quantization.md).

### MTP during streaming

By default the dynamic controller ejects MTP at streaming onset. Keeping it
active lets it speculate on the recent window while the target streams, and the
draft KV now slides (evicting the oldest cells) so it can follow the target past
the pinned window. `--kv-stream-mtp-keep-pages N` sets how long to keep it: MTP
stays active until the target's decode working set exceeds `N` 256-token pages,
then ejects as usual. `0` (default) ejects at streaming onset; a very large `N`
keeps it active throughout. The measurements below are the keep-throughout arm;
at arena 3072 the crossover is about 380 pages (about 97K tokens), so
`--kv-stream-mtp-keep-pages 380` reproduces it. Requires
`--kv-stream-mtp-dynamic`.

| prompt | keep tg t/s | default tg t/s | keep/default | keep pp t/s | default pp t/s |
|---:|---:|---:|---:|---:|---:|
| 50,000 | 30.40 | 22.32 | 1.36x | 686.8 | 711.4 |
| 80,000 | 23.25 | 19.61 | 1.19x | 598.6 | 670.2 |
| 90,000 | 21.47 | 18.81 | 1.14x | 571.7 | 653.4 |
| 100,000 | 17.21 | 18.13 | 0.95x | 548.3 | 636.9 |
| 120,000 | 12.68 | 15.80 | 0.80x | 505.2 | 599.5 |
| 160,000 | 7.15 | 12.15 | 0.59x | 435.1 | 522.7 |

Measured at arena 3072, ctx 160000, q8_0 K / q4_0 V, 256 decode tokens. Keeping
MTP is faster up to about 97,000 tokens and slower beyond; prefill pays 3.5 to 17
percent because the draft runs during prefill too. Reproduce with
`benchmarks/benchmark_mtp_streaming.py`; details in
[docs/next-steps/02-mtp-during-streaming.md](docs/next-steps/02-mtp-during-streaming.md).

## Differences from upstream

### Phase arena: speculative verify batches
- Upstream rejected generation batches whose token count was not exactly
  `n_seq_max` ("phase arena currently supports TG1 without speculative batches").
  Speculative decoding verifies `1 + n_draft` tokens on a single sequence.
- New context parameter `llama_context_params.n_max_spec_draft` ("max speculative
  draft tokens, 0 = none"). `common_context_params_to_llama()` sets it from
  `common_speculative_n_max(&params.speculative)`, so it follows the configured
  spec type (ngram, MTP, DFlash) instead of a hardcoded value.
- `llama_context::sched_reserve()` measures the token-generation graph at
  `max(n_seq_max, 1 + n_max_spec_draft)` tokens instead of `n_seq_max`.
- `llama_context::kv_stream_switch_phase()` re-reserves the decode layout at the
  same width, so the arena compute slab fits a verify batch.
- `llama_context::process_ubatch()` admits generation batches up to
  `max(n_seq_max, 1 + n_max_spec_draft)` and otherwise fails with
  "phase arena decode batch too wide".
- `n_max_spec_draft = 0` reproduces the original behaviour.

### MTP draft context
- The MTP draft context's `n_ubatch` is capped to
  `max(8, draft.n_max + 2) * n_seq_max` so its compute graph stays small.
  `n_batch` is left unchanged, because the Qwen3.5 MTP path runs a prefill
  catch-up decode into its own KV.
- The MTP context's KV cache can be allocated from a pinned region inside the
  target's phase arena instead of a separate full-length F16 `cudaMalloc`.
- The MTP block weights can be allocated from the same pinned region, so an
  evicted MTP context returns both its KV and its weights to the arena pool.
- A separate MTP-only GGUF can be supplied with `-md` while using
  `--spec-type draft-mtp`. The target then skips its embedded MTP tensors
  (`load_mtp = false`), and the draft borrows the target's LM head.

### New arena and context API
- ggml-cuda: `ggml_backend_cuda_phase_arena_set_pinned()`,
  `_reset_pinned()`, and `_pinned_buffer_type()`. The pinned region is
  bump-allocated from the top of the arena; the compute region must stay below
  it. The pinned buffer type shares the arena name so it is recognised as a CUDA
  buffer.
- `llama_kv_stream_pinned_buft()` returns the target context's pinned buffer
  type.
- New experimental context parameters: `spec_mtp`, `mtp_weights_bytes`,
  `n_max_spec_draft`, `kv_stream_mtp_kv_pages`, `kv_stream_mtp_dynamic`. The
  status struct gains `mtp_kv_pages`, the current pinned MTP KV size in pages.
- `llama_model_borrow_output()` lets a draft model share the target's LM head
  (`output` / `output_s`) instead of carrying a duplicate copy.

## Configuration

### Phase arena (upstream)
- `--kv-stream-arena-mib N` (alias `--kv-stream-stage-mib`): size of the shared CUDA arena in MiB; `0` disables it. The phase arena requires `--parallel 1`, `--flash-attn on`, KV offload, and a Qwen3.5-family target.

### Speculative decoding
- `--spec-type draft-mtp`: enable MTP speculative decoding.
- `-md <file>`: optional separate MTP-only GGUF; the target then skips its embedded MTP tensors (`load_mtp = false`) and the draft borrows the target LM head.
- `--spec-draft-n-max N`: number of draft tokens. It also widens the target recurrent-state cache and the decode compute slab.
- `--spec-draft-type-k T` / `--spec-draft-type-v T`: draft KV cache types (default F16). The main `--cache-type-k`/`--cache-type-v` do not affect the draft.

### Creating a separate MTP model

The MTP block can be split out of a merged GGUF with
`gguf-py/gguf/scripts/gguf_extract_mtp.py`:

```sh
python3 gguf-py/gguf/scripts/gguf_extract_mtp.py \
    Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf \
    Qwen3.8-27B-ASCII-Condensed-MTP.gguf
```

The output keeps the target vocab metadata (`token_embd`, `output_norm`) and the
`blk.<mtp>.` block. It deliberately drops `output.weight` so the draft borrows
the target LM head; pass `--with-lm-head` to keep it. Use the result with
`-md <file>`; the target then skips its embedded MTP tensors, saving their VRAM.

### Dynamic MTP eject (this fork, opt-in)
- `--kv-stream-mtp-dynamic`: eject MTP when the decode working set exceeds the MTP-active decode capacity, and re-enable it when it fits again (default: disabled).
- `--kv-stream-mtp-eject-pages N`: eject once the active pages exceed the MTP-active decode capacity by `N` 256-token KV pages (default: 0, i.e. at streaming onset).
- `--kv-stream-mtp-reenable-pages N`: re-enable once the active pages fit at least `N` pages below that capacity. Must be greater than `--kv-stream-mtp-eject-pages` (default: 8).
- `--kv-stream-mtp-stable-decodes N`: consecutive decode batches required before a transition (default: 4).
- `--kv-stream-mtp-kv-pages N`: size of the pinned MTP KV reservation, in 256-token pages. `0` (default) sizes the pin to the MTP-active decode window automatically; a positive `N` pins exactly `N` pages and caps the window there. Requires `--kv-stream-mtp-dynamic`.
- `--kv-stream-mtp-keep-pages N`: keep MTP active until the target's decode working set exceeds `N` 256-token pages, then eject like the default. `0` (default) ejects at streaming onset; use a large value to keep MTP active throughout. The draft KV slides to follow the target. Requires `--kv-stream-mtp-dynamic`.

The `LLAMA_ARG_KV_STREAM_MTP_*` environment variables mirror these options. Ejecting returns the MTP weights, the MTP KV cache, and the widened recurrent-state cache to the arena pool; re-enabling restores them. The default configuration ejects at streaming onset and re-enables with an 8-page hysteresis band.

**MTP must be the only spec type.** Dynamic eject changes only the MTP context.
If `--spec-type` mixes `draft-mtp` with another speculator (for example
`--spec-type draft-mtp,ngram-mod`), the server disables dynamic eject with a
warning and keeps MTP pinned for the whole run.

### Tuning the dynamic MTP window

MTP is kept while the decode working set fits the MTP-active decode pool, which
is what remains of the arena after the pinned reservation (MTP weights,
recurrent-state cache, MTP KV) and the phase compute slab. A larger
`--ctx-size` reserves more and shrinks the window.

The pinned MTP KV is the largest term, and a full-context pin reserves about
4 MiB per 1000 context tokens. MTP only runs while the working set fits the
decode pool, so the MTP KV never needs the full context. The default sizes the
pin to that decode window instead. The two are coupled: a smaller pin leaves
more arena for KV and grows the window, which in turn needs a larger pin. The
default solves that fixed point directly, so the pin matches the decode
capacity with no wasted reservation. Measured at `--kv-stream-arena-mib 3264`
with this model and draft:

| `--ctx-size` | prefill resident pages/layer | decode resident pages/layer | MTP-active window   |
|-------------:|-----------------------------:|----------------------------:|---------------------|
|        32768 |                          267 |                         276 | full context (~32k) |
|        65536 |                          228 |                         239 | full context (~61k) |
|       160000 |                          172 |                         190 | ~49k tokens         |

A page is 256 tokens, so the window is `decode resident pages/layer * 256`
tokens, bounded by `--ctx-size`. When the whole context fits the decode pool,
as at `--ctx-size 32768`, no cap is applied and MTP stays active throughout.

- `--kv-stream-mtp-kv-pages N` overrides the automatic pin and reserves exactly
  `N` pages of MTP KV. The window is then capped at `N` pages minus a small
  catch-up margin (the MTP context decodes every target batch, so it must absorb
  a few batches past the nominal window before the eject lands). Requires
  `--kv-stream-mtp-dynamic`. Use it to trade MTP reach against pool size, or to
  bound a known working set. `--kv-stream-mtp-kv-pages 0` restores the
  automatic sizing.
- `--kv-stream-mtp-eject-pages N` ejects only EARLIER: it fires when the active
  pages come within `N` pages of the capacity. `N = 0` keeps MTP as long as
  possible (eject at streaming onset). It cannot extend the window past the pool
  capacity.
- `--kv-stream-mtp-reenable-pages N` is the hysteresis: MTP is re-enabled once
  the active pages fall `N` pages below the capacity. It must exceed
  `--kv-stream-mtp-eject-pages`; a larger value makes re-enable later and less
  prone to flapping. Default 8.
- `--kv-stream-mtp-stable-decodes N` debounces a transition until `N` consecutive
  decode batches agree. Default 4. Raise it if a mixed workload flaps.

Set `--ctx-size` to the longest prompt you need; beyond that MTP ejects and
generation returns to the baseline rate. Keep `--spec-draft-n-max` small (the
recurrent cache is `149.6 MiB * (1 + n_max)`) and give the arena as much room as
the model leaves.

## Next steps

- [Enable DFlash2](docs/next-steps/03-dflash2-draft.md): build a vocabulary-matching draft for the condensed target and generalize the pin and eject path.

## Scope and status

- Validated on an RTX 5060 Ti 16 GB with Qwen3.8-27B, a Q8_0 K cache, a Q4_0 V
  cache, one server slot (`-np 1`), and Flash Attention enabled.
- Single GPU and `llama-server` only. The phase arena requires
  `n_seq_max == 1`, Flash Attention, KV offload, and a Qwen3.5-family target.
- DFlash2 requires a draft whose vocabulary matches the target. A
  condensed-vocabulary target (129006 tokens) is incompatible with the
  full-vocabulary DFlash2 draft (248320 tokens): the draft has no token
  embedding and embeds through the target's `token_embd`.
- ngram-map and ngram-simple currently produce zero drafts in this configuration.
- Quantizing the MTP draft KV (`-ctkd`/`-ctvd`) shrinks the pin, which grows the
  arena compute side and raises the init peak. At arena 3264 with `-ub 256` this
  can fail to allocate the MTP context's compute buffer. Use arena 3072 or
  `-ub 128` at 3264 for now.
- Research code, no upstream guarantees.

## Upstream

[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)
