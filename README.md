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
  `n_max_spec_draft`.
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

The `LLAMA_ARG_KV_STREAM_MTP_*` environment variables mirror these options. Ejecting returns the MTP weights, the MTP KV cache, and the widened recurrent-state cache to the arena pool; re-enabling restores them. The default configuration ejects at streaming onset and re-enables with an 8-page hysteresis band.

**MTP must be the only spec type.** Dynamic eject changes only the MTP context.
If `--spec-type` mixes `draft-mtp` with another speculator (for example
`--spec-type draft-mtp,ngram-mod`), the server disables dynamic eject with a
warning and keeps MTP pinned for the whole run.

### Tuning the dynamic MTP window

MTP is kept while the decode working set fits the MTP-active decode pool, which
is what remains of the arena after the pinned reservation (MTP weights,
recurrent-state cache, full-context MTP KV) and the phase compute slab. A larger
`--ctx-size` reserves more and shrinks the window. Measured at
`--kv-stream-arena-mib 3072` with this model and draft:

| `--ctx-size` | prefill resident pages/layer | decode resident pages/layer | MTP-active window       |
|-------------:|-----------------------------:|----------------------------:|-------------------------|
|        32768 |                          229 |                         246 | full context (~63k cap) |
|        65536 |                          185 |                         207 | ~53k tokens             |
|       160000 |                           57 |                          93 | ~24k tokens             |

A page is 256 tokens, so the window is `decode resident pages/layer * 256`
tokens (the row with the lowest capacity binds).

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

To keep MTP active to a given context length, use the smallest `--ctx-size` that
covers your workload (each 1000 tokens of ctx reserves about 4 MiB of MTP KV),
keep `--spec-draft-n-max` small (the recurrent cache is
`149.6 MiB * (1 + n_max)`), and give the arena as much room as the model leaves.
At 3072 MiB on a 16 GiB card with a 13.5 GiB model the MTP-active capacity tops
out near 63k tokens at `--ctx-size 32768` (MTP then stays active for the whole
32k context), so an 80k-token context cannot keep MTP active at this arena size;
the three flags tune where the transition happens, they cannot raise that
ceiling.

## MTP generation speed

The tables below compare token generation (TG) throughput with MTP against the
same build with `--spec-type none` (baseline). Measured on an RTX 5060 Ti 16 GB
with Qwen3.8-27B (Q8_0 K cache, Q4_0 V cache), `--kv-stream-arena-mib 2304`,
`--ctx-size 180000`, `-ngl 99`, `--flash-attn on`, `--parallel 1`, and 96
generated tokens at temperature 0. `synthetic` is a repeated sentence; `document`
is a real source tree/README followed by an instruction.

| prompt tokens | prompt   | baseline TG t/s | MTP TG t/s | gain  |
|--------------:|----------|----------------:|-----------:|------:|
|         4,000 | synthetic |            27.9 |       74.7 | 2.67x |
|         8,000 | synthetic |            27.2 |       73.1 | 2.69x |
|        12,000 | synthetic |            26.6 |       67.4 | 2.53x |
|        16,000 | synthetic |            26.1 |       60.1 | 2.30x |
|        20,000 | synthetic |            25.4 |       52.4 | 2.06x |
|        35,429 | document  |            23.5 |       27.1 | 1.15x |
|        54,892 | document  |            21.4 |       16.5 | 0.77x |
|        73,426 | document  |            19.7 |       11.7 | 0.59x |

MTP also costs prompt-processing throughput: about 830 vs 990 t/s at low
context, and 620 vs 760 t/s at the largest tested context.

With `--kv-stream-mtp-dynamic`, MTP is kept while it helps and ejected once the
working set outgrows the MTP-active capacity. After the eject, generation returns
to the baseline rate exactly:

| prompt tokens | prompt   | baseline TG t/s | static MTP TG t/s | dynamic TG t/s |
|--------------:|----------|----------------:|------------------:|---------------:|
|         8,000 | synthetic |            27.2 |              73.1 |           74.6 |
|        35,429 | document  |            23.5 |              27.1 |           23.5 |
|        73,426 | document  |            19.7 |              11.7 |           19.7 |

The crossover depends on how predictable the text is: roughly 44k prompt tokens
for real documents and around 80k for highly repetitive text. Dynamic eject is
most useful above that point, where it keeps the large-context pool without
paying the MTP generation penalty.

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
- Research code, no upstream guarantees.

## Upstream

[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)
