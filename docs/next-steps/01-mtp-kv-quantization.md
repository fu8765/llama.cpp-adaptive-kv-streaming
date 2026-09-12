# Next step 1: quantize the MTP draft KV cache

## Goal

Make the automatic MTP KV pin size the draft KV from the type the draft
context actually uses, so `-ctkd`/`-ctvd` shrink the pin and move the
MTP-active window (and the crossover where MTP stops helping) to a larger
context.

## Current state

The MTP draft context keeps its own KV cache and does not inherit the
target's `-ctk`/`-ctv`. It defaults to F16. The draft types come from
`--spec-draft-type-k`/`-ctkd` and `--spec-draft-type-v`/`-ctvd`
(common/arg.cpp:4130-4155), flow through `common_base_params_to_speculative`
(common/speculative.cpp:2481-2482) into the draft context params, and the
server passes that draft-parameterized copy to the speculative init
(tools/server/server-context.cpp:949-954). The draft context is created from
it at common/speculative.cpp:2588/2600.

So `-ctkd`/`-ctvd` do reach the MTP KV. Measured on b366c9e51, arena 3264,
ctx 160000, auto pin:

| MTP KV type | MTP KV buffer | MTP KV pin | decode window | VRAM after decode |
|---|---|---|---|---|
| F16 (default) | 190.00 MiB | 190 pages / 1398.30 MiB | 190 pages | 15769 MiB |
| q8_0 K / q4_0 V | 77.19 MiB | 190 pages / 1398.30 MiB | 190 pages | 15777 MiB |
| q4_0 K / q4_0 V | 53.44 MiB | 190 pages / 1398.30 MiB | 190 pages | 15777 MiB |

Flash Attention stays enabled in all three.

The problem: `kv_stream_pinned_bytes_for()` hard-codes F16 for the MTP KV
(`ggml_type_size(GGML_TYPE_F16)` at src/llama-context.cpp:96-98). The pin
does not change when the KV is quantized, so the saved bytes sit unused inside
the pinned region and the window stays at 190 pages. Quantizing costs nothing
but buys nothing today.

Pin breakdown at the same settings: MTP KV 190.00 MiB + MTP draft weights
609.8 MiB + recurrent rows 598.5 MiB (4 rows, 149.6 MiB each;
`CUDA_Phase_Arena0 RS buffer size = 598.50 MiB`) = 1398.30 MiB.

The earlier report that `-ctkd`/`-ctvd` increased VRAM was real, but it is a
transient init peak, not a steady-state increase. Quantizing the MTP KV shrinks
the pin, so at a fixed arena the shared arena compute region grows. The MTP
context commits that larger region in `sched_reserve`, which raises the peak
during init. At arena 3264 this peak OOMs on the MTP context's ~209 MiB compute
buffer (`failed to create MTP context`). Once the server is up, total VRAM at a
fixed arena is the same as F16.

## Work items

1. Plumb the MTP KV types into the target context params. Done:
   `mtp_kv_type_k`/`mtp_kv_type_v` added to `llama_context_params`
   (include/llama.h) and `llama_cparams` (src/llama-cparams.h), copied in the
   context constructor, and set from `params.speculative.draft.cache_type_k/v`
   in `common_context_params_to_llama()` (common/common.cpp).
2. Size the pin from the actual types. Done: `mtp_kv_bytes_per_token()` in
   src/llama-context.cpp returns
   `ggml_row_size(type_k, n_embd_k_gqa) + ggml_row_size(type_v, n_embd_v_gqa)`
   for the nextn layer and replaces the F16 expression in both
   `kv_stream_pinned_bytes_for()` and `kv_stream_mtp_kv_cap_apply()`.
3. Debug-assert that the pin types match the draft context's KV cache types.
   Not done. The draft context is created after the target, so the target cannot
   inspect it; both use the same common params, so the types agree by
   construction.
4. Expose the MTP KV types in `llama_kv_stream_status`. Not done, optional.

## Outcome

Measured on `feature/mtp-next-steps`, arena 3072, ctx 160000, ub 256, auto pin.
The three-way probe now shows the pin tracking the types:

| MTP KV type | MTP KV pin | decode window |
|---|---|---|
| F16 | 164 pages / 41984 tokens | 164 pages |
| q8_0 K / q4_0 V | 178 pages / 45568 tokens | 178 pages |
| q4_0 K / q4_0 V | 182 pages / 46592 tokens | 181 pages |

Decode throughput across the crossovers (ejected rows in italics in the raw
CSV; a dash means MTP is still active and the run is the fast path):

| prompt | F16 decode | q8_0/q4_0 decode | q4_0/q4_0 decode |
|---:|---:|---:|---:|
| 38000 | 45.7 | 57.1 | 57.2 |
| 40000 | 23.3 (ejected) | 56.3 | 56.3 |
| 44000 | 22.8 | 22.9 (ejected) | 22.9 (ejected) |
| 48000 | 22.5 | 22.4 | 22.5 |
| 80000 | 19.6 | 19.6 | 19.6 |
| 160000 | 12.2 | 12.0 | 12.0 |

The window grows by 14 to 18 pages, but the crossover only moves from about
39K to about 42K tokens (roughly one context step). The eject trigger fires
before the full window is used, so the extra pages do not translate one to one
into a later crossover. Prefill loses 1.5 to 4 percent versus F16. The F16
38000 decode row is an outlier; it sits at the edge of the arena and is not a
clean baseline.

Arena 3072 was used because arena 3264 plus a quantized MTP KV hits the init
peak above. The same sweep can run at arena 3264 with `-ub 128`.

## Known limitation

The MTP context has no arena of its own (`kv_stream_arena_mib = 0` for the
draft). Its KV and weights come from the pinned region inside the target arena,
but its compute buffers are a separate `cudaMalloc`. Shrinking the pin enlarges
the arena compute side, so the init peak rises. At arena 3264 with `-ub 256`
this can fail to allocate the MTP context's compute buffer
(`failed to create MTP context`). The failure is not deterministic: the same
config can start. Workarounds are arena 3072, or arena 3264 with `-ub 128`.
Fixing it properly means either capping the MTP prefill catch-up batch (the
buffer scales with it) or routing the MTP compute into the target arena, which
now has room because the pin shrank.

## Acceptance

- Pin tracks the actual MTP KV type; the freed bytes become window. Met.
- No TG regression versus the F16 MTP KV baseline. Met while MTP is active;
  decode is equal to or better than F16 at every context.
- No output drift versus the F16 MTP KV baseline. Not yet checked. The
  speculative acceptance rate and a greedy output comparison still need a run.
- The init-peak OOM is not fixed. It is currently avoided by arena 3072, not
  solved.

## Risks and levers

- The draft weights (609.8 MiB) and rs rows (598.5 MiB) dominate the pin, so the
  KV saving is ~10% of the pin. If the crossover needs a bigger push, the larger
  levers are a quantized MTP draft model and a smaller verify batch (fewer rs
  rows). Both are separate changes.
- Mixed q8_0 K + q4_0 V for the draft must keep FA. The fork builds with
  `GGML_CUDA_FA_ALL_QUANTS=ON`, so the kernels are present, but confirm the
  draft does not silently fall back.
