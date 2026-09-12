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

The earlier report that `-ctkd`/`-ctvd` increased VRAM did not reproduce. The
after-decode figures above are within noise. The "increase" was most likely a
measurement taken at a different point: the arena allocates on first decode,
so a reading taken when `/health` first returns can show roughly 3 GiB less
than the settled figure.

## Work items

1. Plumb the MTP KV types into the target context params.
   `common_context_params_to_llama` already sets `cparams.spec_mtp`
   (common/common.cpp:1764); add `mtp_kv_type_k`/`mtp_kv_type_v` from
   `params.speculative.draft.cache_type_k/v` and carry them through
   `llama_context_params` (include/llama.h) and `llama_cparams`
   (src/llama-cparams.h).
2. In `kv_stream_pinned_bytes_for()` (src/llama-context.cpp:85-120) compute the
   MTP KV term from the actual types, K and V separately, with per-row block
   rounding (`ggml_blck_size`). Prefer the KV cache's own size accounting if one
   is reachable instead of duplicating the layout math.
3. Assert in debug that the types used for the pin match the draft context's KV
   cache types.
4. Optionally expose the MTP KV types in `llama_kv_stream_status` for the
   server log and the README.

## Test plan

- Re-run the three-way probe above. Expect `pinned` to drop by ~113 MiB
  (q8_0/q4_0) or ~137 MiB (q4_0/q4_0) and the auto cap to pick a larger decode
  window. At ~6.5 MiB per target decode page that is roughly +17 to +21 pages.
- Measure TG and PP at 20/40/80/120/160K for MTP KV F16 vs q8_0/q4_0 vs
  q4_0/q4_0 and record the new crossover.
- Correctness: greedy output on a fixed prompt should match the F16 MTP KV run;
  watch the speculative acceptance rate for a regression.

## Acceptance

- Pin tracks the actual MTP KV type; the freed bytes become window.
- No TG regression and no output drift versus the F16 MTP KV baseline.

## Risks and levers

- The draft weights (609.8 MiB) and rs rows (598.5 MiB) dominate the pin, so the
  KV saving is ~10% of the pin. If the crossover needs a bigger push, the larger
  levers are a quantized MTP draft model and a smaller verify batch (fewer rs
  rows). Both are separate changes.
- Mixed q8_0 K + q4_0 V for the draft must keep FA. The fork builds with
  `GGML_CUDA_FA_ALL_QUANTS=ON`, so the kernels are present, but confirm the
  draft does not silently fall back.
