# Next step 3: enable DFlash2 with the ASCII-condensed target

## Goal

Run DFlash2 speculative decoding on the condensed-vocabulary target, using the
same phase arena, pinned draft weights, and dynamic eject that MTP uses.

The arena pin/eject generalization is planned in [docs/superpowers/plans/2026-09-13-dflash-arena-generalization.md](../superpowers/plans/2026-09-13-dflash-arena-generalization.md).

## Context

DFlash and DFlash2 live in src/models/dflash.cpp. DFlash2 adds conv/selector
tensors (`dflash_selector_rank`, `dflash_selector_top_k`,
`dflash_conv_kernel_size`, `dflash_conv_group_size`,
src/llama-hparams.h:244-248; src/models/dflash.cpp:130-139). The spec types are
`draft-dflash` and `draft-dspark` (DSpark = DFlash + semi-autoregressive Markov
head), docs/speculative.md:55-83. The target-arch check is QWEN35-only, and the
condensed target is QWEN35.

## Prerequisite: a vocabulary-matching DFlash2 draft

The public DFlash2 draft has a 248320-token vocabulary. The condensed target
has 129006. The draft has no token embedding and embeds through the target's
`token_embd`, so the mismatch is in the draft's vocabulary-dependent tensors
and/or the d2t mapping (src/models/dflash.cpp:107).

The condensed target itself shows the fix is tractable: it was built from a
stock GGUF by row-gathering `token_embd` and `output` to the retained ASCII
rows and rewriting the tokenizer, with no retraining or requantize and no other
weight change. See the model card at
https://huggingface.co/bsaleh03/Qwen3.8-27B-ASCII-Condensed. Extend the existing
MTP-only GGUF extraction script to DFlash2, apply the same row-gather to the
draft's vocabulary tensors, and rewrite the tokenizer with the target's
mapping. Verify survivors are bit-exact and the d2t path is consistent.

## Code work to generalize the MTP pin machinery

1. `kv_stream_pinned_bytes_for()` (src/llama-context.cpp:85-120) assumes a
   single nextn layer and F16. Generalize to the draft context's KV geometry
   (n_layer layers) and its type.
2. `kv_secondary_buft` is wired only for `ctx_type == LLAMA_CONTEXT_TYPE_MTP`
   (src/llama-context.cpp:633-636), and the draft-weight buft override is gated
   on `spec_mtp` (common/speculative.cpp:2566-2573). Add a draft-dflash path for
   both.
3. The draft context window cap and `n_ubatch` cap are MTP-gated
   (common/speculative.cpp:2541-2552). DFlash needs the window cap too.
4. The `kv_stream_mtp_set` toggle is gated on `cparams.spec_mtp`
   (src/llama-context.cpp:1449; flag set common/common.cpp:1764). The rs rebuild
   inside it is already generic in `n_rs`; the MTP-specific part is
   `llama_set_embeddings_nextn` in the server eject/enable
   (tools/server/server-context.cpp:2940, 2957). DFlash uses fused target
   features instead, so re-enable must know what draft state to preserve.
5. The server controller (`spec_mtp_enabled_dynamic`, `mtp_eject()`,
   `mtp_enable()`) and the "MTP must be the only spec type" validation
   (tools/server/server-context.cpp:1104-1112, common/arg.cpp:891) need
   generalizing to the active draft type.

Already generic: the verify width `1 + n_max_spec_draft`; the 32-token decode
layout; the rs row count via `need_n_rs_seq()`, which already returns
`draft.n_max` for MTP/EAGLE3/DFLASH/DSPARK (common/common.h:400-406).

## Sizing

Recurrent rows cost 149.6 MiB x (1 + n_max) (see the README Scope and status
section). DFlash2 blocks can reach ~15, so `n_max` 15 would pin ~2.4 GB of rs
alone, which does not fit a 3264 MiB arena on top of draft weights and KV. Keep
`n_max` small (~3-5) or use a larger GPU. The DFlash draft also runs a full
multi-layer block-diffusion pass per step, heavier than MTP's single nextn
layer, so expect a smaller TG gain on a 16 GB card.

## Test plan

- Build the vocab-matching draft; confirm it loads and that the target-arch
  check passes.
- Verify the draft KV and weights are backed by the pinned region and that
  eject returns them to the pool.
- Measure TG and PP at 20/40/80/120/160K against MTP on the same target and find
  the crossover.

## Acceptance

- `--spec-type draft-dflash` runs on the condensed target with the pinned arena
  and dynamic eject, with measured TG/PP versus MTP.
