# Next step 2: keep MTP active while streaming

## Goal

Stop ejecting MTP the moment KV streaming starts. Keep speculating while the
MTP gain still beats the extra streaming cost, and eject only when it does not.

## Current state

`update_mtp_dynamic()` (tools/server/server-context.cpp:2853-2928) ejects MTP
after `stable_decodes` consecutive checks when either

- `st.streaming` is true, or
- `active_pages + eject_pages > mtp_capacity_pages`.

`st.streaming` is literally `active_pages > resident_pages`
(src/llama-kv-cache.cpp:1469), so it becomes true exactly when the working set
outgrows the MTP-active pool. Re-enable requires `!st.streaming` and
`active_pages + reenable_pages <= mtp_capacity_pages`
(tools/server/server-context.cpp:2909-2911).

Keeping MTP active keeps the pin (MTP KV window + draft weights + rs rows) and
holds the resident pool at the smaller MTP-active capacity, so more KV streams.
The question is whether the roughly 2.5x TG from MTP outweighs that, and where
the crossover sits.

## Hypothesis

Just past the MTP-active capacity, ejecting frees little pool and MTP still
wins. The crossover is somewhere between the MTP-active capacity and the full
context, and it depends on how much the streaming actually costs.

## Work items

Phase 1, cheap and decisive:

1. Add a policy knob that drops the `st.streaming` term from the eject
   condition while keeping the capacity term and the re-enable side. Candidates:
   `--kv-stream-mtp-eject-policy {capacity,pressure,streaming}` or a simple
   `--kv-stream-mtp-keep-streaming`.
2. Re-run 80/120/160K with and without the trigger and compare TG, PP,
   acceptance, and streaming volume.

Phase 2, find the crossover automatically:

3. Expose the streaming runtime's copy-pressure signals through
   `llama_kv_stream_status`. `deadline_miss_ratio` and `copy_busy_ratio` are
   already computed per ubatch (src/llama-kv-cache.cpp:1441-1456) but are not
   reported.
4. Eject on measured copy pressure instead of streaming onset. The controller
   then adapts to the machine instead of a fixed context threshold.

## Notes

- No hard blocker was found. Verify batches are admitted at
  `max(n_seq_max, 1 + n_max_spec_draft)` (src/llama-context.cpp:2174-2185) and
  the concentrated decode layout handles query batches up to 32 tokens
  (src/llama-kv-cache.cpp:1427, 1479-1481), so a 4-token MTP verify is treated
  as a normal decode under streaming.
- The MTP draft window is capped to the pin window
  (common/speculative.cpp:2541-2544). At 160K the draft sees about 49K of
  160K, which may reduce acceptance.
- Staying active also avoids the toggle cost: `synchronize()` + CUDA graph
  reset + scheduler reset + rs rebuild (src/llama-context.cpp:1507-1516).
- Task 1 helps here: a quantized MTP KV makes the same pin hold a larger MTP
  window.

## Acceptance

- A measured crossover and a policy that keeps MTP while it is faster and
  ejects it when streaming dominates.
