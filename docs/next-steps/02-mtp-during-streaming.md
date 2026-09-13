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

## Finding: the draft window was a hard wall

Keeping MTP active past the pin window (the keep-throughout arm of
`benchmarks/benchmark_mtp_streaming.py`, or `--kv-stream-spec-keep-pages` set
above the window) fails without a sliding draft KV.
`benchmarks/benchmark_mtp_streaming.py` (arena 3072, q8_0 K / q4_0 V) ejects
MTP in default mode at 50000 and above. Keeping it active fails at context
50000: the MTP draft context has `n_ctx` capped to the 45568-token pin window
(common/speculative.cpp:2541-2544) and fills its KV during prefill, so the
catch-up decode cannot find a slot:

    W decode: failed to find a memory slot for batch of size 256
    E spec process: llama_decode(ctx_dft) head=0 failed rc=1 (pos=45568)

The failure is slot exhaustion, not a position bound. Nothing evicts old draft
cells, so the draft cannot follow the target past its window. The earlier "no
hard blocker" note was wrong.

Below the window the knob is a no-op: default mode already keeps MTP active
there. It only matters above the window.

## Fix: sliding draft KV

`common_speculative_impl_draft_mtp::process()` (common/speculative.cpp:1478)
now trims the oldest cells before the catch-up decode:

    const int32_t keep = std::max(1, (int32_t) llama_n_ctx(ctx_dft) - 256);
    // per seq: if pos_new >= keep: seq_rm(mem_dft, seq_id, 0, pos_new - keep + 1)

Absolute positions are kept, so RoPE and the causal mask stay correct. The
QWEN35 MTP layer is not SWA, so the built-in path does not apply. This only
changes draft acceptance, never target output. The trim is unconditional but a
no-op below the window, so default mode is unaffected.

## Outcome

`benchmarks/benchmark_mtp_streaming.py`, arena 3072, ctx 160000, q8_0 K /
q4_0 V, 256 decode tokens, repo-source prompt. Default ejects MTP as soon as
the working set passes the pin; keep mode (the benchmark's keep-throughout arm)
leaves it active with zero draft errors.

| prompt | keep tg t/s | default tg t/s | keep/default | keep pp t/s | default pp t/s |
|---:|---:|---:|---:|---:|---:|
| 50,000 | 30.40 | 22.32 | 1.36x | 686.8 | 711.4 |
| 80,000 | 23.25 | 19.61 | 1.19x | 598.6 | 670.2 |
| 90,000 | 21.47 | 18.81 | 1.14x | 571.7 | 653.4 |
| 100,000 | 17.21 | 18.13 | 0.95x | 548.3 | 636.9 |
| 110,000 | 11.46 | 17.48 | 0.66x | 525.3 | 619.0 |
| 120,000 | 12.68 | 15.80 | 0.80x | 505.2 | 599.5 |
| 160,000 | 7.15 | 12.15 | 0.59x | 435.1 | 522.7 |

The decode crossover is about 97,000 tokens (linear between 90K and 100K).
Keeping MTP costs prefill throughout: -3.5% at 50K rising to about -17% at
160K, because the draft runs during prefill too. The 110K keep point is noisier
than its neighbours (acceptance 53% vs 73% at 120K); the 90K-to-100K crossover
is not sensitive to it.

`--kv-stream-spec-keep-pages N` turns this into a knob: MTP stays active until
the target's decode working set exceeds `N` 256-token pages, then ejects like the
default. `0` (default) ejects at streaming onset; setting `N` near the crossover
(about 380 pages at arena 3072) captures most of the gain while ejecting past it.
Verified at arena 3072 with `N=380`: 90K tokens keeps MTP active (21.06 t/s vs
18.82 ejected) and 110K ejects (17.50 t/s vs 17.47 default).

## Remaining work

Phase 2, find the crossover automatically: expose the streaming runtime's
copy-pressure signals through `llama_kv_stream_status`. `deadline_miss_ratio`
and `copy_busy_ratio` are already computed per ubatch
(src/llama-kv-cache.cpp:1441-1456) but are not reported. Eject on measured
copy pressure instead of streaming onset, so the controller adapts to the
machine and prompt rather than a fixed threshold.

## Notes

- Verify batches are admitted at `max(n_seq_max, 1 + n_max_spec_draft)`
  (src/llama-context.cpp:2174-2185) and the concentrated decode layout handles
  query batches up to 32 tokens (src/llama-kv-cache.cpp:1427, 1479-1481), so a
  4-token MTP verify is a normal decode under streaming.
- Staying active also avoids the toggle cost: `synchronize()` + CUDA graph
  reset + scheduler reset + rs rebuild (src/llama-context.cpp:1507-1516).
- Task 1 helps here: a quantized MTP KV makes the same pin hold a larger MTP
  window.

## Acceptance

- Met: the sliding draft KV keeps MTP active past the pin window with no draft
  errors, and keep mode is faster than the ejected baseline below ~97K tokens.
- Not met: the controller still ejects on streaming onset in default mode. The
  copy-pressure policy above would make it eject near the measured crossover.
