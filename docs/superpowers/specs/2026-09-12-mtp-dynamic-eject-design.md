# Dynamic MTP eject / re-enable in the KV-stream phase arena

## Status

Approved design. Implementation to follow. Branch: `feature/kv-stream-phase-arena-spec`.

## Background

The phase-arena KV streaming implementation allocates one fixed CUDA arena and
partitions it into a KV streaming pool, a per-phase compute slab, and a top
"pinned" region. When MTP speculative decoding is enabled on QWEN35, three
things are pinned into that top region so a fixed arena can cover them:

- the target's recurrent-state (rs) cache, widened by `n_rs_seq` (~598.5 MiB at
  `n_rs_seq=3`, vs ~149.6 MiB at `n_rs_seq=0`);
- the separate MTP draft model weights (~605 MiB);
- the MTP context KV cache (~704 MiB).

Measured with ctx 180000, q8_0/q4_0, `--kv-stream-arena-mib 3072`:

| Region (MiB)            | MTP active | MTP ejected | no MTP (ngram) |
|-------------------------|-----------:|------------:|---------------:|
| rs cache pinned         | 598.5      | 149.6       | 0 (outside)    |
| MTP weights pinned      | 605        | 0           | 0              |
| MTP KV pinned           | 704        | 0           | 0              |
| pinned subtotal         | 1907.5     | 149.6       | 0              |
| compute slab (decode)   | 706        | 706         | 737            |
| KV pool, decode         | 453        | ~2216       | 2335           |
| KV pool, prefill        | 199        | ~1962       | 2064           |

MTP is a speed optimization that helps short/medium contexts. It costs pool
space, which matters once the sequence is long enough that the KV working set
exceeds the resident capacity and pages start streaming to host. The goal is to
eject MTP when streaming begins, reclaim its arena space for the pool, and later
re-enable MTP when the context shrinks again.

## Problem

Ejecting MTP must undo the three pinned allocations. Freeing the MTP context and
draft model releases the weights and the MTP KV automatically, but the target rs
cache is a live target allocation that cannot be freed or resized, so the pinned
region can never shrink. Two existing mechanisms block the feature:

- `ggml_backend_cuda_phase_arena_set_pinned` refuses while `pinned_refs != 0`,
  and `reset_pinned` asserts `pinned_refs == 0`. Every live pinned tenant holds a
  ref.
- `llama_memory_recurrent` has no free/realloc path (tensors are allocated once
  into `ctxs_bufs`), is non-copyable/non-movable, and is held as a
  `const std::unique_ptr` inside `llama_memory_hybrid`, so it cannot be swapped.

It is also not possible to simply leave rs outside the arena: a 3072 arena plus
the 598.5 MiB rs allocation does not fit (verified: fails allocating the rs
cache at 2816 and 3072).

## Design

### Invariant

Every transition happens between ubatches, in the decode loop, exactly like
`llama_context::kv_stream_switch_phase`:

1. `synchronize()`;
2. `graph_reset_fn(backend_ptrs[backend_index])`;
3. `sched.reset()`;
4. `gf_res_prev->reset()` and `gf_res_reserve->reset()`;
5. mutate memory / arena;
6. recreate the scheduler and re-run `graph_reserve`.

No graph may hold raw r/s tensor pointers across the mutation, and no rs
rollback may be pending (`rs_idx == 0` for all sequences).

### Component 1: rs cache rebuild

Row layout is `row = plane * mem_size + cell`, plane 0 = committed state,
planes `1..n_rs_seq` = per-token rollback snapshots. The committed state is
therefore rows `[0, mem_size)`.

Add to `llama_memory_recurrent`:

```cpp
// resize the rollback planes, preserving the committed state
bool rebuild(uint32_t n_rs_seq, ggml_backend_buffer_type_t secondary_buft);
```

Semantics:

- refuse if any `rs_idx[seq] != 0` (pending rollback) or if the object is not in
  a clean between-ubatch state;
- stage rows `[0, mem_size)` of `r_l/s_l/p_l` per layer to host;
- drop `ctxs_bufs` (this is what releases the pinned ref);
- allocate new tensors with `n_rows = mem_size * (1 + n_rs_seq)` from the given
  buft, referencing the fresh pinned region;
- restore plane 0, zero planes `1..n_rs_seq`;
- reset `rs_idx` to 0, recompute `rs_z`, keep `cells/head/size/used` (plane- and
  size-independent);
- the class needs a stored way to reproduce per-layer buft selection (today it
  only keeps `hparams`). Store either the model pointer or the resolved
  per-layer buft vector captured at construction.

### Component 2: arena re-pin

- Wire the already-exported `ggml_backend_cuda_phase_arena_reset_pinned` proc
  address into `kv_stream_phase_arena_owner` (new `reset_pinned_fn` field).
- Add a context method that, called with `pinned_refs == 0`:
  - `reset_pinned(arena)`;
  - `set_pinned(arena_full - new_pinned, new_pinned)`;
  - updates `owner.pinned_bytes` and `owner.arena_bytes = arena_full - new_pinned`;
  - forces a re-reserve (new force flag, or make the reserve unconditional for
    this path) so `prefill`/`token_generation` layouts and the pool are
    recomputed.
- `kv_stream_pinned_bytes` is currently a constructor local. Add a helper that
  recomputes the components (rs bytes, MTP weights bytes, MTP KV bytes) from
  `cparams`/`hparams` so both the constructor and the runtime path share one
  source of truth.

### Component 3: KV runtime maximum pool

The streaming runtime is created with `maximum_pool_bytes = effective arena`,
which caps `resize_pool`. Create it with the full arena size instead (the CUDA
check allows `maximum_pool_bytes <= phase_arena->size`), so the pool can grow
after a re-pin without recreating the runtime and losing resident/ring state.
Verify the ring-capacity derivation uses the current `pool_bytes`, not the
maximum, before relying on this.

### Component 4: context API

```cpp
// flip the arena between the MTP-reserved and MTP-free layouts
bool llama_kv_stream_mtp_set(llama_context * ctx, bool mtp_active);

// pool free headroom and streaming state
uint64_t llama_kv_stream_pool_free_pages(llama_context * ctx);
bool     llama_kv_stream_streaming(llama_context * ctx);
```

`mtp_active = false` (eject): requires the MTP context and draft model to have
been destroyed already (so their pinned refs are released); rebuilds rs to
`n_rs_seq = 0`; re-pins to `rs_small`; sets `cparams.spec_mtp = false`,
`cparams.n_max_spec_draft = 0`, `cparams.n_rs_seq = 0`; forces re-reserve.

`mtp_active = true` (enable): requires the pinned region to hold only the small
rs cache; frees it, re-pins to `rs_big + weights + mtp_kv`, rebuilds rs to
`n_rs_seq = n_max`, then returns so the caller can create the MTP context (its
weights and KV allocate from the pinned buft).

The streaming signal already exists internally: `kv_stream_adapt` computes
`active_pages > resident_pages`. Persist it on the owner and surface it.

### Component 5: server orchestration

Eject (`tools/server/server-context.cpp`):

1. stop scheduling verify work for every slot; clear `slot.spec_draft`,
   `spec_ckpt`, `spec_i_batch`, `spec_is_replay`, set drafting false;
2. `spec.reset()` then `spec_init.reset()`;
3. `ctx_dft = nullptr`, `model_dft = nullptr`, update every slot
   (`slot.ctx_dft = nullptr`, `slot.spec = nullptr`,
   `slot.mem.init(ctx_tgt, nullptr)`);
4. `llama_set_embeddings_nextn(ctx_tgt, false, false)`;
5. `llama_kv_stream_mtp_set(ctx_tgt, false)`.

Re-enable: call `llama_kv_stream_mtp_set(ctx_tgt, true)` first, then rebuild the
spec context exactly as the load path does (`common_base_params_to_speculative`
+ `common_speculative_init_from_params` + `common_speculative_init`), then
rewire slots.

Trigger and hysteresis:

- eject when the streaming signal has been true and pool free headroom has
  fallen below a margin for N consecutive decodes;
- re-enable only when active pages again fit the projected MTP-active resident
  capacity with margin, stable for M consecutive decodes (M > N), and only
  between sequences where no rollback is pending.

Thresholds are configurable (CLI or server option) with conservative defaults.

## Ordering summary

Eject: spec teardown (frees MTP weights + KV refs) -> rs rebuild (frees old rs)
-> `pinned_refs == 0` -> re-pin small -> allocate small rs -> re-reserve
(pool grows).

Enable: free small rs -> `pinned_refs == 0` -> re-pin large -> rebuild large rs
-> create MTP (weights + KV allocate from pinned) -> re-reserve (pool shrinks).

## Memory accounting

At arena 3072 the ejected pool is ~2216 MiB decode vs 2335 for a pure ngram run,
within ~5%. MTP-active pool stays ~453 MiB.

## Testing

- Unit-ish: `llama_memory_recurrent::rebuild` plane-0 copy for shrink and grow;
  rs bytes helper; pinned bytes recompute.
- GPU integration (`llama-server`): a long prompt crosses the streaming
  threshold, MTP ejects, log shows the pinned shrink and pool grow, generation
  continues correctly; then a short/compacted context re-enables MTP and
  acceptance returns. Build with the standard flags:
  `cmake --build build --config Release -j 12`.
- Regression: ngram-only runs must be unchanged (no pinned region, full pool).

## Milestones

1. `rebuild` + re-pin + `llama_kv_stream_mtp_set`, driven by a debug hook.
   Validate toggling at runtime (pool sizes, correct output, no assert).
2. Server auto-eject on streaming onset with headroom margin.
3. Server auto re-enable with hysteresis.

## Risks

- rs state migration correctness; must block on pending rollback.
- graph reuse invalidation across the rebuild (`can_reuse` checks `n_rs`/`head`;
  graphs are reset anyway).
- `maximum_pool_bytes` / ring-capacity derivation.
- spec teardown / re-create rewiring (slots, checkpoints). The borrowed target
  LM head is safe: `model_dft` never owns `model_tgt->output`, and destruction
  order frees the draft before the target.
- MTP weight reload latency on re-enable.

## Out of scope

- Re-enabling MTP across a target context recreate (a context-size change still
  recreates the target).
- Non-QWEN35 architectures and non-MTP speculative types.
- Resizing the target context itself.
