# Dynamic MTP Eject - Milestone 2/3 (server auto eject + re-enable) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the phase-arena KV pool starts streaming, automatically eject MTP (freeing its pinned arena reservation for the pool), and automatically re-enable MTP once the context shrinks enough that the reservation fits again.

**Architecture:** Expose a read-only streaming/pool status from the target context, factor the server's spec-context lifecycle into create/destroy/rewire helpers, then add a between-ubatch controller in `server_context_impl` that ejects and re-enables via M1's `llama_kv_stream_mtp_set`. The behavior is opt-in (default off) and threshold-configured, because the crossover point still needs A/B measurement.

**Tech Stack:** C++17, CUDA backend proc-address API, llama-server.

**Spec:** `docs/superpowers/specs/2026-09-12-mtp-dynamic-eject-design.md` (Components 4-5, Milestones 2-3)

**Depends on:** M1 plan `docs/superpowers/plans/2026-09-12-mtp-dynamic-eject.md` (complete; `llama_kv_stream_mtp_set` exists).

## Global Constraints

- ASCII only in code and comments. No em-dashes or unicode arrows.
- Follow surrounding llama.cpp style; comments only where non-obvious.
- Build: `cmake --build build --config Release -j 12` from the repo root.
- Do NOT commit `benchmarks/benchmark_kv_stream.py` (unrelated).
- Per-task commits are pre-approved on this branch. Use `Assisted-by: opencode`, never `Co-authored-by`.
- Do not add files under `tests/` (needs maintainer approval). GPU/integration drivers live in `/tmp/opencode`.
- The feature is OPT-IN: new behavior defaults off; existing runs must be unchanged unless the flag is set.
- Only MTP is affected. ngram and other draft types must be untouched.
- All dynamic transitions happen at the top of `update_slots`, between complete batched decodes, never inside `decode()`/`post_decode()`.

---

### Task 1: Streaming/pool status API on the context

Expose the streaming predicate and pool geometry so the server can decide when to eject and re-enable.

**Files:**
- Modify: `include/llama.h` (new public struct + declaration near the `llama_kv_stream_pinned_buft` block, ~line 606)
- Modify: `src/llama-kv-cache.h` (owner fields ~300-336, small const getters)
- Modify: `src/llama-kv-cache.cpp` (proc fetch ~280-305, `kv_stream_adapt` ~1410-1414, new getters)
- Modify: `src/llama-context.h` (member declaration near `get_kv_stream_pinned_buft`)
- Modify: `src/llama-context.cpp` (public free function near `llama_kv_stream_mtp_set`)

**Interfaces:**
- Consumes: `kv_stream_runtime_owner` (src/llama-kv-cache.h:300), `feedback_fn` outputs (ring_slots, resident_pages_per_layer, controlled_pool_pages), `kv_stream_pinned_bytes_for`.
- Produces: `bool llama_kv_stream_get_status(struct llama_context * ctx, struct llama_kv_stream_status * status);` and `bool llama_context::kv_stream_get_status(llama_kv_stream_status * status) const;`

- [ ] **Step 1: Define the public struct and declare the function**

In `include/llama.h`, add the struct at file scope near the other public structs (for example just before `struct llama_context_params`), and the function next to `llama_kv_stream_pinned_buft`:

```c
    struct llama_kv_stream_status {
        bool     enabled;                    // streaming runtime present and arena configured
        bool     streaming;                  // active pages exceed resident pages
        uint32_t active_pages;               // active KV pages per layer
        uint32_t resident_pages_per_layer;   // resident pages per layer
        uint32_t ring_slots;                 // ring/staging slots
        uint32_t layer_count;                // streaming attention layers
        uint32_t page_bytes;                 // bytes per KV page
        uint64_t pool_free_bytes;            // controlled pool bytes minus the active working set
        uint64_t mtp_reserved_bytes;         // pinned(true) - pinned(false); 0 when MTP is not configured
    };
```

```c
    // Query the phase-arena KV streaming status. Returns false when streaming is not enabled.
    LLAMA_API bool llama_kv_stream_get_status(struct llama_context * ctx, struct llama_kv_stream_status * status);
```

- [ ] **Step 2: Persist the streaming fields in the KV cache owner**

In `src/llama-kv-cache.h`, inside `kv_stream_runtime_owner` (after `resize_pool_fn` or with the other state fields), add:

```cpp
        size_t (*pool_bytes_fn)(void *) = nullptr;
        bool     streaming = false;
        uint32_t active_pages = 0;
        uint32_t resident_pages_per_layer = 0;
        uint32_t ring_slots = 0;
        uint32_t controlled_pool_pages = 0;
```

Add public const getters to `llama_kv_cache` near `kv_stream_adapt` (src/llama-kv-cache.h:165):

```cpp
    bool     kv_stream_streaming() const;
    uint64_t kv_stream_pool_free_bytes() const;
    uint32_t kv_stream_active_pages() const;
    uint32_t kv_stream_resident_pages() const;
    uint32_t kv_stream_ring_slots() const;
    uint32_t kv_stream_layer_count() const;
    uint32_t kv_stream_page_bytes() const;
```

In `src/llama-kv-cache.cpp`, fetch the pool-bytes proc next to the other fetches (around line 280-305), add it to the null check, and store it. Read the actual surrounding code and match the local names exactly:

```cpp
    owner.pool_bytes_fn = (size_t (*)(void *)) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_cuda_kv_stream_pool_bytes");
```

In `llama_kv_cache::kv_stream_adapt` (src/llama-kv-cache.cpp:1381), immediately after the feedback call that produces `resident_pages`/`ring_slots`/`controlled_pages` (around line 1410-1414), store:

```cpp
    kv_stream_runtime.streaming = active_pages > resident_pages;
    kv_stream_runtime.active_pages = active_pages;
    kv_stream_runtime.resident_pages_per_layer = resident_pages;
    kv_stream_runtime.ring_slots = ring_slots;
    kv_stream_runtime.controlled_pool_pages = controlled_pages;
```

Use the exact local names present in that function for the four values.

- [ ] **Step 3: Implement the getters**

In `src/llama-kv-cache.cpp` (near `kv_stream_resize_pool`):

```cpp
bool llama_kv_cache::kv_stream_streaming() const {
    return kv_stream_runtime.streaming;
}

uint64_t llama_kv_cache::kv_stream_pool_free_bytes() const {
    const auto & o = kv_stream_runtime;
    if (o.runtime == nullptr || o.layer_count == 0) {
        return 0;
    }
    const uint64_t controlled = o.controlled_pool_pages;
    const uint64_t used = uint64_t(o.active_pages)*o.layer_count;
    if (controlled <= used) {
        return 0;
    }
    return (controlled - used)*o.page_bytes;
}

uint32_t llama_kv_cache::kv_stream_active_pages() const {
    return kv_stream_runtime.active_pages;
}

uint32_t llama_kv_cache::kv_stream_resident_pages() const {
    return kv_stream_runtime.resident_pages_per_layer;
}

uint32_t llama_kv_cache::kv_stream_ring_slots() const {
    return kv_stream_runtime.ring_slots;
}

uint32_t llama_kv_cache::kv_stream_layer_count() const {
    return kv_stream_runtime.layer_count;
}

uint32_t llama_kv_cache::kv_stream_page_bytes() const {
    return (uint32_t) kv_stream_runtime.page_bytes;
}
```

Confirm the owner field that holds bytes per page (used here as `page_bytes`); if it is named differently, use the real name.

- [ ] **Step 4: Add the context method and C function**

In `src/llama-context.h`, add near `get_kv_stream_pinned_buft`:

```cpp
    bool kv_stream_get_status(llama_kv_stream_status * status) const;
```

In `src/llama-context.cpp`, add the method near `llama_context::get_kv_stream_pinned_buft`:

```cpp
bool llama_context::kv_stream_get_status(llama_kv_stream_status * status) const {
    if (status == nullptr || !kv_stream_phase_arena.configured) {
        return false;
    }

    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (hybrid == nullptr || hybrid->get_mem_attn() == nullptr) {
        return false;
    }

    const llama_kv_cache * kv = hybrid->get_mem_attn();

    status->enabled = true;
    status->streaming = kv->kv_stream_streaming();
    status->active_pages = kv->kv_stream_active_pages();
    status->resident_pages_per_layer = kv->kv_stream_resident_pages();
    status->ring_slots = kv->kv_stream_ring_slots();
    status->layer_count = kv->kv_stream_layer_count();
    status->page_bytes = kv->kv_stream_page_bytes();
    status->pool_free_bytes = kv->kv_stream_pool_free_bytes();
    status->mtp_reserved_bytes = spec_mtp_configured
        ? kv_stream_pinned_bytes_for(true) - kv_stream_pinned_bytes_for(false)
        : 0;

    return true;
}
```

Add the public free function next to `llama_kv_stream_mtp_set`:

```cpp
bool llama_kv_stream_get_status(llama_context * ctx, llama_kv_stream_status * status) {
    return ctx->kv_stream_get_status(status);
}
```

If `llama_context::kv_stream_mtp_set` is not const and `memory.get()` in a const method causes a compile error, make `kv_stream_get_status` non-const and adjust the header.

- [ ] **Step 5: Extend the harness to assert the status (RED then GREEN)**

In `/tmp/opencode/arena_toggle_test.cpp`, after the target context is created and after the prefill decode, call the API and check the fields:

```cpp
llama_kv_stream_status st = {};
if (!llama_kv_stream_get_status(ctx, &st)) { printf("TEST: FAIL status not enabled\n"); return 1; }
if (!st.enabled) { printf("TEST: FAIL status enabled flag\n"); return 1; }
if (st.mtp_reserved_bytes == 0) { printf("TEST: FAIL mtp reserved bytes zero\n"); return 1; }
if (st.pool_free_bytes == 0) { printf("TEST: FAIL pool free bytes zero\n"); return 1; }
if (st.layer_count == 0 || st.page_bytes == 0) { printf("TEST: FAIL geometry zero\n"); return 1; }
printf("TEST: status streaming=%d active=%u resident=%u ring=%u layers=%u page_bytes=%u free=%llu reserved=%llu\n",
       (int) st.streaming, st.active_pages, st.resident_pages_per_layer, st.ring_slots,
       st.layer_count, st.page_bytes, (unsigned long long) st.pool_free_bytes,
       (unsigned long long) st.mtp_reserved_bytes);
```

Also assert the behavior invariant across a toggle: capture `st.mtp_reserved_bytes` before any toggle, then after `llama_kv_stream_mtp_set(ctx,false)` the reserved value is unchanged and `pool_free_bytes` has grown.

- [ ] **Step 6: Build and run**

```bash
cmake --build build --config Release -j 12
g++ -std=c++17 -O2 -I include -I ggml/include /tmp/opencode/arena_toggle_test.cpp \
    -L build/bin -lllama -lggml -lggml-base -Wl,-rpath,'$ORIGIN' -o build/bin/arena_toggle_test
cd build/bin && ./arena_toggle_test 2>&1 | tail -8
```

Expected: `TEST: PASS` and a status line with non-zero geometry, free bytes, and reserved bytes. Before Step 1-4 the harness fails to compile/link (RED).

- [ ] **Step 7: Commit**

```bash
git add include/llama.h src/llama-kv-cache.h src/llama-kv-cache.cpp src/llama-context.h src/llama-context.cpp
git commit -m "kv-stream : expose phase arena streaming status

Assisted-by: opencode"
```

---

### Task 2: Factor the server spec lifecycle into helpers

Extract the MTP/spec context create, destroy and slot-rewire logic from `load_model`/`destroy` so the dynamic controller can reuse it. No behavior change.

**Files:**
- Modify: `tools/server/server-context.cpp` (`server_context_impl` methods)

**Interfaces:**
- Consumes: `common_base_params_to_speculative`, `common_speculative_init_from_params`, `common_speculative_init`, existing `has_draft`/`spec_mtp`/`has_spec` locals (tools/server/server-context.cpp:1019-1023).
- Produces: `bool spec_create()`, `void spec_destroy()`, `void spec_rewire_slots(bool enable)`.

- [ ] **Step 1: Read the existing blocks**

Read `tools/server/server-context.cpp` around 938-952 (`destroy`), 1019-1023 (`has_spec` flags), 1120-1151 (creation), 1258-1294 (spec init + slot wiring). These are the exact lines to factor.

- [ ] **Step 2: Add the helpers**

In `server_context_impl`, add (declare in the class and define in the .cpp):

```cpp
bool spec_create() {
    common_params params_dft = common_base_params_to_speculative(params_base);
    spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    model_dft = spec_init->model();
    ctx_dft = spec_init->context();
    if (ctx_dft == nullptr) {
        spec_init.reset();
        model_dft = nullptr;
        return false;
    }
    params_base.speculative.draft.ctx_tgt = ctx_tgt;
    params_base.speculative.draft.ctx_dft = ctx_dft;
    ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
    ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
    spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
    if (spec == nullptr) {
        spec_init.reset();
        ctx_dft = nullptr;
        model_dft = nullptr;
        params_base.speculative.draft.ctx_dft = nullptr;
        return false;
    }
    return true;
}

void spec_destroy() {
    spec.reset();
    spec_init.reset();
    ctx_dft = nullptr;
    model_dft = nullptr;
    params_base.speculative.draft.ctx_dft = nullptr;
}

void spec_rewire_slots(bool enable) {
    for (auto & slot : slots) {
        slot.ctx_dft = enable ? ctx_dft : nullptr;
        slot.spec = enable ? spec.get() : nullptr;
        slot.mem.init(ctx_tgt, enable ? ctx_dft : nullptr);
        slot.spec_draft.clear();
        slot.spec_i_batch.clear();
        slot.spec_ckpt.clear();
        slot.spec_is_replay = false;
        slot.spec_prompt.clear();
    }
}
```

Match the real member names and types exactly; `slots` may be accessed differently. Preserve the existing order and any extra bookkeeping the original blocks contain (for example the `ctx_tgt_seq_rm_type != NO` guard around `common_speculative_init`).

- [ ] **Step 3: Replace the inlined blocks with the helpers**

- In `destroy()` (around 938-952), replace the first four statements with `spec_destroy();` (keep the target teardown that follows).
- In `load_model` (around 1120-1151), replace the creation block body with `spec_create()` and keep the surrounding progress-logging and the `has_spec` guard.
- Replace the spec-init + slot-wiring block (around 1258-1294) so the successful path calls `spec_rewire_slots(true)`, and the existing failure path calls `spec_destroy()`.

- [ ] **Step 4: Build and smoke test**

```bash
cmake --build build --config Release -j 12
```

Start the server with MTP and with ngram, confirm both still load, and confirm a `/completion` returns text. Use the existing harness:

```bash
bash /tmp/opencode/vram2.sh 3072 draft-mtp t2factor
bash /tmp/opencode/vram2.sh 3072 ngram-mod t2factorng
```

Expected: both `UP=1` with no load errors. No functional change.

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "server : factor speculative context lifecycle helpers

Assisted-by: opencode"
```

---

### Task 3: Server auto-eject (M2)

Add the opt-in dynamic controller that ejects MTP when the pool is under pressure.

**Files:**
- Modify: `common/common.h` (`common_params_speculative`, ~370-401)
- Modify: `common/arg.cpp` (near the other `--spec-*` options and the kv-stream options ~2423-2432)
- Modify: `tools/server/server-context.cpp` (state near `n_empty_consecutive` ~913; controller at the top of `update_slots` ~2777-2818)

**Interfaces:**
- Consumes: `llama_kv_stream_get_status`, `llama_kv_stream_mtp_set`, `llama_set_embeddings_nextn`, `spec_create`/`spec_destroy`/`spec_rewire_slots`.
- Produces: config fields `kv_stream_mtp_dynamic`, `kv_stream_mtp_eject_mib`, `kv_stream_mtp_reenable_pages`, `kv_stream_mtp_stable_decodes`; methods `void update_mtp_dynamic();` `bool mtp_eject();`.

- [ ] **Step 1: Add the config fields**

In `common/common.h`, in `struct common_params_speculative` (near `draft.n_max`):

```cpp
    bool    kv_stream_mtp_dynamic        = false; // eject MTP under pool pressure, re-enable when it fits
    int32_t kv_stream_mtp_eject_mib      = 128;   // eject when free pool bytes <= this
    int32_t kv_stream_mtp_reenable_pages = 16;    // re-enable when resident - active pages >= this
    int32_t kv_stream_mtp_stable_decodes = 8;     // consecutive controller checks needed
```

In `common/arg.cpp`, add four options next to the existing `--spec-*` group, following the existing `common_arg(...).set_env(...)` pattern:

```cpp
    arg = common_arg("--kv-stream-mtp-dynamic", &params.speculative.kv_stream_mtp_dynamic, "eject MTP when the KV pool streams and re-enable it when it fits again");
    arg = common_arg("--kv-stream-mtp-eject-mib", &params.speculative.kv_stream_mtp_eject_mib, "free pool MiB at or below which MTP is ejected");
    arg = common_arg("--kv-stream-mtp-reenable-pages", &params.speculative.kv_stream_mtp_reenable_pages, "resident-page headroom required to re-enable MTP");
    arg = common_arg("--kv-stream-mtp-stable-decodes", &params.speculative.kv_stream_mtp_stable_decodes, "consecutive controller checks before a transition");
```

Read the real `common_arg` constructor/overloads in the surrounding code and match them (some use `.set_env("LLAMA_ARG_...")`). Keep the descriptions ASCII and short.

- [ ] **Step 2: Add controller state and the trigger call**

In `server_context_impl`, add members and a helper:

```cpp
    bool     spec_mtp_enabled_dynamic = false;
    bool     mtp_ejected = false;
    uint32_t mtp_stable = 0;
    uint32_t mtp_resident_pages_capture = 0;
    bool     mtp_capture_valid = false;
```

At the top of `update_slots()` (before the all-idle early return around 2792-2807), call `update_mtp_dynamic();`.

Implement:

```cpp
void update_mtp_dynamic() {
    if (!params_base.speculative.kv_stream_mtp_dynamic) {
        return;
    }
    if (!spec_mtp_enabled_dynamic || ctx_tgt == nullptr) {
        return;
    }

    llama_kv_stream_status st = {};
    if (!llama_kv_stream_get_status(ctx_tgt, &st) || !st.enabled || st.mtp_reserved_bytes == 0) {
        return;
    }

    const uint64_t eject_bytes = uint64_t(std::max(0, params_base.speculative.kv_stream_mtp_eject_mib))*1024ull*1024ull;

    if (!mtp_ejected) {
        if (!streaming_pressure_reached(st, eject_bytes)) {
            mtp_stable = 0;
            return;
        }
        if (++mtp_stable < (uint32_t) std::max(1, params_base.speculative.kv_stream_mtp_stable_decodes)) {
            return;
        }
        if (mtp_eject()) {
            mtp_stable = 0;
        }
        return;
    }

    if (!mtp_capture_valid) {
        return;
    }
    const int32_t margin = params_base.speculative.kv_stream_mtp_reenable_pages;
    if ((int64_t) mtp_resident_pages_capture - (int64_t) st.active_pages < margin) {
        mtp_stable = 0;
        return;
    }
    if (++mtp_stable < (uint32_t) std::max(1, params_base.speculative.kv_stream_mtp_stable_decodes)) {
        return;
    }
    if (mtp_enable()) {
        mtp_stable = 0;
    }
}
```

`streaming_pressure_reached(st, eject_bytes)` is `st.streaming || st.pool_free_bytes <= eject_bytes`.

Guard `spec_mtp` using the same condition the load path uses (tools/server/server-context.cpp:1019-1023: the types vector contains `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`). Store it in a new member at load time, for example `spec_mtp_enabled_dynamic = spec_mtp;`. Do not call the controller when it is false.

- [ ] **Step 3: Implement `mtp_eject()` (M2; `mtp_enable()` is Task 4)**

```cpp
bool mtp_eject() {
    llama_kv_stream_status st = {};
    if (!llama_kv_stream_get_status(ctx_tgt, &st)) {
        return false;
    }
    mtp_resident_pages_capture = st.resident_pages_per_layer;
    mtp_capture_valid = true;

    for (auto & slot : slots) {
        slot.spec_draft.clear();
        slot.spec_i_batch.clear();
        slot.spec_ckpt.clear();
        slot.spec_is_replay = false;
        slot.spec_prompt.clear();
    }

    spec_destroy();
    llama_set_embeddings_nextn(ctx_tgt, false, false);
    if (!llama_kv_stream_mtp_set(ctx_tgt, false)) {
        return false;
    }
    spec_rewire_slots(false);
    mtp_ejected = true;
    LLAMA_LOG_INFO("%s: MTP ejected, pool free = %llu bytes\n", __func__, (unsigned long long) llama_kv_stream_pool_free_bytes(ctx_tgt));
    return true;
}
```

`llama_kv_stream_pool_free_bytes` is not a public symbol (Task 1 made the query a status call). Use a status call to log the free bytes instead:

```cpp
    llama_kv_stream_status st2 = {};
    if (llama_kv_stream_get_status(ctx_tgt, &st2)) {
        LLAMA_LOG_INFO("%s: MTP ejected, pool free = %llu bytes\n", __func__, (unsigned long long) st2.pool_free_bytes);
    }
```

- [ ] **Step 4: Build and GPU validation**

```bash
cmake --build build --config Release -j 12
```

Run the server with MTP plus the new flag and a long prompt, then inspect the log:

```bash
bash /tmp/opencode/pp_test.sh 3072 draft-mtp m2 --kv-stream-mtp-dynamic --ctx-size 180000
```

`/tmp/opencode/pp_test.sh` is a template; confirm it forwards extra args. The expected log sequence is: the streaming/decode reserve lines, then `MTP ejected`, then the RS buffer line dropping from about `598.50 MiB` to about `149.62 MiB`, and generation continuing with `draft acceptance = 0` while ejected. If the prompt is too short to stream, use a longer prompt or a smaller arena.

- [ ] **Step 5: Commit**

```bash
git add common/common.h common/arg.cpp tools/server/server-context.cpp
git commit -m "server : auto-eject MTP when the KV pool streams

Assisted-by: opencode"
```

---

### Task 4: Server auto re-enable (M3)

Re-enable MTP once the captured MTP-active capacity again exceeds the active working set by the configured margin.

**Files:**
- Modify: `tools/server/server-context.cpp` (add `mtp_enable()`; the re-enable branch in `update_mtp_dynamic` is already written in Task 3)

**Interfaces:**
- Consumes: `mtp_resident_pages_capture`, `mtp_capture_valid`, `spec_create`, `spec_rewire_slots`.
- Produces: `bool mtp_enable();`.

- [ ] **Step 1: Implement `mtp_enable()`**

```cpp
bool mtp_enable() {
    llama_set_embeddings_nextn(ctx_tgt, true, false);
    if (!llama_kv_stream_mtp_set(ctx_tgt, true)) {
        llama_set_embeddings_nextn(ctx_tgt, false, false);
        return false;
    }
    if (!spec_create()) {
        llama_kv_stream_mtp_set(ctx_tgt, false);
        llama_set_embeddings_nextn(ctx_tgt, false, false);
        return false;
    }
    spec_rewire_slots(true);
    mtp_ejected = false;
    LLAMA_LOG_INFO("%s: MTP re-enabled\n", __func__);
    return true;
}
```

Note the hard ordering: re-pin/enable must run before `spec_create`, because the MTP draft weights and KV allocate from the pinned region. If `spec_create` fails, roll the toggle back.

- [ ] **Step 2: Build and GPU validation**

```bash
cmake --build build --config Release -j 12
```

Run the dynamic server, cross the eject threshold with a long prompt, then reset/compact the slot (new short request on the same slot, or clear the conversation) so the active KV drops below the captured capacity minus the margin. Expected log: `MTP re-enabled`, the RS line returning to about `598.50 MiB`, `draft acceptance` returning to non-zero, and no assert. Confirm no thrash: the sequence eject -> re-enable must not repeat on consecutive checks.

- [ ] **Step 3: Commit**

```bash
git add tools/server/server-context.cpp
git commit -m "server : auto re-enable MTP when the pool has headroom

Assisted-by: opencode"
```

---

### Task 5: End-to-end A/B validation (no commit)

**Files:**
- Test only: `/tmp/opencode` scripts and logs.

- [ ] **Step 1: Measure the MTP crossover**

Run the same long-prompt workload with `--kv-stream-mtp-dynamic` off and on, and record prompt-eval tok/s, generation tok/s, the ejection point (tokens), and the re-enable point. Vary `--kv-stream-mtp-eject-mib` and `--kv-stream-mtp-reenable-pages` to find where MTP stops paying off.

- [ ] **Step 2: Confirm no regression when disabled**

Run with the flag absent and confirm the log, VRAM, and throughput match the pre-M2 behavior.

- [ ] **Step 3: Report**

Report the crossover numbers and the recommended defaults. No repo files change.

---

## Self-Review

- Spec coverage: Component 4 (streaming/pool query) -> Task 1; Component 5 (server eject/re-enable, hysteresis) -> Tasks 2-4; Milestone 2 -> Task 3; Milestone 3 -> Task 4; the A/B measurement the user asked for -> Task 5. The streaming-onset trigger and headroom/hysteresis are all represented.
- Deviation from the spec: the spec named `llama_kv_stream_streaming` and `llama_kv_stream_pool_free_pages`; this plan uses one `llama_kv_stream_get_status` struct because the controller also needs active/resident pages and the MTP reservation size. Ruling recorded; the spec's names are illustrative.
- No placeholders: each code step contains the code to write; the server extraction steps name exact line ranges and preserve behavior.
- Type consistency: `llama_kv_stream_status`, `llama_kv_stream_get_status`, `update_mtp_dynamic`, `mtp_eject`, `mtp_enable`, `spec_create`/`spec_destroy`/`spec_rewire_slots` are used with the same signatures throughout.
