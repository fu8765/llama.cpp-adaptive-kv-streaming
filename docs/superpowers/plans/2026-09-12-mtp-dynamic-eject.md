# Dynamic MTP Eject - Milestone 1 (rs rebuild + arena re-pin + context API) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a live QWEN35 phase-arena context shrink the pinned arena region and rebuild its recurrent-state cache at runtime, so the KV pool can reclaim the MTP reservation.

**Architecture:** Refactor `llama_memory_recurrent` so its buffers can be freed and reallocated with a different `n_rs_seq`, preserving plane-0 (committed) state. Wire the CUDA `phase_arena_reset_pinned` proc and add a `llama_context` method that frees the rs buffers, re-pins the arena top, rebuilds rs, and re-reserves. This is the riskiest layer; the server trigger (M2) and re-enable (M3) are separate plans.

**Tech Stack:** C++17, CUDA backend proc-address API, ggml-backend buffers.

**Spec:** `docs/superpowers/specs/2026-09-12-mtp-dynamic-eject-design.md`

## Global Constraints

- ASCII only in code and comments. No em-dashes or unicode arrows.
- Follow surrounding llama.cpp style; comments only where non-obvious.
- Build: `cmake --build build --config Release -j 12` from the repo root.
- Do NOT commit `benchmarks/benchmark_kv_stream.py` (unrelated).
- Every commit requires explicit user approval first (AGENTS.md). Use `Assisted-by: opencode`, never `Co-authored-by`.
- The uncommitted 8-file rs-pinning change is a prerequisite of this plan and stays in the working tree.
- Do not add files under `tests/` (needs maintainer approval). The integration program lives in `/tmp/opencode`.

---

### Task 1: Public API stub + failing integration test (RED)

Declare the toggle API, stub it, and write the integration program that must fail before the real implementation.

**Files:**
- Modify: `include/llama.h` (near the `llama_kv_stream_pinned_buft` declaration, ~line 606)
- Modify: `src/llama-context.cpp` (free function near `llama_kv_stream_pinned_buft`, ~line 4904)
- Create: `/tmp/opencode/arena_toggle_test.cpp`

**Interfaces:**
- Consumes: existing `llama_context_params.spec_mtp`, `.n_rs_seq`, `.mtp_weights_bytes`, `.kv_stream_arena_mib`, `.type_k`, `.type_v`, `.offload_kqv`.
- Produces: `LLAMA_API bool llama_kv_stream_mtp_set(struct llama_context * ctx, bool mtp_active);`

- [ ] **Step 1: Declare the API and add a stub**

In `include/llama.h`, after the `llama_kv_stream_pinned_buft` declaration:

```c
    // Reconfigure the phase arena between the MTP-reserved and MTP-free layouts.
    // Setting false requires the MTP context and draft model to be destroyed first.
    LLAMA_API bool llama_kv_stream_mtp_set(struct llama_context * ctx, bool mtp_active);
```

In `src/llama-context.cpp`, next to `llama_kv_stream_pinned_buft`:

```cpp
bool llama_kv_stream_mtp_set(llama_context * ctx, bool mtp_active) {
    return ctx->kv_stream_mtp_set(mtp_active);
}
```

Add the member declaration in `src/llama-context.h` (public section, next to `get_kv_stream_pinned_buft`):

```cpp
    bool kv_stream_mtp_set(bool mtp_active);
```

In `src/llama-context.cpp`, add a stub body temporarily:

```cpp
bool llama_context::kv_stream_mtp_set(bool mtp_active) {
    // replaced in Task 5
    (void) mtp_active;
    return false;
}
```

- [ ] **Step 2: Write the integration test**

Create `/tmp/opencode/arena_toggle_test.cpp`:

```cpp
#include "llama.h"

#include <cstdio>
#include <vector>

static bool decode_ids(llama_context * ctx, const llama_model * model,
                       const std::vector<llama_token> & toks, bool logits_last) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (logits_last && i + 1 == toks.size()) ? 1 : 0;
    }
    batch.n_tokens = (int32_t) toks.size();
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret == 0;
}

int main() {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(
        "/home/troed/llm-models/Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf", mparams);
    if (model == nullptr) { printf("TEST: model load failed\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 4096;
    cparams.n_batch          = 256;
    cparams.n_ubatch         = 256;
    cparams.n_seq_max        = 1;
    cparams.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k           = GGML_TYPE_Q8_0;
    cparams.type_v           = GGML_TYPE_Q4_0;
    cparams.offload_kqv      = true;
    cparams.kv_stream_arena_mib = 3072;
    cparams.spec_mtp         = true;
    cparams.n_rs_seq         = 3;
    cparams.mtp_weights_bytes = 605ULL * 1024 * 1024;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) { printf("TEST: context init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(8);
    const int n = llama_tokenize(vocab, "The capital of France is", 23, toks.data(), (int) toks.size(), true, false);
    toks.resize(n);

    if (!decode_ids(ctx, model, toks, true)) { printf("TEST: prefill decode failed\n"); return 1; }

    if (!llama_kv_stream_mtp_set(ctx, false)) { printf("TEST: FAIL eject returned false\n"); return 1; }
    if (!decode_ids(ctx, model, { 1234 }, true)) { printf("TEST: FAIL decode after eject\n"); return 1; }

    if (!llama_kv_stream_mtp_set(ctx, true)) { printf("TEST: FAIL enable returned false\n"); return 1; }
    if (!decode_ids(ctx, model, { 2345 }, true)) { printf("TEST: FAIL decode after enable\n"); return 1; }

    printf("TEST: PASS\n");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```

- [ ] **Step 3: Build and run to verify it fails**

```bash
cmake --build build --config Release -j 12
g++ -std=c++17 -O2 -I include -I ggml/include /tmp/opencode/arena_toggle_test.cpp \
    -L build/bin -lllama -lggml -lggml-base -Wl,-rpath,'$ORIGIN' -o build/bin/arena_toggle_test
cd build/bin && ./arena_toggle_test 2>&1 | tail -5
```

Expected: `TEST: FAIL eject returned false`.

- [ ] **Step 4: Commit**

Request user approval, then:

```bash
git add include/llama.h src/llama-context.h src/llama-context.cpp
git commit -m "kv-stream : add phase arena mtp toggle stub

Assisted-by: opencode"
```

---

### Task 2: Refactor recurrent allocation so buffers can be rebuilt

Extract allocation from the constructor into a reusable method and remember the allocation parameters.

**Files:**
- Modify: `src/llama-memory-recurrent.h:118-137` (private members/methods)
- Modify: `src/llama-memory-recurrent.cpp:20-145` (constructor)

**Interfaces:**
- Produces: private `void alloc_buffers();`; members `type_r`, `type_s`, `offload`, `filter`, `secondary_buft`, `layer_buft` (per model layer).

- [ ] **Step 1: Add members and the helper declaration**

In `src/llama-memory-recurrent.h`, in the private section (after `n_seq_max`):

```cpp
    // kept so the buffers can be rebuilt at runtime
    ggml_type type_r = GGML_TYPE_F32;
    ggml_type type_s = GGML_TYPE_F32;
    bool offload = true;
    layer_filter_cb filter;
    ggml_backend_buffer_type_t secondary_buft = nullptr;
    std::vector<ggml_backend_buffer_type_t> layer_buft; // one entry per model layer, null when skipped

    void alloc_buffers();
```

Add `#include <functional>` to the header includes.

- [ ] **Step 2: Refactor the constructor**

In `src/llama-memory-recurrent.cpp`, replace the body from the `ctx_map` declaration (line 48) through the buffer-allocation loop (line 132) so the constructor only records the parameters and calls `alloc_buffers()`. The constructor becomes:

```cpp
llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter,
    ggml_backend_buffer_type_t secondary_buft) : hparams(model.hparams), n_seq_max(n_seq_max) {
    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);

    cells.clear();
    cells.resize(mem_size);

    this->type_r = type_r;
    this->type_s = type_s;
    this->offload = offload;
    this->filter = filter;
    this->secondary_buft = secondary_buft;

    const int32_t n_layer = hparams.n_layer();
    layer_buft.assign(n_layer, nullptr);
    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            continue;
        }
        if (offload) {
            layer_buft[i] = ggml_backend_dev_buffer_type(model.dev_layer(i));
        } else {
            layer_buft[i] = ggml_backend_cpu_buffer_type();
        }
        if (secondary_buft != nullptr) {
            layer_buft[i] = secondary_buft;
        }
    }

    alloc_buffers();
}
```

- [ ] **Step 3: Write `alloc_buffers()`**

Add above the constructor (or below the class block) in `src/llama-memory-recurrent.cpp`:

```cpp
void llama_memory_recurrent::alloc_buffers() {
    const int32_t n_layer = hparams.n_layer();

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((hparams.ple_conv_state() > 0 ? 3u : 2u)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }
            ctx_map.emplace(buft, ctx);
            return ctx;
        }
        return it->second.get();
    };

    r_l.assign(n_layer, nullptr);
    s_l.assign(n_layer, nullptr);
    p_l.assign(n_layer, nullptr);

    for (int i = 0; i < n_layer; i++) {
        ggml_backend_buffer_type_t buft = layer_buft[i];
        if (buft == nullptr) {
            continue;
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows = size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        r_l[i] = r;
        s_l[i] = s;

        if (hparams.ple_conv_state() > 0 && hparams.is_ple(i)) {
            ggml_tensor * p = ggml_new_tensor_2d(ctx, type_r, hparams.ple_conv_state(), n_rows);
            ggml_format_name(p, "cache_ple_r_l%d", i);
            p_l[i] = p;
        }
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();
        const size_t memory_size_p = size_p_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB, P (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s + memory_size_p) / (1024.0f * 1024.0f), size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f),
                ggml_type_name(type_r), (float)memory_size_p / (1024.0f * 1024.0f));
    }
}
```

- [ ] **Step 4: Build**

Run: `cmake --build build --config Release -j 12`
Expected: clean build.

- [ ] **Step 5: Commit**

Request approval, then commit `src/llama-memory-recurrent.h` and `.cpp` with message `llama : factor recurrent cache buffer allocation`.

---

### Task 3: Add `llama_memory_recurrent::rebuild`

**Files:**
- Modify: `src/llama-memory-recurrent.h` (public method)
- Modify: `src/llama-memory-recurrent.cpp` (implementation)

**Interfaces:**
- Consumes: `layer_buft`, `secondary_buft`, `alloc_buffers()` from Task 2.
- Produces: `bool llama_memory_recurrent::rebuild(uint32_t n_rs_seq, ggml_backend_buffer_type_t secondary_buft, const std::function<void()> & repin);`

- [ ] **Step 1: Declare the method**

In `src/llama-memory-recurrent.h`, public section (after `set_rs_idx`):

```cpp
    // Reallocate the rollback planes, preserving the committed state (plane 0).
    // repin() runs between releasing the old buffers and allocating the new ones.
    bool rebuild(uint32_t n_rs_seq, ggml_backend_buffer_type_t secondary_buft,
                 const std::function<void()> & repin);
```

- [ ] **Step 2: Implement it**

In `src/llama-memory-recurrent.cpp` (after `alloc_buffers`):

```cpp
bool llama_memory_recurrent::rebuild(uint32_t n_rs_seq, ggml_backend_buffer_type_t secondary_buft,
                                     const std::function<void()> & repin) {
    for (uint32_t v : rs_idx) {
        if (v != 0) {
            LLAMA_LOG_ERROR("%s: cannot rebuild with a pending rollback\n", __func__);
            return false;
        }
    }

    const int32_t n_layer = hparams.n_layer();
    const size_t row_r = hparams.n_embd_r()*ggml_type_size(type_r);
    const size_t row_s = hparams.n_embd_s()*ggml_type_size(type_s);
    const size_t row_p = hparams.ple_conv_state()*ggml_type_size(type_r);

    std::vector<std::vector<uint8_t>> stage_r(n_layer), stage_s(n_layer), stage_p(n_layer);
    for (int i = 0; i < n_layer; i++) {
        if (r_l[i] == nullptr) {
            continue;
        }
        stage_r[i].resize(row_r*size);
        ggml_backend_tensor_get(r_l[i], stage_r[i].data(), 0, stage_r[i].size());
        stage_s[i].resize(row_s*size);
        ggml_backend_tensor_get(s_l[i], stage_s[i].data(), 0, stage_s[i].size());
        if (p_l[i] != nullptr) {
            stage_p[i].resize(row_p*size);
            ggml_backend_tensor_get(p_l[i], stage_p[i].data(), 0, stage_p[i].size());
        }
    }

    this->secondary_buft = secondary_buft;
    ctxs_bufs.clear();
    if (repin) {
        repin();
    }

    this->n_rs_seq = n_rs_seq;
    alloc_buffers();

    for (int i = 0; i < n_layer; i++) {
        if (r_l[i] == nullptr) {
            continue;
        }
        ggml_backend_tensor_set(r_l[i], stage_r[i].data(), 0, stage_r[i].size());
        ggml_backend_tensor_set(s_l[i], stage_s[i].data(), 0, stage_s[i].size());
        if (p_l[i] != nullptr) {
            ggml_backend_tensor_set(p_l[i], stage_p[i].data(), 0, stage_p[i].size());
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
    rs_z = -1;
    return true;
}
```

- [ ] **Step 3: Build**

Run: `cmake --build build --config Release -j 12`
Expected: clean build.

- [ ] **Step 4: Commit**

Request approval, then commit both recurrent files with message `llama : allow rebuild of the recurrent cache planes`.

---

### Task 4: Arena re-pin plumbing + pinned-bytes helper

**Files:**
- Modify: `src/llama-context.h` (owner struct, new members)
- Modify: `src/llama-context.cpp` (proc fetch, helper, ctor use)

**Interfaces:**
- Produces: owner `reset_pinned_fn`; owner `arena_total_bytes`; members `spec_mtp_configured`, `spec_n_rs_seq`, `spec_n_max_spec_draft`; method `uint64_t kv_stream_pinned_bytes_for(bool mtp_active) const`.

- [ ] **Step 1: Add owner fields**

In `src/llama-context.h`, inside `kv_stream_phase_arena_owner`, after `set_pinned_fn`:

```cpp
        void (*reset_pinned_fn)(void *) = nullptr;
```

after `arena_bytes`:

```cpp
        size_t arena_total_bytes = 0;
```

In the private members of `llama_context` (near the owner), add:

```cpp
    // captured at construction so the MTP reservation can be restored
    bool     spec_mtp_configured = false;
    uint32_t spec_n_rs_seq = 0;
    uint32_t spec_n_max_spec_draft = 0;

    uint64_t kv_stream_pinned_bytes_for(bool mtp_active) const;
```

- [ ] **Step 2: Add the pinned-bytes helper and use it in the constructor**

In `src/llama-context.cpp`, add the member definition before the constructor (or near it):

```cpp
uint64_t llama_context::kv_stream_pinned_bytes_for(bool mtp_active) const {
    if (!spec_mtp_configured) {
        return 0;
    }

    uint64_t pinned = 0;

    if (mtp_active && hparams.n_layer_nextn > 0) {
        const uint32_t il = hparams.n_layer();
        const uint64_t per_token =
            uint64_t(hparams.n_embd_k_gqa(il) + hparams.n_embd_v_gqa(il))*ggml_type_size(GGML_TYPE_F16);
        pinned += (per_token*cparams.n_ctx_seq + 127ULL) & ~127ULL;
    }
    if (mtp_active) {
        pinned += (cparams.mtp_weights_bytes + 127ULL) & ~127ULL;
    }

    const uint32_t n_rs = mtp_active ? spec_n_rs_seq : 0;
    const uint64_t n_rows = uint64_t(std::max(1u, cparams.n_seq_max))*(1ULL + n_rs);
    uint64_t rs_bytes = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.is_recr(il)) {
            continue;
        }
        uint64_t elems = uint64_t(hparams.n_embd_r()) + hparams.n_embd_s();
        if (hparams.ple_conv_state() > 0 && hparams.is_ple(il)) {
            elems += hparams.ple_conv_state();
        }
        rs_bytes += n_rows*elems*ggml_type_size(GGML_TYPE_F32);
    }
    pinned += (rs_bytes + 127ULL) & ~127ULL;

    return pinned;
}
```

In the constructor, before the arena block, capture:

```cpp
        spec_mtp_configured  = cparams.spec_mtp;
        spec_n_rs_seq        = cparams.n_rs_seq;
        spec_n_max_spec_draft = cparams.n_max_spec_draft;
```

Then replace the inline pinned computation at lines 484-516 with:

```cpp
            const uint64_t kv_stream_pinned_bytes =
                kv_stream_pinned_bytes_for(spec_mtp_configured);
```

- [ ] **Step 3: Fetch the reset proc and record the total**

In `src/llama-context.cpp`, add the typedef near the other arena typedefs:

```cpp
            using arena_reset_pinned_fn_t = void (*)(void *);
```

Add the proc fetch after `arena_set_pinned_fn`:

```cpp
            auto arena_reset_pinned_fn = (arena_reset_pinned_fn_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_phase_arena_reset_pinned");
```

Add `arena_reset_pinned_fn == nullptr` to the null check, store it:

```cpp
            kv_stream_phase_arena.reset_pinned_fn = arena_reset_pinned_fn;
            kv_stream_phase_arena.arena_total_bytes = kv_stream_arena_bytes;
```

- [ ] **Step 4: Create the KV runtime with the full arena cap**

Change `params_mem` line ~605 from `kv_stream_phase_arena.arena_bytes` to the full arena:

```cpp
            /*.kv_stream_maximum_pool_bytes =*/ kv_stream_phase_arena.arena_total_bytes,
```

Also change the `rs_secondary_buft` gate at line ~613 to use the captured flag:

```cpp
            /*.rs_secondary_buft    =*/ cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
                                        spec_mtp_configured
                                            ? kv_stream_phase_arena.pinned_buffer_type
                                            : nullptr,
```

- [ ] **Step 5: Build**

Run: `cmake --build build --config Release -j 12`
Expected: clean build. If the ring-capacity derivation depends on `maximum_pool_bytes`, run the Task 6 test and check the decode pool still matches the measured ~453 MiB with MTP; if not, revert this sub-step and instead add a runtime max setter in M3.

- [ ] **Step 6: Commit**

Request approval, then commit with message `kv-stream : wire the arena re-pin proc and share pinned sizing`.

---

### Task 5: Implement `llama_context::kv_stream_mtp_set`

**Files:**
- Modify: `src/llama-context.cpp` (replace the Task 1 stub)

**Interfaces:**
- Consumes: `kv_stream_pinned_bytes_for`, `owner.reset_pinned_fn`, `owner.arena_total_bytes`, `llama_memory_recurrent::rebuild`, `sched_reserve()`.
- Produces: working `bool llama_context::kv_stream_mtp_set(bool mtp_active)`.

- [ ] **Step 1: Replace the stub**

```cpp
bool llama_context::kv_stream_mtp_set(bool mtp_active) {
    if (!kv_stream_phase_arena.configured) {
        return false;
    }
    if (cparams.spec_mtp == mtp_active) {
        return true;
    }

    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (hybrid == nullptr || hybrid->get_mem_recr() == nullptr) {
        return false;
    }
    llama_memory_recurrent * mem_recr = hybrid->get_mem_recr();

    const uint64_t new_pinned = kv_stream_pinned_bytes_for(mtp_active);

    synchronize();
    kv_stream_phase_arena.graph_reset_fn(backend_ptrs[kv_stream_phase_arena.backend_index]);
    gf_res_prev->reset();
    gf_res_reserve->reset();
    sched.reset();

    const bool ok = mem_recr->rebuild(
        mtp_active ? spec_n_rs_seq : 0,
        kv_stream_phase_arena.pinned_buffer_type,
        [&]() {
            kv_stream_phase_arena.reset_pinned_fn(kv_stream_phase_arena.arena);
            if (new_pinned != 0) {
                kv_stream_phase_arena.set_pinned_fn(
                    kv_stream_phase_arena.arena,
                    kv_stream_phase_arena.arena_total_bytes - new_pinned,
                    new_pinned);
            }
            kv_stream_phase_arena.pinned_bytes = new_pinned;
            kv_stream_phase_arena.arena_bytes =
                kv_stream_phase_arena.arena_total_bytes - new_pinned;
        });
    if (!ok) {
        return false;
    }

    cparams.spec_mtp         = mtp_active;
    cparams.n_rs_seq         = mtp_active ? spec_n_rs_seq : 0;
    cparams.n_max_spec_draft = mtp_active ? spec_n_max_spec_draft : 0;

    sched_need_reserve = true;
    sched_reserve();

    LLAMA_LOG_INFO("%s: MTP %s, pinned = %.2f MiB, arena = %.2f MiB\n", __func__,
            mtp_active ? "enabled" : "ejected",
            new_pinned/1024.0/1024.0, kv_stream_phase_arena.arena_bytes/1024.0/1024.0);
    return true;
}
```

Note: `get_mem_recr()` returns `llama_memory_recurrent *` directly (`src/llama-memory-hybrid.h:88`), so no cast is needed.

- [ ] **Step 2: Build**

Run: `cmake --build build --config Release -j 12`
Expected: clean build.

- [ ] **Step 3: Run the integration test (GREEN)**

```bash
cd build/bin && ./arena_toggle_test 2>&1 | tail -20
```

Expected: `TEST: PASS`, with log lines showing the RS buffer size drop from `598.50 MiB` to `149.62 MiB` on eject and back on enable, and the arena line changing from `~1164 MiB` to `~2922 MiB` and back.

- [ ] **Step 4: Commit**

Request approval, then commit `src/llama-context.cpp` with message `kv-stream : add runtime MTP arena toggle`.

---

### Task 6: Optional arena resize validation

**Files:**
- Test only: `/tmp/opencode/arena_toggle_test.cpp` (temporary modification)

- [ ] **Step 1: Confirm the pool actually grows**

Add a print of the decode-phase KV bytes from `llama_get_memory_breakdown` after each toggle, or read the `activated decode phase: KV ... MiB` log lines. Verify: MTP active -> ~453 MiB decode KV; MTP ejected -> ~2216 MiB. If the pool does not grow, the KV runtime max cap from Task 4 Step 4 is the cause; fix per the spec (max-pool setter).

- [ ] **Step 2: Confirm generation is still coherent**

Decode several tokens after each toggle and print `llama_get_logits` argmax; the text must stay coherent (no NaN/garbage).

---

## Self-Review

- Spec coverage: rs rebuild (Tasks 2-3), re-pin + full-arena cap (Task 4), context API (Task 5); the streaming signal, server trigger, and re-enable are M2/M3 plans, called out in the spec milestones.
- No placeholders; all code steps contain full code.
- Type consistency: `kv_stream_pinned_bytes_for`, `rebuild(...)`, `llama_kv_stream_mtp_set`, `reset_pinned_fn`, `arena_total_bytes` used with the same signatures throughout.
