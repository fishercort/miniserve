# Phase 0 — Mechanical foundations

Goal: load the metal facts before writing engine code.

- Trace one forward pass end to end. The goal is a mechanical, not paper-level,
  account of prefill (compute-bound) vs decode (memory-bandwidth-bound).
- Compute KV cache size by hand for the target model:
  `num_layers x 2 x num_kv_heads x head_dim x dtype_bytes x seq_len`. Know the
  real bytes per token and per block.
- Read PagedAttention and RadixAttention; skim Dynamo's KV router and KVBM docs
  to establish exactly what this project is NOT rebuilding.
- Read the concurrent/adjacent systems papers and extract each policy's exact
  scoring rule into notes. They are the related work and the baseline specs for
  Phase 3b:
  - **Continuum (2511.02230, Tensormesh + Tsinghua)** — names the end-of-turn
    eviction failure mode and proposes KV TTL. A baseline (see Phase 3b), and
    evidence the LMCache community already owns this problem.
  - **2605.06472** — lifecycle / retired-cache eviction via workflow
    termination messages.
  - **2605.00528 (SAGA)** — Workflow-Aware LRU.
  - **2606.09916 (IntentKV)** — VERIFY which lane this occupies: the title says
    "pruning," which may mean token-level. If so, it drops from the baseline
    set to related-work-only.
- Internalize and document the two-lane distinction (token-level eviction vs
  block/request-level management). The authored paragraph lives in the
  benchmark spec's Scope section (`agentic-kv-bench/docs/benchmark-spec.md`) and
  is reused verbatim as the benchmark's scope statement — write it once, there.
  It is both the scoping guard and the gap statement.
- Pick a small open-weights model (1B–8B) that fits one GPU. Model size is not
  the point; the engineering is.
- Set up: one rented GPU for measurement, torch profiler and nsys.

**Deliverable:** `MECHANICS.md` in the repo.
