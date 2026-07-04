# miniserve
Continuous-batching inference engine with paged KV cache, from scratch, single GPU, PyTorch.

Architecture is specced in docs/ — read before making design decisions:
- docs/overview.md         — project overview, the three artifacts, non-goal, phase map
- docs/phase0-mechanics.md — Phase 0 mechanical foundations
- docs/phase1-engine.md    — Phase 1 batching engine (module layout, scheduler loop, API)
- docs/phase2-cost-model.md — Phase 2 calibrated cost model

Build/test (uv-based): `uv run pytest`, `uv run ruff check`.
Sibling repo: agentic-kv-bench (policy benchmark harness; depends on this engine, never the reverse).
PRIVATE.md (gitignored) holds non-technical project strategy — not needed for engineering work.
