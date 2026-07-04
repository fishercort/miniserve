# miniserve — project overview

miniserve is a continuous-batching LLM inference engine with a paged KV cache,
built from scratch in pure PyTorch for a single GPU, to understand and
instrument the internals. It doubles as the measurement instrument for the
phases that follow.

## The three artifacts

This repo is one part of a three-artifact project on KV cache management for
agentic inference:

1. **miniserve** (this repo) — the continuous-batching engine with a paged KV
   cache. Built from scratch to understand and instrument the internals; the
   measurement instrument for everything after.
2. **The policy engine** — published eviction/placement policies reimplemented
   as swappable strategies, plus a cost-model-driven joint policy as the
   contender, plus the lifecycle hint interface wired through the engine. Its
   design lives in the benchmark repo's policy layer; see
   `agentic-kv-bench/docs/policy-interface.md`.
3. **agentic-kv-bench** — the adoptable benchmark: real Claude Code traces, a
   synthetic sweep generator, a Belady-style oracle, and a small policy
   interface, designed for researchers to pull and use in their own work.
   Sibling repo. Dependency direction is bench → engine only.

## Explicit non-goal: learned eviction

A well-motivated heuristic measured against an oracle beats a half-trained
predictor, and keeps the scope systems, not ML. The prediction-based paper
(2605.06472) is a baseline to reimplement in its non-learned form, not a
direction to extend.

## Phase map

- **Phase 0 — Mechanical foundations** (`phase0-mechanics.md`): load the metal
  facts before writing engine code.
- **Phase 1 — Batching engine** (`phase1-engine.md`): the continuous-batching
  server with a paged KV cache. A complete, usable engine on its own.
- **Phase 2 — Cost model** (`phase2-cost-model.md`): a calibrated
  migrate-vs-recompute cost model measured on this engine.
- **Phase 3 — Policy engine, hint interface, benchmark**: specced in
  `agentic-kv-bench/docs/`.
- **Phase 4 — Validation on a production engine**: port the winning policy to
  vLLM via Dynamo's KV-event interface (KVBM), and propose the lifecycle-hint
  contract upstream. Specced in `agentic-kv-bench/docs/validation.md`.

Phases are ordered so each ends at a usable milestone; if scope compresses, the
Phase 4 vLLM port goes first — it confirms the findings rather than extending
them.

## Hardware

The scratch engine targets a single consumer-class GPU with a 1B–8B model —
model size is not the point, the engineering is. Phases 2 and 4 use an
A100/H100-class instance for measured numbers on datacenter hardware.
