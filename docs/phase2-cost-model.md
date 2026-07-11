# Phase 2 — Cost model from first principles

Goal: ground every cost in measured hardware behavior — no magic constants.

Measured on the miniserve engine itself, cross-checked on real hardware:

- Measure prefill latency as a function of prompt length; derive the
  compute-cost curve. (The per-step prefill metrics logged in Phase 1 are the
  raw data — this phase is mostly analysis, not new instrumentation.)
- Compute KV block byte size for the target model; measure transfer time for
  GPU ↔ CPU ↔ disk. Get real bandwidth numbers per tier.
- Derive the migrate-vs-recompute crossover: at what prefix length does moving
  a cached block beat recomputing it, for each tier and interconnect?
- Measure activation peak vs batch composition: MECHANICS.md assumes a
  1.5–2 GiB overhead band, but activation memory scales with tokens in flight
  per step (a large mixed prefill+decode batch peaks above steady-state
  decode). Replace the assumption with a curve.
- Validate predicted vs measured; report the error.

**Deliverable:** a calibrated cost model with measured constants and plotted
crossover curves, plus a derivation writeup. It grounds the memory hierarchy,
prefill/decode behavior, and KV layout in numbers rather than assumptions, and
is consumed directly by the Phase 3 policies and oracle.
