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

## GPU-day verdict (2026-07-21, H100 80GB SXM5)

Ran clean end to end on a single H100 80GB SXM5 (a named target from the
master plan; no hardware deviation). All curves measured, config emitted
(results/gpu_day/cost_model.json), same-day committed. Numbers below cite that
artifact; none are retyped.

Spot-validation contract: FAILED (0 of 3), and the failure is the finding. The
pre-committed contract (25 percent at 2 of 3 points per tier) fired exactly as
designed and refused to let a degenerate crossover graduate into Phase 3.

Diagnosis, fully determined from the data: the v1 gather-attention forward has
a fixed overhead floor of about 24 ms per pass on this hardware. Prefill
latency is flat at ~23.7 ms from 16 to 512 tokens and only exceeds the floor
near 2048 tokens (45 ms) and 4096 (129 ms), where the quadratic attention tail
appears; decode is 23 ms/step for the same reason. This is the Phase 1
gather-based-attention perf-gap tradeoff, now quantified. Because recompute
costs at least ~24 ms at any length while migrating a prefix's KV costs
microseconds, migration wins at 1 token, the crossover band collapses to
[1, 1], and validation at a single token is pure timing noise (migrate
predicted 0.056 ms vs measured 0.023 ms). Curve A itself fits well (quadratic,
2 percent relative error, gate passed); the crossover derived from it is
degenerate, not the fit.

What this means: the migrate-vs-recompute crossover is instrument-bound, not
hardware-bound, and does not graduate. A production kernel would have a
~1-3 ms forward floor, moving the crossover to a real, non-degenerate prefix
length. The calibrated crossover is therefore deferred to Phase 4 (vLLM real
kernel), as the plan already contemplated. Phase 3 consumes the cost model
structurally: recompute cost is swept as a parameter rather than pinned to the
v1 engine's launch-bound constant, and the joint policy's evict/offload/
recompute decision is evaluated across that sweep. This also sharpens the
project's own hypothesis: the recompute-vs-migrate tradeoff only becomes
interesting when recompute is compute-bound, so the economic policy's decisive
regime is large prefixes and efficient kernels, not the launch-bound floor.

What does graduate (measured, hardware-real, useful now):
- Activation-peak curve: 5.8 to 7.2 GiB across the batch/context/composition
  grid, peaking on a large prefill joining a full decode batch. This is the
  overhead band MECHANICS.md assumed, now measured.
- Transfer bandwidths (contiguous, non-pinned; an upper bound the scattered
  and pinned corrections adjust): cuda->cpu 16.0 GB/s, cpu->cuda 15.6 GB/s,
  disk write+fsync 1.1 GB/s (ext4), warm page-cached read 11.0 GB/s.
- Chunked-overhead coefficient: 0.018 ms per chunk, projected 1.85x slowdown
  at the model's 56 KV regions (the scattered-transfer correction, now a
  number).

## Measurement plan

Every protocol decision below is recorded before its script runs. The plan
governs; scripts implement.

### 1. Curve A: prefill latency vs prompt length (the recompute cost curve)

- Probes drive the engine and read the per-step metrics log (the
  instrumentation built in Phase 1); no new timing code around the forward.
- One probe = one prompt of the target length, max_tokens=1: exactly one
  prefill StepRecord, no decode contamination.
- Prompt ids are random and valid for the loaded model; vocab derives from
  the model's embedding table, never a literal.
- One cache, preallocated at max probe length, reused for every probe.
  Warmup is three max-length probes, discarded: the FIFO free list hands
  consecutive probes different physical blocks, so multiple cycles are needed
  to fault every page before recording. Reuse matches how the calibrate CLI
  must behave on user hardware.
- Probes run round-robin across lengths (rep 1 of each, then rep 2) so
  thermal and clock drift spread across cells instead of biasing one.
- Grid: 16 to 4096, log2-spaced. Kept value per length: median of repeats
  (timing noise is one-sided; interference only adds).
- Fits: linear and quadratic, both reported with mean relative error.
  Preferred form is quadratic only when it beats linear by 20 percent or
  more on residuals. Acceptance gate, numeric, two failure modes: (a) the
  preferred fit's mean relative error must be at or below 5 percent across
  the grid, and (b) residuals must be structureless: a run of four or more
  consecutive same-sign residuals (each beyond 1 percent) marks the fit
  structured and unacceptable, because a saturation knee can hide under a
  passing mean error. Either failure means the grid is densified and refit
  (or the form revisited) before any downstream use.

### 2. Curve B: decode step latency vs (batch size, context length)

- Grid: batch {1, 2, 4, 8} x context {256, 1024, 4096}.
- Independent-variable control (the cap-vs-realized trap again): with B
  sequences decoding together, contexts grow during the probe. Convention:
  prompts are sized to target minus k (k=8); the measured steps are exactly
  steps 1..k after all B sequences are in DECODE; the report carries the
  realized mean context per measured step alongside the target label. Spec
  targets, harness reports realized.
- Kept value per cell: median over measured steps and repeats.

### 3. Curve C: transfer time per tier

- GPU to CPU and back: batches of 1 to 64 blocks, timed around explicit
  device syncs on both sides; fit t = latency + bytes/bandwidth so
  per-transfer latency separates from bandwidth.
- CPU to disk: block-sized buffers, fsync included on the write path.
  Page-cache honesty clause: a timed read of a just-written file measures
  RAM, not disk. The v1 disk tier therefore measures the OS-cached path and
  is labeled `disk_warm_page_cached` in the emitted config; cold-disk is
  reported separately only if measurable without root or host disruption.
  This clause exists because a disk number that is secretly a RAM number
  mis-prices the migrate-vs-recompute boundary exactly where recompute most
  plausibly wins.
- Scattered-transfer caveat, structural: these tiers measure CONTIGUOUS
  copies, but a prefix's KV is num_layers x 2 separate tensors (56 regions
  for the target model) sliced at block granularity. Many small copies pay
  many launch-and-sync overheads, so measured bandwidth is an upper bound on
  achievable migration bandwidth and the crossover's migrate-favorable edge
  is optimistic. Production engines close the gap with staging buffers
  (gather scattered KV into one region, one fat copy); small-transfer
  overhead, not raw bandwidth, is what historically pushed swap-vs-recompute
  toward recompute for short sequences. The emitted config carries
  transfer_measurement: "contiguous", and the GPU-day batch includes a
  chunked-copy probe (same bytes, N discrete copies, N swept) that measures
  the overhead directly and turns this caveat into a coefficient.

### 4. Sync policy

On cuda and mps, the model's forward is wrapped so an explicit device sync
lands inside the timed region. Timing never relies on the sampling .item()
sync as a coincidence. Verification: at three lengths, shim timings are
compared against torch.cuda.Event timings; agreement within 5 percent is
required before the run counts.

### 5. Crossover derivation and validation

From curves A and C, derive L* per tier: the prefix length where migrating
cached blocks costs the same as recomputing them. The point alone can be
numerically meaningless (near-parallel cost lines put it anywhere), so the
config also emits the crossover BAND: the range where the two costs are
within 25 percent of each other. Policies treat the band as the soft
boundary; a wide band is itself a finding. Spot-validation targets the band,
not just the point: per tier, measure both paths at the band's lower edge,
the crossover, and the band's upper edge. Contract, pre-committed: the
model's predictions must land within 25 percent of measurement at 2 of 3
points per tier, or the cost model is revised before Phase 3 inherits it.
Transfer tiers are DIRECTIONAL (gpu_to_cpu and cpu_to_gpu are separate
entries; disk read and write likewise), and one-way vs round-trip pricing is
named in the consumption module, because the offload decision pays down plus
up against recompute's zero.

### 6. Hardware rule and emitted config

- CPU and MPS runs validate pipeline mechanics and fit plumbing only. Two
  legitimate uses: end-to-end script validation, and testing the fit and
  schema code. CPU curves have the wrong shape for the target hardware (no
  GPU knee, different memory hierarchy): no CPU number graduates to the
  cost-model config or this doc.
- GPU target, pinned: A100-80GB. Form factor matters at exactly the thing
  this phase measures (SXM and PCIe differ in memory bandwidth), so the
  emitted config records gpu name, form factor, driver, and VRAM, and every
  reported number states them.
- Emitted config (versioned schema): schema_version, model, gpu {name,
  form_factor, driver, vram_gb}, dtype, curves {prefill fit + preferred +
  rel_err, decode table, transfer per tier {latency, bandwidth}}, provenance
  {timestamp, platform, commit}. This file is what the simulator consumes
  and what the benchmark's calibrate CLI later emits on user hardware.

### 7. Ordering note and budget

The prefill probe script preceded this plan; its protocol block was written
first and generalized here. Recorded once, not repeated: all subsequent
scripts are written against this plan. Budget: one GPU day, batched: curves
A through C, the chunked-copy overhead probe (scattered-transfer
coefficient), the activation-peak-vs-batch-composition measurement, and the
crossover spot-validation in a single rental session, with the config, raw
JSONLs, and session log committed the same day. Doc numbers cite the config
artifact; they are never retyped.
