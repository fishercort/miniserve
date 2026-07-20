# Phase 1 — Batching engine from scratch

Goal: understand the internals by building them. A minimal continuous-batching
inference server with a paged KV cache, single GPU, pure PyTorch, built so it
becomes the instrument for Phases 2–4. Where a design choice exists, this picks
the tractable-but-correct option and flags the harder version as a stretch.

## Module layout

```
miniserve/
  model.py       # load HF weights, forward pass with paged KV
  kv_cache.py    # PagedKVCache: physical tensors, free list, block tables, Evictor seam
  scheduler.py   # waiting/running queues, admission, the step() loop
  engine.py      # ties model + cache + scheduler, runs the loop
  server.py      # HTTP API, token streaming
  metrics.py     # per-request and per-step collection
  bench/         # load generator, static-vs-continuous comparison
```

## Data structure 1: the paged KV cache

This is the structure Phase 3 later instruments, so build the seams now even
though v1 does not use them.

```python
# Physical storage: preallocate to fill the GPU memory budget.
# One pair (K, V) per layer. Shape per tensor:
#   [num_blocks, block_size, num_kv_heads, head_dim]
# block_size = tokens per block, e.g. 16.

class PagedKVCache:
    k_cache: list[Tensor]          # len = num_layers
    v_cache: list[Tensor]
    block_size: int
    free_blocks: deque[int]        # stack/queue of free physical block ids
    block_tables: dict[ReqId, list[int]]   # logical token -> physical block
    meta: dict[BlockId, BlockMeta] # reserved for Phase 3, unused in v1

    def free_count(self) -> int
    def allocate(self, req_id, num_blocks) -> bool   # pops from free_blocks
    def append_block(self, req_id) -> bool           # one more block for a seq
    def free(self, req_id)                           # returns blocks to free list

# Reserved now, populated in Phase 3. Leave the fields present and unused.
class BlockMeta:
    last_access: float = 0.0
    access_count: int = 0
    session_id: str | None = None
    lifecycle_class: str = "unknown"   # durable | ephemeral | unknown
    recompute_cost: float = 0.0        # filled from the Phase 2 cost model
```

The free list plus per-sequence block table is the whole PagedAttention idea.
Logical token position `i` for a sequence maps to physical block
`block_table[i // block_size]`, offset `i % block_size`.

## Data structure 2: the sequence

Carries everything the scheduler and the metrics need.

```python
class Sequence:
    req_id: str
    prompt_ids: list[int]
    output_ids: list[int]
    status: Status              # WAITING | PREFILL | DECODE | FINISHED
    sampling: SamplingParams
    arrival_time: float
    first_token_time: float | None   # set when the first output token is produced
    completion_time: float | None

    def total_len(self) -> int       # len(prompt) + len(output)
    def needs_new_block(self, block_size) -> bool
    def is_finished(self, eos_id) -> bool
```

TTFT = `first_token_time - arrival_time`. Record it the moment prefill produces
its first token.

## The scheduler loop (the heart)

Iteration-level scheduling is the thing that makes this "continuous" batching:
the batch composition changes every step, finished sequences leave and waiting
ones join at iteration boundaries, instead of waiting for a whole static batch
to drain.

```python
def step():
    # 1. Admission: pull from waiting while blocks are available
    while waiting and len(running) < max_batch:
        seq = waiting[0]
        need = ceil(seq.total_len() / block_size)   # resume: prompt + generated
        if kv.free_count() < ceil((seq.total_len() + 1) / block_size):
            break   # exact-form progress headroom; see Named failure modes
        waiting.popleft()
        kv.allocate(seq.req_id, need)
        seq.status = PREFILL
        running.append(seq)

    if not running:
        return

    # 2. Partition this iteration's work
    prefill = [s for s in running if s.status == PREFILL]
    decode  = [s for s in running if s.status == DECODE]

    # 3. Forward pass. Paged attention reads K,V via each seq's block table.
    #    v1: gather KV blocks into a contiguous buffer per sequence, then run
    #    standard attention. Simpler and pure-PyTorch. Note the perf gap vs a
    #    real paged kernel in the writeup. (Stretch: write a paged kernel.)
    logits = model.forward(prefill + decode, kv)

    # 4. Sample, append, grow KV
    for s in prefill + decode:
        tok = sample(logits[s.req_id], s.sampling)
        if s.status == PREFILL:
            s.first_token_time = now()   # <-- TTFT marker
            s.status = DECODE
        s.output_ids.append(tok)
        if s.needs_new_block(block_size):
            if kv.free_count() == 0:
                evictor.make_room(s)     # Phase 3 seam. v1: abort-and-requeue youngest.
            kv.append_block(s.req_id)

    # 5. Retire finished, return their blocks
    for s in list(running):
        if s.is_finished(eos_id):
            s.completion_time = now()
            kv.free(s.req_id)
            running.remove(s)
            emit_result(s)

    metrics.record_step(
        n_prefill=len(prefill), n_decode=len(decode),
        kv_used=kv.used_count(), step_latency=...,
    )
```

The engine runs `step()` in a tight loop on a dedicated engine thread.
`server.py` communicates with it only by message passing; see the concurrency
model decision in Named failure modes.

### v1 simplifications (scope control)

- Greedy or basic top-k/top-p sampling. Not the interesting part yet.
- Single GPU, no tensor parallelism.
- Gather-based attention rather than a custom paged kernel. Flag the tradeoff.
- No swapping/offload in v1. Preemption exists in exactly one form —
  abort-and-requeue-youngest when `make_room` is hit (see Named failure modes
  below). Richer victim selection is the Phase 3 policy engine's job.
- Prefill handling: simplest correct version processes the admitted prefill plus
  the running decode in one step as above. Chunked prefill (cap prefill tokens
  per step to protect decode latency) is the nicer version, list it as a stretch.

### Named failure modes (open design questions)

Two scheduler-semantics decisions are deliberately left open until
implementation; both are places a serving engineer will look first.

- **Full-occupancy livelock.** Blocking admission protects the waiting queue,
  but if every block is allocated and every running decode needs a new block,
  `make_room` has nothing trivially evictable and without preemption, the step
  loop could wedge. Candidate v1 answers: block and accept the wedge under
  adversarial load; abort-and-requeue the youngest running sequence;
  preempt-by-recompute (vLLM's answer).

  v1 policy:
  victim = two-tier youngest: youngest among sequences that have produced at
  least one token (finish-what-you-started fairness, plus cheapest recompute —
  and it preserves the admission-implies-progress invariant below). A first
  token sampled earlier in the same step counts, and the requester itself is
  not excluded — self-preemption is legal: the invariant is already satisfied,
  the resume path already exists, and a special-case wait path is extra code
  for a corner the capacity check already bounds. Fallback,
  when no running sequence has decoded yet (the all-fresh corner, where the
  invariant is unsatisfiable and someone must die), youngest overall — tagged
  with a distinct preemption reason code so the corner is visible in logs.
  (Traced after implementation: the fallback is unreachable through step()
  itself — the requester has always just sampled a token, so the first tier is
  never empty. It is kept as a defensive tier, unit-tested directly, for
  whatever calls make_room in Phase 3.);
  mechanism = abort-and-requeue to front of waiting, with a `preemptions_total`
  counter and the admission-time capacity check (reject any request whose worst
  case `ceil((len(prompt) + max_tokens) / block_size)` exceeds total blocks —
  that's what makes the mechanism provably terminate);
  framing = v1's hardcoded victim choice is the degenerate case of the Phase 3
  policy engine

  Measured (mechanism sweep, pressure ablation): under the same memory
  pressure, continuous batching preempts about three times as often as static
  (43-45 vs 12-18 events per run) and still wins both metrics (140 vs 88
  tok/s, p95 TTFT 9 s vs 18 s). The scheduler spends preemptions freely and
  profits, which is the empirical case that abort-and-requeue was the right
  v1 mechanism: cheap enough to use, not just correct enough to pass tests.

- **Admission headroom.** Admission reserves `ceil(len(prompt)/block_size)`
  blocks — exactly the prompt, zero decode headroom. A prompt that fills its
  last block needs a new block on the first generated token, so at high
  occupancy admission feeds straight into the `make_room` path. Candidate v1
  answers: admit at `need` (maximum utilization, earlier pressure) or `need + 1`
  (one-token headroom, slightly lower utilization).

  v1 policy: admission needs enough free blocks to cover the request's current
  content plus one token, i.e. `ceil((total_len + 1) / block_size)`; for a
  resumed request, content means prompt plus everything generated before
  preemption. That equals need + 1 blocks only when the content exactly fills
  its last block; otherwise it is just need. The extra token's room guarantees
  every admitted sequence completes at least one decode step before it can
  face preemption: admission implies progress. The exact form matters at the
  boundary: flat need + 1 strands a whole-cache-sized request at re-admission,
  demanding a block the cache does not contain and breaking the termination
  promise the capacity check made, with the stranded head then blocking the
  strict-FIFO queue behind it. Amended from the original need + 1 wording when
  implementation found that hole. (The all-fresh corner above remains the one
  bounded exception, with its own reason code.)

- **Head-of-line blocking.** When the head of the waiting queue does not fit
  but a smaller sequence behind it would, admission could hold the line or
  skip ahead.

  v1 policy: strict FIFO — the head blocks the queue. Skip-ahead is a fairness
  policy with a starvation risk for large requests; it deserves the Phase 3
  cost-aware treatment, and FIFO-strict is one sentence to defend.

- **Concurrency model.** The engine loop and the HTTP server need a boundary;
  the allocator is guard-then-mutate and documented single-threaded.

  v1 policy: the engine runs on a dedicated thread that exclusively owns the
  scheduler and KV cache; the HTTP server communicates only by message passing
  (a submission queue in, per-request token queues out, futures for capacity
  rejections). The allocator's single-threaded assumption holds because no
  other thread can reach the scheduler, not because callers promise to behave.
  What crosses the queue is token ids, never text: tokenization happens in
  the HTTP handler thread, so a long prompt cannot stall running decodes.
  Queues are unbounded in v1: a slow SSE client grows its token queue without
  limit; accepted and documented (bounding it is a resource-eviction policy,
  which is Phase 3's shape of problem). Cost accepted: capacity rejections
  resolve at the next step boundary rather than synchronously, and the
  CapacityError must cross the thread boundary as the same exception type and
  message. Tests drive step() directly and stay deterministic; thread
  lifecycle, including clean termination of an in-flight stream at shutdown,
  is quarantined to integration tests.

  Amendment (step 4 audit): /metrics and /health read counters directly from
  handler threads. This is a scoped exception, not a violation: the line
  exists to protect the allocator's mutation invariants, and read-only,
  GIL-atomic snapshots (single field reads and len() calls, values allowed to
  be torn across fields and stale) cannot touch those. Routing them through
  the engine would also make observability depend on the engine loop being
  healthy, which is backwards: a wedged engine is exactly when /metrics must
  stay readable. The rule that keeps the carve-out safe: no iteration over
  scheduler or cache collections from a handler thread, ever; a handler that
  needs more than a counter goes through the engine.

## API surface

```
POST /generate
  body:  { prompt: str, max_tokens: int, temperature?: float, top_p?: float }
  resp:  server-sent events streaming tokens, then a final usage summary
         { ttft_ms, total_ms, output_tokens, throughput_tok_s }

GET /metrics
  Prometheus-style or JSON: TTFT histogram, inter-token latency, throughput,
  num_running, num_waiting, kv_utilization_pct

GET /health
  { status, model, kv_blocks_total, kv_blocks_free }
```

Requests are async: they land in `waiting`, the loop picks them up, tokens
stream back as produced.

Sampling parameters come from the request and are executed exactly. The
checkpoint's `generation_config.json` is deliberately ignored — its defaults
are client-side suggestions, not model contract. Note that HF `generate()`
does apply them: a shipped `repetition_penalty` modifies logits even under
`do_sample=False`, which is why the golden tests neutralize it with a bare
`GenerationConfig`. Supporting `repetition_penalty` as a request parameter
would be a new API feature.

Two more v1 API decisions:
- Greedy only: requests with temperature != 0 or top_p != 1 are rejected with
  a 400. Accepting and silently ignoring sampling parameters is the same
  failure the golden tests caught in generate(); v1 refuses instead.
- No cancel on disconnect: if the client hangs up mid-stream, the server
  stops writing but the sequence completes and its blocks free at natural
  retirement. Client-initiated abort is a flagged extension (a new scheduler
  operation). Default max_tokens is 128 when omitted; /health reports it.

## Metrics (build in from day one, Phase 2 depends on them)

Per request: arrival, first-token, completion times; prompt length; output
length. Per step: batch size, prefill-token count vs decode-token count, KV
blocks used vs free, step latency.

The per-step prefill-tokens-vs-latency log is exactly what the Phase 2 cost
model reads to derive the prefill compute curve. Logging it now means Phase 2 is
mostly analysis, not new instrumentation.

## The Phase 1 deliverable: the benchmark

The result that shows why continuous batching matters.

- Baseline: static batching. Wait to fill a fixed batch, run it to completion,
  then take the next batch.
- This engine: continuous batching, the loop above.
- Load: bursty Poisson arrivals with a spread of output lengths.
- Result to show: under bursty load with mixed output lengths, continuous
  batching gives markedly higher throughput and lower p95 latency, because a
  short sequence finishing does not have to wait for the longest one in its
  batch. Plot throughput and p95 TTFT vs arrival rate for both.
- Baseline strength: static is given its best timeout (max_wait 0, it never
  fill-waits; a batch forms from whatever is queued when the previous batch
  retires). The measured gap is therefore run-to-completion slot waste only,
  not wait-to-fill idling. Both arms share the same max_batch, so the
  parallelism ceiling is identical and only admission differs; a single
  static cell at double max_batch checks the result is not a ceiling
  artifact. Mechanism runs use a flat-cost fake model (a step costs the same
  regardless of batch composition): batching is free, which is the GPU
  serving regime and is stated in the chart caption.
- Measurement rules: TTFT and latency are measured client-side by the load
  generator (server-side stamps quantize to step boundaries and would flatter
  TTFT). The two arms differ only in the scheduler's admission hook. Spec
  targets, harness reports realized values: burst profile, output lengths,
  and driver schedule lag are reported as observed, not assumed. Client-side
  timestamps include collector-thread wakeup latency, which biases against
  the continuous arm (it runs more concurrent streams); the bias is
  conservative with respect to the expected result and is named here so the
  TTFT floor is explained before it is asked about.

Explain the mechanism in the writeup, not just the numbers.

Scope of the claim, stated before the numbers exist: v1's forward processes
sequences one at a time, so batching exists at the scheduling layer (shared
step cadence, shared admission, iteration-level composition) and not at the
compute layer (no shared matmuls — weights are not read once and applied
across N sequences). The benchmark can therefore demonstrate scheduling wins —
throughput and p95 TTFT under bursty load, queue behavior, preemption cost —
and cannot demonstrate the memory-bandwidth throughput win that compute-layer
batching is famous for. Compute-layer batching is the flagged v2 extension;
naming this boundary here is what keeps the chart honest.

## How this feeds the later phases (the coherence thread)

- The `Evictor` seam in `make_room` is where Phase 3's policy plugs in. v1
  leaves it trivial.
- `BlockMeta` fields are reserved now, populated in Phase 3.
- `recompute_cost` is filled by the Phase 2 cost model, which is itself derived
  from the per-step prefill metrics logged here.

Because the engine owns this cache, every later decision (evict, migrate,
recompute) is measurable rather than inferred from a black box. That is the
whole reason the two-layer project hangs together.
