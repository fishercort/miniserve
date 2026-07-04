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
        need = ceil(len(seq.prompt_ids) / block_size)
        if kv.free_count() < need:
            break                      # no room, stop admitting this step
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
                evictor.make_room(s)     # Phase 3 seam. v1: preempt or block.
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

The engine runs `step()` in a tight loop on a background thread/async task.
`server.py` pushes new requests into `waiting` and streams emitted tokens back.

### v1 simplifications (scope control)

- Greedy or basic top-k/top-p sampling. Not the interesting part yet.
- Single GPU, no tensor parallelism.
- Gather-based attention rather than a custom paged kernel. Flag the tradeoff.
- No preemption/swapping in v1. If `make_room` is hit, simplest is to block
  admission. Preemption is a clean Phase 3 extension.
- Prefill handling: simplest correct version processes the admitted prefill plus
  the running decode in one step as above. Chunked prefill (cap prefill tokens
  per step to protect decode latency) is the nicer version, list it as a stretch.

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

No HTML forms anywhere. Requests are async: they land in `waiting`, the loop
picks them up, tokens stream back as produced.

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

Explain the mechanism in the writeup, not just the numbers.

## How this feeds the later phases (the coherence thread)

- The `Evictor` seam in `make_room` is where Phase 3's policy plugs in. v1
  leaves it trivial.
- `BlockMeta` fields are reserved now, populated in Phase 3.
- `recompute_cost` is filled by the Phase 2 cost model, which is itself derived
  from the per-step prefill metrics logged here.

Because the engine owns this cache, every later decision (evict, migrate,
recompute) is measurable rather than inferred from a black box. That is the
whole reason the two-layer project hangs together.
