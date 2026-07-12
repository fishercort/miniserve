"""Waiting/running queues, admission, and the step() loop. See docs/phase1-engine.md.

Implements the v1 policies from the Named failure modes section: capacity
rejection at submit, strict-FIFO admission with progress headroom, two-tier
youngest preemption (self-preemption allowed, all-fresh fallback with its own
reason code), abort-and-requeue to the front of the waiting queue.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import ceil

from miniserve.kv_cache import PagedKVCache
from miniserve.metrics import Metrics


class Status(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


class PreemptionReason(Enum):
    YOUNGEST_DECODED = "youngest_decoded"
    # Defensive tier: unreachable from step() under the progress-headroom
    # admission rule (the requester has always just sampled, so tier 1 is never
    # empty), but make_room may gain callers in Phase 3. The distinct code is
    # for the debugging session where that happens.
    ALL_FRESH_FALLBACK = "all_fresh_fallback"


class CapacityError(ValueError):
    """The request can never fit even in an empty cache. Rejected at submit —
    this is what makes abort-and-requeue provably terminate."""


@dataclass
class SamplingParams:
    temperature: float = 0.0  # v1: greedy; carried for the API surface
    top_p: float = 1.0


@dataclass
class Sequence:
    req_id: str
    prompt_ids: list[int]
    max_tokens: int
    sampling: SamplingParams
    arrival_time: float
    seq_no: int  # submission order; youngest = highest. Survives preemption.
    output_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    first_token_time: float | None = None
    completion_time: float | None = None
    emitted_count: int = 0  # tokens already streamed; survives preemption (no re-emit)
    preempted_count: int = 0

    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def is_finished(self, eos_id: int) -> bool:
        return bool(self.output_ids) and (
            self.output_ids[-1] == eos_id or len(self.output_ids) >= self.max_tokens
        )


@dataclass
class PreemptionEvent:
    req_id: str
    reason: PreemptionReason
    self_preempted: bool  # victim was the sequence whose growth triggered make_room


class Scheduler:
    """Iteration-level scheduler: batch composition changes every step."""

    def __init__(
        self,
        kv: PagedKVCache,
        model,
        max_batch: int,
        eos_id: int,
        clock=time.monotonic,
        on_token=None,
        on_finish=None,
        metrics: Metrics | None = None,
    ) -> None:
        self.kv = kv
        self.model = model
        self.max_batch = max_batch
        self.eos_id = eos_id
        self.clock = clock
        self.on_token = on_token or (lambda req_id, token: None)
        self.on_finish = on_finish or (lambda seq: None)
        self.metrics = metrics if metrics is not None else Metrics()
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.preemptions_total = 0
        self.preemption_log: list[PreemptionEvent] = []
        self._seq_counter = 0

    # -- submission -----------------------------------------------------------

    def submit(
        self,
        req_id: str,
        prompt_ids: list[int],
        max_tokens: int,
        sampling: SamplingParams | None = None,
    ) -> Sequence:
        if not prompt_ids:
            raise ValueError("empty prompt")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        worst = self._blocks_for(len(prompt_ids) + max_tokens)
        if worst > self.kv.num_blocks:
            raise CapacityError(
                f"request {req_id!r} worst case {worst} blocks exceeds cache of "
                f"{self.kv.num_blocks} blocks"
            )
        seq = Sequence(
            req_id=req_id,
            prompt_ids=list(prompt_ids),
            max_tokens=max_tokens,
            sampling=sampling or SamplingParams(),
            arrival_time=self.clock(),
            seq_no=self._seq_counter,
        )
        self._seq_counter += 1
        self.waiting.append(seq)
        return seq

    # -- the step loop --------------------------------------------------------

    def step(self) -> None:
        t_start = self.clock()

        # 1. Admission: strict FIFO — a head that does not fit blocks the queue.
        while self.waiting and len(self.running) < self.max_batch:
            seq = self.waiting[0]
            # Progress headroom: room for current content plus one token — the
            # exact form of the need+1 rule ("admission implies progress").
            # When the content doesn't fill its last block, the next token is
            # already covered and no extra block is demanded; this is what lets
            # a whole-cache-sized request re-admit near its finish line.
            if self.kv.free_count() < self._blocks_for(seq.total_len() + 1):
                break
            self.waiting.popleft()
            self.kv.allocate(seq.req_id, self._blocks_for(seq.total_len()))
            seq.status = Status.PREFILL
            self.running.append(seq)

        if not self.running:
            return

        # 2. Partition this iteration's work.
        prefill = [s for s in self.running if s.status is Status.PREFILL]
        decode = [s for s in self.running if s.status is Status.DECODE]
        # Fresh vs recompute, kept separate: a resumed prefill re-computes KV
        # that existed before its preemption (prompt AND generated tokens), and
        # tokens-recomputed is a headline metric of the benchmark this feeds.
        n_prefill_tokens_fresh = sum(
            s.total_len() for s in prefill if s.preempted_count == 0
        )
        n_prefill_tokens_recompute = sum(
            s.total_len() for s in prefill if s.preempted_count > 0
        )
        n_decode_tokens = len(decode)

        # 3. Forward pass. v1 tests drive this with a fake model; the paged
        #    gather-attention forward lands in model.py.
        logits = self.model.forward(prefill + decode, self.kv)

        # 4. Sample, emit, retire-or-grow. Retirement is inlined (not a separate
        #    phase) so a finished sequence's blocks are back on the free list
        #    before later sequences in the same step reach make_room.
        for s in prefill + decode:
            if s.status is Status.WAITING:
                continue  # preempted moments ago, by an earlier sequence in this loop
            tok = self._sample(logits[s.req_id], s.sampling)
            if s.status is Status.PREFILL:
                if s.first_token_time is None:  # TTFT is first token EVER; resume keeps it
                    s.first_token_time = self.clock()
                s.status = Status.DECODE
            s.output_ids.append(tok)
            while s.emitted_count < len(s.output_ids):
                self.on_token(s.req_id, s.output_ids[s.emitted_count])
                s.emitted_count += 1
            if s.is_finished(self.eos_id):
                self._finish(s)
                continue
            # Grow: the token just sampled needs its KV slot written next step.
            if self._blocks_for(s.total_len()) > len(self.kv.block_tables[s.req_id]):
                if not self.kv.append_block(s.req_id):
                    self._make_room(requester=s)
                    if s.status is Status.WAITING:
                        continue  # self-preempted; resumes via admission later
                    if not self.kv.append_block(s.req_id):
                        # The beneficiary must get its block in the same step —
                        # a make_room that freed nothing is a scheduler bug.
                        raise RuntimeError(f"make_room freed no block for {s.req_id!r}")

        self.metrics.record_step(
            t_start=t_start,
            n_prefill=len(prefill),
            n_decode=len(decode),
            n_prefill_tokens_fresh=n_prefill_tokens_fresh,
            n_prefill_tokens_recompute=n_prefill_tokens_recompute,
            n_decode_tokens=n_decode_tokens,
            kv_used=self.kv.used_count(),
            kv_free=self.kv.free_count(),
            step_latency_s=self.clock() - t_start,
        )

    # -- internals ------------------------------------------------------------

    def _blocks_for(self, n_tokens: int) -> int:
        return ceil(n_tokens / self.kv.block_size)

    @staticmethod
    def _sample(logits, sampling: SamplingParams) -> int:
        return int(logits.argmax().item())  # v1: greedy

    def _finish(self, s: Sequence) -> None:
        s.completion_time = self.clock()
        s.status = Status.FINISHED
        self.kv.free(s.req_id)
        self.running.remove(s)
        self.metrics.record_request(s)
        self.on_finish(s)

    def _make_room(self, requester: Sequence) -> None:
        """Two-tier youngest victim selection; see the decision line in
        docs/phase1-engine.md. The requester itself is eligible (self-preemption)."""
        decoded = [s for s in self.running if s.output_ids]
        if decoded:
            victim = max(decoded, key=lambda s: s.seq_no)
            reason = PreemptionReason.YOUNGEST_DECODED
        else:
            victim = max(self.running, key=lambda s: s.seq_no)
            reason = PreemptionReason.ALL_FRESH_FALLBACK
        self._preempt(victim, reason, self_preempted=victim is requester)

    def _preempt(
        self, victim: Sequence, reason: PreemptionReason, self_preempted: bool
    ) -> None:
        """Abort-and-requeue: free the victim's blocks, front of the waiting
        queue. On re-admission it resumes (re-prefills prompt + generated
        tokens), it does not restart — emitted_count survives, so nothing is
        streamed twice."""
        self.kv.free(victim.req_id)
        victim.status = Status.WAITING
        victim.preempted_count += 1
        self.running.remove(victim)
        self.waiting.appendleft(victim)
        self.preemptions_total += 1
        self.preemption_log.append(PreemptionEvent(victim.req_id, reason, self_preempted))
