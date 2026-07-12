"""Per-request and per-step metrics collection. See docs/phase1-engine.md (metrics).

The per-step prefill-tokens-vs-latency log is the raw data the Phase 2 cost
model reads — logging it now means Phase 2 is mostly analysis, not new
instrumentation.
"""

from dataclasses import dataclass


@dataclass
class StepRecord:
    t_start: float  # step start time; makes occupancy-over-time plots possible
    n_prefill: int
    n_decode: int
    n_prefill_tokens_fresh: int  # first-admission prompts
    n_prefill_tokens_recompute: int  # resumed prefills: KV that existed pre-preemption
    n_decode_tokens: int
    kv_used: int
    kv_free: int
    step_latency_s: float

    @property
    def n_prefill_tokens(self) -> int:
        return self.n_prefill_tokens_fresh + self.n_prefill_tokens_recompute


@dataclass
class RequestRecord:
    """Exactly one record per request LIFETIME, written at completion.

    Preemption episodes do not produce records — they aggregate into
    ``preempted_count``, and ``first_token_time`` is the first token ever
    (it survives preemption)."""

    req_id: str
    arrival_time: float
    first_token_time: float | None
    completion_time: float | None
    prompt_len: int
    output_len: int
    preempted_count: int

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time


class Metrics:
    """In-memory collector; the server exposes these via GET /metrics later."""

    def __init__(self) -> None:
        self.steps: list[StepRecord] = []
        self.requests: list[RequestRecord] = []

    def record_step(self, **kwargs) -> None:
        self.steps.append(StepRecord(**kwargs))

    def record_request(self, seq) -> None:
        self.requests.append(
            RequestRecord(
                req_id=seq.req_id,
                arrival_time=seq.arrival_time,
                first_token_time=seq.first_token_time,
                completion_time=seq.completion_time,
                prompt_len=len(seq.prompt_ids),
                output_len=len(seq.output_ids),
                preempted_count=seq.preempted_count,
            )
        )
