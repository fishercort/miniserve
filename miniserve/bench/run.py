"""Benchmark driver: replays a workload against the engine in real time.

TTFT and completion latency are measured CLIENT-SIDE, from the caller's side
of the token stream: server-side stamps quantize to step boundaries and would
flatter TTFT. Arrival time is stamped immediately before submit().

Two measurement caveats, named up front:
- sched_lag_s per request records how late the driver submitted relative to
  the nominal schedule. If lag is microseconds the schedule was delivered; if
  it is milliseconds during bursts, the realized workload differs from spec
  and the report says so. A benchmark that cannot prove it delivered its own
  workload cannot defend its chart.
- Client-side timestamps include collector-thread wakeup latency (N collectors
  compete for the GIL). This biases AGAINST the continuous arm, which runs
  more concurrent streams; a conservative bias is publishable, a favorable
  one is not.
"""

import threading
import time
from dataclasses import dataclass

from miniserve.bench.workload import Arrival
from miniserve.engine import FINISH, Engine


@dataclass
class RequestResult:
    req_id: str
    submitted_s: float
    sched_lag_s: float  # submitted_s minus nominal at_s: driver-delivery proof
    first_token_s: float | None
    done_s: float | None
    output_tokens: int
    aborted: bool

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.submitted_s

    @property
    def latency_s(self) -> float | None:
        if self.done_s is None:
            return None
        return self.done_s - self.submitted_s


def run_workload(
    engine: Engine,
    schedule: list[Arrival],
    on_all_submitted=None,
    collect_timeout_s: float = 300.0,
) -> list[RequestResult]:
    """Submit each arrival at its scheduled time; collect every stream in its
    own thread. on_all_submitted fires after the last submit (the static
    baseline uses it to set the flush flag)."""
    t0 = time.monotonic()
    results: list[RequestResult] = []
    results_lock = threading.Lock()
    collectors: list[threading.Thread] = []

    def collect(req_id: str, stream, submitted_s: float, sched_lag_s: float) -> None:
        first = None
        while True:
            kind, payload = stream.get()
            now = time.monotonic() - t0
            if kind == FINISH:
                with results_lock:
                    results.append(
                        RequestResult(
                            req_id=req_id,
                            submitted_s=submitted_s,
                            sched_lag_s=sched_lag_s,
                            first_token_s=first,
                            done_s=now,
                            output_tokens=payload.output_tokens,
                            aborted=payload.aborted,
                        )
                    )
                return
            if first is None:
                first = now

    for a in schedule:
        delay = a.at_s - (time.monotonic() - t0)
        if delay > 0:
            time.sleep(delay)
        submitted_s = time.monotonic() - t0
        _, stream = engine.submit(a.req_id, a.prompt_ids, a.max_tokens)
        th = threading.Thread(
            target=collect,
            args=(a.req_id, stream, submitted_s, submitted_s - a.at_s),
            daemon=True,
        )
        th.start()
        collectors.append(th)

    if on_all_submitted is not None:
        on_all_submitted()
    for th in collectors:
        th.join(timeout=collect_timeout_s)
        if th.is_alive():
            raise RuntimeError("collector hung: a stream never terminated")
    return sorted(results, key=lambda r: r.req_id)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(p / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def summarize(results: list[RequestResult]) -> dict:
    finished = [r for r in results if not r.aborted and r.done_s is not None]
    n_aborted = sum(1 for r in results if r.aborted)
    ttfts = [r.ttft_s for r in finished if r.ttft_s is not None]
    latencies = [r.latency_s for r in finished]
    lags = [r.sched_lag_s for r in results]
    total_tokens = sum(r.output_tokens for r in finished)
    # Makespan over ALL results: excluding aborted tails would shorten the
    # window and quietly inflate throughput.
    makespan = max((r.done_s for r in results if r.done_s is not None), default=0.0)
    return {
        "n_finished": len(finished),
        "n_aborted": n_aborted,
        # A chart must never quietly average over a partially failed run.
        "run_valid_for_chart": n_aborted == 0 and len(finished) == len(results),
        "total_output_tokens": total_tokens,
        "makespan_s": makespan,
        "throughput_tok_s": total_tokens / makespan if makespan > 0 else 0.0,
        "ttft_p50_s": percentile(ttfts, 50) if ttfts else None,
        "ttft_p95_s": percentile(ttfts, 95) if ttfts else None,
        "latency_p95_s": percentile(latencies, 95) if latencies else None,
        # Driver-delivery proof: microsecond lags mean the nominal schedule
        # was delivered; millisecond lags during bursts mean it was not, and
        # the report must say so rather than let the chart imply otherwise.
        "sched_lag_p95_s": percentile(lags, 95) if lags else None,
        "sched_lag_max_s": max(lags) if lags else None,
        # realized output lengths: caps in mechanism runs, observed for real
        # models (EOS can fire early); reported, not assumed.
        "output_tokens_mean": total_tokens / len(finished) if finished else 0.0,
    }
