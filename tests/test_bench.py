"""Bench harness tests: workload determinism and validation, static-batching
admission semantics (deterministic, thread-free), and a real-time smoke run of
both arms with a slow fake model. No performance assertions: CI machines make
perf claims flaky, and the harness's job here is to be correct, not fast."""

import time

import pytest
import torch

from miniserve.bench.run import run_workload, summarize
from miniserve.bench.static import StaticScheduler
from miniserve.bench.workload import (
    Arrival,
    WorkloadSpec,
    generate,
    realized_burst_profile,
)
from miniserve.engine import Engine
from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import Scheduler

BS = 4
EOS = 0


class BenchModel:
    """Never emits EOS: the cap IS the realized length (the mechanism-run
    convention). A few ms per forward so real-time runs behave like runs."""

    def __init__(self, delay_s: float = 0.002):
        self.delay_s = delay_s

    def forward(self, seqs, kv):
        time.sleep(self.delay_s)
        out = {}
        for s in seqs:
            v = torch.zeros(16)
            v[(len(s.output_ids) % 5) + 1] = 1.0
            out[s.req_id] = v
        return out


# -- workload -------------------------------------------------------------------


def test_workload_deterministic_same_seed():
    spec = WorkloadSpec(n_requests=20, seed=7)
    assert generate(spec) == generate(spec)
    assert generate(spec) != generate(WorkloadSpec(n_requests=20, seed=8))


def test_workload_times_increase_and_lengths_bounded():
    spec = WorkloadSpec(n_requests=40, seed=1)
    schedule = generate(spec)
    times = [a.at_s for a in schedule]
    assert times == sorted(times) and times[0] > 0
    lo = min(lo for _, (lo, _) in spec.output_mix)
    hi = max(hi for _, (_, hi) in spec.output_mix)
    for a in schedule:
        assert lo <= a.max_tokens <= hi
        assert spec.prompt_len[0] <= len(a.prompt_ids) <= spec.prompt_len[1]


def test_workload_rejects_bad_mix_weights():
    with pytest.raises(ValueError, match="weights sum"):
        generate(WorkloadSpec(output_mix=((0.5, (1, 2)), (0.3, (3, 4)))))


def test_realized_burst_profile_counts_every_arrival():
    spec = WorkloadSpec(n_requests=30, seed=3, burst_factor=3.0, rate_rps=8.0)
    schedule = generate(spec)
    profile = realized_burst_profile(schedule, spec)
    assert sum(profile) == len(schedule)


# -- static admission semantics (deterministic, no threads) ---------------------


def make_static(max_batch=2, num_blocks=16):
    kv = PagedKVCache(1, num_blocks, BS, 1, 2, poison_on_free=True)

    class TinyModel:
        def forward(self, seqs, kv):
            out = {}
            for s in seqs:
                v = torch.zeros(16)
                v[1] = 1.0
                out[s.req_id] = v
            return out

    return StaticScheduler(kv, TinyModel(), max_batch=max_batch, eos_id=EOS), kv


def test_static_admits_only_full_batches_then_flush_drains():
    sched, kv = make_static(max_batch=2)
    for i in range(3):
        sched.submit(f"r{i}", [7, 7], max_tokens=2)
    sched.step()
    assert len(sched.running) == 2  # full batch admitted together
    assert len(sched.waiting) == 1
    # batch runs to completion; the leftover single is NOT admitted meanwhile
    for _ in range(4):
        sched.step()
        assert all(s.req_id != "r2" for s in sched.running)
    assert not sched.running  # first batch done
    sched.step()
    assert not sched.running and len(sched.waiting) == 1  # still waiting: no flush
    sched.flush = True  # driver says: no more arrivals are coming
    sched.step()
    assert [s.req_id for s in sched.running] == ["r2"]  # partial batch drains
    while sched.running:
        sched.step()
    kv.check_invariants()


def test_static_never_joins_mid_batch():
    """The defining property: a request arriving mid-batch waits for the batch
    to fully retire, even with free blocks and batch room available."""
    sched, kv = make_static(max_batch=4)
    sched.submit("a", [7, 7], max_tokens=4)
    sched.flush = True
    sched.step()
    assert [s.req_id for s in sched.running] == ["a"]
    sched.submit("late", [7, 7], max_tokens=2)
    sched.step()
    assert all(s.req_id != "late" for s in sched.running)  # room exists; policy says no
    while any(s.req_id == "a" for s in sched.running):
        sched.step()
    while sched.waiting or sched.running:
        sched.step()
    kv.check_invariants()


def test_static_max_wait_opens_partial_batch():
    """At low arrival rates, the partial-batch timer opens admission without
    flush: static must not stall forever waiting to fill."""

    class FakeClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            self.t += 1.0
            return self.t

    kv = PagedKVCache(1, 16, BS, 1, 2, poison_on_free=True)

    class TinyModel:
        def forward(self, seqs, kv):
            out = {}
            for s in seqs:
                v = torch.zeros(16)
                v[1] = 1.0
                out[s.req_id] = v
            return out

    sched = StaticScheduler(
        kv, TinyModel(), max_batch=4, eos_id=EOS, clock=FakeClock(), max_wait_s=10.0
    )
    sched.submit("solo", [7, 7], max_tokens=2)
    sched.step()
    assert not sched.running  # not full, no flush, timer not yet expired
    admitted_at_step = None
    for i in range(30):
        sched.step()
        if sched.running or not sched.waiting:
            admitted_at_step = i
            break
    assert admitted_at_step is not None and admitted_at_step > 0  # waited, then opened
    while sched.waiting or sched.running:
        sched.step()
    kv.check_invariants()


def test_preemption_preserves_bench_measurements():
    """The continuous arm under memory pressure preempts and resumes; the
    bench must still report exact realized lengths (cap == realized survives
    preemption because emitted_count does) with zero aborts."""
    kv = PagedKVCache(1, 3, BS, 1, 2, poison_on_free=True)
    sched = Scheduler(kv, BenchModel(delay_s=0.001), max_batch=4, eos_id=EOS)
    engine = Engine(sched, idle_wait_s=0.002)
    engine.start()
    schedule = [
        Arrival(0.0, "o", [7] * 4, 5),  # worst 3 blocks of 3: the whole cache
        Arrival(0.0, "y", [7] * 2, 6),  # forces contention, y gets preempted
    ]
    try:
        results = run_workload(engine, schedule)
    finally:
        engine.stop()
    kv.check_invariants()
    assert sched.preemptions_total >= 1  # pressure was real
    by_id = {r.req_id: r for r in results}
    assert by_id["o"].output_tokens == 5 and by_id["y"].output_tokens == 6
    for r in results:
        assert not r.aborted and r.ttft_s is not None and r.latency_s is not None


# -- real-time smoke: both arms end to end ---------------------------------------


def smoke_spec():
    return WorkloadSpec(
        n_requests=10,
        rate_rps=40.0,
        seed=2,
        prompt_len=(2, 6),
        output_mix=((0.6, (2, 4)), (0.4, (5, 8))),
        vocab=16,
    )


def run_arm(static: bool):
    kv = PagedKVCache(1, 24, BS, 1, 2, poison_on_free=True)
    model = BenchModel()
    cls = StaticScheduler if static else Scheduler
    sched = cls(kv, model, max_batch=4, eos_id=EOS)
    engine = Engine(sched, idle_wait_s=0.002)
    engine.start()
    try:
        flush = (lambda: setattr(sched, "flush", True)) if static else None
        results = run_workload(engine, generate(smoke_spec()), on_all_submitted=flush)
    finally:
        engine.stop()
    kv.check_invariants()
    return results


@pytest.mark.parametrize("static", [False, True], ids=["continuous", "static"])
def test_smoke_arm_completes_and_measures(static):
    results = run_arm(static)
    assert len(results) == 10
    for r in results:
        assert not r.aborted
        assert r.ttft_s is not None and r.ttft_s >= 0
        assert r.latency_s is not None and r.latency_s >= r.ttft_s
        assert r.output_tokens >= 1  # cap == realized: BenchModel never EOSes
    summary = summarize(results)
    assert summary["n_finished"] == 10 and summary["n_aborted"] == 0
    assert summary["run_valid_for_chart"] is True
    assert summary["throughput_tok_s"] > 0
    assert summary["ttft_p95_s"] >= summary["ttft_p50_s"]
    assert summary["sched_lag_max_s"] is not None  # driver-delivery proof present
