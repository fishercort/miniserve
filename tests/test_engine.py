"""Engine tests. Deterministic tests drive step_once() on the test thread with
no engine thread at all; the integration trio (clean finish, shutdown mid
stream, submit-vs-stop race) plus crash-to-sentinel run the real thread."""

import threading
import time

import pytest
import torch

from miniserve.engine import FINISH, TOKEN, Engine, EngineStopped
from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import CapacityError, Scheduler

EOS = 0
BS = 4


class ScriptModel:
    """Token n for request r = scripts[r][n]. Requires known req_ids."""

    def __init__(self, scripts):
        self.scripts = scripts

    def forward(self, seqs, kv):
        out = {}
        for s in seqs:
            t = self.scripts[s.req_id][len(s.output_ids)]
            v = torch.zeros(16)
            v[t] = 1.0
            out[s.req_id] = v
        return out


class AnyModel:
    """Request-id independent: token n = (n % 5) + 1, never EOS."""

    def forward(self, seqs, kv):
        out = {}
        for s in seqs:
            v = torch.zeros(16)
            v[(len(s.output_ids) % 5) + 1] = 1.0
            out[s.req_id] = v
        return out


class CrashModel(AnyModel):
    """Healthy for the first N forwards, then raises."""

    def __init__(self, crash_after: int):
        self.calls = 0
        self.crash_after = crash_after

    def forward(self, seqs, kv):
        self.calls += 1
        if self.calls > self.crash_after:
            raise RuntimeError("model exploded")
        return super().forward(seqs, kv)


class SlowModel(AnyModel):
    """A few ms per forward, so generation reliably outlives a stop() call."""

    def forward(self, seqs, kv):
        time.sleep(0.005)
        return super().forward(seqs, kv)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def make_engine(model, num_blocks=8, max_batch=4, clock=None):
    kv = PagedKVCache(1, num_blocks, BS, 1, 2, poison_on_free=True)
    kwargs = {"clock": clock} if clock else {}
    sched = Scheduler(kv, model, max_batch=max_batch, eos_id=EOS, **kwargs)
    return Engine(sched, idle_wait_s=0.005), kv


def drain(stream, timeout=5.0):
    """Blocking-collect until the finish sentinel. Raises queue.Empty on hang."""
    items = []
    while True:
        kind, payload = stream.get(timeout=timeout)
        items.append((kind, payload))
        if kind == FINISH:
            return items


# -- deterministic (no engine thread) -----------------------------------------


def test_manual_stream_and_summary():
    engine, kv = make_engine(ScriptModel({"a": [1, 2, EOS]}), clock=FakeClock())
    fut, stream = engine.submit("a", [7] * 4, max_tokens=8)
    assert not fut.done()  # nothing happens until the loop turns
    for _ in range(5):
        engine.step_once()
    assert fut.result(timeout=0) == "a"
    items = drain(stream, timeout=0.1)
    assert [p for k, p in items if k == TOKEN] == [1, 2, EOS]
    summary = items[-1][1]
    assert summary.output_tokens == 3
    assert summary.ttft_ms is not None and summary.ttft_ms > 0
    assert summary.total_ms is not None and summary.throughput_tok_s is not None
    assert summary.aborted is False
    kv.check_invariants()
    assert kv.free_count() == kv.num_blocks


def test_capacity_error_crosses_boundary_intact():
    engine, kv = make_engine(AnyModel(), num_blocks=2)
    fut, _ = engine.submit("big", [7] * 6, max_tokens=8)  # worst case 3 > 2 blocks
    engine.step_once()
    with pytest.raises(CapacityError, match="worst case"):
        fut.result(timeout=0)
    assert not kv.block_tables  # zero allocator state, behavior 26
    kv.check_invariants()


def test_validation_error_crosses_boundary():
    engine, _ = make_engine(AnyModel())
    fut, _ = engine.submit("empty", [], max_tokens=4)
    engine.step_once()
    with pytest.raises(ValueError, match="empty prompt"):
        fut.result(timeout=0)


# -- integration trio (real thread) --------------------------------------------


def test_thread_clean_serve_and_finish():
    engine, _ = make_engine(ScriptModel({"a": [3, 1, EOS]}))
    engine.start()
    try:
        fut, stream = engine.submit("a", [7] * 4, max_tokens=8)
        assert fut.result(timeout=5) == "a"
        items = drain(stream)
        assert [p for k, p in items if k == TOKEN] == [3, 1, EOS]
        assert items[-1][1].aborted is False
    finally:
        engine.stop()


def test_shutdown_mid_stream_delivers_aborted_sentinel():
    engine, _ = make_engine(SlowModel())
    engine.start()
    # max_tokens sized to fit the capacity check (worst 8 blocks == cache),
    # slow enough per step that stop() lands mid-generation.
    fut, stream = engine.submit("long", [7] * 4, max_tokens=28)
    assert fut.result(timeout=5) == "long"
    first = stream.get(timeout=5)
    assert first[0] == TOKEN  # streaming has begun
    engine.stop()  # asserts the thread joined
    items = drain(stream)  # must terminate cleanly, not hang
    summary = items[-1][1]
    assert summary.aborted is True
    assert summary.output_tokens >= 1


def test_submit_racing_stop_never_hangs():
    engine, _ = make_engine(AnyModel(), num_blocks=8, max_batch=8)
    engine.start()
    stopper = threading.Thread(target=lambda: (time.sleep(0.02), engine.stop()))
    stopper.start()
    futures = []
    for i in range(300):
        fut, _ = engine.submit(f"r{i}", [7], max_tokens=1)
        futures.append(fut)
    stopper.join(timeout=15)
    assert not stopper.is_alive()
    for fut in futures:
        # exception() blocks until resolution; TimeoutError here = a hung
        # client, the exact bug the doorway lock exists to prevent.
        exc = fut.exception(timeout=10)
        assert exc is None or isinstance(exc, EngineStopped | ValueError)


# -- crash path -----------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_engine_crash_delivers_sentinel_and_refuses_new_work():
    """run() re-raises after sentineling, so the thread dies loudly by design;
    the filtered warning is that death being noticed."""
    engine, _ = make_engine(CrashModel(crash_after=2))
    engine.start()
    fut, stream = engine.submit("a", [7] * 4, max_tokens=20)  # worst 6 of 8 blocks
    assert fut.result(timeout=5) == "a"
    items = drain(stream)  # crash must terminate the stream, not strand it
    assert items[-1][1].aborted is True
    engine._thread.join(timeout=5)
    assert not engine._thread.is_alive()
    fut2, _ = engine.submit("b", [7], max_tokens=1)
    with pytest.raises(EngineStopped):
        fut2.result(timeout=1)
