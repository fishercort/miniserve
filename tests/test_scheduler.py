"""Scheduler tests: one test (at least) per numbered behavior in the step-2
test list. Behavior numbers are cited in each docstring. Fake model + fake
clock; block conservation is re-checked after every step of every run.
"""

import pytest
import torch

from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import (
    CapacityError,
    PreemptionReason,
    Scheduler,
    Status,
)

EOS = 0
BS = 4  # block size for all tests


class FakeModel:
    """Deterministic: next token for a sequence = script[req_id][len(output_ids)].

    A function of sequence CONTENT, not call count — so a preempted-and-resumed
    sequence replays byte-identically (behavior 15's precondition)."""

    def __init__(self, scripts: dict[str, list[int]], vocab_size: int = 32):
        self.scripts = scripts
        self.vocab_size = vocab_size

    def forward(self, seqs, kv):
        out = {}
        for s in seqs:
            tok = self.scripts[s.req_id][len(s.output_ids)]
            logits = torch.zeros(self.vocab_size)
            logits[tok] = 1.0
            out[s.req_id] = logits
        return out


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def make(scripts, num_blocks=8, max_batch=8):
    kv = PagedKVCache(1, num_blocks, BS, 1, 2, poison_on_free=True)
    tokens: list[tuple[str, int]] = []
    finished: list[str] = []
    sched = Scheduler(
        kv,
        FakeModel(scripts),
        max_batch=max_batch,
        eos_id=EOS,
        clock=FakeClock(),
        on_token=lambda r, t: tokens.append((r, t)),
        on_finish=lambda s: finished.append(s.req_id),
    )
    return sched, kv, tokens, finished


def run_all(sched, kv, limit=300):
    """Drive to completion; behavior 19 (never wedges) and 20 (conservation
    after every step) are asserted on every run that uses this."""
    steps = 0
    while (sched.waiting or sched.running) and steps < limit:
        sched.step()
        kv.check_invariants()
        steps += 1
    assert not sched.waiting and not sched.running, "engine wedged"
    return steps


def step_until(sched, kv, pred, limit=300):
    steps = 0
    while not pred() and steps < limit:
        sched.step()
        kv.check_invariants()
        steps += 1
    assert pred(), "condition never reached"
    return steps


def stream_of(tokens, req_id):
    return [t for r, t in tokens if r == req_id]


# -- admission (1-6) ----------------------------------------------------------


def test_admission_claims_blocks():  # behavior 1
    sched, kv, _, _ = make({"a": [1, 1, EOS]})
    seq = sched.submit("a", [7] * 5, max_tokens=3)
    sched.step()
    assert seq.status is Status.DECODE
    assert "a" in kv.block_tables
    assert kv.used_count() == 2  # ceil(5/4) prompt blocks... plus growth as needed


def test_no_room_waits_then_admitted_later():  # behavior 2
    scripts = {"a": [1, EOS], "b": [1, EOS]}
    sched, kv, _, finished = make(scripts, num_blocks=2)
    sched.submit("a", [7] * 4, max_tokens=2)
    sched.submit("b", [7] * 4, max_tokens=2)
    sched.step()
    assert len(sched.waiting) == 1 and sched.waiting[0].req_id == "b"  # not dropped
    run_all(sched, kv)
    assert set(finished) == {"a", "b"}


def test_max_batch_respected():  # behavior 3
    scripts = {"a": [1, EOS], "b": [1, EOS]}
    sched, kv, _, finished = make(scripts, num_blocks=8, max_batch=1)
    sched.submit("a", [7] * 4, max_tokens=2)
    sched.submit("b", [7] * 4, max_tokens=2)
    sched.step()
    assert len(sched.running) == 1  # blocks were plentiful; batch cap held anyway
    run_all(sched, kv)
    assert set(finished) == {"a", "b"}


def test_strict_fifo_head_blocks_queue():  # behavior 4 (decision: strict FIFO)
    scripts = {"a": [1] * 4, "h": [1, EOS], "s": [EOS]}
    sched, kv, _, _ = make(scripts, num_blocks=3)
    sched.submit("a", [7] * 4, max_tokens=4)
    sched.step()  # a admitted, grows to 2 blocks; 1 free
    sched.submit("h", [7] * 4, max_tokens=2)  # head: needs 2, doesn't fit
    sched.submit("s", [7] * 2, max_tokens=1)  # would fit in 1 — must NOT skip ahead
    sched.step()
    assert [s.req_id for s in sched.waiting] == ["h", "s"]
    assert [s.req_id for s in sched.running] == ["a"]


def test_capacity_reject_and_boundary_fit():  # behaviors 5, 26 (seam: exact headroom)
    sched, kv, _, finished = make({"big": [1] * 8}, num_blocks=2)
    # Worst case 3 blocks > 2: rejected at submit, exactly one client-visible
    # error, zero allocator state (behavior 26).
    with pytest.raises(CapacityError):
        sched.submit("no", [7] * 6, max_tokens=3)
    assert not kv.block_tables and not sched.waiting
    kv.check_invariants()
    # Worst case exactly the whole cache: accepted AND completes. Under a
    # literal need+1 admission rule this wedges; the exact-form headroom is
    # what makes it terminate.
    sched.submit("big", [7] * 4, max_tokens=4)
    run_all(sched, kv)
    assert finished == ["big"]


def test_admission_implies_progress():  # behavior 6
    scripts = {"a": [1] * 8, "b": [1, EOS]}
    sched, kv, _, _ = make(scripts, num_blocks=3)
    sched.submit("a", [7] * 4, max_tokens=8)
    sched.step()  # a holds 2 blocks; 1 free
    b = sched.submit("b", [7] * 3, max_tokens=2)  # fits: 3+1 tokens = 1 block
    sched.step()
    assert len(b.output_ids) >= 1  # produced a token...
    assert b.preempted_count == 0  # ...without being preempted by its own growth
    assert sched.preemptions_total == 0


# -- lifecycle (7-12) ---------------------------------------------------------


def test_prefill_once_then_decode():  # behavior 7
    sched, kv, _, _ = make({"a": [1, 1, EOS]})
    sched.submit("a", [7] * 4, max_tokens=3)
    sched.step()
    assert sched.metrics.steps[0].n_prefill == 1
    sched.step()
    assert sched.metrics.steps[1].n_prefill == 0
    assert sched.metrics.steps[1].n_decode == 1


def test_ttft_stamped_at_first_token():  # behavior 8
    sched, kv, _, _ = make({"a": [1, EOS]})
    seq = sched.submit("a", [7] * 4, max_tokens=2)
    assert seq.first_token_time is None
    sched.step()
    assert seq.first_token_time is not None
    assert seq.first_token_time > seq.arrival_time
    assert seq.completion_time is None or seq.completion_time >= seq.first_token_time


def test_retirement_eos_and_max_tokens():  # behavior 9
    scripts = {"eos": [5, EOS], "cap": [1, 1, 1]}
    sched, kv, _, finished = make(scripts)
    sched.submit("eos", [7] * 4, max_tokens=8)
    sched.submit("cap", [7] * 4, max_tokens=3)
    run_all(sched, kv)
    assert sorted(finished) == ["cap", "eos"]
    assert finished.count("eos") == 1  # emitted exactly once
    assert kv.free_count() == kv.num_blocks  # all blocks returned
    by_id = {r.req_id: r for r in sched.metrics.requests}
    assert by_id["eos"].output_len == 2  # stopped at EOS
    assert by_id["cap"].output_len == 3  # stopped at max_tokens
    assert all(r.completion_time is not None for r in sched.metrics.requests)


def test_continuous_batching_short_finishes_first():  # behavior 10
    scripts = {"long": [1] * 10, "short": [1, EOS]}
    sched, kv, _, finished = make(scripts)
    sched.submit("long", [7] * 4, max_tokens=10)
    sched.submit("short", [7] * 4, max_tokens=2)
    step_until(sched, kv, lambda: "short" in finished)
    assert "long" not in finished
    assert [s.req_id for s in sched.running] == ["long"]  # kept running throughout
    run_all(sched, kv)


def test_block_boundary_growth():  # behavior 11
    sched, kv, _, _ = make({"a": [1] * 6})
    sched.submit("a", [7] * 4, max_tokens=6)  # prompt exactly fills 1 block
    sched.step()  # first token crosses into a new block
    assert len(kv.block_tables["a"]) == 2


def test_empty_step_is_noop():  # behavior 12
    sched, kv, _, _ = make({})
    sched.step()
    assert not sched.metrics.steps
    kv.check_invariants()


# -- preemption (13-19, 23-25) -------------------------------------------------


def pressured_pair():
    """Old 'o' and young 'y', both decoded, contention forces one preemption."""
    scripts = {"o": [1] * 8, "y": [1] * 8}
    sched, kv, tokens, finished = make(scripts, num_blocks=3)
    sched.submit("o", [7] * 4, max_tokens=5)  # worst 3 blocks
    sched.submit("y", [7] * 2, max_tokens=6)  # worst 2 blocks
    return sched, kv, tokens, finished


def test_two_tier_victim_older_decoded_survives():  # behaviors 13, 14, 18
    sched, kv, _, _ = pressured_pair()
    step_until(sched, kv, lambda: sched.preemptions_total >= 1)
    ev = sched.preemption_log[0]
    assert ev.req_id == "y"  # younger decoded dies...
    assert any(s.req_id == "o" for s in sched.running)  # ...older decoded survives
    assert ev.reason is PreemptionReason.YOUNGEST_DECODED
    assert sched.waiting[0].req_id == "y"  # behavior 14: front of the queue
    assert sched.preemptions_total == 1  # behavior 18: counter
    run_all(sched, kv)


def test_beneficiary_gets_block_same_step():  # behavior 23
    """The preemption's beneficiary never stalls: 'o' produces one token every
    step it runs, so its completion step-count matches a contention-free run."""
    sched, kv, _, finished = pressured_pair()
    steps_pressured = step_until(sched, kv, lambda: "o" in finished)

    solo_sched, solo_kv, _, solo_finished = make({"o": [1] * 8}, num_blocks=3)
    solo_sched.submit("o", [7] * 4, max_tokens=5)
    steps_solo = step_until(solo_sched, solo_kv, lambda: "o" in solo_finished)

    assert steps_pressured == steps_solo  # no silent one-step-per-preemption stall
    run_all(sched, kv)


def test_resume_not_restart_byte_identical():  # behaviors 15, 16
    """PAIR COVERAGE: proves the scheduler's real preempt→readmit path yields a
    correct resume at the token-stream level; that the MODEL's forward treats
    such a resumed state correctly (RoPE positions × mask) is proven by
    test_model.test_resumed_prefill_equals_fresh_prefill. Complete only as a
    pair — do not delete either believing the other covers it."""
    sched, kv, tokens, finished = pressured_pair()
    run_all(sched, kv)
    assert sched.preemptions_total >= 1  # pressure actually happened

    solo_sched, solo_kv, solo_tokens, _ = make({"y": [1] * 8}, num_blocks=3)
    solo_sched.submit("y", [7] * 2, max_tokens=6)
    run_all(solo_sched, solo_kv)

    pressured_stream = stream_of(tokens, "y")
    solo_stream = stream_of(solo_tokens, "y")
    assert pressured_stream == solo_stream  # byte-identical under greedy
    assert len(pressured_stream) == 6  # nothing emitted twice (behavior 16)


def test_all_fresh_fallback_reason_code():  # behavior 17
    """Unreachable from step() under exact-form headroom (the requester has
    always just sampled), so pinned as a unit test of the selection function —
    the defensive tier and its distinct reason code."""
    scripts = {"f1": [1] * 4, "f2": [1] * 4}
    sched, kv, _, _ = make(scripts, num_blocks=4)
    sched.submit("f1", [7] * 4, max_tokens=2)
    sched.submit("f2", [7] * 4, max_tokens=2)
    # Admit both without running the sample loop: mimic mid-step all-fresh state.
    while sched.waiting:
        seq = sched.waiting.popleft()
        kv.allocate(seq.req_id, 1)
        seq.status = Status.PREFILL
        sched.running.append(seq)
    sched._make_room(requester=sched.running[0])
    ev = sched.preemption_log[0]
    assert ev.reason is PreemptionReason.ALL_FRESH_FALLBACK
    assert ev.req_id == "f2"  # youngest overall
    kv.check_invariants()


def test_self_preemption():  # behavior 24 (decision: self-preemption legal)
    scripts = {"c": [1, 1, 1, 1], "a": [1] * 8}
    sched, kv, tokens, finished = make(scripts, num_blocks=3)
    sched.submit("c", [7] * 4, max_tokens=4)  # old, tops out at 2 blocks
    sched.submit("a", [7] * 2, max_tokens=8)  # young, will need a 2nd block
    step_until(sched, kv, lambda: sched.preemptions_total >= 1)
    ev = sched.preemption_log[0]
    assert ev.req_id == "a" and ev.self_preempted  # youngest-decoded == requester
    run_all(sched, kv)
    assert set(finished) == {"a", "c"}
    # and the resumed self-preempted sequence streamed correctly:
    solo_sched, solo_kv, solo_tokens, _ = make({"a": [1] * 8}, num_blocks=3)
    solo_sched.submit("a", [7] * 2, max_tokens=8)
    run_all(solo_sched, solo_kv)
    assert stream_of(tokens, "a") == stream_of(solo_tokens, "a")


def test_same_step_first_token_counts_for_eligibility():  # behavior 25
    """A sequence that prefilled and sampled earlier in the same step is
    victim-eligible at the instant make_room runs (normal reason, no fallback)."""
    scripts = {"o": [1] * 8, "c": [1, 1, EOS]}
    sched, kv, _, _ = make(scripts, num_blocks=3)
    sched.submit("o", [7] * 4, max_tokens=6)
    for _ in range(4):
        sched.step()  # o reaches total 8: two full blocks, 1 block free
    c = sched.submit("c", [7] * 2, max_tokens=3)
    # One step, three events in order: c is admitted into the last free block
    # and samples its first token; then o (decode) crosses into its 3rd block,
    # the free list is empty, and c — exactly one same-step token old — must be
    # the youngest DECODED victim, not an all-fresh fallback.
    sched.step()
    kv.check_invariants()
    assert sched.preemptions_total == 1
    ev = sched.preemption_log[0]
    assert ev.req_id == "c"
    assert ev.reason is PreemptionReason.YOUNGEST_DECODED  # counted as decoded
    assert len(c.output_ids) == 1  # it held only its same-step first token
    run_all(sched, kv)


def test_sustained_overload_terminates():  # behavior 19 (termination property)
    scripts = {f"r{i}": [1] * 12 for i in range(6)}
    sched, kv, _, finished = make(scripts, num_blocks=4, max_batch=4)
    for i in range(6):
        sched.submit(f"r{i}", [7] * 4, max_tokens=4 + (i % 3))
    run_all(sched, kv)  # asserts: everything finishes, invariants each step
    assert len(finished) == 6
    assert sched.preemptions_total >= 1  # pressure was real, not a soft pass
    assert kv.free_count() == kv.num_blocks


# -- bookkeeping (20-22) and submit validation (27) ----------------------------


def test_step_metrics_recorded():  # behavior 21
    sched, kv, _, _ = make({"a": [1, 1, EOS]})
    sched.submit("a", [7] * 5, max_tokens=3)
    run_all(sched, kv)
    assert sched.metrics.steps
    first = sched.metrics.steps[0]
    assert first.n_prefill == 1 and first.n_prefill_tokens_fresh == 5
    assert first.n_prefill_tokens_recompute == 0  # never preempted
    for rec in sched.metrics.steps:
        assert rec.kv_used + rec.kv_free == kv.num_blocks


def test_recompute_tokens_metered_separately():  # behavior 29 (added: metrics gap)
    """A resumed prefill's tokens are recompute volume, not fresh prompt volume
    — the split Phase 2's cost model and the benchmark's headline metric need.
    The volume must equal the victim's total_len() at resume exactly: prompt
    plus every token generated before the preemption destroyed its KV."""
    sched, kv, _, _ = pressured_pair()
    step_until(sched, kv, lambda: sched.preemptions_total >= 1)
    victim_id = sched.preemption_log[0].req_id
    victim = next(s for s in sched.waiting if s.req_id == victim_id)
    expected = victim.total_len()  # frozen while WAITING: nothing generates
    run_all(sched, kv)
    recompute = [r for r in sched.metrics.steps if r.n_prefill_tokens_recompute > 0]
    assert recompute  # the resume was metered as recompute...
    assert recompute[0].n_prefill_tokens_recompute == expected  # ...all of it, exactly


def test_request_metrics_include_preempted():  # behavior 22
    sched, kv, _, _ = pressured_pair()
    run_all(sched, kv)
    by_id = {r.req_id: r for r in sched.metrics.requests}
    assert set(by_id) == {"o", "y"}
    assert by_id["y"].preempted_count >= 1
    for rec in by_id.values():
        assert rec.ttft_s is not None and rec.completion_time is not None


def test_submit_validation():  # behavior 27 (added during implementation)
    sched, kv, _, _ = make({})
    with pytest.raises(ValueError, match="empty prompt"):
        sched.submit("a", [], max_tokens=4)
    with pytest.raises(ValueError, match="max_tokens"):
        sched.submit("a", [7], max_tokens=0)
    assert not sched.waiting and not kv.block_tables


def test_finish_on_first_token():  # behavior 28 (added during implementation)
    sched, kv, tokens, finished = make({"a": [EOS]})
    seq = sched.submit("a", [7] * 4, max_tokens=4)
    sched.step()
    assert finished == ["a"]
    assert seq.first_token_time is not None and seq.completion_time is not None
    assert stream_of(tokens, "a") == [EOS]  # emitted exactly once
    assert kv.free_count() == kv.num_blocks
