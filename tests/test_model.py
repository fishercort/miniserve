"""PagedModel correctness against the HF reference implementation — the three
rungs in miniature, on a tiny random-weight Qwen2 config (float32, CPU,
hermetic: no downloads). Rung 1 catches loading/RoPE/GQA; rung 2 catches the
decode loop; rung 3 catches paging and the gather. The real-weights golden
versions live in test_golden_real.py (opt-in)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from miniserve.kv_cache import PagedKVCache  # noqa: E402
from miniserve.model import PagedModel  # noqa: E402
from miniserve.scheduler import SamplingParams, Scheduler, Sequence, Status  # noqa: E402

BS = 4  # block size
VOCAB = 128


def tiny_model():
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        use_sliding_window=False,
        attention_dropout=0.0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    hf = Qwen2ForCausalLM(cfg).eval()
    return hf, PagedModel(hf)


def make_kv(num_blocks=16):
    # 2 layers, 2 KV heads, head_dim 16, float32 for tight comparison.
    return PagedKVCache(
        2, num_blocks, BS, 2, 16, dtype=torch.float32, poison_on_free=True
    )


def prefill_seq(req_id, prompt, kv):
    seq = Sequence(
        req_id,
        list(prompt),
        max_tokens=64,
        sampling=SamplingParams(),
        arrival_time=0.0,
        seq_no=0,
    )
    seq.status = Status.PREFILL
    assert kv.allocate(req_id, -(-len(prompt) // BS))
    return seq


def hf_last_logits(hf, ids):
    with torch.inference_mode():
        return hf(torch.tensor([ids])).logits[0, -1]


def test_rung1_prefill_logits_match_reference():
    hf, pm = tiny_model()
    kv = make_kv()
    prompt = list(range(3, 12))  # length 9: two full blocks + one partial
    seq = prefill_seq("a", prompt, kv)
    mine = pm.forward([seq], kv)["a"]
    ref = hf_last_logits(hf, prompt)
    torch.testing.assert_close(mine, ref, atol=1e-4, rtol=1e-4)
    assert int(mine.argmax()) == int(ref.argmax())


def test_rung2_stepwise_decode_matches_full_context():
    """Incremental paged decode == full-context reference at every length."""
    hf, pm = tiny_model()
    kv = make_kv()
    prompt = [5, 17, 90, 41, 7]
    seq = prefill_seq("a", prompt, kv)
    logits = pm.forward([seq], kv)["a"]
    seq.status = Status.DECODE

    for _ in range(8):
        ref = hf_last_logits(hf, seq.prompt_ids + seq.output_ids)
        torch.testing.assert_close(logits, ref, atol=1e-4, rtol=1e-4)
        tok = int(ref.argmax())  # follow the reference's greedy path
        seq.output_ids.append(tok)
        if -(-seq.total_len() // BS) > len(kv.block_tables["a"]):
            assert kv.append_block("a")
        logits = pm.forward([seq], kv)["a"]


def test_rung3a_fragmented_blocks_same_logits():
    """Correctness is independent of WHICH physical blocks a sequence holds."""
    hf, pm = tiny_model()
    prompt = list(range(10, 19))

    kv_clean = make_kv()
    seq = prefill_seq("a", prompt, kv_clean)
    clean = pm.forward([seq], kv_clean)["a"]

    kv_frag = make_kv()
    kv_frag.allocate("junk1", 3)  # scramble the free list so "a" gets
    kv_frag.allocate("junk2", 2)  # non-contiguous, non-zero-based blocks
    kv_frag.free("junk1")
    seq2 = prefill_seq("a", prompt, kv_frag)
    assert kv_frag.block_tables["a"] != list(range(len(kv_frag.block_tables["a"])))
    frag = pm.forward([seq2], kv_frag)["a"]

    torch.testing.assert_close(frag, clean, atol=1e-5, rtol=1e-5)


def test_resumed_prefill_equals_fresh_prefill():
    """The hairiest interaction in the file, pinned directly: RoPE positions ×
    causal mask × preemption. A resume (prefill over prompt + previously
    generated tokens) must produce logits identical to a fresh prefill over the
    same content — and both must match the HF full-context reference.

    PAIR COVERAGE: this test builds the resumed state BY HAND to isolate the
    model layer; that the scheduler's real preempt→readmit path produces
    exactly this state is proven by
    test_scheduler.test_resume_not_restart_byte_identical (behavior 15).
    Coverage is complete only as a pair — do not delete either believing the
    other covers it."""
    hf, pm = tiny_model()
    prompt, generated = [5, 17, 90, 41, 7], [22, 9, 104]

    kv_fresh = make_kv()
    fresh = prefill_seq("a", prompt + generated, kv_fresh)
    fresh_logits = pm.forward([fresh], kv_fresh)["a"]

    kv_resume = make_kv()
    resumed = Sequence(
        "a", list(prompt), max_tokens=64, sampling=SamplingParams(),
        arrival_time=0.0, seq_no=0,
    )
    resumed.output_ids = list(generated)  # its first life's tokens
    resumed.preempted_count = 1
    resumed.status = Status.PREFILL  # re-admission re-prefills ALL content
    assert kv_resume.allocate("a", -(-resumed.total_len() // BS))
    resume_logits = pm.forward([resumed], kv_resume)["a"]

    torch.testing.assert_close(resume_logits, fresh_logits, atol=1e-6, rtol=1e-6)
    ref = hf_last_logits(hf, prompt + generated)
    torch.testing.assert_close(resume_logits, ref, atol=1e-4, rtol=1e-4)


def test_rung3_end_to_end_scheduler_matches_hf_generate():
    """The whole engine (scheduler + paged model), two concurrent sequences,
    greedy — token streams must equal HF generate() run per-prompt."""
    hf, pm = tiny_model()
    kv = make_kv()
    tokens: list[tuple[str, int]] = []
    sched = Scheduler(
        kv,
        pm,
        max_batch=4,
        eos_id=-1,  # unreachable: run to max_tokens
        on_token=lambda r, t: tokens.append((r, t)),
    )
    prompts = {"a": [5, 17, 90, 41], "b": [3, 3, 99, 12, 61, 8]}
    for rid, p in prompts.items():
        sched.submit(rid, p, max_tokens=6)
    steps = 0
    while (sched.waiting or sched.running) and steps < 50:
        sched.step()
        kv.check_invariants()
        steps += 1
    assert not sched.running and not sched.waiting

    for rid, p in prompts.items():
        with torch.inference_mode():
            ref = hf.generate(
                torch.tensor([p]),
                max_new_tokens=6,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )[0, len(p):].tolist()
        mine = [t for r, t in tokens if r == rid]
        assert mine == ref, f"{rid}: {mine} != {ref}"
