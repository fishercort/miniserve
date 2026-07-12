"""Golden tests on the REAL Qwen2.5-1.5B-Instruct weights — the step-3 exit
criterion. Opt-in (downloads ~3 GB on first run, minutes of CPU):

    MINISERVE_REAL_MODEL=1 uv run pytest tests/test_golden_real.py -s

Rung 1: single-forward parity (argmax must agree; logits close).
Rung 2: fifty greedy tokens, this engine vs HF generate, token-identical.
Rung 3: two concurrent sequences through the full scheduler, token-identical.
Float32 on CPU so both sides share numerics exactly.
"""

import os
from math import ceil

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISERVE_REAL_MODEL") != "1",
    reason="set MINISERVE_REAL_MODEL=1 to run real-weights golden tests",
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "The key idea behind paged attention is"
BS = 16  # block size, per the docs' example


@pytest.fixture(scope="module")
def rig():
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from miniserve.model import PagedModel

    hf = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
    )
    # The checkpoint SHIPS an opinionated generation_config (found the hard
    # way): repetition_penalty=1.1 is a logits PROCESSOR, not a sampling
    # warper, so it applies even under do_sample=False and silently changes
    # the "greedy" reference. Neutralize wholesale: pure argmax, no penalties,
    # no eos stopping — the only thing this engine's greedy implements.
    hf.generation_config = GenerationConfig()
    tok = AutoTokenizer.from_pretrained(MODEL)
    return hf, tok, PagedModel(hf)


def make_kv(num_blocks=64):
    from miniserve.kv_cache import PagedKVCache

    return PagedKVCache(28, num_blocks, BS, 2, 128, dtype=torch.float32)


def prefill_seq(req_id, prompt_ids, kv):
    from miniserve.scheduler import SamplingParams, Sequence, Status

    seq = Sequence(
        req_id, list(prompt_ids), max_tokens=64, sampling=SamplingParams(),
        arrival_time=0.0, seq_no=0,
    )
    seq.status = Status.PREFILL
    assert kv.allocate(req_id, ceil(len(prompt_ids) / BS))
    return seq


def test_rung1_single_forward_parity(rig):
    hf, tok, pm = rig
    ids = tok(PROMPT)["input_ids"]
    kv = make_kv()
    seq = prefill_seq("a", ids, kv)
    mine = pm.forward([seq], kv)["a"]
    with torch.inference_mode():
        ref = hf(torch.tensor([ids])).logits[0, -1]
    assert int(mine.argmax()) == int(ref.argmax())  # the claim that matters
    torch.testing.assert_close(mine, ref, atol=2e-3, rtol=2e-3)  # close, loosely
    print(f"\nrung1: argmax agrees ({int(mine.argmax())}); "
          f"max |Δlogit| = {(mine - ref).abs().max().item():.2e}")


def test_rung2_fifty_greedy_tokens_identical(rig):
    from miniserve.scheduler import Status

    hf, tok, pm = rig
    ids = tok(PROMPT)["input_ids"]

    with torch.inference_mode():
        ref = hf.generate(
            torch.tensor([ids]), max_new_tokens=50, do_sample=False,
            pad_token_id=0,
        )[0, len(ids):].tolist()

    kv = make_kv()
    seq = prefill_seq("a", ids, kv)
    logits = pm.forward([seq], kv)["a"]
    seq.status = Status.DECODE
    mine = []
    for _ in range(50):
        tok_id = int(logits.argmax())
        mine.append(tok_id)
        seq.output_ids.append(tok_id)
        if ceil(seq.total_len() / BS) > len(kv.block_tables["a"]):
            assert kv.append_block("a")
        logits = pm.forward([seq], kv)["a"]

    if mine != ref:
        first_bad = next(
            i for i, (m, r) in enumerate(zip(mine, ref, strict=True)) if m != r
        )
        raise AssertionError(f"diverged at index {first_bad}: {mine} != {ref}")
    print("\nrung2: 50/50 greedy tokens identical.")
    print(f"prompt: {PROMPT!r}")
    print(f"tokens: {mine}")
    print(f"continuation: {tok.decode(mine)!r}")


def test_rung3_scheduler_two_sequences_identical(rig):
    from miniserve.scheduler import Scheduler

    hf, tok, pm = rig
    prompts = {
        "a": tok("The key idea behind paged attention is")["input_ids"],
        "b": tok("Continuous batching improves GPU utilization by")["input_ids"],
    }
    kv = make_kv()
    tokens: list[tuple[str, int]] = []
    sched = Scheduler(kv, pm, max_batch=4, eos_id=-1,
                      on_token=lambda r, t: tokens.append((r, t)))
    for rid, p in prompts.items():
        sched.submit(rid, p, max_tokens=12)
    steps = 0
    while (sched.waiting or sched.running) and steps < 60:
        sched.step()
        kv.check_invariants()
        steps += 1
    assert not sched.running and not sched.waiting

    for rid, p in prompts.items():
        with torch.inference_mode():
            ref = hf.generate(
                torch.tensor([p]), max_new_tokens=12, do_sample=False,
                pad_token_id=0,
            )[0, len(p):].tolist()
        mine = [t for r, t in tokens if r == rid]
        assert mine == ref, f"{rid}: {mine} != {ref}"
    print(f"\nrung3: both concurrent sequences token-identical through the "
          f"scheduler ({steps} steps).")
