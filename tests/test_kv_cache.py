"""PagedKVCache unit tests: free-list accounting, all-or-nothing allocation,
growth, retirement, poison-on-free, and the executable conservation invariant
(checked after every test's mutation sequence via fixture teardown)."""

import pytest
import torch

from miniserve.kv_cache import InvariantViolation, PagedKVCache

# Tiny cache: 2 layers, 8 blocks of 4 tokens, 2 KV heads, head_dim 8.
NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM = 2, 8, 4, 2, 8


def make_cache(poison_on_free: bool = True) -> PagedKVCache:
    return PagedKVCache(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM,
        poison_on_free=poison_on_free,
    )


@pytest.fixture
def kv():
    cache = make_cache()
    yield cache
    # Every test's final state must conserve blocks — including tests whose
    # last action was a failure path or a raise.
    cache.check_invariants()


def test_init_shapes_and_counts(kv):
    assert kv.free_count() == NUM_BLOCKS
    assert kv.used_count() == 0
    assert len(kv.k_cache) == len(kv.v_cache) == NUM_LAYERS
    for t in (*kv.k_cache, *kv.v_cache):
        assert t.shape == (NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM)
        assert t.dtype == torch.bfloat16


def test_allocate_claims_blocks(kv):
    assert kv.allocate("a", 3) is True
    assert kv.free_count() == NUM_BLOCKS - 3
    assert kv.used_count() == 3
    assert len(kv.block_tables["a"]) == 3


def test_allocate_all_or_nothing(kv):
    assert kv.allocate("a", NUM_BLOCKS - 1) is True
    free_before = kv.free_count()
    assert kv.allocate("b", 2) is False  # only 1 free
    assert kv.free_count() == free_before  # no partial claim
    assert "b" not in kv.block_tables


def test_allocate_duplicate_req_raises(kv):
    kv.allocate("a", 1)
    with pytest.raises(ValueError, match="already has an allocation"):
        kv.allocate("a", 1)


def test_allocate_zero_blocks_raises(kv):
    with pytest.raises(ValueError, match="zero-block"):
        kv.allocate("a", 0)
    assert "a" not in kv.block_tables


def test_allocations_are_disjoint(kv):
    kv.allocate("a", 3)
    kv.allocate("b", 3)
    kv.allocate("c", 2)
    all_blocks = kv.block_tables["a"] + kv.block_tables["b"] + kv.block_tables["c"]
    assert len(all_blocks) == len(set(all_blocks)) == NUM_BLOCKS
    assert kv.free_count() == 0


def test_append_block_grows_by_one(kv):
    kv.allocate("a", 1)
    assert kv.append_block("a") is True
    assert len(kv.block_tables["a"]) == 2
    assert kv.free_count() == NUM_BLOCKS - 2


def test_append_block_exhausted_is_clean_false(kv):
    kv.allocate("a", NUM_BLOCKS)
    table_before = list(kv.block_tables["a"])
    assert kv.append_block("a") is False
    assert kv.block_tables["a"] == table_before  # no state change


def test_append_block_unknown_req_raises(kv):
    with pytest.raises(KeyError):
        kv.append_block("ghost")


def test_free_returns_blocks_for_reuse(kv):
    kv.allocate("a", NUM_BLOCKS)
    kv.free("a")
    assert kv.free_count() == NUM_BLOCKS
    assert "a" not in kv.block_tables
    assert kv.allocate("b", NUM_BLOCKS) is True  # every block reusable


def test_double_free_raises(kv):
    kv.allocate("a", 1)
    kv.free("a")
    with pytest.raises(KeyError):
        kv.free("a")


def test_block_table_order_is_stable(kv):
    """The scheduler maps logical position i -> block_table[i // block_size];
    the table must preserve allocation-then-append order."""
    kv.allocate("a", 2)
    first_two = list(kv.block_tables["a"])
    kv.append_block("a")
    assert kv.block_tables["a"][:2] == first_two
    assert len(kv.block_tables["a"]) == 3


def test_preempt_then_readmit_conserves_blocks(kv):
    """Preemption is free() on a running sequence followed by later
    re-admission; the allocator must not care that the free was mid-flight."""
    kv.allocate("a", 3)
    kv.append_block("a")
    kv.free("a")  # preempted
    assert kv.free_count() == NUM_BLOCKS
    assert kv.allocate("a", 5) is True  # re-admitted, bigger footprint
    assert kv.free_count() == NUM_BLOCKS - 5


def test_poison_on_free_detonates_stale_reads(kv):
    """A freed block's K/V slots become NaN, so a read-after-free shows up in
    the logits immediately instead of as plausible stale values."""
    kv.allocate("a", 2)
    block = kv.block_tables["a"][0]
    kv.k_cache[0][block] = 1.0
    kv.v_cache[1][block] = 2.0
    kv.free("a")
    assert torch.isnan(kv.k_cache[0][block]).all()
    assert torch.isnan(kv.v_cache[1][block]).all()


def test_no_poison_when_flag_off():
    cache = make_cache(poison_on_free=False)
    cache.allocate("a", 1)
    block = cache.block_tables["a"][0]
    cache.k_cache[0][block] = 1.0
    cache.free("a")
    assert (cache.k_cache[0][block] == 1.0).all()  # stale by contract, not zeroed
    cache.check_invariants()


def test_check_invariants_is_not_vacuous():
    """Corrupt the state deliberately; the checker must catch both failure
    modes (count mismatch and duplicate id)."""
    dup = make_cache()
    dup.free_blocks.append(0)  # duplicate: block 0 is free twice
    with pytest.raises(InvariantViolation, match="leak or double-count"):
        dup.check_invariants()

    leak = make_cache()
    leak.free_blocks.popleft()  # leak: block 0 is nowhere
    with pytest.raises(InvariantViolation):
        leak.check_invariants()


def test_full_lifecycle_accounting(kv):
    """Interleaved allocate/grow/free never leaks or double-counts a block."""
    kv.allocate("a", 2)
    kv.allocate("b", 3)
    kv.append_block("a")
    kv.free("b")
    kv.allocate("c", 4)
    assert kv.free_count() + kv.used_count() == NUM_BLOCKS
    held = kv.block_tables["a"] + kv.block_tables["c"]
    assert len(held) == len(set(held)) == kv.used_count() == 7
