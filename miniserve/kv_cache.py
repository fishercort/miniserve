"""PagedKVCache: physical tensors, free list, block tables, and the Evictor seam.

See docs/phase1-engine.md (data structure 1). Logical token position ``i`` for a
sequence maps to physical block ``block_table[i // block_size]``, offset
``i % block_size``.
"""

from collections import deque
from dataclasses import dataclass

import torch

ReqId = str
BlockId = int


class InvariantViolation(RuntimeError):
    """Block conservation broken — a bug in the allocator or its caller.

    A real exception, not ``assert``: the invariant check must survive
    ``python -O``, or the executable correctness argument silently vanishes.
    """


@dataclass
class BlockMeta:
    """Reserved for Phase 3. Fields present and unused in v1."""

    last_access: float = 0.0
    access_count: int = 0
    session_id: str | None = None
    lifecycle_class: str = "unknown"  # durable | ephemeral | unknown
    recompute_cost: float = 0.0  # filled from the Phase 2 cost model


class PagedKVCache:
    """Preallocated paged KV storage with a free list and per-sequence block tables.

    Physical storage: one (K, V) tensor pair per layer, each shaped
    ``[num_blocks, block_size, num_kv_heads, head_dim]``. All blocks are
    preallocated up front; allocation and freeing only move block ids between
    the free list and per-request block tables — no tensor memory moves.

    Freed blocks are NOT zeroed (the malloc-doesn't-memset contract): readers
    rely on write-before-read discipline, enforced by the scheduler's length
    accounting. ``poison_on_free=True`` fills released blocks with NaN so any
    read-after-free detonates in the logits instead of masquerading as
    sampling weirdness — on in tests, off in serving.

    Concurrency: guard-then-mutate throughout, safe only because the engine
    loop is single-threaded. A multi-threaded scheduler would need locking
    around every mutator or the allocate() comprehension can leak blocks.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        poison_on_free: bool = False,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.poison_on_free = poison_on_free
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        # FIFO reuse: deterministic, and oldest-freed-first is the trivial
        # retired-LRU — the seam Phase 3's reusable-block pool extends.
        self.free_blocks: deque[BlockId] = deque(range(num_blocks))
        self.block_tables: dict[ReqId, list[BlockId]] = {}
        self.meta: dict[BlockId, BlockMeta] = {}  # reserved for Phase 3, unused in v1

    def free_count(self) -> int:
        return len(self.free_blocks)

    def used_count(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def allocate(self, req_id: ReqId, num_blocks: int) -> bool:
        """Claim ``num_blocks`` for a new request. All-or-nothing.

        Returns False (no state change) if not enough free blocks. A duplicate
        allocation or a zero-block request is a scheduler bug, not a
        recoverable condition — raises.
        """
        if req_id in self.block_tables:
            raise ValueError(f"request {req_id!r} already has an allocation")
        if num_blocks < 1:
            raise ValueError(f"zero-block allocation for {req_id!r} is a scheduler bug")
        if len(self.free_blocks) < num_blocks:
            return False
        self.block_tables[req_id] = [
            self.free_blocks.popleft() for _ in range(num_blocks)
        ]
        return True

    def append_block(self, req_id: ReqId) -> bool:
        """Grow an existing request by one block. False if no block is free.

        Unknown ``req_id`` raises KeyError — the scheduler must only grow
        sequences it has admitted.
        """
        table = self.block_tables[req_id]
        if not self.free_blocks:
            return False
        table.append(self.free_blocks.popleft())
        return True

    def free(self, req_id: ReqId) -> None:
        """Return a request's blocks to the free list (retire or preempt path).

        Unknown ``req_id`` raises KeyError — a double-free is a scheduler bug.

        Phase 3: must also clear ``self.meta`` for the released ids — stale
        lifecycle metadata across block owners quietly mis-prices evictions,
        a nastier bug family than stale KV (which at least poisons loudly).
        """
        blocks = self.block_tables.pop(req_id)
        if self.poison_on_free:
            for k, v in zip(self.k_cache, self.v_cache, strict=True):
                k[blocks] = float("nan")
                v[blocks] = float("nan")
        self.free_blocks.extend(blocks)

    def check_invariants(self) -> None:
        """Block conservation, executable: free + held == total, no id twice.

        Test-support: the suite calls this after every mutation sequence, so
        the conservation argument is re-proven by CI instead of living in a
        review that was once made.
        """
        held = [b for table in self.block_tables.values() for b in table]
        all_ids = list(self.free_blocks) + held
        if len(all_ids) != self.num_blocks:
            raise InvariantViolation(
                f"block leak or double-count: {len(all_ids)} ids for "
                f"{self.num_blocks} blocks"
            )
        if len(set(all_ids)) != self.num_blocks:
            raise InvariantViolation("duplicate block id across free list and tables")
