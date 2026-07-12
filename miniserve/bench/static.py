"""Static batching baseline: wait for a full batch, run it to completion, take
the next batch. Differs from continuous batching by exactly one thing, the
admission-open condition, so the comparison isolates the scheduling policy.
"""

from miniserve.scheduler import Scheduler


class StaticScheduler(Scheduler):
    """Admission opens only when the batch is empty AND a full batch is
    waiting (or flush says no more arrivals are coming, so partial batches
    drain instead of waiting forever)."""

    def __init__(self, *args, max_wait_s: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Bench-only, GIL-atomic bool store from the driver thread; same
        # class of exception as the metrics carve-out (no structure mutation).
        self.flush = False
        # Optional partial-batch timer: open when the oldest waiting request
        # has waited this long, so low arrival rates do not stall forever.
        self.max_wait_s = max_wait_s

    def _admission_open(self) -> bool:
        if self.running:
            return False  # the current batch runs to completion first
        if not self.waiting:
            return False
        if len(self.waiting) >= self.max_batch or self.flush:
            return True
        if self.max_wait_s is not None:
            return self.clock() - self.waiting[0].arrival_time >= self.max_wait_s
        return False
