"""Ties model + cache + scheduler together and runs the loop. See docs/phase1-engine.md.

Concurrency model (the decision line in Named failure modes): the engine thread
exclusively owns the scheduler and KV cache. Server threads reach it only
through message passing: a submission queue in, per-request token queues out,
futures for rejections. The allocator's single-threaded assumption holds
because no other thread can reach the scheduler.

Only ids cross the queue, never text: tokenization belongs to the caller's
thread.
"""

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass

from miniserve.scheduler import SamplingParams, Scheduler, Sequence

TOKEN = "token"
FINISH = "finish"


class EngineStopped(RuntimeError):
    """The engine is shutting down or stopped; the request was not admitted."""


@dataclass
class FinishSummary:
    req_id: str
    output_tokens: int
    ttft_ms: float | None
    total_ms: float | None
    throughput_tok_s: float | None
    aborted: bool = False  # True when the engine went down mid-stream


class Engine:
    """Runs the scheduler loop on a dedicated thread.

    Two drive modes:
    - ``start()`` / ``stop()``: the real thread, for serving and integration
      tests only.
    - ``step_once()``: drain + one scheduler step on the calling thread, for
      deterministic tests. Same code path, no threads.

    Nothing hangs, period: shutdown and engine-thread crashes both resolve
    every pending future and deliver a terminal sentinel to every live stream.
    Refusal paths sentinel too, so the stream contract is unconditional:
    every stream submit() ever returns terminates with exactly one FINISH.

    Engine takes ownership of scheduler.on_token and scheduler.on_finish; any
    callbacks set before construction are replaced.
    """

    def __init__(self, scheduler: Scheduler, idle_wait_s: float = 0.01) -> None:
        self.scheduler = scheduler
        scheduler.on_token = self._on_token
        scheduler.on_finish = self._on_finish
        self._submissions: queue.Queue = queue.Queue()
        # Ownership rule: _streams is written and read by the engine thread
        # ONLY (_admit registers, _on_finish/_finalize_shutdown pop). Server
        # threads hold only the queue object submit() handed back to them.
        self._streams: dict[str, queue.Queue] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.idle_wait_s = idle_wait_s
        # Guards the stop-check-plus-put pair in submit() against stop()
        # setting the event between them. Without it, a put landing after
        # _finalize_shutdown's final drain leaves a future that never
        # resolves: a client hung forever. The lock guards the doorway,
        # not the scheduler; message passing stays the only channel.
        self._doorway = threading.Lock()

    # -- handle API: the only surface server threads may touch ----------------

    def submit(
        self,
        req_id: str,
        prompt_ids: list[int],
        max_tokens: int,
        sampling: SamplingParams | None = None,
    ) -> tuple[Future, queue.Queue]:
        """Queue a request. Returns (future, token stream).

        The future resolves to the req_id on admission to the waiting queue,
        or raises; CapacityError/ValueError cross the thread boundary as the
        same exception type and message. The stream yields ("token", id)
        tuples then exactly one ("finish", FinishSummary).

        The stream's sentinel only guarantees termination; it does not carry
        the cause. A rejected-at-the-door request and a crash-aborted request
        that produced nothing both sentinel with aborted=True and zero tokens.
        Check the future to tell them apart.
        """
        fut: Future = Future()
        stream: queue.Queue = queue.Queue()
        with self._doorway:
            if self._stop.is_set():
                fut.set_exception(EngineStopped("engine stopped"))
                self._refusal_sentinel(req_id, stream)
                return fut, stream
            self._submissions.put(
                (req_id, list(prompt_ids), max_tokens, sampling, fut, stream)
            )
        return fut, stream

    @staticmethod
    def _refusal_sentinel(req_id: str, stream: queue.Queue) -> None:
        """Refused requests still terminate their stream: the contract is
        unconditional, not a tendency."""
        stream.put(
            (
                FINISH,
                FinishSummary(
                    req_id=req_id,
                    output_tokens=0,
                    ttft_ms=None,
                    total_ms=None,
                    throughput_tok_s=None,
                    aborted=True,
                ),
            )
        )

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.run, name="miniserve-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        with self._doorway:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("engine thread did not stop")

    # -- engine-thread internals -----------------------------------------------

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                has_work = bool(self.scheduler.waiting or self.scheduler.running)
                if not has_work and self._submissions.empty():
                    # Idle: block briefly for the next submission instead of
                    # spinning. Admit the blocking get's result before
                    # step_once's drain so queue FIFO order is preserved.
                    try:
                        self._admit(self._submissions.get(timeout=self.idle_wait_s))
                    except queue.Empty:
                        continue
                self.step_once()
        except BaseException:
            # A crash must not strand clients: refuse new submissions, then
            # fall through to the sentinel drain before re-raising.
            with self._doorway:
                self._stop.set()
            raise
        finally:
            self._finalize_shutdown()

    def step_once(self) -> None:
        """Drain submissions, then one scheduler step. Callable directly from
        tests for deterministic, thread-free drives."""
        while True:
            try:
                self._admit(self._submissions.get_nowait())
            except queue.Empty:
                break
        self.scheduler.step()

    def _admit(self, item) -> None:
        req_id, prompt_ids, max_tokens, sampling, fut, stream = item
        try:
            self.scheduler.submit(req_id, prompt_ids, max_tokens, sampling)
        except ValueError as e:  # CapacityError included: crosses boundary intact
            fut.set_exception(e)
            self._refusal_sentinel(req_id, stream)
            return
        self._streams[req_id] = stream
        fut.set_result(req_id)

    def _on_token(self, req_id: str, token: int) -> None:
        self._streams[req_id].put((TOKEN, token))

    def _on_finish(self, seq: Sequence) -> None:
        stream = self._streams.pop(seq.req_id)
        # is_finished requires non-empty output_ids, so first_token_time
        # should always be set here; guarded anyway, house style.
        ttft_ms = (
            None
            if seq.first_token_time is None
            else (seq.first_token_time - seq.arrival_time) * 1000.0
        )
        total_s = (
            None
            if seq.completion_time is None
            else seq.completion_time - seq.arrival_time
        )
        n = len(seq.output_ids)
        stream.put(
            (
                FINISH,
                FinishSummary(
                    req_id=seq.req_id,
                    output_tokens=n,
                    ttft_ms=ttft_ms,
                    total_ms=None if total_s is None else total_s * 1000.0,
                    throughput_tok_s=n / total_s if total_s else None,
                ),
            )
        )

    def _finalize_shutdown(self) -> None:
        """Nothing hangs: pending futures get EngineStopped, live streams get
        an aborted finish sentinel. The doorway lock guarantees no submission
        can land after this drain.

        Must remain READ-ONLY against scheduler state: on the crash path this
        runs against a half-stepped scheduler (possibly mid-preemption, block
        tables in flux). Scans of running/waiting are safe; any mutation, KV
        cleanup included, is not."""
        while True:
            try:
                item = self._submissions.get_nowait()
            except queue.Empty:
                break
            item[4].set_exception(EngineStopped("engine stopped"))
        for req_id, stream in self._streams.items():
            seq = next(
                (s for s in self.scheduler.running if s.req_id == req_id), None
            ) or next((s for s in self.scheduler.waiting if s.req_id == req_id), None)
            stream.put(
                (
                    FINISH,
                    FinishSummary(
                        req_id=req_id,
                        output_tokens=len(seq.output_ids) if seq else 0,
                        ttft_ms=None,
                        total_ms=None,
                        throughput_tok_s=None,
                        aborted=True,
                    ),
                )
            )
        self._streams.clear()
