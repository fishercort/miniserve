"""HTTP API and token streaming. See docs/phase1-engine.md (API surface).

Handler threads never touch the scheduler: they submit through the engine
handle and block on their own token queue. Tokenization happens here, in the
handler thread; only ids cross into the engine.
"""

import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from miniserve.engine import FINISH, Engine, EngineStopped
from miniserve.scheduler import CapacityError, SamplingParams

DEFAULT_MAX_TOKENS = 128


def make_server(
    engine: Engine,
    host: str = "127.0.0.1",
    port: int = 8000,
    tokenizer=None,
    model_name: str = "",
    submit_timeout_s: float = 30.0,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep test output quiet
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            sched = engine.scheduler
            kv = sched.kv
            if self.path == "/health":
                self._json(
                    200,
                    {
                        "status": "ok",
                        "model": model_name,
                        "kv_blocks_total": kv.num_blocks,
                        "kv_blocks_free": kv.free_count(),
                        "max_tokens_default": DEFAULT_MAX_TOKENS,
                    },
                )
            elif self.path == "/metrics":
                self._json(
                    200,
                    {
                        "num_running": len(sched.running),
                        "num_waiting": len(sched.waiting),
                        "kv_utilization_pct": 100.0 * kv.used_count() / kv.num_blocks,
                        "preemptions_total": sched.preemptions_total,
                        "requests_finished": len(sched.metrics.requests),
                        "steps_total": len(sched.metrics.steps),
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/generate":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid JSON"})
                return

            # Tokenize here, in the handler thread. Only ids cross the queue.
            if "prompt_ids" in body:
                prompt_ids = body["prompt_ids"]
                if not isinstance(prompt_ids, list) or not all(
                    isinstance(t, int) for t in prompt_ids
                ):
                    self._json(400, {"error": "prompt_ids must be a list of ints"})
                    return
            elif "prompt" in body and tokenizer is not None:
                prompt_ids = tokenizer(body["prompt"])["input_ids"]
            else:
                self._json(
                    400,
                    {"error": "provide prompt_ids, or prompt with a tokenizer configured"},
                )
                return

            sampling = SamplingParams(
                temperature=float(body.get("temperature", 0.0)),
                top_p=float(body.get("top_p", 1.0)),
            )
            # v1 is greedy-only. Accepting and silently ignoring sampling
            # params is the failure the golden tests caught in generate();
            # refuse instead of pretending.
            if sampling.temperature != 0.0 or sampling.top_p != 1.0:
                self._json(
                    400,
                    {"error": "v1 is greedy-only: temperature must be 0 and top_p 1"},
                )
                return
            req_id = uuid.uuid4().hex[:12]
            fut, stream = engine.submit(
                req_id, prompt_ids, int(body.get("max_tokens", DEFAULT_MAX_TOKENS)), sampling
            )
            try:
                fut.result(timeout=submit_timeout_s)
            except (CapacityError, ValueError) as e:
                self._json(400, {"error": str(e)})  # exactly one client-visible error
                return
            except EngineStopped as e:
                self._json(503, {"error": str(e)})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            # Dead-client policy (v1): if the client hangs up mid-stream, stop
            # writing and stop draining, but the sequence completes server-side
            # and frees its blocks at natural retirement. Cancel-on-disconnect
            # is a flagged extension; it is a new scheduler operation.
            try:
                while True:
                    kind, payload = stream.get()
                    if kind == FINISH:
                        event = {
                            "done": True,
                            "aborted": payload.aborted,
                            "output_tokens": payload.output_tokens,
                            "ttft_ms": payload.ttft_ms,
                            "total_ms": payload.total_ms,
                            "throughput_tok_s": payload.throughput_tok_s,
                        }
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                        self.wfile.flush()
                        return
                    self.wfile.write(
                        f"data: {json.dumps({'token_id': payload})}\n\n".encode()
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return  # client left; generation continues to retirement

    return ThreadingHTTPServer((host, port), Handler)
