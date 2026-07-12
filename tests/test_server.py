"""HTTP server tests over a real socket with the fake model: SSE streaming,
capacity rejection as exactly one 400, health and metrics endpoints."""

import json
import threading

import pytest
import torch

from miniserve.engine import Engine
from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import Scheduler
from miniserve.server import make_server

BS = 4


class AnyModel:
    """Request-id independent: token n = (n % 5) + 1, never EOS."""

    def forward(self, seqs, kv):
        out = {}
        for s in seqs:
            v = torch.zeros(16)
            v[(len(s.output_ids) % 5) + 1] = 1.0
            out[s.req_id] = v
        return out


@pytest.fixture
def rig():
    kv = PagedKVCache(1, 16, BS, 1, 2, poison_on_free=True)
    sched = Scheduler(kv, AnyModel(), max_batch=4, eos_id=0)
    engine = Engine(sched, idle_wait_s=0.005)
    server = make_server(engine, port=0, model_name="fake-tiny")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    engine.start()
    host, port = server.server_address[:2]
    yield host, port
    server.shutdown()
    engine.stop()


def post_generate(host, port, body):
    from http.client import HTTPConnection

    conn = HTTPConnection(host, port, timeout=10)
    conn.request(
        "POST", "/generate", json.dumps(body), {"Content-Type": "application/json"}
    )
    return conn, conn.getresponse()


def get_json(host, port, path):
    from http.client import HTTPConnection

    conn = HTTPConnection(host, port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()
    return resp.status, body


def read_sse(resp):
    events = []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith(b"data: "):
            events.append(json.loads(line[len(b"data: "):]))
            if events[-1].get("done"):
                break
    return events


def test_generate_streams_tokens_then_summary(rig):
    host, port = rig
    conn, resp = post_generate(
        host, port, {"prompt_ids": [7, 7, 7], "max_tokens": 3}
    )
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "text/event-stream"
    events = read_sse(resp)
    conn.close()
    tokens = [e["token_id"] for e in events if "token_id" in e]
    assert tokens == [1, 2, 3]  # AnyModel's position sequence
    done = events[-1]
    assert done["done"] is True and done["aborted"] is False
    assert done["output_tokens"] == 3
    assert done["ttft_ms"] > 0 and done["throughput_tok_s"] > 0


def test_capacity_rejection_is_one_400(rig):
    host, port = rig
    conn, resp = post_generate(
        host, port, {"prompt_ids": [7] * 4, "max_tokens": 100_000}
    )
    assert resp.status == 400
    body = json.loads(resp.read())
    conn.close()
    assert "worst case" in body["error"]  # same message that CapacityError raised


def test_missing_prompt_is_400(rig):
    host, port = rig
    conn, resp = post_generate(host, port, {"max_tokens": 4})
    assert resp.status == 400
    assert "prompt" in json.loads(resp.read())["error"]
    conn.close()


def test_health(rig):
    host, port = rig
    status, body = get_json(host, port, "/health")
    assert status == 200
    assert body["status"] == "ok" and body["model"] == "fake-tiny"
    assert body["kv_blocks_total"] == 16


def test_disconnect_mid_stream_completes_server_side():
    """Client hangs up mid-stream: the handler stops, the server survives, and
    the sequence runs to natural retirement (v1's documented no-cancel policy)."""
    import time

    class SlowModel(AnyModel):
        def forward(self, seqs, kv):
            time.sleep(0.003)
            return super().forward(seqs, kv)

    kv = PagedKVCache(1, 16, BS, 1, 2, poison_on_free=True)
    sched = Scheduler(kv, SlowModel(), max_batch=4, eos_id=0)
    engine = Engine(sched, idle_wait_s=0.005)
    server = make_server(engine, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    engine.start()
    host, port = server.server_address[:2]
    try:
        conn, resp = post_generate(host, port, {"prompt_ids": [7, 7], "max_tokens": 20})
        line = resp.readline()
        assert line.startswith(b"data: ")  # streaming began
        conn.close()  # client leaves mid-stream

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status, body = get_json(host, port, "/metrics")
            if body["requests_finished"] >= 1:
                break
            time.sleep(0.02)
        assert body["requests_finished"] >= 1  # ran to retirement, server alive
        assert kv.free_count() == kv.num_blocks  # blocks freed naturally
    finally:
        server.shutdown()
        engine.stop()


def test_sampling_params_rejected_not_ignored(rig):
    """v1 is greedy-only and says so: accept-and-ignore is the generate() sin."""
    host, port = rig
    conn, resp = post_generate(
        host, port, {"prompt_ids": [7, 7], "max_tokens": 2, "temperature": 0.9}
    )
    assert resp.status == 400
    assert "greedy-only" in json.loads(resp.read())["error"]
    conn.close()


def test_prompt_ids_validated_at_boundary(rig):
    host, port = rig
    conn, resp = post_generate(host, port, {"prompt_ids": ["a", 1.5], "max_tokens": 2})
    assert resp.status == 400
    assert "list of ints" in json.loads(resp.read())["error"]
    conn.close()


def test_metrics_after_traffic(rig):
    host, port = rig
    conn, resp = post_generate(host, port, {"prompt_ids": [7, 7], "max_tokens": 2})
    read_sse(resp)
    conn.close()
    status, body = get_json(host, port, "/metrics")
    assert status == 200
    assert body["requests_finished"] >= 1
    assert body["steps_total"] >= 1
    assert 0.0 <= body["kv_utilization_pct"] <= 100.0
