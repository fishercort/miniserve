"""Curve B and C probe mechanics, hermetic: cell bookkeeping and realized
context control against a stub model; disk-tier fitting on a real tmp path."""

import torch

from miniserve.bench.decode_probe import K_MEASURED_STEPS, measure_decode_cell
from miniserve.bench.transfer_probe import (
    _fit_latency_bandwidth,
    measure_disk_tiers,
)


class StubModel:
    num_layers = 1
    n_kv_heads = 1
    head_dim = 2
    embed = torch.zeros(50, 2)

    def forward(self, seqs, kv):
        out = {}
        for s in seqs:
            v = torch.zeros(8)
            v[1] = 1.0
            out[s.req_id] = v
        return out


def test_decode_cell_control_and_schema():
    cell = measure_decode_cell(
        StubModel(), batch=2, target_context=64, repeats=2, block_size=4
    )
    # exactly the schema costmodel's decode validation requires
    for key in ("batch", "target_context", "realized_mean_context",
                "decode_ms_median"):
        assert key in cell
    assert cell["batch"] == 2 and cell["target_context"] == 64
    # realized context: prompts at target-k, measured steps 1..k, so the
    # realized mean sits just under target and above target-k
    assert 64 - K_MEASURED_STEPS <= cell["realized_mean_context"] < 64
    assert cell["decode_ms_median"] >= 0
    assert len(cell["decode_ms_all"]) == 2 * K_MEASURED_STEPS


def test_disk_tiers_fit_and_labels(tmp_path):
    tiers = measure_disk_tiers(str(tmp_path), block_bytes=4096, repeats=2)
    assert set(tiers) == {"cpu_to_disk", "disk_to_cpu"}
    for t in tiers.values():
        assert t["latency_ms"] >= 0 and t["bandwidth_gb_s"] > 0
        assert len(t["points"]) == 7
    assert "warm_page_cached" in tiers["disk_to_cpu"]["label"]  # honesty label
    assert "fs=" in tiers["cpu_to_disk"]["label"]  # filesystem captured


def test_latency_bandwidth_fit_recovers_known_line():
    # t = 2ms + bytes * 1e-6 ms/byte  ->  bandwidth 1 GB/s exactly
    points = [{"bytes": b, "ms": 2.0 + b * 1e-6} for b in (1e5, 2e5, 4e5, 8e5)]
    fit = _fit_latency_bandwidth(points)
    assert abs(fit["latency_ms"] - 2.0) < 1e-6
    assert abs(fit["bandwidth_gb_s"] - 1.0) < 1e-6
