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


def test_chunk_overhead_fit_recovers_coefficient():
    from miniserve.bench.transfer_probe import fit_chunk_overhead

    # t(N) = 5ms + 0.2ms per chunk; at 56 regions: (5+11.2)/(5+0.2) = ~3.115x
    points = [{"n_chunks": n, "ms": 5.0 + 0.2 * n} for n in (1, 2, 4, 8, 16, 56)]
    fit = fit_chunk_overhead(points, n_regions=56)
    assert abs(fit["per_chunk_overhead_ms"] - 0.2) < 1e-9
    assert abs(fit["base_ms"] - 5.0) < 1e-9
    assert abs(fit["projected_slowdown"]["factor"] - (16.2 / 5.2)) < 1e-9


def test_activation_grid_shape():
    from miniserve.bench.activation_probe import plan_cells

    cells = plan_cells()
    kinds = {c["kind"] for c in cells}
    assert kinds == {"decode", "mixed"}
    assert sum(1 for c in cells if c["kind"] == "decode") == 9  # 3 batches x 3 ctx
    assert all("prefill_len" in c for c in cells if c["kind"] == "mixed")


def test_spot_validation_contract_logic():
    from miniserve.bench.spot_validate import evaluate_contract, select_points

    def point(tokens, factor_r, factor_m):
        return {
            "tokens": tokens,
            "predicted_recompute_ms": 100.0,
            "measured_recompute_ms": 100.0 * factor_r,
            "predicted_migrate_ms": 50.0,
            "measured_migrate_ms": 50.0 * factor_m,
        }

    # two points inside 25%, one wildly out: contract passes at 2 of 3
    verdict = evaluate_contract([point(10, 1.1, 0.9), point(20, 1.2, 1.2),
                                 point(40, 2.0, 1.0)])
    assert verdict["n_passing"] == 2 and verdict["contract_passes"] is True
    # one good point only: contract fails
    verdict = evaluate_contract([point(10, 1.0, 1.0), point(20, 1.4, 1.0),
                                 point(40, 1.0, 1.4)])
    assert verdict["contract_passes"] is False

    band = {"lo": 15, "hi": 28, "crossover": 20, "hi_clamped": False}
    assert select_points(band) == [15, 20, 28]
    clamped = {"lo": 200, "hi": 1 << 20, "crossover": 800, "hi_clamped": True}
    assert select_points(clamped) == [200, 800, 1600]  # 2x crossover substitutes
