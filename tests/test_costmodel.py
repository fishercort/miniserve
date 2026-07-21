"""Cost-model contract tests: the config is emittable only around an
acceptable fit, validates its inputs, and the crossover plus its band are
computable from the config ALONE with hand-checkable numbers."""

import pytest

from miniserve.bench.costmodel import (
    crossover_band,
    crossover_tokens,
    emit_config,
    migrate_ms,
    recompute_ms,
    round_trip_ms,
)

# dims chosen so kv_bytes_per_token = 10*2*5*10*2 = 2000 bytes exactly
DIMS = {"num_layers": 10, "num_kv_heads": 5, "head_dim": 10, "dtype_bytes": 2}
HARDWARE = {
    "gpu": {"name": "stub", "form_factor": "SXM", "driver": "0.0", "vram_gb": 80},
    "filesystem": "apfs",
}
PROVENANCE = {"timestamp": "t", "platform": "test", "commit": "deadbeef"}


def linear_fit(a_ms: float, b_ms: float) -> dict:
    return {
        "preferred": "linear",
        "linear": {"a_ms": a_ms, "b_ms_per_token": b_ms, "mean_rel_err": 0.0},
        "quadratic": {
            "a_ms": 0.0, "b_ms_per_token": 0.0, "c_ms_per_token2": 0.0,
            "mean_rel_err": 1.0,
        },
        "fit_gate_rel_err": 0.05,
        "residual_structure": False,
        "fit_acceptable": True,
    }


def make_config(b_ms: float = 0.5, tiers: dict | None = None) -> dict:
    # bandwidth 0.02 GB/s over 2000 B/token -> exactly 0.1 ms/token transfer
    tiers = tiers or {
        "cpu_to_gpu": {"latency_ms": 10.0, "bandwidth_gb_s": 0.02, "label": "test"}
    }
    return emit_config(
        model_name="stub",
        model_dims=DIMS,
        prefill_fit=linear_fit(2.0, b_ms),
        decode_table=[],
        transfer_tiers=tiers,
        hardware=HARDWARE,
        provenance=PROVENANCE,
    )


def test_config_refuses_failed_fit():
    bad = linear_fit(2.0, 0.5)
    bad["fit_acceptable"] = False
    with pytest.raises(ValueError, match="acceptance gate"):
        emit_config(
            model_name="stub", model_dims=DIMS, prefill_fit=bad,
            decode_table=[], transfer_tiers={}, hardware=HARDWARE,
            provenance=PROVENANCE,
        )


def test_config_validates_hardware_and_decode_cells():
    with pytest.raises(ValueError, match="filesystem"):
        emit_config(
            model_name="stub", model_dims=DIMS, prefill_fit=linear_fit(2, 0.5),
            decode_table=[], transfer_tiers={},
            hardware={"gpu": HARDWARE["gpu"]}, provenance=PROVENANCE,
        )
    with pytest.raises(ValueError, match="realized_mean_context"):
        emit_config(
            model_name="stub", model_dims=DIMS, prefill_fit=linear_fit(2, 0.5),
            decode_table=[{"batch": 1, "target_context": 256, "decode_ms_median": 5}],
            transfer_tiers={}, hardware=HARDWARE, provenance=PROVENANCE,
        )


def test_costs_and_crossover_from_config_alone():
    """recompute: 2 + 0.5L. migrate: 10 + 0.1L. Equal at L=20 exactly."""
    config = make_config()
    assert config["kv_bytes_per_token"] == 2000
    assert recompute_ms(config, 20) == pytest.approx(12.0)
    assert migrate_ms(config, 20, "cpu_to_gpu") == pytest.approx(12.0)
    assert crossover_tokens(config, "cpu_to_gpu") == 20


def test_crossover_band_hand_numbers():
    """near_lo: 10+0.1L <= 1.25(2+0.5L) -> L >= 14.3 -> lo 15.
    past_hi: 2+0.5L > 1.25(10+0.1L) -> L > 28 -> hi 28."""
    band = crossover_band(make_config(), "cpu_to_gpu", ratio=1.25)
    assert band == {
        "lo": 15, "hi": 28, "crossover": 20, "ratio": 1.25, "hi_clamped": False,
    }


def test_near_parallel_lines_produce_wide_clamped_band():
    """b=0.11 vs 0.1 ms/token transfer: crossover exists (L=800) but the
    band never closes above, which is exactly the soft-boundary finding the
    band exists to surface."""
    config = make_config(b_ms=0.11)
    assert crossover_tokens(config, "cpu_to_gpu") == 800
    band = crossover_band(config, "cpu_to_gpu", ratio=1.25)
    assert band["lo"] == 200
    assert band["hi_clamped"] is True
    assert band["hi"] > 100 * band["crossover"]


def test_round_trip_prices_both_directions():
    tiers = {
        "gpu_to_cpu": {"latency_ms": 10.0, "bandwidth_gb_s": 0.02, "label": "down"},
        "cpu_to_gpu": {"latency_ms": 5.0, "bandwidth_gb_s": 0.01, "label": "up"},
    }
    config = make_config(tiers=tiers)
    down = migrate_ms(config, 100, "gpu_to_cpu")  # 10 + 10 = 20
    up = migrate_ms(config, 100, "cpu_to_gpu")  # 5 + 20 = 25
    assert down == pytest.approx(20.0)
    assert up == pytest.approx(25.0)
    assert round_trip_ms(config, 100, "gpu_to_cpu", "cpu_to_gpu") == pytest.approx(45.0)


def test_no_crossover_when_recompute_dominates():
    """Transfer strictly worse than recompute at every length: per-token
    transfer 0.1 ms with recompute slope 0.05 and higher fixed latency."""
    config = make_config(b_ms=0.05)
    assert crossover_tokens(config, "cpu_to_gpu") is None
    assert crossover_band(config, "cpu_to_gpu") is None