"""Calibration mechanics, hermetic: probe bookkeeping against a stub model,
vocab derivation from the embedding table, and the fit/gate math on synthetic
curves with known answers."""

import torch

from miniserve.bench.calibrate import fit_curves, measure_prefill


class StubModel:
    """Exposes the PagedModel surface calibrate reads: dims and an embedding
    table (vocab derives from its shape, never a literal)."""

    num_layers = 1
    n_kv_heads = 1
    head_dim = 2
    embed = torch.zeros(50, 2)  # vocab = 50

    def forward(self, seqs, kv):
        out = {}
        for s in seqs:
            v = torch.zeros(8)
            v[1] = 1.0
            out[s.req_id] = v
        return out


def test_measure_prefill_bookkeeping():
    records = measure_prefill(
        StubModel(), lengths=[4, 8], repeats=2, warmup=1, block_size=4
    )
    assert len(records) == 4  # warmup discarded, repeats x lengths kept
    assert sorted({r["prompt_tokens"] for r in records}) == [4, 8]
    assert all(r["prefill_ms"] >= 0 for r in records)
    # round-robin order: rep 1 of each length, then rep 2
    assert [r["prompt_tokens"] for r in records] == [4, 8, 4, 8]


def test_fit_recovers_linear_and_gates():
    records = [
        {"prompt_tokens": n, "prefill_ms": 2.0 + 0.5 * n}
        for n in (16, 32, 64, 128)
        for _ in range(3)
    ]
    fits = fit_curves(records)
    assert abs(fits["linear"]["b_ms_per_token"] - 0.5) < 1e-6
    assert abs(fits["linear"]["a_ms"] - 2.0) < 1e-6
    assert fits["preferred"] == "linear"  # quadratic can't beat exact by 20%
    assert fits["fit_acceptable"] is True


def test_fit_prefers_quadratic_when_it_earns_it():
    records = [
        {"prompt_tokens": n, "prefill_ms": 1.0 + 0.1 * n + 0.01 * n * n}
        for n in (16, 32, 64, 128, 256)
    ]
    fits = fit_curves(records)
    assert fits["preferred"] == "quadratic"
    assert abs(fits["quadratic"]["c_ms_per_token2"] - 0.01) < 1e-6
    assert fits["fit_acceptable"] is True
    assert fits["residual_structure"] is False


def test_gate_rejects_saturation_knee():
    """The negative case is the whole point of a gate: a piecewise curve
    (linear, then a sharp knee, the shape a real GPU produces at saturation)
    must not be accepted, and the gate must say which failure mode fired."""

    def knee(n: int) -> float:
        if n <= 256:
            return 2.0 + 0.05 * n
        return 2.0 + 0.05 * 256 + 1.5 * (n - 256)

    records = [
        {"prompt_tokens": n, "prefill_ms": knee(n)}
        for n in (16, 32, 64, 128, 256, 512, 1024, 2048)
    ]
    fits = fit_curves(records)
    assert fits["fit_acceptable"] is False
    preferred_err = fits[fits["preferred"]]["mean_rel_err"]
    assert preferred_err > fits["fit_gate_rel_err"] or fits["residual_structure"]
