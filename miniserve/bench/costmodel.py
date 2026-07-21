"""Cost-model config assembly and consumption (plan sections 5 and 6).

The emitted config is the contract Phase 3 holds: the simulator, the
policies, and the oracle price decisions from this file alone. Load-bearing
properties, pinned by tests: the migrate-vs-recompute crossover AND its band
are computable from the config with no other inputs, and a config cannot be
emitted around a failed fit.

Transfer tiers are directional: `gpu_to_cpu` and `cpu_to_gpu` are separate
entries (bandwidths differ; disk read and write differ more). migrate_ms
prices ONE direction; the offload decision prices down plus up against
recompute, which is what round_trip_ms exists to make un-missable.
"""

SCHEMA_VERSION = 1

_REQUIRED_HARDWARE = ("gpu", "filesystem")
_REQUIRED_GPU = ("name", "form_factor", "driver", "vram_gb")
_REQUIRED_DECODE_CELL = (
    "batch",
    "target_context",
    "realized_mean_context",  # plan section 2: spec targets, harness reports
    "decode_ms_median",
)


def emit_config(
    *,
    model_name: str,
    model_dims: dict,  # num_layers, num_kv_heads, head_dim, dtype_bytes
    prefill_fit: dict,  # fit_curves() output (preferred + params + gate)
    decode_table: list[dict],  # curve B cells (empty until curve B runs)
    transfer_tiers: dict,  # directional tier -> {latency_ms, bandwidth_gb_s, label}
    hardware: dict,  # gpu {name, form_factor, driver, vram_gb}, filesystem
    provenance: dict,  # timestamp, platform, commit
) -> dict:
    for key in _REQUIRED_HARDWARE:
        if key not in hardware:
            raise ValueError(f"hardware missing required field {key!r}")
    for key in _REQUIRED_GPU:
        if key not in hardware["gpu"]:
            raise ValueError(f"hardware.gpu missing required field {key!r}")
    for cell in decode_table:
        for key in _REQUIRED_DECODE_CELL:
            if key not in cell:
                raise ValueError(
                    f"decode cell missing required field {key!r} "
                    "(plan section 2: realized fields travel with the cell)"
                )
    if not prefill_fit.get("fit_acceptable", False):
        raise ValueError(
            "prefill fit failed its acceptance gate; densify and refit "
            "before emitting a config (plan section 1)"
        )
    bytes_per_token = (
        model_dims["num_layers"]
        * 2
        * model_dims["num_kv_heads"]
        * model_dims["head_dim"]
        * model_dims["dtype_bytes"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "model_dims": dict(model_dims),
        "kv_bytes_per_token": bytes_per_token,
        "prefill": prefill_fit,
        "decode": list(decode_table),
        "transfer": dict(transfer_tiers),
        "hardware": hardware,
        "provenance": provenance,
    }


def recompute_ms(config: dict, tokens: int) -> float:
    """Cost of recomputing a prefix of this length: the prefill curve.
    Recompute pays nothing at evict time; that asymmetry is the offload
    decision's whole shape."""
    fit = config["prefill"]
    if fit["preferred"] == "quadratic":
        q = fit["quadratic"]
        return q["a_ms"] + q["b_ms_per_token"] * tokens + (
            q["c_ms_per_token2"] * tokens * tokens
        )
    lin = fit["linear"]
    return lin["a_ms"] + lin["b_ms_per_token"] * tokens


def migrate_ms(config: dict, tokens: int, tier: str) -> float:
    """ONE-WAY transfer cost through a directional tier: latency plus
    bytes over bandwidth. The offload decision prices down + up against
    recompute; reach for round_trip_ms for that, not this."""
    t = config["transfer"][tier]
    # GB/s = 1e9 bytes/s = 1e6 bytes/ms, so ms/byte = 1 / (GB/s * 1e6).
    ms_per_byte = 1.0 / (t["bandwidth_gb_s"] * 1e6)
    return t["latency_ms"] + tokens * config["kv_bytes_per_token"] * ms_per_byte


def round_trip_ms(config: dict, tokens: int, down_tier: str, up_tier: str) -> float:
    """Offload-then-restore: the cost the eviction decision compares against
    recompute (which pays zero at evict time)."""
    return migrate_ms(config, tokens, down_tier) + migrate_ms(config, tokens, up_tier)


def _bisect_first(pred, lo: int, hi: int) -> int:
    """Smallest L in (lo, hi] with pred true; caller guarantees pred(lo) is
    False, pred(hi) is True, and pred is monotone across the range."""
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


def crossover_tokens(config: dict, tier: str, max_tokens: int = 1 << 20) -> int | None:
    """Smallest prefix length at which one-way migration beats recomputing,
    from the config ALONE (the Phase 3 consumption contract). None if
    recompute wins everywhere in range."""

    def migrate_wins(tokens: int) -> bool:
        return migrate_ms(config, tokens, tier) <= recompute_ms(config, tokens)

    if not migrate_wins(max_tokens):
        return None
    if migrate_wins(1):
        return 1
    return _bisect_first(migrate_wins, 1, max_tokens)


def crossover_band(
    config: dict, tier: str, ratio: float = 1.25, max_tokens: int = 1 << 20
) -> dict | None:
    """The soft boundary: the range where the two costs are within `ratio`
    of each other. Near-parallel cost lines make the point crossover
    exquisitely sensitive to fit error; the band is what policies should
    treat as the decision boundary, and a wide band is itself a finding.
    Assumes the cost ratio is monotone through the band, which holds for the
    linear and quadratic forms this module emits."""
    crossover = crossover_tokens(config, tier, max_tokens)
    if crossover is None:
        return None

    def near_lo(tokens: int) -> bool:  # migrate within ratio of recompute
        return migrate_ms(config, tokens, tier) <= ratio * recompute_ms(config, tokens)

    def past_hi(tokens: int) -> bool:  # recompute decisively lost
        return recompute_ms(config, tokens) > ratio * migrate_ms(config, tokens, tier)

    lo = 1 if near_lo(1) else _bisect_first(near_lo, 1, crossover)
    clamped = not past_hi(max_tokens)
    hi = max_tokens if clamped else _bisect_first(past_hi, crossover, max_tokens) - 1
    return {
        "lo": lo,
        "hi": hi,
        "crossover": crossover,
        "ratio": ratio,
        "hi_clamped": clamped,
    }
