"""Crossover spot-validation (plan section 5): at each tier's band edges and
crossover, measure BOTH paths end to end and adjudicate the pre-committed
contract: predictions within 25 percent of measurement at 2 of 3 points per
tier, or the cost model is revised before Phase 3 inherits it.

v1 scope: device-interconnect tiers (the restore direction, cpu_to_gpu).
Migration is measured as a contiguous transfer, consistent with the config's
transfer_measurement label; the chunked-overhead coefficient is reported
separately by the transfer probe.
"""

import argparse
import json
import pathlib
import statistics
import time

from miniserve.bench.calibrate import device_sync, measure_prefill
from miniserve.bench.costmodel import (
    crossover_band,
    migrate_ms,
    recompute_ms,
)

CONTRACT_REL_ERR = 0.25
CONTRACT_MIN_PASSING = 2


def select_points(band: dict) -> list[int]:
    """Band lower edge, crossover, band upper edge; a clamped band substitutes
    2x crossover for the missing edge."""
    hi = band["crossover"] * 2 if band["hi_clamped"] else band["hi"]
    return [band["lo"], band["crossover"], hi]


def evaluate_contract(points: list[dict]) -> dict:
    """A point passes when BOTH path predictions are within the contract
    error of their measurements; the tier passes at 2 of 3."""
    for p in points:
        err_r = abs(p["predicted_recompute_ms"] - p["measured_recompute_ms"]) / (
            p["measured_recompute_ms"]
        )
        err_m = abs(p["predicted_migrate_ms"] - p["measured_migrate_ms"]) / (
            p["measured_migrate_ms"]
        )
        p["rel_err_recompute"] = err_r
        p["rel_err_migrate"] = err_m
        p["ok"] = err_r <= CONTRACT_REL_ERR and err_m <= CONTRACT_REL_ERR
    return {
        "points": points,
        "n_passing": sum(1 for p in points if p["ok"]),
        "contract_passes": sum(1 for p in points if p["ok"]) >= CONTRACT_MIN_PASSING,
    }


def measure_migrate_contiguous(
    device: str, tokens: int, kv_bytes_per_token: int, repeats: int = 5
) -> float:
    """The restore direction: cpu -> device, contiguous, explicit sync."""
    import torch

    elems = tokens * kv_bytes_per_token // 2
    src = torch.zeros(elems, dtype=torch.bfloat16)
    samples = []
    for _ in range(repeats):
        device_sync(device)
        t0 = time.monotonic()
        src.to(device)
        device_sync(device)
        samples.append((time.monotonic() - t0) * 1000.0)
    return statistics.median(samples)


def validate_tier(config: dict, tier: str, model, device: str, dtype) -> dict | None:
    band = crossover_band(config, tier)
    if band is None:
        return {"tier": tier, "band": None, "note": "no crossover in range"}
    points = []
    for tokens in select_points(band):
        prefill_records = measure_prefill(
            model, [tokens], repeats=3, warmup=1, device=device, dtype=dtype
        )
        measured_recompute = statistics.median(
            r["prefill_ms"] for r in prefill_records
        )
        measured_migrate = measure_migrate_contiguous(
            device, tokens, config["kv_bytes_per_token"]
        )
        points.append(
            {
                "tokens": tokens,
                "predicted_recompute_ms": recompute_ms(config, tokens),
                "measured_recompute_ms": measured_recompute,
                "predicted_migrate_ms": migrate_ms(config, tokens, tier),
                "measured_migrate_ms": measured_migrate,
            }
        )
    verdict = evaluate_contract(points)
    return {"tier": tier, "band": band, **verdict}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="results/gpu_day/cost_model.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/gpu_day/spot_validation.json")
    args = ap.parse_args(argv)

    import torch

    from miniserve.model import PagedModel

    config = json.loads(pathlib.Path(args.config).read_text())
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    pm = PagedModel.from_pretrained(args.model, device=args.device, dtype=dtype)
    results = []
    for tier in config["transfer"]:
        if not tier.startswith("cpu_to_") or tier == "cpu_to_disk":
            continue  # v1 scope: restore-direction device tiers
        result = validate_tier(config, tier, pm, args.device, dtype)
        results.append(result)
        if result.get("band"):
            print(
                f"{tier}: band {result['band']['lo']}..{result['band']['hi']}, "
                f"{result['n_passing']}/3 points in contract, "
                f"{'PASS' if result['contract_passes'] else 'FAIL: revise model'}",
                flush=True,
            )
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
