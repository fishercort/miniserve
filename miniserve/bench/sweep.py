"""Mechanism sweep: continuous vs static across arrival rates.

Main sweep runs with ample blocks and asserts zero preemptions per cell, so
the chart shows exactly one mechanism (batching/admission policy) and not a
blend with preemption cost. The pressure ablation is a separate cell family
(mid rate, tight blocks) where preemption behavior is shown explicitly.

Static arm runs max_wait_s=0, and that is a decision with a name:
drain-in-batches, the STRONGEST static configuration. It never idles waiting
to fill; a batch is whatever is queued when the previous batch retires. The
wait-to-fill pathology is absent by construction, so the measured gap is
run-to-completion slot waste only, the mechanism the output mix exists to
expose. Beating the best static configuration is the claim; it also collapses
the W sweep dimension legitimately, and makes the arms identical at idle (the
0.5 rps left edge is a harness-fairness exhibit).

Both arms share max_batch: identical parallelism ceiling, only admission
differs. The n16 cell (static at double the ceiling) checks the result is not
a ceiling artifact, replacing the full N sweep with one sentence and one cell.

Usage: uv run python -m miniserve.bench.sweep --out results/mechanism.jsonl
"""

import argparse
import json
import pathlib
import time

from miniserve.bench.fake import FlatCostModel
from miniserve.bench.run import run_workload, summarize
from miniserve.bench.static import StaticScheduler
from miniserve.bench.workload import WorkloadSpec, generate, realized_burst_profile
from miniserve.engine import Engine
from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import Scheduler

EOS = 0  # FlatCostModel never emits it


def run_cell(
    arm: str,
    rate_rps: float,
    seed: int,
    n_requests: int,
    delay_s: float,
    max_batch: int,
    num_blocks: int,
    block_size: int,
    cell: str = "main",
) -> dict:
    spec = WorkloadSpec(
        n_requests=n_requests, rate_rps=rate_rps, seed=seed, burst_factor=3.0
    )
    schedule = generate(spec)
    # poison_on_free deliberately OFF: per-free tensor fills would pollute
    # step timing in a perf run. Tests run it on; benchmarks run it off.
    kv = PagedKVCache(1, num_blocks, block_size, 1, 2)
    model = FlatCostModel(delay_s=delay_s)
    if arm == "static":
        sched = StaticScheduler(
            kv, model, max_batch=max_batch, eos_id=EOS, max_wait_s=0.0
        )
        # Redundant under W=0 (the timer already opens partial batches);
        # kept as belt-and-suspenders for the tail.
        flush = lambda: setattr(sched, "flush", True)  # noqa: E731
    else:
        sched = Scheduler(kv, model, max_batch=max_batch, eos_id=EOS)
        flush = None
    engine = Engine(sched, idle_wait_s=0.002)
    engine.start()
    try:
        results = run_workload(engine, schedule, on_all_submitted=flush)
    finally:
        engine.stop()
    kv.check_invariants()
    summary = summarize(results)
    # Validity: a main-sweep cell must be preemption-free, or it is measuring
    # two mechanisms at once and cannot go on the chart.
    valid = summary["run_valid_for_chart"] and (
        cell != "main" or sched.preemptions_total == 0
    )
    return {
        "cell": cell,
        "arm": arm,
        "rate_rps": rate_rps,
        "seed": seed,
        "valid_for_chart": valid,
        "preemptions_total": sched.preemptions_total,
        "realized_burst_profile": realized_burst_profile(schedule, spec),
        **summary,
        "config": {
            "n_requests": n_requests,
            "delay_s": delay_s,
            "max_batch": max_batch,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "burst_factor": spec.burst_factor,
        },
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mechanism_sweep.jsonl")
    ap.add_argument("--rates", default="0.5,1,2,4,8,16")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--delay-ms", type=float, default=20.0)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--pressure-blocks", type=int, default=20)
    ap.add_argument("--pressure-rate", type=float, default=4.0)
    ap.add_argument("--skip-pressure", action="store_true")
    args = ap.parse_args(argv)

    rates = [float(r) for r in args.rates.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # (cell, blocks, rate, arms, max_batch)
    both = ("continuous", "static")
    cells = [("main", args.blocks, r, both, args.max_batch) for r in rates]
    # Ceiling-robustness check: static at double max_batch, near saturation.
    cells.append(("n16", args.blocks, 8.0, ("static",), args.max_batch * 2))
    if not args.skip_pressure:
        cells.append(
            ("pressure", args.pressure_blocks, args.pressure_rate, both, args.max_batch)
        )

    t_sweep = time.monotonic()
    with out.open("w") as f:
        for cell, blocks, rate, arms, max_batch in cells:
            for arm in arms:
                for seed in seeds:
                    t_cell = time.monotonic()
                    row = run_cell(
                        arm,
                        rate,
                        seed,
                        n_requests=args.n,
                        delay_s=args.delay_ms / 1000.0,
                        max_batch=max_batch,
                        num_blocks=blocks,
                        block_size=args.block_size,
                        cell=cell,
                    )
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    ttft = row["ttft_p95_s"]
                    lag = row["sched_lag_max_s"]
                    print(
                        f"[{time.monotonic() - t_sweep:7.1f}s] {cell}/{arm} "
                        f"rate={rate} seed={seed}: "
                        f"tput={row['throughput_tok_s']:.1f} tok/s "
                        f"ttft_p95={f'{ttft:.3f}s' if ttft is not None else 'n/a'} "
                        f"preempt={row['preemptions_total']} "
                        f"lag_max={f'{lag * 1000:.2f}ms' if lag is not None else 'n/a'} "
                        f"valid={row['valid_for_chart']} "
                        f"({time.monotonic() - t_cell:.0f}s)",
                        flush=True,
                    )
    print(f"done in {time.monotonic() - t_sweep:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
