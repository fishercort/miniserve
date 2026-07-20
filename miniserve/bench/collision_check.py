"""Collision measurement at low arrival rate, static arm: what fraction of
arrivals find a batch already running?

Operational definitions (they travel with the data, results/collisions.txt):
- collided: client-side TTFT > 0.1 s. Open-window TTFT is about two step
  times (40-60 ms including collector wakeup); collided arrivals wait
  hundreds of ms to seconds. The TTFT distribution is bimodal and 0.1 s sits
  in the gap between modes.
- mean service time: realized output tokens x step delay, averaged over the
  per-request records. Under the EOS-free fake model, cap == realized.
- busy fraction: sum of recorded step latencies over makespan. Idle stretches
  record no steps.

The comparison the numbers feed: utilization arithmetic (rate x mean service)
predicts the collision fraction Poisson arrivals would see; bursty arrivals
oversample busy periods, so the measured fraction runs higher.

Usage: uv run python -m miniserve.bench.collision_check
"""

from miniserve.bench.fake import FlatCostModel
from miniserve.bench.run import run_workload
from miniserve.bench.static import StaticScheduler
from miniserve.bench.workload import WorkloadSpec, generate
from miniserve.engine import Engine
from miniserve.kv_cache import PagedKVCache

THRESH_S = 0.1
RATE_RPS = 0.5
DELAY_S = 0.02


def main() -> None:
    total, collided, svc_sum = 0, 0, 0.0
    busy_fracs = []
    for seed in (0, 1, 2):
        spec = WorkloadSpec(
            n_requests=60, rate_rps=RATE_RPS, seed=seed, burst_factor=3.0
        )
        kv = PagedKVCache(1, 256, 16, 1, 2)
        sched = StaticScheduler(
            kv, FlatCostModel(DELAY_S), max_batch=8, eos_id=0, max_wait_s=0.0
        )
        engine = Engine(sched, idle_wait_s=0.002)
        engine.start()
        try:
            results = run_workload(
                engine,
                generate(spec),
                on_all_submitted=(lambda s=sched: setattr(s, "flush", True)),
            )
        finally:
            engine.stop()
        ttfts = [r.ttft_s for r in results if r.ttft_s is not None]
        hits = sum(1 for t in ttfts if t > THRESH_S)
        total += len(ttfts)
        collided += hits
        svc_sum += sum(r.output_tokens * DELAY_S for r in results)
        makespan = max(r.done_s for r in results)
        busy = sum(s.step_latency_s for s in sched.metrics.steps)
        busy_fracs.append(busy / makespan)
        print(
            f"seed {seed}: {hits}/{len(ttfts)} collided "
            f"({100 * hits / len(ttfts):.0f}%)",
            flush=True,
        )

    mean_svc = svc_sum / total
    print(f"\nmean service time: {mean_svc:.2f}s (realized tokens x step cost)")
    print(f"utilization arithmetic ({RATE_RPS} rps x {mean_svc:.2f}s): "
          f"{RATE_RPS * mean_svc:.2f}")
    print(f"measured busy fraction of wall clock: {sum(busy_fracs) / 3:.2f}")
    print(f"MEASURED COLLISION FRACTION: {collided}/{total} = "
          f"{100 * collided / total:.0f}%")


if __name__ == "__main__":
    main()
