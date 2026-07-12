"""Load generator: bursty Poisson arrivals with a spread of output lengths.

Deterministic: same spec (including seed) produces the identical schedule.

Output-length control, stated as the experiment's convention: the scheduler
stops a sequence at exactly max_tokens, and the bench fake model never emits
EOS, so in mechanism runs the cap IS the realized length; the independent
variable is controlled, not observed. Real-model runs can hit EOS early, so
there max_tokens is a cap and the report states realized lengths.

Prompts are random token ids: content is meaningless by design, because the
benchmark measures scheduling, not generation quality. For real-model runs
this also shifts EOS timing, which is one more reason realized lengths get
reported rather than assumed.
"""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Arrival:
    at_s: float
    req_id: str
    prompt_ids: list[int]
    max_tokens: int


@dataclass(frozen=True)
class WorkloadSpec:
    n_requests: int = 50
    rate_rps: float = 4.0
    seed: int = 0
    prompt_len: tuple[int, int] = (8, 64)
    # Mixture of output lengths: (weight, (lo, hi)). The spread is the point:
    # static batching's slot waste comes from short sequences waiting on long.
    output_mix: tuple = ((0.7, (4, 32)), (0.3, (48, 128)))
    # 1.0 = plain Poisson. >1.0 alternates hot/cold windows: rate*f in hot,
    # rate/f in cold, switching every burst_period_s. Cheap Markov modulation.
    # Note: in a cold window the gaps are long enough that t can leap across
    # an entire hot window without an arrival in it, so the realized burst
    # pattern differs from nominal; report realized_burst_profile, not spec.
    burst_factor: float = 1.0
    burst_period_s: float = 4.0
    vocab: int = 1000


def generate(spec: WorkloadSpec) -> list[Arrival]:
    weight_sum = sum(w for w, _ in spec.output_mix)
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"output_mix weights sum to {weight_sum}, expected 1.0")
    rng = random.Random(spec.seed)
    t = 0.0
    arrivals = []
    for i in range(spec.n_requests):
        if spec.burst_factor > 1.0:
            hot = int(t / spec.burst_period_s) % 2 == 0
            rate = spec.rate_rps * (
                spec.burst_factor if hot else 1.0 / spec.burst_factor
            )
        else:
            rate = spec.rate_rps
        t += rng.expovariate(rate)
        plen = rng.randint(*spec.prompt_len)
        roll, acc, max_tokens = rng.random(), 0.0, spec.output_mix[-1][1][1]
        for weight, (lo, hi) in spec.output_mix:
            acc += weight
            if roll <= acc:
                max_tokens = rng.randint(lo, hi)
                break
        prompt = [rng.randrange(3, spec.vocab) for _ in range(plen)]
        arrivals.append(Arrival(t, f"req-{i:04d}", prompt, max_tokens))
    return arrivals


def realized_burst_profile(
    schedule: list[Arrival], spec: WorkloadSpec
) -> list[int]:
    """Arrivals per burst window, realized. Spec targets, harness reports:
    same honesty rule as the eval plan's realized ephemeral fraction."""
    if not schedule:
        return []
    n_windows = int(schedule[-1].at_s / spec.burst_period_s) + 1
    counts = [0] * n_windows
    for a in schedule:
        counts[int(a.at_s / spec.burst_period_s)] += 1
    return counts
