"""Cost-model calibration, curve A: prefill latency vs prompt length on the
real model, through the engine's own per-step metrics log.

Protocol lives in docs/phase2-cost-model.md (Measurement plan, section 1);
this file implements it. The same script later ships as part of the
benchmark's calibrate CLI, measuring on the user's hardware.

Usage: uv run python -m miniserve.bench.calibrate --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import json
import pathlib
import platform
import random
import statistics

from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import Scheduler

FIT_GATE_REL_ERR = 0.05  # plan section 1: preferred fit must beat this


def device_sync(device: str) -> None:
    """THE sync primitive (plan section 4). Every timed region in curves A,
    B, and C ends through this function; no curve grows its own sync."""
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class _SyncedForward:
    """Wraps a model so the device sync lands INSIDE the timed step (plan
    section 4: timing never relies on the sampling .item() sync)."""

    def __init__(self, inner, device: str):
        self.inner = inner
        self.device = device

    def forward(self, seqs, kv):
        out = self.inner.forward(seqs, kv)
        device_sync(self.device)
        return out


def measure_prefill(
    model,
    lengths: list[int],
    repeats: int = 3,
    warmup: int = 3,
    block_size: int = 16,
    device: str = "cpu",
    dtype=None,
    vocab: int | None = None,
) -> list[dict]:
    """One engine probe per (length, repeat), against a single reused cache.

    Model must expose num_layers, n_kv_heads, head_dim (PagedModel does);
    vocab derives from model.embed when not given (never a literal).
    """
    import torch

    if vocab is None:
        vocab = int(model.embed.shape[0])
    rng = random.Random(0)
    max_len = max(lengths)
    kv = PagedKVCache(
        model.num_layers,
        -(-max_len // block_size) + 4,
        block_size,
        model.n_kv_heads,
        model.head_dim,
        dtype=dtype or torch.float32,
        device=device,
    )
    timed_model = _SyncedForward(model, device) if device != "cpu" else model
    sched = Scheduler(kv, timed_model, max_batch=1, eos_id=-1)

    def probe(i: int, length: int) -> dict:
        prompt = [rng.randrange(3, vocab) for _ in range(length)]
        sched.submit(f"probe-{i}", prompt, max_tokens=1)
        while sched.waiting or sched.running:
            sched.step()
        rec = sched.metrics.steps[-1]
        assert rec.n_prefill == 1 and rec.n_prefill_tokens_fresh == length
        return {"prompt_tokens": length, "prefill_ms": rec.step_latency_s * 1000.0}

    for i in range(warmup):
        probe(-(i + 1), max_len)  # discarded: faults every page (FIFO reuse)
    records = []
    i = 0
    for _ in range(repeats):  # round-robin: drift spreads across lengths
        for length in lengths:
            records.append(probe(i, length))
            i += 1
    return records


def _residual_structure(y, pred) -> bool:
    """Plan section 1, failure mode (b): four or more consecutive same-sign
    residuals (each beyond 1 percent) means structure, e.g. a saturation
    knee hiding under a passing mean error."""
    run_sign, run_len = 0, 0
    for actual, fitted in zip(y, pred, strict=True):
        rel = (fitted - actual) / actual
        sign = 0 if abs(rel) < 0.01 else (1 if rel > 0 else -1)
        if sign != 0 and sign == run_sign:
            run_len += 1
        else:
            run_sign, run_len = sign, (1 if sign != 0 else 0)
        if run_len >= 4:
            return True
    return False


def fit_curves(records: list[dict]) -> dict:
    import numpy as np

    by_len: dict[int, list[float]] = {}
    for r in records:
        by_len.setdefault(r["prompt_tokens"], []).append(r["prefill_ms"])
    lengths = sorted(by_len)
    medians = [statistics.median(by_len[length]) for length in lengths]
    x = np.array(lengths, dtype=float)
    y = np.array(medians, dtype=float)

    def rel_err(pred) -> float:
        return float(np.mean(np.abs(pred - y) / y))

    lin = np.polyfit(x, y, 1)
    quad = np.polyfit(x, y, 2)
    lin_err = rel_err(np.polyval(lin, x))
    quad_err = rel_err(np.polyval(quad, x))
    preferred = "quadratic" if quad_err < 0.8 * lin_err else "linear"
    preferred_err = quad_err if preferred == "quadratic" else lin_err
    preferred_pred = np.polyval(quad if preferred == "quadratic" else lin, x)
    structured = _residual_structure(y, preferred_pred)
    return {
        "lengths": lengths,
        "median_prefill_ms": medians,
        "linear": {
            "b_ms_per_token": float(lin[0]),
            "a_ms": float(lin[1]),
            "mean_rel_err": lin_err,
        },
        "quadratic": {
            "c_ms_per_token2": float(quad[0]),
            "b_ms_per_token": float(quad[1]),
            "a_ms": float(quad[2]),
            "mean_rel_err": quad_err,
        },
        "preferred": preferred,
        # Plan section 1 acceptance gate, both failure modes: mean error
        # threshold AND structureless residuals. Either failure means the
        # grid is densified and refit before any downstream use.
        "fit_gate_rel_err": FIT_GATE_REL_ERR,
        "residual_structure": structured,
        "fit_acceptable": preferred_err <= FIT_GATE_REL_ERR and not structured,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lengths", default="16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/prefill_curve.json")
    args = ap.parse_args(argv)

    import torch

    from miniserve.model import PagedModel

    if not args.device.startswith("cuda"):
        print(
            "non-GPU run: pipeline validation only; constants do not graduate "
            "to the cost model (docs/phase2-cost-model.md, section 6).",
            flush=True,
        )
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    pm = PagedModel.from_pretrained(args.model, device=args.device, dtype=dtype)
    lengths = [int(x) for x in args.lengths.split(",")]
    print(
        f"probing {len(lengths)} lengths x {args.repeats} repeats "
        f"on {args.device}/{dtype} ...",
        flush=True,
    )
    records = measure_prefill(
        pm, lengths, repeats=args.repeats, device=args.device, dtype=dtype
    )
    fits = fit_curves(records)
    out = {
        "model": args.model,
        "device": args.device,
        "dtype": str(dtype),
        "platform": platform.platform(),
        "graduates_to_cost_model": args.device.startswith("cuda"),
        "raw": records,
        **fits,
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    for key in ("lengths", "median_prefill_ms", "linear", "quadratic",
                "preferred", "fit_acceptable"):
        print(f"{key}: {out[key]}")
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
