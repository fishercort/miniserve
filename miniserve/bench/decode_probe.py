"""Cost-model calibration, curve B: decode step latency vs (batch size,
context length). Protocol: docs/phase2-cost-model.md, Measurement plan §2.

The independent-variable control: with B sequences decoding together,
contexts grow during the probe. Prompts are sized to target minus k; the
measured steps are exactly steps 1..k after all B sequences are in DECODE;
the cell carries the realized mean context per measured step alongside the
target label. Spec targets, harness reports realized.

Usage: uv run python -m miniserve.bench.decode_probe --device cpu \
    --batches 1,2 --contexts 64,128   (small grids for CPU validation)
"""

import argparse
import json
import pathlib
import platform
import random
import statistics

from miniserve.bench.calibrate import _SyncedForward
from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import Scheduler, Status

K_MEASURED_STEPS = 8


def measure_decode_cell(
    model,
    batch: int,
    target_context: int,
    repeats: int = 3,
    block_size: int = 16,
    device: str = "cpu",
    dtype=None,
    vocab: int | None = None,
) -> dict:
    import torch

    if vocab is None:
        vocab = int(model.embed.shape[0])
    rng = random.Random(0)
    prompt_len = target_context - K_MEASURED_STEPS
    assert prompt_len > 0, "target_context must exceed the measured-step count"
    timed_model = _SyncedForward(model, device) if device != "cpu" else model

    step_ms: list[float] = []
    realized: list[float] = []
    for rep in range(repeats):
        blocks = batch * (-(-(target_context + 2) // block_size)) + 4
        kv = PagedKVCache(
            model.num_layers, blocks, block_size, model.n_kv_heads,
            model.head_dim, dtype=dtype or torch.float32, device=device,
        )
        sched = Scheduler(kv, timed_model, max_batch=batch, eos_id=-1)
        for b in range(batch):
            prompt = [rng.randrange(3, vocab) for _ in range(prompt_len)]
            sched.submit(f"c{rep}-{b}", prompt, max_tokens=K_MEASURED_STEPS + 1)
        # Step until every sequence is in DECODE (the prefill step).
        while any(s.status is not Status.DECODE for s in sched.running) or (
            sched.waiting
        ):
            sched.step()
        # Measured steps: exactly 1..k, all B sequences decoding together.
        for _ in range(K_MEASURED_STEPS):
            realized.append(
                sum(s.total_len() for s in sched.running) / len(sched.running)
            )
            before = len(sched.metrics.steps)
            sched.step()
            rec = sched.metrics.steps[before]
            assert rec.n_prefill == 0 and rec.n_decode == batch
            step_ms.append(rec.step_latency_s * 1000.0)
        while sched.waiting or sched.running:  # drain to retirement
            sched.step()
    return {
        "batch": batch,
        "target_context": target_context,
        "realized_mean_context": sum(realized) / len(realized),
        "decode_ms_median": statistics.median(step_ms),
        "decode_ms_all": step_ms,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--contexts", default="256,1024,4096")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/decode_table.json")
    args = ap.parse_args(argv)

    import torch

    from miniserve.model import PagedModel

    if not args.device.startswith("cuda"):
        print(
            "non-GPU run: pipeline validation only; constants do not graduate "
            "(docs/phase2-cost-model.md, section 6).",
            flush=True,
        )
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    pm = PagedModel.from_pretrained(args.model, device=args.device, dtype=dtype)
    batches = [int(x) for x in args.batches.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    # Warmup: one discarded run of the largest cell (pages, caches, clocks).
    measure_decode_cell(
        pm, max(batches), max(contexts), repeats=1,
        device=args.device, dtype=dtype,
    )
    cells = []
    for target in contexts:
        for batch in batches:
            cell = measure_decode_cell(
                pm, batch, target, repeats=args.repeats,
                device=args.device, dtype=dtype,
            )
            cells.append(cell)
            print(
                f"batch={batch} target_ctx={target}: "
                f"median {cell['decode_ms_median']:.2f} ms, "
                f"realized ctx {cell['realized_mean_context']:.1f}",
                flush=True,
            )
    out = {
        "model": args.model,
        "device": args.device,
        "dtype": str(dtype),
        "platform": platform.platform(),
        "graduates_to_cost_model": args.device.startswith("cuda"),
        "k_measured_steps": K_MEASURED_STEPS,
        "cells": cells,
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
