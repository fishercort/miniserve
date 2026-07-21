"""Activation peak vs batch composition (plan section 7 batch item; the task
MECHANICS.md's overhead band assumption created).

CUDA only: peak measurement uses torch.cuda.max_memory_allocated, which has
no CPU or MPS analog worth trusting. Non-CUDA invocations exit with a
skipped notice rather than fake numbers.

Usage: uv run python -m miniserve.bench.activation_probe --device cuda
"""

import argparse
import json
import pathlib
import platform
import random

from miniserve.kv_cache import PagedKVCache
from miniserve.scheduler import Scheduler, Status


def plan_cells(
    batches: tuple = (1, 4, 8),
    contexts: tuple = (256, 1024, 4096),
    prefill_lens: tuple = (512, 2048),
) -> list[dict]:
    """The composition grid: steady-state decode cells, plus mixed cells
    (one long prefill joining a running decode batch), which is where the
    peak lives."""
    cells = [
        {"kind": "decode", "batch": b, "context": c}
        for b in batches
        for c in contexts
    ]
    cells += [
        {"kind": "mixed", "batch": b, "context": 1024, "prefill_len": p}
        for b in (4, 8)
        for p in prefill_lens
    ]
    return cells


def measure_cell(model, cell: dict, device: str, dtype, block_size: int = 16) -> dict:
    import torch

    rng = random.Random(0)
    vocab = int(model.embed.shape[0])
    ctx = cell["context"]
    extra = cell.get("prefill_len", 0)
    batch = cell["batch"]
    blocks = (batch + 1) * (-(-(ctx + extra + 8) // block_size)) + 4
    kv = PagedKVCache(
        model.num_layers, blocks, block_size, model.n_kv_heads, model.head_dim,
        dtype=dtype, device=device,
    )
    sched = Scheduler(kv, model, max_batch=batch + 1, eos_id=-1)
    for b in range(batch):
        prompt = [rng.randrange(3, vocab) for _ in range(ctx)]
        sched.submit(f"d{b}", prompt, max_tokens=64)
    while any(s.status is not Status.DECODE for s in sched.running) or sched.waiting:
        sched.step()
    if cell["kind"] == "mixed":
        prompt = [rng.randrange(3, vocab) for _ in range(cell["prefill_len"])]
        sched.submit("mix", prompt, max_tokens=8)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(4):
        sched.step()
    peak = int(torch.cuda.max_memory_allocated())
    return {**cell, "peak_bytes": peak}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/activation_peak.json")
    args = ap.parse_args(argv)

    import torch

    if not args.device.startswith("cuda"):
        print("activation probe requires CUDA; skipped (no fake numbers).")
        return

    from miniserve.model import PagedModel

    dtype = torch.bfloat16
    pm = PagedModel.from_pretrained(args.model, device=args.device, dtype=dtype)
    cells = []
    for cell in plan_cells():
        result = measure_cell(pm, cell, args.device, dtype)
        cells.append(result)
        print(f"{result['kind']} batch={result['batch']} "
              f"ctx={result['context']}: peak {result['peak_bytes'] / 2**30:.2f} GiB",
              flush=True)
    out = {
        "model": args.model,
        "device": args.device,
        "platform": platform.platform(),
        "cells": cells,
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
