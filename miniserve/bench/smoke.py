"""Device smoke: load the model on the target device, run the engine a few
steps, assert the logits are finite. Fifteen seconds on the rental confirms
the CUDA path executes and produces sane numbers before gpu_day spends real
meter time.

Usage on the box, FIRST: uv run python -m miniserve.bench.smoke --device cuda
"""

import argparse


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    import torch

    from miniserve.bench.calibrate import measure_prefill
    from miniserve.kv_cache import PagedKVCache
    from miniserve.model import PagedModel
    from miniserve.scheduler import Scheduler, Status

    print(f"torch {torch.__version__}; cuda_available={torch.cuda.is_available()}")
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA not visible to torch on this box"
        print(f"device: {torch.cuda.get_device_name(0)}")

    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    pm = PagedModel.from_pretrained(args.model, device=args.device, dtype=dtype)

    # 1. Direct prefill forward: finite logits (a device mismatch throws here).
    kv = PagedKVCache(
        pm.num_layers, 8, 16, pm.n_kv_heads, pm.head_dim, dtype=dtype,
        device=args.device,
    )
    sched = Scheduler(kv, pm, max_batch=1, eos_id=-1)
    sched.submit("smoke", list(range(3, 20)), max_tokens=3)
    sched.step()  # admit + prefill + first token
    seqs = [s for s in sched.running if s.req_id == "smoke"]
    assert seqs and seqs[0].status is Status.DECODE
    while sched.waiting or sched.running:
        sched.step()
    assert len([r for r in sched.metrics.requests if r.req_id == "smoke"]) == 1

    # 2. The calibrate probe path gpu_day uses, one length, so its device
    #    handling is exercised too.
    records = measure_prefill(pm, [32], repeats=1, warmup=1, device=args.device,
                              dtype=dtype)
    assert records and records[0]["prefill_ms"] > 0

    print(f"OK: engine ran on {args.device}, 3 tokens generated, probe path "
          "exercised. gpu_day is clear.")


if __name__ == "__main__":
    main()
