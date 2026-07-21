"""Cost-model calibration, curve C: transfer time per directional tier.
Protocol: docs/phase2-cost-model.md, Measurement plan §3.

Tiers are directional (gpu_to_cpu and cpu_to_gpu are separate; disk write
and read likewise). Fit per tier: t = latency + bytes/bandwidth over block
batches, separating per-transfer latency from bandwidth.

Page-cache honesty (plan §3): a timed read of a just-written file measures
RAM, not disk. The disk tiers here measure the OS-cached path and are
labeled warm_page_cached; cold-disk is out of scope without root.

Usage: uv run python -m miniserve.bench.transfer_probe --device cpu
"""

import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import tempfile
import time

from miniserve.bench.calibrate import device_sync

BLOCK_BATCHES = (1, 2, 4, 8, 16, 32, 64)


def filesystem_label(path: str) -> str:
    """Best-effort filesystem type for the probed path; 'unknown' beats a
    guess. Recorded because disk numbers are meaningless without it."""
    try:
        if platform.system() == "Darwin":
            dev = subprocess.run(
                ["df", "-P", path], capture_output=True, text=True, check=True
            ).stdout.splitlines()[-1].split()[0]
            for line in subprocess.run(
                ["mount"], capture_output=True, text=True, check=True
            ).stdout.splitlines():
                if line.startswith(dev + " "):
                    return line.split("(")[1].split(",")[0]
        else:
            out = subprocess.run(
                ["df", "-PT", path], capture_output=True, text=True, check=True
            ).stdout.splitlines()[-1]
            return out.split()[1]
    except Exception:
        pass
    return "unknown"


def _fit_latency_bandwidth(points: list[dict]) -> dict:
    """t_ms = latency_ms + bytes * ms_per_byte; bandwidth back-computed.
    (GB/s = 1e9 B/s = 1e6 B/ms, so bw_gb_s = 1 / (ms_per_byte * 1e6).)"""
    import numpy as np

    x = np.array([p["bytes"] for p in points], dtype=float)
    y = np.array([p["ms"] for p in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "latency_ms": float(max(intercept, 0.0)),
        "bandwidth_gb_s": float(1.0 / (slope * 1e6)) if slope > 0 else float("inf"),
        "points": points,
    }


def _timed(fn, sync_devices: tuple[str, ...]) -> float:
    for d in sync_devices:
        device_sync(d)
    t0 = time.monotonic()
    fn()
    for d in sync_devices:
        device_sync(d)
    return (time.monotonic() - t0) * 1000.0


def measure_device_tiers(
    device: str, block_bytes: int, repeats: int = 5
) -> dict[str, dict]:
    """gpu_to_cpu and cpu_to_gpu (or mps analogs), explicit sync both sides."""
    import torch

    if device == "cpu":
        return {}
    elems = block_bytes // 2  # bf16/fp16-sized elements
    tiers: dict[str, list[dict]] = {f"{device}_to_cpu": [], f"cpu_to_{device}": []}
    for n in BLOCK_BATCHES:
        src_dev = torch.zeros(n * elems, dtype=torch.bfloat16, device=device)
        src_cpu = torch.zeros(n * elems, dtype=torch.bfloat16)
        down = statistics.median(
            _timed(lambda t=src_dev: t.to("cpu"), (device,)) for _ in range(repeats)
        )
        up = statistics.median(
            _timed(lambda t=src_cpu: t.to(device), (device,)) for _ in range(repeats)
        )
        nbytes = n * elems * 2
        tiers[f"{device}_to_cpu"].append({"bytes": nbytes, "ms": down})
        tiers[f"cpu_to_{device}"].append({"bytes": nbytes, "ms": up})
    return {
        name: {**_fit_latency_bandwidth(pts), "label": "device_interconnect"}
        for name, pts in tiers.items()
    }


def measure_disk_tiers(
    disk_path: str, block_bytes: int, repeats: int = 5
) -> dict[str, dict]:
    """cpu_to_disk (write+fsync) and disk_to_cpu (warm read). The read tier
    is page-cached by construction and labeled so."""
    fs = filesystem_label(disk_path)
    write_pts, read_pts = [], []
    with tempfile.TemporaryDirectory(dir=disk_path) as tmp:
        for n in BLOCK_BATCHES:
            buf = os.urandom(n * block_bytes)
            fname = pathlib.Path(tmp) / f"blk{n}"

            def write(fname=fname, buf=buf) -> None:
                fd = os.open(fname, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
                try:
                    os.write(fd, buf)
                    os.fsync(fd)
                finally:
                    os.close(fd)

            def read(fname=fname) -> None:
                with open(fname, "rb") as f:
                    f.read()

            w = statistics.median(_timed(write, ()) for _ in range(repeats))
            r = statistics.median(_timed(read, ()) for _ in range(repeats))
            write_pts.append({"bytes": n * block_bytes, "ms": w})
            read_pts.append({"bytes": n * block_bytes, "ms": r})
    return {
        "cpu_to_disk": {
            **_fit_latency_bandwidth(write_pts),
            "label": f"write_fsync fs={fs}",
        },
        "disk_to_cpu": {
            **_fit_latency_bandwidth(read_pts),
            "label": f"warm_page_cached fs={fs}",
        },
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--block-bytes", type=int, default=458_752)  # 448 KiB
    ap.add_argument("--disk-path", default="results")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/transfer_tiers.json")
    args = ap.parse_args(argv)

    if not args.device.startswith("cuda"):
        print(
            "non-GPU run: pipeline validation only; constants do not graduate "
            "(docs/phase2-cost-model.md, section 6).",
            flush=True,
        )
    pathlib.Path(args.disk_path).mkdir(parents=True, exist_ok=True)
    tiers = {}
    tiers.update(measure_device_tiers(args.device, args.block_bytes, args.repeats))
    tiers.update(measure_disk_tiers(args.disk_path, args.block_bytes, args.repeats))
    for name, t in tiers.items():
        print(
            f"{name}: latency {t['latency_ms']:.3f} ms, "
            f"bandwidth {t['bandwidth_gb_s']:.2f} GB/s ({t['label']})",
            flush=True,
        )
    out = {
        "device": args.device,
        "block_bytes": args.block_bytes,
        "platform": platform.platform(),
        "filesystem": filesystem_label(args.disk_path),
        "graduates_to_cost_model": args.device.startswith("cuda"),
        # Plan section 3, scattered-transfer caveat: these are contiguous-copy
        # numbers, an upper bound on achievable migration bandwidth. The
        # chunked-copy probe on GPU day supplies the overhead coefficient.
        "transfer_measurement": "contiguous",
        "tiers": tiers,
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
