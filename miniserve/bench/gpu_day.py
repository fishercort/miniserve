"""GPU-day orchestrator (plan section 7): the whole batched session in one
command, so nothing gets discovered while the meter runs.

Sequence: hardware capture -> curve A (gate-checked) -> curve B -> curve C
plus chunked overhead -> activation peaks -> config assembly -> spot
validation. Everything prints to stdout AND results/gpu_day/session.log;
the same-day commit is the operator's last step and the script ends by
printing exactly what to commit.

The model loads once and is reused across curves A, B, and spot validation.

Usage (on the rental):
    uv run python -m miniserve.bench.gpu_day --form-factor SXM
"""

import argparse
import datetime
import json
import pathlib
import platform
import subprocess
import sys


class _Tee:
    def __init__(self, path: pathlib.Path):
        self.file = path.open("w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def capture_hardware(form_factor: str) -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        driver = "unknown"
    return {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "form_factor": form_factor,
            "driver": driver,
            "vram_gb": round(props.total_memory / 2**30, 1),
        },
        "filesystem": "recorded_by_transfer_probe",
    }


def git_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--form-factor", required=True, choices=["SXM", "PCIe"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/gpu_day")
    args = ap.parse_args(argv)

    import torch

    if not args.device.startswith("cuda"):
        raise SystemExit("gpu_day runs on the rental only (plan section 6)")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / "session.log")  # session log, same-day committed

    from miniserve.bench import activation_probe, spot_validate, transfer_probe
    from miniserve.bench.calibrate import fit_curves, measure_prefill
    from miniserve.bench.costmodel import crossover_band, emit_config
    from miniserve.bench.decode_probe import measure_decode_cell
    from miniserve.model import PagedModel

    hardware = capture_hardware(args.form_factor)
    print(f"hardware: {json.dumps(hardware['gpu'])}", flush=True)
    dtype = torch.bfloat16
    pm = PagedModel.from_pretrained(args.model, device=args.device, dtype=dtype)

    # Curve A, gate-checked before anything else spends meter time on it.
    print("== curve A: prefill ==", flush=True)
    records = measure_prefill(
        pm, [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        repeats=3, device=args.device, dtype=dtype,
    )
    fits = fit_curves(records)
    (out_dir / "prefill.json").write_text(json.dumps({"raw": records, **fits}, indent=2))
    print(f"prefill fit: preferred={fits['preferred']} "
          f"acceptable={fits['fit_acceptable']}", flush=True)
    if not fits["fit_acceptable"]:
        raise SystemExit(
            "curve A failed its gate (plan section 1): densify the grid and "
            "rerun before continuing; do not proceed on a bad fit"
        )

    print("== curve B: decode ==", flush=True)
    cells = []
    for target in (256, 1024, 4096):
        for batch in (1, 2, 4, 8):
            cell = measure_decode_cell(
                pm, batch, target, repeats=3, device=args.device, dtype=dtype
            )
            cells.append(cell)
            print(f"batch={batch} ctx={target}: {cell['decode_ms_median']:.2f} ms",
                  flush=True)
    (out_dir / "decode.json").write_text(json.dumps({"cells": cells}, indent=2))

    print("== curve C: transfers + chunked overhead ==", flush=True)
    transfer_probe.main([
        "--device", args.device, "--disk-path", str(out_dir),
        "--out", str(out_dir / "transfer.json"),
    ])
    transfer = json.loads((out_dir / "transfer.json").read_text())
    hardware["filesystem"] = transfer["filesystem"]

    print("== activation peaks ==", flush=True)
    activation_probe.main([
        "--model", args.model, "--device", args.device,
        "--out", str(out_dir / "activation_peak.json"),
    ])

    print("== config assembly ==", flush=True)
    config = emit_config(
        model_name=args.model,
        model_dims={
            "num_layers": pm.num_layers,
            "num_kv_heads": pm.n_kv_heads,
            "head_dim": pm.head_dim,
            "dtype_bytes": 2,
        },
        prefill_fit=fits,
        decode_table=cells,
        transfer_tiers=transfer["tiers"],
        hardware=hardware,
        provenance={
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "platform": platform.platform(),
            "commit": git_commit_hash(),
        },
    )
    config["transfer_measurement"] = transfer["transfer_measurement"]
    config["chunked_overhead"] = transfer["chunked_overhead"]
    config_path = out_dir / "cost_model.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    for tier in config["transfer"]:
        band = crossover_band(config, tier)
        print(f"{tier}: band={band}", flush=True)

    print("== spot validation ==", flush=True)
    spot_validate.main([
        "--config", str(config_path), "--model", args.model,
        "--device", args.device, "--out", str(out_dir / "spot_validation.json"),
    ])

    print(
        "\nSAME-DAY COMMIT (plan section 7):\n"
        f"  git add {args.out_dir} && git commit -m "
        f"'phase2: GPU day on {hardware['gpu']['name']} ({args.form_factor})' "
        "&& git push",
        flush=True,
    )


if __name__ == "__main__":
    main()
