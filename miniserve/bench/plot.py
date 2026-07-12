"""Chart: throughput and p95 TTFT vs arrival rate, both arms, seed bands.

Refuses to plot invalid cells (run_valid_for_chart false or preemptions in a
main cell): a chart must never quietly average over a partially failed run.

Usage: uv run python -m miniserve.bench.plot --inp results/mechanism_sweep.jsonl
"""

import argparse
import json
import pathlib
from collections import defaultdict

# Validated palette (dataviz reference instance, light mode). Aqua is sub-3:1
# on the light surface: relief rule satisfied by direct labels + the results
# table printed alongside.
SERIES = {"continuous": "#2a78d6", "static": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/mechanism_sweep.jsonl")
    ap.add_argument("--out", default="results/mechanism_chart.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in load_rows(args.inp) if r["cell"] == "main"]
    bad = [r for r in rows if not r["valid_for_chart"]]
    if bad:
        raise SystemExit(
            f"refusing to chart: {len(bad)} invalid main cells "
            f"(aborts or preemptions present)"
        )

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["rate_rps"])].append(r)
    rates = sorted({rate for _, rate in cells})

    panels = (
        ("throughput_tok_s", "Throughput (tok/s)", False),
        ("ttft_p95_s", "p95 TTFT (s)", True),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label, logy) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        for arm, color in SERIES.items():
            means, lows, highs = [], [], []
            for rate in rates:
                vals = [r[key] for r in cells[(arm, rate)]]
                means.append(sum(vals) / len(vals))
                lows.append(min(vals))
                highs.append(max(vals))
            ax.plot(
                rates, means, color=color, linewidth=2, marker="o",
                markersize=6, label=arm,
            )
            ax.fill_between(rates, lows, highs, color=color, alpha=0.18, linewidth=0)
            # Direct label at the line's right end (relief for the aqua WARN).
            ax.annotate(
                arm, (rates[-1], means[-1]), xytext=(8, 0),
                textcoords="offset points", color=color, fontsize=10,
                fontweight="bold", va="center",
            )
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(rates)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("arrival rate (req/s)", color=INK_2)
        ax.set_ylabel(label, color=INK_2)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.margins(x=0.06)
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper left")

    fig.suptitle(
        "Continuous vs static batching under bursty Poisson load",
        color=INK, fontsize=13, fontweight="bold",
    )
    cfg = rows[0]["config"]
    fig.text(
        0.01, 0.005,
        (
            f"Flat-cost fake model ({cfg['delay_s'] * 1000:.0f} ms/step): batching is "
            "free, the GPU regime. Static gets its best timeout (W=0, never "
            "fill-waits): the gap is run-to-completion slot waste only.\n"
            f"{cfg['n_requests']} requests/cell, burst factor "
            f"{cfg['burst_factor']}, max_batch {cfg['max_batch']}, 3 seeds; band = "
            "min-max across seeds. All cells preemption-free; driver lag in results."
        ),
        color=INK_2, fontsize=7.5, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"chart -> {out}")

    # Table view (the relief rule's second leg, and the numbers behind the chart).
    print(
        f"\n{'cell':9} {'arm':11} {'rate':>5} {'tput tok/s':>11} "
        f"{'p95 TTFT s':>11} {'p95 lag ms':>11} {'preempt':>7}"
    )
    ordering = lambda r: (r["cell"], r["rate_rps"], r["arm"], r["seed"])  # noqa: E731
    for r in sorted(load_rows(args.inp), key=ordering):
        ttft = f"{r['ttft_p95_s']:.3f}" if r["ttft_p95_s"] is not None else "n/a"
        lag = f"{r['sched_lag_p95_s'] * 1000:.2f}" if r["sched_lag_p95_s"] is not None else "n/a"
        print(
            f"{r['cell']:9} {r['arm']:11} {r['rate_rps']:>5} "
            f"{r['throughput_tok_s']:>11.1f} {ttft:>11} {lag:>11} "
            f"{r['preemptions_total']:>7}"
        )


if __name__ == "__main__":
    main()
