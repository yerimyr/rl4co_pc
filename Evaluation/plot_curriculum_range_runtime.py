"""Plot mean per-instance runtime by test-size range and algorithm."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANGES = ("N10-20", "N21-30", "N31-40")
METHODS = ("ortools", "cpccd", "nco-curriculum")
LABELS = {
    "ortools": "OR-Tools (serial)",
    "cpccd": "CPCCD",
    "nco-curriculum": "NCO (curriculum)",
}
COLORS = {
    "ortools": "#4e9bc6",
    "cpccd": "#f28e2b",
    "nco-curriculum": "#9467bd",
}
MARKERS = {"ortools": "o", "cpccd": "s", "nco-curriculum": "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data[
        data["test_range"].isin(RANGES) & data["method"].isin(METHODS)
    ].copy()
    return (
        filtered.groupby(["test_range", "method"], as_index=False)
        .agg(
            n=("wall_elapsed_sec", "size"),
            mean_sec=("wall_elapsed_sec", "mean"),
            std_sec=("wall_elapsed_sec", "std"),
            sem_sec=("wall_elapsed_sec", "sem"),
            median_sec=("wall_elapsed_sec", "median"),
            min_sec=("wall_elapsed_sec", "min"),
            max_sec=("wall_elapsed_sec", "max"),
        )
        .sort_values(["test_range", "method"])
    )


def time_label(seconds: float) -> str:
    if seconds < 0.01:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 1.0:
        return f"{seconds:.3f} s"
    return f"{seconds:.1f} s"


def draw(summary: pd.DataFrame, output_path: Path, log_scale: bool) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    x = np.arange(len(RANGES))
    for method in METHODS:
        values = (
            summary[summary["method"] == method]
            .set_index("test_range")
            .reindex(RANGES)
        )
        y = values["mean_sec"].to_numpy(float)
        ax.plot(
            x,
            y,
            marker=MARKERS[method],
            markersize=8,
            linewidth=2.3,
            color=COLORS[method],
            label=LABELS[method],
        )
        for x_value, y_value in zip(x, y):
            if not log_scale and method == "cpccd":
                offset, alignment = (-10, 8), "right"
            elif not log_scale and method == "nco-curriculum":
                offset, alignment = (10, 8), "left"
            else:
                offset, alignment = (0, 8), "center"
            ax.annotate(
                time_label(y_value),
                (x_value, y_value),
                xytext=offset,
                textcoords="offset points",
                ha=alignment,
                fontsize=9,
                color=COLORS[method],
            )
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Mean wall-clock time per instance (seconds, log scale)")
        scale_note = "logarithmic y-axis"
    else:
        ax.set_ylabel("Mean wall-clock time per instance (seconds)")
        scale_note = "linear y-axis"
    ax.set_xticks(x, RANGES)
    ax.set_xlabel("Test-size range")
    ax.set_title(
        "Mean solution time per instance by algorithm and size range\n"
        f"Averaged over 5 gammas × 100 instances ({scale_note})"
    )
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(title="Algorithm")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input.resolve())
    required = {"test_range", "method", "wall_elapsed_sec"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    summary = summarize(data)
    expected = len(RANGES) * len(METHODS)
    if len(summary) != expected:
        raise ValueError(f"Expected {expected} range-method groups, found {len(summary)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "runtime_per_instance_summary.csv", index=False)
    draw(summary, output / "runtime_per_instance_by_range_linear.png", log_scale=False)
    draw(summary, output / "runtime_per_instance_by_range_log.png", log_scale=True)
    print(f"Saved runtime plots: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
