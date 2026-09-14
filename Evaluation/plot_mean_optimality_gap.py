"""Create one mean OR-Tools optimality-gap chart from gamma-sweep results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("cpccd", "nco-custom")
LABELS = {"cpccd": "CPCCD", "nco-custom": "NCO"}
COLORS = {"cpccd": "#F28E2B", "nco-custom": "#59A14F"}


def summarize_gaps(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["gamma", "instance_idx"]
    reference = (
        df.loc[df["method"].eq("ortools"), keys + ["score"]]
        .rename(columns={"score": "ortools_score"})
        .drop_duplicates(keys)
    )
    values = df.loc[df["method"].isin(METHODS), keys + ["method", "score"]]
    values = values.merge(reference, on=keys, how="left", validate="many_to_one")
    if values["ortools_score"].isna().any():
        raise ValueError("Some algorithm rows have no matching OR-Tools reference.")
    values["optimality_gap"] = values["ortools_score"] - values["score"]
    return (
        values.groupby(["gamma", "method"])["optimality_gap"]
        .agg(gap_mean="mean", gap_std="std", gap_variance="var", n="count")
        .reset_index()
        .sort_values(["gamma", "method"])
    )


def save_mean_gap_plot(summary: pd.DataFrame, output_dir: Path, dataset_name: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gammas = sorted(summary["gamma"].astype(float).unique())
    x = np.arange(len(gammas), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    for index, method in enumerate(METHODS):
        method_rows = summary.loc[summary["method"].eq(method)].set_index("gamma")
        means = np.array([method_rows.loc[gamma, "gap_mean"] for gamma in gammas])
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            means,
            width,
            label=LABELS[method],
            color=COLORS[method],
            alpha=0.88,
        )
        for bar, value in zip(bars, means):
            ax.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if value >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    sample_sizes = sorted(summary["n"].astype(int).unique())
    sample_label = str(sample_sizes[0]) if len(sample_sizes) == 1 else "/".join(map(str, sample_sizes))
    ax.axhline(
        0.0,
        color="#4E79A7",
        linestyle="--",
        linewidth=1.5,
        label="OR-Tools gap = 0",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{gamma:g}" for gamma in gammas])
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Mean optimality gap (OR-Tools score - algorithm score)")
    ax.set_title(f"Mean OR-Tools optimality gap | {dataset_name} | n={sample_label}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mean_optimality_gap_mean_only.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    raw_results = args.raw_results.resolve()
    output_dir = args.output_dir or raw_results.parent / "plots" / "optimality_gap"
    summary = summarize_gaps(pd.read_csv(raw_results))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "optimality_gap_summary.csv", index=False)
    plot_path = save_mean_gap_plot(summary, output_dir, raw_results.parent.name)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
