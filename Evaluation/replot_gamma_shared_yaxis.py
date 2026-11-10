"""Regenerate gamma score plots with shared y-axis limits across result folders."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from Evaluation.gamma_sweep import save_gamma_boxplot, save_gamma_score_barplot


def rounded_limits(low: float, high: float, step: float = 50.0) -> tuple[float, float]:
    span = max(high - low, step)
    padding = 0.05 * span
    return (
        step * math.floor((low - padding) / step),
        step * math.ceil((high + padding) / step),
    )


def valid_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path / "raw_results.csv")
    invalid_ortools = (
        df["method"].astype(str).str.startswith("ortools")
        & ~df["solver_status"].isin(["OPTIMAL", "FEASIBLE"])
    )
    return df.loc[~invalid_ortools].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    datasets = [(path, valid_rows(path)) for path in args.result_dirs]
    combined = pd.concat([df for _, df in datasets], ignore_index=True)

    box_limits = rounded_limits(float(combined["score"].min()), float(combined["score"].max()))
    summary = combined.groupby(["gamma", "method",], as_index=False)["score"].agg(["mean", "std"])
    bar_limits = rounded_limits(
        float((summary["mean"] - summary["std"]).min()),
        float((summary["mean"] + summary["std"]).max()),
    )

    print(f"Shared bar-plot y-axis: {bar_limits}")
    print(f"Shared boxplot y-axis: {box_limits}")
    for path, df in datasets:
        save_gamma_score_barplot(df, path / "plots", y_limits=bar_limits)
        save_gamma_boxplot(df, path / "plots", y_limits=box_limits)


if __name__ == "__main__":
    main()
