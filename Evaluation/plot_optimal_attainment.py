"""Plot CPCCD/NCO performance relative to the OR-Tools reference score.

Attainment uses the original OR-Tools-relative formula::

    attainment (%) = 100 * (1 - (ortools_score - algorithm_score)
                                  / abs(ortools_score))

When both scores are exactly zero, the algorithm ties the OR-Tools reference
and attainment is defined as 100%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("cpccd", "nco-custom")
DISPLAY_NAMES = {"cpccd": "CPCCD", "nco-custom": "NCO"}
COLORS = {"cpccd": "#F28E2B", "nco-custom": "#59A14F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mean OR-Tools attainment bar charts."
    )
    parser.add_argument("raw_results", type=Path, help="Path to raw_results.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <raw-results-directory>/plots/optimal_attainment",
    )
    return parser.parse_args()


def prepare_attainment(df: pd.DataFrame) -> pd.DataFrame:
    required = {"gamma", "instance_idx", "method", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    keys = ["gamma", "instance_idx"]
    reference_method = (
        "ortools-parallel"
        if df["method"].eq("ortools-parallel").any()
        else "ortools"
    )
    reference = (
        df.loc[df["method"].eq(reference_method), keys + ["score", "solver_status"]]
        .rename(columns={"score": "ortools_score", "solver_status": "ortools_status"})
        .drop_duplicates(keys)
    )
    algorithms = df.loc[df["method"].isin(METHODS), keys + ["method", "score"]]
    result = algorithms.merge(reference, on=keys, how="left", validate="many_to_one")

    if result["ortools_score"].isna().any():
        bad = result.loc[result["ortools_score"].isna(), keys].drop_duplicates()
        raise ValueError(f"Missing OR-Tools reference rows:\n{bad.to_string(index=False)}")
    denominator = result["ortools_score"].abs()
    degenerate = np.isclose(denominator.to_numpy(dtype=float), 0.0)
    tied_at_bound = degenerate & np.isclose(
        result["score"].to_numpy(dtype=float),
        result["ortools_score"].to_numpy(dtype=float),
    )
    safe_denominator = denominator.mask(degenerate, 1.0)
    result["attainment_pct"] = np.where(
        tied_at_bound,
        100.0,
        np.where(
            degenerate,
            np.nan,
            100.0
            * (1.0 - (result["ortools_score"] - result["score"]) / safe_denominator),
        ),
    )
    return result.sort_values(keys + ["method"]).reset_index(drop=True)


def _set_ratio_axis(ax, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    low = min(0.0, float(finite.min()) if finite.size else 0.0)
    high = max(100.0, float(finite.max()) if finite.size else 100.0)
    padding = max(6.0, 0.08 * (high - low or 100.0))
    ax.set_ylim(low - padding, high + padding)
    ax.axhline(100.0, color="#4E79A7", linewidth=1.5, linestyle="--", label="OR-Tools (100%)")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)


def save_plots(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float | str]] = []

    for gamma, gamma_df in df.groupby("gamma", sort=True):
        gamma_label = f"{float(gamma):g}".replace(".", "p")
        gamma_dir = output_dir / f"gamma_{gamma_label}"
        gamma_dir.mkdir(parents=True, exist_ok=True)

        stats = (
            gamma_df.groupby("method")["attainment_pct"]
            .agg(["mean", "std", "var", "count"])
            .reindex(METHODS)
            .dropna(subset=["mean"])
        )
        labels = [DISPLAY_NAMES[m] for m in stats.index]
        means = stats["mean"].to_numpy(dtype=float)
        colors = [COLORS[m] for m in stats.index]

        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        bars = ax.bar(labels, means, color=colors, width=0.58, alpha=0.82)
        _set_ratio_axis(ax, means)
        ax.set_ylabel("Mean OR-Tools attainment (%)")
        sample_sizes = sorted(stats["count"].astype(int).unique())
        sample_label = str(sample_sizes[0]) if len(sample_sizes) == 1 else "/".join(map(str, sample_sizes))
        ax.set_title(
            f"Mean attainment | gamma={float(gamma):g} | valid n={sample_label}"
        )
        ax.legend(loc="best", fontsize=8)
        for bar, (_, row) in zip(bars, stats.iterrows()):
            value = float(row["mean"])
            ax.annotate(
                f"{value:.2f}%",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5 if value >= 0 else -5),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )
        table_values = [
            [f"{value:.2f}" for value in stats["mean"]],
            [f"{value:.2f}" for value in stats["std"]],
            [f"{value:.2f}" for value in stats["var"]],
        ]
        table = ax.table(
            cellText=table_values,
            rowLabels=["Mean (%)", "Std (%p)", "Variance"],
            colLabels=labels,
            cellLoc="center",
            rowLoc="center",
            bbox=[0.16, -0.34, 0.76, 0.24],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for method, row in stats.iterrows():
            summary_rows.append(
                {
                    "gamma": float(gamma),
                    "method": method,
                    "attainment_mean_pct": float(row["mean"]),
                    "attainment_std_pct": float(row["std"]),
                    "attainment_variance": float(row["var"]),
                    "n": int(row["count"]),
                }
            )
        fig.subplots_adjust(bottom=0.28)
        fig.savefig(gamma_dir / "mean_with_std_variance.png", dpi=180)
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(output_dir / "attainment_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    raw_results = args.raw_results.resolve()
    output_dir = args.output_dir or raw_results.parent / "plots" / "optimal_attainment"
    df = prepare_attainment(pd.read_csv(raw_results))
    save_plots(df, output_dir)
    print(f"Saved attainment plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
