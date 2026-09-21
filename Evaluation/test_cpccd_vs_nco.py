"""Paired statistical comparison of CPCCD and NCO scores across PC sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm family-wise error correction without an extra dependency."""
    values = pvalues.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)
    return pd.Series(adjusted, index=pvalues.index)


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def compare_file(raw_results: Path, num_parts: int) -> list[dict[str, float | int | str]]:
    data = pd.read_csv(raw_results)
    selected = data[data["method"].isin(["cpccd", "nco-custom"])].copy()
    rows: list[dict[str, float | int | str]] = []

    for gamma, gamma_df in selected.groupby("gamma", sort=True):
        paired = gamma_df.pivot(
            index="instance_idx", columns="method", values="score"
        ).dropna(subset=["cpccd", "nco-custom"])
        cpccd = paired["cpccd"].to_numpy(dtype=float)
        nco = paired["nco-custom"].to_numpy(dtype=float)
        diff = nco - cpccd
        count = len(diff)
        mean_diff = float(diff.mean())
        std_diff = float(diff.std(ddof=1)) if count > 1 else np.nan
        sem = std_diff / np.sqrt(count) if count > 1 else np.nan
        critical = stats.t.ppf(0.975, count - 1) if count > 1 else np.nan

        t_result = stats.ttest_rel(nco, cpccd)
        try:
            w_result = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            wilcoxon_stat = float(w_result.statistic)
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0

        shapiro_p = float(stats.shapiro(diff).pvalue) if 3 <= count <= 5000 else np.nan
        rows.append(
            {
                "num_parts": num_parts,
                "gamma": float(gamma),
                "n_pairs": count,
                "cpccd_mean": float(cpccd.mean()),
                "cpccd_std": float(cpccd.std(ddof=1)),
                "nco_mean": float(nco.mean()),
                "nco_std": float(nco.std(ddof=1)),
                "mean_diff_nco_minus_cpccd": mean_diff,
                "median_diff_nco_minus_cpccd": float(np.median(diff)),
                "mean_diff_ci95_low": mean_diff - critical * sem,
                "mean_diff_ci95_high": mean_diff + critical * sem,
                "nco_wins": int((diff > 0).sum()),
                "ties": int(np.isclose(diff, 0.0).sum()),
                "cpccd_wins": int((diff < 0).sum()),
                "shapiro_diff_p": shapiro_p,
                "paired_t_stat": float(t_result.statistic),
                "paired_t_p": float(t_result.pvalue),
                "cohen_dz": mean_diff / std_diff if std_diff > 0 else np.nan,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "rank_biserial": rank_biserial(diff),
                "raw_results": str(raw_results.resolve()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/evaluation/gamma_sweep"),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--num-parts", default="30,40,50")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/gamma_sweep/statistical_tests"),
    )
    args = parser.parse_args()

    rows = []
    for value in args.num_parts.split(","):
        num_parts = int(value.strip())
        folder = (
            args.base_dir
            / f"n{num_parts}_seed{args.seed}_test{args.test_size}_paper(ortools_5m)"
        )
        raw_results = folder / "raw_results.csv"
        if not raw_results.exists():
            raise FileNotFoundError(raw_results)
        result_rows = compare_file(raw_results, num_parts)
        rows.extend(result_rows)
        per_size = pd.DataFrame(result_rows)
        stats_dir = folder / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)
        per_size.to_csv(stats_dir / "cpccd_vs_nco_unadjusted.csv", index=False)

    result = pd.DataFrame(rows).sort_values(["num_parts", "gamma"]).reset_index(drop=True)
    result["paired_t_p_holm"] = holm_adjust(result["paired_t_p"])
    result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p"])
    result["wilcoxon_significant_holm_0p05"] = result["wilcoxon_p_holm"] < 0.05
    result["favored_by_mean"] = np.select(
        [result["mean_diff_nco_minus_cpccd"] > 0, result["mean_diff_nco_minus_cpccd"] < 0],
        ["NCO", "CPCCD"],
        default="Tie",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "cpccd_vs_nco_paired_tests.csv"
    result.to_csv(output, index=False)
    columns = [
        "num_parts", "gamma", "n_pairs", "cpccd_mean", "nco_mean",
        "mean_diff_nco_minus_cpccd", "mean_diff_ci95_low", "mean_diff_ci95_high",
        "nco_wins", "ties", "cpccd_wins", "wilcoxon_p", "wilcoxon_p_holm",
        "rank_biserial", "paired_t_p", "paired_t_p_holm", "favored_by_mean",
    ]
    print(result[columns].to_string(index=False))
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()
