"""Paired tests for serial OR-Tools, CPCCD, and NCO on PC gamma sweeps."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METHODS = ("ortools", "cpccd", "nco-custom")
TIE_ATOL = 1e-9


def holm_adjust(values: pd.Series) -> pd.Series:
    pvalues = values.to_numpy(dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running_max = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * pvalues[index])
        adjusted[index] = min(running_max, 1.0)
    return pd.Series(adjusted, index=values.index)


def rank_biserial(diff: np.ndarray) -> float:
    nonzero = diff[~np.isclose(diff, 0.0, rtol=0.0, atol=TIE_ATOL)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/evaluation/gamma_sweep"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--num-parts", default="20,30,40")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    descriptive_rows = []
    friedman_rows = []
    pairwise_rows = []
    status_rows = []

    for value in args.num_parts.split(","):
        num_parts = int(value.strip())
        raw_path = (
            args.base_dir
            / f"n{num_parts}_seed{args.seed}_test{args.test_size}_paper(ortools_5m)"
            / "raw_results.csv"
        )
        data = pd.read_csv(raw_path)
        data = data[data["method"].isin(METHODS)].copy()

        serial = data[data["method"] == "ortools"]
        for (gamma, status), group in serial.groupby(["gamma", "solver_status"], dropna=False):
            status_rows.append(
                {"num_parts": num_parts, "gamma": gamma, "solver_status": status, "count": len(group)}
            )

        for gamma, group in data.groupby("gamma", sort=True):
            pivot = group.pivot(index="instance_idx", columns="method", values="score")
            pivot = pivot.dropna(subset=list(METHODS))
            if len(pivot) == 0:
                raise ValueError(f"No complete pairs for N={num_parts}, gamma={gamma}")

            for method in METHODS:
                values = pivot[method].to_numpy(float)
                descriptive_rows.append(
                    {
                        "num_parts": num_parts,
                        "gamma": gamma,
                        "method": method,
                        "n": len(values),
                        "mean": values.mean(),
                        "std": values.std(ddof=1),
                        "median": np.median(values),
                    }
                )

            friedman = stats.friedmanchisquare(*(pivot[method] for method in METHODS))
            friedman_rows.append(
                {
                    "num_parts": num_parts,
                    "gamma": gamma,
                    "n_pairs": len(pivot),
                    "friedman_stat": friedman.statistic,
                    "friedman_p": friedman.pvalue,
                }
            )

            condition_rows = []
            for method_a, method_b in combinations(METHODS, 2):
                a = pivot[method_a].to_numpy(float)
                b = pivot[method_b].to_numpy(float)
                diff = a - b
                tied = np.isclose(diff, 0.0, rtol=0.0, atol=TIE_ATOL)
                # Use the same numerical-zero definition for the test, effect
                # size, and mutually exclusive win/tie/loss counts.
                test_diff = diff.copy()
                test_diff[tied] = 0.0
                try:
                    test = stats.wilcoxon(
                        test_diff, alternative="two-sided", zero_method="wilcox"
                    )
                    statistic, pvalue = float(test.statistic), float(test.pvalue)
                except ValueError:
                    statistic, pvalue = 0.0, 1.0
                condition_rows.append(
                    {
                        "num_parts": num_parts,
                        "gamma": gamma,
                        "n_pairs": len(diff),
                        "method_a": method_a,
                        "method_b": method_b,
                        "mean_a": a.mean(),
                        "mean_b": b.mean(),
                        "mean_diff_a_minus_b": diff.mean(),
                        "median_diff_a_minus_b": np.median(diff),
                        "a_wins": int(((diff > 0) & ~tied).sum()),
                        "ties": int(tied.sum()),
                        "b_wins": int(((diff < 0) & ~tied).sum()),
                        "wilcoxon_stat": statistic,
                        "wilcoxon_p": pvalue,
                        "rank_biserial_a_minus_b": rank_biserial(test_diff),
                    }
                )
            condition = pd.DataFrame(condition_rows)
            condition["wilcoxon_p_holm_within_condition"] = holm_adjust(condition["wilcoxon_p"])
            pairwise_rows.extend(condition.to_dict("records"))

    descriptives = pd.DataFrame(descriptive_rows)
    friedman = pd.DataFrame(friedman_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["wilcoxon_p_holm_global"] = holm_adjust(pairwise["wilcoxon_p"])
    pairwise["significant_holm_0p05"] = pairwise["wilcoxon_p_holm_within_condition"] < 0.05
    pairwise["favored_method"] = np.where(
        pairwise["mean_diff_a_minus_b"] > 0, pairwise["method_a"], pairwise["method_b"]
    )
    pairwise["conclusion"] = np.where(
        pairwise["significant_holm_0p05"],
        pairwise["favored_method"] + " significantly better",
        "No significant difference",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptives.to_csv(args.output_dir / "descriptive_statistics.csv", index=False)
    friedman.to_csv(args.output_dir / "friedman_tests.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_wilcoxon_holm.csv", index=False)
    pd.DataFrame(status_rows).to_csv(args.output_dir / "ortools_serial_status_counts.csv", index=False)
    pairwise[
        [
            "num_parts", "gamma", "method_a", "method_b", "mean_diff_a_minus_b",
            "wilcoxon_p_holm_within_condition", "rank_biserial_a_minus_b", "conclusion",
        ]
    ].to_csv(args.output_dir / "conclusion_summary.csv", index=False)
    print(pairwise.to_string(index=False))
    print(f"\nSaved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
