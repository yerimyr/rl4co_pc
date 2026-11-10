"""Paired statistical tests for curriculum range evaluation results."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METHODS = ("ortools", "cpccd", "mcl", "nco-curriculum")
METHOD_LABELS = {
    "ortools": "CP-SAT",
    "cpccd": "CPCCD",
    "mcl": "MCL",
    "nco-curriculum": "RL-based NCO",
}
TIE_ATOL = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


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
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def validate_input(data: pd.DataFrame) -> None:
    required = {"test_range", "gamma", "method", "instance_idx", "score"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    unknown = sorted(set(data["method"].unique()) - set(METHODS))
    if unknown:
        raise ValueError(f"Unexpected methods: {unknown}")
    duplicates = data.duplicated(
        subset=["test_range", "gamma", "method", "instance_idx"]
    )
    if duplicates.any():
        raise ValueError(f"Duplicate result rows: {int(duplicates.sum())}")


def effect_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.1:
        return "negligible"
    if magnitude < 0.3:
        return "small"
    if magnitude < 0.5:
        return "medium"
    return "large"


def run_tests(data: pd.DataFrame, alpha: float):
    descriptive_rows = []
    friedman_rows = []
    pairwise_rows = []
    integrity_rows = []

    grouped = data.groupby(["test_range", "gamma"], sort=True)
    for (test_range, gamma), group in grouped:
        counts = group.groupby("method")["instance_idx"].nunique().to_dict()
        pivot = group.pivot(index="instance_idx", columns="method", values="score")
        paired = pivot.dropna(subset=list(METHODS))[list(METHODS)]
        integrity_rows.append(
            {
                "test_range": test_range,
                "gamma": gamma,
                **{f"{method}_n": int(counts.get(method, 0)) for method in METHODS},
                "complete_pairs": int(len(paired)),
                "all_methods_have_identical_instance_ids": bool(
                    len(paired) == len(pivot)
                    and all(counts.get(method, 0) == len(paired) for method in METHODS)
                ),
            }
        )
        if len(paired) == 0:
            raise ValueError(f"No complete pairs: {test_range}, gamma={gamma}")

        for method in METHODS:
            values = paired[method].to_numpy(float)
            descriptive_rows.append(
                {
                    "test_range": test_range,
                    "gamma": gamma,
                    "method": method,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

        friedman = stats.friedmanchisquare(*(paired[method] for method in METHODS))
        kendalls_w = float(friedman.statistic) / (len(paired) * (len(METHODS) - 1))
        friedman_rows.append(
            {
                "test_range": test_range,
                "gamma": gamma,
                "n_pairs": len(paired),
                "friedman_stat": float(friedman.statistic),
                "friedman_p": float(friedman.pvalue),
                "kendalls_w": kendalls_w,
            }
        )

        condition_rows = []
        for method_a, method_b in combinations(METHODS, 2):
            a = paired[method_a].to_numpy(float)
            b = paired[method_b].to_numpy(float)
            diff = a - b
            tied = np.isclose(diff, 0.0, rtol=0.0, atol=TIE_ATOL)
            test_diff = diff.copy()
            test_diff[tied] = 0.0
            try:
                result = stats.wilcoxon(
                    test_diff,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(result.statistic)
                pvalue = float(result.pvalue)
            except ValueError:
                statistic, pvalue = 0.0, 1.0
            rbc = rank_biserial(test_diff)
            if np.isclose(diff.mean(), 0.0, atol=TIE_ATOL):
                favored = "tie"
            else:
                favored = method_a if diff.mean() > 0 else method_b
            condition_rows.append(
                {
                    "test_range": test_range,
                    "gamma": gamma,
                    "n_pairs": len(diff),
                    "method_a": method_a,
                    "method_b": method_b,
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "median_diff_a_minus_b": float(np.median(diff)),
                    "a_wins": int(((diff > 0) & ~tied).sum()),
                    "ties": int(tied.sum()),
                    "b_wins": int(((diff < 0) & ~tied).sum()),
                    "wilcoxon_stat": statistic,
                    "wilcoxon_p": pvalue,
                    "rank_biserial_a_minus_b": rbc,
                    "effect_magnitude": effect_label(rbc),
                    "favored_by_mean": favored,
                }
            )
        condition = pd.DataFrame(condition_rows)
        condition["wilcoxon_p_holm_within_condition"] = holm_adjust(
            condition["wilcoxon_p"]
        )
        pairwise_rows.extend(condition.to_dict("records"))

    descriptives = pd.DataFrame(descriptive_rows)
    friedman = pd.DataFrame(friedman_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    integrity = pd.DataFrame(integrity_rows)

    friedman["friedman_p_holm_global"] = holm_adjust(friedman["friedman_p"])
    friedman["significant_raw_0p05"] = friedman["friedman_p"] < alpha
    friedman["significant_holm_global_0p05"] = (
        friedman["friedman_p_holm_global"] < alpha
    )

    pairwise["wilcoxon_p_holm_global"] = holm_adjust(pairwise["wilcoxon_p"])
    pairwise = pairwise.merge(
        friedman[["test_range", "gamma", "friedman_p", "significant_raw_0p05"]],
        on=["test_range", "gamma"],
        how="left",
    )
    pairwise["significant_holm_within_condition_0p05"] = (
        pairwise["wilcoxon_p_holm_within_condition"] < alpha
    )
    pairwise["significant_after_friedman_and_holm_0p05"] = (
        pairwise["significant_raw_0p05"]
        & pairwise["significant_holm_within_condition_0p05"]
    )
    pairwise["significant_holm_global_0p05"] = (
        pairwise["wilcoxon_p_holm_global"] < alpha
    )
    pairwise["conclusion_within_condition"] = np.where(
        pairwise["significant_after_friedman_and_holm_0p05"],
        pairwise["favored_by_mean"] + " significantly better",
        "No significant difference",
    )
    pairwise["conclusion_global"] = np.where(
        pairwise["significant_holm_global_0p05"],
        pairwise["favored_by_mean"] + " significantly better",
        "No significant difference",
    )
    return descriptives, friedman, pairwise, integrity


def save_heatmap(pairwise: pd.DataFrame, output: Path) -> None:
    pair_order = list(combinations(METHODS, 2))
    ranges = list(dict.fromkeys(pairwise["test_range"]))
    gammas = sorted(pairwise["gamma"].unique())
    fig, axes = plt.subplots(len(ranges), 1, figsize=(13.5, 3.7 * len(ranges)))
    if len(ranges) == 1:
        axes = [axes]
    image = None
    for ax, test_range in zip(axes, ranges):
        subset = pairwise[pairwise["test_range"] == test_range]
        matrix = np.full((len(pair_order), len(gammas)), np.nan)
        annotations = [["" for _ in gammas] for _ in pair_order]
        for row_idx, (method_a, method_b) in enumerate(pair_order):
            for col_idx, gamma in enumerate(gammas):
                row = subset[
                    (subset["method_a"] == method_a)
                    & (subset["method_b"] == method_b)
                    & np.isclose(subset["gamma"], gamma)
                ].iloc[0]
                matrix[row_idx, col_idx] = row["rank_biserial_a_minus_b"]
                star = "*" if row["significant_after_friedman_and_holm_0p05"] else ""
                annotations[row_idx][col_idx] = (
                    f"{row['rank_biserial_a_minus_b']:.2f}{star}\n"
                    f"p={row['wilcoxon_p_holm_within_condition']:.3g}"
                )
        image = ax.imshow(matrix, cmap="RdBu", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(gammas)), [f"{gamma:g}" for gamma in gammas])
        ax.set_yticks(
            range(len(pair_order)),
            [f"{METHOD_LABELS[a]} vs {METHOD_LABELS[b]}" for a, b in pair_order],
        )
        ax.set_xlabel("Gamma")
        ax.set_title(test_range)
        for row_idx in range(len(pair_order)):
            for col_idx in range(len(gammas)):
                ax.text(
                    col_idx,
                    row_idx,
                    annotations[row_idx][col_idx],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(matrix[row_idx, col_idx]) > 0.55 else "black",
                )
    fig.subplots_adjust(left=0.2, right=0.82, top=0.88, bottom=0.07, hspace=0.55)
    colorbar_ax = fig.add_axes([0.86, 0.17, 0.018, 0.64])
    fig.colorbar(
        image,
        cax=colorbar_ax,
        label="Rank-biserial effect (positive favors first method)",
    )
    fig.suptitle("Pairwise Wilcoxon tests with within-condition Holm correction\n* p < 0.05 after Friedman and Holm", fontsize=14)
    fig.savefig(output / "pairwise_rank_biserial_holm_heatmap.png", dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    friedman: pd.DataFrame,
    pairwise: pd.DataFrame,
    alpha: float,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.select_dtypes(include=["float"]).columns:
            display[column] = display[column].map(lambda value: f"{value:.6g}")
        headers = [str(column) for column in display.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in display.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    pairwise_report = pairwise[
        [
            "test_range",
            "gamma",
            "method_a",
            "method_b",
            "mean_diff_a_minus_b",
            "a_wins",
            "ties",
            "b_wins",
            "wilcoxon_p_holm_within_condition",
            "rank_biserial_a_minus_b",
            "conclusion_within_condition",
        ]
    ]
    lines = [
        "# Curriculum range statistical tests",
        "",
        f"- Significance level: alpha={alpha}",
        f"- Omnibus test: Friedman test (paired instances, {len(METHODS)} algorithms)",
        "- Post-hoc test: two-sided Wilcoxon signed-rank test",
        "- Multiple comparisons: Holm correction within each range x gamma condition",
        f"- Additional conservative correction: global Holm correction across all {len(pairwise)} pairwise tests",
        "- Error/effect direction: positive rank-biserial favors method A",
        "",
        "## Friedman tests",
        "",
        markdown_table(friedman),
        "",
        "## Pairwise conclusions",
        "",
        markdown_table(pairwise_report),
        "",
    ]
    (output / "statistical_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input.resolve())
    validate_input(data)
    descriptives, friedman, pairwise, integrity = run_tests(data, args.alpha)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    descriptives.to_csv(output / "descriptive_statistics.csv", index=False)
    friedman.to_csv(output / "friedman_tests.csv", index=False)
    pairwise.to_csv(output / "pairwise_wilcoxon_holm.csv", index=False)
    integrity.to_csv(output / "paired_data_integrity.csv", index=False)
    if "solver_status" in data.columns:
        (
            data[data["method"] == "ortools"]
            .groupby(["test_range", "gamma", "solver_status"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .to_csv(output / "ortools_status_counts.csv", index=False)
        )
    conclusion_columns = [
        "test_range",
        "gamma",
        "method_a",
        "method_b",
        "mean_diff_a_minus_b",
        "a_wins",
        "ties",
        "b_wins",
        "wilcoxon_p_holm_within_condition",
        "wilcoxon_p_holm_global",
        "rank_biserial_a_minus_b",
        "effect_magnitude",
        "conclusion_within_condition",
        "conclusion_global",
    ]
    pairwise[conclusion_columns].to_csv(output / "conclusion_summary.csv", index=False)
    save_heatmap(pairwise, output)
    write_report(output, friedman, pairwise, args.alpha)
    print(f"Saved statistical tests: {output}")
    print(f"Conditions: {len(friedman)}, pairwise comparisons: {len(pairwise)}")


if __name__ == "__main__":
    main()
