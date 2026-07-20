from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

try:
    import pandas as pd
    from scipy.stats import friedmanchisquare, wilcoxon
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing dependency: {missing}. Install required packages with: "
        "pip install pandas scipy matplotlib"
    ) from exc


DEFAULT_CSV = Path("outputs/pc20_compare_100.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/statistical_hypothesis_testing")
DEFAULT_METHODS = ["cpccd", "sa", "ga", "nco"]
DEFAULT_METRICS = ["score", "num_groups", "wall_elapsed_sec"]
METRIC_ALIASES = {
    "groups": "num_groups",
    "time": "wall_elapsed_sec",
    "wall_time": "wall_elapsed_sec",
    "solver_time": "solver_elapsed_sec",
}


def parse_list(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip() for item in value.split(",") if item.strip())
    return parsed


def normalize_metric(metric: str) -> str:
    return METRIC_ALIASES.get(metric, metric)


def holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (original_idx, p_value) in enumerate(indexed):
        adjusted_p = (m - rank) * p_value
        running_max = max(running_max, adjusted_p)
        adjusted[original_idx] = min(running_max, 1.0)
    return adjusted


def rank_biserial_from_wilcoxon(x: pd.Series, y: pd.Series) -> float:
    diffs = (x - y).astype(float)
    diffs = diffs[diffs != 0.0]
    n = len(diffs)
    if n == 0:
        return 0.0

    abs_ranks = diffs.abs().rank(method="average")
    pos_rank_sum = float(abs_ranks[diffs > 0].sum())
    neg_rank_sum = float(abs_ranks[diffs < 0].sum())
    total_rank_sum = n * (n + 1) / 2.0
    if total_rank_sum == 0:
        return 0.0
    return (pos_rank_sum - neg_rank_sum) / total_rank_sum


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"algorithm", "instance_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["algorithm"] = df["algorithm"].astype(str).str.lower()
    df["instance_idx"] = df["instance_idx"].astype(int)
    return df


def build_pivot(df: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    metric = normalize_metric(metric)
    methods = [method.lower() for method in methods]

    if metric not in df.columns:
        raise ValueError(f"Metric column not found: {metric}")

    filtered = df[df["algorithm"].isin(methods)].copy()
    pivot = filtered.pivot_table(
        index="instance_idx",
        columns="algorithm",
        values=metric,
        aggfunc="mean",
    )
    missing_methods = [method for method in methods if method not in pivot.columns]
    if missing_methods:
        raise ValueError(f"No rows found for methods: {missing_methods}")

    pivot = pivot[methods].dropna()
    if len(pivot) == 0:
        raise ValueError(
            f"No complete paired rows found for metric '{metric}' and methods {methods}."
        )
    return pivot


def describe_metric(pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in pivot.columns:
        values = pivot[method].astype(float)
        rows.append(
            {
                "method": method,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def save_metric_boxplot(pivot: pd.DataFrame, metric: str, output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skip boxplot for '{metric}': {exc}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(pivot.columns)
    data = [pivot[method].astype(float).values for method in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    boxplot = ax.boxplot(data, patch_artist=True, labels=methods)
    colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc", "#fdd0a2"]
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    means = [float(pivot[method].astype(float).mean()) for method in methods]
    ax.scatter(range(1, len(methods) + 1), means, color="#d62728", marker="D", s=36, label="Mean")
    ax.set_title(f"{metric} distribution by algorithm")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def run_friedman_and_posthoc(
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric = normalize_metric(metric)
    pivot = build_pivot(df, metric, methods)

    arrays = [pivot[method].astype(float).values for method in pivot.columns]
    friedman_stat, friedman_p = friedmanchisquare(*arrays)
    kendalls_w = friedman_stat / (len(pivot) * (len(pivot.columns) - 1))

    pair_rows = []
    raw_p_values = []
    for left, right in combinations(pivot.columns, 2):
        x = pivot[left].astype(float)
        y = pivot[right].astype(float)
        diff = x - y
        if bool((diff == 0.0).all()):
            stat = 0.0
            p_value = 1.0
        else:
            stat, p_value = wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="two-sided",
                mode="auto",
            )

        pair_rows.append(
            {
                "comparison": f"{left} vs {right}",
                "mean_diff_left_minus_right": diff.mean(),
                "median_diff_left_minus_right": diff.median(),
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
                "rank_biserial_left_minus_right": rank_biserial_from_wilcoxon(x, y),
            }
        )
        raw_p_values.append(float(p_value))

    for row, adjusted_p in zip(pair_rows, holm_correction(raw_p_values)):
        row["holm_p_value"] = adjusted_p

    overall = {
        "metric": metric,
        "methods": ",".join(pivot.columns),
        "n_instances": int(len(pivot)),
        "friedman_stat": float(friedman_stat),
        "friedman_p_value": float(friedman_p),
        "kendalls_w": float(kendalls_w),
    }
    return overall, describe_metric(pivot), pd.DataFrame(pair_rows), pivot


def print_block(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Friedman and pairwise Wilcoxon tests on main.py PC result CSV files."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Algorithms to compare. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to test. Common choices: score, num_groups, wall_elapsed_sec, solver_elapsed_sec.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_list(args.methods, DEFAULT_METHODS)
    metrics = [normalize_metric(metric) for metric in parse_list(args.metrics, DEFAULT_METRICS)]

    df = load_results(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input CSV: {args.csv}")
    print(f"Methods: {methods}")
    print(f"Metrics: {metrics}")
    print(f"Rows: {len(df)}")

    summary_rows = []
    for metric in metrics:
        overall, descriptives, pairwise, pivot = run_friedman_and_posthoc(df, metric, methods)

        print(f"\n\n##### METRIC: {metric} #####")
        print(
            "Friedman: "
            f"stat={overall['friedman_stat']:.6f}, "
            f"p={overall['friedman_p_value']:.6g}, "
            f"Kendall_W={overall['kendalls_w']:.6f}, "
            f"n={overall['n_instances']}"
        )
        print_block(f"{metric} descriptives", descriptives)
        print_block(f"{metric} pairwise Wilcoxon + Holm", pairwise)

        summary_rows.append(overall)
        descriptives.to_csv(args.output_dir / f"{metric}_descriptives.csv", index=False)
        pairwise.to_csv(args.output_dir / f"{metric}_pairwise.csv", index=False)
        pivot.to_csv(args.output_dir / f"{metric}_paired_values.csv")

        if not args.no_plots:
            plot_path = args.output_dir / f"{metric}_boxplot.png"
            if save_metric_boxplot(pivot, metric, plot_path):
                print(f"Saved boxplot: {plot_path}")

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "friedman_summary.csv", index=False)
    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
