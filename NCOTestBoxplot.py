from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_PATTERN = "logs/train/runs/**/test_results/nco_test_results.csv"
DEFAULT_OUTPUT_DIR = Path("outputs/nco_test_boxplot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw REINFORCE/PPO/POMO boxplots from saved NCO test result CSV files."
    )
    parser.add_argument(
        "--csv",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit nco_test_results.csv files. If omitted, --pattern is used.",
    )
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-name", type=str, default="nco_test_score_boxplot.png")
    parser.add_argument("--combined-csv-name", type=str, default="nco_test_scores_combined.csv")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["reinforce", "ppo", "pomo"],
        help="Methods to include in the plot.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source_csv"] = str(path)
        row["algorithm"] = row["algorithm"].lower()
        row["instance_idx"] = int(row["instance_idx"])
        row["score"] = float(row["score"])
        row["reward"] = float(row.get("reward", row["score"]))
    return rows


def discover_csvs(args: argparse.Namespace) -> list[Path]:
    if args.csv:
        return [path.resolve() for path in args.csv]
    return sorted(Path(".").glob(args.pattern))


def write_combined_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "instance_idx",
        "score",
        "reward",
        "batch_idx",
        "local_idx",
        "source_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_boxplot(rows: list[dict], methods: list[str], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: matplotlib. Install it with: pip install matplotlib") from exc

    methods = [method.lower() for method in methods]
    data = [
        [row["score"] for row in rows if row["algorithm"] == method]
        for method in methods
    ]
    missing = [method for method, values in zip(methods, data) if not values]
    if missing:
        raise ValueError(f"No rows found for methods: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        boxplot = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=[method.upper() for method in methods],
        )
    except TypeError:
        boxplot = ax.boxplot(
            data,
            patch_artist=True,
            labels=[method.upper() for method in methods],
        )

    colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc", "#fdd0a2"]
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    means = [float(np.mean(values)) for values in data]
    ax.scatter(range(1, len(methods) + 1), means, color="#d62728", marker="D", s=40, label="Mean")
    ax.set_title("PC NCO Test Score Distribution")
    ax.set_xlabel("NCO model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(rows: list[dict], methods: list[str]) -> None:
    print("\nSummary")
    print("-" * 72)
    for method in methods:
        values = [row["score"] for row in rows if row["algorithm"] == method]
        if not values:
            continue
        print(
            f"{method:>9} | n={len(values):4d} | "
            f"mean={np.mean(values): .6f} | median={np.median(values): .6f} | "
            f"std={np.std(values, ddof=1) if len(values) > 1 else 0.0: .6f}"
        )


def main() -> None:
    args = parse_args()
    csv_paths = discover_csvs(args)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found. Pattern: {args.pattern}")

    rows: list[dict] = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        file_rows = read_rows(path)
        rows.extend(file_rows)
        print(f"Loaded {len(file_rows):4d} rows from {path}")

    methods = [method.lower() for method in args.methods]
    rows = [row for row in rows if row["algorithm"] in methods]
    if not rows:
        raise ValueError(f"No rows left after filtering methods: {methods}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.output_dir / args.combined_csv_name
    output_plot = args.output_dir / args.plot_name

    write_combined_csv(combined_csv, rows)
    save_boxplot(rows, methods, output_plot)
    print_summary(rows, methods)
    print(f"\nSaved combined CSV: {combined_csv}")
    print(f"Saved boxplot: {output_plot}")


if __name__ == "__main__":
    main()
