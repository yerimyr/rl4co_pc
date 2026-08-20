from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_PATTERN = "logs/train/runs/**/test_results/nco_test_results.csv"
DEFAULT_OUTPUT_DIR = Path("outputs/nco_test_boxplot")
METHOD_LABELS = {
    "reinforce": "CONSTRUCTIVE_REINFORCE",
    "ppo": "CONSTRUCTIVE_PPO",
    "pomo": "CONSTRUCTIVE_POMO",
    "original": "ORIGINAL",
    "no_compatibility_matrix": "NO_COMPATIBILITY_MATRIX",
    "no_action0": "NO_ACTION_0",
    "no_encoder_masking": "NO_ENCODER_MASKING",
    "all_removed": "ALL_REMOVED",
    "reinforce_edge": "ORIGINAL",
    "reinforce_edge_no_compat": "NO_COMPATIBILITY_MATRIX",
    "reinforce_edge_no_sep_encoder": "NO_ACTION_0",
    "reinforce_edge_no_message_mask": "NO_ENCODER_MASKING",
    "reinforce_edge_all_removed": "ALL_REMOVED",
    "ppo_edge_all_removed": "PPO_EDGE_ALL_REMOVED",
    "pomo_edge_all_removed": "POMO_EDGE_ALL_REMOVED",
    "reinforce_edge_no_compat_no_mask": "CURRENT_ENCODER",
    "ppo_edge_no_compat_no_mask": "PPO_EDGE_NO_COMPAT_NO_MASK",
    "pomo_edge_no_compat_no_mask": "POMO_EDGE_NO_COMPAT_NO_MASK",
    "reinforce_matnet": "ONLY_MATNET",
    "reinforce_split_hybrid": "CURRENT_ENCODER+MATNET",
}


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
        default=[
            "reinforce_edge",
            "reinforce_edge_no_compat",
            "reinforce_edge_no_message_mask",
            "reinforce_edge_all_removed",
        ],
        help="Methods to include in the plot.",
    )
    parser.add_argument(
        "--aliases",
        nargs="*",
        default=None,
        help="Optional algorithm names to assign to each --csv file in order.",
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
    labels = [METHOD_LABELS.get(method, method.upper()) for method in methods]
    fig_width = max(8.5, 1.8 * len(methods))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    try:
        boxplot = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=labels,
        )
    except TypeError:
        boxplot = ax.boxplot(
            data,
            patch_artist=True,
            labels=labels,
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
    ax.tick_params(axis="x", labelsize=7)
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
    if args.aliases is not None and len(args.aliases) != len(csv_paths):
        raise ValueError(
            f"--aliases must have the same length as --csv files: "
            f"{len(args.aliases)} aliases for {len(csv_paths)} files"
        )

    for idx, path in enumerate(csv_paths):
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        file_rows = read_rows(path)
        if args.aliases is not None:
            alias = args.aliases[idx].lower()
            for row in file_rows:
                row["algorithm"] = alias
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
