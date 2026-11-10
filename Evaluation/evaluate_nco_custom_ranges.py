"""Evaluate existing fixed-size NCO models on curriculum test-size ranges.

The fixed-size checkpoint for each range/gamma is read from the corresponding
paper gamma-sweep raw results.  This guarantees that the same independent NCO
model used in the paper comparison is reused here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.common import ROOT, save_dataframe
from Evaluation.evaluate_curriculum_forgetting_ranges import parse_ranges, range_label
from Evaluation.evaluate_curriculum_range_baselines import summarize, test_dataset_path
from Evaluation.gamma_sweep import evaluate_nco_custom
from rl4co.data.utils import load_npz_to_tensordict


METHODS = ("ortools", "cpccd", "nco-custom", "nco-curriculum")
METHOD_LABELS = {
    "ortools": "CP-SAT",
    "cpccd": "CPCCD",
    "nco-custom": "NCO (fixed-size)",
    "nco-curriculum": "NCO (curriculum)",
}
COLORS = {
    "ortools": "#4e9bc6",
    "cpccd": "#f28e2b",
    "nco-custom": "#59a14f",
    "nco-curriculum": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-ranges", default="10-20,21-30,31-40")
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--test-seed", type=int, default=1234)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--base-results",
        type=Path,
        default=Path("outputs/evaluation/curriculum_range_test_seed1234"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/curriculum_range_ablation_seed1234"),
    )
    return parser.parse_args()


def fixed_result_path(num_parts: int) -> Path:
    return (
        ROOT
        / "outputs"
        / "evaluation"
        / "gamma_sweep"
        / f"n{num_parts}_seed1234_test100_paper(ortools_5m)"
        / "raw_results.csv"
    )


def fixed_checkpoint(num_parts: int, gamma: float) -> Path:
    path = fixed_result_path(num_parts)
    raw = pd.read_csv(path)
    rows = raw[
        (raw["method"] == "nco-custom")
        & np.isclose(raw["gamma"].astype(float), gamma)
    ]
    checkpoints = rows["ckpt"].dropna().astype(str).unique().tolist()
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected one recorded checkpoint for N={num_parts}, gamma={gamma:g}; "
            f"found {checkpoints} in {path}"
        )
    checkpoint = Path(checkpoints[0]).resolve()
    if not checkpoint.exists():
        # Some paper runs were moved from logs/train/runs after evaluation.
        # Recover the same checkpoint by preserving the recorded run name.
        run_name = checkpoint.parents[1].name
        moved = (
            ROOT
            / "logs"
            / "train"
            / f"runs(paper,N{num_parts})"
            / run_name
            / "checkpoints"
            / checkpoint.name
        )
        if moved.exists():
            checkpoint = moved.resolve()
        else:
            raise FileNotFoundError(
                f"Recorded checkpoint does not exist: {checkpoint}; "
                f"moved candidate also missing: {moved}"
            )
    return checkpoint


def save_plots(
    summary: pd.DataFrame,
    output: Path,
    labels: list[str],
    gammas: list[float],
) -> None:
    for label in labels:
        subset = summary[summary["test_range"] == label]
        x = np.arange(len(gammas), dtype=float)
        width = 0.19
        fig, ax = plt.subplots(figsize=(12, 7))
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))
        for offset, method in zip(offsets, METHODS):
            values = subset[subset["method"] == method].set_index("gamma").reindex(gammas)
            ax.bar(
                x + offset,
                values["score_mean"],
                width,
                yerr=values["score_std"],
                capsize=3,
                color=COLORS[method],
                alpha=0.9,
                label=METHOD_LABELS[method],
                error_kw={"elinewidth": 2.0, "capthick": 2.0},
            )
        ax.set_xticks(x, [f"{gamma:g}" for gamma in gammas])
        ax.set_xlabel("Gamma", fontsize=22)
        ax.set_ylabel("Score", fontsize=22)
        ax.tick_params(axis="both", labelsize=19)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=16)
        fig.tight_layout()
        fig.savefig(output / f"{label.lower()}_curriculum_ablation_mean_std.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    base = args.base_results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_raw = pd.read_csv(base / "raw_results.csv")
    raw = source_raw[source_raw["method"] != "nco-custom"].copy()
    gammas = [float(value) for value in args.gammas.split(",")]
    ranges = parse_ranges(args.test_ranges)
    labels = [range_label(low, high) for low, high in ranges]

    new_rows: list[dict] = []
    for low, high in ranges:
        label = range_label(low, high)
        dataset_path = test_dataset_path(low, high, args.test_seed, args.test_size)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Range test dataset not found: {dataset_path}")
        dataset = load_npz_to_tensordict(dataset_path)
        limit = min(args.test_size, len(dataset))
        sampled_sizes = dataset["num_parts"].cpu().numpy()

        # The independent baseline for a band is the model trained at its upper N.
        for gamma in gammas:
            checkpoint = fixed_checkpoint(high, gamma)
            print(
                f"Evaluate fixed-size NCO: range={label}, gamma={gamma:g}, "
                f"checkpoint={checkpoint}",
                flush=True,
            )
            rows = evaluate_nco_custom(
                dataset,
                limit,
                gamma,
                checkpoint,
                args.device,
                args.batch_size,
            )
            for row in rows:
                row["method"] = "nco-custom"
                row["algorithm"] = "nco-custom"
                row["test_range"] = label
                row["sampled_num_parts"] = int(sampled_sizes[row["instance_idx"]])
                row["test_seed"] = args.test_seed
                row["fixed_size_checkpoint"] = str(checkpoint)
            new_rows.extend(rows)

    raw = pd.concat([raw, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    raw = raw.drop_duplicates(
        subset=["test_range", "gamma", "method", "instance_idx"], keep="last"
    ).sort_values(["test_range", "gamma", "method", "instance_idx"])
    save_dataframe(raw, output / "raw_results.csv")

    summary = summarize(raw)
    save_dataframe(summary, output / "summary.csv")
    for label, group in raw.groupby("test_range"):
        save_dataframe(group, output / f"{label.lower()}_raw_results.csv")
    save_plots(summary, output, labels, gammas)
    print(f"Saved fixed-size versus curriculum range comparison: {output}")


if __name__ == "__main__":
    main()
