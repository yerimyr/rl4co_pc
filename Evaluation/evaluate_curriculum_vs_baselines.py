"""Compare final curriculum NCO models with existing PC evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.common import ROOT, save_dataframe
from Evaluation.gamma_sweep import evaluate_nco_custom
from rl4co.data.utils import load_npz_to_tensordict


METHODS = ("ortools", "cpccd", "nco-custom", "nco-curriculum")
METHOD_LABELS = {
    "ortools": "OR-Tools (serial)",
    "cpccd": "CPCCD",
    "nco-custom": "NCO (independent)",
    "nco-curriculum": "NCO (curriculum)",
}
COLORS = {
    "ortools": "#9ecae1",
    "cpccd": "#f28e2b",
    "nco-custom": "#59a14f",
    "nco-curriculum": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/evaluation/gamma_sweep"))
    parser.add_argument("--num-parts", default="20,30,40")
    parser.add_argument("--gammas", default="0,0.25,0.5")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/gamma_sweep/curriculum_comparison_seed1234"),
    )
    return parser.parse_args()


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def stage3_best(run_root: Path, gamma: float) -> Path:
    checkpoint_dir = (
        run_root / f"gamma_{gamma_label(gamma)}" / "stage3_n10-40" / "checkpoints"
    )
    checkpoints = sorted(
        (path for path in checkpoint_dir.glob("*.ckpt") if path.name != "last.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"Completed Stage 3 best checkpoint not found: {checkpoint_dir}")
    return checkpoints[0].resolve()


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["num_parts", "gamma", "method"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            score_sem=("score", "sem"),
            num_groups_mean=("num_groups", "mean"),
            feasible_rate=("feasible", "mean"),
            wall_elapsed_sec_mean=("wall_elapsed_sec", "mean"),
            n=("score", "size"),
        )
        .sort_values(["num_parts", "gamma", "method"])
    )


def plot_one(ax, summary: pd.DataFrame, num_parts: int, gammas: list[float]) -> None:
    subset = summary[summary["num_parts"] == num_parts]
    x = np.arange(len(gammas), dtype=float)
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))
    for offset, method in zip(offsets, METHODS):
        values = (
            subset[subset["method"] == method]
            .set_index("gamma")
            .reindex(gammas)
        )
        ax.bar(
            x + offset,
            values["score_mean"],
            width,
            yerr=values["score_std"],
            capsize=3,
            color=COLORS[method],
            alpha=0.9,
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(x, [f"{gamma:g}" for gamma in gammas])
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Score")
    ax.set_title(f"N={num_parts}, test seed=1234 (error bars: ±1 SD)")
    ax.grid(True, axis="y", alpha=0.25)


def save_plots(summary: pd.DataFrame, output_dir: Path, sizes: list[int], gammas: list[float]) -> None:
    lower = float((summary["score_mean"] - summary["score_std"]).min())
    upper = float((summary["score_mean"] + summary["score_std"]).max())
    padding = max((upper - lower) * 0.06, 1.0)
    y_limits = (min(0.0, lower - padding), upper + padding)

    for num_parts in sizes:
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_one(ax, summary, num_parts, gammas)
        ax.set_ylim(*y_limits)
        ax.legend(title="Method")
        fig.tight_layout()
        fig.savefig(output_dir / f"n{num_parts}_curriculum_comparison.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(sizes), figsize=(7 * len(sizes), 6), sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    for ax, num_parts in zip(axes, sizes):
        plot_one(ax, summary, num_parts, gammas)
        ax.set_ylim(*y_limits)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), title="Method")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "n20_n30_n40_curriculum_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    sizes = [int(value) for value in args.num_parts.split(",")]
    gammas = [float(value) for value in args.gammas.split(",")]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {gamma: stage3_best(run_root, gamma) for gamma in gammas}
    all_rows = []
    for num_parts in sizes:
        data_path = ROOT / "data" / "pc" / f"pc{num_parts}_newdist_test_seed{args.seed}.npz"
        dataset = load_npz_to_tensordict(data_path)
        limit = min(args.limit, len(dataset))

        existing_path = (
            args.base_dir
            / f"n{num_parts}_seed{args.seed}_test{args.limit}_paper(ortools_5m)"
            / "raw_results.csv"
        )
        existing = pd.read_csv(existing_path)
        existing = existing[
            existing["method"].isin(METHODS[:3])
            & existing["gamma"].astype(float).isin(gammas)
            & (existing["instance_idx"] < limit)
        ].copy()
        existing["num_parts"] = num_parts
        all_rows.extend(existing.to_dict("records"))

        for gamma in gammas:
            print(f"\nEvaluate curriculum NCO: N={num_parts}, gamma={gamma:g}")
            rows = evaluate_nco_custom(
                dataset,
                limit,
                gamma,
                checkpoints[gamma],
                args.device,
                args.batch_size,
            )
            for row in rows:
                row["method"] = "nco-curriculum"
                row["algorithm"] = "nco-curriculum"
                row["num_parts"] = num_parts
            all_rows.extend(rows)

    raw = pd.DataFrame(all_rows)
    summary = summarize(raw)
    save_dataframe(raw, output_dir / "raw_results.csv")
    save_dataframe(summary, output_dir / "summary.csv")
    for num_parts, group in raw.groupby("num_parts"):
        save_dataframe(group, output_dir / f"n{int(num_parts)}_raw_results.csv")
    save_plots(summary, output_dir, sizes, gammas)
    print(f"\nSaved curriculum comparison: {output_dir}")


if __name__ == "__main__":
    main()
