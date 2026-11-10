"""Evaluate curriculum checkpoints on fixed variable-size validation bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from Evaluation.common import ROOT, save_dataframe
from Evaluation.gamma_sweep import evaluate_nco_custom
from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from rl4co.envs.pc.env import PartConsolidationEnv


STAGES = (
    ("stage1", "stage1_n10-20"),
    ("stage2", "stage2_n10-30"),
    ("stage3", "stage3_n10-40"),
)
REFERENCE_STAGE = {"N10-20": "stage1", "N21-30": "stage2", "N31-40": "stage3"}
TIE_ATOL = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--validation-ranges", default="10-20,21-30,31-40")
    parser.add_argument("--validation-seed", type=int, default=4321)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def parse_ranges(value: str) -> list[tuple[int, int]]:
    ranges = []
    for token in value.split(","):
        low, high = (int(part) for part in token.strip().split("-", 1))
        if low > high:
            raise ValueError(f"Invalid validation range: {token}")
        ranges.append((low, high))
    return ranges


def range_label(low: int, high: int) -> str:
    return f"N{low}-{high}"


def dataset_path(low: int, high: int, seed: int, size: int) -> Path:
    return ROOT / "data" / "pc" / f"pc{low}-{high}_newdist_val{size}_seed{seed}.npz"


def generate_range_dataset(low: int, high: int, seed: int, size: int, overwrite: bool) -> Path:
    path = dataset_path(low, high, seed, size)
    if path.exists() and not overwrite:
        return path
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = PartConsolidationEnv(
        generator_params={
            "num_parts": low,
            "max_num_parts": high,
            "material_types": 2,
            "topology_mode": "mixed",
            "p_maint_H_low": 0.01,
            "p_maint_H_high": 0.02,
            "p_standard_low": 0.01,
            "p_standard_high": 0.02,
            "p_relative_motion_low": 0.01,
            "p_relative_motion_high": 0.02,
        }
    )
    dataset = env.generator(size)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tensordict_to_npz(dataset, path)
    counts = pd.Series(dataset["num_parts"].cpu().numpy()).value_counts().sort_index()
    print(f"Saved fixed validation dataset: {path}")
    print(f"N counts: {counts.to_dict()}")
    return path


def best_checkpoint(run_root: Path, gamma: float, stage_folder: str) -> Path:
    checkpoint_dir = run_root / f"gamma_{gamma_label(gamma)}" / stage_folder / "checkpoints"
    checkpoints = sorted(
        (path for path in checkpoint_dir.glob("epoch_*.ckpt") if path.name != "last.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_dir}")
    return checkpoints[0].resolve()


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["gamma", "stage", "validation_range"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            score_sem=("score", "sem"),
            num_groups_mean=("num_groups", "mean"),
            feasible_rate=("feasible", "mean"),
            sampled_n_mean=("sampled_num_parts", "mean"),
            n=("score", "size"),
        )
        .sort_values(["gamma", "validation_range", "stage"])
    )


def summarize_range_normalized(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize scores for comparing checkpoints within each size band.

    Dividing by the sampled number of parts removes a simple size factor.  The
    resulting stage mean is then indexed to the best checkpoint for the same
    gamma and validation band.  This keeps comparisons paired to one band and
    avoids interpreting the raw score scales of different N ranges as equal.
    """
    normalized = raw.copy()
    normalized["score_per_part"] = (
        normalized["score"].astype(float)
        / normalized["sampled_num_parts"].astype(float)
    )
    summary = (
        normalized.groupby(["gamma", "stage", "validation_range"], as_index=False)
        .agg(
            score_per_part_mean=("score_per_part", "mean"),
            score_per_part_std=("score_per_part", "std"),
            score_per_part_sem=("score_per_part", "sem"),
            n=("score_per_part", "size"),
        )
        .sort_values(["gamma", "validation_range", "stage"])
    )
    best = summary.groupby(["gamma", "validation_range"])[
        "score_per_part_mean"
    ].transform("max")
    # Gap-to-best form also behaves correctly if a future experiment has a
    # negative best score (a direct score/best ratio would reverse ordering).
    scale = best.abs().clip(lower=np.finfo(float).eps)
    summary["relative_to_range_best_pct"] = 100.0 * (
        1.0 - (best - summary["score_per_part_mean"]) / scale
    )
    summary["is_range_best"] = np.isclose(
        summary["relative_to_range_best_pct"], 100.0, rtol=0.0, atol=1e-9
    )
    return summary


def paired_changes(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (gamma, validation_range), group in raw.groupby(["gamma", "validation_range"]):
        reference = REFERENCE_STAGE[validation_range]
        pivot = group.pivot(index="instance_idx", columns="stage", values="score")
        for stage in ("stage1", "stage2", "stage3"):
            delta = (pivot[stage] - pivot[reference]).dropna().to_numpy(float)
            tied = np.isclose(delta, 0.0, rtol=0.0, atol=TIE_ATOL)
            test_delta = delta.copy()
            test_delta[tied] = 0.0
            try:
                pvalue = float(stats.wilcoxon(test_delta, zero_method="wilcox").pvalue)
            except ValueError:
                pvalue = 1.0
            rows.append(
                {
                    "gamma": gamma,
                    "validation_range": validation_range,
                    "reference_stage": reference,
                    "evaluated_stage": stage,
                    "score_change_vs_reference": delta.mean(),
                    "forgetting": -delta.mean(),
                    "wins": int(((delta > 0) & ~tied).sum()),
                    "ties": int(tied.sum()),
                    "losses": int(((delta < 0) & ~tied).sum()),
                    "wilcoxon_p": pvalue,
                    "n": len(delta),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["gamma", "validation_range", "evaluated_stage"]
    )


def summarize_paired_transition_gains(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize checkpoint-to-checkpoint gains on exactly paired instances."""
    transitions = (
        ("stage1_to_stage2", "stage1", "stage2", "N21-30"),
        ("stage2_to_stage3", "stage2", "stage3", "N31-40"),
    )
    rows = []
    for (gamma, validation_range), group in raw.groupby(
        ["gamma", "validation_range"]
    ):
        pivot = group.pivot(index="instance_idx", columns="stage", values="score")
        for transition, previous_stage, current_stage, newly_emphasized_range in transitions:
            paired = pivot[[previous_stage, current_stage]].dropna()
            delta = (
                paired[current_stage].astype(float)
                - paired[previous_stage].astype(float)
            ).to_numpy()
            tied = np.isclose(delta, 0.0, rtol=0.0, atol=TIE_ATOL)
            rows.append(
                {
                    "gamma": float(gamma),
                    "transition": transition,
                    "previous_stage": previous_stage,
                    "current_stage": current_stage,
                    "newly_emphasized_range": newly_emphasized_range,
                    "validation_range": validation_range,
                    "paired_score_change_mean": float(np.mean(delta)),
                    "paired_score_change_std": float(np.std(delta, ddof=1)),
                    "paired_score_change_sem": float(stats.sem(delta)),
                    "wins": int(((delta > 0) & ~tied).sum()),
                    "ties": int(tied.sum()),
                    "losses": int(((delta < 0) & ~tied).sum()),
                    "n": int(len(delta)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["gamma", "transition", "validation_range"]
    )


def save_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    stages = ["stage1", "stage2", "stage3"]
    for gamma, group in summary.groupby("gamma"):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for validation_range, values in group.groupby("validation_range"):
            values = values.set_index("stage").reindex(stages)
            ax.errorbar(
                range(1, 4), values["score_mean"], yerr=values["score_sem"],
                marker="o", capsize=4, label=validation_range,
            )
        ax.set_xticks(range(1, 4), ["Stage 1", "Stage 2", "Stage 3"])
        ax.set_xlabel("Curriculum checkpoint")
        ax.set_ylabel("Mean validation score (error bars: SEM)")
        ax.set_title(f"Curriculum retention by training-size range | gamma={gamma:g}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="Fixed validation range")
        fig.tight_layout()
        fig.savefig(output_dir / f"curriculum_range_validation_gamma_{gamma_label(gamma)}.png", dpi=180)
        plt.close(fig)


def save_range_normalized_heatmap(normalized: pd.DataFrame, output_dir: Path) -> None:
    stages = ["stage1", "stage2", "stage3"]
    stage_labels = ["Stage 1", "Stage 2", "Stage 3"]
    ranges = ["N10-20", "N21-30", "N31-40"]
    gammas = sorted(normalized["gamma"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    image = None
    for ax, gamma in zip(axes, gammas):
        values = (
            normalized[normalized["gamma"] == gamma]
            .pivot(index="stage", columns="validation_range", values="relative_to_range_best_pct")
            .reindex(index=stages, columns=ranges)
        )
        image = ax.imshow(values.to_numpy(float), cmap="YlGn", vmin=0.0, vmax=100.0)
        ax.set_xticks(range(3), ranges)
        ax.set_yticks(range(3), stage_labels)
        ax.set_title(f"gamma={gamma:g}")
        for row in range(3):
            for column in range(3):
                value = float(values.iloc[row, column])
                ax.text(
                    column,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value < 60 else "black",
                    fontweight="bold" if np.isclose(value, 100.0) else "normal",
                )
            # Diagonal boxes mark the stage associated with each new size band.
            ax.add_patch(
                plt.Rectangle(
                    (row - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="tab:red",
                    linewidth=2.2,
                )
            )
        ax.set_xlabel("Validation size range")
        ax.set_ylabel("Checkpoint")
    for ax in axes[len(gammas):]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.82, label="Relative performance (% of best stage in range)")
    fig.suptitle(
        "Curriculum stage specialization after per-part and within-range normalization\n"
        "Red diagonal: stage associated with each newly introduced size band",
        fontsize=14,
    )
    fig.savefig(output_dir / "curriculum_stage_range_normalized_heatmap.png", dpi=180)
    plt.close(fig)


def save_per_part_normalized_line_plots(
    normalized: pd.DataFrame, output_dir: Path
) -> None:
    """Plot score/N while preserving the original range-validation layout."""
    stages = ["stage1", "stage2", "stage3"]
    stage_labels = ["Stage 1", "Stage 2", "Stage 3"]
    for gamma, group in normalized.groupby("gamma"):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for validation_range, values in group.groupby("validation_range"):
            values = values.set_index("stage").reindex(stages)
            ax.errorbar(
                range(1, 4),
                values["score_per_part_mean"].astype(float),
                yerr=values["score_per_part_sem"].astype(float),
                marker="o",
                capsize=4,
                label=validation_range,
            )
        ax.set_xticks(range(1, 4), stage_labels)
        ax.set_xlabel("Curriculum checkpoint")
        ax.set_ylabel("Mean score per actual part (score / N; error bars: SEM)")
        ax.set_title(f"Per-part normalized curriculum retention | gamma={gamma:g}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="Fixed validation range")
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"curriculum_range_validation_normalized_gamma_{gamma_label(gamma)}.png",
            dpi=180,
        )
        plt.close(fig)


def save_paired_transition_gain_plots(
    transition_summary: pd.DataFrame, output_dir: Path
) -> None:
    """Plot learning gain and forgetting without comparing range difficulty."""
    transitions = ["stage1_to_stage2", "stage2_to_stage3"]
    transition_labels = [
        "Stage 1 -> Stage 2\n(new emphasis: N21-30)",
        "Stage 2 -> Stage 3\n(new emphasis: N31-40)",
    ]
    ranges = ["N10-20", "N21-30", "N31-40"]
    colors = {"N10-20": "tab:blue", "N21-30": "tab:orange", "N31-40": "tab:green"}
    x = np.arange(len(transitions), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)

    for gamma, group in transition_summary.groupby("gamma"):
        fig, ax = plt.subplots(figsize=(9, 5.8))
        for offset, validation_range in zip(offsets, ranges):
            values = (
                group[group["validation_range"] == validation_range]
                .set_index("transition")
                .reindex(transitions)
            )
            ax.bar(
                x + offset,
                values["paired_score_change_mean"].astype(float),
                width,
                yerr=values["paired_score_change_sem"].astype(float),
                capsize=4,
                color=colors[validation_range],
                alpha=0.9,
                label=validation_range,
            )
        ax.axhline(0.0, color="black", linewidth=1.1)
        ax.set_xticks(x, transition_labels)
        ax.set_ylabel("Mean paired score change (error bars: SEM)")
        ax.set_title(f"Curriculum learning gain and forgetting | gamma={gamma:g}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="Fixed validation range")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"curriculum_paired_transition_gain_gamma_{gamma_label(gamma)}.png",
            dpi=180,
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    gammas = [float(value) for value in args.gammas.split(",")]
    ranges = parse_ranges(args.validation_ranges)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_root / "forgetting_evaluation_ranges"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for low, high in ranges:
        path = generate_range_dataset(
            low, high, args.validation_seed, args.validation_size, args.overwrite_data
        )
        datasets[range_label(low, high)] = load_npz_to_tensordict(path)

    rows = []
    for gamma in gammas:
        for stage, stage_folder in STAGES:
            checkpoint = best_checkpoint(run_root, gamma, stage_folder)
            for label, dataset in datasets.items():
                print(f"\nEvaluate gamma={gamma:g}, {stage}, validation={label}")
                evaluated = evaluate_nco_custom(
                    dataset, len(dataset), gamma, checkpoint, args.device, args.batch_size
                )
                sampled_sizes = dataset["num_parts"].cpu().numpy()
                for row in evaluated:
                    row["stage"] = stage
                    row["validation_range"] = label
                    row["sampled_num_parts"] = int(sampled_sizes[row["instance_idx"]])
                rows.extend(evaluated)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    changes = paired_changes(raw)
    transition_summary = summarize_paired_transition_gains(raw)
    save_dataframe(raw, output_dir / "raw_results.csv")
    save_dataframe(summary, output_dir / "summary.csv")
    save_dataframe(changes, output_dir / "forgetting.csv")
    save_dataframe(
        transition_summary, output_dir / "paired_transition_gain_summary.csv"
    )
    save_plots(summary, output_dir)
    save_paired_transition_gain_plots(transition_summary, output_dir)
    print(f"\nSaved range-based forgetting evaluation: {output_dir}")


if __name__ == "__main__":
    main()
