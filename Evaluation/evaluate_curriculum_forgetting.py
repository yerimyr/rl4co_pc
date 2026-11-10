from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from Evaluation.common import ROOT, save_dataframe
from Evaluation.gamma_sweep import evaluate_nco_custom
from rl4co.data.utils import load_npz_to_tensordict


STAGES = (
    ("stage1", "stage1_n10-20", 20),
    ("stage2", "stage2_n10-30", 30),
    ("stage3", "stage3_n10-40", 40),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-evaluate curriculum checkpoints for catastrophic forgetting."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gammas", default="0,0.25")
    parser.add_argument("--validation-sizes", default="20,30,40")
    parser.add_argument("--validation-seed", type=int, default=4321)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def best_checkpoint(stage_dir: Path) -> Path:
    checkpoint_dir = stage_dir / "checkpoints"
    checkpoints = sorted(
        (path for path in checkpoint_dir.glob("*.ckpt") if path.name != "last.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_dir}")
    return checkpoints[0].resolve()


def validation_path(size: int, seed: int) -> Path:
    path = ROOT / "data" / "pc" / f"pc{size}_newdist_val_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {path}")
    return path


def make_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["gamma", "stage", "validation_n"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            score_sem=("score", "sem"),
            num_groups_mean=("num_groups", "mean"),
            feasible_rate=("feasible", "mean"),
            n=("score", "size"),
        )
        .sort_values(["gamma", "validation_n", "stage"])
    )


def make_forgetting(raw: pd.DataFrame) -> pd.DataFrame:
    reference_stage = {20: "stage1", 30: "stage2", 40: "stage3"}
    rows = []
    for (gamma, validation_n), group in raw.groupby(["gamma", "validation_n"]):
        reference = reference_stage.get(int(validation_n))
        if reference is None:
            continue
        pivot = group.pivot(index="instance_idx", columns="stage", values="score")
        if reference not in pivot:
            continue
        reference_scores = pivot[reference]
        for stage in ("stage1", "stage2", "stage3"):
            if stage not in pivot:
                continue
            paired = pd.concat([reference_scores, pivot[stage]], axis=1).dropna()
            paired.columns = ["reference", "current"]
            delta = paired["current"] - paired["reference"]
            rows.append(
                {
                    "gamma": gamma,
                    "validation_n": validation_n,
                    "reference_stage": reference,
                    "evaluated_stage": stage,
                    "score_change_vs_reference": delta.mean(),
                    "score_change_std": delta.std(),
                    "forgetting": -delta.mean(),
                    "n": len(delta),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["gamma", "validation_n", "evaluated_stage"]
    )


def save_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    stages = ["stage1", "stage2", "stage3"]
    for gamma, group in summary.groupby("gamma"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for validation_n, values in group.groupby("validation_n"):
            values = values.set_index("stage").reindex(stages)
            ax.errorbar(
                range(1, 4),
                values["score_mean"],
                yerr=values["score_sem"],
                marker="o",
                capsize=4,
                label=f"Validation N={int(validation_n)}",
            )
        ax.set_xticks(range(1, 4), ["Stage 1", "Stage 2", "Stage 3"])
        ax.set_xlabel("Curriculum checkpoint")
        ax.set_ylabel("Mean validation score (error bars: SEM)")
        ax.set_title(f"Curriculum retention and transfer | gamma={gamma:g}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"curriculum_validation_gamma_{gamma_label(gamma)}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    gammas = [float(value) for value in args.gammas.split(",")]
    validation_sizes = [int(value) for value in args.validation_sizes.split(",")]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_root / "forgetting_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        size: load_npz_to_tensordict(validation_path(size, args.validation_seed))
        for size in validation_sizes
    }
    all_rows = []
    for gamma in gammas:
        gamma_dir = run_root / f"gamma_{gamma_label(gamma)}"
        for stage, stage_folder, _ in STAGES:
            checkpoint = best_checkpoint(gamma_dir / stage_folder)
            for validation_n, dataset in datasets.items():
                limit = min(args.limit, len(dataset))
                print(
                    f"\nEvaluate gamma={gamma:g}, {stage}, validation N={validation_n}, "
                    f"instances={limit}"
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
                    row["stage"] = stage
                    row["validation_n"] = validation_n
                all_rows.extend(rows)

    raw = pd.DataFrame(all_rows)
    summary = make_summary(raw)
    forgetting = make_forgetting(raw)
    save_dataframe(raw, output_dir / "raw_results.csv")
    save_dataframe(summary, output_dir / "summary.csv")
    save_dataframe(forgetting, output_dir / "forgetting.csv")
    save_plot(summary, output_dir)
    print(f"\nSaved curriculum forgetting evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
