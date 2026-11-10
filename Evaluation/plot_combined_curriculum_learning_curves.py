"""Build continuous three-stage learning curves from two curriculum runs.

The original run supplies gamma 0, 0.25, and 0.5.  For gamma 0.75 and 1.0,
Stage 1 comes from the original run and Stages 2-3 come from the low-LR retry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
STAGES = (
    (1, "stage1_n10-20", "N=10-20"),
    (2, "stage2_n10-30", "N=10-20 (20%) + N=21-30 (80%)"),
    (3, "stage3_n10-40", "N=10-20 (10%) + N=21-30 (10%) + N=31-40 (80%)"),
)
TAGS = {
    "train_reward": "train_epoch/reward",
    "validation_reward": "val_epoch/reward",
    "validation_sampling_reward": "val_epoch/sampling_reward",
    "loss": "train_epoch/loss",
    "entropy": "train_epoch/entropy",
    "log_likelihood": "train_epoch/log_likelihood",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--retry-run", type=Path, required=True)
    parser.add_argument(
        "--gamma0p5-retry-run",
        type=Path,
        default=None,
        help=(
            "Optional gamma=0.5 retry root. When supplied, Stage 2 and Stage 3 "
            "for gamma=0.5 are read from this run instead of the original run."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=5)
    return parser.parse_args()


def gamma_dir(gamma: float) -> str:
    return f"gamma_{gamma:g}".replace(".", "p")


def event_dir(stage_dir: Path) -> Path:
    versions = sorted(stage_dir.glob("tensorboard/*/version_*"))
    if not versions:
        raise FileNotFoundError(f"TensorBoard version directory not found: {stage_dir}")
    return versions[-1]


def scalar_rows(version_dir: Path, tag: str) -> list[dict]:
    accumulator = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []

    # Multiple event files may contain the same step after a restart.  Keep the
    # most recently written event for each step.
    latest = {}
    for event in accumulator.Scalars(tag):
        if event.step not in latest or event.wall_time >= latest[event.step].wall_time:
            latest[event.step] = event
    return [
        {
            "epoch_in_stage": step + 1,
            "value": latest[step].value,
            "wall_time": latest[step].wall_time,
        }
        for step in sorted(latest)
    ]


def collect(
    original: Path,
    retry: Path,
    gamma0p5_retry: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    sources: list[dict] = []
    for gamma in GAMMAS:
        for stage, stage_name, stage_distribution in STAGES:
            use_gamma0p5_retry = (
                gamma0p5_retry is not None
                and gamma == 0.5
                and stage in (2, 3)
            )
            use_high_gamma_retry = gamma in (0.75, 1.0) and stage in (2, 3)
            if use_gamma0p5_retry:
                root = gamma0p5_retry
                source_label = "retry_gamma0p5_lr1e-5"
            elif use_high_gamma_retry:
                root = retry
                source_label = "retry_lr1e-5"
            else:
                root = original
                source_label = "original"
            stage_dir = root / gamma_dir(gamma) / stage_name
            version_dir = event_dir(stage_dir)
            sources.append(
                {
                    "gamma": gamma,
                    "stage": stage,
                    "stage_distribution": stage_distribution,
                    "source": source_label,
                    "tensorboard_dir": str(version_dir.resolve()),
                }
            )
            for metric, tag in TAGS.items():
                for item in scalar_rows(version_dir, tag):
                    rows.append(
                        {
                            "gamma": gamma,
                            "stage": stage,
                            "stage_distribution": stage_distribution,
                            "source": source_label,
                            "metric": metric,
                            "epoch_in_stage": item["epoch_in_stage"],
                            "cumulative_epoch": (stage - 1) * 100 + item["epoch_in_stage"],
                            "value": item["value"],
                            "wall_time": item["wall_time"],
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(sources)


def add_stage_guides(ax: plt.Axes) -> None:
    for x in (100.5, 200.5):
        ax.axvline(x, color="0.35", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.axvspan(0.5, 100.5, color="#4e79a7", alpha=0.045)
    ax.axvspan(100.5, 200.5, color="#f28e2b", alpha=0.045)
    ax.axvspan(200.5, 300.5, color="#59a14f", alpha=0.045)
    y0, y1 = ax.get_ylim()
    y = y1 - 0.035 * (y1 - y0)
    for x, label in ((50.5, "Stage 1"), (150.5, "Stage 2"), (250.5, "Stage 3")):
        ax.text(x, y, label, ha="center", va="top", fontsize=9, color="0.3")
    ax.set_xlim(1, 300)
    ax.set_xlabel("Cumulative epoch")
    ax.grid(True, alpha=0.22)


def draw_gamma(ax: plt.Axes, raw: pd.DataFrame, gamma: float, window: int) -> None:
    subset = raw[raw["gamma"] == gamma]
    styles = {
        "train_reward": ("#4e79a7", "Train reward"),
        "validation_reward": ("#e15759", "Validation reward (greedy)"),
    }
    for metric, (color, label) in styles.items():
        values = subset[subset["metric"] == metric].sort_values("cumulative_epoch").copy()
        smooth = values["value"].rolling(window, center=True, min_periods=1).mean()
        ax.plot(
            values["cumulative_epoch"], values["value"], color=color,
            alpha=0.16, linewidth=0.8,
        )
        ax.plot(
            values["cumulative_epoch"], smooth, color=color,
            linewidth=2.0, label=f"{label} ({window}-epoch mean)",
        )
    ax.set_title(f"gamma={gamma:g}")
    ax.set_ylabel("Reward")
    add_stage_guides(ax)


def save_plots(raw: pd.DataFrame, output: Path, window: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for gamma in GAMMAS:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        draw_gamma(ax, raw, gamma, window)
        ax.legend(loc="best")
        fig.suptitle("Three-stage curriculum learning curve", fontsize=15)
        fig.tight_layout()
        gamma_name = f"{gamma:g}".replace(".", "p")
        fig.savefig(output / f"curriculum_learning_curve_gamma_{gamma_name}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    flat = axes.ravel()
    for ax, gamma in zip(flat, GAMMAS):
        draw_gamma(ax, raw, gamma, window)
    flat[-1].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.93, 0.08))
    fig.suptitle("Combined curriculum learning curves", fontsize=18)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output / "all_gammas_curriculum_learning_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    original = args.original_run.resolve()
    retry = args.retry_run.resolve()
    gamma0p5_retry = (
        args.gamma0p5_retry_run.resolve()
        if args.gamma0p5_retry_run is not None
        else None
    )
    output = args.output_dir.resolve()
    raw, sources = collect(original, retry, gamma0p5_retry)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "learning_curve_raw.csv", index=False)
    sources.to_csv(output / "learning_curve_sources.csv", index=False)
    save_plots(raw, output, args.smooth_window)
    print(f"Saved combined learning curves: {output}")
    print(sources.to_string(index=False))


if __name__ == "__main__":
    main()
