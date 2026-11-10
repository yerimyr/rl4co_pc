from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Stage 1/2/3 curriculum learning curves for each gamma."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing learning_curve_raw.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <log-dir>/learning_curve_figures.",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=7,
        help="EMA span used to make convergence trends readable.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def gamma_name(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def plot_gamma(rows: pd.DataFrame, gamma: float, output_dir: Path, ema_span: int, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    plotted_values: list[float] = []

    for stage, color in zip((1, 2, 3), colors):
        stage_rows = rows[rows["stage"] == stage].sort_values("epoch_in_stage")
        if stage_rows.empty:
            raise ValueError(f"Missing train reward rows: gamma={gamma:g}, stage={stage}")

        reward = stage_rows["value"].astype(float)
        smooth = reward.ewm(span=ema_span, adjust=False).mean()
        epoch = stage_rows["epoch_in_stage"].astype(int)
        plotted_values.extend(smooth.tolist())

        ax.plot(
            epoch,
            smooth,
            color=color,
            linewidth=2.6,
            label=f"Stage {stage}",
        )

    y_min = min(plotted_values)
    y_max = max(plotted_values)
    padding = max((y_max - y_min) * 0.07, 1.0)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.set_xlabel("Epoch", fontsize=17)
    ax.set_ylabel("Reward", fontsize=17)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", color="#b0b0b0", alpha=0.28, linewidth=0.8)
    ax.legend(frameon=False, fontsize=14, loc="best")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    stem = f"curriculum_learning_curve_gamma_{gamma_name(gamma)}"
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    input_csv = log_dir / "learning_curve_raw.csv"
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else log_dir / "learning_curve_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv)
    train = data[data["metric"] == "train_reward"].copy()
    if train.empty:
        raise ValueError("No train_reward rows found in learning_curve_raw.csv")

    for gamma in sorted(train["gamma"].unique()):
        plot_gamma(
            train[train["gamma"] == gamma],
            float(gamma),
            output_dir,
            args.ema_span,
            args.dpi,
        )

    print(f"Saved learning-curve figures: {output_dir}")


if __name__ == "__main__":
    main()
