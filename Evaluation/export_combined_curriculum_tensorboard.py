"""Export three curriculum stages as one continuous TensorBoard run per gamma."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from Evaluation.plot_combined_curriculum_learning_curves import TAGS, collect


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
    parser.add_argument(
        "--stage-runs-only",
        action="store_true",
        help="Append separate Stage 1/2/3 runs without rewriting combined runs.",
    )
    return parser.parse_args()


def gamma_name(gamma: float) -> str:
    return f"gamma_{gamma:g}".replace(".", "p")


def add_layout(writer: SummaryWriter) -> None:
    writer.add_custom_scalars(
        {
            "Curriculum reward": {
                "Train and validation": [
                    "Multiline",
                    [
                        "train_epoch/reward",
                        "val_epoch/reward",
                        "val_epoch/sampling_reward",
                    ],
                ]
            },
            "Policy diagnostics": {
                "Entropy and negative log-likelihood": [
                    "Multiline",
                    ["train_epoch/entropy", "train_epoch/log_likelihood"],
                ]
            },
        }
    )


def write_metrics(
    writer: SummaryWriter,
    rows,
    metric_to_tag: dict[str, str],
    step_column: str,
    wall_time_column: str,
) -> None:
    for metric, metric_rows in rows.groupby("metric", sort=False):
        tag = metric_to_tag[metric]
        ordered = metric_rows.sort_values(step_column)
        for row in ordered.itertuples(index=False):
            writer.add_scalar(
                tag,
                float(row.value),
                int(getattr(row, step_column)) - 1,
                walltime=float(getattr(row, wall_time_column)),
            )


def add_continuous_wall_time(rows):
    """Remove downtime between stages while retaining actual training durations."""
    rows = rows.copy()
    base_wall_time = float(rows[rows["stage"] == 1]["wall_time"].min())
    elapsed_offset = 0.0
    for stage in sorted(rows["stage"].unique()):
        mask = rows["stage"] == stage
        stage_start = float(rows.loc[mask, "wall_time"].min())
        stage_end = float(rows.loc[mask, "wall_time"].max())
        rows.loc[mask, "continuous_wall_time"] = (
            base_wall_time + elapsed_offset + rows.loc[mask, "wall_time"] - stage_start
        )
        elapsed_offset += stage_end - stage_start
    return rows


def main() -> None:
    args = parse_args()
    gamma0p5_retry = (
        args.gamma0p5_retry_run.resolve()
        if args.gamma0p5_retry_run is not None
        else None
    )
    raw, sources = collect(
        args.original_run.resolve(),
        args.retry_run.resolve(),
        gamma0p5_retry,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metric_to_tag = {metric: tag for metric, tag in TAGS.items()}
    for gamma, gamma_rows in raw.groupby("gamma", sort=True):
        gamma_rows = add_continuous_wall_time(gamma_rows)
        source_rows = sources[sources["gamma"] == gamma]
        source_text = "\n".join(
            f"Stage {int(row.stage)}: {row.source} — {row.stage_distribution}"
            for row in source_rows.itertuples(index=False)
        )

        if not args.stage_runs_only:
            run_dir = output / gamma_name(float(gamma)) / "combined"
            writer = SummaryWriter(log_dir=str(run_dir))
            add_layout(writer)
            write_metrics(
                writer,
                gamma_rows,
                metric_to_tag,
                "cumulative_epoch",
                "continuous_wall_time",
            )
            # A scalar stage indicator makes boundaries 100 and 200 inspectable.
            for step in range(300):
                time_row = gamma_rows[
                    (gamma_rows["metric"] == "train_reward")
                    & (gamma_rows["cumulative_epoch"] == step + 1)
                ].iloc[0]
                writer.add_scalar(
                    "curriculum/stage",
                    step // 100 + 1,
                    step,
                    walltime=float(time_row["continuous_wall_time"]),
                )
            writer.add_text("curriculum/source_manifest", source_text, 0)
            writer.close()

        for stage, stage_rows in gamma_rows.groupby("stage", sort=True):
            stage_dir = output / gamma_name(float(gamma)) / f"stage_{int(stage)}"
            writer = SummaryWriter(log_dir=str(stage_dir))
            add_layout(writer)
            write_metrics(
                writer,
                stage_rows,
                metric_to_tag,
                "epoch_in_stage",
                "wall_time",
            )
            writer.add_text(
                "curriculum/source_manifest",
                source_text.splitlines()[int(stage) - 1],
                0,
            )
            writer.close()

    raw.to_csv(output / "learning_curve_raw.csv", index=False)
    sources.to_csv(output / "learning_curve_sources.csv", index=False)
    print(f"Saved merged TensorBoard runs: {output}")


if __name__ == "__main__":
    main()
