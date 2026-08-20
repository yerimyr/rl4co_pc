from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Evaluation.common import (
    AlgorithmSpec,
    DEFAULT_GENERATOR_PARAMS,
    OUTPUT_ROOT,
    ROOT,
    add_bks_gap,
    algorithms_for_num_parts,
    evaluate_specs,
    load_or_generate_dataset,
    save_dataframe,
    save_json,
    save_metric_boxplots,
    save_nonparametric_tests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PC performance evaluation. This combines effectiveness, "
            "efficiency, and stability. Running this for multiple n values "
            "covers scalability."
        )
    )
    parser.add_argument("--num-parts", type=int, default=20)
    parser.add_argument(
        "--checkpoint-num-parts",
        type=int,
        default=None,
        help=(
            "Problem size used by the trained NCO checkpoints. Defaults to "
            "--num-parts. Use --checkpoint-num-parts 20 with --num-parts 50 "
            "for size generalization from n=20 to n=50."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)  # stability
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--nco-batch-size", type=int, default=1)
    parser.add_argument("--ga-pop-size", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=300)
    parser.add_argument("--sa-iterations", type=int, default=300)
    parser.add_argument("--cpccd-alpha", type=float, default=0.5)
    parser.add_argument(
        "--nco-current-ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint override for nco-custom/current encoder.",
    )
    parser.add_argument(
        "--nco-matnet-ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint override for nco-matnet.",
    )
    parser.add_argument(
        "--nco-new-ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint override for nco-new/part-matrix encoder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/evaluation/performance/n{num_parts}_seed{seed}.",
    )
    return parser.parse_args()


def default_data_path(num_parts: int, seed: int) -> Path:
    return ROOT / "data" / "pc" / f"pc{num_parts}_newdist_test_seed{seed}.npz"


def apply_checkpoint_overrides(
    algorithms: list[AlgorithmSpec],
    checkpoint_num_parts: int,
    args: argparse.Namespace,
) -> list[AlgorithmSpec]:
    overrides = {
        f"nco_current_n{checkpoint_num_parts}": args.nco_current_ckpt,
        f"nco_matnet_n{checkpoint_num_parts}": args.nco_matnet_ckpt,
        f"nco_new_n{checkpoint_num_parts}": args.nco_new_ckpt,
    }
    out: list[AlgorithmSpec] = []
    for spec in algorithms:
        ckpt = overrides.get(spec.name)
        out.append(
            AlgorithmSpec(spec.name, spec.kind, str(ckpt) if ckpt is not None else spec.ckpt)
        )
    return out


def make_config(args: argparse.Namespace) -> dict[str, Any]:
    generator_params = dict(DEFAULT_GENERATOR_PARAMS)
    generator_params["num_parts"] = args.num_parts
    checkpoint_num_parts = args.checkpoint_num_parts or args.num_parts
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif checkpoint_num_parts == args.num_parts:
        output_dir = OUTPUT_ROOT / "performance" / f"n{args.num_parts}_seed{args.seed}"
    else:
        output_dir = (
            OUTPUT_ROOT
            / "performance"
            / f"train_n{checkpoint_num_parts}_test_n{args.num_parts}_seed{args.seed}"
        )
    algorithms = apply_checkpoint_overrides(
        algorithms_for_num_parts(checkpoint_num_parts), checkpoint_num_parts, args
    )
    return {
        "name": "performance",
        "num_parts": args.num_parts,
        "checkpoint_num_parts": checkpoint_num_parts,
        "test_size": args.test_size,
        "test_seed": args.seed,
        "limit": args.limit,
        "repeats": args.repeats,
        "data": args.data or default_data_path(args.num_parts, args.seed),
        "overwrite_dataset": args.overwrite_dataset,
        "device": args.device,
        "nco_batch_size": args.nco_batch_size,
        "ga_pop_size": args.ga_pop_size,
        "ga_generations": args.ga_generations,
        "sa_iterations": args.sa_iterations,
        "cpccd_alpha": args.cpccd_alpha,
        "generator_params": generator_params,
        "algorithms": algorithms,
        "output_dir": output_dir,
    }


def summarize_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("algorithm")
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            bks_gap_pct_mean=("bks_gap_pct", "mean"),
            bks_gap_pct_std=("bks_gap_pct", "std"),
            num_groups_mean=("num_groups", "mean"),
            feasible_mean=("feasible", "mean"),
            n=("score", "count"),
        )
        .reset_index()
        .sort_values("algorithm")
    )


def summarize_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("algorithm")
        .agg(
            wall_elapsed_sec_mean=("wall_elapsed_sec", "mean"),
            wall_elapsed_sec_std=("wall_elapsed_sec", "std"),
            wall_elapsed_sec_total=("wall_elapsed_sec", "sum"),
            solver_elapsed_sec_mean=("solver_elapsed_sec", "mean"),
            solver_elapsed_sec_total=("solver_elapsed_sec", "sum"),
            n=("wall_elapsed_sec", "count"),
        )
        .reset_index()
        .sort_values("algorithm")
    )


def summarize_stability(df: pd.DataFrame) -> pd.DataFrame:
    per_instance = (
        df.groupby(["algorithm", "instance_idx"])
        .agg(
            score_repeat_mean=("score", "mean"),
            score_repeat_std=("score", "std"),
            bks_gap_pct_repeat_mean=("bks_gap_pct", "mean"),
            bks_gap_pct_repeat_std=("bks_gap_pct", "std"),
        )
        .reset_index()
    )
    per_instance[["score_repeat_std", "bks_gap_pct_repeat_std"]] = per_instance[
        ["score_repeat_std", "bks_gap_pct_repeat_std"]
    ].fillna(0.0)
    return (
        per_instance.groupby("algorithm")
        .agg(
            mean_score=("score_repeat_mean", "mean"),
            mean_bks_gap_pct=("bks_gap_pct_repeat_mean", "mean"),
            mean_within_instance_score_std=("score_repeat_std", "mean"),
            mean_within_instance_gap_std=("bks_gap_pct_repeat_std", "mean"),
            n_instances=("instance_idx", "count"),
        )
        .reset_index()
        .sort_values("algorithm")
    )


def make_overall_summary(
    effectiveness: pd.DataFrame,
    efficiency: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    out = effectiveness.merge(efficiency, on=["algorithm", "n"], how="outer")
    return out.merge(stability, on="algorithm", how="outer", suffixes=("", "_stability"))


def run(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    data_path = Path(config["data"])
    dataset = load_or_generate_dataset(
        path=data_path,
        size=config["test_size"],
        seed=config["test_seed"],
        generator_params=config["generator_params"],
        overwrite=config["overwrite_dataset"],
    )

    df = evaluate_specs(
        dataset=dataset,
        algorithms=config["algorithms"],
        limit=config["limit"],
        output_csv=output_dir / "raw_results.csv",
        seed=config["test_seed"],
        repeats=config["repeats"],
        device=config["device"],
        nco_batch_size=config["nco_batch_size"],
        ga_pop_size=config["ga_pop_size"],
        ga_generations=config["ga_generations"],
        sa_iterations=config["sa_iterations"],
        cpccd_alpha=config["cpccd_alpha"],
        extra_columns={
            "num_parts_case": config["num_parts"],
            "checkpoint_num_parts": config["checkpoint_num_parts"],
        },
    )
    df = add_bks_gap(df, group_cols=["repeat", "instance_idx"])
    save_dataframe(df, output_dir / "results_with_bks_gap.csv")

    effectiveness = summarize_effectiveness(df)
    efficiency = summarize_efficiency(df)
    stability = summarize_stability(df)
    overall = make_overall_summary(effectiveness, efficiency, stability)

    save_dataframe(effectiveness, output_dir / "effectiveness_summary.csv")
    save_dataframe(efficiency, output_dir / "efficiency_summary.csv")
    save_dataframe(stability, output_dir / "stability_summary.csv")
    save_dataframe(overall, output_dir / "overall_summary.csv")

    save_metric_boxplots(
        df,
        metrics=["score", "bks_gap_pct", "wall_elapsed_sec"],
        output_dir=output_dir / "plots",
        title_prefix=f"Performance n={config['num_parts']}",
    )
    save_nonparametric_tests(
        df,
        methods=sorted(df["algorithm"].unique()),
        metrics=["score", "bks_gap_pct", "wall_elapsed_sec"],
        output_dir=output_dir / "stat_tests",
    )
    save_json(
        output_dir / "config.json",
        {
            key: str(value)
            if isinstance(value, Path)
            else value
            for key, value in config.items()
            if key != "algorithms"
        },
    )
    return df


def main() -> None:
    args = parse_args()
    run(make_config(args))


if __name__ == "__main__":
    main()
