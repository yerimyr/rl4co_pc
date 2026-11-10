"""Evaluate CPCCD on the exact instances used by an existing GA scenario sweep."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.cpccd_solver import CPCCDSolver
from main import strip_sep_instance
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.pc.evaluator import evaluate_groups, score_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ga-dir",
        type=Path,
        default=Path("outputs/evaluation/genetic_algorithm"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def save_markdown(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        "num_parts",
        "gamma",
        "n",
        "ga_mean",
        "ga_std",
        "cpccd_mean",
        "cpccd_std",
        "cpccd_minus_ga_mean",
    ]
    header = (
        "| N | Gamma | n | GA mean | GA SD | CPCCD mean | CPCCD SD | "
        "CPCCD-GA mean |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for row in summary[columns].itertuples(index=False):
        lines.append(
            f"| {int(row.num_parts)} | {row.gamma:g} | {int(row.n)} | "
            f"{row.ga_mean:.2f} | {row.ga_std:.2f} | "
            f"{row.cpccd_mean:.2f} | {row.cpccd_std:.2f} | "
            f"{row.cpccd_minus_ga_mean:+.2f} |"
        )
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ga_dir = args.ga_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ga_dir / "cpccd_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ga_path = ga_dir / "raw_results.csv"
    ga = pd.read_csv(ga_path).sort_values(["num_parts", "gamma", "instance_idx"])
    keys = ["num_parts", "gamma", "instance_idx"]
    if ga.duplicated(keys).any():
        raise ValueError("GA raw results contain duplicate scenario-instance keys")
    scenario_counts = ga.groupby(["num_parts", "gamma"]).size()
    if len(ga) != 90 or len(scenario_counts) != 9 or not scenario_counts.eq(10).all():
        raise ValueError(
            "Expected 90 GA rows: 9 scenarios with 10 instances each; "
            f"got rows={len(ga)}, scenarios={len(scenario_counts)}, "
            f"counts={scenario_counts.to_dict()}"
        )

    datasets = {}
    cpccd_rows = []
    for row in ga.itertuples(index=False):
        dataset_path = Path(row.dataset_path).resolve()
        if dataset_path not in datasets:
            datasets[dataset_path] = load_npz_to_tensordict(dataset_path)
        dataset = datasets[dataset_path]
        instance_idx = int(row.instance_idx)
        inst = strip_sep_instance(dataset[instance_idx])

        # Threshold-free CPCCD, matching the current paper-reproduction logic.
        solver = CPCCDSolver(alpha=None)
        started = time.perf_counter()
        groups, solver_elapsed = solver.solve(inst)
        wall_elapsed = time.perf_counter() - started
        metrics = score_metric_rows(
            [evaluate_groups(groups, inst, gamma=float(row.gamma))]
        )[0]
        cpccd_rows.append(
            {
                "num_parts": int(row.num_parts),
                "gamma": float(row.gamma),
                "instance_idx": instance_idx,
                "dataset_path": str(dataset_path),
                "algorithm": "cpccd",
                "score": float(metrics["score"]),
                "solver_elapsed_sec": float(solver_elapsed),
                "wall_elapsed_sec": float(wall_elapsed),
                "groups": json.dumps(groups),
                "cpccd_modularity": float(solver.last_modularity),
                "cpccd_strategy": solver.last_strategy,
                **{key: float(value) for key, value in metrics.items()},
            }
        )
        print(
            f"CPCCD N={int(row.num_parts)}, gamma={float(row.gamma):g}, "
            f"instance={instance_idx + 1}/10, score={float(metrics['score']):.6f}",
            flush=True,
        )

    cpccd = pd.DataFrame(cpccd_rows).sort_values(keys)
    cpccd.to_csv(output_dir / "cpccd_raw_results.csv", index=False)

    ga_selected = ga[
        keys
        + [
            "dataset_path",
            "score",
            "solver_elapsed_sec",
            "wall_elapsed_sec",
            "groups",
            "feasible",
            "num_groups",
            "Q_gamma",
            "Q_observed",
            "Q_expected",
        ]
    ].copy()
    ga_selected["algorithm"] = "ga"
    combined = pd.concat([ga_selected, cpccd], ignore_index=True, sort=False)
    combined = combined.sort_values(keys + ["algorithm"])
    combined.to_csv(output_dir / "ga_cpccd_raw_results.csv", index=False)

    paired = ga[keys + ["score"]].rename(columns={"score": "ga_score"}).merge(
        cpccd[keys + ["score"]].rename(columns={"score": "cpccd_score"}),
        on=keys,
        validate="one_to_one",
    )
    paired["cpccd_minus_ga"] = paired["cpccd_score"] - paired["ga_score"]
    paired.to_csv(output_dir / "paired_scores.csv", index=False)

    summary = (
        paired.groupby(["num_parts", "gamma"], as_index=False)
        .agg(
            n=("instance_idx", "size"),
            ga_mean=("ga_score", "mean"),
            ga_std=("ga_score", "std"),
            cpccd_mean=("cpccd_score", "mean"),
            cpccd_std=("cpccd_score", "std"),
            cpccd_minus_ga_mean=("cpccd_minus_ga", "mean"),
            cpccd_minus_ga_std=("cpccd_minus_ga", "std"),
        )
        .sort_values(["num_parts", "gamma"])
    )
    summary.to_csv(output_dir / "ga_cpccd_summary.csv", index=False)
    save_markdown(summary, output_dir / "ga_cpccd_summary.md")
    print(f"Saved GA-CPCCD comparison: {output_dir}")


if __name__ == "__main__":
    main()
