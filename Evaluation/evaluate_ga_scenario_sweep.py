"""Run a diagnostic-rich GA sweep over PC sizes and modularity gamma values.

The evaluator is restartable: one result row and one diagnostic directory are
written after every completed instance. Re-running the same command skips
completed (N, gamma, instance) keys unless --rerun-completed is supplied.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline.ga_solver import GASolver
from Evaluation.common import ROOT
from Evaluation.gamma_sweep import default_data_path
from main import strip_sep_instance
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.pc.evaluator import evaluate_groups, score_metric_rows


DEFAULT_SIZES = (20, 30, 40)
DEFAULT_GAMMAS = (0.0, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GA with complete convergence diagnostics over 9 PC scenarios."
    )
    parser.add_argument("--num-parts", default="20,30,40")
    parser.add_argument("--gammas", default="0,0.5,1.0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=3000)
    parser.add_argument("--elite-size", type=int, default=0)
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--crossover-rate", type=float, default=0.90)
    parser.add_argument("--mutation-rate", type=float, default=0.30)
    parser.add_argument("--strong-mutation-prob", type=float, default=0.50)
    parser.add_argument("--strong-mutation-steps", type=int, default=5)
    parser.add_argument("--init-new-group-bias", type=float, default=0.60)
    parser.add_argument("--init-diverse-fraction", type=float, default=0.70)
    parser.add_argument("--enable-post-merge-repair", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/genetic_algorithm"),
    )
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    parsed = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not parsed:
        raise ValueError("At least one num-parts value is required")
    return parsed


def parse_floats(value: str) -> list[float]:
    parsed = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not parsed:
        raise ValueError("At least one gamma value is required")
    return parsed


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def append_log(path: Path, message: str) -> None:
    timestamped = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(timestamped, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(timestamped + "\n")


def make_solver(args: argparse.Namespace, gamma: float, run_seed: int) -> GASolver:
    return GASolver(
        pop_size=args.pop_size,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        strong_mutation_prob=args.strong_mutation_prob,
        strong_mutation_steps=args.strong_mutation_steps,
        init_new_group_bias=args.init_new_group_bias,
        enable_post_merge_repair=args.enable_post_merge_repair,
        init_diverse_fraction=args.init_diverse_fraction,
        gamma=gamma,
        seed=run_seed,
    )


def scenario_generation_summary(scenario_dir: Path, limit: int) -> pd.DataFrame:
    frames = []
    for instance_idx in range(limit):
        path = (
            scenario_dir
            / f"instance_{instance_idx:04d}"
            / "fitness_history_diagnostics.csv"
        )
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["instance_idx"] = instance_idx
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    diagnostics = pd.concat(frames, ignore_index=True)
    return (
        diagnostics.groupby("generation", as_index=False)
        .agg(
            best_fitness_mean=("best_raw_fitness", "mean"),
            best_fitness_std=("best_raw_fitness", "std"),
            all_time_best_fitness_mean=("all_time_best_raw_fitness", "mean"),
            all_time_best_fitness_std=("all_time_best_raw_fitness", "std"),
            population_fitness_mean=("mean_raw_fitness", "mean"),
            population_fitness_std=("mean_raw_fitness", "std"),
            unique_fitness_mean=("unique_raw_score_count", "mean"),
            unique_grouping_mean=("unique_grouping_count", "mean"),
            duplicate_ratio_mean=("duplicate_grouping_ratio", "mean"),
            parent_child_similarity_mean=("parent_child_mean_similarity", "mean"),
            mutation_count_mean=("mutation_count", "mean"),
            strong_mutation_ratio_mean=("strong_mutation_ratio", "mean"),
            completed_instances=("instance_idx", "nunique"),
        )
    )


def save_scenario_plot(summary: pd.DataFrame, path: Path, title: str) -> None:
    if summary.empty:
        return
    x = summary["generation"].to_numpy(float)
    best = summary["best_fitness_mean"].to_numpy(float)
    best_std = summary["best_fitness_std"].fillna(0.0).to_numpy(float)
    all_time_best = summary["all_time_best_fitness_mean"].to_numpy(float)
    mean = summary["population_fitness_mean"].to_numpy(float)
    mean_std = summary["population_fitness_std"].fillna(0.0).to_numpy(float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, best, label="Mean generation-best fitness", linewidth=2)
    axes[0].fill_between(x, best - best_std, best + best_std, alpha=0.18)
    axes[0].plot(
        x,
        all_time_best,
        label="Mean all-time-best fitness",
        linewidth=2.0,
        color="tab:green",
    )
    axes[0].plot(x, mean, label="Mean population fitness", linewidth=1.7)
    axes[0].fill_between(x, mean - mean_std, mean + mean_std, alpha=0.14)
    axes[0].set_ylabel("Fitness")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        x,
        summary["unique_grouping_mean"],
        label="Unique groupings",
        linewidth=1.7,
    )
    axes[1].plot(
        x,
        100.0 * summary["duplicate_ratio_mean"],
        label="Duplicate ratio (%)",
        linewidth=1.7,
    )
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Diversity diagnostic")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overall_summary(raw: pd.DataFrame, output_dir: Path) -> None:
    if raw.empty:
        return
    summary = (
        raw.groupby(["num_parts", "gamma"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            score_sem=("score", "sem"),
            num_groups_mean=("num_groups", "mean"),
            feasible_rate=("feasible", "mean"),
            solver_elapsed_sec_mean=("solver_elapsed_sec", "mean"),
            solver_elapsed_sec_total=("solver_elapsed_sec", "sum"),
            n=("score", "size"),
        )
        .sort_values(["num_parts", "gamma"])
    )
    atomic_csv(summary, output_dir / "summary.csv")


def main() -> None:
    args = parse_args()
    sizes = parse_ints(args.num_parts)
    gammas = parse_floats(args.gammas)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = output_dir / "run.log"
    raw_path = output_dir / "raw_results.csv"

    config = {
        "num_parts": sizes,
        "gammas": gammas,
        "test_seed": args.seed,
        "instances_per_scenario": args.limit,
        "scenario_count": len(sizes) * len(gammas),
        "total_ga_runs": len(sizes) * len(gammas) * args.limit,
        "ga": {
            "population_size": args.pop_size,
            "generations": args.generations,
            "elite_size": args.elite_size,
            "tournament_size": args.tournament_size,
            "crossover_rate": args.crossover_rate,
            "mutation_rate": args.mutation_rate,
            "strong_mutation_probability": args.strong_mutation_prob,
            "strong_mutation_steps": args.strong_mutation_steps,
            "initial_new_group_bias": args.init_new_group_bias,
            "initial_diverse_fraction": args.init_diverse_fraction,
            "post_merge_repair": args.enable_post_merge_repair,
        },
        "fitness": {
            "name": "scaled generalized modularity",
            "formula": "score = 1000 * (Q_observed - gamma * Q_expected_unweighted)",
            "implementation": "rl4co.envs.pc.evaluator.evaluate_groups",
            "constraints": "GA initialization, crossover, mutation, and repair preserve feasible groupings",
        },
        "output_dir": str(output_dir),
    }
    config_path = output_dir / "config.json"
    if config_path.exists() and raw_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_previous = dict(previous_config)
        comparable_current = dict(config)
        comparable_previous.pop("output_dir", None)
        comparable_current.pop("output_dir", None)
        if comparable_previous != comparable_current:
            raise ValueError(
                "The output directory already contains GA results produced with "
                "different settings. Resume with the original settings or choose "
                f"another --output-dir: {output_dir}"
            )
    save_json(config, config_path)
    (output_dir / "fitness_definition.txt").write_text(
        "GA fitness function\n"
        "===================\n"
        "score = Q_gamma = 1000 * (Q_observed - gamma * Q_expected_unweighted)\n\n"
        "Q_observed measures within-group observed connection strength.\n"
        "Q_expected_unweighted is the configuration-model expected term before gamma weighting.\n"
        "All candidate groupings are repaired/stabilized to satisfy the PC grouping constraints.\n",
        encoding="utf-8",
    )

    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    completed = set()
    if not raw.empty and not args.rerun_completed:
        completed = {
            (int(row.num_parts), round(float(row.gamma), 12), int(row.instance_idx))
            for row in raw.itertuples()
        }

    total_runs = len(sizes) * len(gammas) * args.limit
    append_log(
        progress_log,
        f"Start/resume GA sweep: {total_runs} runs, completed={len(completed)}, "
        f"population={args.pop_size}, generations={args.generations}",
    )

    rows = raw.to_dict("records") if not raw.empty else []
    for num_parts in sizes:
        data_path = default_data_path(num_parts, args.seed)
        if not data_path.exists():
            raise FileNotFoundError(f"Missing fixed test dataset: {data_path}")
        dataset = load_npz_to_tensordict(data_path)
        limit = min(args.limit, len(dataset))

        for gamma_index, gamma in enumerate(gammas):
            scenario_dir = output_dir / "scenarios" / f"n{num_parts}" / f"gamma_{gamma_label(gamma)}"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            append_log(
                progress_log,
                f"Scenario N={num_parts}, gamma={gamma:g}, instances={limit}",
            )
            for instance_idx in range(limit):
                key = (num_parts, round(gamma, 12), instance_idx)
                if key in completed:
                    append_log(progress_log, f"SKIP completed N={num_parts}, gamma={gamma:g}, i={instance_idx}")
                    continue

                run_seed = args.seed + num_parts * 1_000_000 + gamma_index * 10_000 + instance_idx
                instance_dir = scenario_dir / f"instance_{instance_idx:04d}"
                instance_dir.mkdir(parents=True, exist_ok=True)
                inst = strip_sep_instance(dataset[instance_idx])
                solver = make_solver(args, gamma, run_seed)

                append_log(progress_log, f"RUN N={num_parts}, gamma={gamma:g}, i={instance_idx}, seed={run_seed}")
                wall_started = time.perf_counter()
                groups, solver_elapsed = solver.solve(inst)
                wall_elapsed = time.perf_counter() - wall_started
                metrics = score_metric_rows(
                    [evaluate_groups(groups, inst, gamma=gamma)]
                )[0]
                if solver.last_best_score is None or not np.isclose(
                    float(metrics["score"]), float(solver.last_best_score), rtol=1e-8, atol=1e-8
                ):
                    raise RuntimeError(
                        "Final reported score does not match the GA fitness: "
                        f"metric={metrics['score']}, GA={solver.last_best_score}"
                    )

                fitness_plot = instance_dir / "fitness_history.png"
                solver.plot_fitness_history(str(fitness_plot), show=False)
                result = {
                    "num_parts": num_parts,
                    "gamma": gamma,
                    "instance_idx": instance_idx,
                    "run_seed": run_seed,
                    "dataset_path": str(data_path.resolve()),
                    "score": float(metrics["score"]),
                    "ga_best_fitness": float(solver.last_best_score),
                    "solver_elapsed_sec": float(solver_elapsed),
                    "wall_elapsed_sec": float(wall_elapsed),
                    "groups": groups,
                    **{key_name: float(value) for key_name, value in metrics.items()},
                    "fitness_plot": str(fitness_plot.resolve()),
                    "generation_diagnostics_csv": str(
                        fitness_plot.with_name("fitness_history_diagnostics.csv").resolve()
                    ),
                    "parent_child_examples_csv": str(
                        fitness_plot.with_name("fitness_history_parent_child_examples.csv").resolve()
                    ),
                }
                save_json(result, instance_dir / "result.json")

                rows = [
                    row
                    for row in rows
                    if not (
                        int(row["num_parts"]) == num_parts
                        and np.isclose(float(row["gamma"]), gamma)
                        and int(row["instance_idx"]) == instance_idx
                    )
                ]
                csv_row = dict(result)
                csv_row["groups"] = json.dumps(groups)
                rows.append(csv_row)
                raw = pd.DataFrame(rows).sort_values(["num_parts", "gamma", "instance_idx"])
                atomic_csv(raw, raw_path)
                save_overall_summary(raw, output_dir)
                completed.add(key)
                append_log(
                    progress_log,
                    f"DONE N={num_parts}, gamma={gamma:g}, i={instance_idx}, "
                    f"score={metrics['score']:.6f}, sec={solver_elapsed:.2f}",
                )

            generation_summary = scenario_generation_summary(scenario_dir, limit)
            if not generation_summary.empty:
                atomic_csv(generation_summary, scenario_dir / "generation_summary.csv")
                save_scenario_plot(
                    generation_summary,
                    scenario_dir / "convergence_summary.png",
                    f"GA convergence over {int(generation_summary['completed_instances'].max())} instances "
                    f"| N={num_parts}, gamma={gamma:g}",
                )

    final_raw = pd.read_csv(raw_path)
    save_overall_summary(final_raw, output_dir)
    append_log(progress_log, f"Completed GA sweep: rows={len(final_raw)}, output={output_dir}")


if __name__ == "__main__":
    main()
