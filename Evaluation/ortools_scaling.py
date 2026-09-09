from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Evaluation.common import OUTPUT_ROOT, ROOT, save_dataframe, save_json
from baseline.ortools_pc_solver import ORToolsPCSolver
from main import scalar, strip_sep_instance
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.pc.evaluator import evaluate_groups, score_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure OR-Tools status and runtime scaling as PC problem size N increases."
    )
    parser.add_argument(
        "--num-parts",
        type=str,
        default="10,20,30,50,70",
        help="Comma-separated N values to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument(
        "--distribution",
        choices=["newdist", "shifted"],
        default="newdist",
        help="Dataset suffix: pc{N}_{distribution}_test_seed{seed}.npz.",
    )
    parser.add_argument(
        "--data-template",
        type=str,
        default=None,
        help=(
            "Optional dataset template. Available fields: {N}, {seed}, {distribution}. "
            "Example: data/pc/pc{N}_newdist_test_seed{seed}.npz"
        ),
    )
    parser.add_argument("--ortools-time-limit", type=float, default=120.0)
    parser.add_argument("--ortools-workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/evaluation/ortools_scaling/{distribution}_seed{seed}_gamma{gamma}.",
    )
    return parser.parse_args()


def parse_num_parts(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("At least one N value is required.")
    return out


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def dataset_path(
    *,
    num_parts: int,
    seed: int,
    distribution: str,
    data_template: str | None,
) -> Path:
    if data_template:
        rendered = data_template.format(N=num_parts, seed=seed, distribution=distribution)
        return Path(rendered)
    return ROOT / "data" / "pc" / f"pc{num_parts}_{distribution}_test_seed{seed}.npz"


def evaluate_one_n(
    *,
    num_parts: int,
    data_path: Path,
    limit: int,
    gamma: float,
    time_limit_sec: float,
    workers: int,
) -> list[dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset for N={num_parts}: {data_path}")

    dataset = load_npz_to_tensordict(data_path)
    actual_limit = len(dataset) if limit <= 0 else min(limit, len(dataset))
    rows: list[dict[str, Any]] = []
    for idx in range(actual_limit):
        print(f"  OR-Tools N={num_parts}: instance {idx + 1}/{actual_limit}", flush=True)
        inst = strip_sep_instance(dataset[idx])
        solver = ORToolsPCSolver(
            gamma=gamma,
            time_limit_sec=time_limit_sec,
            num_workers=workers,
        )
        result = solver.solve_with_status(inst)
        metrics = score_metric_rows([evaluate_groups(result.groups, inst, gamma=gamma)])[0]
        rows.append(
            {
                "num_parts_case": num_parts,
                "instance_idx": idx,
                "dataset_path": str(data_path),
                "requested_num_parts": num_parts,
                "actual_num_parts": int(scalar(dataset[idx]["num_parts"])),
                "gamma": gamma,
                "status": result.status,
                "is_optimal": int(result.status == "OPTIMAL"),
                "is_feasible": int(result.status in {"OPTIMAL", "FEASIBLE"}),
                "solver_elapsed_sec": result.elapsed_sec,
                "wall_elapsed_sec": result.elapsed_sec,
                "ortools_objective_value": result.objective_value,
                "ortools_best_objective_bound": result.best_objective_bound,
                "groups": json.dumps(result.groups),
                **metrics,
            }
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    status_counts = (
        raw.pivot_table(
            index="num_parts_case",
            columns="status",
            values="instance_idx",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    grouped = (
        raw.groupby("num_parts_case")
        .agg(
            n=("instance_idx", "count"),
            optimal_count=("is_optimal", "sum"),
            feasible_count=("is_feasible", "sum"),
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            num_groups_mean=("num_groups", "mean"),
            wall_elapsed_sec_mean=("wall_elapsed_sec", "mean"),
            wall_elapsed_sec_std=("wall_elapsed_sec", "std"),
            wall_elapsed_sec_median=("wall_elapsed_sec", "median"),
            wall_elapsed_sec_max=("wall_elapsed_sec", "max"),
            wall_elapsed_sec_total=("wall_elapsed_sec", "sum"),
        )
        .reset_index()
    )
    grouped["optimal_ratio"] = grouped["optimal_count"] / grouped["n"]
    grouped["feasible_ratio"] = grouped["feasible_count"] / grouped["n"]
    return grouped.merge(status_counts, on="num_parts_case", how="left").sort_values(
        "num_parts_case"
    )


def save_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skip OR-Tools scaling plots: {exc}")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    x = summary["num_parts_case"].astype(int).to_numpy()
    optimal = summary["optimal_count"].astype(float).to_numpy()
    n = summary["n"].astype(float).to_numpy()
    not_optimal = n - optimal

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, optimal, label="OPTIMAL", color="#74c476")
    ax.bar(x, not_optimal, bottom=optimal, label="Not OPTIMAL", color="#fd8d3c")
    ax.set_xlabel("N")
    ax.set_ylabel("Number of instances")
    ax.set_title("OR-Tools optimal status by problem size")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "optimal_count_by_n.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, summary["optimal_ratio"].astype(float), marker="o", color="#238b45")
    ax.set_xlabel("N")
    ax.set_ylabel("Optimal ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("OR-Tools optimal ratio by problem size")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "optimal_ratio_by_n.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        summary["wall_elapsed_sec_mean"].astype(float),
        yerr=summary["wall_elapsed_sec_std"].fillna(0).astype(float),
        marker="o",
        capsize=4,
        color="#3182bd",
        label="Mean +/- std",
    )
    ax.plot(
        x,
        summary["wall_elapsed_sec_median"].astype(float),
        marker="s",
        color="#756bb1",
        label="Median",
    )
    ax.set_xlabel("N")
    ax.set_ylabel("Time / instance (sec)")
    ax.set_title("OR-Tools runtime by problem size")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "time_per_instance_by_n.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        summary["wall_elapsed_sec_total"].astype(float),
        marker="o",
        color="#08519c",
    )
    ax.set_xlabel("N")
    ax.set_ylabel("Total time (sec)")
    ax.set_title("OR-Tools total runtime by problem size")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "total_time_by_n.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def run(args: argparse.Namespace) -> pd.DataFrame:
    num_parts_values = parse_num_parts(args.num_parts)
    output_dir = args.output_dir or (
        OUTPUT_ROOT
        / "ortools_scaling"
        / f"{args.distribution}_seed{args.seed}_gamma{gamma_label(args.gamma)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for n in num_parts_values:
        path = dataset_path(
            num_parts=n,
            seed=args.seed,
            distribution=args.distribution,
            data_template=args.data_template,
        )
        print("")
        print("=" * 100)
        print(f"OR-Tools scaling: N={n}, data={path}")
        print("=" * 100)
        rows.extend(
            evaluate_one_n(
                num_parts=n,
                data_path=path,
                limit=args.limit,
                gamma=args.gamma,
                time_limit_sec=args.ortools_time_limit,
                workers=args.ortools_workers,
            )
        )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    save_dataframe(raw, output_dir / "raw_results.csv")
    save_dataframe(summary, output_dir / "summary.csv")
    save_json(
        output_dir / "config.json",
        {
            "num_parts": num_parts_values,
            "seed": args.seed,
            "limit": args.limit,
            "gamma": args.gamma,
            "distribution": args.distribution,
            "data_template": args.data_template,
            "ortools_time_limit": args.ortools_time_limit,
            "ortools_workers": args.ortools_workers,
            "output_dir": str(output_dir),
        },
    )
    save_plots(summary, output_dir)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
