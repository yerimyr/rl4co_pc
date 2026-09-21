from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Evaluation.common import ROOT, OUTPUT_ROOT, add_bks_gap, save_dataframe, save_json
from main import (
    load_nco_model,
    node_groups_to_part_groups,
    resolve_device,
    scalar,
    select_nco_actions,
    strip_sep_instance,
)
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.evaluator import evaluate_groups, score_metric_rows


DEFAULT_GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OR-Tools, CPCCD, and gamma-specific nco-custom checkpoints."
    )
    parser.add_argument("--num-parts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Defaults to data/pc/pc{num_parts}_newdist_test_seed{seed}.npz.",
    )
    parser.add_argument(
        "--gammas",
        type=str,
        default=",".join(str(gamma) for gamma in DEFAULT_GAMMAS),
        help="Comma-separated gamma values.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="ortools,cpccd,nco-custom",
        help="Comma-separated methods from: ortools, cpccd, nco-custom.",
    )
    parser.add_argument("--run-root", type=Path, default=ROOT / "logs" / "train" / "runs")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--nco-batch-size", type=int, default=1)
    parser.add_argument("--cpccd-alpha", type=float, default=0.5)
    parser.add_argument("--ortools-time-limit", type=float, default=30.0)
    parser.add_argument("--ortools-workers", type=int, default=8)
    parser.add_argument(
        "--ortools-result-name",
        type=str,
        default="ortools",
        help="Method label stored for newly evaluated OR-Tools rows.",
    )
    parser.add_argument("--allow-missing-nco", action="store_true")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help=(
            "Merge newly evaluated rows into an existing raw_results.csv in the "
            "output directory. Rows with the same gamma, method, and instance_idx "
            "are replaced; all other existing rows are preserved."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/evaluation/gamma_sweep/n{num_parts}_seed{seed}.",
    )
    return parser.parse_args()


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def parse_csv_floats(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("At least one gamma value is required.")
    return out


def parse_csv_methods(value: str) -> list[str]:
    aliases = {
        "nco": "nco-custom",
        "nco_custom": "nco-custom",
        "nco-current": "nco-custom",
        "nco_current": "nco-custom",
        "google": "ortools",
        "google-or-tools": "ortools",
        "or-tools": "ortools",
    }
    methods = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        methods.append(aliases.get(key, key))
    allowed = {"ortools", "cpccd", "nco-custom"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    return methods


def default_data_path(num_parts: int, seed: int) -> Path:
    return ROOT / "data" / "pc" / f"pc{num_parts}_newdist_test_seed{seed}.npz"


def find_gamma_checkpoint(run_root: Path, num_parts: int, gamma: float) -> Path:
    label = gamma_label(gamma)
    expected_name = f"reinforce_edge_n{num_parts}_gamma{label}"
    expected_lower = expected_name.lower()
    matches = []
    for run in run_root.glob("*"):
        if not run.is_dir():
            continue
        run_lower = run.name.lower()
        matched = run_lower == expected_lower or f"({expected_lower})" in run_lower
        tensorboard_dir = run / "tensorboard"
        if not matched and tensorboard_dir.exists():
            matched = any(
                child.is_dir() and child.name.lower() == expected_lower
                for child in tensorboard_dir.iterdir()
            )
        if not matched:
            overrides_path = run / ".hydra" / "overrides.yaml"
            if overrides_path.exists():
                override_num_parts = None
                override_gamma = None
                for line in overrides_path.read_text(encoding="utf-8").splitlines():
                    value = line.removeprefix("- ").strip()
                    if value.startswith("env.generator_params.num_parts="):
                        override_num_parts = int(value.split("=", 1)[1])
                    elif value.startswith("env.modularity_gamma="):
                        override_gamma = float(value.split("=", 1)[1])
                matched = (
                    override_num_parts == num_parts
                    and override_gamma is not None
                    and np.isclose(override_gamma, gamma)
                )
        if matched:
            matches.append(run)

    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for run in matches:
        checkpoint_dir = run / "checkpoints"
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt.resolve()
        ckpts = sorted(
            checkpoint_dir.glob("*.ckpt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if ckpts:
            return ckpts[0].resolve()

    raise FileNotFoundError(f"No checkpoint found for run name {expected_name} under {run_root}")


def make_env_from_dataset(dataset: Any, device: torch.device, gamma: float) -> PartConsolidationEnv:
    sample = dataset[0]
    generator_params = {
        "num_parts": int(scalar(sample["num_parts"])),
        "material_types": int(scalar(sample["material_type_count"]))
        if "material_type_count" in sample.keys()
        else 3,
    }
    return PartConsolidationEnv(
        generator_params=generator_params,
        modularity_gamma=gamma,
        device=str(device),
    )


def evaluate_cpccd(dataset, limit: int, gamma: float, seed: int, cpccd_alpha: float) -> list[dict[str, Any]]:
    from baseline.cpccd_solver import CPCCDSolver

    rows = []
    for idx in range(limit):
        print(f"  Running cpccd gamma={gamma:g}: instance {idx + 1}/{limit}", flush=True)
        np.random.seed(seed + idx)
        torch.manual_seed(seed + idx)
        inst = strip_sep_instance(dataset[idx])
        solver = CPCCDSolver(alpha=cpccd_alpha)
        started = time.perf_counter()
        groups, solver_elapsed = solver.solve(inst)
        wall_elapsed = time.perf_counter() - started
        metrics = score_metric_rows([evaluate_groups(groups, inst, gamma=gamma)])[0]
        rows.append(
            {
                "gamma": gamma,
                "method": "cpccd",
                "algorithm": "cpccd",
                "instance_idx": idx,
                "score": float(metrics["score"]),
                "solver_elapsed_sec": float(solver_elapsed),
                "wall_elapsed_sec": float(wall_elapsed),
                "groups": json.dumps(groups),
                "solver_status": "OK",
                **{key: value for key, value in metrics.items()},
            }
        )
    return rows


def evaluate_ortools(
    dataset,
    limit: int,
    gamma: float,
    time_limit_sec: float,
    workers: int,
    result_name: str = "ortools",
) -> list[dict[str, Any]]:
    from baseline.ortools_pc_solver import ORToolsPCSolver

    rows = []
    for idx in range(limit):
        print(f"  Running ortools gamma={gamma:g}: instance {idx + 1}/{limit}", flush=True)
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
                "gamma": gamma,
                "method": result_name,
                "algorithm": result_name,
                "instance_idx": idx,
                "score": float(metrics["score"]),
                "solver_elapsed_sec": float(result.elapsed_sec),
                "wall_elapsed_sec": float(result.elapsed_sec),
                "groups": json.dumps(result.groups),
                "solver_status": result.status,
                "solver_has_solution": result.status in {"OPTIMAL", "FEASIBLE"},
                "ortools_time_limit_sec": float(time_limit_sec),
                "ortools_workers": int(workers),
                "ortools_objective_value": result.objective_value,
                "ortools_best_objective_bound": result.best_objective_bound,
                **{key: value for key, value in metrics.items()},
            }
        )
    return rows


def evaluate_nco_custom(
    dataset,
    limit: int,
    gamma: float,
    ckpt_path: Path,
    device_name: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    device = resolve_device(device_name)
    env = make_env_from_dataset(dataset, device, gamma)
    model, model_kind = load_nco_model(ckpt_path, env, device)
    num_starts = int(getattr(model, "num_starts", 0) or 0)
    rows = []

    print(f"  NCO gamma={gamma:g} checkpoint: {ckpt_path}")
    for start in range(0, limit, batch_size):
        end = min(start + batch_size, limit)
        print(f"  Running nco-custom gamma={gamma:g}: instances {start + 1}-{end}/{limit}", flush=True)
        batch = dataset[start:end].to(device)
        td = env.reset(batch)
        started = time.perf_counter()
        with torch.no_grad():
            if num_starts > 1:
                out = model.policy(td.clone(), env, phase="test", num_starts=num_starts)
            else:
                out = model.policy(td.clone(), env, phase="test")
        wall_elapsed = time.perf_counter() - started
        actions, model_reward = select_nco_actions(out, num_starts)
        node_groups = env.actions_to_groups(actions, td.cpu())
        per_instance_elapsed = wall_elapsed / max(end - start, 1)

        for local_idx, groups_with_sep in enumerate(node_groups):
            idx = start + local_idx
            inst = strip_sep_instance(dataset[idx])
            groups = node_groups_to_part_groups(groups_with_sep)
            metrics = score_metric_rows([evaluate_groups(groups, inst, gamma=gamma)])[0]
            rows.append(
                {
                    "gamma": gamma,
                    "method": "nco-custom",
                    "algorithm": "nco-custom",
                    "instance_idx": idx,
                    "score": float(metrics["score"]),
                    "model_reward": float(model_reward[local_idx].item()),
                    "solver_elapsed_sec": float(per_instance_elapsed),
                    "wall_elapsed_sec": float(per_instance_elapsed),
                    "groups": json.dumps(groups),
                    "solver_status": model_kind,
                    "ckpt": str(ckpt_path),
                    **{key: value for key, value in metrics.items()},
                }
            )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["gamma", "method"])
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            bks_gap_mean=("bks_gap", "mean"),
            bks_gap_std=("bks_gap", "std"),
            num_groups_mean=("num_groups", "mean"),
            wall_elapsed_sec_mean=("wall_elapsed_sec", "mean"),
            wall_elapsed_sec_total=("wall_elapsed_sec", "sum"),
            n=("score", "count"),
        )
        .reset_index()
        .sort_values(["gamma", "method"])
    )


def save_gamma_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skip gamma boxplot: {exc}")
        return

    methods = ["ortools", "ortools-parallel", "cpccd", "nco-custom"]
    methods = [method for method in methods if method in set(df["method"])]
    gammas = sorted(float(gamma) for gamma in df["gamma"].dropna().unique())
    colors = {
        "ortools": "#9ecae1",
        "ortools-parallel": "#3182bd",
        "cpccd": "#fdae6b",
        "nco-custom": "#a1d99b",
    }

    fig, ax = plt.subplots(figsize=(max(10.0, 1.25 * len(gammas) * len(methods)), 6.0))
    positions = []
    data = []
    colors_used = []
    labels = []
    width = 0.22
    offsets = np.linspace(-width, width, len(methods)) if len(methods) > 1 else np.array([0.0])

    for gamma_idx, gamma in enumerate(gammas, start=1):
        for offset, method in zip(offsets, methods):
            values = df.loc[
                (df["gamma"].astype(float) == gamma) & (df["method"] == method),
                "score",
            ].dropna()
            if values.empty:
                continue
            data.append(values.astype(float).to_numpy())
            positions.append(gamma_idx + float(offset))
            colors_used.append(colors.get(method, "#cccccc"))
            labels.append(method)

    boxplot = ax.boxplot(data, positions=positions, widths=0.18, patch_artist=True, showfliers=True)
    for patch, color in zip(boxplot["boxes"], colors_used):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for method in methods:
        ax.plot([], [], color=colors.get(method, "#cccccc"), linewidth=8, label=method)

    ax.set_xticks(range(1, len(gammas) + 1))
    ax.set_xticklabels([f"{gamma:g}" for gamma in gammas])
    ax.set_xlabel("gamma")
    ax.set_ylabel("score")
    ax.set_title("PC gamma sweep score distribution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="method")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gamma_score_boxplot.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_gamma_score_barplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Save mean score bars with one-standard-deviation error bars."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skip gamma score bar plot: {exc}")
        return

    method_order = ["ortools", "ortools-parallel", "cpccd", "nco-custom"]
    display_names = {
        "ortools": "OR-Tools (serial)",
        "ortools-parallel": "OR-Tools (10 workers)",
        "cpccd": "CPCCD",
        "nco-custom": "NCO",
    }
    colors = {
        "ortools": "#9ECAE1",
        "ortools-parallel": "#3182BD",
        "cpccd": "#F28E2B",
        "nco-custom": "#59A14F",
    }
    methods = [method for method in method_order if method in set(df["method"])]
    gammas = sorted(float(gamma) for gamma in df["gamma"].dropna().unique())
    summary = (
        df.groupby(["gamma", "method"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    x = np.arange(len(gammas), dtype=float)
    total_width = 0.78
    width = total_width / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(gammas)), 6.0))

    for method_idx, method in enumerate(methods):
        method_summary = summary[summary["method"] == method].set_index("gamma")
        means = np.array([method_summary.loc[gamma, "mean"] for gamma in gammas])
        stds = np.array([method_summary.loc[gamma, "std"] for gamma in gammas])
        offset = (method_idx - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width * 0.9,
            yerr=stds,
            capsize=4,
            color=colors[method],
            alpha=0.9,
            label=display_names[method],
            error_kw={"elinewidth": 1.3, "capthick": 1.3},
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{gamma:g}" for gamma in gammas])
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Score")
    ax.set_title("Mean score over test instances (error bars: ±1 SD)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Method")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gamma_score_barplot_mean_std.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    save_dataframe(summary.sort_values(["gamma", "method"]), output_dir / "gamma_score_barplot_summary.csv")
    print(f"Saved: {output_path}")


def run(args: argparse.Namespace) -> pd.DataFrame:
    gammas = parse_csv_floats(args.gammas)
    methods = parse_csv_methods(args.methods)
    data_path = args.data or default_data_path(args.num_parts, args.seed)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = load_npz_to_tensordict(data_path)
    limit = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    output_dir = args.output_dir or (
        OUTPUT_ROOT / "gamma_sweep" / f"n{args.num_parts}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for gamma in gammas:
        print("")
        print("=" * 100)
        print(f"Gamma = {gamma:g}")
        print("=" * 100)
        if "ortools" in methods:
            rows.extend(
                evaluate_ortools(
                    dataset,
                    limit,
                    gamma,
                    args.ortools_time_limit,
                    args.ortools_workers,
                    args.ortools_result_name,
                )
            )
        if "cpccd" in methods:
            rows.extend(evaluate_cpccd(dataset, limit, gamma, args.seed, args.cpccd_alpha))
        if "nco-custom" in methods:
            try:
                ckpt = find_gamma_checkpoint(args.run_root, args.num_parts, gamma)
            except FileNotFoundError:
                if args.allow_missing_nco:
                    print(f"  Skip nco-custom gamma={gamma:g}: checkpoint not found.")
                else:
                    raise
            else:
                rows.extend(
                    evaluate_nco_custom(
                        dataset,
                        limit,
                        gamma,
                        ckpt,
                        args.device,
                        args.nco_batch_size,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows were generated.")

    raw_results_path = output_dir / "raw_results.csv"
    if args.merge_existing and raw_results_path.exists():
        existing = pd.read_csv(raw_results_path)
        keys = ["gamma", "method", "instance_idx"]
        replacement_keys = df[keys].drop_duplicates()
        existing = existing.merge(
            replacement_keys.assign(_replace=True),
            on=keys,
            how="left",
        )
        kept = existing.loc[existing["_replace"].isna()].drop(columns="_replace")
        df = pd.concat([kept, df], ignore_index=True, sort=False)
        df = df.sort_values(keys).reset_index(drop=True)
        print(
            f"Merged results: kept {len(kept)} existing rows and added/replaced "
            f"{len(replacement_keys)} rows."
        )

    save_dataframe(df, raw_results_path)
    invalid_ortools = (
        df["method"].astype(str).str.startswith("ortools")
        & ~df["solver_status"].isin(["OPTIMAL", "FEASIBLE"])
    )
    if invalid_ortools.any():
        print(
            f"Exclude {int(invalid_ortools.sum())} OR-Tools rows without a solver "
            "solution from summaries and plots. They remain in raw_results.csv."
        )
    analysis_df = df.loc[~invalid_ortools].copy()
    analysis_df = add_bks_gap(analysis_df, group_cols=["gamma", "instance_idx"])
    save_dataframe(analysis_df, output_dir / "results_with_bks_gap.csv")
    save_dataframe(summarize(analysis_df), output_dir / "summary.csv")
    save_gamma_boxplot(analysis_df, output_dir / "plots")
    save_gamma_score_barplot(analysis_df, output_dir / "plots")
    from Evaluation.plot_optimal_attainment import prepare_attainment, save_plots

    save_plots(
        prepare_attainment(analysis_df),
        output_dir / "plots" / "optimal_attainment",
    )
    save_json(
        output_dir / "config.json",
        {
            "num_parts": args.num_parts,
            "seed": args.seed,
            "limit": limit,
            "data": str(data_path),
            # Describe the complete merged artifact, not only this invocation.
            "gammas": sorted(float(value) for value in df["gamma"].dropna().unique()),
            "methods": sorted(str(value) for value in df["method"].dropna().unique()),
            "last_invocation_gammas": gammas,
            "last_invocation_methods": methods,
            "run_root": str(args.run_root),
            "device": args.device,
            "nco_batch_size": args.nco_batch_size,
            "cpccd_alpha": args.cpccd_alpha,
            "ortools_time_limit": args.ortools_time_limit,
            "ortools_workers": args.ortools_workers,
            "ortools_result_name": args.ortools_result_name,
            "merge_existing": args.merge_existing,
        },
    )
    return df


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
