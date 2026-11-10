"""Compare final curriculum NCO with PC baselines on size ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Evaluation.common import ROOT, save_dataframe, save_json
from Evaluation.evaluate_curriculum_forgetting_ranges import parse_ranges, range_label
from Evaluation.gamma_sweep import (
    evaluate_cpccd,
    evaluate_mcl,
    evaluate_nco_custom,
    evaluate_ortools,
)
from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from rl4co.envs.pc.env import PartConsolidationEnv


METHODS = ("ortools", "cpccd", "mcl", "nco-curriculum")
METHOD_LABELS = {
    "ortools": "CP-SAT",
    "cpccd": "CPCCD",
    "mcl": "MCL",
    "nco-curriculum": "RL-based NCO",
}
COLORS = {
    "ortools": "#4e9bc6",
    "cpccd": "#f28e2b",
    "mcl": "#59a14f",
    "nco-curriculum": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="One curriculum root containing all five completed gamma folders.",
    )
    parser.add_argument("--original-run", type=Path, default=None)
    parser.add_argument("--retry-run", type=Path, default=None)
    parser.add_argument(
        "--retry-gammas",
        default="0.75,1.0",
        help=(
            "Comma-separated gammas loaded from --retry-run when using a mixed "
            "original/retry checkpoint set."
        ),
    )
    parser.add_argument("--test-ranges", default="10-20,21-30,31-40")
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated methods to evaluate. Existing unselected methods are preserved.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace existing rows for the requested ranges, gammas, and methods.",
    )
    parser.add_argument(
        "--plot-gammas",
        default=None,
        help="Gammas shown in plots; by default all gammas present after merging are used.",
    )
    parser.add_argument("--test-seed", type=int, default=1234)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Instances evaluated per range; 0 uses the complete test dataset.",
    )
    parser.add_argument("--material-types", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nco-batch-size", type=int, default=100)
    parser.add_argument("--ortools-time-limit", type=float, default=300.0)
    parser.add_argument("--ortools-workers", type=int, default=1)
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/curriculum_range_test_seed1234"),
    )
    return parser.parse_args()


def gamma_name(gamma: float) -> str:
    return f"gamma_{gamma:g}".replace(".", "p")


def test_dataset_path(
    low: int,
    high: int,
    seed: int,
    size: int,
    material_types: int = 2,
) -> Path:
    material_suffix = "" if material_types == 2 else f"_material{material_types}"
    return (
        ROOT
        / "data"
        / "pc"
        / f"pc{low}-{high}_newdist{material_suffix}_test{size}_seed{seed}.npz"
    )


def load_or_generate_range_test(
    low: int,
    high: int,
    seed: int,
    size: int,
    material_types: int,
    overwrite: bool,
):
    path = test_dataset_path(low, high, seed, size, material_types)
    if path.exists() and not overwrite:
        return load_npz_to_tensordict(path), path

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = PartConsolidationEnv(
        generator_params={
            "num_parts": low,
            "max_num_parts": high,
            "material_types": material_types,
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
    print(f"Saved fixed test dataset: {path}")
    print(f"N counts: {counts.to_dict()}")
    return dataset, path


def final_checkpoint(
    gamma: float,
    *,
    run_root: Path | None,
    original: Path | None,
    retry: Path | None,
    retry_gammas: set[float],
) -> Path:
    if run_root is not None:
        root = run_root
    else:
        if original is None or retry is None:
            raise ValueError(
                "Provide --run-root, or provide both --original-run and --retry-run."
            )
        use_retry = any(np.isclose(gamma, value) for value in retry_gammas)
        root = retry if use_retry else original
    checkpoint_dir = root / gamma_name(gamma) / "stage3_n10-40" / "checkpoints"
    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"Final curriculum checkpoint not found: {checkpoint_dir}")
    return checkpoints[0].resolve()


def completed(raw: pd.DataFrame, size_range: str, gamma: float, method: str, n: int) -> bool:
    if raw.empty:
        return False
    mask = (
        (raw["test_range"] == size_range)
        & np.isclose(raw["gamma"].astype(float), gamma)
        & (raw["method"] == method)
    )
    return int(mask.sum()) >= n


def append_and_save(raw: pd.DataFrame, rows: list[dict], path: Path) -> pd.DataFrame:
    updated = pd.concat([raw, pd.DataFrame(rows)], ignore_index=True, sort=False)
    updated = updated.drop_duplicates(
        subset=["test_range", "gamma", "method", "instance_idx"], keep="last"
    ).sort_values(["test_range", "gamma", "method", "instance_idx"])
    save_dataframe(updated, path)
    return updated


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["test_range", "gamma", "method"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            score_sem=("score", "sem"),
            num_groups_mean=("num_groups", "mean"),
            feasible_rate=("feasible", "mean"),
            wall_elapsed_sec_mean=("wall_elapsed_sec", "mean"),
            wall_elapsed_sec_total=("wall_elapsed_sec", "sum"),
            n=("score", "size"),
        )
        .sort_values(["test_range", "gamma", "method"])
    )


def draw_range(
    ax,
    summary: pd.DataFrame,
    label: str,
    gammas: list[float],
    show_range_title: bool = False,
) -> None:
    subset = summary[summary["test_range"] == label]
    x = np.arange(len(gammas), dtype=float)
    width = 0.19
    offsets = np.linspace(
        -width * (len(METHODS) - 1) / 2,
        width * (len(METHODS) - 1) / 2,
        len(METHODS),
    )
    for offset, method in zip(offsets, METHODS):
        values = subset[subset["method"] == method].set_index("gamma").reindex(gammas)
        ax.bar(
            x + offset,
            values["score_mean"],
            width,
            yerr=values["score_std"],
            capsize=3,
            color=COLORS[method],
            alpha=0.9,
            label=METHOD_LABELS[method],
            error_kw={"elinewidth": 2.2, "capthick": 2.2},
        )
    ax.set_xticks(x, [f"{gamma:g}" for gamma in gammas])
    ax.set_xlabel("Gamma", fontsize=24)
    ax.set_ylabel("Score", fontsize=24)
    ax.tick_params(axis="both", labelsize=21, width=1.5, length=6)
    if show_range_title:
        ax.set_title(label, fontsize=22)
    ax.grid(True, axis="y", alpha=0.25)


def save_plots(summary: pd.DataFrame, output: Path, labels: list[str], gammas: list[float]) -> None:
    for label in labels:
        fig, ax = plt.subplots(figsize=(12, 7))
        draw_range(ax, summary, label, gammas)
        ax.legend(fontsize=20, frameon=True)
        fig.tight_layout()
        fig.savefig(output / f"{label.lower()}_method_comparison_mean_std.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 6))
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        draw_range(ax, summary, label, gammas, show_range_title=True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(METHODS), fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output / "all_ranges_method_comparison_mean_std.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve() if args.run_root is not None else None
    original = args.original_run.resolve() if args.original_run is not None else None
    retry = args.retry_run.resolve() if args.retry_run is not None else None
    if args.material_types < 1:
        raise ValueError("--material-types must be at least 1")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.csv"
    raw = (
        pd.DataFrame()
        if args.restart or not raw_path.exists()
        else pd.read_csv(raw_path)
    )

    gammas = [float(value) for value in args.gammas.split(",")]
    retry_gammas = {
        float(value) for value in args.retry_gammas.split(",") if value.strip()
    }
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(
            f"Unknown --methods values: {unknown_methods}. Available methods: {METHODS}"
        )
    if not methods:
        raise ValueError("--methods must contain at least one method")
    ranges = parse_ranges(args.test_ranges)
    labels = [range_label(low, high) for low, high in ranges]
    checkpoints = (
        {
            gamma: final_checkpoint(
                gamma,
                run_root=run_root,
                original=original,
                retry=retry,
                retry_gammas=retry_gammas,
            )
            for gamma in gammas
        }
        if "nco-curriculum" in methods
        else {}
    )

    if args.replace_existing and not raw.empty:
        gamma_mask = raw["gamma"].astype(float).map(
            lambda value: any(np.isclose(value, gamma) for gamma in gammas)
        )
        replace_mask = (
            raw["test_range"].isin(labels)
            & gamma_mask
            & raw["method"].isin(methods)
        )
        removed = int(replace_mask.sum())
        raw = raw.loc[~replace_mask].copy()
        save_dataframe(raw, raw_path)
        print(f"Removed {removed} existing rows before reevaluation.")

    result_gammas = sorted(
        set(gammas)
        | (set(raw["gamma"].astype(float).tolist()) if not raw.empty else set())
    )
    result_methods = sorted(
        set(methods)
        | (set(raw["method"].astype(str).tolist()) if not raw.empty else set())
    )
    config_path = output / "config.json"
    existing_config = {}
    if config_path.exists() and not args.restart:
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_config = {}
    recorded_checkpoints = dict(existing_config.get("checkpoints", {}))
    recorded_checkpoints.update(
        {f"{gamma:g}": str(path) for gamma, path in checkpoints.items()}
    )
    save_json(
        config_path,
        {
            "run_root": str(run_root) if run_root is not None else None,
            "original_run": str(original) if original is not None else None,
            "retry_run": str(retry) if retry is not None else None,
            "retry_gammas": sorted(retry_gammas),
            "test_ranges": args.test_ranges,
            "gammas": result_gammas,
            "evaluation_gammas": gammas,
            "test_seed": args.test_seed,
            "test_size": args.test_size,
            "limit": args.limit,
            "material_types": args.material_types,
            "device": args.device,
            "nco_batch_size": args.nco_batch_size,
            "ortools_time_limit": args.ortools_time_limit,
            "ortools_workers": args.ortools_workers,
            "methods": result_methods,
            "evaluation_methods": methods,
            "replace_existing": args.replace_existing,
            "checkpoints": recorded_checkpoints,
        },
    )

    for low, high in ranges:
        label = range_label(low, high)
        dataset, path = load_or_generate_range_test(
            low,
            high,
            args.test_seed,
            args.test_size,
            args.material_types,
            args.overwrite_data,
        )
        requested_limit = args.test_size if args.limit <= 0 else args.limit
        limit = min(requested_limit, len(dataset))
        sampled_sizes = dataset["num_parts"].cpu().numpy()
        print(f"\nTest range {label}: {path}")

        for gamma in gammas:
            for method in methods:
                if completed(raw, label, gamma, method, limit):
                    print(f"Skip completed: {label}, gamma={gamma:g}, {method}")
                    continue

                print(f"\nEvaluate {label}, gamma={gamma:g}, method={method}")
                if method == "ortools":
                    rows = evaluate_ortools(
                        dataset,
                        limit,
                        gamma,
                        args.ortools_time_limit,
                        args.ortools_workers,
                        "ortools",
                    )
                elif method == "cpccd":
                    rows = evaluate_cpccd(dataset, limit, gamma, args.test_seed, None)
                elif method == "mcl":
                    rows = evaluate_mcl(dataset, limit, gamma)
                else:
                    rows = evaluate_nco_custom(
                        dataset,
                        limit,
                        gamma,
                        checkpoints[gamma],
                        args.device,
                        args.nco_batch_size,
                    )
                    for row in rows:
                        row["method"] = "nco-curriculum"
                        row["algorithm"] = "nco-curriculum"

                for row in rows:
                    row["test_range"] = label
                    row["sampled_num_parts"] = int(sampled_sizes[row["instance_idx"]])
                    row["test_seed"] = args.test_seed
                    row["material_types_max"] = args.material_types
                    if method == "nco-curriculum":
                        row["curriculum_checkpoint"] = str(checkpoints[gamma])
                raw = append_and_save(raw, rows, raw_path)

    summary = summarize(raw)
    save_dataframe(summary, output / "summary.csv")
    for label, group in raw.groupby("test_range"):
        save_dataframe(group, output / f"{label.lower()}_raw_results.csv")
    plot_gammas = (
        [float(value) for value in args.plot_gammas.split(",")]
        if args.plot_gammas
        else sorted(raw["gamma"].astype(float).unique().tolist())
    )
    save_plots(summary, output, labels, plot_gammas)
    print(f"\nSaved curriculum range comparison: {output}")


if __name__ == "__main__":
    main()
