from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from rl4co.envs.pc.env import PartConsolidationEnv

from main import evaluate_algorithm, evaluate_nco, write_csv
from StatisticalHypothesisTesting import run_friedman_and_posthoc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "evaluation"
DATA_ROOT = ROOT / "data" / "pc" / "evaluation"


DEFAULT_GENERATOR_PARAMS: dict[str, Any] = {
    "num_parts": 20,
    "max_num_parts": None,
    "material_types": None,
    "topology_mode": "mixed",
    "L_low": 5.0,
    "L_high": 100.0,
    "W_low": 5.0,
    "W_high": 100.0,
    "H_low": 5.0,
    "H_high": 100.0,
    "p_maint_H_low": 0.10,
    "p_maint_H_high": 0.50,
    "p_standard_low": 0.10,
    "p_standard_high": 0.50,
    "p_relative_motion_low": 0.10,
    "p_relative_motion_high": 0.50,
}


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    kind: str
    ckpt: str | None = None


DEFAULT_ALGORITHMS: list[AlgorithmSpec] = [
    AlgorithmSpec("cpccd", "baseline"),
    AlgorithmSpec("sa", "baseline"),
    AlgorithmSpec("ga", "baseline"),
    AlgorithmSpec(
        "nco_current_n10",
        "nco",
        ckpt=str(
            ROOT
            / "logs"
            / "train"
            / "runs"
            / "2026-08-14_00-14-46"
            / "checkpoints"
            / "epoch_029.ckpt"
        ),
    ),
    AlgorithmSpec(
        "nco_matnet_n10",
        "nco",
        ckpt=str(
            ROOT
            / "logs"
            / "train"
            / "runs"
            / "2026-08-14_01-50-10"
            / "checkpoints"
            / "epoch_004.ckpt"
        ),
    ),
    AlgorithmSpec(
        "nco_current_n20",
        "nco",
        ckpt=str(
            ROOT
            / "logs"
            / "train"
            / "runs"
            / "2026-08-13_17-47-58"
            / "checkpoints"
            / "epoch_059.ckpt"
        ),
    ),
    AlgorithmSpec(
        "nco_matnet_n20",
        "nco",
        ckpt=str(
            ROOT
            / "logs"
            / "train"
            / "runs"
            / "2026-08-13_21-03-29"
            / "checkpoints"
            / "epoch_004.ckpt"
        ),
    ),
    AlgorithmSpec(
        "nco_current_n50",
        "nco",
        ckpt=str(
            ROOT
            / "logs"
            / "train"
            / "runs"
            / "2026-08-14_03-23-30"
            / "checkpoints"
            / "epoch_004.ckpt"
        ),
    ),
    AlgorithmSpec(
        "nco_matnet_n50",
        "nco",
        ckpt=str(
            ROOT
            / "logs"
            / "train"
            / "runs"
            / "2026-08-14_20-37-28"
            / "checkpoints"
            / "epoch_004.ckpt"
        ),
    ),
]


N20_ALGORITHMS: list[AlgorithmSpec] = [
    spec
    for spec in DEFAULT_ALGORITHMS
    if spec.kind == "baseline" or spec.name in {"nco_current_n20", "nco_matnet_n20"}
]


def active_algorithms(algorithms: list[AlgorithmSpec]) -> list[AlgorithmSpec]:
    active: list[AlgorithmSpec] = []
    for spec in algorithms:
        if spec.kind == "nco" and not spec.ckpt:
            print(f"Skip {spec.name}: checkpoint path is not configured.")
            continue
        active.append(spec)
    return active


def make_main_args(
    *,
    seed: int,
    output: Path,
    ckpt: str | None = None,
    device: str = "cpu",
    nco_batch_size: int = 1000,
    ga_pop_size: int = 120,
    ga_generations: int = 3000,
    sa_iterations: int = 3000,
    cpccd_alpha: float = 0.5,
) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        output=output,
        ckpt=Path(ckpt) if ckpt else None,
        auto_ckpt=False,
        ckpt_root=ROOT / "logs" / "train" / "runs",
        device=device,
        nco_batch_size=nco_batch_size,
        ga_pop_size=ga_pop_size,
        ga_generations=ga_generations,
        sa_iterations=sa_iterations,
        cpccd_alpha=cpccd_alpha,
        plot_history=False,
        plot_dir=OUTPUT_ROOT / "plots",
    )


def dataset_path(name: str, num_parts: int, size: int, seed: int) -> Path:
    return DATA_ROOT / f"{name}_pc{num_parts}_size{size}_seed{seed}.npz"


def load_or_generate_dataset(
    *,
    path: Path,
    size: int,
    seed: int,
    generator_params: dict[str, Any],
    overwrite: bool = False,
):
    if path.exists() and not overwrite:
        return load_npz_to_tensordict(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = PartConsolidationEnv(generator_params=generator_params)
    td = env.generator(size)
    save_tensordict_to_npz(td, path)
    return td


def evaluate_specs(
    *,
    dataset,
    algorithms: list[AlgorithmSpec],
    limit: int,
    output_csv: Path,
    seed: int,
    repeats: int = 1,
    device: str = "cpu",
    nco_batch_size: int = 1000,
    ga_pop_size: int = 120,
    ga_generations: int = 3000,
    sa_iterations: int = 3000,
    cpccd_alpha: float = 0.5,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = active_algorithms(algorithms)
    if not specs:
        raise ValueError("No algorithms are active. Configure at least one checkpoint or baseline.")

    for repeat in range(repeats):
        repeat_seed = seed + repeat * 100_000
        for spec in specs:
            print(f"\n[{output_csv.stem}] {spec.name} repeat {repeat + 1}/{repeats}")
            args = make_main_args(
                seed=repeat_seed,
                output=output_csv,
                ckpt=spec.ckpt,
                device=device,
                nco_batch_size=nco_batch_size,
                ga_pop_size=ga_pop_size,
                ga_generations=ga_generations,
                sa_iterations=sa_iterations,
                cpccd_alpha=cpccd_alpha,
            )
            started = time.perf_counter()
            if spec.kind == "nco":
                spec_rows = evaluate_nco(dataset, limit, args)
                for row in spec_rows:
                    row["algorithm"] = spec.name
            else:
                spec_rows = evaluate_algorithm(spec.name, dataset, limit, args)

            for row in spec_rows:
                row["repeat"] = repeat
                row["run_seed"] = repeat_seed
                row["algorithm_wall_total_sec"] = time.perf_counter() - started
                if extra_columns:
                    row.update(extra_columns)
            rows.extend(spec_rows)

    write_csv(output_csv, rows)
    return pd.DataFrame(rows)


def add_bks_gap(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    score_col: str = "score",
    eps: float = 1e-8,
) -> pd.DataFrame:
    group_cols = group_cols or ["instance_idx"]
    out = df.copy()
    out["bks_score"] = out.groupby(group_cols)[score_col].transform("max")
    out["bks_gap_pct"] = (
        (out["bks_score"] - out[score_col]) / (out["bks_score"].abs() + eps) * 100.0
    )
    return out


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def summarize_by_algorithm(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    group_col: str = "algorithm",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algorithm, group in df.groupby(group_col):
        row: dict[str, Any] = {group_col: algorithm, "n": len(group)}
        for metric in metrics:
            if metric in group.columns:
                values = group[metric].astype(float)
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1)
                row[f"{metric}_median"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col)


def save_nonparametric_tests(
    df: pd.DataFrame,
    *,
    methods: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        if metric not in df.columns:
            continue
        try:
            overall, descriptives, pairwise, _ = run_friedman_and_posthoc(df, metric, methods)
        except Exception as exc:
            print(f"Skip test for {metric}: {exc}")
            continue
        save_dataframe(pd.DataFrame([overall]), output_dir / f"{metric}_friedman.csv")
        save_dataframe(descriptives, output_dir / f"{metric}_descriptives.csv")
        save_dataframe(pairwise, output_dir / f"{metric}_wilcoxon_holm.csv")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


def write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")
