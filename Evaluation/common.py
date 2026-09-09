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

from main import evaluate_algorithm, evaluate_nco, strip_sep_instance, write_csv
from StatisticalHypothesisTesting import run_friedman_and_posthoc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "evaluation"
DATA_ROOT = ROOT / "data" / "pc" / "evaluation"


DEFAULT_GENERATOR_PARAMS: dict[str, Any] = {
    "num_parts": 20,
    "max_num_parts": None,
    "material_types": 2,
    "topology_mode": "mixed",
    "p_maint_H_low": 0.01,
    "p_maint_H_high": 0.02,
    "p_standard_low": 0.01,
    "p_standard_high": 0.02,
    "p_relative_motion_low": 0.01,
    "p_relative_motion_high": 0.02,
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
    AlgorithmSpec("nco_current_n10", "nco"),
    AlgorithmSpec("nco_matnet_n10", "nco"),
    AlgorithmSpec("nco_new_n10", "nco"),
    AlgorithmSpec("nco_current_n20", "nco"),
    AlgorithmSpec("nco_matnet_n20", "nco"),
    AlgorithmSpec("nco_new_n20", "nco"),
    AlgorithmSpec("nco_current_n30", "nco"),
    AlgorithmSpec("nco_matnet_n30", "nco"),
    AlgorithmSpec("nco_new_n30", "nco"),
    AlgorithmSpec("nco_current_n50", "nco"),
    AlgorithmSpec("nco_matnet_n50", "nco"),
    AlgorithmSpec("nco_new_n50", "nco"),
]


N20_ALGORITHMS: list[AlgorithmSpec] = [
    spec
    for spec in DEFAULT_ALGORITHMS
    if spec.kind == "baseline" or spec.name in {"nco_current_n20", "nco_matnet_n20", "nco_new_n20"}
]


def algorithms_for_num_parts(num_parts: int) -> list[AlgorithmSpec]:
    nco_names = {
        f"nco_current_n{num_parts}",
        f"nco_matnet_n{num_parts}",
        f"nco_new_n{num_parts}",
    }
    return [
        spec
        for spec in DEFAULT_ALGORITHMS
        if spec.kind == "baseline" or spec.name in nco_names
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
    plot_history: bool = False,
    plot_dir: Path | None = None,
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
        plot_history=plot_history,
        plot_dir=plot_dir or (OUTPUT_ROOT / "plots"),
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
    plot_history: bool = False,
    plot_dir: Path | None = None,
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
                plot_history=plot_history,
                plot_dir=plot_dir,
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
) -> pd.DataFrame:
    group_cols = group_cols or ["instance_idx"]
    out = df.copy()
    out["bks_score"] = out.groupby(group_cols)[score_col].transform("max")
    out["bks_gap"] = out["bks_score"] - out[score_col]
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_instance_plot(inst: dict[str, Any], path: Path, instance_idx: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skip instance plot: {exc}")
        return

    matrices = [
        ("Relation Weight W", np.asarray(inst.get("W", []), dtype=float)),
        ("Material Difference", np.asarray(inst.get("mat_var", []), dtype=float)),
        ("Maintenance Difference", np.asarray(inst.get("maint_diff", []), dtype=float)),
        ("Relative Motion", np.asarray(inst.get("rel_motion", []), dtype=float)),
    ]
    matrices = [(title, mat) for title, mat in matrices if mat.ndim == 2 and mat.size > 0]
    if not matrices:
        return

    cols = min(2, len(matrices))
    rows = int(np.ceil(len(matrices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.2 * rows))
    axes_arr = np.atleast_1d(axes).reshape(-1)

    for ax, (title, matrix) in zip(axes_arr, matrices):
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Part j")
        ax.set_ylabel("Part i")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_arr[len(matrices) :]:
        ax.axis("off")

    fig.suptitle(f"PC Instance {instance_idx:04d}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _format_array(value: Any, *, precision: int = 3) -> str:
    array = np.asarray(value)
    if array.dtype.kind in {"b", "i", "u"}:
        array = array.astype(int)
    with np.printoptions(precision=precision, suppress=True, linewidth=220):
        return np.array2string(array)


def _compat_matrix(inst: dict[str, Any]) -> np.ndarray | None:
    required = ("mat_var", "maint_diff", "rel_motion")
    if not all(key in inst for key in required):
        return None
    mat_var = np.asarray(inst["mat_var"]).astype(bool)
    maint_diff = np.asarray(inst["maint_diff"]).astype(bool)
    rel_motion = np.asarray(inst["rel_motion"]).astype(bool)
    return ~(mat_var | maint_diff | rel_motion)


def _search_space_proxy(compat: np.ndarray | None) -> dict[str, Any]:
    if compat is None or compat.ndim != 2:
        return {}

    comp = np.asarray(compat).astype(bool).copy()
    n = int(comp.shape[0])
    np.fill_diagonal(comp, False)
    pairs = int(np.triu(comp, k=1).sum())
    total_pairs = max(n * (n - 1) // 2, 1)

    triples = 0
    quads = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not comp[i, j]:
                continue
            for k in range(j + 1, n):
                if comp[i, k] and comp[j, k]:
                    triples += 1
                    for l in range(k + 1, n):
                        if comp[i, l] and comp[j, l] and comp[k, l]:
                            quads += 1

    # Fast greedy proxy, not an exact maximum clique solver.
    remaining = set(range(n))
    clique: list[int] = []
    degrees = comp.sum(axis=1)
    while remaining:
        node = max(remaining, key=lambda x: degrees[x])
        if all(comp[node, other] for other in clique):
            clique.append(node)
            remaining = {other for other in remaining if comp[node, other]}
        else:
            remaining.remove(node)

    return {
        "compat_density": pairs / total_pairs,
        "pairs": pairs,
        "triples": triples,
        "quads": quads,
        "greedy_max_clique": len(clique),
    }


def save_instance_report(inst: dict[str, Any], path: Path, instance_idx: int) -> None:
    compat = _compat_matrix(inst)
    proxy = _search_space_proxy(compat)

    lines = [
        f"===== PC EXPERIMENT INSTANCE {instance_idx} =====",
        "num_parts:",
        str(inst.get("num_parts", "")),
    ]

    if proxy:
        lines.extend(
            [
                "search_space_proxy:",
                (
                    f"compat_density={proxy['compat_density']:.3f}, "
                    f"pairs={proxy['pairs']}, triples={proxy['triples']}, "
                    f"quads={proxy['quads']}, greedy_max_clique={proxy['greedy_max_clique']}"
                ),
            ]
        )

    if "material_type_count" in inst:
        lines.extend(["material_type_count:", str(inst["material_type_count"])])

    for key, label in (
        ("material", "material"),
        ("maintfreq", "maintfreq"),
        ("isstandard", "isstandard"),
        ("assembly_adj", "assembly_adj"),
        ("mat_var", "mat_var"),
        ("maint_diff", "maint_diff"),
        ("rel_motion", "rel_motion"),
        ("W", "W"),
    ):
        if key in inst:
            lines.extend([f"{label}:", _format_array(inst[key])])

    if compat is not None:
        lines.extend(["compat:", _format_array(compat.astype(int))])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_instance_csvs(inst: dict[str, Any], csv_dir: Path) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)

    part_columns: dict[str, Any] = {"part_id": np.arange(1, int(inst["num_parts"]) + 1)}
    for key in ("material", "maintfreq", "isstandard"):
        if key in inst:
            part_columns[key] = np.asarray(inst[key])
    pd.DataFrame(part_columns).to_csv(csv_dir / "part_features.csv", index=False)

    for key in ("assembly_adj", "mat_var", "maint_diff", "rel_motion", "W"):
        if key in inst:
            pd.DataFrame(np.asarray(inst[key])).to_csv(
                csv_dir / f"{key}.csv", index=False, header=False
            )

    compat = _compat_matrix(inst)
    if compat is not None:
        pd.DataFrame(compat.astype(int)).to_csv(
            csv_dir / "compat.csv", index=False, header=False
        )


def save_instance_artifacts(dataset, limit: int, output_dir: Path) -> None:
    instance_dir = output_dir / "instances"
    json_dir = instance_dir / "json"
    report_dir = instance_dir / "txt"
    csv_dir = instance_dir / "csv"
    instance_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    actual_limit = min(int(limit), len(dataset))
    try:
        subset = dataset[:actual_limit]
        if hasattr(subset, "cpu"):
            subset = subset.cpu()
        save_tensordict_to_npz(subset, instance_dir / "instances_used.npz")
        print(f"Saved: {instance_dir / 'instances_used.npz'}")
    except Exception as exc:
        print(f"Skip instance npz export: {exc}")

    for idx in range(actual_limit):
        inst = strip_sep_instance(dataset[idx])
        json_path = json_dir / f"instance_{idx:04d}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(inst), f, indent=2, ensure_ascii=False)
        save_instance_report(inst, report_dir / f"instance_{idx:04d}.txt", idx)
        save_instance_csvs(inst, csv_dir / f"instance_{idx:04d}")

    print(f"Saved instance artifacts: {instance_dir}")


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


def save_metric_boxplots(
    df: pd.DataFrame,
    *,
    metrics: list[str],
    output_dir: Path,
    group_col: str = "algorithm",
    title_prefix: str = "",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"Skip boxplots: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(str(value) for value in df[group_col].dropna().unique())
    if not groups:
        print("Skip boxplots: no groups found.")
        return

    for metric in metrics:
        if metric not in df.columns:
            print(f"Skip boxplot for {metric}: column not found.")
            continue

        data = []
        labels = []
        for group in groups:
            values = df.loc[df[group_col].astype(str) == group, metric].dropna().astype(float)
            if len(values) == 0:
                continue
            data.append(values.to_numpy())
            labels.append(group)

        if not data:
            print(f"Skip boxplot for {metric}: no numeric values.")
            continue

        fig_width = max(8.0, 1.6 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_width, 5.2))
        boxplot = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=True)
        colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc", "#fdd0a2", "#c7e9c0"]
        for patch, color in zip(boxplot["boxes"], colors * ((len(labels) // len(colors)) + 1)):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        means = [float(np.mean(values)) for values in data]
        ax.scatter(
            range(1, len(labels) + 1),
            means,
            color="#d62728",
            marker="D",
            s=38,
            label="Mean",
            zorder=3,
        )
        title = f"{title_prefix} {metric} distribution".strip()
        ax.set_title(title)
        ax.set_xlabel(group_col)
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        output_path = output_dir / f"{metric}_boxplot.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Saved: {output_path}")


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
