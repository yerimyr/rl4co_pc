"""Audit PC training logs and evaluation artifacts for consistency.

The audit is read-only except for files written below ``--output-dir``.
It performs structural checks over every gamma-sweep result and deep score /
group validation for the paper N=30/40/50 result directories.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from Evaluation.gamma_sweep import default_data_path
from main import strip_sep_instance
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.pc.evaluator import evaluate_groups, score_metric_rows


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["gamma", "method", "instance_idx"]
REQUIRED_RAW_COLUMNS = {
    "gamma", "method", "instance_idx", "score", "groups", "solver_status",
    "feasible", "infeasible_solution", "infeasible_groups", "num_groups",
    "Q_gamma", "Q_observed", "Q_expected",
}


def add_issue(issues: list[dict[str, Any]], severity: str, scope: str, message: str) -> None:
    issues.append({"severity": severity, "scope": scope, "message": message})


def parse_groups(value: Any) -> list[list[int]]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or any(not isinstance(group, list) for group in parsed):
        raise ValueError("groups is not a list of lists")
    return [[int(node) for node in group] for group in parsed]


def structural_raw_audit(path: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    scope = str(path.relative_to(ROOT))
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        add_issue(issues, "ERROR", scope, f"cannot read CSV: {exc}")
        return {"path": scope, "rows": 0}

    missing = sorted(REQUIRED_RAW_COLUMNS.difference(data.columns))
    if missing:
        add_issue(issues, "ERROR", scope, f"missing columns: {missing}")
    if set(KEYS).issubset(data.columns):
        duplicates = int(data.duplicated(KEYS).sum())
        if duplicates:
            add_issue(issues, "ERROR", scope, f"duplicate gamma/method/instance rows: {duplicates}")
    numeric = [
        column for column in ["gamma", "instance_idx", "score", "num_groups", "Q_gamma",
                              "Q_observed", "Q_expected", "wall_elapsed_sec"]
        if column in data.columns
    ]
    for column in numeric:
        values = pd.to_numeric(data[column], errors="coerce")
        invalid = int((~np.isfinite(values)).sum())
        if invalid:
            add_issue(issues, "ERROR", scope, f"{column} has {invalid} non-finite/missing values")
    if {"score", "Q_gamma"}.issubset(data.columns):
        error = np.max(np.abs(data["score"].astype(float) - data["Q_gamma"].astype(float)))
        if error > 1e-6:
            add_issue(issues, "ERROR", scope, f"score != Q_gamma; max absolute error={error:.6g}")
    if "feasible" in data.columns and (data["feasible"].astype(float) != 1).any():
        count = int((data["feasible"].astype(float) != 1).sum())
        add_issue(issues, "ERROR", scope, f"contains {count} infeasible algorithm solutions")

    counts = {}
    if {"gamma", "method"}.issubset(data.columns):
        counts = {
            f"{gamma:g}/{method}": int(count)
            for (gamma, method), count in data.groupby(["gamma", "method"]).size().items()
        }
    return {"path": scope, "rows": len(data), "counts": counts}


def deep_paper_audit(path: Path, num_parts: int, seed: int, issues: list[dict[str, Any]]) -> dict[str, Any]:
    scope = str(path.relative_to(ROOT))
    data = pd.read_csv(path)
    dataset_path = default_data_path(num_parts, seed)
    if not dataset_path.exists():
        add_issue(issues, "ERROR", scope, f"dataset missing: {dataset_path}")
        return {"path": scope, "checked_rows": 0}
    dataset = load_npz_to_tensordict(dataset_path)

    max_errors = Counter()
    invalid_partitions = 0
    parse_errors = 0
    metric_mismatches = 0
    for row in data.itertuples(index=False):
        try:
            groups = parse_groups(row.groups)
        except Exception:
            parse_errors += 1
            continue
        flattened = [node for group in groups for node in group]
        if sorted(flattened) != list(range(num_parts)) or len(flattened) != len(set(flattened)):
            invalid_partitions += 1
            continue
        if int(round(float(row.num_groups))) != len(groups):
            max_errors["num_groups"] = max(max_errors["num_groups"], 1.0)

        idx = int(row.instance_idx)
        if idx < 0 or idx >= len(dataset):
            add_issue(issues, "ERROR", scope, f"instance_idx out of range: {idx}")
            continue
        inst = strip_sep_instance(dataset[idx])
        recalculated = score_metric_rows(
            [evaluate_groups(groups, inst, gamma=float(row.gamma))]
        )[0]
        for column in ["score", "Q_gamma", "Q_observed", "Q_expected", "num_groups",
                       "feasible", "infeasible_solution", "infeasible_groups"]:
            error = abs(float(getattr(row, column)) - float(recalculated[column]))
            max_errors[column] = max(max_errors[column], error)
            tolerance = 1e-5 if column not in {"feasible", "infeasible_solution", "infeasible_groups", "num_groups"} else 1e-8
            if error > tolerance:
                metric_mismatches += 1

    if parse_errors:
        add_issue(issues, "ERROR", scope, f"unparseable groups rows: {parse_errors}")
    if invalid_partitions:
        add_issue(issues, "ERROR", scope, f"invalid part partitions: {invalid_partitions}")
    if metric_mismatches:
        add_issue(issues, "ERROR", scope, f"recalculated metric mismatches: {metric_mismatches}")

    valid_for_analysis = data.loc[
        ~(
            data["method"].astype(str).str.startswith("ortools")
            & ~data["solver_status"].isin(["OPTIMAL", "FEASIBLE"])
        )
    ]
    summary_path = path.parent / "summary.csv"
    if summary_path.exists():
        stored = pd.read_csv(summary_path).sort_values(["gamma", "method"]).reset_index(drop=True)
        recomputed = (
            valid_for_analysis.groupby(["gamma", "method"])
            .agg(score_mean=("score", "mean"), score_std=("score", "std"), n=("score", "count"))
            .reset_index().sort_values(["gamma", "method"]).reset_index(drop=True)
        )
        merged = stored.merge(recomputed, on=["gamma", "method"], suffixes=("_stored", "_new"))
        if len(merged) != len(recomputed):
            add_issue(issues, "ERROR", scope, "summary.csv is missing gamma/method groups")
        else:
            for column in ["score_mean", "score_std", "n"]:
                delta = np.max(np.abs(merged[f"{column}_stored"] - merged[f"{column}_new"]))
                if delta > 1e-8:
                    add_issue(issues, "ERROR", scope, f"stale summary {column}; max delta={delta:.6g}")
    else:
        add_issue(issues, "WARNING", scope, "summary.csv missing")

    expected_plot = path.parent / "plots" / "gamma_score_barplot_mean_std.png"
    if not expected_plot.exists() or expected_plot.stat().st_mtime < path.stat().st_mtime:
        add_issue(issues, "WARNING", scope, "score bar plot missing or older than raw_results.csv")

    config_path = path.parent / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        raw_gammas = sorted(map(float, data["gamma"].unique()))
        raw_methods = sorted(map(str, data["method"].unique()))
        if sorted(map(float, config.get("gammas", []))) != raw_gammas:
            add_issue(issues, "WARNING", scope, "config.json gammas do not describe merged raw results")
        if sorted(map(str, config.get("methods", []))) != raw_methods:
            add_issue(issues, "WARNING", scope, "config.json methods do not describe merged raw results")
        configured_data = Path(str(config.get("data", "")))
        if str(configured_data) and not configured_data.exists():
            add_issue(issues, "WARNING", scope, f"config.json data path is stale: {configured_data}")
        if "5m" in path.parent.name and float(config.get("ortools_time_limit", 300.0)) != 300.0:
            add_issue(
                issues,
                "ERROR",
                scope,
                f"folder says ortools_5m but last recorded time limit is "
                f"{config.get('ortools_time_limit')} seconds",
            )

    if "ckpt" in data.columns:
        stale = sorted(
            set(
                str(value)
                for value in data.loc[data["method"].eq("nco-custom"), "ckpt"].dropna().unique()
                if not Path(str(value)).exists()
            )
        )
        if stale:
            add_issue(issues, "WARNING", scope, f"{len(stale)} recorded checkpoint paths are stale")

    status_counts = (
        data[data["method"].astype(str).str.startswith("ortools")]
        .groupby(["method", "solver_status"]).size().to_dict()
    )
    invalid_solver_rows = data[
        data["method"].astype(str).str.startswith("ortools")
        & ~data["solver_status"].isin(["OPTIMAL", "FEASIBLE"])
    ]
    if len(invalid_solver_rows):
        add_issue(
            issues,
            "ERROR",
            scope,
            f"{len(invalid_solver_rows)} OR-Tools rows have no solver solution "
            "(status is neither OPTIMAL nor FEASIBLE) and must be excluded",
        )
    feasible_reference = sum(count for (method, status), count in status_counts.items() if status != "OPTIMAL")
    if feasible_reference:
        add_issue(
            issues, "WARNING", scope,
            f"{feasible_reference} OR-Tools rows are not OPTIMAL; label them references, not proven optima",
        )
    return {
        "path": scope,
        "checked_rows": len(data),
        "max_recalculation_errors": dict(max_errors),
        "ortools_status_counts": {f"{k[0]}/{k[1]}": int(v) for k, v in status_counts.items()},
    }


def audit_training_runs(log_root: Path, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    if not log_root.exists():
        return records
    for run in sorted(path for path in log_root.rglob("20??-??-??_??-??-??") if path.is_dir()):
        overrides = run / ".hydra" / "overrides.yaml"
        checkpoints = list((run / "checkpoints").glob("*.ckpt")) if (run / "checkpoints").exists() else []
        version_dirs = list((run / "tensorboard").glob("*/version_*")) if (run / "tensorboard").exists() else []
        scope = str(run.relative_to(ROOT))
        if not overrides.exists():
            add_issue(issues, "WARNING", scope, "Hydra overrides missing")
        if not checkpoints:
            add_issue(issues, "ERROR", scope, "checkpoint missing")
        if not version_dirs:
            add_issue(issues, "ERROR", scope, "TensorBoard version directory missing")
            continue
        try:
            accumulator = EventAccumulator(str(version_dirs[0]), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
        except Exception as exc:
            add_issue(issues, "ERROR", scope, f"cannot load TensorBoard events: {exc}")
            continue
        scalar_info = {}
        for tag in tags:
            events = accumulator.Scalars(tag)
            values = np.array([event.value for event in events], dtype=float)
            steps = [event.step for event in events]
            if not np.isfinite(values).all():
                add_issue(issues, "ERROR", scope, f"non-finite TensorBoard scalar: {tag}")
            if len(steps) != len(set(steps)):
                add_issue(issues, "WARNING", scope, f"duplicate scalar steps: {tag}")
            scalar_info[tag] = {
                "count": len(events), "last_step": steps[-1] if steps else None,
                "first": float(values[0]) if len(values) else None,
                "last": float(values[-1]) if len(values) else None,
                "min": float(values.min()) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
            }
        if "train_epoch/entropy" in tags and "train_epoch/log_likelihood" in tags:
            entropy = {e.step: e.value for e in accumulator.Scalars("train_epoch/entropy")}
            ll = {e.step: e.value for e in accumulator.Scalars("train_epoch/log_likelihood")}
            common = sorted(set(entropy).intersection(ll))
            error = max((abs(entropy[s] + ll[s]) for s in common), default=0.0)
            if error > 0.25:
                add_issue(issues, "WARNING", scope, f"entropy and -log_likelihood diverge; max error={error:.4g}")
        records.append({"run": scope, "checkpoints": len(checkpoints), "scalars": scalar_info})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "audit")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    issues: list[dict[str, Any]] = []
    gamma_root = ROOT / "outputs" / "evaluation" / "gamma_sweep"
    structural = [
        structural_raw_audit(path, issues)
        for path in sorted(gamma_root.rglob("raw_results.csv"))
    ]
    deep = []
    for num_parts in (30, 40, 50):
        path = gamma_root / f"n{num_parts}_seed{args.seed}_test100_paper(ortools_5m)" / "raw_results.csv"
        if path.exists():
            deep.append(deep_paper_audit(path, num_parts, args.seed, issues))
        else:
            add_issue(issues, "ERROR", str(path.relative_to(ROOT)), "paper raw results missing")

    logs = []
    for name in ["runs", "runs(paper,N30)", "runs(paper,N40)", "runs(paper,N50)"]:
        logs.extend(audit_training_runs(ROOT / "logs" / "train" / name, issues))

    issue_df = pd.DataFrame(issues, columns=["severity", "scope", "message"])
    issue_df.to_csv(args.output_dir / "issues.csv", index=False)
    report = {"structural_results": structural, "deep_results": deep, "training_runs": logs, "issues": issues}
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    counts = Counter(item["severity"] for item in issues)
    print(f"Audited {len(structural)} raw result files, {len(deep)} paper datasets, {len(logs)} runs")
    print("Issues:", dict(counts))
    if issues:
        print(issue_df.to_string(index=False))
    print(f"Saved: {(args.output_dir / 'audit_report.json').resolve()}")


if __name__ == "__main__":
    main()
