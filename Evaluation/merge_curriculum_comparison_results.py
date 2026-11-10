"""Merge disjoint-gamma curriculum comparison result folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Evaluation.common import save_dataframe
from Evaluation.evaluate_curriculum_vs_baselines import save_plots, summarize


KEY_COLUMNS = ["num_parts", "gamma", "method", "instance_idx"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for input_dir in args.inputs:
        path = input_dir.resolve() / "raw_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
        frame = pd.read_csv(path)
        missing = [column for column in KEY_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing key columns in {path}: {missing}")
        frame["source_result_dir"] = str(input_dir.resolve())
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_mask = raw.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_keys = raw.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates()
        raise ValueError(
            "The inputs overlap on result keys; refusing an ambiguous merge:\n"
            + duplicate_keys.to_string(index=False)
        )

    raw = raw.sort_values(KEY_COLUMNS).reset_index(drop=True)
    sizes = sorted(raw["num_parts"].astype(int).unique().tolist())
    gammas = sorted(raw["gamma"].astype(float).unique().tolist())
    methods = sorted(raw["method"].astype(str).unique().tolist())

    instances_per_scenario = raw.groupby(["num_parts", "gamma", "method"])[
        "instance_idx"
    ].nunique()
    if instances_per_scenario.nunique() != 1:
        raise ValueError(
            "Merged scenarios do not have a consistent instance count:\n"
            + instances_per_scenario.to_string()
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(raw)
    save_dataframe(raw, output_dir / "raw_results.csv")
    save_dataframe(summary, output_dir / "summary.csv")
    for num_parts, group in raw.groupby("num_parts"):
        save_dataframe(group, output_dir / f"n{int(num_parts)}_raw_results.csv")
    save_plots(summary, output_dir, sizes, gammas)

    print(f"Merged rows: {len(raw)}")
    print(f"N: {sizes}")
    print(f"Gamma: {gammas}")
    print(f"Methods: {methods}")
    print(f"Instances per N/gamma/method: {int(instances_per_scenario.iloc[0])}")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
