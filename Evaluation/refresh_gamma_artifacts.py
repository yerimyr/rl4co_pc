"""Rebuild gamma-sweep summaries and plots from authoritative raw results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from Evaluation.common import add_bks_gap, save_dataframe
from Evaluation.gamma_sweep import (
    default_data_path,
    save_gamma_boxplot,
    save_gamma_score_barplot,
    summarize,
)
from Evaluation.plot_optimal_attainment import prepare_attainment, save_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    for result_dir in args.result_dirs:
        result_dir = result_dir.resolve()
        raw_path = result_dir / "raw_results.csv"
        data = pd.read_csv(raw_path)
        invalid_ortools = (
            data["method"].astype(str).str.startswith("ortools")
            & ~data["solver_status"].isin(["OPTIMAL", "FEASIBLE"])
        )
        analysis = data.loc[~invalid_ortools].copy()
        analysis = add_bks_gap(analysis, group_cols=["gamma", "instance_idx"])

        save_dataframe(analysis, result_dir / "results_with_bks_gap.csv")
        save_dataframe(summarize(analysis), result_dir / "summary.csv")
        save_gamma_boxplot(analysis, result_dir / "plots")
        save_gamma_score_barplot(analysis, result_dir / "plots")
        save_plots(prepare_attainment(analysis), result_dir / "plots" / "optimal_attainment")

        config_path = result_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        num_parts = int(config.get("num_parts") or result_dir.name.split("_", 1)[0].removeprefix("n"))
        current_data_path = default_data_path(num_parts, args.seed)
        config.update(
            {
                "num_parts": num_parts,
                "seed": int(config.get("seed", args.seed)),
                "data": str(current_data_path.resolve()) if current_data_path.exists() else str(current_data_path),
                "gammas": sorted(float(value) for value in data["gamma"].dropna().unique()),
                "methods": sorted(str(value) for value in data["method"].dropna().unique()),
                "excluded_from_derived_artifacts": {
                    "reason": "OR-Tools status is neither OPTIMAL nor FEASIBLE",
                    "rows": int(invalid_ortools.sum()),
                },
            }
        )
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"Refreshed {result_dir} | raw={len(data)} | analyzed={len(analysis)} "
            f"| excluded={int(invalid_ortools.sum())}"
        )


if __name__ == "__main__":
    main()
