from __future__ import annotations

import pandas as pd

from Evaluation.common import (
    DEFAULT_GENERATOR_PARAMS,
    N20_ALGORITHMS,
    OUTPUT_ROOT,
    add_bks_gap,
    dataset_path,
    evaluate_specs,
    load_or_generate_dataset,
    save_dataframe,
    save_json,
    save_metric_boxplots,
    save_nonparametric_tests,
    summarize_by_algorithm,
)


SHIFTED_DISTRIBUTION = {
    **DEFAULT_GENERATOR_PARAMS,
    "L_low": 20.0,
    "L_high": 150.0,
    "W_low": 20.0,
    "W_high": 150.0,
    "H_low": 20.0,
    "H_high": 150.0,
    "p_maint_H_low": 0.35,
    "p_maint_H_high": 0.80,
    "p_standard_low": 0.02,
    "p_standard_high": 0.20,
    "p_relative_motion_low": 0.35,
    "p_relative_motion_high": 0.80,
}


CONFIG = {
    "name": "generalization",
    "test_size": 100,
    "test_seed": 1234,
    "overwrite_dataset": False,
    "limit": 100,
    "device": "cpu",
    "nco_batch_size": 100,
    "ga_pop_size": 100,
    "ga_generations": 3000,
    "sa_iterations": 3000,
    "cpccd_alpha": 0.5,
    "cases": [
        {
            "case": "size_n10_default_dist",
            "num_parts": 10,
            "generator_params": dict(DEFAULT_GENERATOR_PARAMS),
        },
        {
            "case": "size_n20_default_dist",
            "num_parts": 20,
            "generator_params": dict(DEFAULT_GENERATOR_PARAMS),
        },
        {
            "case": "size_n50_default_dist",
            "num_parts": 50,
            "generator_params": dict(DEFAULT_GENERATOR_PARAMS),
        },
        {
            "case": "type_n20_shifted_dist",
            "num_parts": 20,
            "generator_params": dict(SHIFTED_DISTRIBUTION),
        },
        {
            "case": "size_type_n50_shifted_dist",
            "num_parts": 50,
            "generator_params": dict(SHIFTED_DISTRIBUTION),
        },
    ],
    "algorithms": N20_ALGORITHMS,
}


def run(config: dict = CONFIG):
    output_dir = OUTPUT_ROOT / config["name"]
    all_frames: list[pd.DataFrame] = []

    for case in config["cases"]:
        case_name = case["case"]
        num_parts = case["num_parts"]
        generator_params = dict(case["generator_params"])
        generator_params["num_parts"] = num_parts

        dataset = load_or_generate_dataset(
            path=dataset_path(case_name, num_parts, config["test_size"], config["test_seed"]),
            size=config["test_size"],
            seed=config["test_seed"],
            generator_params=generator_params,
            overwrite=config["overwrite_dataset"],
        )
        df = evaluate_specs(
            dataset=dataset,
            algorithms=config["algorithms"],
            limit=config["limit"],
            output_csv=output_dir / f"raw_results_{case_name}.csv",
            seed=config["test_seed"],
            device=config["device"],
            nco_batch_size=config["nco_batch_size"],
            ga_pop_size=config["ga_pop_size"],
            ga_generations=config["ga_generations"],
            sa_iterations=config["sa_iterations"],
            cpccd_alpha=config["cpccd_alpha"],
            extra_columns={"generalization_case": case_name, "num_parts_case": num_parts},
        )
        df = add_bks_gap(df, group_cols=["generalization_case", "instance_idx"])
        save_dataframe(df, output_dir / f"results_with_bks_gap_{case_name}.csv")
        save_metric_boxplots(
            df,
            metrics=["score", "bks_gap_pct", "wall_elapsed_sec"],
            output_dir=output_dir / f"plots_{case_name}",
            title_prefix=f"Generalization {case_name}",
        )
        summary = summarize_by_algorithm(df, metrics=["score", "bks_gap_pct", "wall_elapsed_sec"])
        summary["generalization_case"] = case_name
        save_dataframe(summary, output_dir / f"summary_{case_name}.csv")
        save_nonparametric_tests(
            df,
            methods=sorted(df["algorithm"].unique()),
            metrics=["score", "wall_elapsed_sec"],
            output_dir=output_dir / f"stat_tests_{case_name}",
        )
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    save_dataframe(combined, output_dir / "combined_results_with_bks_gap.csv")
    save_metric_boxplots(
        combined,
        metrics=["score", "bks_gap_pct", "wall_elapsed_sec"],
        output_dir=output_dir / "plots",
        group_col="generalization_case",
        title_prefix="Generalization by case",
    )
    combined_summary = (
        combined.groupby(["generalization_case", "algorithm"])
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            bks_gap_pct_mean=("bks_gap_pct", "mean"),
            bks_gap_pct_std=("bks_gap_pct", "std"),
            wall_elapsed_sec_mean=("wall_elapsed_sec", "mean"),
            wall_elapsed_sec_std=("wall_elapsed_sec", "std"),
            n=("score", "count"),
        )
        .reset_index()
    )
    save_dataframe(combined_summary, output_dir / "summary.csv")
    save_json(output_dir / "config.json", {k: v for k, v in config.items() if k not in {"algorithms", "cases"}})
    return combined


if __name__ == "__main__":
    run()
