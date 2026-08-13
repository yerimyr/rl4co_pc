from __future__ import annotations

import pandas as pd

from Evaluation.common import (
    DEFAULT_ALGORITHMS,
    DEFAULT_GENERATOR_PARAMS,
    OUTPUT_ROOT,
    add_bks_gap,
    dataset_path,
    evaluate_specs,
    load_or_generate_dataset,
    save_dataframe,
    save_json,
    save_nonparametric_tests,
    summarize_by_algorithm,
)


CONFIG = {
    "name": "scalability",
    "num_parts_list": [10, 20, 30, 40, 50],
    "test_size": 1000,
    "test_seed": 1234,
    "overwrite_dataset": False,
    "limit": 1000,
    "device": "cpu",
    "nco_batch_size": 1000,
    "ga_pop_size": 120,
    "ga_generations": 3000,
    "sa_iterations": 3000,
    "cpccd_alpha": 0.5,
    "generator_params": dict(DEFAULT_GENERATOR_PARAMS),
    "algorithms": DEFAULT_ALGORITHMS,
}


def run(config: dict = CONFIG):
    output_dir = OUTPUT_ROOT / config["name"]
    all_frames: list[pd.DataFrame] = []

    for num_parts in config["num_parts_list"]:
        generator_params = dict(config["generator_params"])
        generator_params["num_parts"] = num_parts
        dataset = load_or_generate_dataset(
            path=dataset_path(config["name"], num_parts, config["test_size"], config["test_seed"]),
            size=config["test_size"],
            seed=config["test_seed"],
            generator_params=generator_params,
            overwrite=config["overwrite_dataset"],
        )
        df = evaluate_specs(
            dataset=dataset,
            algorithms=config["algorithms"],
            limit=config["limit"],
            output_csv=output_dir / f"raw_results_n{num_parts}.csv",
            seed=config["test_seed"],
            device=config["device"],
            nco_batch_size=config["nco_batch_size"],
            ga_pop_size=config["ga_pop_size"],
            ga_generations=config["ga_generations"],
            sa_iterations=config["sa_iterations"],
            cpccd_alpha=config["cpccd_alpha"],
            extra_columns={"num_parts_case": num_parts},
        )
        df = add_bks_gap(df, group_cols=["num_parts_case", "instance_idx"])
        save_dataframe(df, output_dir / f"results_with_bks_gap_n{num_parts}.csv")
        summary = summarize_by_algorithm(df, metrics=["score", "bks_gap_pct", "wall_elapsed_sec"])
        summary["num_parts_case"] = num_parts
        save_dataframe(summary, output_dir / f"summary_n{num_parts}.csv")
        save_nonparametric_tests(
            df,
            methods=sorted(df["algorithm"].unique()),
            metrics=["score", "wall_elapsed_sec"],
            output_dir=output_dir / f"stat_tests_n{num_parts}",
        )
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    save_dataframe(combined, output_dir / "combined_results_with_bks_gap.csv")
    combined_summary = (
        combined.groupby(["num_parts_case", "algorithm"])
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
    save_json(output_dir / "config.json", {k: v for k, v in config.items() if k != "algorithms"})
    return combined


if __name__ == "__main__":
    run()
