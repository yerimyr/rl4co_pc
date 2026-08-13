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
)


CONFIG = {
    "name": "stability",
    "num_parts": 20,
    "test_size": 1000,
    "test_seed": 1234,
    "overwrite_dataset": False,
    "limit": 1000,
    "repeats": 10,
    "device": "cpu",
    "nco_batch_size": 1000,
    "ga_pop_size": 120,
    "ga_generations": 3000,
    "sa_iterations": 3000,
    "cpccd_alpha": 0.5,
    "generator_params": dict(DEFAULT_GENERATOR_PARAMS),
    "algorithms": DEFAULT_ALGORITHMS,
}


def summarize_stability(df: pd.DataFrame) -> pd.DataFrame:
    per_instance = (
        df.groupby(["algorithm", "instance_idx"])["score"]
        .agg(score_repeat_mean="mean", score_repeat_std="std")
        .reset_index()
    )
    per_instance["score_repeat_std"] = per_instance["score_repeat_std"].fillna(0.0)
    return (
        per_instance.groupby("algorithm")
        .agg(
            mean_score=("score_repeat_mean", "mean"),
            mean_within_instance_std=("score_repeat_std", "mean"),
            std_of_within_instance_std=("score_repeat_std", "std"),
            n_instances=("instance_idx", "count"),
        )
        .reset_index()
    )


def run(config: dict = CONFIG):
    output_dir = OUTPUT_ROOT / config["name"]
    generator_params = dict(config["generator_params"])
    generator_params["num_parts"] = config["num_parts"]

    dataset = load_or_generate_dataset(
        path=dataset_path(config["name"], config["num_parts"], config["test_size"], config["test_seed"]),
        size=config["test_size"],
        seed=config["test_seed"],
        generator_params=generator_params,
        overwrite=config["overwrite_dataset"],
    )
    df = evaluate_specs(
        dataset=dataset,
        algorithms=config["algorithms"],
        limit=config["limit"],
        output_csv=output_dir / "raw_results.csv",
        seed=config["test_seed"],
        repeats=config["repeats"],
        device=config["device"],
        nco_batch_size=config["nco_batch_size"],
        ga_pop_size=config["ga_pop_size"],
        ga_generations=config["ga_generations"],
        sa_iterations=config["sa_iterations"],
        cpccd_alpha=config["cpccd_alpha"],
    )
    df = add_bks_gap(df, group_cols=["repeat", "instance_idx"])
    save_dataframe(df, output_dir / "results_with_bks_gap.csv")
    save_dataframe(summarize_stability(df), output_dir / "summary.csv")
    save_json(output_dir / "config.json", {k: v for k, v in config.items() if k != "algorithms"})
    return df


if __name__ == "__main__":
    run()
