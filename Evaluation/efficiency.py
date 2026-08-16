from __future__ import annotations

from Evaluation.common import (
    DEFAULT_GENERATOR_PARAMS,
    N20_ALGORITHMS,
    OUTPUT_ROOT,
    dataset_path,
    evaluate_specs,
    load_or_generate_dataset,
    save_dataframe,
    save_json,
    save_nonparametric_tests,
    summarize_by_algorithm,
)


CONFIG = {
    "name": "efficiency",
    "num_parts": 20,
    "test_size": 100,
    "test_seed": 1234,
    "overwrite_dataset": False,
    "limit": 100,
    "device": "cpu",
    "nco_batch_size": 100,
    "ga_pop_size": 120,
    "ga_generations": 3000,
    "sa_iterations": 3000,
    "cpccd_alpha": 0.5,
    "generator_params": dict(DEFAULT_GENERATOR_PARAMS),
    "algorithms": N20_ALGORITHMS,
}


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
        device=config["device"],
        nco_batch_size=config["nco_batch_size"],
        ga_pop_size=config["ga_pop_size"],
        ga_generations=config["ga_generations"],
        sa_iterations=config["sa_iterations"],
        cpccd_alpha=config["cpccd_alpha"],
    )
    summary = summarize_by_algorithm(df, metrics=["wall_elapsed_sec", "solver_elapsed_sec", "score"])
    save_dataframe(summary, output_dir / "summary.csv")
    save_nonparametric_tests(
        df,
        methods=sorted(df["algorithm"].unique()),
        metrics=["wall_elapsed_sec", "solver_elapsed_sec"],
        output_dir=output_dir / "stat_tests",
    )
    save_json(output_dir / "config.json", {k: v for k, v in config.items() if k != "algorithms"})
    return df


if __name__ == "__main__":
    run()
