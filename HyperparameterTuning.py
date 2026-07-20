from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from typing import Any

import torch

try:
    import optuna
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: optuna. Install it with: pip install optuna") from exc

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from rl4co.tasks.train import run


ALGORITHMS = ("am", "am_ppo", "pomo")
EXPERIMENT_BY_ALGORITHM = {
    "am": "pc/am_pc",
    "am_ppo": "pc/am_ppo_pc",
    "pomo": "pc/pomo_pc",
}
DEFAULT_OUTPUT_DIR = Path("outputs/optuna")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Optuna hyperparameter tuning for PC NCO models.")
    parser.add_argument(
        "--algorithm",
        choices=("am", "am_ppo", "pomo", "all"),
        default="all",
        help="Which NCO algorithm to tune.",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL. Defaults to per-algorithm SQLite.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--train-data-size", type=int, default=512)
    parser.add_argument("--val-data-size", type=int, default=256)
    parser.add_argument("--test-data-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--objective-metric", type=str, default="val/reward")
    parser.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--resume", action="store_true", help="Resume an existing study with the same name/storage.")
    parser.add_argument("--run-test", action="store_true", help="Run test loop during each Optuna trial.")
    return parser.parse_args()


def metric_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def get_metric(metric_dict: dict[str, Any], metric_name: str) -> float:
    if metric_name in metric_dict:
        return metric_to_float(metric_dict[metric_name])

    candidates = [
        metric_name.replace("_epoch", ""),
        f"{metric_name}_epoch",
        "val/reward",
        "val/reward_epoch",
        "test/reward",
        "reward",
    ]
    for key in candidates:
        if key in metric_dict:
            return metric_to_float(metric_dict[key])

    available = ", ".join(sorted(metric_dict.keys()))
    raise KeyError(f"Metric '{metric_name}' not found. Available metrics: {available}")


def suggest_common_params(trial: optuna.Trial) -> dict[str, Any]:
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
    valid_heads = [heads for heads in [4, 8] if embed_dim % heads == 0]
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-7, 1e-6, 1e-5]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "embed_dim": embed_dim,
        "num_encoder_layers": trial.suggest_categorical("num_encoder_layers", [2, 3, 4]),
        "num_heads": trial.suggest_categorical("num_heads", valid_heads),
    }


def suggest_algorithm_params(trial: optuna.Trial, algorithm: str) -> dict[str, Any]:
    params = suggest_common_params(trial)

    if algorithm == "am":
        params.update(
            {
                "baseline": trial.suggest_categorical("baseline", ["no", "mean", "exponential"]),
            }
        )
    elif algorithm == "am_ppo":
        params.update(
            {
                "mini_batch_size": trial.suggest_categorical("mini_batch_size", [32, 64, 128]),
                "ppo_epochs": trial.suggest_categorical("ppo_epochs", [1, 2, 3]),
                "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
                "vf_lambda": trial.suggest_categorical("vf_lambda", [0.1, 0.5, 1.0]),
                "entropy_lambda": trial.suggest_categorical("entropy_lambda", [0.0, 0.001, 0.01]),
                "normalize_adv": trial.suggest_categorical("normalize_adv", [False, True]),
                "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0]),
            }
        )
    elif algorithm == "pomo":
        params.update(
            {
                "num_starts": trial.suggest_categorical("num_starts", [2, 4, 8]),
                "normalization": trial.suggest_categorical("normalization", ["batch", "instance"]),
                "use_graph_context": trial.suggest_categorical("use_graph_context", [True, False]),
            }
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return params


def compose_trial_cfg(args: argparse.Namespace, algorithm: str, params: dict[str, Any], trial_dir: Path):
    GlobalHydra.instance().clear()
    with initialize(config_path="configs", version_base="1.3"):
        cfg = compose(
            config_name="main.yaml",
            return_hydra_config=True,
            overrides=[f"experiment={EXPERIMENT_BY_ALGORITHM[algorithm]}"],
        )

    HydraConfig().set_config(cfg)

    with open_dict(cfg):
        cfg.paths.root_dir = str(Path.cwd())
        cfg.paths.output_dir = str(trial_dir)
        cfg.paths.log_dir = str(args.output_dir)
        cfg.paths.work_dir = str(Path.cwd())

        cfg.seed = int(args.seed)
        cfg.test = bool(args.run_test)
        cfg.optimized_metric = args.objective_metric
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False

        cfg.logger = None
        cfg.callbacks.rich_progress_bar = None
        cfg.callbacks.model_summary = None
        cfg.callbacks.learning_rate_monitor = None
        cfg.callbacks.speed_monitor = None
        cfg.callbacks.tensorboard_logger = None

        cfg.trainer.max_epochs = int(args.max_epochs)
        cfg.trainer.accelerator = "gpu" if args.accelerator == "cuda" else args.accelerator
        cfg.trainer.devices = int(args.devices)
        cfg.trainer.precision = args.precision
        cfg.trainer.enable_progress_bar = False
        cfg.trainer.enable_model_summary = False
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.log_every_n_steps = 1

        cfg.model.generate_default_data = False
        cfg.model.batch_size = int(params["batch_size"])
        cfg.model.val_batch_size = int(args.val_batch_size)
        cfg.model.test_batch_size = int(args.test_batch_size)
        cfg.model.train_data_size = int(args.train_data_size)
        cfg.model.val_data_size = int(args.val_data_size)
        cfg.model.test_data_size = int(args.test_data_size)
        cfg.model.optimizer_kwargs.lr = float(params["lr"])
        cfg.model.optimizer_kwargs.weight_decay = float(params["weight_decay"])
        cfg.model.policy_kwargs.embed_dim = int(params["embed_dim"])
        cfg.model.policy_kwargs.num_encoder_layers = int(params["num_encoder_layers"])
        cfg.model.policy_kwargs.num_heads = int(params["num_heads"])

        if algorithm == "am":
            cfg.model.baseline = params["baseline"]
        elif algorithm == "am_ppo":
            cfg.model.critic_kwargs.embed_dim = int(params["embed_dim"])
            cfg.model.mini_batch_size = int(params["mini_batch_size"])
            cfg.model.ppo_epochs = int(params["ppo_epochs"])
            cfg.model.clip_range = float(params["clip_range"])
            cfg.model.vf_lambda = float(params["vf_lambda"])
            cfg.model.entropy_lambda = float(params["entropy_lambda"])
            cfg.model.normalize_adv = bool(params["normalize_adv"])
            cfg.model.max_grad_norm = float(params["max_grad_norm"])
        elif algorithm == "pomo":
            cfg.model.num_starts = int(params["num_starts"])
            cfg.model.num_augment = 1
            cfg.model.policy_kwargs.normalization = params["normalization"]
            cfg.model.policy_kwargs.use_graph_context = bool(params["use_graph_context"])

    return cfg


def objective_factory(args: argparse.Namespace, algorithm: str):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_algorithm_params(trial, algorithm)
        trial_dir = args.output_dir / algorithm / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        cfg = compose_trial_cfg(args, algorithm, params, trial_dir)
        OmegaConf.save(cfg, trial_dir / "config.yaml")

        try:
            metric_dict, object_dict = run(cfg)
            objective_value = get_metric(metric_dict, args.objective_metric)
            trial.set_user_attr("output_dir", str(trial_dir))
            trainer = object_dict.get("trainer")
            if trainer is not None and getattr(trainer, "checkpoint_callback", None) is not None:
                trial.set_user_attr("best_model_path", trainer.checkpoint_callback.best_model_path)
            for key, value in metric_dict.items():
                if isinstance(value, (int, float, torch.Tensor)):
                    try:
                        trial.set_user_attr(key, metric_to_float(value))
                    except Exception:
                        pass
            return objective_value
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def make_sampler(args: argparse.Namespace):
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=args.seed)
    return optuna.samplers.TPESampler(seed=args.seed, multivariate=True)


def storage_for(args: argparse.Namespace, algorithm: str) -> str:
    if args.storage is not None:
        return args.storage
    db_path = args.output_dir / algorithm / "study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def flatten_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    row: dict[str, Any] = {
        "number": trial.number,
        "value": trial.value,
        "state": trial.state.name,
    }
    for key, value in trial.params.items():
        row[f"param_{key}"] = value
    for key, value in trial.user_attrs.items():
        row[f"user_{key}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_study(args: argparse.Namespace, algorithm: str) -> optuna.Study:
    study_name = args.study_name or f"pc_{algorithm}"
    if args.algorithm == "all" and args.study_name is not None:
        study_name = f"{args.study_name}_{algorithm}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_for(args, algorithm),
        direction="maximize",
        sampler=make_sampler(args),
        load_if_exists=args.resume,
    )
    study.optimize(objective_factory(args, algorithm), n_trials=args.n_trials, gc_after_trial=True)

    out_dir = args.output_dir / algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "trials.csv", [flatten_trial(trial) for trial in study.trials])

    best = {
        "algorithm": algorithm,
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_user_attrs": dict(study.best_trial.user_attrs),
    }
    OmegaConf.save(OmegaConf.create(best), out_dir / "best_params.yaml")

    print(f"\n[{algorithm}] best value: {study.best_value:.6f}")
    print(f"[{algorithm}] best params: {study.best_params}")
    print(f"[{algorithm}] saved trials: {out_dir / 'trials.csv'}")
    print(f"[{algorithm}] saved best params: {out_dir / 'best_params.yaml'}")
    return study


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    algorithms = ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    studies = []
    for algorithm in algorithms:
        studies.append(run_study(args, algorithm))

    summary_rows = []
    for study, algorithm in zip(studies, algorithms):
        summary_rows.append(
            {
                "algorithm": algorithm,
                "study_name": study.study_name,
                "best_value": study.best_value,
                "best_trial_number": study.best_trial.number,
                **{f"param_{key}": value for key, value in study.best_params.items()},
            }
        )
    write_csv(args.output_dir / "summary.csv", summary_rows)
    print(f"\nSaved Optuna summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
