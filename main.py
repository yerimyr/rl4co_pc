from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.evaluator import evaluate_groups, score_metric_rows
from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.amppo import AMPPO
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.ops import unbatchify


DEFAULT_DATA = Path("data/pc/pc20_test_seed1234.npz")  # 1234
DEFAULT_CKPT_ROOT = Path("logs/train/runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PC baselines and trained NCO checkpoints."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--algorithms",
        type=str,
        default="cpccd,sa,ga",
        help="Comma-separated list from: cpccd, sa, ga, nco.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of instances to evaluate. Use 0 to evaluate all instances.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("baseline_results.csv"))
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to an RL4CO checkpoint for the nco algorithm.",
    )
    parser.add_argument(
        "--auto-ckpt",
        action="store_true",
        help="Automatically use the latest logs/train/runs/**/checkpoints/epoch_*.ckpt.",
    )
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for NCO evaluation.",
    )
    parser.add_argument("--nco-batch-size", type=int, default=256)

    parser.add_argument("--ga-pop-size", type=int, default=120)
    parser.add_argument("--ga-generations", type=int, default=20000)  # 300
    parser.add_argument("--sa-iterations", type=int, default=10_000)
    parser.add_argument("--cpccd-alpha", type=float, default=0.3)
    parser.add_argument(
        "--plot-history",
        action="store_true",
        help="Save GA/SA optimization history plots under outputs/plots.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory used when --plot-history is enabled.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def find_latest_checkpoint(root: Path) -> Path:
    candidates = sorted(
        root.glob("**/checkpoints/epoch_*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found under {root}")
    return candidates[0]


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any) -> Any:
    arr = to_numpy(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def strip_sep_instance(td_item: Any) -> dict[str, Any]:
    """Convert an RL4CO PC instance to the no-SEP format expected by baselines."""
    num_parts = int(scalar(td_item["num_parts"]))
    node_slice = slice(1, num_parts + 1)

    inst: dict[str, Any] = {
        "num_parts": num_parts,
    }

    if "material_type_count" in td_item.keys():
        inst["material_type_count"] = int(scalar(td_item["material_type_count"]))

    for key in (
        "node_features",
        "valid_part_mask",
        "material",
        "size",
        "maintfreq",
        "isstandard",
        "pos1d",
    ):
        if key in td_item.keys():
            inst[key] = to_numpy(td_item[key])[node_slice]

    for key in (
        "edge_features",
        "W",
        "assembly_adj",
        "mat_var",
        "stack_size",
        "maint_diff",
        "rel_motion",
        "compat",
        "relation_valid",
        "relation_consistent",
    ):
        if key in td_item.keys():
            value = to_numpy(td_item[key])
            if value.ndim >= 2:
                inst[key] = value[node_slice, node_slice]
            else:
                inst[key] = clean_value(value)

    if "build_limit" in td_item.keys():
        inst["build_limit"] = to_numpy(td_item["build_limit"])

    return inst


def make_solver(name: str, args: argparse.Namespace, seed: int) -> Any:
    name = name.lower()
    if name == "cpccd":
        from baseline.cpccd_solver import CPCCDSolver

        return CPCCDSolver(alpha=args.cpccd_alpha)
    if name == "sa":
        from baseline.sa_solver import SASolver

        return SASolver(iterations=args.sa_iterations, seed=seed)
    if name == "ga":
        try:
            from baseline.ga_solver import GASolver
        except ImportError as exc:
            raise RuntimeError(
                "GA baseline requires deap. Install it with: pip install deap"
            ) from exc

        return GASolver(
            pop_size=args.ga_pop_size,
            generations=args.ga_generations,
            seed=seed,
        )
    raise ValueError(f"Unknown algorithm: {name}")


def save_solver_plot(name: str, solver: Any, idx: int, args: argparse.Namespace) -> str | None:
    if not args.plot_history:
        return None

    name = name.lower()
    if name == "ga" and hasattr(solver, "plot_fitness_history"):
        plot_dir = args.plot_dir / "ga_graph"
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / f"instance_{idx:04d}_ga_fitness_history.png"
        return solver.plot_fitness_history(str(save_path), show=False)

    if name == "sa" and hasattr(solver, "plot_history"):
        plot_dir = args.plot_dir / "sa_graph"
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / f"instance_{idx:04d}_sa_history.png"
        return solver.plot_history(str(save_path), show=False)

    return None


def make_env_from_dataset(dataset: Any, device: torch.device) -> PartConsolidationEnv:
    sample = dataset[0]
    generator_params = {
        "num_parts": int(scalar(sample["num_parts"])),
        "material_types": int(scalar(sample["material_type_count"]))
        if "material_type_count" in sample.keys()
        else 3,
    }
    return PartConsolidationEnv(generator_params=generator_params, device=str(device))


def infer_model_kind(state_dict: dict[str, Any], hparams: dict[str, Any]) -> str:
    if any(key.startswith("critic.") for key in state_dict):
        return "am_ppo"
    if "num_starts" in hparams and "num_augment" in hparams:
        return "pomo"
    return "am"


def load_nco_model(ckpt_path: Path, env: PartConsolidationEnv, device: torch.device):
    # Load checkpoints on CPU first. Lightning checkpoints may contain serialized
    # env objects, and their RNG state must remain a CPU ByteTensor during unpickle.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    hparams = ckpt.get("hyper_parameters", {})
    policy_kwargs = dict(hparams.get("policy_kwargs", {}))
    metrics = hparams.get("metrics", {})
    kind = infer_model_kind(state_dict, hparams)

    if kind == "am_ppo":
        critic_kwargs = {"embed_dim": policy_kwargs.get("embed_dim", 128)}
        model = AMPPO(
            env=env,
            policy_kwargs=policy_kwargs,
            critic_kwargs=critic_kwargs,
            metrics=metrics,
        )
    elif kind == "pomo":
        model = POMO(
            env=env,
            policy_kwargs=policy_kwargs,
            baseline=hparams.get("baseline", "shared"),
            num_augment=1,
            num_starts=hparams.get("num_starts", 0),
            metrics=metrics,
        )
    else:
        model = AttentionModel(
            env=env,
            policy_kwargs=policy_kwargs,
            baseline=hparams.get("baseline", "no"),
            metrics=metrics,
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"NCO checkpoint missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"NCO checkpoint unexpected keys: {len(incompatible.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model, kind


def clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def evaluate_algorithm(
    name: str,
    dataset: Any,
    limit: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx in range(limit):
        print(
            f"  Running {name}: instance {idx + 1}/{limit} "
            f"(dataset index {idx})...",
            flush=True,
        )
        seed = args.seed + idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        inst = strip_sep_instance(dataset[idx])
        solver = make_solver(name, args, seed)

        started = time.perf_counter()
        groups, solver_elapsed = solver.solve(inst)
        wall_elapsed = time.perf_counter() - started
        plot_path = save_solver_plot(name, solver, idx, args)

        metrics = score_metric_rows([evaluate_groups(groups, inst)])[0]
        score = metrics["score"]

        row = {
            "algorithm": name,
            "instance_idx": idx,
            "score": float(score),
            "reward": float(score),
            "solver_elapsed_sec": float(solver_elapsed),
            "wall_elapsed_sec": float(wall_elapsed),
            "groups": json.dumps(groups),
        }
        if plot_path is not None:
            row["plot_path"] = plot_path
        row.update({key: clean_value(value) for key, value in metrics.items()})
        rows.append(row)

    return rows


def node_groups_to_part_groups(groups: list[list[int]]) -> list[list[int]]:
    return [[int(node) - 1 for node in group if int(node) > 0] for group in groups]


def select_nco_actions(out: dict[str, torch.Tensor], num_starts: int) -> tuple[torch.Tensor, torch.Tensor]:
    actions = out["actions"]
    reward = out["reward"]
    if num_starts and num_starts > 1:
        reward_by_start = unbatchify(reward, num_starts)
        actions_by_start = unbatchify(actions, num_starts)
        best_idx = reward_by_start.argmax(dim=1)
        batch_idx = torch.arange(actions_by_start.size(0), device=actions_by_start.device)
        actions = actions_by_start[batch_idx, best_idx]
        reward = reward_by_start[batch_idx, best_idx]
    return actions.detach().cpu(), reward.detach().cpu()


def evaluate_nco(
    dataset: Any,
    limit: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        if not args.auto_ckpt:
            print("NCO checkpoint not provided; using latest checkpoint automatically.")
        ckpt_path = find_latest_checkpoint(args.ckpt_root)
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"NCO checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    env = make_env_from_dataset(dataset, device)
    model, model_kind = load_nco_model(ckpt_path, env, device)
    num_starts = int(getattr(model, "num_starts", 0) or 0)

    print(f"NCO checkpoint: {ckpt_path}")
    print(f"NCO model kind: {model_kind}")
    print(f"NCO device: {device}")

    rows: list[dict[str, Any]] = []
    for start in range(0, limit, args.nco_batch_size):
        end = min(start + args.nco_batch_size, limit)
        batch = dataset[start:end].to(device)
        td = env.reset(batch)

        started = time.perf_counter()
        with torch.no_grad():
            if num_starts > 1:
                out = model.policy(td.clone(), env, phase="test", num_starts=num_starts)
            else:
                out = model.policy(td.clone(), env, phase="test")
        wall_elapsed = time.perf_counter() - started

        actions, model_reward = select_nco_actions(out, num_starts)
        td_cpu = td.cpu()
        node_groups = env.actions_to_groups(actions, td_cpu)
        per_instance_elapsed = wall_elapsed / max(end - start, 1)

        for local_idx, groups_with_sep in enumerate(node_groups):
            idx = start + local_idx
            inst = strip_sep_instance(dataset[idx])
            groups = node_groups_to_part_groups(groups_with_sep)
            metrics = score_metric_rows([evaluate_groups(groups, inst)])[0]
            score = metrics["score"]
            row = {
                "algorithm": "nco",
                "instance_idx": idx,
                "score": float(score),
                "reward": float(score),
                "model_reward": float(model_reward[local_idx].item()),
                "solver_elapsed_sec": float(per_instance_elapsed),
                "wall_elapsed_sec": float(per_instance_elapsed),
                "groups": json.dumps(groups),
                "ckpt": str(ckpt_path),
                "model_kind": model_kind,
            }
            row.update({key: clean_value(value) for key, value in metrics.items()})
            rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    preferred = [
        "algorithm",
        "instance_idx",
        "score",
        "reward",
        "feasible",
        "num_groups",
        "Q_observed",
        "Q_expected",
        "solver_elapsed_sec",
        "wall_elapsed_sec",
        "groups",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = [key for key in preferred if key in rows[0]] + extra

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)

    print("\nSummary")
    print("-" * 80)
    for name, alg_rows in grouped.items():
        score = np.mean([row["score"] for row in alg_rows])
        feasible = np.mean([float(row.get("feasible", 0.0)) for row in alg_rows])
        num_groups = np.mean([float(row.get("num_groups", 0.0)) for row in alg_rows])
        elapsed = np.mean([row["wall_elapsed_sec"] for row in alg_rows])
        total_elapsed = np.sum([row["wall_elapsed_sec"] for row in alg_rows])
        print(
            f"{name:>6} | n={len(alg_rows):4d} | score={score: .6f} | "
            f"feasible={feasible: .3f} | groups={num_groups: .2f} | "
            f"time={elapsed: .4f}s/inst | total={total_elapsed: .4f}s"
        )


def main() -> None:
    args = parse_args()
    algorithms = [item.strip().lower() for item in args.algorithms.split(",") if item.strip()]
    if not algorithms:
        raise ValueError("At least one algorithm must be selected.")

    dataset = load_npz_to_tensordict(args.data)
    dataset_size = int(dataset.batch_size[0])
    limit = dataset_size if args.limit <= 0 else min(args.limit, dataset_size)

    print(f"Data: {args.data}")
    print(f"Instances: {limit}/{dataset_size}")
    print(f"Algorithms: {', '.join(algorithms)}")

    all_rows: list[dict[str, Any]] = []
    for name in algorithms:
        print(f"\nRunning {name}...")
        if name == "nco":
            rows = evaluate_nco(dataset, limit, args)
        else:
            rows = evaluate_algorithm(name, dataset, limit, args)
        all_rows.extend(rows)

    write_csv(args.output, all_rows)
    print_summary(all_rows)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
