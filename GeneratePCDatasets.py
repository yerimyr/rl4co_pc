from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl4co.data.utils import save_tensordict_to_npz
from rl4co.envs.pc.env import PartConsolidationEnv


DEFAULT_PARAMS: dict[str, Any] = {
    "max_num_parts": None,
    "material_types": None,
    "topology_mode": "mixed",
    "p_maint_H_low": 0.10,
    "p_maint_H_high": 0.50,
    "p_standard_low": 0.10,
    "p_standard_high": 0.50,
    "p_relative_motion_low": 0.10,
    "p_relative_motion_high": 0.50,
}


SHIFTED_PARAMS: dict[str, Any] = {
    **DEFAULT_PARAMS,
    "p_maint_H_low": 0.35,
    "p_maint_H_high": 0.80,
    "p_standard_low": 0.02,
    "p_standard_high": 0.20,
    "p_relative_motion_low": 0.35,
    "p_relative_motion_high": 0.80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed PC validation/test datasets for RL4CO training and evaluation."
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50],
        help="Problem sizes to generate.",
    )
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--val-seed", type=int, default=4321)
    parser.add_argument("--test-seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pc"))
    parser.add_argument(
        "--distribution",
        choices=["default", "shifted", "both"],
        default="default",
        help="Which instance distribution to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even if they already exist.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use npz compression. Smaller files, slower save/load.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_file(
    *,
    path: Path,
    num_parts: int,
    size: int,
    seed: int,
    params: dict[str, Any],
    overwrite: bool,
    compress: bool,
) -> None:
    if path.exists() and not overwrite:
        print(f"Skip existing: {path}")
        return

    generator_params = dict(params)
    generator_params["num_parts"] = num_parts
    generator_params["max_num_parts"] = None

    set_seed(seed)
    env = PartConsolidationEnv(generator_params=generator_params)
    td = env.generator(size)

    path.parent.mkdir(parents=True, exist_ok=True)
    save_tensordict_to_npz(td, path, compress=compress)
    print(
        f"Saved {path} | size={size} | seed={seed} | "
        f"node_features={tuple(td['node_features'].shape)}"
    )


def distribution_items(distribution: str) -> list[tuple[str, dict[str, Any]]]:
    if distribution == "default":
        return [("newdist", DEFAULT_PARAMS)]
    if distribution == "shifted":
        return [("shifted", SHIFTED_PARAMS)]
    return [("newdist", DEFAULT_PARAMS), ("shifted", SHIFTED_PARAMS)]


def main() -> None:
    args = parse_args()

    for num_parts in args.num_parts:
        for suffix, params in distribution_items(args.distribution):
            generate_file(
                path=args.output_dir / f"pc{num_parts}_{suffix}_val_seed{args.val_seed}.npz",
                num_parts=num_parts,
                size=args.val_size,
                seed=args.val_seed,
                params=params,
                overwrite=args.overwrite,
                compress=args.compress,
            )
            generate_file(
                path=args.output_dir / f"pc{num_parts}_{suffix}_test_seed{args.test_seed}.npz",
                num_parts=num_parts,
                size=args.test_size,
                seed=args.test_seed,
                params=params,
                overwrite=args.overwrite,
                compress=args.compress,
            )


if __name__ == "__main__":
    main()
