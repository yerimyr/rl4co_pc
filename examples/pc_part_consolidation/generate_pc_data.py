from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rl4co.data.utils import save_tensordict_to_npz
from rl4co.envs.pc.generator import FPIGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed PC validation/test datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pc"))
    parser.add_argument("--num-parts", type=int, default=20)
    parser.add_argument("--max-num-parts", type=int, default=None)
    parser.add_argument("--material-types", type=int, default=3)
    parser.add_argument("--topology-mode", type=str, default="mixed")
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--val-seed", type=int, default=4321)
    parser.add_argument("--test-seed", type=int, default=1234)
    parser.add_argument("--compress", action="store_true")
    return parser.parse_args()


def generate_and_save(generator: FPIGenerator, size: int, seed: int, path: Path, compress: bool):
    torch.manual_seed(seed)
    td = generator(size)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tensordict_to_npz(td, str(path), compress=compress)
    print(f"saved {size} instances to {path}")


def main():
    args = parse_args()
    generator = FPIGenerator(
        num_parts=args.num_parts,
        max_num_parts=args.max_num_parts,
        material_types=args.material_types,
        topology_mode=args.topology_mode,
    )

    prefix = f"pc{args.num_parts}"
    val_path = args.data_dir / f"{prefix}_val_seed{args.val_seed}.npz"
    test_path = args.data_dir / f"{prefix}_test_seed{args.test_seed}.npz"

    generate_and_save(generator, args.val_size, args.val_seed, val_path, args.compress)
    generate_and_save(generator, args.test_size, args.test_seed, test_path, args.compress)


if __name__ == "__main__":
    main()

