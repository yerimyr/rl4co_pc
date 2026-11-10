"""Generate fixed-size PC test datasets for one or more problem sizes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from Evaluation.common import DEFAULT_GENERATOR_PARAMS, load_or_generate_dataset


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument(
        "--num-parts",
        default="30,40,50",
        help="Comma-separated fixed sizes, for example 20,30,40.",
    )
    sizes.add_argument(
        "--num-parts-ranges",
        default=None,
        help="Comma-separated inclusive ranges, for example 10-20,21-30,31-40.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--material-types", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.material_types < 1:
        raise ValueError("--material-types must be at least 1")

    material_suffix = "" if args.material_types == 2 else f"_material{args.material_types}"

    if args.num_parts_ranges:
        cases: list[tuple[int, int]] = []
        for token in args.num_parts_ranges.split(","):
            low, high = (int(part) for part in token.strip().split("-", 1))
            if low < 1 or low > high:
                raise ValueError(f"Invalid part-count range: {token}")
            cases.append((low, high))
    else:
        cases = [(int(value.strip()), int(value.strip())) for value in args.num_parts.split(",")]

    for low, high in cases:
        is_range = low != high
        size_label = f"{low}-{high}" if is_range else str(low)
        split_label = f"{args.split}{args.size}" if is_range else args.split
        path = (
            ROOT
            / "data"
            / "pc"
            / f"pc{size_label}_newdist{material_suffix}_{split_label}_seed{args.seed}.npz"
        )
        generator_params = dict(DEFAULT_GENERATOR_PARAMS)
        generator_params["num_parts"] = low
        generator_params["max_num_parts"] = high if is_range else None
        generator_params["material_types"] = args.material_types
        dataset = load_or_generate_dataset(
            path=path,
            size=args.size,
            seed=args.seed,
            generator_params=generator_params,
            overwrite=args.overwrite,
        )
        if len(dataset) != args.size:
            raise ValueError(f"{path} contains {len(dataset)} instances, expected {args.size}")
        print(f"Saved/verified {len(dataset)} instances: {path}")
        if is_range:
            sampled_sizes = [int(value) for value in dataset["num_parts"].cpu().tolist()]
            print(f"N counts: {dict(sorted(Counter(sampled_sizes).items()))}")
        material_counts = [
            int(value) for value in dataset["material_type_count"].cpu().tolist()
        ]
        print(
            "Material-type-count distribution: "
            f"{dict(sorted(Counter(material_counts).items()))}"
        )


if __name__ == "__main__":
    main()
