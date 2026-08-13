from __future__ import annotations

import argparse

from Evaluation import effectiveness, efficiency, stability, scalability, generalization


EXPERIMENTS = {
    "effectiveness": effectiveness.run,
    "efficiency": efficiency.run,
    "stability": stability.run,
    "scalability": scalability.run,
    "generalization": generalization.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PC evaluation experiments sequentially.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="Experiments to run in order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.experiments:
        print("\n" + "=" * 100)
        print(f"Running evaluation experiment: {name}")
        print("=" * 100)
        EXPERIMENTS[name]()


if __name__ == "__main__":
    main()
