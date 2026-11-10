"""Create CPCCD-vs-NCO gamma bar plots across problem sizes and test seeds."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GAMMA_ROOT = ROOT / "outputs" / "evaluation" / "gamma_sweep"
METHODS = ("cpccd", "nco-custom")
LABELS = {"cpccd": "CPCCD", "nco-custom": "NCO"}
COLORS = {"cpccd": "#F28E2B", "nco-custom": "#59A14F"}


def result_dir(num_parts: int, seed: int) -> Path:
    suffix = "paper(ortools_5m)" if seed == 1234 else "paper(cpccd_nco)"
    return GAMMA_ROOT / f"n{num_parts}_seed{seed}_test100_{suffix}"


def shared_limits(frames: list[pd.DataFrame]) -> tuple[float, float]:
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby(["num_parts", "test_seed", "gamma", "method"])["score"].agg(["mean", "std"])
    low = float((summary["mean"] - summary["std"]).min())
    high = float((summary["mean"] + summary["std"]).max())
    span = max(high - low, 50.0)
    padding = span * 0.05
    return 50.0 * math.floor((low - padding) / 50.0), 50.0 * math.ceil((high + padding) / 50.0)


def save_plot(df: pd.DataFrame, num_parts: int, seed: int, output: Path, y_limits: tuple[float, float]) -> None:
    gammas = sorted(df["gamma"].astype(float).unique())
    summary = df.groupby(["gamma", "method"])["score"].agg(["mean", "std"])
    x = np.arange(len(gammas), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_idx, method in enumerate(METHODS):
        means = np.array([summary.loc[(gamma, method), "mean"] for gamma in gammas])
        stds = np.array([summary.loc[(gamma, method), "std"] for gamma in gammas])
        offset = (-0.5 if method_idx == 0 else 0.5) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=5,
            color=COLORS[method],
            alpha=0.9,
            label=LABELS[method],
            error_kw={"elinewidth": 1.4, "capthick": 1.4},
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{gamma:g}" for gamma in gammas])
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Score")
    ax.set_ylim(*y_limits)
    ax.set_title(f"CPCCD vs NCO mean score (error bars: ±1 SD) | N={num_parts}, test seed={seed}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=GAMMA_ROOT / "seed_plot")
    args = parser.parse_args()

    cases: list[tuple[int, int, pd.DataFrame]] = []
    for num_parts in (20, 30, 40, 50):
        for seed in (1234, 2026):
            path = result_dir(num_parts, seed) / "raw_results.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            df = df[df["method"].isin(METHODS)].copy()
            df["num_parts"] = num_parts
            df["test_seed"] = seed
            cases.append((num_parts, seed, df))

    y_limits = shared_limits([df for _, _, df in cases])
    print(f"Shared y-axis: {y_limits}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for num_parts, seed, df in cases:
        output = args.output_dir / f"cpccd_nco_N{num_parts}_test_seed{seed}.png"
        save_plot(df, num_parts, seed, output, y_limits)


if __name__ == "__main__":
    main()
