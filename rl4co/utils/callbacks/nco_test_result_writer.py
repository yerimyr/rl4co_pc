from __future__ import annotations

import csv
from pathlib import Path

import torch

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class NCOTestResultWriter(Callback):
    """Save per-instance NCO test rewards for boxplots and paired comparisons."""

    def __init__(
        self,
        output_dir: str | Path,
        filename: str = "nco_test_results.csv",
        algorithm: str | None = None,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.algorithm = algorithm
        self._rows: list[dict] = []

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._rows.clear()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not isinstance(outputs, dict) or "_test_rewards" not in outputs:
            return

        rewards = outputs["_test_rewards"]
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().float().reshape(-1).cpu().tolist()

        algorithm = self.algorithm or self._infer_algorithm(pl_module)
        batch_size = len(rewards)
        for local_idx, reward in enumerate(rewards):
            self._rows.append(
                {
                    "algorithm": algorithm,
                    "instance_idx": batch_idx * batch_size + local_idx,
                    "score": float(reward),
                    "reward": float(reward),
                    "batch_idx": int(batch_idx),
                    "local_idx": int(local_idx),
                }
            )

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if not self._rows:
            return

        path = self.output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "algorithm",
                    "instance_idx",
                    "score",
                    "reward",
                    "batch_idx",
                    "local_idx",
                ],
            )
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"Saved NCO test results: {path}")

    @staticmethod
    def _infer_algorithm(pl_module) -> str:
        cls_name = pl_module.__class__.__name__.lower()
        if cls_name == "attentionmodel":
            return "reinforce"
        if cls_name == "ampppo":
            return "ppo"
        if cls_name == "pomo":
            return "pomo"
        return cls_name
