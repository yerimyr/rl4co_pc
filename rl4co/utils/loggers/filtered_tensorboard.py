from __future__ import annotations

from typing import Any

from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class FilteredTensorBoardLogger(TensorBoardLogger):
    """TensorBoard logger that keeps only custom epoch-level metric groups.

    RL4CO still calls ``self.log`` for progress/checkpoint metrics and callbacks
    such as LR/time monitors may emit their own metrics. For PC experiments we
    keep TensorBoard focused on the custom callback outputs:
    ``train_epoch/*``, ``val_epoch/*``, and ``test_epoch/*``.
    """

    allowed_prefixes = ("train_epoch/", "val_epoch/", "test_epoch/", "overfitting_check")

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        filtered = {
            key: value
            for key, value in metrics.items()
            if any(str(key).startswith(prefix) for prefix in self.allowed_prefixes)
        }
        if filtered:
            super().log_metrics(filtered, step=step)

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None) -> None:
        # Avoid the automatic hp_metric card in TensorBoard.
        return None
