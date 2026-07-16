from collections import defaultdict
from collections.abc import Mapping

import torch

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class TensorBoardLogger(Callback):
    """Log epoch-averaged metrics to TensorBoard with epoch as the x-axis."""

    def __init__(
        self,
        train_metrics: list[str] | tuple[str, ...] = ("reward", "loss", "entropy"),
        val_metrics: list[str] | tuple[str, ...] = ("reward", "loss"),
        test_metrics: list[str] | tuple[str, ...] = ("reward",),
        train_prefix: str = "train_epoch",
        val_prefix: str = "val_epoch",
        test_prefix: str = "test_epoch",
    ):
        super().__init__()
        self.train_metrics = tuple(train_metrics)
        self.val_metrics = tuple(val_metrics)
        self.test_metrics = tuple(test_metrics)
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.test_prefix = test_prefix
        self._buffers = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._buffers["train"].clear()

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._buffers["val"].clear()

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._buffers["test"].clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._collect("train", outputs, self.train_metrics)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not trainer.sanity_checking:
            self._collect("val", outputs, self.val_metrics)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self._collect("test", outputs, self.test_metrics)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._flush(trainer, "train", self.train_prefix, trainer.current_epoch)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._flush(trainer, "val", self.val_prefix, trainer.current_epoch)

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._flush(trainer, "test", self.test_prefix, trainer.current_epoch)

    def _collect(self, phase: str, outputs, metric_names: tuple[str, ...]) -> None:
        if not isinstance(outputs, Mapping):
            return

        for metric_name in metric_names:
            value = self._find_metric(outputs, phase, metric_name)
            if value is None:
                continue
            value = self._to_scalar_tensor(value)
            if value is not None:
                self._buffers[phase][metric_name].append(value)

    @staticmethod
    def _find_metric(outputs: Mapping, phase: str, metric_name: str):
        candidates = (
            f"{phase}/{metric_name}",
            metric_name,
            f"{phase}/{metric_name}_step",
            f"{metric_name}_step",
        )
        for key in candidates:
            if key in outputs:
                return outputs[key]
        return None

    @staticmethod
    def _to_scalar_tensor(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.detach().float().mean().cpu()
        try:
            return torch.tensor(float(value), dtype=torch.float32)
        except (TypeError, ValueError):
            return None

    def _flush(self, trainer, phase: str, prefix: str, step: int) -> None:
        if not self._buffers[phase]:
            return

        loggers = getattr(trainer, "loggers", None)
        if not loggers:
            logger = getattr(trainer, "logger", None)
            loggers = [logger] if logger is not None else []

        for metric_name, values in self._buffers[phase].items():
            if not values:
                continue
            value = torch.stack(values).mean().item()
            tag = f"{prefix}/{metric_name}"
            for logger in loggers:
                experiment = getattr(logger, "experiment", None)
                if hasattr(experiment, "add_scalar"):
                    experiment.add_scalar(tag, value, step)

        self._buffers[phase].clear()
