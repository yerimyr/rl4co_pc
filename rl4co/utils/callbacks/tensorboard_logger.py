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
        train_groups: dict[str, list[str]] | None = None,
        val_groups: dict[str, list[str]] | None = None,
        test_groups: dict[str, list[str]] | None = None,
        train_prefix: str = "train_epoch",
        val_prefix: str = "val_epoch",
        test_prefix: str = "test_epoch",
        train_decode_check: bool = False,
        train_decode_check_tag: str = "train_decode_check/reward",
    ):
        super().__init__()
        self.train_metrics = tuple(train_metrics)
        self.val_metrics = tuple(val_metrics)
        self.test_metrics = tuple(test_metrics)
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.test_prefix = test_prefix
        self.train_decode_check = train_decode_check
        self.train_decode_check_tag = train_decode_check_tag
        self._groups = {
            "train": train_groups or {},
            "val": val_groups or {},
            "test": test_groups or {},
        }
        self._buffers = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }
        self._last_train_epoch_values: dict[str, float] = {}
        self._last_val_epoch_values: dict[str, float] = {}
        self._train_decode_check_batch = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._buffers["train"].clear()
        self._train_decode_check_batch = None

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._buffers["val"].clear()

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._buffers["test"].clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._collect("train", outputs, self.train_metrics)
        if self.train_decode_check and self._train_decode_check_batch is None:
            self._train_decode_check_batch = self._copy_batch_to_cpu(batch)

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
        train_values = self._flush(trainer, "train", self.train_prefix, trainer.current_epoch)
        self._log_overfitting_train(trainer, train_values, trainer.current_epoch)
        if self.train_decode_check:
            self._log_train_decode_check(trainer, pl_module, trainer.current_epoch)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            val_values = self._flush(trainer, "val", self.val_prefix, trainer.current_epoch)
            self._last_val_epoch_values = val_values
            self._log_overfitting_validation(trainer, val_values, trainer.current_epoch)

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

    def _flush(self, trainer, phase: str, prefix: str, step: int) -> dict[str, float]:
        if not self._buffers[phase]:
            return {}

        loggers = getattr(trainer, "loggers", None)
        if not loggers:
            logger = getattr(trainer, "logger", None)
            loggers = [logger] if logger is not None else []

        flushed_values = {}
        for metric_name, values in self._buffers[phase].items():
            if not values:
                continue
            value = torch.stack(values).mean().item()
            flushed_values[metric_name] = value
            tag = f"{prefix}/{metric_name}"
            for logger in loggers:
                experiment = getattr(logger, "experiment", None)
                if hasattr(experiment, "add_scalar"):
                    experiment.add_scalar(tag, value, step)

        for group_name, metric_names in self._groups[phase].items():
            scalars = {}
            for metric_name in metric_names:
                values = self._buffers[phase].get(metric_name)
                if values:
                    scalars[metric_name] = torch.stack(values).mean().item()
            if not scalars:
                continue
            tag = f"{prefix}/{group_name}"
            for logger in loggers:
                experiment = getattr(logger, "experiment", None)
                if hasattr(experiment, "add_scalars"):
                    experiment.add_scalars(tag, scalars, step)

        self._buffers[phase].clear()
        if phase == "train":
            self._last_train_epoch_values = flushed_values
        return flushed_values

    def _log_overfitting_train(self, trainer, train_values: dict[str, float], step: int) -> None:
        if "reward" not in train_values:
            return
        self._log_overfitting_scalars(
            trainer,
            {"train_reward": train_values["reward"]},
            step,
        )

    def _log_overfitting_validation(self, trainer, val_values: dict[str, float], step: int) -> None:
        if "reward" not in val_values:
            return
        self._log_overfitting_scalars(
            trainer,
            {"validation_reward": val_values["reward"]},
            step,
        )

    def _log_overfitting_scalars(self, trainer, scalars: dict[str, float], step: int) -> None:
        loggers = getattr(trainer, "loggers", None)
        if not loggers:
            logger = getattr(trainer, "logger", None)
            loggers = [logger] if logger is not None else []

        for logger in loggers:
            experiment = getattr(logger, "experiment", None)
            if hasattr(experiment, "add_scalars"):
                experiment.add_scalars("overfitting_check/reward", scalars, step)

    @staticmethod
    def _copy_batch_to_cpu(batch):
        if hasattr(batch, "clone"):
            batch = batch.clone()
        if hasattr(batch, "detach"):
            batch = batch.detach()
        if hasattr(batch, "cpu"):
            batch = batch.cpu()
        return batch

    @staticmethod
    def _move_batch_to_device(batch, device):
        if hasattr(batch, "to"):
            return batch.to(device)
        return batch

    def _log_train_decode_check(self, trainer, pl_module, step: int) -> None:
        if self._train_decode_check_batch is None:
            return
        if not hasattr(pl_module, "policy") or not hasattr(pl_module, "env"):
            return

        batch = self._move_batch_to_device(
            self._copy_batch_to_cpu(self._train_decode_check_batch),
            pl_module.device,
        )

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            sampling_td = pl_module.env.reset(batch.clone() if hasattr(batch, "clone") else batch)
            sampling_out = pl_module.policy(
                sampling_td,
                pl_module.env,
                phase="train",
                select_best=False,
            )
            greedy_td = pl_module.env.reset(batch.clone() if hasattr(batch, "clone") else batch)
            greedy_out = pl_module.policy(
                greedy_td,
                pl_module.env,
                phase="val",
                select_best=True,
            )
        if was_training:
            pl_module.train()

        if "reward" not in sampling_out or "reward" not in greedy_out:
            return

        scalars = {
            "sampling_reward": self._to_scalar_tensor(sampling_out["reward"]).item(),
            "greedy_reward": self._to_scalar_tensor(greedy_out["reward"]).item(),
        }

        loggers = getattr(trainer, "loggers", None)
        if not loggers:
            logger = getattr(trainer, "logger", None)
            loggers = [logger] if logger is not None else []

        for logger in loggers:
            experiment = getattr(logger, "experiment", None)
            if hasattr(experiment, "add_scalars"):
                experiment.add_scalars(self.train_decode_check_tag, scalars, step)
