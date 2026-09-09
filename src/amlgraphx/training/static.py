"""Lightning orchestration for static binary node classification.

AMLGraphX owns the training protocol while researchers own the neural network.
The model receives one PyTorch Geometric-style batch and must return one binary
logit per node. This module deliberately does not implement a GNN architecture.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import Metric, MetricCollection


class ModelContractError(ValueError):
    """Raised when a batch or researcher model violates the static contract."""


OptimizerFactory = Callable[..., Optimizer]
SchedulerFactory = Callable[..., Any]


class StaticBinaryNodePredictor(LightningModule):
    """Train and evaluate a researcher-defined binary node classifier.

    The wrapped model is called as ``model(batch)`` and must return a tensor of
    shape ``[num_nodes]`` or ``[num_nodes, 1]`` containing raw logits. A batch
    must expose ``node_y`` (or the configured ``target_attr``) and boolean
    ``train_mask``, ``validation_mask`` and ``test_mask`` fields. The default
    names match :func:`amlgraphx.graph.to_pyg_data` and AMLGraphX temporal split
    conventions.

    Loss receives masked raw logits and floating-point binary targets. Metrics
    receive masked sigmoid scores and targets, and are accumulated over the
    complete split before ``compute`` is called. ``predict_step`` returns
    sigmoid risk scores for every node in the supplied batch.

    Args:
        model: Researcher-defined ``torch.nn.Module`` accepting one batch.
        loss: Callable accepting ``(logits, target)``.
        metrics: Optional named TorchMetrics instances. Each instance is copied
            for train, validation, and test, so split state cannot mix.
        optimizer: Optimizer class or factory. Defaults to Adam.
        learning_rate: Learning rate used when ``optimizer_kwargs`` does not
            specify ``lr``. Defaults to ``1e-3``.
        optimizer_kwargs: Additional keyword arguments for ``optimizer``.
        scheduler: Optional scheduler class or factory receiving the optimizer.
        scheduler_kwargs: Additional keyword arguments for ``scheduler``.
        scheduler_monitor: Metric name monitored by a scheduler such as
            ``ReduceLROnPlateau``. Set to ``None`` for schedulers without a
            monitored quantity.
        target_attr: Batch attribute containing binary node labels.
        train_mask_attr: Batch attribute selecting training nodes.
        validation_mask_attr: Batch attribute selecting validation nodes.
        test_mask_attr: Batch attribute selecting test nodes.

    Raises:
        ModelContractError: If model output, labels, or masks are malformed.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[Tensor, Tensor], Tensor],
        *,
        metrics: Mapping[str, Metric] | None = None,
        optimizer: OptimizerFactory = torch.optim.Adam,
        learning_rate: float = 1e-3,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        scheduler: SchedulerFactory | None = None,
        scheduler_kwargs: Mapping[str, Any] | None = None,
        scheduler_monitor: str | None = "val_loss",
        target_attr: str = "node_y",
        train_mask_attr: str = "train_mask",
        validation_mask_attr: str = "validation_mask",
        test_mask_attr: str = "test_mask",
    ) -> None:
        """Create a static binary node-classification predictor."""
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not callable(loss):
            raise TypeError("loss must be callable")
        if not callable(optimizer):
            raise TypeError("optimizer must be an optimizer class or factory")
        if scheduler is not None and not callable(scheduler):
            raise TypeError("scheduler must be a scheduler class or factory")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if metrics is not None:
            if not isinstance(metrics, Mapping):
                raise TypeError("metrics must be a mapping of names to Metric objects")
            if any(not isinstance(metric, Metric) for metric in metrics.values()):
                raise TypeError("metrics values must be torchmetrics.Metric objects")

        self.model = model
        self.loss_fn = loss
        self.optimizer_factory = optimizer
        self.learning_rate = learning_rate
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.scheduler_factory = scheduler
        self.scheduler_kwargs = dict(scheduler_kwargs or {})
        self.scheduler_monitor = scheduler_monitor
        self.target_attr = target_attr
        self.train_mask_attr = train_mask_attr
        self.validation_mask_attr = validation_mask_attr
        self.test_mask_attr = test_mask_attr

        metric_dict = dict(metrics or {})
        self.train_metrics = MetricCollection(deepcopy(metric_dict), prefix="train_")
        self.validation_metrics = MetricCollection(deepcopy(metric_dict), prefix="val_")
        self.test_metrics = MetricCollection(deepcopy(metric_dict), prefix="test_")

    def forward(self, batch: Any) -> Tensor:
        """Return validated raw logits for every node in ``batch``."""
        logits = self.model(batch)
        return _validate_logits(logits, _batch_num_nodes(batch))

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Optimize the loss over nodes selected by ``train_mask``."""
        del batch_idx
        return self._step(batch, self.train_mask_attr, self.train_metrics, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Evaluate validation loss and accumulate validation metrics."""
        del batch_idx
        return self._step(
            batch, self.validation_mask_attr, self.validation_metrics, "val"
        )

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Evaluate test loss and accumulate test metrics."""
        del batch_idx
        return self._step(batch, self.test_mask_attr, self.test_metrics, "test")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Return sigmoid risk scores for every node in ``batch``."""
        del batch_idx, dataloader_idx
        return torch.sigmoid(self.forward(batch))

    def on_train_epoch_end(self) -> None:
        """Log and reset metrics accumulated during the training epoch."""
        self._finish_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        """Log and reset metrics accumulated during validation."""
        self._finish_metrics(self.validation_metrics)

    def on_test_epoch_end(self) -> None:
        """Log and reset metrics accumulated during testing."""
        self._finish_metrics(self.test_metrics)

    def configure_optimizers(self) -> Any:
        """Construct the configured optimizer and optional epoch scheduler."""
        optimizer_kwargs = dict(self.optimizer_kwargs)
        optimizer_kwargs.setdefault("lr", self.learning_rate)
        optimizer = self.optimizer_factory(self.parameters(), **optimizer_kwargs)
        if self.scheduler_factory is None:
            return optimizer

        scheduler = self.scheduler_factory(optimizer, **self.scheduler_kwargs)
        scheduler_config: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": "epoch",
        }
        if self.scheduler_monitor is not None:
            scheduler_config["monitor"] = self.scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def _step(
        self,
        batch: Any,
        mask_attr: str,
        metrics: MetricCollection,
        stage: str,
    ) -> Tensor:
        """Run one masked loss/metric update for a training stage."""
        target = self._target(batch)
        logits = self.forward(batch)
        mask = _validate_mask(
            _get_batch_field(batch, mask_attr), target.numel(), mask_attr
        )
        masked_logits = logits[mask]
        masked_target = target[mask].to(dtype=logits.dtype)
        if masked_target.numel() == 0:
            raise ModelContractError(f"{mask_attr} must select at least one node")

        loss = self.loss_fn(masked_logits, masked_target)
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ModelContractError("loss must return a scalar torch.Tensor")
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if len(metrics) > 0:
            metrics.update(torch.sigmoid(masked_logits), target[mask])
        return loss

    def _finish_metrics(self, metrics: MetricCollection) -> None:
        """Log split-level metric values, then clear their accumulated state."""
        if len(metrics) == 0:
            return
        values = metrics.compute()
        self.log_dict(values, on_step=False, on_epoch=True)
        metrics.reset()

    def _target(self, batch: Any) -> Tensor:
        """Read and validate binary labels from one batch."""
        value = _get_batch_field(batch, self.target_attr)
        if not isinstance(value, Tensor):
            raise ModelContractError(f"{self.target_attr} must be a torch.Tensor")
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1 or value.numel() == 0:
            raise ModelContractError(f"{self.target_attr} must have shape [num_nodes]")
        if not torch.isin(value, torch.tensor((0, 1), device=value.device)).all():
            raise ModelContractError(
                f"{self.target_attr} must contain only binary labels 0 and 1"
            )
        return value.to(dtype=torch.long)


def _get_batch_field(batch: Any, name: str) -> Any:
    """Read a named field from a PyG data object or mapping."""
    if isinstance(batch, Mapping):
        if name not in batch:
            raise ModelContractError(f"batch is missing required field {name!r}")
        return batch[name]
    try:
        return getattr(batch, name)
    except AttributeError as exc:
        raise ModelContractError(f"batch is missing required field {name!r}") from exc


def _batch_num_nodes(batch: Any) -> int:
    """Infer node count without requiring labels, including at prediction time."""
    value = None
    if isinstance(batch, Mapping):
        value = batch.get("num_nodes")
    else:
        value = getattr(batch, "num_nodes", None)
    if isinstance(value, int) and value > 0:
        return value

    features = (
        batch.get("x") if isinstance(batch, Mapping) else getattr(batch, "x", None)
    )
    if isinstance(features, Tensor) and features.ndim >= 1 and features.shape[0] > 0:
        return int(features.shape[0])

    raise ModelContractError(
        "batch must expose a positive num_nodes value or a non-empty x tensor"
    )


def _validate_logits(logits: Any, node_count: int) -> Tensor:
    """Validate and normalize the model's one-logit-per-node output."""
    if not isinstance(logits, Tensor):
        raise ModelContractError("model must return a torch.Tensor of binary logits")
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1 or logits.numel() != node_count:
        raise ModelContractError(
            "model output must have shape [num_nodes] or [num_nodes, 1]"
        )
    if not logits.is_floating_point():
        raise ModelContractError("model output logits must use a floating-point dtype")
    return logits


def _validate_mask(value: Any, node_count: int, name: str) -> Tensor:
    """Validate one boolean node-selection mask."""
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ModelContractError(f"{name} must be a boolean torch.Tensor")
    if value.ndim != 1 or value.numel() != node_count:
        raise ModelContractError(f"{name} must have shape [num_nodes]")
    return value


__all__ = ["ModelContractError", "StaticBinaryNodePredictor"]
