"""Lightning orchestration for snapshot-sequence node classification."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torchmetrics import Metric

from .static import (
    ModelContractError,
    StaticBinaryNodePredictor,
    _validate_mask,
)


class SnapshotBinaryNodePredictor(StaticBinaryNodePredictor):
    """Train a researcher-defined node model over chronologically ordered snapshots.

    Each batch is one snapshot-like object accepted by the researcher's model.
    The model returns one raw binary logit per node. Labels are read from
    ``node_y`` by default, and ``target_mask`` selects only nodes belonging to
    the current prediction window; when the mask is absent, every node is a
    target. This matches :class:`amlgraphx.data.GraphSnapshot` semantics after
    conversion to a PyG ``Data`` object.

    Snapshot batches may expose a scalar ``snapshot_index`` (or a configured
    field) for order validation. If the researcher model defines
    ``reset_state()``, the hook is called at the start of every train,
    validation, test, and prediction sequence. The predictor does not impose a
    neural architecture or maintain hidden graph state itself.

    Args:
        model: Researcher-defined ``torch.nn.Module`` accepting one snapshot.
        loss: Callable accepting masked ``(logits, target)`` tensors.
        metrics: Optional named TorchMetrics instances.
        target_mask_attr: Optional snapshot field selecting prediction targets.
        snapshot_index_attr: Field used to verify chronological order. Set to
            ``None`` to disable index checking for a loader that guarantees
            ordering externally.
        reset_state: Whether to call an optional model ``reset_state()`` hook.
        **kwargs: Optimizer, scheduler, metric, and target configuration passed
            to :class:`StaticBinaryNodePredictor`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[Tensor, Tensor], Tensor],
        *,
        metrics: Mapping[str, Metric] | None = None,
        target_mask_attr: str = "target_mask",
        snapshot_index_attr: str | None = "snapshot_index",
        reset_state: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a snapshot-sequence binary node predictor."""
        super().__init__(model, loss, metrics=metrics, **kwargs)
        self.target_mask_attr = target_mask_attr
        self.snapshot_index_attr = snapshot_index_attr
        self.reset_state = reset_state
        self._last_snapshot_index: dict[str, int | None] = {
            "train": None,
            "val": None,
            "test": None,
            "predict": None,
        }

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Train on one ordered snapshot and its target nodes."""
        del batch_idx
        self._check_snapshot_order(batch, "train")
        return self._snapshot_step(batch, self.train_metrics, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Validate on one ordered snapshot and its target nodes."""
        del batch_idx
        self._check_snapshot_order(batch, "val")
        return self._snapshot_step(batch, self.validation_metrics, "val")

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Test on one ordered snapshot and its target nodes."""
        del batch_idx
        self._check_snapshot_order(batch, "test")
        return self._snapshot_step(batch, self.test_metrics, "test")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Return sigmoid scores for every node in one ordered snapshot."""
        del batch_idx, dataloader_idx
        self._check_snapshot_order(batch, "predict")
        return torch.sigmoid(self.forward(batch))

    def on_train_epoch_start(self) -> None:
        """Reset sequence order and optional model state before training."""
        self._start_sequence("train")

    def on_validation_epoch_start(self) -> None:
        """Reset sequence order and optional model state before validation."""
        self._start_sequence("val")

    def on_test_epoch_start(self) -> None:
        """Reset sequence order and optional model state before testing."""
        self._start_sequence("test")

    def on_predict_epoch_start(self) -> None:
        """Reset sequence order and optional model state before prediction."""
        self._start_sequence("predict")

    def _snapshot_step(self, batch: Any, metrics: Any, stage: str) -> Tensor:
        """Apply target masking and update one snapshot metric collection."""
        target = self._target(batch)
        logits = self.forward(batch)
        mask = _snapshot_mask(
            batch, self.target_mask_attr, target.numel(), target.device
        )
        masked_logits = logits[mask]
        masked_target = target[mask].to(dtype=logits.dtype)
        if masked_target.numel() == 0:
            raise ModelContractError(
                f"{self.target_mask_attr} must select at least one node"
            )
        loss = self.loss_fn(masked_logits, masked_target)
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ModelContractError("loss must return a scalar torch.Tensor")
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if len(metrics) > 0:
            metrics.update(torch.sigmoid(masked_logits), target[mask])
        return loss

    def _start_sequence(self, stage: str) -> None:
        """Reset order tracking and an optional researcher model state."""
        self._last_snapshot_index[stage] = None
        if self.reset_state:
            reset = getattr(self.model, "reset_state", None)
            if reset is not None:
                if not callable(reset):
                    raise ModelContractError("model.reset_state must be callable")
                reset()

    def _check_snapshot_order(self, batch: Any, stage: str) -> None:
        """Reject snapshots that move backwards in their declared sequence."""
        if self.snapshot_index_attr is None:
            return
        value = getattr(batch, self.snapshot_index_attr, None)
        if value is None and isinstance(batch, Mapping):
            value = batch.get(self.snapshot_index_attr)
        if value is None:
            return
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ModelContractError(
                    f"{self.snapshot_index_attr} must be a scalar integer"
                )
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, int):
            raise ModelContractError(
                f"{self.snapshot_index_attr} must be a scalar integer"
            )
        previous = self._last_snapshot_index[stage]
        if previous is not None and value <= previous:
            raise ModelContractError(
                f"snapshot indices must increase strictly; received {value} after {previous}"
            )
        self._last_snapshot_index[stage] = value


def _snapshot_mask(
    batch: Any, name: str, node_count: int, device: torch.device
) -> Tensor:
    """Return an optional target mask, defaulting to every snapshot node."""
    value = getattr(batch, name, None)
    if value is None and isinstance(batch, Mapping):
        value = batch.get(name)
    if value is None:
        return torch.ones(node_count, dtype=torch.bool, device=device)
    return _validate_mask(value, node_count, name)


__all__ = ["SnapshotBinaryNodePredictor"]
