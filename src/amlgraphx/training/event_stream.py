"""Lightning orchestration for chronological link/event streams."""

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


class EventStreamBinaryPredictor(StaticBinaryNodePredictor):
    """Train a researcher-defined binary classifier on ordered transaction events.

    The wrapped model receives one PyG ``TemporalData``-style batch with
    ``src``, ``dst``, ``t`` and optional ``msg`` fields and returns one raw
    binary logit per event. Labels default to ``y``. Events are never silently
    sorted: timestamps must be non-decreasing within a batch and must not move
    backwards across batches, preventing accidental future-information access.

    If the model defines ``reset_state()``, it is called at each split or
    prediction sequence start. If it defines ``update_state(batch)``, the hook
    is called after logits and loss inputs are computed, implementing the
    predict-before-update convention used by stateful temporal models such as
    JODIE and TGN. Models without either hook remain ordinary stateless PyTorch
    modules.

    Args:
        model: Researcher-defined event model accepting one event batch.
        loss: Callable accepting masked ``(logits, target)`` tensors.
        metrics: Optional named TorchMetrics instances.
        event_mask_attr: Optional field selecting labelled events. Missing means
            every event in the supplied batch is a target.
        timestamp_attr: Event timestamp field, normally ``t``.
        reset_state: Whether to call an optional model ``reset_state()`` hook.
        **kwargs: Optimizer, scheduler, and target configuration passed to
            :class:`StaticBinaryNodePredictor`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[Tensor, Tensor], Tensor],
        *,
        metrics: Mapping[str, Metric] | None = None,
        event_mask_attr: str | None = None,
        timestamp_attr: str = "t",
        reset_state: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a chronological event-stream binary predictor."""
        super().__init__(
            model,
            loss,
            metrics=metrics,
            target_attr=kwargs.pop("target_attr", "y"),
            **kwargs,
        )
        self.event_mask_attr = event_mask_attr
        self.timestamp_attr = timestamp_attr
        self.reset_state = reset_state
        self._last_event_time: dict[str, Tensor | None] = {
            "train": None,
            "val": None,
            "test": None,
            "predict": None,
        }

    def forward(self, batch: Any) -> Tensor:
        """Return validated raw logits for every event in ``batch``."""
        logits = self.model(batch)
        return _validate_event_logits(logits, _num_events(batch))

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Train on one chronological event batch before updating model state."""
        del batch_idx
        return self._event_step(batch, self.train_metrics, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Validate on one chronological event batch."""
        del batch_idx
        return self._event_step(batch, self.validation_metrics, "val")

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Test on one chronological event batch."""
        del batch_idx
        return self._event_step(batch, self.test_metrics, "test")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Score one event batch, then apply its optional state update hook."""
        del batch_idx, dataloader_idx
        self._check_event_order(batch, "predict")
        scores = torch.sigmoid(self.forward(batch))
        self._update_state(batch)
        return scores

    def on_train_epoch_start(self) -> None:
        """Reset event order and optional model state before training."""
        self._start_sequence("train")

    def on_validation_epoch_start(self) -> None:
        """Reset event order and optional model state before validation."""
        self._start_sequence("val")

    def on_test_epoch_start(self) -> None:
        """Reset event order and optional model state before testing."""
        self._start_sequence("test")

    def on_predict_epoch_start(self) -> None:
        """Reset event order and optional model state before prediction."""
        self._start_sequence("predict")

    def _event_step(self, batch: Any, metrics: Any, stage: str) -> Tensor:
        """Validate order, score events, compute loss, and update state last."""
        self._check_event_order(batch, stage)
        target = self._target(batch)
        logits = self.forward(batch)
        mask = _event_mask(batch, self.event_mask_attr, target.numel(), target.device)
        masked_logits = logits[mask]
        masked_target = target[mask].to(dtype=logits.dtype)
        if masked_target.numel() == 0:
            raise ModelContractError("event mask must select at least one event")
        loss = self.loss_fn(masked_logits, masked_target)
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ModelContractError("loss must return a scalar torch.Tensor")
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if len(metrics) > 0:
            metrics.update(torch.sigmoid(masked_logits), target[mask])
        self._update_state(batch)
        return loss

    def _start_sequence(self, stage: str) -> None:
        """Reset timestamp tracking and an optional researcher model state."""
        self._last_event_time[stage] = None
        if self.reset_state:
            reset = getattr(self.model, "reset_state", None)
            if reset is not None:
                if not callable(reset):
                    raise ModelContractError("model.reset_state must be callable")
                reset()

    def _check_event_order(self, batch: Any, stage: str) -> None:
        """Reject a batch whose timestamps would expose future events."""
        value = getattr(batch, self.timestamp_attr, None)
        if value is None and isinstance(batch, Mapping):
            value = batch.get(self.timestamp_attr)
        if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() == 0:
            raise ModelContractError(
                f"{self.timestamp_attr} must be a non-empty one-dimensional tensor"
            )
        if not torch.isfinite(value).all():
            raise ModelContractError(
                f"{self.timestamp_attr} must contain finite values"
            )
        if value.numel() > 1 and torch.any(value[1:] < value[:-1]):
            raise ModelContractError(
                "event timestamps must be non-decreasing within each batch"
            )
        previous = self._last_event_time[stage]
        if previous is not None and value[0] < previous:
            raise ModelContractError(
                "event timestamps must be non-decreasing across batches"
            )
        self._last_event_time[stage] = value[-1].detach()

    def _update_state(self, batch: Any) -> None:
        """Call the optional post-prediction model state transition hook."""
        update = getattr(self.model, "update_state", None)
        if update is not None:
            if not callable(update):
                raise ModelContractError("model.update_state must be callable")
            update(batch)


def _num_events(batch: Any) -> int:
    """Infer event count from a TemporalData-style timestamp or source field."""
    value = getattr(batch, "t", None)
    if value is None and isinstance(batch, Mapping):
        value = batch.get("t")
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() == 0:
        value = getattr(batch, "src", None)
        if value is None and isinstance(batch, Mapping):
            value = batch.get("src")
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ModelContractError(
            "event batch must expose a non-empty one-dimensional t or src tensor"
        )
    return int(value.numel())


def _validate_event_logits(logits: Any, event_count: int) -> Tensor:
    """Validate and normalize one raw binary logit per event."""
    if not isinstance(logits, Tensor):
        raise ModelContractError("model must return a torch.Tensor of event logits")
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1 or logits.numel() != event_count:
        raise ModelContractError(
            "model output must have shape [num_events] or [num_events, 1]"
        )
    if not logits.is_floating_point():
        raise ModelContractError("model output logits must use a floating-point dtype")
    return logits


def _event_mask(
    batch: Any, name: str | None, event_count: int, device: torch.device
) -> Tensor:
    """Return an optional event mask, defaulting to every event."""
    if name is None:
        return torch.ones(event_count, dtype=torch.bool, device=device)
    value = getattr(batch, name, None)
    if value is None and isinstance(batch, Mapping):
        value = batch.get(name)
    if value is None:
        return torch.ones(event_count, dtype=torch.bool, device=device)
    return _validate_mask(value, event_count, name)


__all__ = ["EventStreamBinaryPredictor"]
