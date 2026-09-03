"""Tests for snapshot and continuous event-stream training orchestration."""

import pytest
import torch
from pytorch_lightning import Trainer
from torch import nn
from torch_geometric.data import Data, TemporalData
from torch_geometric.loader import DataLoader, TemporalDataLoader

from amlgraphx.training import (
    EventStreamBinaryPredictor,
    ModelContractError,
    SnapshotBinaryNodePredictor,
)


def _snapshot(index: int) -> Data:
    """Return one small snapshot with a target mask and sequence index."""
    return Data(
        x=torch.tensor([[0.0], [1.0], [2.0]]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        node_y=torch.tensor([0, 1, 0]),
        target_mask=torch.tensor([False, True, True]),
        snapshot_index=torch.tensor(index),
    )


class _NodeLinear(nn.Module):
    """Small node model used by the snapshot smoke test."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, batch: Data) -> torch.Tensor:
        return self.linear(batch.x)


def test_snapshot_predictor_masks_context_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot loss excludes historical context outside target_mask."""
    seen: list[int] = []

    class RecordingLoss:
        def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            seen.extend(target.to(dtype=torch.int64).tolist())
            return nn.functional.binary_cross_entropy_with_logits(logits, target)

    predictor = SnapshotBinaryNodePredictor(_NodeLinear(), RecordingLoss(), metrics={})
    monkeypatch.setattr(predictor, "log", lambda *args, **kwargs: None)
    predictor.on_train_epoch_start()
    predictor.training_step(_snapshot(1), 0)

    assert seen == [1, 0]


def test_snapshot_predictor_rejects_backward_sequence() -> None:
    """Snapshot order is validated instead of silently being rearranged."""
    predictor = SnapshotBinaryNodePredictor(
        _NodeLinear(), nn.BCEWithLogitsLoss(), metrics={}
    )
    predictor.on_train_epoch_start()
    predictor._check_snapshot_order(_snapshot(2), "train")

    with pytest.raises(ModelContractError, match="increase strictly"):
        predictor._check_snapshot_order(_snapshot(1), "train")


def test_snapshot_predictor_runs_with_lightning() -> None:
    """A snapshot sequence can use the standard Lightning trainer."""
    loader = DataLoader([_snapshot(1), _snapshot(2)], batch_size=1)
    predictor = SnapshotBinaryNodePredictor(
        _NodeLinear(), nn.BCEWithLogitsLoss(), metrics={}, learning_rate=0.01
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=2,
        limit_val_batches=2,
    )

    trainer.fit(predictor, train_dataloaders=loader, val_dataloaders=loader)

    assert "val_loss" in trainer.callback_metrics


def _events() -> TemporalData:
    """Return a chronological labelled event stream."""
    return TemporalData(
        src=torch.tensor([0, 1, 0]),
        dst=torch.tensor([1, 0, 1]),
        t=torch.tensor([1, 2, 3]),
        msg=torch.tensor([[0.0], [1.0], [2.0]]),
        y=torch.tensor([0, 1, 0]),
    )


class _EventModel(nn.Module):
    """Event model recording the predict-before-update ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.calls: list[str] = []

    def forward(self, batch: TemporalData) -> torch.Tensor:
        self.calls.append("forward")
        return self.linear(batch.msg)

    def update_state(self, batch: TemporalData) -> None:
        del batch
        self.calls.append("update")


class _MutableStateEventModel(nn.Module):
    """Event model whose state cannot be mutated before backward."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("state", torch.tensor([1.0]))

    def forward(self, batch: TemporalData) -> torch.Tensor:
        return self.weight * self.state.expand(batch.t.numel())

    def update_state(self, batch: TemporalData) -> None:
        del batch
        self.state.add_(1)


def test_event_predictor_updates_state_after_scoring() -> None:
    """The optional state hook runs after the current event scores exist."""
    model = _EventModel()
    predictor = EventStreamBinaryPredictor(model, nn.BCEWithLogitsLoss(), metrics={})

    scores = predictor.predict_step(_events(), 0)

    assert scores.shape == (3,)
    assert model.calls == ["forward", "update"]


def test_event_predictor_rejects_temporal_leakage() -> None:
    """Unsorted event timestamps fail before a model can consume them."""
    predictor = EventStreamBinaryPredictor(
        _EventModel(), nn.BCEWithLogitsLoss(), metrics={}
    )
    events = _events()
    events.t = torch.tensor([1, 3, 2])

    with pytest.raises(ModelContractError, match="non-decreasing"):
        predictor.training_step(events, 0)


def test_event_predictor_runs_with_temporal_dataloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TemporalDataLoader batches preserve the predictor event contract."""
    predictor = EventStreamBinaryPredictor(
        _EventModel(), nn.BCEWithLogitsLoss(), metrics={}, learning_rate=0.01
    )
    loader = TemporalDataLoader(_events(), batch_size=2)
    batch = next(iter(loader))
    monkeypatch.setattr(predictor, "log", lambda *args, **kwargs: None)

    assert predictor.training_step(batch, 0).ndim == 0


def test_event_predictor_defers_mutable_state_until_after_backward() -> None:
    """State used by forward is not mutated while autograd still needs it."""
    model = _MutableStateEventModel()
    predictor = EventStreamBinaryPredictor(
        model, nn.BCEWithLogitsLoss(), metrics={}, learning_rate=0.01
    )
    loader = TemporalDataLoader(_events(), batch_size=2)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(predictor, train_dataloaders=loader)

    assert model.state.item() == 3.0
