"""Tests for static binary node training orchestration."""

import pytest
import torch
from pytorch_lightning import Trainer
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from amlgraphx.evaluation import AveragePrecision, Precision
from amlgraphx.training import ModelContractError, StaticBinaryNodePredictor


def _data() -> Data:
    """Return one deterministic full-graph batch with three split masks."""
    return Data(
        x=torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        node_y=torch.tensor([0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, False, False]),
        validation_mask=torch.tensor([False, False, True, True]),
        test_mask=torch.tensor([False, False, True, True]),
    )


class _LinearNodeModel(nn.Module):
    """Minimal researcher model used to verify the model boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, batch: Data) -> torch.Tensor:
        return self.linear(batch.x)


class _RecordingLoss:
    """Record the labels passed to the loss while returning BCE loss."""

    def __init__(self) -> None:
        self.targets: torch.Tensor | None = None

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.targets = target.detach().clone()
        return nn.functional.binary_cross_entropy_with_logits(logits, target)


def test_training_step_uses_only_the_selected_node_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training loss receives labels selected by train_mask only."""
    loss = _RecordingLoss()
    predictor = StaticBinaryNodePredictor(
        _LinearNodeModel(), loss, metrics={}, learning_rate=0.01
    )
    monkeypatch.setattr(predictor, "log", lambda *args, **kwargs: None)

    predictor.training_step(_data(), 0)

    assert loss.targets is not None
    assert torch.equal(loss.targets, torch.tensor([0.0, 1.0]))


def test_metrics_receive_sigmoid_scores_not_raw_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metric thresholds operate on the documented sigmoid risk score."""

    class FixedModel(nn.Module):
        def forward(self, batch: Data) -> torch.Tensor:
            return torch.tensor([0.5, 0.5, 0.0, 0.0])

    predictor = StaticBinaryNodePredictor(
        FixedModel(),
        nn.BCEWithLogitsLoss(),
        metrics={"precision": Precision(threshold=0.6)},
    )
    monkeypatch.setattr(predictor, "log", lambda *args, **kwargs: None)

    predictor.training_step(_data(), 0)

    assert predictor.train_metrics["precision"].compute().item() == pytest.approx(0.5)


def test_wrong_model_output_shape_raises_contract_error() -> None:
    """A model must return one binary logit per graph node."""

    class BadModel(nn.Module):
        def forward(self, batch: Data) -> torch.Tensor:
            return torch.zeros((batch.num_nodes, 2))

    predictor = StaticBinaryNodePredictor(
        BadModel(), nn.BCEWithLogitsLoss(), metrics={}
    )

    with pytest.raises(ModelContractError, match="one|shape"):
        predictor.forward(_data())


def test_predict_returns_sigmoid_scores_for_all_nodes() -> None:
    """Prediction exposes probabilities while the model emits raw logits."""

    class FixedModel(nn.Module):
        def forward(self, batch: Data) -> torch.Tensor:
            return torch.tensor([-1.0, 0.0, 1.0, 2.0])

    predictor = StaticBinaryNodePredictor(
        FixedModel(), nn.BCEWithLogitsLoss(), metrics={}
    )

    scores = predictor.predict_step(_data(), 0)

    assert torch.allclose(scores, torch.sigmoid(torch.tensor([-1.0, 0.0, 1.0, 2.0])))


def test_predict_accepts_an_unlabelled_batch() -> None:
    """Inference does not require labels or split masks."""

    class FixedModel(nn.Module):
        def forward(self, batch: Data) -> torch.Tensor:
            return torch.zeros(batch.num_nodes)

    predictor = StaticBinaryNodePredictor(
        FixedModel(), nn.BCEWithLogitsLoss(), metrics={}
    )
    batch = Data(x=torch.ones((3, 1)))

    scores = predictor.predict_step(batch, 0)

    assert torch.equal(scores, torch.full((3,), 0.5))


def test_lightning_fit_test_and_predict_smoke() -> None:
    """The predictor works with ordinary Lightning lifecycle methods."""
    loader = DataLoader([_data()], batch_size=1)
    predictor = StaticBinaryNodePredictor(
        _LinearNodeModel(),
        nn.BCEWithLogitsLoss(),
        metrics={"precision": Precision(), "average_precision": AveragePrecision()},
        learning_rate=0.01,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
    )

    trainer.fit(predictor, train_dataloaders=loader, val_dataloaders=loader)
    result = trainer.test(predictor, dataloaders=loader, verbose=False)
    predictions = trainer.predict(predictor, dataloaders=loader)[0]

    assert result and "test_loss" in result[0]
    assert predictions.shape == (4,)
    assert torch.isfinite(predictions).all()
