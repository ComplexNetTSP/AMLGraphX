"""Behavioral tests for the lightweight experiment lifecycle."""

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data, TemporalData
from torch_geometric.loader import DataLoader

from amlgraphx.data import SnapshotBatch, event_stream_loader
from amlgraphx.experiments import BinaryRiskTask, Experiment, TabularExperiment
from amlgraphx.training import (
    ModelContractError,
    SnapshotBinaryEdgePredictor,
    StaticBinaryEdgePredictor,
)


def _node_data() -> Data:
    """Return a small labelled transaction-node graph for every lifecycle stage."""
    return Data(
        x=torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        node_y=torch.tensor([0, 1, 0, 1]),
        target_node_mask=torch.tensor([True, True, True, True]),
    )


class _NodeModel(nn.Module):
    """Return one transaction-node logit from the selected scalar feature."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, batch: Data) -> torch.Tensor:
        return self.linear(batch.x)


def test_experiment_runs_fit_predict_and_binary_risk_metrics() -> None:
    """Experiment validates a model then returns aligned held-out risk scores."""
    loader = DataLoader([_node_data()], batch_size=1)
    experiment = Experiment(
        _NodeModel(),
        task=BinaryRiskTask(
            representation="static",
            target_kind="node",
            target_mask_attr="target_node_mask",
        ),
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
        },
        evaluation_kwargs={"top_k": (1,)},
    )

    result = experiment.run(
        train_dataloaders=loader,
        validation_dataloaders=loader,
        test_dataloaders=loader,
    )

    assert result.predictions.labels.tolist() == [0, 1, 0, 1]
    assert result.predictions.scores.shape == (4,)
    assert result.risk_metrics is not None
    assert result.risk_metrics.sample_count == 4


def test_experiment_rejects_model_with_wrong_target_axis() -> None:
    """The pre-fit dummy check fails before Lightning starts optimisation."""

    class BadModel(nn.Module):
        def forward(self, batch: Data) -> torch.Tensor:
            return torch.zeros(batch.num_nodes + 1)

    loader = DataLoader([_node_data()], batch_size=1)
    experiment = Experiment(
        BadModel(),
        task=BinaryRiskTask(representation="static", target_kind="node"),
        trainer_kwargs={"logger": False, "enable_progress_bar": False},
    )

    with pytest.raises(ModelContractError, match="input/output contract"):
        experiment.fit(loader)


def test_edge_predictors_accept_account_edge_labels_and_snapshot_targets() -> None:
    """Account graph edges and SnapshotBatch targets use one-logit-per-edge rules."""
    graph = Data(
        x=torch.ones((3, 1)),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.tensor([[1.0], [2.0]]),
        edge_y=torch.tensor([0, 1]),
        target_edge_mask=torch.tensor([True, True]),
    )

    class EdgeModel(nn.Module):
        def forward(self, batch: Data | SnapshotBatch) -> torch.Tensor:
            target = batch.target if isinstance(batch, SnapshotBatch) else batch
            return target.edge_attr[:, 0]

    edge_predictor = StaticBinaryEdgePredictor(EdgeModel(), nn.BCEWithLogitsLoss())
    assert edge_predictor.forward(graph).shape == (2,)

    snapshot = SnapshotBatch(
        context=(Batch.from_data_list([graph]),), target=Batch.from_data_list([graph])
    )
    snapshot_predictor = SnapshotBinaryEdgePredictor(
        EdgeModel(), nn.BCEWithLogitsLoss()
    )
    assert snapshot_predictor.forward(snapshot).shape == (2,)


def test_event_experiment_keeps_one_score_per_temporal_event() -> None:
    """TemporalData prediction collection remains aligned across event batches."""

    class EventModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, batch: TemporalData) -> torch.Tensor:
            return self.linear(batch.msg).squeeze(-1)

    events = TemporalData(
        src=torch.tensor([0, 1, 0, 1, 0]),
        dst=torch.tensor([1, 0, 1, 0, 1]),
        t=torch.tensor([1, 2, 3, 4, 5]),
        msg=torch.ones((5, 1)),
        y=torch.tensor([0, 1, 0, 1, 0]),
    )
    loader = event_stream_loader(events, batch_size=2)
    experiment = Experiment(
        EventModel(),
        task=BinaryRiskTask("event_stream", "event"),
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
        },
    )

    result = experiment.run(
        train_dataloaders=loader,
        validation_dataloaders=loader,
        test_dataloaders=loader,
    )

    assert result.predictions.labels.numel() == 5
    assert result.predictions.scores.numel() == 5


def test_tabular_experiment_fits_only_train_data_and_returns_risk_scores() -> None:
    """A sklearn-style estimator shares the test risk-score output boundary."""

    class Estimator:
        def __init__(self) -> None:
            self.seen: tuple[torch.Tensor, torch.Tensor] | None = None

        def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
            self.seen = (features, labels)

        def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
            positive = features[:, 0].clamp(0, 1)
            return torch.stack((1 - positive, positive), dim=1)

    train_x = torch.tensor([[0.0], [1.0]])
    train_y = torch.tensor([0, 1])
    test_x = torch.tensor([[0.1], [0.9]])
    test_y = torch.tensor([0, 1])
    estimator = Estimator()
    experiment = TabularExperiment(
        estimator, metrics={"mean_score": lambda _, score: score.mean()}
    )

    result = experiment.run(train_x, train_y, test_x, test_y)

    assert estimator.seen is not None
    assert torch.equal(estimator.seen[0], train_x)
    assert torch.equal(estimator.seen[1], train_y)
    assert result.predictions.target_kind == "row"
    assert result.predictions.scores.tolist() == pytest.approx([0.1, 0.9])
    assert result.metrics["mean_score"] == pytest.approx(0.5)
