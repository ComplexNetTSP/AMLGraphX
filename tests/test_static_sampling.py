"""Tests for strictly causal time-aware static graph sampling."""

from datetime import timedelta

import torch
from torch import nn
from torch_geometric.data import Data

from amlgraphx.experiments import BinaryRiskTask, Experiment
from amlgraphx.sampling import causal_static_edge_loader, causal_static_node_loader
from amlgraphx.training import StaticBinaryEdgePredictor

_DAY_NS = 86_400_000_000_000


def _node_data() -> Data:
    """Return a transaction graph with past, equal-time, and future neighbors."""
    data = Data(
        x=torch.arange(6, dtype=torch.float32).view(-1, 1),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [4, 4, 4, 4, 5]]),
        node_time=torch.tensor([0, 1, 2, 3, 2, 3]),
        edge_time=torch.tensor([2, 2, 2, 2, 3]),
        node_y=torch.tensor([0, 1, 0, 1, 0, 1]),
        train_mask=torch.tensor([False, False, False, False, True, False]),
    )
    data.num_nodes = 6
    return data


def _edge_data() -> Data:
    """Return an account graph with a selected edge at a repeated timestamp."""
    data = Data(
        x=torch.arange(6, dtype=torch.float32).view(-1, 1),
        edge_index=torch.tensor([[0, 1, 2, 3], [5, 5, 5, 5]]),
        edge_time=torch.tensor([0, 1, 2, 2]),
        edge_y=torch.tensor([0, 1, 0, 1]),
        train_edge_mask=torch.tensor([False, False, False, True]),
    )
    data.num_nodes = 6
    return data


def test_full_graph_node_sampling_excludes_equal_and_future_context() -> None:
    """A target sees only strictly earlier transaction nodes from a full graph."""
    batch = next(
        iter(
            causal_static_node_loader(
                _node_data(),
                target_mask_attr="train_mask",
                num_neighbors=[-1],
                batch_size=1,
            )
        )
    )

    target = batch.target_node_mask
    assert batch.n_id[target].tolist() == [4]
    assert set(batch.n_id[~target].tolist()) == {0, 1}
    assert torch.all(batch.node_time[~target] < batch.node_time[target][0])


def test_window_sampling_applies_a_target_specific_time_cutoff() -> None:
    """A later target in one window never leaks into an earlier target batch."""
    data = Data(
        x=torch.arange(4, dtype=torch.float32).view(-1, 1),
        edge_index=torch.tensor([[0, 1, 2], [2, 2, 3]]),
        edge_time=torch.tensor([2 * _DAY_NS, 2 * _DAY_NS, 3 * _DAY_NS]),
        node_time=torch.tensor([0, _DAY_NS, 2 * _DAY_NS, 3 * _DAY_NS]),
        node_y=torch.tensor([0, 1, 0, 1]),
    )
    data.num_nodes = 4

    loader = causal_static_node_loader(
        data,
        num_neighbors=[-1],
        batch_size=1,
        window_size=timedelta(days=2),
        lookback=timedelta(days=2),
    )
    target_two_days = next(batch for batch in loader if batch.n_id[0].item() == 2)

    assert target_two_days.target_node_mask.tolist() == [True, False, False]
    assert 3 not in target_two_days.n_id.tolist()
    assert torch.all(target_two_days.node_time[1:] < 2 * _DAY_NS)


def test_window_sampling_skips_windows_without_the_requested_split() -> None:
    """A validation split may begin after several otherwise valid target windows."""
    data = Data(
        x=torch.arange(4, dtype=torch.float32).view(-1, 1),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_time=torch.tensor([_DAY_NS, 2 * _DAY_NS, 3 * _DAY_NS]),
        node_time=torch.tensor([0, _DAY_NS, 2 * _DAY_NS, 3 * _DAY_NS]),
        node_y=torch.tensor([0, 1, 0, 1]),
        validation_mask=torch.tensor([False, False, True, True]),
    )
    data.num_nodes = 4

    loader = causal_static_node_loader(
        data,
        target_mask_attr="validation_mask",
        num_neighbors=[-1],
        batch_size=1,
        window_size=timedelta(days=1),
        lookback=timedelta(days=1),
    )

    assert [batch.n_id[0].item() for batch in loader] == [2, 3]


def test_edge_sampling_excludes_the_target_and_exposes_link_labels() -> None:
    """Account-edge batches retain labels outside their strict-past context graph."""
    batch = next(
        iter(
            causal_static_edge_loader(
                _edge_data(),
                target_mask_attr="train_edge_mask",
                num_neighbors=[-1],
                batch_size=1,
            )
        )
    )

    assert batch.edge_label.tolist() == [1]
    assert batch.target_edge_mask.tolist() == [True]
    assert batch.target_edge_time.tolist() == [2]
    assert batch.edge_time.tolist() == [0, 1]
    assert 3 not in batch.e_id.tolist()


def test_sampled_node_batches_run_through_experiment_metrics() -> None:
    """Experiment collects exactly one frozen risk score per sampled node target."""
    data = Data(
        x=torch.arange(6, dtype=torch.float32).view(-1, 1),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        edge_time=torch.tensor([1, 2, 3, 4, 5]),
        node_time=torch.arange(6),
        node_y=torch.tensor([0, 1, 0, 1, 0, 1]),
        train_mask=torch.tensor([True, True, False, False, False, False]),
        validation_mask=torch.tensor([False, False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
    )
    data.num_nodes = 6

    class NodeModel(nn.Module):
        """Return one trainable logit per sampled transaction node."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, batch: Data) -> torch.Tensor:
            return self.linear(batch.x).squeeze(-1)

    def loader(mask: str):
        return causal_static_node_loader(
            data,
            target_mask_attr=mask,
            num_neighbors=[-1],
            batch_size=2,
        )

    experiment = Experiment(
        NodeModel(),
        task=BinaryRiskTask("static", "node", target_mask_attr="target_node_mask"),
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
        train_dataloaders=loader("train_mask"),
        validation_dataloaders=loader("validation_mask"),
        test_dataloaders=loader("test_mask"),
    )

    assert result.predictions.labels.tolist() == [0, 1]
    assert result.predictions.scores.shape == (2,)
    assert result.risk_metrics is not None


def test_edge_predictor_uses_pyg_link_target_axis() -> None:
    """The edge predictor validates logits against ``edge_label``, not context edges."""

    class EdgeModel(nn.Module):
        """Score each sampled target edge from its remapped source endpoint."""

        def forward(self, batch: Data) -> torch.Tensor:
            return batch.x[batch.edge_label_index[0], 0]

    batch = next(
        iter(
            causal_static_edge_loader(
                _edge_data(),
                target_mask_attr="train_edge_mask",
                num_neighbors=[-1],
                batch_size=1,
            )
        )
    )
    predictor = StaticBinaryEdgePredictor(EdgeModel(), nn.BCEWithLogitsLoss())

    assert predictor.forward(batch).shape == (1,)
