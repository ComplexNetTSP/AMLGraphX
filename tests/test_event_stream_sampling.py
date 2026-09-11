"""Tests for strict-causal stateful and stateless event-stream sampling."""

import pytest
import torch
from torch import nn
from torch_geometric.data import Data, TemporalData

from amlgraphx.experiments import BinaryRiskTask, Experiment
from amlgraphx.sampling import (
    causal_event_neighbor_loader,
    causal_event_stream_loader,
    recent_event_neighbors,
)
from amlgraphx.training import EventStreamBinaryPredictor


def _events() -> TemporalData:
    """Return a small ordered account event stream with one equal-time pair."""
    return TemporalData(
        src=torch.tensor([0, 1, 0, 2, 1, 3]),
        dst=torch.tensor([1, 0, 2, 0, 3, 1]),
        t=torch.tensor([1, 2, 3, 3, 4, 5]),
        msg=torch.arange(6, dtype=torch.float32).reshape(-1, 1),
        y=torch.tensor([0, 1, 0, 1, 0, 1]),
    )


def test_stateful_loader_groups_timestamps_and_marks_warmup() -> None:
    """History is update-only and equal-time targets remain one atomic batch."""
    events = _events()
    history = events.index_select(torch.tensor([0, 1]))
    targets = events.index_select(torch.tensor([2, 3, 4]))

    batches = list(causal_event_stream_loader(targets, history=history))

    assert [batch.t.tolist() for batch in batches] == [[1], [2], [3, 3], [4]]
    assert [batch.target_event_mask.tolist() for batch in batches] == [
        [False],
        [False],
        [True, True],
        [True],
    ]


def test_stateful_loader_rejects_equal_time_split_boundary() -> None:
    """A target cannot use a same-timestamp event from a preceding split."""
    events = _events()
    history = events.index_select(torch.tensor([0, 1, 2]))
    targets = events.index_select(torch.tensor([3, 4]))

    with pytest.raises(ValueError, match="strictly before"):
        causal_event_stream_loader(targets, history=history)


class _StatefulRecorder(nn.Module):
    """Record whether warm-up batches are updated without forward scoring."""

    def __init__(self) -> None:
        """Create a differentiable scorer and a compact call trace."""
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.calls: list[str] = []

    def forward(self, batch: TemporalData) -> torch.Tensor:
        """Score target event messages when a target batch is supplied."""
        self.calls.append("forward")
        return self.linear(batch.msg).squeeze(-1)

    def update_state(self, batch: TemporalData) -> None:
        """Record each timestamp group after its prediction opportunity."""
        self.calls.append(f"update:{batch.t[0].item()}")


def test_event_predictor_updates_history_without_scoring_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm-up state transitions happen before the first validation score."""
    events = _events()
    loader = causal_event_stream_loader(
        events.index_select(torch.tensor([2, 3])),
        history=events.index_select(torch.tensor([0, 1])),
    )
    model = _StatefulRecorder()
    predictor = EventStreamBinaryPredictor(
        model,
        nn.BCEWithLogitsLoss(),
        metrics={},
        event_mask_attr="target_event_mask",
    )
    monkeypatch.setattr(predictor, "log", lambda *args, **kwargs: None)
    predictor.on_validation_epoch_start()

    losses = [
        predictor.validation_step(batch, index) for index, batch in enumerate(loader)
    ]

    assert losses[:2] == [None, None]
    assert losses[2] is not None
    assert model.calls == ["update:1", "update:2", "forward", "update:3"]


def test_stateless_loader_excludes_target_equal_time_and_future_context() -> None:
    """Every sampled message-passing edge is strictly older than its target."""
    events = _events()
    events.test_mask = torch.tensor([False, False, False, False, True, False])

    batch = next(
        iter(
            causal_event_neighbor_loader(
                events,
                num_neighbors=[-1],
                batch_size=1,
                target_mask_attr="test_mask",
            )
        )
    )

    assert batch.event_id.tolist() == [4]
    assert batch.event_y.tolist() == [0]
    assert batch.event_time.tolist() == [4]
    assert batch.event_msg.tolist() == [[4.0]]
    assert batch.target_event_mask.tolist() == [True]
    assert torch.all(events.t[batch.e_id] < batch.event_time[0])


def test_recent_event_neighbors_is_native_pyg_history_index() -> None:
    """The stateful helper returns a resettable PyG recent-neighbor component."""
    history = recent_event_neighbors(_events(), size=2)
    history.insert(torch.tensor([0]), torch.tensor([1]))

    node_id, edge_index, event_id = history(torch.tensor([0]))

    assert node_id.tolist() == [0, 1]
    assert edge_index.tolist() == [[1], [0]]
    assert event_id.tolist() == [0]


class _StatelessEventModel(nn.Module):
    """Use target transaction features from one sampled local event graph."""

    def __init__(self) -> None:
        """Create the smallest learnable event-target scorer."""
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, batch: Data) -> torch.Tensor:
        """Return one raw logit for every sampled target event feature row."""
        return self.linear(batch.event_msg).squeeze(-1)


def test_stateless_sampled_events_run_through_experiment() -> None:
    """Sampled target labels, logits, and held-out risk scores remain aligned."""
    events = _events()
    events.train_mask = torch.tensor([True, True, True, False, False, False])
    events.validation_mask = torch.tensor([False, False, False, True, False, False])
    events.test_mask = torch.tensor([False, False, False, False, True, True])
    experiment = Experiment(
        _StatelessEventModel(),
        task=BinaryRiskTask(
            "event_stream",
            "event",
            label_attr="event_y",
            target_mask_attr="target_event_mask",
        ),
        predictor_kwargs={"timestamp_attr": "event_time"},
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "enable_progress_bar": False,
        },
    )
    options = {"num_neighbors": [2], "batch_size": 1}

    result = experiment.run(
        train_dataloaders=causal_event_neighbor_loader(
            events, target_mask_attr="train_mask", **options
        ),
        validation_dataloaders=causal_event_neighbor_loader(
            events, target_mask_attr="validation_mask", **options
        ),
        test_dataloaders=causal_event_neighbor_loader(
            events, target_mask_attr="test_mask", **options
        ),
    )

    assert result.predictions.labels.tolist() == [0, 1]
    assert result.predictions.scores.numel() == 2
