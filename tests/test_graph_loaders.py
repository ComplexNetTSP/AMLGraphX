"""Tests for PyG-ready AMLGraphX graph batching."""

from datetime import UTC, datetime, timedelta

import pytest
import torch
from torch_geometric.data import Batch, Data, TemporalData

from amlgraphx.data import (
    SnapshotDataLoader,
    SnapshotWindowDataset,
    StaticGraphWindowDataset,
    event_stream_loader,
    static_graph_loader,
)

_DAY_NS = 86_400_000_000_000


def _transaction_data() -> Data:
    """Return a small causal transaction graph with node-level targets."""
    data = Data(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.tensor([[1.0], [1.0]]),
        node_y=torch.tensor([0, 1, 0]),
        node_time=torch.tensor([0, _DAY_NS, 2 * _DAY_NS]),
        edge_time=torch.tensor([_DAY_NS, 2 * _DAY_NS]),
    )
    data.num_nodes = 3
    return data


def _account_data() -> Data:
    """Return a small account graph with transaction edge targets."""
    data = Data(
        x=torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_y=torch.tensor([0, 1, 0]),
        edge_time=torch.tensor([0, _DAY_NS, 2 * _DAY_NS]),
    )
    data.num_nodes = 4
    return data


def _snapshot(index: int) -> Data:
    """Return one account snapshot with metadata from the public graph API."""
    data = Data(
        x=torch.tensor([[float(index)], [float(index + 1)]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.tensor([[float(index)]]),
        edge_y=torch.tensor([index % 2]),
        edge_time=torch.tensor([index * _DAY_NS]),
    )
    data.num_nodes = 2
    data.snapshot_index = index
    data.start_time = datetime(2025, 1, index, tzinfo=UTC)
    data.end_time = datetime(2025, 1, index + 1, tzinfo=UTC)
    return data


def test_transaction_static_windows_keep_causal_context_and_pyg_batching() -> None:
    """Disjoint targets retain one-day predecessor context and batch in PyG."""
    dataset = StaticGraphWindowDataset(
        _transaction_data(),
        window_size=timedelta(days=1),
        lookback=timedelta(days=1),
    )

    assert len(dataset) == 3
    assert dataset[1].node_time.tolist() == [0, _DAY_NS]
    assert dataset[1].target_node_mask.tolist() == [False, True]
    assert dataset[1].edge_index.tolist() == [[0], [1]]

    batch = next(
        iter(
            static_graph_loader(
                _transaction_data(),
                window_size=timedelta(days=1),
                lookback=timedelta(days=1),
                batch_size=2,
            )
        )
    )
    assert isinstance(batch, Batch)
    assert batch.num_graphs == 2
    assert batch.target_node_mask.tolist() == [True, False, True]
    assert batch.window_id.tolist() == [0, 1]


def test_account_static_windows_compact_nodes_and_mask_target_edges() -> None:
    """Account windows retain selected transaction edges without dense padding."""
    dataset = StaticGraphWindowDataset(
        _account_data(),
        window_size=timedelta(days=1),
        lookback=timedelta(days=1),
    )

    assert len(dataset) == 3
    assert dataset[1].num_nodes == 3
    assert dataset[1].edge_index.tolist() == [[0, 1], [1, 2]]
    assert dataset[1].edge_y.tolist() == [0, 1]
    assert dataset[1].target_edge_mask.tolist() == [False, True]


def test_snapshot_loader_batches_each_time_position_with_pyg_batch() -> None:
    """Two temporal windows become two disconnected graphs at each time step."""
    dataset = SnapshotWindowDataset(
        [_snapshot(1), _snapshot(2), _snapshot(3), _snapshot(4)],
        context_size=2,
    )
    batch = next(iter(SnapshotDataLoader(dataset, batch_size=2)))

    assert len(dataset) == 2
    assert len(batch.context) == 2
    assert isinstance(batch.context[0], Batch)
    assert batch.context[0].num_graphs == 2
    assert batch.context[0].edge_index.tolist() == [[0, 2], [1, 3]]
    assert batch.target.edge_y.tolist() == [1, 0]
    assert "snapshot_index" not in batch.target
    assert "start_time" not in batch.target
    assert batch.to("cpu").target.edge_y.tolist() == [1, 0]


def test_snapshot_loader_rejects_reversed_metadata() -> None:
    """Snapshot ordering failures are rejected before a training epoch starts."""
    first = _snapshot(2)
    second = _snapshot(1)
    with pytest.raises(ValueError, match="increase"):
        SnapshotWindowDataset([first, second], context_size=1)


def test_event_stream_loader_uses_pyg_successive_event_batches() -> None:
    """The event wrapper delegates batching to PyG without reordering events."""
    events = TemporalData(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 3]),
        t=torch.tensor([1, 2, 3]),
        msg=torch.ones((3, 1)),
        y=torch.tensor([0, 1, 0]),
    )

    batches = list(event_stream_loader(events, batch_size=2))
    assert [batch.t.tolist() for batch in batches] == [[1, 2], [3]]

    events.t = torch.tensor([2, 1, 3])
    with pytest.raises(ValueError, match="non-decreasing"):
        event_stream_loader(events, batch_size=2)
