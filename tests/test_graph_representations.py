"""Tests for account temporal representations and standard PyG conversion."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
import torch
from torch_geometric.data import Batch, Data, TemporalData

from amlgraphx.graph import (
    AccountEventStream,
    AccountGraph,
    TransactionGraph,
    prepare_graph,
    to_pyg_data,
    to_pyg_temporal_data,
)


def _time(day: int) -> datetime:
    return datetime(2025, 1, day, tzinfo=UTC)


def _account_transactions() -> pl.DataFrame:
    # Deliberately unordered to verify that only event streams reorder rows.
    return pl.DataFrame(
        {
            "transaction_id": ["t2", "t1", "t3"],
            "source": ["A", "A", "B"],
            "target": ["B", "B", "C"],
            "timestamp": [_time(2), _time(1), _time(3)],
            "amount": [20.0, 10.0, 30.0],
            "channel": ["wire", "cash", "wire"],
            "label": [0, 1, 0],
        }
    )


def _account_metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "node_id": ["A", "B", "C"],
            "balance": [100.0, 200.0, 300.0],
            "country": ["FR", "DE", "GB"],
        }
    )


def test_account_static_graph_keeps_edge_features_and_typed_time() -> None:
    """Time-aware account graphs keep one complete row per transaction edge."""
    graph = prepare_graph(
        _account_transactions(),
        node_type="account",
        temporal="static",
        account_metadata=_account_metadata(),
    )

    assert isinstance(graph, AccountGraph)
    assert graph.edges["transaction_id"].to_list() == ["t2", "t1", "t3"]
    assert graph.edges["amount"].to_list() == [20.0, 10.0, 30.0]
    assert graph.edges["channel"].to_list() == ["wire", "cash", "wire"]
    assert graph.edges.schema["timestamp"].base_type() == pl.Datetime
    assert graph.nodes["balance"].to_list() == [100.0, 200.0, 300.0]


def test_account_snapshots_preserve_parallel_edges_and_edge_features() -> None:
    """Account snapshots select edge windows without transaction aggregation."""
    snapshots = list(
        prepare_graph(
            _account_transactions(),
            node_type="account",
            temporal="snapshot",
            account_metadata=_account_metadata(),
            bin_size=timedelta(days=2),
            stride=timedelta(days=2),
            start_time=_time(1),
            end_time=_time(4),
            drop_last=False,
        )
    )

    assert len(snapshots) == 2
    assert snapshots[0].graph.edges["amount"].to_list() == [20.0, 10.0]
    assert snapshots[0].graph.edges["channel"].to_list() == ["wire", "cash"]
    assert snapshots[0].edge_index.tolist() == [[0, 0], [1, 1]]
    assert snapshots[1].graph.edges["transaction_id"].to_list() == ["t3"]


def test_account_event_stream_is_ordered_and_keeps_messages() -> None:
    """Each account transaction becomes one chronologically ordered event."""
    stream = prepare_graph(
        _account_transactions(),
        node_type="account",
        temporal="event_stream",
        account_metadata=_account_metadata(),
    )

    assert isinstance(stream, AccountEventStream)
    assert stream.events["transaction_id"].to_list() == ["t1", "t2", "t3"]
    assert stream.events["amount"].to_list() == [10.0, 20.0, 30.0]
    assert stream.events["channel"].to_list() == ["cash", "wire", "wire"]

    with pytest.raises(NotImplementedError, match="node-arrival"):
        prepare_graph(
            _account_transactions(),
            node_type="transaction",
            temporal="event_stream",
            edge_delta=timedelta(days=1),
        )


def test_both_node_types_convert_to_standard_pyg_data() -> None:
    """Account and transaction semantics share one standard PyG Data output."""
    account_graph = prepare_graph(
        _account_transactions(),
        node_type="account",
        temporal="static",
        account_metadata=_account_metadata(),
    )
    account_data = to_pyg_data(
        account_graph,
        node_feature_columns=["balance"],
        edge_feature_columns=["amount"],
        edge_label_column="label",
    )

    assert type(account_data) is Data
    assert account_data.x.shape == (3, 1)
    assert account_data.edge_index.tolist() == [[0, 0, 1], [1, 1, 2]]
    assert account_data.edge_attr[:, 0].tolist() == [20.0, 10.0, 30.0]
    assert account_data.edge_time.shape == (3,)
    assert account_data.is_edge_attr("edge_time")
    assert account_data.edge_y.tolist() == [0, 1, 0]

    transactions = pl.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "source": ["A", "B", "C"],
            "target": ["B", "C", "D"],
            "timestamp": [_time(1), _time(2), _time(3)],
            "amount": [10.0, 20.0, 30.0],
            "label": [1, 0, 0],
        }
    )
    transaction_graph = prepare_graph(
        transactions,
        node_type="transaction",
        temporal="static",
        edge_delta=timedelta(days=1),
    )
    transaction_data = to_pyg_data(
        transaction_graph,
        node_feature_columns=["amount"],
        edge_feature_columns=["time_delta"],
        node_label_column="label",
    )

    assert isinstance(transaction_graph, TransactionGraph)
    assert type(transaction_data) is Data
    assert transaction_data.edge_index.tolist() == [[0, 1], [1, 2]]
    assert transaction_data.edge_attr[:, 0].tolist() == [86_400.0, 86_400.0]
    assert transaction_data.edge_time.tolist() == [
        int(_time(2).timestamp() * 1_000_000_000),
        int(_time(3).timestamp() * 1_000_000_000),
    ]
    assert transaction_data.node_y.tolist() == [1, 0, 0]

    # PyG batches custom edge_time exactly like other edge-level tensors; no
    # AMLGraphX Data subclass is needed.
    batch = Batch.from_data_list([account_data, account_data])
    assert batch.edge_index.shape == (2, 6)
    assert batch.edge_time.shape == (6,)


def test_account_stream_converts_to_standard_temporal_data() -> None:
    """The event representation maps directly to PyG src/dst/t/msg fields."""
    stream = prepare_graph(
        _account_transactions(),
        node_type="account",
        temporal="event_stream",
        account_metadata=_account_metadata(),
    )
    events = to_pyg_temporal_data(
        stream,
        message_columns=["amount"],
        label_column="label",
    )

    assert type(events) is TemporalData
    assert events.src.tolist() == [0, 0, 1]
    assert events.dst.tolist() == [1, 1, 2]
    assert events.msg[:, 0].tolist() == [10.0, 20.0, 30.0]
    assert events.y.tolist() == [1, 0, 0]
    assert torch.all(events.t[:-1] <= events.t[1:])


def test_pyg_conversion_rejects_unencoded_categorical_features() -> None:
    """Categorical encoding remains an explicit research preprocessing step."""
    graph = prepare_graph(
        _account_transactions(),
        node_type="account",
        temporal="static",
    )

    with pytest.raises(TypeError, match="must be numerical"):
        to_pyg_data(graph, edge_feature_columns=["channel"])
