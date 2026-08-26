"""Tests for temporal graph splitting and sparse snapshots."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
import torch

from amlgraphx.data import (
    TransactionGraphDataModule,
    sliding_snapshots,
    split_transaction_graph,
)
from amlgraphx.graph import build_transaction_graph


def _transactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "source": ["A", "B", "C", "D", "E", "F"],
            "target": ["B", "C", "D", "E", "F", "G"],
            "timestamp": [
                _date(1),
                _date(2),
                _date(3),
                _date(4),
                _date(5),
                _date(6),
            ],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "label": [0, 0, 1, 0, 1, 0],
        }
    )


def _date(day: int) -> datetime:
    return datetime(2025, 1, day, tzinfo=UTC)


def test_temporal_split_removes_cross_partition_edges() -> None:
    """Each temporal partition is an induced graph without future edges."""
    graph = build_transaction_graph(_transactions(), delta=timedelta(days=2))

    splits = split_transaction_graph(
        graph,
        train_end=_date(3),
        validation_end=_date(5),
    )

    assert splits.train.nodes["transaction_id"].to_list() == ["t1", "t2"]
    assert splits.validation.nodes["transaction_id"].to_list() == ["t3", "t4"]
    assert splits.test.nodes["transaction_id"].to_list() == ["t5", "t6"]
    assert splits.train.num_edges == 1
    assert splits.validation.num_edges == 1
    assert splits.test.num_edges == 1


def test_sliding_snapshots_use_local_sparse_edge_indices() -> None:
    """Snapshots store local COO-style edges instead of dense adjacency."""
    graph = build_transaction_graph(_transactions(), delta=timedelta(days=2))

    snapshots = list(
        sliding_snapshots(
            graph,
            window_size=timedelta(days=3),
            stride=timedelta(days=2),
            start_time=_date(1),
            end_time=_date(7),
        )
    )

    assert len(snapshots) == 2
    assert snapshots[0].index == 1
    assert snapshots[0].graph.nodes["transaction_id"].to_list() == ["t1", "t2", "t3"]
    assert snapshots[0].edge_index.dtype == torch.long
    assert snapshots[0].edge_index.shape == (2, 2)
    assert snapshots[0].edge_index.tolist() == [[0, 1], [1, 2]]
    assert snapshots[1].edge_index.shape[0] == 2


def test_data_module_builds_graph_before_partition_snapshots() -> None:
    """The data module orchestrates full graph, split, and window creation."""
    data_module = TransactionGraphDataModule(
        _transactions(),
        edge_delta=timedelta(days=2),
        train_end=_date(3),
        validation_end=_date(5),
        test_end=_date(7),
        window_size=timedelta(days=2),
        stride=timedelta(days=1),
    )

    with pytest.raises(RuntimeError, match="setup"):
        _ = data_module.splits

    data_module.setup()

    assert data_module.full_graph.num_nodes == 6
    assert len(list(data_module.train_snapshots())) == 1
    assert len(list(data_module.validation_snapshots())) == 1
    assert len(list(data_module.test_snapshots())) == 1


def test_data_module_rejects_invalid_time_configuration() -> None:
    """Unordered cutoffs and non-positive windows fail clearly."""
    with pytest.raises(ValueError, match="train_end"):
        TransactionGraphDataModule(
            _transactions(),
            edge_delta=timedelta(days=1),
            train_end=_date(5),
            validation_end=_date(4),
            window_size=timedelta(days=1),
            stride=timedelta(days=1),
        )

    graph = build_transaction_graph(_transactions(), delta=timedelta(days=1))
    with pytest.raises(ValueError, match="window_size"):
        list(
            sliding_snapshots(
                graph,
                window_size=timedelta(0),
                stride=timedelta(days=1),
            )
        )
