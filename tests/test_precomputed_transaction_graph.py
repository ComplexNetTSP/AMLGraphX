"""Tests for importing transaction-node datasets and ordinal time."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from amlgraphx.data import logical_timestamp_from_step
from amlgraphx.graph import build_precomputed_transaction_graph


def test_logical_timestamp_from_step_preserves_order_and_raw_step() -> None:
    """Ordinal time uses the declared duration without dropping step values."""
    result = logical_timestamp_from_step(
        pl.LazyFrame({"step": [0, 3]}),
        step_column="step",
        step_size=timedelta(hours=2),
    ).collect()

    assert result["step"].to_list() == [0, 3]
    assert result["timestamp"].to_list() == [
        datetime(1970, 1, 1, tzinfo=UTC),
        datetime(1970, 1, 1, 6, tzinfo=UTC),
    ]


def test_precomputed_transaction_graph_preserves_supplied_edges() -> None:
    """A supplied relation does not require artificial account endpoints."""
    graph = build_precomputed_transaction_graph(
        pl.DataFrame({"tx_id": ["a", "b"], "step": [1, 2], "amount": [1.0, 2.0]}),
        pl.DataFrame({"from": ["a"], "to": ["b"]}),
        node_id_column="tx_id",
        edge_source_column="from",
        edge_target_column="to",
        time_column="step",
        step_size=timedelta(days=1),
    )

    assert graph.nodes["transaction_id"].to_list() == ["a", "b"]
    assert graph.edges.select(
        "source_transaction_id", "target_transaction_id"
    ).to_dicts() == [{"source_transaction_id": "a", "target_transaction_id": "b"}]


def test_precomputed_transaction_graph_rejects_unknown_edge_endpoints() -> None:
    """Every imported edge must point to a supplied transaction node."""
    with pytest.raises(ValueError, match="unknown transaction IDs"):
        build_precomputed_transaction_graph(
            pl.DataFrame({"tx": ["a"]}),
            pl.DataFrame({"from": ["a"], "to": ["missing"]}),
            node_id_column="tx",
            edge_source_column="from",
            edge_target_column="to",
        )
