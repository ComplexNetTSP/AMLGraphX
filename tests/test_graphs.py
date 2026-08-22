"""Tests for AMLGraphX account and transaction graph views."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from amlgraphx.datasets import clean_lazy_frame
from amlgraphx.graphs import (
    AccountGraph,
    TransactionGraph,
    build_account_graph,
    build_transaction_graph,
)


def _transaction_frame(
    rows: list[tuple[str, str, str, str, float, int]],
) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=[
            "transaction_id",
            "source",
            "target",
            "timestamp",
            "amount",
            "label",
        ],
        orient="row",
    ).with_columns(pl.col("timestamp").str.to_datetime())


def test_account_graph_preserves_nodes_edges_and_attributes() -> None:
    """Account graphs preserve accounts, repeated edges, and attributes."""
    frame = _transaction_frame(
        [
            ("t1", "A", "B", "2025-01-01 09:00", 10.0, 0),
            ("t2", "A", "B", "2025-01-01 09:05", 20.0, 1),
            ("t3", "B", "C", "2025-01-01 09:10", 30.0, 0),
        ]
    )

    graph = build_account_graph(frame)

    assert isinstance(graph, AccountGraph)
    assert graph.num_nodes == 3
    assert graph.num_edges == 3
    assert graph.nodes["node_id"].to_list() == ["A", "B", "C"]
    assert graph.edges.select("transaction_id").to_series().to_list() == [
        "t1",
        "t2",
        "t3",
    ]
    assert graph.edges["amount"].to_list() == [10.0, 20.0, 30.0]
    assert graph.edges["label"].to_list() == [0, 1, 0]


def test_account_graph_joins_account_metadata() -> None:
    """Account metadata is preserved on matching account nodes."""
    transactions = pl.DataFrame({"from": ["A"], "to": ["B"]})
    accounts = pl.DataFrame(
        {
            "Account Number": ["A", "B"],
            "Bank ID": [1, 2],
            "Entity ID": ["e1", "e2"],
        }
    )

    graph = build_account_graph(transactions, account_metadata=accounts)

    assert graph.nodes["Bank ID"].to_list() == [1, 2]
    assert graph.nodes["Entity ID"].to_list() == ["e1", "e2"]


def test_account_graph_strips_account_metadata_ids() -> None:
    """Whitespace around metadata account IDs does not prevent a join."""
    transactions = pl.DataFrame({"from": ["A"], "to": ["B"]})
    accounts = pl.DataFrame({"Account Number": [" A ", " B "], "Bank ID": [1, 2]})

    graph = build_account_graph(transactions, account_metadata=accounts)

    assert graph.nodes["Bank ID"].to_list() == [1, 2]


def test_account_graph_accepts_lazy_frames() -> None:
    """Account graph construction accepts a lazy Polars frame."""
    frame = pl.LazyFrame({"Account": ["A"], "Account.1": ["B"]})

    graph = AccountGraph.from_transactions(frame)

    assert graph.edges.select(["source", "target"]).to_dicts() == [
        {"source": "A", "target": "B"}
    ]


def test_account_graph_requires_endpoints() -> None:
    """Missing endpoint columns produce a clear validation error."""
    with pytest.raises(ValueError, match="source account"):
        build_account_graph(pl.DataFrame({"amount": [1.0]}))


def test_graphs_generate_deterministic_transaction_ids() -> None:
    """Missing transaction IDs are generated from stable input row order."""
    frame = pl.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "timestamp": [
                datetime(2025, 1, 1, 9, tzinfo=UTC),
                datetime(2025, 1, 1, 10, tzinfo=UTC),
            ],
        }
    )

    first = build_account_graph(frame)
    second = build_account_graph(frame)

    assert first.edges["transaction_id"].to_list() == ["tx_0", "tx_1"]
    assert first.edges["transaction_id"].to_list() == second.edges[
        "transaction_id"
    ].to_list()


def test_transaction_graph_repairs_null_and_duplicate_ids() -> None:
    """Invalid source IDs become deterministic unique transaction IDs."""
    frame = _transaction_frame(
        [
            ("duplicate", "A", "B", "2025-01-01 09:00", 1.0, 0),
            ("duplicate", "B", "C", "2025-01-01 09:10", 2.0, 0),
            (None, "C", "D", "2025-01-01 09:20", 3.0, 0),
        ]
    )

    graph = build_transaction_graph(frame, delta=timedelta(hours=1))
    node_ids = graph.nodes["transaction_id"].to_list()

    assert node_ids == ["tx_0", "tx_1", "tx_2"]
    assert len(node_ids) == len(set(node_ids))
    assert all(value is not None for value in node_ids)
    assert graph.edges.select(
        ["source_transaction_id", "target_transaction_id"]
    ).to_dicts() == [
        {"source_transaction_id": "tx_0", "target_transaction_id": "tx_1"},
        {"source_transaction_id": "tx_1", "target_transaction_id": "tx_2"},
    ]


def test_transaction_graph_combines_parsed_date_with_time() -> None:
    """A parsed datetime date column combines correctly with a time string."""
    frame = pl.LazyFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "Time": ["09:00:00", "09:30:00"],
            "Date": ["2025-01-01", "2025-01-01"],
        }
    )
    cleaned = clean_lazy_frame(
        frame,
        source_column="source",
        target_column="target",
        timestamp_columns=("Date",),
    )

    graph = build_transaction_graph(cleaned, delta=timedelta(hours=1))

    assert graph.num_edges == 1
    assert graph.edges["time_delta"].to_list() == [timedelta(minutes=30)]


def test_transaction_graph_keeps_full_timestamp_when_date_also_exists() -> None:
    """Full timestamps are not combined again with a separate date column."""
    frame = pl.LazyFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "Time": ["2025-01-01 09:00:00", "2025-01-01 09:30:00"],
            "Date": ["2025-01-01", "2025-01-01"],
        }
    )

    graph = build_transaction_graph(frame, delta=timedelta(hours=1))

    assert graph.num_edges == 1
    assert graph.edges["time_delta"].to_list() == [timedelta(minutes=30)]


def test_transaction_graph_accepts_date_only_strings() -> None:
    """Date-only timestamp columns retain their midnight timestamp."""
    graph = build_transaction_graph(
        pl.DataFrame(
            {
                "source": ["A"],
                "target": ["B"],
                "Date": ["2025-01-01"],
            }
        ),
        delta=timedelta(hours=1),
    )

    assert graph.num_nodes == 1


def test_transaction_graph_preserves_nanosecond_ordering() -> None:
    """Sub-microsecond timestamps remain ordered and measurable."""
    frame = pl.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "timestamp": [0, 500],
        },
        schema={"source": pl.String, "target": pl.String, "timestamp": pl.Datetime("ns")},
    )

    graph = build_transaction_graph(frame, delta=timedelta(microseconds=1))

    assert graph.num_edges == 1
    assert graph.edges["time_delta"].cast(pl.Int64).to_list() == [500]


def test_transaction_graph_rejects_exceeded_delta() -> None:
    """Successors beyond the inclusive time window are excluded."""
    frame = _transaction_frame(
        [
            ("t1", "A", "B", "2025-01-01 09:00", 1.0, 0),
            ("t2", "B", "C", "2025-01-01 10:01", 2.0, 0),
        ]
    )

    graph = build_transaction_graph(frame, delta=timedelta(hours=1))

    assert graph.num_edges == 0


def test_transaction_graph_creates_directional_temporal_edges() -> None:
    """Transaction edges follow account direction and the inclusive window."""
    frame = _transaction_frame(
        [
            ("t1", "A", "B", "2025-01-01 09:00", 10.0, 0),
            ("t2", "B", "C", "2025-01-01 09:20", 20.0, 0),
            ("t3", "B", "D", "2025-01-01 10:00", 30.0, 1),
            ("t4", "X", "B", "2025-01-01 09:30", 40.0, 0),
        ]
    )

    graph = build_transaction_graph(frame, delta=timedelta(hours=1))

    assert isinstance(graph, TransactionGraph)
    assert graph.num_nodes == 4
    assert graph.edges.select(
        ["source_transaction_id", "target_transaction_id"]
    ).to_dicts() == [
        {"source_transaction_id": "t1", "target_transaction_id": "t2"},
        {"source_transaction_id": "t1", "target_transaction_id": "t3"},
        {"source_transaction_id": "t4", "target_transaction_id": "t3"},
    ]
    assert graph.edges["via_account"].to_list() == ["B", "B", "B"]
    assert graph.edges["time_delta"].to_list() == [
        timedelta(minutes=20),
        timedelta(hours=1),
        timedelta(minutes=30),
    ]


def test_transaction_graph_rejects_same_or_backward_time() -> None:
    """Only strictly later transactions may receive temporal edges."""
    frame = _transaction_frame(
        [
            ("t1", "A", "B", "2025-01-01 10:00", 1.0, 0),
            ("t2", "B", "C", "2025-01-01 10:00", 2.0, 0),
            ("t3", "B", "D", "2025-01-01 09:00", 3.0, 0),
        ]
    )

    graph = build_transaction_graph(frame, delta=timedelta(hours=2))

    assert graph.num_edges == 0


def test_transaction_graph_requires_timestamp_and_delta() -> None:
    """Required timestamps and non-negative timedelta values are validated."""
    frame = pl.DataFrame({"source": ["A"], "target": ["B"]})

    with pytest.raises(ValueError, match="timestamp"):
        build_transaction_graph(frame, delta=timedelta(hours=1))
    with pytest.raises(ValueError, match="non-negative"):
        build_transaction_graph(
            pl.DataFrame(
                {
                    "source": ["A"],
                    "target": ["B"],
                    "timestamp": [datetime(2025, 1, 1, tzinfo=UTC)],
                }
            ),
            delta=timedelta(seconds=-1),
        )
