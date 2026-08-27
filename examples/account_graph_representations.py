"""Build account static, snapshot, and event-stream representations.

Run / 运行：
    ``uv run python examples/account_graph_representations.py``

Private CSV or Parquet data can replace the small Polars frame below. Every
transaction column remains available as an edge feature until the researcher
explicitly selects numerical columns for PyG.
"""

from datetime import UTC, datetime, timedelta

import polars as pl

from amlgraphx.graph import prepare_graph, to_pyg_data, to_pyg_temporal_data


def main() -> None:
    """Show the three natural account-node outputs and their PyG adapters."""
    transactions = pl.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "source": ["A", "A", "B"],
            "target": ["B", "B", "C"],
            "timestamp": [
                datetime(2025, 1, 1, 9, tzinfo=UTC),
                datetime(2025, 1, 1, 10, tzinfo=UTC),
                datetime(2025, 1, 2, 9, tzinfo=UTC),
            ],
            "amount": [10.0, 20.0, 30.0],
            "label": [0, 1, 0],
            "channel": ["cash", "wire", "wire"],
        }
    )
    accounts = pl.DataFrame(
        {"node_id": ["A", "B", "C"], "balance": [100.0, 200.0, 300.0]}
    )

    static_graph = prepare_graph(
        transactions,
        node_type="account",
        temporal="static",
        account_metadata=accounts,
    )
    pyg_graph = to_pyg_data(
        static_graph,
        node_feature_columns=["balance"],
        edge_feature_columns=["amount"],
        edge_label_column="label",
    )
    print("Static / 静态图:", static_graph.num_nodes, static_graph.num_edges)
    print("PyG Data:", pyg_graph)

    snapshots = prepare_graph(
        transactions,
        node_type="account",
        temporal="snapshot",
        account_metadata=accounts,
        bin_size=timedelta(days=1),
        stride=timedelta(days=1),
        drop_last=False,
    )
    for snapshot in snapshots:
        # ``snapshot.graph.edges`` still contains amount, timestamp, channel,
        # label, and every other transaction feature.
        print("Snapshot:", snapshot.index, snapshot.graph.edges.shape)

    stream = prepare_graph(
        transactions,
        node_type="account",
        temporal="event_stream",
        account_metadata=accounts,
    )
    temporal_data = to_pyg_temporal_data(
        stream,
        message_columns=["amount"],
        label_column="label",
    )
    print("Event stream / 事件流:", temporal_data)


if __name__ == "__main__":
    main()
