"""Inspect account-node static, snapshot, and event batching on IBM HI-Small.

Run / 运行::

    uv run python examples/ibm_account_batching.py

The default input is a small deterministic temporal sample spread across six
time bands of IBM HI-Small. The script prints the object type, tensor shapes,
and short value previews at every batching boundary.

For account snapshots, ``SnapshotDataLoader`` creates samples of the form
``(G[t-k], ..., G[t-1]) -> G[t]``. A batch contains one PyG ``Batch`` per
relative time position, so the temporal axis remains explicit instead of
being flattened into one graph.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import polars as pl
from torch import Tensor
from torch_geometric.data import Batch, Data, TemporalData

from amlgraphx.data import (
    SnapshotDataLoader,
    SnapshotWindowDataset,
    event_stream_loader,
    static_graph_loader,
)
from amlgraphx.datasets import load_dataset
from amlgraphx.graph import GraphFeatureSpec, prepare_pyg_graph

DEFAULT_LIMIT = 12_000
EDGE_DELTA = timedelta(hours=4)
WINDOW_SIZE = timedelta(days=1)
CONTEXT_SIZE = 5
TEMPORAL_BANDS = 6


def _load_temporally_spread_transactions(
    *, cache_dir: Path | None, limit: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load bounded IBM transactions and account metadata."""
    dataset = load_dataset("ibm-aml", variant="hi-small", cache_dir=cache_dir)
    transactions = dataset.transactions()
    accounts = dataset.accounts().collect()
    if limit == 0:
        return transactions.collect(), accounts
    if limit < 0:
        raise ValueError("limit must be non-negative")

    bounds = (
        transactions.select(
            pl.col("timestamp").min().alias("minimum"),
            pl.col("timestamp").max().alias("maximum"),
        )
        .collect()
        .row(0)
    )
    minimum, maximum = bounds
    if minimum is None or maximum is None:
        raise ValueError("IBM HI-Small contains no timestamped transactions")

    per_band = max(1, limit // TEMPORAL_BANDS)
    span = maximum - minimum
    pieces: list[pl.DataFrame] = []
    for band in range(TEMPORAL_BANDS):
        start = minimum + span * band / TEMPORAL_BANDS
        end = (
            minimum + span * (band + 1) / TEMPORAL_BANDS
            if band + 1 < TEMPORAL_BANDS
            else maximum + timedelta(microseconds=1)
        )
        piece = (
            transactions.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
            )
            .limit(per_band)
            .collect()
        )
        if not piece.is_empty():
            pieces.append(piece)
    if not pieces:
        raise ValueError("the selected IBM time bands contain no transactions")
    return pl.concat(pieces, how="vertical_relaxed").sort("timestamp"), accounts


def _preview(value: object) -> str:
    """Return a short shape/dtype/value preview for a tensor field."""
    if not isinstance(value, Tensor):
        return repr(value)
    if value.ndim == 2 and value.shape[0] == 2:
        values = value[:, :3].tolist()
    else:
        values = value[:3].tolist()
    return f"shape={tuple(value.shape)}, dtype={value.dtype}, head={values}"


def _describe_data(name: str, data: Data | TemporalData) -> None:
    """Print standard PyG fields for static or event data."""
    print(f"\n{name}")
    if isinstance(data, TemporalData):
        print(
            f"type={type(data).__name__}, num_nodes={data.num_nodes}, "
            f"num_events={data.num_events}"
        )
        fields = ("src", "dst", "t", "msg", "y", "x")
    else:
        print(
            f"type={type(data).__name__}, num_nodes={data.num_nodes}, "
            f"num_edges={data.num_edges}"
        )
        fields = ("x", "edge_index", "edge_attr", "edge_time", "edge_y")
    for field in fields:
        value = getattr(data, field, None)
        if value is not None:
            print(f"{field}: {_preview(value)}")


def _describe_snapshot_batch(batch: object) -> None:
    """Print the nested shape of one account snapshot sequence batch."""
    print("\nSnapshotBatch / snapshot 序列 batch")
    print(f"context positions={len(batch.context)}")  # type: ignore[union-attr]
    for position, graph in enumerate(batch.context):  # type: ignore[union-attr]
        assert isinstance(graph, Batch)
        print(
            f"context[{position}]: type=Batch, num_graphs={graph.num_graphs}, "
            f"num_nodes={graph.num_nodes}, num_edges={graph.num_edges}, "
            f"edge_y_shape={tuple(graph.edge_y.shape)}"
        )
        print(f"  edge_index: {_preview(graph.edge_index)}")
    target = batch.target  # type: ignore[union-attr]
    print(
        f"target: type=Batch, num_graphs={target.num_graphs}, "
        f"num_nodes={target.num_nodes}, num_edges={target.num_edges}, "
        f"edge_y_shape={tuple(target.edge_y.shape)}"
    )
    print(f"target.edge_y: {_preview(target.edge_y)}")


def main(*, cache_dir: Path | None = None, limit: int = DEFAULT_LIMIT) -> None:
    """Build and inspect all three account-node batching paths."""
    transactions, accounts = _load_temporally_spread_transactions(
        cache_dir=cache_dir,
        limit=limit,
    )
    print("IBM HI-Small transaction input / 交易输入")
    print(f"shape={transactions.shape}; account metadata shape={accounts.shape}")
    print(transactions.select("transaction_id", "timestamp", "amount", "label").head(3))

    features = GraphFeatureSpec(
        node_columns=("Bank ID",),
        edge_columns=("amount",),
        label_column="label",
    )

    account_data = prepare_pyg_graph(
        transactions,
        node_type="account",
        temporal="static",
        account_metadata=accounts,
        features=features,
    )
    _describe_data("1. Account static graph / 账户静态图", account_data)

    static_loader = static_graph_loader(
        account_data,
        window_size=WINDOW_SIZE,
        lookback=EDGE_DELTA,
        batch_size=2,
        shuffle=False,
    )
    static_batch = next(iter(static_loader))
    if int(static_batch.target_edge_mask.sum()) == static_batch.num_edges:
        static_batch = next(
            (
                candidate
                for candidate in static_loader
                if int(candidate.target_edge_mask.sum()) < candidate.num_edges
            ),
            static_batch,
        )
    _describe_data("1b. Account static window Batch / 账户静态窗口 Batch", static_batch)
    print(f"num_graphs={static_batch.num_graphs}")
    print(f"batch assignment head={static_batch.batch[:10].tolist()}")
    print(f"window_id={static_batch.window_id.tolist()}")
    print(
        "target_edge_mask selects "
        f"{int(static_batch.target_edge_mask.sum())} of "
        f"{static_batch.num_edges} edges."
    )

    snapshot_data = list(
        prepare_pyg_graph(
            transactions,
            node_type="account",
            temporal="snapshot",
            account_metadata=accounts,
            features=features,
            bin_size=WINDOW_SIZE,
            stride=WINDOW_SIZE,
            drop_last=False,
        )
    )
    print(f"\n2. Account snapshots / 账户 snapshots: count={len(snapshot_data)}")
    print(
        "snapshot indices / snapshot 编号:",
        [int(snapshot.snapshot_index) for snapshot in snapshot_data[:6]],
    )
    _describe_data("First account snapshot / 第一张账户 snapshot", snapshot_data[0])
    snapshot_dataset = SnapshotWindowDataset(
        snapshot_data,
        context_size=CONTEXT_SIZE,
    )
    snapshot_loader = SnapshotDataLoader(
        snapshot_dataset,
        batch_size=2,
        shuffle=False,
    )
    snapshot_batch = next(iter(snapshot_loader))
    _describe_snapshot_batch(snapshot_batch)
    print(
        "The tuple keeps time order; each inner Batch contains disconnected "
        "graphs from different samples."
    )

    events = prepare_pyg_graph(
        transactions,
        node_type="account",
        temporal="event_stream",
        account_metadata=accounts,
        features=features,
    )
    _describe_data("3. Account event stream / 账户事件流", events)
    event_batch = next(iter(event_stream_loader(events, batch_size=8)))
    _describe_data("3b. Event batch / 事件 Batch", event_batch)
    print(
        "TemporalDataLoader slices events, not the global account feature table: "
        "event endpoints index the full x matrix."
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for the manual example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Rows sampled across six time bands; 0 means the full dataset.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(cache_dir=args.cache_dir, limit=args.limit)
