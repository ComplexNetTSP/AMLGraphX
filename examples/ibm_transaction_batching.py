"""Inspect transaction-as-node PyG inputs and batching on IBM HI-Small.

Run / 运行::

    uv run python examples/ibm_transaction_batching.py

The default input is a small deterministic temporal sample spread across six
time bands of IBM HI-Small. Use ``--limit 0`` only when enough memory is
available for the complete dataset graph.

This example contrasts two valid uses of a transaction-as-node graph:

* one complete time-aware static ``Data`` object;
* disjoint target windows returned as ordinary PyG ``Batch`` objects.

The windows are a memory/batching strategy, not a transaction snapshot
evolution. ``lookback`` preserves causal predecessor context while the target
mask selects only transactions in the current window.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import polars as pl
from torch import Tensor
from torch_geometric.data import Data

from amlgraphx.data import static_graph_loader
from amlgraphx.datasets import load_dataset
from amlgraphx.graph import GraphFeatureSpec, prepare_pyg_graph

DEFAULT_LIMIT = 12_000
EDGE_DELTA = timedelta(hours=4)
WINDOW_SIZE = timedelta(days=1)
TEMPORAL_BANDS = 6


def _load_temporally_spread_transactions(
    *, cache_dir: Path | None, limit: int
) -> pl.DataFrame:
    """Load a bounded IBM sample while retaining more than one time window."""
    dataset = load_dataset("ibm-aml", variant="hi-small", cache_dir=cache_dir)
    transactions = dataset.transactions()
    if limit == 0:
        return transactions.collect()
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
    return pl.concat(pieces, how="vertical_relaxed").sort("timestamp")


def _preview(value: object) -> str:
    """Return a short shape/dtype/value preview for a tensor field."""
    if not isinstance(value, Tensor):
        return repr(value)
    if value.ndim == 2 and value.shape[0] == 2:
        values = value[:, :3].tolist()
    else:
        values = value[:3].tolist()
    return f"shape={tuple(value.shape)}, dtype={value.dtype}, head={values}"


def _describe_data(name: str, data: Data) -> None:
    """Print the standard PyG fields a researcher model receives."""
    print(f"\n{name}")
    print(
        f"type={type(data).__name__}, num_nodes={data.num_nodes}, "
        f"num_edges={data.num_edges}"
    )
    for field in (
        "x",
        "edge_index",
        "edge_attr",
        "edge_time",
        "node_time",
        "node_y",
        "target_node_mask",
    ):
        value = getattr(data, field, None)
        if value is not None:
            print(f"{field}: {_preview(value)}")


def main(*, cache_dir: Path | None = None, limit: int = DEFAULT_LIMIT) -> None:
    """Build and inspect transaction static graph batches."""
    transactions = _load_temporally_spread_transactions(
        cache_dir=cache_dir,
        limit=limit,
    )
    print("IBM HI-Small transaction input / 交易输入")
    print(f"shape={transactions.shape}")
    print(transactions.select("transaction_id", "timestamp", "amount", "label").head(3))

    data = prepare_pyg_graph(
        transactions,
        node_type="transaction",
        temporal="static",
        edge_delta=EDGE_DELTA,
        features=GraphFeatureSpec(
            node_columns=("amount",),
            edge_columns=("time_delta",),
            label_column="label",
        ),
    )
    print(
        "\nA full graph is passed directly to a model as Data; no mini-batch is "
        "needed when it fits memory."
    )
    _describe_data("Full transaction graph / 完整交易图", data)

    loader = static_graph_loader(
        data,
        window_size=WINDOW_SIZE,
        lookback=EDGE_DELTA,
        batch_size=2,
        shuffle=False,
    )
    print(
        "\nSliding-window dataset / 滑动窗口数据集: "
        f"{len(loader.dataset)} non-empty target windows"
    )
    batch = next(iter(loader))
    if int(batch.target_node_mask.sum()) == batch.num_nodes:
        # The first window may have no earlier observations. Pick a later
        # batch when available so the printed example visibly shows lookback
        # context being excluded from the target mask.
        batch = next(
            (
                candidate
                for candidate in loader
                if int(candidate.target_node_mask.sum()) < candidate.num_nodes
            ),
            batch,
        )
    _describe_data("Window batch / 窗口 PyG Batch", batch)
    print(f"num_graphs={batch.num_graphs}")
    print(f"batch assignment head={batch.batch[:10].tolist()}")
    print(f"window_id={batch.window_id.tolist()}")
    print(
        "target_node_mask selects "
        f"{int(batch.target_node_mask.sum())} of {batch.num_nodes} nodes; "
        "the remaining nodes are lookback context."
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
