"""Temporal graph splitting and sparse sliding-window snapshots."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import torch
from torch import Tensor

from amlgraphx.graph import TransactionGraph, build_transaction_graph

type TransactionTable = pl.DataFrame | pl.LazyFrame


@dataclass(frozen=True, slots=True)
class TransactionGraphSplit:
    """Hold leakage-free temporal partitions of a transaction graph.

    Each partition is an induced subgraph. Edges whose endpoints belong to
    different time partitions are excluded.

    Args:
        train: Transactions before the training cutoff.
        validation: Transactions between the two cutoffs.
        test: Transactions at or after the validation cutoff.
    """

    train: TransactionGraph
    validation: TransactionGraph
    test: TransactionGraph


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """Represent one temporal transaction-graph window with sparse edges.

    ``edge_index`` uses local node positions and has shape ``[2, num_edges]``.
    No dense adjacency matrix is materialized.

    Args:
        graph: Induced transaction graph for the window.
        edge_index: Sparse COO-style source and target node indices.
        start_time: Inclusive beginning of the window.
        end_time: Exclusive end of the window.
        index: One-based snapshot number within its partition.
    """

    graph: TransactionGraph
    edge_index: Tensor
    start_time: datetime
    end_time: datetime
    index: int

    @property
    def num_nodes(self) -> int:
        """Return the number of transaction nodes in the snapshot."""
        return self.graph.num_nodes

    @property
    def num_edges(self) -> int:
        """Return the number of sparse edges in the snapshot."""
        return self.edge_index.shape[1]


def split_transaction_graph(
    graph: TransactionGraph,
    *,
    train_end: datetime,
    validation_end: datetime,
) -> TransactionGraphSplit:
    """Split a transaction graph into chronological induced subgraphs.

    The intervals are ``[-inf, train_end)``,
    ``[train_end, validation_end)``, and ``[validation_end, +inf)``.

    Args:
        graph: Complete temporal transaction graph.
        train_end: Exclusive end of the training period.
        validation_end: Exclusive end of the validation period.

    Returns:
        Train, validation, and test transaction subgraphs.

    Raises:
        TypeError: If a cutoff is not a ``datetime``.
        ValueError: If cutoffs are unordered or graph columns are missing.
    """
    _validate_cutoffs(train_end, validation_end)
    _validate_graph(graph)

    timestamp = pl.col("timestamp")
    train_nodes = graph.nodes.filter(timestamp < train_end)
    validation_nodes = graph.nodes.filter(
        (timestamp >= train_end) & (timestamp < validation_end)
    )
    test_nodes = graph.nodes.filter(timestamp >= validation_end)

    return TransactionGraphSplit(
        train=_induced_graph(graph, train_nodes),
        validation=_induced_graph(graph, validation_nodes),
        test=_induced_graph(graph, test_nodes),
    )


def sliding_snapshots(
    graph: TransactionGraph,
    *,
    window_size: timedelta,
    stride: timedelta,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    drop_last: bool = True,
) -> Iterator[GraphSnapshot]:
    """Yield sparse graph snapshots over half-open time windows.

    Empty windows are skipped. By default, a final window that crosses
    ``end_time`` is omitted. Set ``drop_last=False`` to emit that final window
    with its end clipped to the requested range.

    Args:
        graph: Transaction graph to sample.
        window_size: Positive duration represented by each full snapshot.
        stride: Positive distance between consecutive window starts.
        start_time: Optional inclusive first window start.
        end_time: Optional exclusive sampling boundary.
        drop_last: Whether to omit a final incomplete window.

    Yields:
        Snapshots containing local nodes, edge attributes, and sparse edges.

    Raises:
        TypeError: If durations or explicit boundaries have invalid types.
        ValueError: If durations are not positive or boundaries are unordered.
    """
    _validate_duration(window_size, "window_size")
    _validate_duration(stride, "stride")
    _validate_graph(graph)

    bounds = _resolve_sampling_bounds(graph, start_time, end_time)
    if bounds is None:
        return

    range_start, range_end = bounds
    window_start = range_start
    snapshot_index = 1
    while window_start < range_end:
        window_end = window_start + window_size
        if window_end > range_end and drop_last:
            break
        window_end = min(window_end, range_end)

        snapshot = _build_snapshot(
            graph,
            start_time=window_start,
            end_time=window_end,
            index=snapshot_index,
        )
        if snapshot.num_nodes > 0:
            yield snapshot
            snapshot_index += 1
        window_start += stride


class TransactionGraphDataModule:
    """Prepare temporal splits and sliding snapshots for transaction GNNs.

    The module builds one complete transaction graph first, divides it by node
    timestamps, and samples each partition independently. Call ``setup()``
    before requesting graphs or snapshots.

    Args:
        transactions: Canonical transaction table.
        edge_delta: Maximum interval for temporal-flow graph edges.
        train_end: Exclusive end of the training partition.
        validation_end: Exclusive end of the validation partition.
        window_size: Duration represented by each full snapshot.
        stride: Distance between consecutive snapshot starts.
        test_end: Optional exclusive end of test snapshot generation.
        drop_last: Whether to omit incomplete final snapshots.
    """

    def __init__(
        self,
        transactions: TransactionTable,
        *,
        edge_delta: timedelta,
        train_end: datetime,
        validation_end: datetime,
        window_size: timedelta,
        stride: timedelta,
        test_end: datetime | None = None,
        drop_last: bool = True,
    ) -> None:
        _validate_cutoffs(train_end, validation_end)
        _validate_duration(edge_delta, "edge_delta", allow_zero=True)
        _validate_duration(window_size, "window_size")
        _validate_duration(stride, "stride")
        if test_end is not None and test_end <= validation_end:
            raise ValueError("test_end must be later than validation_end")

        self.transactions = transactions
        self.edge_delta = edge_delta
        self.train_end = train_end
        self.validation_end = validation_end
        self.window_size = window_size
        self.stride = stride
        self.test_end = test_end
        self.drop_last = drop_last
        self._full_graph: TransactionGraph | None = None
        self._splits: TransactionGraphSplit | None = None

    def setup(self) -> None:
        """Build the complete graph and its chronological partitions once."""
        if self._full_graph is not None:
            return

        self._full_graph = build_transaction_graph(
            self.transactions,
            delta=self.edge_delta,
        )
        self._splits = split_transaction_graph(
            self._full_graph,
            train_end=self.train_end,
            validation_end=self.validation_end,
        )

    @property
    def full_graph(self) -> TransactionGraph:
        """Return the complete graph created by ``setup()``."""
        if self._full_graph is None:
            raise RuntimeError("Call setup() before accessing full_graph")
        return self._full_graph

    @property
    def splits(self) -> TransactionGraphSplit:
        """Return the temporal graph partitions created by ``setup()``."""
        if self._splits is None:
            raise RuntimeError("Call setup() before accessing splits")
        return self._splits

    def train_snapshots(self) -> Iterator[GraphSnapshot]:
        """Yield snapshots contained entirely in the training period."""
        return sliding_snapshots(
            self.splits.train,
            window_size=self.window_size,
            stride=self.stride,
            end_time=self.train_end,
            drop_last=self.drop_last,
        )

    def validation_snapshots(self) -> Iterator[GraphSnapshot]:
        """Yield snapshots contained entirely in the validation period."""
        return sliding_snapshots(
            self.splits.validation,
            window_size=self.window_size,
            stride=self.stride,
            start_time=self.train_end,
            end_time=self.validation_end,
            drop_last=self.drop_last,
        )

    def test_snapshots(self) -> Iterator[GraphSnapshot]:
        """Yield snapshots contained entirely in the test period."""
        return sliding_snapshots(
            self.splits.test,
            window_size=self.window_size,
            stride=self.stride,
            start_time=self.validation_end,
            end_time=self.test_end,
            drop_last=self.drop_last,
        )


def _build_snapshot(
    graph: TransactionGraph,
    *,
    start_time: datetime,
    end_time: datetime,
    index: int,
) -> GraphSnapshot:
    timestamp = pl.col("timestamp")
    nodes = graph.nodes.filter((timestamp >= start_time) & (timestamp < end_time))
    snapshot_graph = _induced_graph(graph, nodes)
    return GraphSnapshot(
        graph=snapshot_graph,
        edge_index=_edge_index(snapshot_graph),
        start_time=start_time,
        end_time=end_time,
        index=index,
    )


def _induced_graph(
    graph: TransactionGraph,
    nodes: pl.DataFrame,
) -> TransactionGraph:
    if nodes.is_empty():
        return TransactionGraph(nodes=nodes, edges=graph.edges.head(0))

    source_ids = nodes.select(pl.col("transaction_id").alias("source_transaction_id"))
    target_ids = nodes.select(pl.col("transaction_id").alias("target_transaction_id"))
    edges = graph.edges.join(
        source_ids,
        on="source_transaction_id",
        how="semi",
    ).join(
        target_ids,
        on="target_transaction_id",
        how="semi",
    )
    return TransactionGraph(nodes=nodes, edges=edges)


def _edge_index(graph: TransactionGraph) -> Tensor:
    if graph.edges.is_empty():
        return torch.empty((2, 0), dtype=torch.long)

    node_indices = graph.nodes.select("transaction_id").with_row_index("node_index")
    source_indices = node_indices.rename(
        {
            "transaction_id": "source_transaction_id",
            "node_index": "source_index",
        }
    )
    target_indices = node_indices.rename(
        {
            "transaction_id": "target_transaction_id",
            "node_index": "target_index",
        }
    )
    indexed_edges = graph.edges.join(
        source_indices,
        on="source_transaction_id",
        how="inner",
    ).join(
        target_indices,
        on="target_transaction_id",
        how="inner",
    )
    indices = indexed_edges.select("source_index", "target_index").to_numpy()
    return torch.as_tensor(indices.T, dtype=torch.long).contiguous()


def _resolve_sampling_bounds(
    graph: TransactionGraph,
    start_time: datetime | None,
    end_time: datetime | None,
) -> tuple[datetime, datetime] | None:
    if start_time is not None and not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime")
    if end_time is not None and not isinstance(end_time, datetime):
        raise TypeError("end_time must be a datetime")
    if graph.nodes.is_empty():
        return None

    timestamps = graph.nodes.get_column("timestamp")
    resolved_start = start_time or timestamps.min()
    resolved_end = end_time or timestamps.max() + timedelta(microseconds=1)
    if resolved_start >= resolved_end:
        raise ValueError("start_time must be earlier than end_time")
    return resolved_start, resolved_end


def _validate_graph(graph: TransactionGraph) -> None:
    if not isinstance(graph, TransactionGraph):
        raise TypeError("graph must be a TransactionGraph")
    required = {"transaction_id", "timestamp"}
    missing = required.difference(graph.nodes.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Transaction graph nodes are missing: {columns}")


def _validate_cutoffs(train_end: datetime, validation_end: datetime) -> None:
    if not isinstance(train_end, datetime):
        raise TypeError("train_end must be a datetime")
    if not isinstance(validation_end, datetime):
        raise TypeError("validation_end must be a datetime")
    if train_end >= validation_end:
        raise ValueError("train_end must be earlier than validation_end")


def _validate_duration(
    value: timedelta,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    if not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a datetime.timedelta")
    minimum = timedelta(0)
    if value < minimum or (value == minimum and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
