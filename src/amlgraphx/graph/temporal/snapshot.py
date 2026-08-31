"""Snapshot graph temporal representation.

This module owns the public concept of a snapshot sequence: temporal bins,
window movement, and snapshot-specific output. Transaction-node snapshots
reuse the existing data module. Account-node snapshots select transaction
edges by time and keep all edge feature columns unchanged.

中文：
    本模块负责 snapshot 序列的公共语义：时间 bin、窗口移动和 snapshot 输出。
    当前实现暂时委托给已有的 transaction data module，从而保持只有一份算法。

Account snapshots are disjoint or overlapping edge windows depending on
``bin_size`` and ``stride``. Parallel transactions are not aggregated.
账户 snapshot 按边时间切窗；是否重叠由 ``bin_size`` 与 ``stride`` 决定，
同一对账户的多笔交易不会被聚合。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import polars as pl
import torch
from torch import Tensor

from amlgraphx.data.datamodule import GraphSnapshot, sliding_snapshots
from amlgraphx.graph.graphs import AccountGraph, TransactionGraph


def build_account_snapshots(
    graph: AccountGraph,
    *,
    bin_size: timedelta,
    stride: timedelta,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    drop_last: bool = True,
) -> Iterator[GraphSnapshot]:
    """Yield account-node snapshots over transaction-edge time windows.

    Every transaction in ``[window_start, window_end)`` remains one edge with
    its complete feature row. Only accounts active in that window are included
    as nodes; their stable ``node_id`` values remain in ``graph.nodes`` while
    ``edge_index`` uses local contiguous positions.

    中文：每个半开窗口内的交易都作为独立边保留完整特征。snapshot 只包含该
    窗口活跃账户，稳定身份保存在 ``node_id``，``edge_index`` 使用局部连续索引。
    """
    _validate_duration(bin_size, "bin_size")
    _validate_duration(stride, "stride")
    _validate_account_graph(graph)

    bounds = _account_bounds(graph, start_time, end_time)
    if bounds is None:
        return

    window_start, range_end = bounds
    snapshot_index = 1
    while window_start < range_end:
        window_end = window_start + bin_size
        if window_end > range_end and drop_last:
            break
        window_end = min(window_end, range_end)

        snapshot = _account_snapshot(
            graph,
            start_time=window_start,
            end_time=window_end,
            index=snapshot_index,
        )
        if snapshot.num_edges > 0:
            yield snapshot
            snapshot_index += 1
        window_start += stride


def build_transaction_snapshots(
    graph: TransactionGraph,
    *,
    bin_size: timedelta,
    stride: timedelta,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    drop_last: bool = True,
) -> Iterator[GraphSnapshot]:
    """Yield transaction-node snapshots over half-open temporal bins.

    ``bin_size`` defines the duration of one snapshot and ``stride`` defines
    how far the next bin starts. If they are equal, bins are disjoint; if the
    stride is smaller, snapshots overlap. The function preserves the current
    induced-subgraph semantics and local sparse ``edge_index`` representation.

    中文：
        ``bin_size`` 定义一张 snapshot 覆盖的时间长度，``stride`` 定义下一张
        snapshot 的起点移动距离。两者相等时窗口不重叠，stride 更小时窗口
        重叠。本函数保留当前诱导子图和局部稀疏 ``edge_index`` 语义。

    This wrapper is intentionally small. The implementation can move here in a
    later change without changing the public function name.
    / 当前只保留薄包装；之后可以把实现迁移到这里而不改变公共函数名。
    """
    return sliding_snapshots(
        graph,
        window_size=bin_size,
        stride=stride,
        start_time=start_time,
        end_time=end_time,
        drop_last=drop_last,
    )


def _account_snapshot(
    graph: AccountGraph,
    *,
    start_time: datetime,
    end_time: datetime,
    index: int,
) -> GraphSnapshot:
    """Select one account edge window without aggregating parallel edges."""
    timestamp = pl.col("timestamp")
    edges = graph.edges.filter((timestamp >= start_time) & (timestamp < end_time))
    if edges.is_empty():
        empty_graph = AccountGraph(nodes=graph.nodes.head(0), edges=edges)
        return GraphSnapshot(
            graph=empty_graph,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            start_time=start_time,
            end_time=end_time,
            index=index,
        )

    active_nodes = pl.concat(
        [
            edges.select(pl.col("source").alias("node_id")),
            edges.select(pl.col("target").alias("node_id")),
        ]
    ).unique("node_id")
    nodes = graph.nodes.join(active_nodes, on="node_id", how="semi")
    snapshot_graph = AccountGraph(nodes=nodes, edges=edges)
    return GraphSnapshot(
        graph=snapshot_graph,
        edge_index=_account_edge_index(snapshot_graph),
        start_time=start_time,
        end_time=end_time,
        index=index,
    )


def _account_edge_index(graph: AccountGraph) -> Tensor:
    """Map stable account IDs to local PyG-style edge positions."""
    if graph.edges.is_empty():
        return torch.empty((2, 0), dtype=torch.long)

    node_indices = graph.nodes.select("node_id").with_row_index("node_index")
    source_indices = node_indices.rename(
        {"node_id": "source", "node_index": "source_index"}
    )
    target_indices = node_indices.rename(
        {"node_id": "target", "node_index": "target_index"}
    )
    indexed = (
        graph.edges.with_row_index("__edge_order")
        .join(source_indices, on="source", how="inner")
        .join(target_indices, on="target", how="inner")
        .sort("__edge_order")
    )
    values = indexed.select("source_index", "target_index").to_numpy()
    return torch.tensor(values.T, dtype=torch.long).contiguous()


def _account_bounds(
    graph: AccountGraph,
    start_time: datetime | None,
    end_time: datetime | None,
) -> tuple[datetime, datetime] | None:
    """Resolve requested bounds from account edge timestamps."""
    if start_time is not None and not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime")
    if end_time is not None and not isinstance(end_time, datetime):
        raise TypeError("end_time must be a datetime")
    if graph.edges.is_empty():
        return None

    timestamps = graph.edges["timestamp"]
    resolved_start = start_time or timestamps.min()
    resolved_end = end_time or timestamps.max() + timedelta(microseconds=1)
    if resolved_start >= resolved_end:
        raise ValueError("start_time must be earlier than end_time")
    return resolved_start, resolved_end


def _validate_account_graph(graph: AccountGraph) -> None:
    """Require canonical account nodes, endpoints, and typed edge time."""
    if not isinstance(graph, AccountGraph):
        raise TypeError("graph must be an AccountGraph")
    required_nodes = {"node_id"}
    required_edges = {"source", "target", "timestamp"}
    missing = required_nodes.difference(graph.nodes.columns)
    missing.update(required_edges.difference(graph.edges.columns))
    if missing:
        raise ValueError(f"Account graph is missing: {', '.join(sorted(missing))}")
    if graph.edges.schema["timestamp"].base_type() != pl.Datetime:
        raise TypeError("account graph timestamp must be a Polars Datetime")


def _validate_duration(value: timedelta, name: str) -> None:
    """Require a positive window duration."""
    if not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a datetime.timedelta")
    if value <= timedelta(0):
        raise ValueError(f"{name} must be positive")


__all__ = [
    "GraphSnapshot",
    "build_account_snapshots",
    "build_transaction_snapshots",
    "sliding_snapshots",
]
