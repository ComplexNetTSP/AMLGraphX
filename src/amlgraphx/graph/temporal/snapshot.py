"""Snapshot graph temporal representation.

This module owns the public concept of a snapshot sequence: temporal bins,
window movement, and snapshot-specific output. The current implementation is
delegated to the existing transaction data module so there is one algorithm
and no duplicate graph logic.

中文：
    本模块负责 snapshot 序列的公共语义：时间 bin、窗口移动和 snapshot 输出。
    当前实现暂时委托给已有的 transaction data module，从而保持只有一份算法。

The current scope is transaction-as-node snapshots. Account-as-node snapshots
need a separate edge aggregation policy and are intentionally not guessed here.
当前范围是 transaction-as-node snapshot；账户为节点的 snapshot 需要单独的边
聚合语义，本模块不会擅自假设。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

from amlgraphx.data.datamodule import GraphSnapshot, sliding_snapshots
from amlgraphx.graph.graphs import TransactionGraph


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


__all__ = ["GraphSnapshot", "build_transaction_snapshots", "sliding_snapshots"]
