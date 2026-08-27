"""High-level graph preparation entry points.

English:
    This module is the small convenience layer for research experiments. It
    selects the graph node semantics and the temporal representation, while
    the concrete builders and temporal modules own the actual semantics.

中文：
    本模块是面向研究实验的轻量便捷入口。它只选择图的节点语义和时间表示，
    具体的图构建与 temporal 逻辑仍由对应的 builder 和 temporal 模块负责。

The API deliberately does not expose a single ambiguous ``DynamicGraph``
object. A static request returns a concrete graph, while a snapshot request
returns an iterator of snapshots. / 本 API 不创建含义模糊的万能
``DynamicGraph``；静态请求返回具体图对象，snapshot 请求返回 snapshot 迭代器。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl

from .builders.account import AccountGraph, build_account_graph
from .builders.transaction import TransactionGraph, build_transaction_graph

if TYPE_CHECKING:
    from .temporal.snapshot import GraphSnapshot

type TransactionTable = pl.DataFrame | pl.LazyFrame
type GraphBuildResult = AccountGraph | TransactionGraph | Iterator["GraphSnapshot"]


class GraphNodeType(StrEnum):
    """Supported node meanings for transaction-derived graphs."""

    ACCOUNT = "account"
    TRANSACTION = "transaction"


class GraphTemporalMode(StrEnum):
    """Supported temporal representations for graph preparation."""

    STATIC = "static"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class GraphBuildSpec:
    """Describe one reproducible graph preparation choice.

    ``node_type`` chooses what becomes a graph node. ``temporal`` chooses one
    complete time-aware graph or a snapshot sequence. ``edge_delta`` belongs to
    transaction-flow edge construction; ``bin_size`` and ``stride`` belong to
    snapshot generation and are intentionally separate.

    中文：
        ``node_type`` 决定图节点的含义；``temporal`` 决定输出一张完整的
        时间感知图还是 snapshot 序列。``edge_delta`` 只控制交易流边，
        ``bin_size`` 与 ``stride`` 只控制 snapshot，三者不能混用。

    Account snapshots are reserved for a future implementation. Keeping that
    combination explicit prevents silently returning a transaction snapshot.
    / 账户 snapshot 暂未实现；显式保留这个组合可以避免错误地返回交易
    snapshot。
    """

    node_type: GraphNodeType | str
    temporal: GraphTemporalMode | str
    edge_delta: timedelta | None = None
    bin_size: timedelta | None = None
    stride: timedelta | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    drop_last: bool = True

    def __post_init__(self) -> None:
        """Normalize enum-like strings and validate mode-specific settings."""
        object.__setattr__(self, "node_type", GraphNodeType(self.node_type))
        object.__setattr__(self, "temporal", GraphTemporalMode(self.temporal))

        if self.temporal is GraphTemporalMode.STATIC and (
            self.bin_size is not None or self.stride is not None
        ):
            raise ValueError("bin_size and stride require temporal='snapshot'")
        if self.temporal is GraphTemporalMode.SNAPSHOT and (
            self.bin_size is None or self.stride is None
        ):
            raise ValueError("snapshot preparation requires both bin_size and stride")
        if self.node_type is GraphNodeType.TRANSACTION and self.edge_delta is None:
            raise ValueError("transaction graphs require edge_delta")


def prepare_graph(
    transactions: TransactionTable,
    *,
    node_type: GraphNodeType | str,
    temporal: GraphTemporalMode | str,
    edge_delta: timedelta | None = None,
    bin_size: timedelta | None = None,
    stride: timedelta | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    drop_last: bool = True,
    account_metadata: TransactionTable | None = None,
) -> GraphBuildResult:
    """Prepare a graph using explicit node and temporal semantics.

    Example / 示例：

    ``prepare_graph(tx, node_type="transaction", temporal="static",\
    edge_delta=timedelta(hours=1))`` returns a ``TransactionGraph``.

    ``prepare_graph(tx, node_type="transaction", temporal="snapshot",\
    edge_delta=timedelta(hours=1), bin_size=timedelta(days=1),\
    stride=timedelta(days=1))`` returns an iterator of ``GraphSnapshot``.

    The function is intentionally a thin dispatcher. It does not perform
    dataset loading, train/validation/test splitting, or backend conversion.
    / 本函数只是薄分发层，不负责数据集读取、train/validation/test 切分或后端
    转换。

    Raises:
        NotImplementedError: If account-as-node snapshots are requested.
        ValueError: If the requested combination is incomplete or invalid.
    """
    spec = GraphBuildSpec(
        node_type=node_type,
        temporal=temporal,
        edge_delta=edge_delta,
        bin_size=bin_size,
        stride=stride,
        start_time=start_time,
        end_time=end_time,
        drop_last=drop_last,
    )

    if spec.node_type is GraphNodeType.ACCOUNT:
        if spec.temporal is GraphTemporalMode.SNAPSHOT:
            raise NotImplementedError(
                "account-as-node snapshot preparation is not implemented yet"
            )
        return build_account_graph(
            transactions,
            account_metadata=account_metadata,
        )

    graph = build_transaction_graph(transactions, delta=spec.edge_delta)  # type: ignore[arg-type]
    if spec.temporal is GraphTemporalMode.STATIC:
        return graph

    from .temporal.snapshot import build_transaction_snapshots

    return build_transaction_snapshots(
        graph,
        bin_size=spec.bin_size,  # type: ignore[arg-type]
        stride=spec.stride,  # type: ignore[arg-type]
        start_time=spec.start_time,
        end_time=spec.end_time,
        drop_last=spec.drop_last,
    )


__all__ = [
    "GraphBuildResult",
    "GraphBuildSpec",
    "GraphNodeType",
    "GraphTemporalMode",
    "prepare_graph",
]
