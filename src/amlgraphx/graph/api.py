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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl
from torch_geometric.data import Data, TemporalData

from .builders.account import (
    AccountGraph,
    build_time_aware_account_graph,
)
from .builders.transaction import TransactionGraph, build_transaction_graph
from .pyg import to_pyg_data, to_pyg_temporal_data
from .temporal.event_stream import AccountEventStream

if TYPE_CHECKING:
    from .temporal.snapshot import GraphSnapshot

type TransactionTable = pl.DataFrame | pl.LazyFrame
type GraphBuildResult = (
    AccountGraph | TransactionGraph | AccountEventStream | Iterator["GraphSnapshot"]
)
type PyGGraphBuildResult = Data | TemporalData | Iterator[Data]


class GraphNodeType(StrEnum):
    """Supported node meanings for transaction-derived graphs."""

    ACCOUNT = "account"
    TRANSACTION = "transaction"


class GraphTemporalMode(StrEnum):
    """Supported temporal representations for graph preparation."""

    STATIC = "static"
    SNAPSHOT = "snapshot"
    EVENT_STREAM = "event_stream"


@dataclass(frozen=True, slots=True)
class GraphFeatureSpec:
    """Select model features without changing their graph semantics.

    Account graphs store account attributes on nodes and transaction attributes
    on edges. Transaction graphs store transaction attributes on nodes and
    relation attributes, such as ``time_delta``, on edges. ``label_column`` is
    therefore mapped automatically to ``edge_y`` for account-node graphs and
    to ``node_y`` for transaction-node graphs.

    Only numerical columns can be converted directly. Encode categorical
    columns before graph preparation or keep using the Polars graph tables.
    """

    node_columns: Sequence[str] = ()
    edge_columns: Sequence[str] = ()
    label_column: str | None = None

    def __post_init__(self) -> None:
        """Freeze feature selections as tuples and reject invalid names."""
        node_columns = _feature_columns(self.node_columns, "node_columns")
        edge_columns = _feature_columns(self.edge_columns, "edge_columns")
        if self.label_column is not None and not isinstance(self.label_column, str):
            raise TypeError("label_column must be a string or None")
        if self.label_column == "":
            raise ValueError("label_column must not be empty")
        object.__setattr__(self, "node_columns", node_columns)
        object.__setattr__(self, "edge_columns", edge_columns)


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

    Account graphs support all three modes because accounts have stable identity
    and each transaction is naturally an edge event. Transaction-as-node is a
    causal static graph: separate time windows may still be used for batching,
    but they are not a snapshot evolution of stable transaction nodes. A
    transaction-node stream would additionally require explicit node-arrival
    semantics and is not implemented.
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
        if self.temporal is GraphTemporalMode.EVENT_STREAM and (
            self.bin_size is not None or self.stride is not None
        ):
            raise ValueError("bin_size and stride require temporal='snapshot'")
        if self.temporal is GraphTemporalMode.SNAPSHOT and (
            self.bin_size is None or self.stride is None
        ):
            raise ValueError("snapshot preparation requires both bin_size and stride")
        if (
            self.node_type is GraphNodeType.TRANSACTION
            and self.temporal is not GraphTemporalMode.STATIC
        ):
            raise NotImplementedError(
                "transaction-as-node supports temporal='static'; transaction "
                "windows are a batching strategy, and event streams require "
                "explicit node-arrival semantics"
            )
        if self.node_type is GraphNodeType.TRANSACTION and self.edge_delta is None:
            raise ValueError("transaction graphs require edge_delta")
        if self.node_type is GraphNodeType.ACCOUNT and self.edge_delta is not None:
            raise ValueError("edge_delta applies only to transaction-node graphs")


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
    source_column: str | None = None,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    transaction_id_column: str | None = None,
    account_id_column: str | None = None,
) -> GraphBuildResult:
    """Prepare a graph using explicit node and temporal semantics.

    Example / 示例：

    ``prepare_graph(tx, node_type="transaction", temporal="static",\
    edge_delta=timedelta(hours=1))`` returns a ``TransactionGraph``.

    ``prepare_graph(tx, node_type="account", temporal="snapshot",\
    bin_size=timedelta(days=1), stride=timedelta(days=1))`` returns an iterator
    of account snapshots with stable account identities.

    The function is intentionally a thin dispatcher. It does not perform
    dataset loading, train/validation/test splitting, or backend conversion.
    / 本函数只是薄分发层，不负责数据集读取、train/validation/test 切分或后端
    转换。

    Account static and snapshot outputs preserve one edge row per transaction,
    including amount and all dataset-specific edge features. Account event
    streams preserve the same rows as time-ordered events.

    Raises:
        NotImplementedError: If transaction-as-node snapshot or event stream is
            requested.
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
        graph = build_time_aware_account_graph(
            transactions,
            account_metadata=account_metadata,
            source_column=source_column,
            target_column=target_column,
            timestamp_column=timestamp_column,
            transaction_id_column=transaction_id_column,
            account_id_column=account_id_column,
        )
        if spec.temporal is GraphTemporalMode.STATIC:
            return graph
        if spec.temporal is GraphTemporalMode.EVENT_STREAM:
            return AccountEventStream.from_graph(graph)

        from .temporal.snapshot import build_account_snapshots

        return build_account_snapshots(
            graph,
            bin_size=spec.bin_size,  # type: ignore[arg-type]
            stride=spec.stride,  # type: ignore[arg-type]
            start_time=spec.start_time,
            end_time=spec.end_time,
            drop_last=spec.drop_last,
        )

    graph = build_transaction_graph(
        transactions,
        delta=spec.edge_delta,  # type: ignore[arg-type]
        source_column=source_column,
        target_column=target_column,
        timestamp_column=timestamp_column,
        transaction_id_column=transaction_id_column,
    )
    return graph


def prepare_pyg_graph(
    transactions: TransactionTable,
    *,
    node_type: GraphNodeType | str,
    temporal: GraphTemporalMode | str,
    features: GraphFeatureSpec | None = None,
    edge_delta: timedelta | None = None,
    bin_size: timedelta | None = None,
    stride: timedelta | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    drop_last: bool = True,
    account_metadata: TransactionTable | None = None,
    source_column: str | None = None,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    transaction_id_column: str | None = None,
    account_id_column: str | None = None,
) -> PyGGraphBuildResult:
    """Build a model-ready PyG representation through one high-level call.

    Feature placement follows node semantics. For account-node graphs,
    ``features.node_columns`` selects account metadata, ``edge_columns`` selects
    transaction attributes, and the label becomes ``edge_y``. For
    transaction-node graphs, transaction attributes and labels become ``x``
    and ``node_y`` while relation attributes such as ``time_delta`` become
    ``edge_attr``. In account event streams, edge features become ``msg``.

    Returns:
        ``Data`` for static graphs, an iterator of ``Data`` for account
        snapshots, or ``TemporalData`` for account event streams.
    """
    if features is not None and not isinstance(features, GraphFeatureSpec):
        raise TypeError("features must be a GraphFeatureSpec or None")
    feature_spec = features or GraphFeatureSpec()
    graph = prepare_graph(
        transactions,
        node_type=node_type,
        temporal=temporal,
        edge_delta=edge_delta,
        bin_size=bin_size,
        stride=stride,
        start_time=start_time,
        end_time=end_time,
        drop_last=drop_last,
        account_metadata=account_metadata,
        source_column=source_column,
        target_column=target_column,
        timestamp_column=timestamp_column,
        transaction_id_column=transaction_id_column,
        account_id_column=account_id_column,
    )

    if isinstance(graph, AccountEventStream):
        return to_pyg_temporal_data(
            graph,
            node_feature_columns=feature_spec.node_columns,
            message_columns=feature_spec.edge_columns,
            label_column=feature_spec.label_column,
        )
    if isinstance(graph, AccountGraph):
        return _graph_to_pyg(graph, feature_spec)
    if isinstance(graph, TransactionGraph):
        return _graph_to_pyg(graph, feature_spec)
    return _account_snapshots_to_pyg(graph, feature_spec)


def _graph_to_pyg(
    graph: AccountGraph | TransactionGraph,
    features: GraphFeatureSpec,
) -> Data:
    """Convert one graph and place its label on the prediction entity."""
    label_arguments: dict[str, str] = {}
    if features.label_column is not None:
        label_key = (
            "edge_label_column"
            if isinstance(graph, AccountGraph)
            else "node_label_column"
        )
        label_arguments[label_key] = features.label_column
    return to_pyg_data(
        graph,
        node_feature_columns=features.node_columns,
        edge_feature_columns=features.edge_columns,
        **label_arguments,
    )


def _account_snapshots_to_pyg(
    snapshots: Iterator[GraphSnapshot],
    features: GraphFeatureSpec,
) -> Iterator[Data]:
    """Convert account snapshots lazily while preserving their time metadata."""
    for snapshot in snapshots:
        if not isinstance(snapshot.graph, AccountGraph):
            raise TypeError("high-level snapshots must contain account-node graphs")
        data = _graph_to_pyg(snapshot.graph, features)
        data.snapshot_index = snapshot.index
        data.start_time = snapshot.start_time
        data.end_time = snapshot.end_time
        yield data


def _feature_columns(columns: Sequence[str], name: str) -> tuple[str, ...]:
    """Normalize a user feature selection into a validated immutable tuple."""
    if isinstance(columns, str):
        raise TypeError(f"{name} must be a sequence of column names, not a string")
    normalized = tuple(columns)
    if any(not isinstance(column, str) or not column for column in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized


__all__ = [
    "GraphBuildResult",
    "GraphBuildSpec",
    "GraphFeatureSpec",
    "GraphNodeType",
    "GraphTemporalMode",
    "PyGGraphBuildResult",
    "prepare_graph",
    "prepare_pyg_graph",
]
