"""Temporal graph splitting and sparse sliding-window snapshots.

English:
    The data module follows the current AMLGraphX experiment route:
    canonical transactions -> one complete transaction graph -> chronological
    train/validation/test partitions -> independent sliding-window snapshots.
    Splits are made before snapshots are sampled, so a snapshot never crosses a
    split boundary.

中文：
    本模块遵循 AMLGraphX 当前的实验路线：统一交易表 -> 完整交易图 -> 按时间
    划分 train/validation/test -> 在各自区间内独立滑动采样 snapshot。先切分
    再采样可以保证一个 snapshot 不会跨越数据集边界。

Polars quick guide / Polars 快速提示：
    ``pl.col("timestamp")`` creates a column expression. It can be combined
    with ``&`` and ``|`` and passed to ``filter``; for example,
    ``frame.filter((pl.col("amount") > 0) & pl.col("label").is_not_null())``.
    ``select`` creates a smaller table, ``join`` matches rows by keys, and a
    ``semi`` join keeps left rows whose key exists on the right without adding
    right-hand columns. These operations are used below to keep only nodes and
    edges that belong to one temporal window.

    ``pl.col("timestamp")`` 会构造列表达式，可以用 ``&``、``|`` 组合后交给
    ``filter``；例如 ``frame.filter((pl.col("amount") > 0) &
    pl.col("label").is_not_null())``。``select`` 创建更小的表，``join`` 按键
    匹配行，而 ``semi`` join 只保留左表中“右表存在相同键”的行，不添加右表
    的列。下面正是用这些操作保留某个时间窗口内的节点和边。
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import torch
from torch import Tensor

from amlgraphx.graph import TransactionGraph, build_transaction_graph

type TransactionTable = pl.DataFrame | pl.LazyFrame

# English: This module currently samples transaction-node graphs. Account-node
# snapshots will need a different temporal edge-to-node construction path.
# 中文：当前模块采样“交易为节点”的图；账户为节点的 snapshot 需要另一条
# 基于交易边时间的构建路径，不能简单复用这里的节点时间筛选。


@dataclass(frozen=True, slots=True)
class TransactionGraphSplit:
    """Hold leakage-free temporal partitions of a transaction graph.

    English: The three fields are induced subgraphs. An edge survives only if
    both endpoint transactions are in the same partition.

    中文：三个字段都是诱导子图；只有当一条边的两个交易节点属于同一时间区间
    时，这条边才会被保留。

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

    English: ``graph.nodes`` stores the transaction attributes and
    ``edge_index`` stores local integer endpoints for PyTorch/PyG-style
    message passing. The node IDs in the Polars table remain the stable
    transaction IDs; the tensor uses row positions only inside this snapshot.

    中文：``graph.nodes`` 保存交易属性，``edge_index`` 保存适合
    PyTorch/PyG message passing 的局部整数端点。Polars 表中仍保留稳定的交易
    ID；tensor 只在当前 snapshot 内使用行号。

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
        """Return snapshot node count / 返回 snapshot 中的交易节点数。"""
        return self.graph.num_nodes

    @property
    def num_edges(self) -> int:
        """Return sparse edge count / 返回 snapshot 中的稀疏边数量。"""
        return self.edge_index.shape[1]


def split_transaction_graph(
    graph: TransactionGraph,
    *,
    train_end: datetime,
    validation_end: datetime,
) -> TransactionGraphSplit:
    """Split a transaction graph into chronological induced subgraphs.

    English: The timestamp belongs to a transaction node. We first select node
    rows for each half-open interval, then call ``_induced_graph`` so no edge
    can leak a transaction from another partition.

    中文：这里的时间戳属于交易节点。先按半开区间筛选节点，再调用
    ``_induced_graph``，因此不会保留连接到其他时间分区的边。

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

    # A Polars expression is reusable in several filters; it is evaluated only
    # when ``filter`` runs. / Polars 表达式可以复用，交给 ``filter`` 时才执行。
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

    English example: with ``window_size=7 days`` and ``stride=3 days``, starts
    at day 0 produce ``[0, 7)``, ``[3, 10)``, ``[6, 13)``, and so on. The stride
    controls movement; it does not change the graph edge semantics.

    中文例子：``window_size=7 天``、``stride=3 天`` 时，窗口依次是
    ``[0, 7)``、``[3, 10)``、``[6, 13)``。stride 控制窗口起点移动距离，
    不会改变图边的定义。

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

    English pipeline:
        1. Build one full transaction graph.
        2. Split its transaction nodes by timestamp.
        3. Sample each split independently with the same window/stride policy.

    中文流程：
        1. 先构建一张完整交易图。
        2. 按交易节点时间划分数据集。
        3. 在每个分区内使用同一组 window/stride 独立采样。

    ``setup`` is intentionally explicit because graph construction materializes
    data and can be expensive. / ``setup`` 需要显式调用，因为建图会物化数据，
    可能是昂贵操作。

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
        """Build the full graph and chronological partitions once.

        中文：只在第一次调用时建图和切分；重复调用直接复用缓存，避免重复扫描
        大型交易表。
        """
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
        """Return the complete graph created by ``setup()`` / 返回完整交易图。"""
        if self._full_graph is None:
            raise RuntimeError("Call setup() before accessing full_graph")
        return self._full_graph

    @property
    def splits(self) -> TransactionGraphSplit:
        """Return temporal partitions created by ``setup()`` / 返回时间分区。"""
        if self._splits is None:
            raise RuntimeError("Call setup() before accessing splits")
        return self._splits

    def train_snapshots(self) -> Iterator[GraphSnapshot]:
        """Yield snapshots fully inside training / 生成完全位于训练区间的 snapshot。"""
        return sliding_snapshots(
            self.splits.train,
            window_size=self.window_size,
            stride=self.stride,
            end_time=self.train_end,
            drop_last=self.drop_last,
        )

    def validation_snapshots(self) -> Iterator[GraphSnapshot]:
        """Yield validation snapshots / 生成完全位于验证区间的 snapshot。"""
        return sliding_snapshots(
            self.splits.validation,
            window_size=self.window_size,
            stride=self.stride,
            start_time=self.train_end,
            end_time=self.validation_end,
            drop_last=self.drop_last,
        )

    def test_snapshots(self) -> Iterator[GraphSnapshot]:
        """Yield test snapshots / 生成完全位于测试区间的 snapshot。"""
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
    """Select one window and convert its graph endpoints to local indices.

    中文：先用半开区间选择窗口节点，再构造诱导子图，最后把稳定的交易 ID
    映射为当前 snapshot 内的连续整数索引。
    """
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
    """Keep only edges whose two transaction endpoints are selected.

    Polars / Polars：
        ``select`` creates one-column key tables. A ``semi`` join behaves like
        an existence filter: ``left.join(right, on="id", how="semi")`` keeps
        left rows whose ``id`` appears in right, but does not copy right columns.
        We apply it once to sources and once to targets, which is exactly the
        definition of an induced subgraph.

    中文：
        ``select`` 创建只包含键的一列表。``semi`` join 相当于存在性筛选：
        ``left.join(right, on="id", how="semi")`` 只保留左表中 id 出现在右表
        的行，也不会复制右表列。这里分别对 source 和 target 做一次，因此
        得到严格的诱导子图。
    """
    if nodes.is_empty():
        # ``head(0)`` preserves the edge schema while returning zero rows.
        # ``head(0)`` 保留边表 schema，但返回 0 行，方便处理空窗口。
        return TransactionGraph(nodes=nodes, edges=graph.edges.head(0))

    # These one-column tables are membership sets for the two endpoint IDs.
    # 这两张单列表就是 source/target 端点 ID 的成员集合。
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
    """Map transaction IDs to local PyTorch edge indices.

    English: ``with_row_index`` gives each node its position in the current
    snapshot. Two inner joins replace source/target transaction IDs in the
    edge table with those positions. The result is transposed to PyG's
    ``[2, num_edges]`` convention.

    中文：``with_row_index`` 为当前 snapshot 的每个节点生成行位置。两次
    inner join 把边表中的 source/target 交易 ID 换成节点位置，最后转置成 PyG
    使用的 ``[2, num_edges]`` 格式。

    No dense adjacency matrix is created. / 全程不创建 dense adjacency matrix。
    """
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
    """Resolve optional sampling bounds from explicit values or node times."""
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
    """Validate the graph shape required by temporal sampling."""
    if not isinstance(graph, TransactionGraph):
        raise TypeError("graph must be a TransactionGraph")
    required = {"transaction_id", "timestamp"}
    missing = required.difference(graph.nodes.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Transaction graph nodes are missing: {columns}")


def _validate_cutoffs(train_end: datetime, validation_end: datetime) -> None:
    """Validate chronological train/validation cutoffs."""
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
    """Validate a duration used by graph construction or window sampling."""
    if not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a datetime.timedelta")
    minimum = timedelta(0)
    if value < minimum or (value == minimum and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
