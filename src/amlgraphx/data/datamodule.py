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

Observed smoke run / 实际 smoke run：
    The following measurements used ``edge_delta=1 hour``, temporal cutoffs at
    60% and 80% of each dataset's timestamp span, ``window_size=7 days``,
    ``stride=7 days``, and ``drop_last=False``. Snapshot values are the first
    yielded snapshot in each partition; they are not totals for all windows.

    | dataset | full graph (N, E) | train split (N, E) | validation split (N, E) | test split (N, E) |
    |---|---:|---:|---:|---:|
    | PaySim | (6,362,620, 56) | (6,010,949, 56) | (228,091, 0) | (123,580, 0) |
    | IBM HI-Small | (5,078,345, 2,176,494) | (5,077,481, 2,176,448) | (741, 36) | (123, 8) |
    | IBM LI-Small | (6,924,049, 3,882,235) | (6,920,489, 3,882,032) | (3,514, 55) | (46, 6) |

    First snapshot ``(num_nodes, num_edges, edge_index.shape)``:

    | dataset | train | validation | test |
    |---|---:|---:|---:|
    | PaySim | (1,930,180, 15, (2, 15)) | (228,091, 0, (2, 0)) | (123,580, 0, (2, 0)) |
    | IBM HI-Small | (3,731,672, 1,318,328, (2, 1,318,328)) | (741, 36, (2, 36)) | (123, 8, (2, 8)) |
    | IBM LI-Small | (5,091,438, 2,304,938, (2, 2,304,938)) | (3,514, 55, (2, 55)) | (46, 6, (2, 6)) |

    以上结果使用 ``edge_delta=1 hour``、时间范围 60%/80% 位置作为切分点、
    ``window_size=7 days``、``stride=7 days``、``drop_last=False``。snapshot
    数字是每个分区第一个输出的 snapshot，不是全部窗口的总和。
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import torch
from torch import Tensor

from amlgraphx.graph.graphs import (
    AccountGraph,
    TransactionGraph,
    build_transaction_graph,
)

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

    Attributes / 属性：
        ``train``, ``validation``, and ``test`` are each
        ``TransactionGraph`` objects. For each graph, ``nodes`` is a Polars
        table shaped ``(N_split, C)`` and ``edges`` is shaped ``(E_split, 4)``.
        Cross-partition edges are removed.

        ``train``、``validation`` 和 ``test`` 都是 ``TransactionGraph``。每个
        图的 ``nodes`` 是 shape ``(N_split, C)`` 的 Polars 表，``edges`` 是
        shape ``(E_split, 4)`` 的边表；跨分区的边会被删除。

    Smoke-run values / 实际运行值：
        With the documented seven-day smoke configuration, the split sizes
        were PaySim ``(6,010,949,56)/(228,091,0)/(123,580,0)``, IBM HI-Small
        ``(5,077,481,2,176,448)/(741,36)/(123,8)``, and IBM LI-Small
        ``(6,920,489,3,882,032)/(3,514,55)/(46,6)`` for
        train/validation/test ``(nodes, edges)``.

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
    """Represent one temporal graph window with sparse local edges.

    English: ``graph.nodes`` and ``graph.edges`` preserve the Polars attributes
    of either account-node or transaction-node graphs. ``edge_index`` stores
    local integer endpoints for PyTorch/PyG-style message passing. Stable IDs
    remain in the tables; the tensor uses positions only inside this snapshot.

    中文：``graph.nodes`` 和 ``graph.edges`` 保留账户图或交易图的 Polars
    属性，``edge_index`` 保存适合 PyTorch/PyG message passing 的局部整数端点。
    稳定 ID 保留在表中，tensor 只在当前 snapshot 内使用行位置。

    Output shape / 输出尺寸：
        ``graph.nodes`` has shape ``(N_snapshot, C_node)`` and
        ``graph.edges`` has shape ``(E_snapshot, C_edge)``. ``edge_index`` is a
        contiguous ``torch.long`` tensor with shape ``(2, E_snapshot)``; its
        values are in ``[0, N_snapshot)``. An empty snapshot graph uses shape
        ``(2, 0)``.

        ``graph.nodes`` 的 shape 是 ``(N_snapshot, C_node)``，``graph.edges``
        的 shape 是 ``(E_snapshot, C_edge)``。``edge_index`` 是连续的 ``torch.long``
        tensor，shape 为 ``(2, E_snapshot)``，其中的值位于
        ``[0, N_snapshot)``；空边图使用 ``(2, 0)``。

    Smoke-run examples / 实际运行示例：
        Under the documented configuration, the first training snapshots had
        ``(N_snapshot, E_snapshot, edge_index.shape)`` of PaySim
        ``(1,930,180, 15, (2,15))``, IBM HI-Small
        ``(3,731,672, 1,318,328, (2,1,318,328))``, and IBM LI-Small
        ``(5,091,438, 2,304,938, (2,2,304,938))``.

    ``edge_index`` uses local node positions and has shape ``[2, num_edges]``.
    No dense adjacency matrix is materialized.

    Args:
        graph: Account or transaction graph selected for the window.
        edge_index: Sparse COO-style source and target node indices.
        start_time: Inclusive beginning of the window.
        end_time: Exclusive end of the window.
        index: One-based snapshot number within its partition.
        target_mask: Optional local boolean mask for prediction targets. A
            ``None`` value means every node is a target.
    """

    graph: AccountGraph | TransactionGraph
    edge_index: Tensor
    start_time: datetime
    end_time: datetime
    index: int
    target_mask: Tensor | None = None

    @property
    def num_nodes(self) -> int:
        """Return snapshot node count / 返回 snapshot 中的节点数。"""
        return self.graph.num_nodes

    @property
    def num_edges(self) -> int:
        """Return sparse edge count / 返回 snapshot 中的稀疏边数量。"""
        return self.edge_index.shape[1]

    @property
    def num_target_nodes(self) -> int:
        """Return prediction-target count / 返回预测目标节点数量。"""
        return (
            self.num_nodes if self.target_mask is None else int(self.target_mask.sum())
        )


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
        A ``TransactionGraphSplit`` containing three materialized
        ``TransactionGraph`` objects. Each object's node table contains only
        transactions in its half-open time interval, and its edge table keeps
        only edges whose two endpoints are in that same interval.

        返回 ``TransactionGraphSplit``，包含三个已经物化的
        ``TransactionGraph``。每个节点表只包含对应半开时间区间内的交易，
        每个边表只保留两个端点都属于该区间的边。

        Shapes / shape：for the three outputs, ``train.nodes.shape`` is
        ``(N_train, C)``, ``validation.nodes.shape`` is ``(N_validation, C)``,
        and ``test.nodes.shape`` is ``(N_test, C)``. Every edge table has shape
        ``(E_partition, 4)``.

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
    context_size: timedelta | None = None,
) -> Iterator[GraphSnapshot]:
    """Yield sparse graph snapshots over half-open time windows.

    English example: with ``window_size=7 days`` and ``stride=3 days``, starts
    at day 0 produce ``[0, 7)``, ``[3, 10)``, ``[6, 13)``, and so on. The stride
    controls movement; it does not change the graph edge semantics.

    中文例子：``window_size=7 天``、``stride=3 天`` 时，窗口依次是
    ``[0, 7)``、``[3, 10)``、``[6, 13)``。stride 控制窗口起点移动距离，
    不会改变图边的定义。

    ``context_size`` optionally prepends historical nodes to each snapshot.
    ``target_mask`` then marks only nodes in ``[start_time, end_time)`` as
    prediction targets, preserving causal context without scoring it.

    Empty target windows are skipped. By default, a final window that crosses
    ``end_time`` is omitted. Set ``drop_last=False`` to emit that final window
    with its end clipped to the requested range.

    Args:
        graph: Transaction graph to sample.
        window_size: Positive duration represented by each full snapshot.
        stride: Positive distance between consecutive window starts.
        start_time: Optional inclusive first window start.
        end_time: Optional exclusive sampling boundary.
        drop_last: Whether to omit a final incomplete window.
        context_size: Optional non-negative history retained before each window.

    Yields:
        A lazy ``Iterator[GraphSnapshot]``. Each yielded snapshot contains a
        node table shaped ``(N_window, C)``, an edge table shaped
        ``(E_window, 4)``, and ``edge_index`` shaped ``(2, E_window)`` with
        dtype ``torch.long``. Empty windows are not yielded.

        返回惰性的 ``Iterator[GraphSnapshot]``。每个 snapshot 包含 shape 为
        ``(N_window, C)`` 的节点表、shape 为 ``(E_window, 4)`` 的边表，以及
        dtype 为 ``torch.long``、shape 为 ``(2, E_window)`` 的 ``edge_index``。
        空窗口不会被 yield。

    Raises:
        TypeError: If durations or explicit boundaries have invalid types.
        ValueError: If durations are not positive or boundaries are unordered.
    """
    _validate_duration(window_size, "window_size")
    _validate_duration(stride, "stride")
    if context_size is not None:
        _validate_duration(context_size, "context_size", allow_zero=True)
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
            context_size=context_size,
        )
        if snapshot.num_target_nodes > 0:
            yield snapshot
            snapshot_index += 1
        window_start += stride


class TransactionGraphDataModule:
    """Prepare temporal splits and sliding snapshots for transaction GNNs.

    English pipeline:
        1. Build one full transaction graph.
        2. Split its transaction nodes by timestamp.
        3. Sample targets by split while retaining ``edge_delta`` history for
           validation and test windows.

    中文流程：
        1. 先构建一张完整交易图。
        2. 按交易节点时间划分数据集。
        3. 按分区采样目标节点，并为验证/测试窗口保留 ``edge_delta`` 历史。

    ``setup`` is intentionally explicit because graph construction materializes
    data and can be expensive. / ``setup`` 需要显式调用，因为建图会物化数据，
    可能是昂贵操作。

    The module builds one complete transaction graph first. It retains strict
    induced subgraphs in ``splits`` for that protocol, while validation and
    test snapshots draw their ``edge_delta`` lookback context from the full
    graph and expose only in-window nodes through ``target_mask``. Call
    ``setup()`` before requesting graphs or snapshots.

    Input / 输入：
        ``transactions`` is a canonical transaction ``DataFrame`` or
        ``LazyFrame``. It must provide ``transaction_id`` and ``timestamp``
        after graph preparation. The constructor stores configuration but does
        not build or collect the graph.

        ``transactions`` 是统一交易 ``DataFrame`` 或 ``LazyFrame``，建图后必须
        能提供 ``transaction_id`` 和 ``timestamp``。构造函数只保存配置，不会
        立即建图或 collect。

    State and output / 状态与输出：
        After ``setup()``, ``full_graph`` is a ``TransactionGraph``, ``splits``
        is a ``TransactionGraphSplit``, and the three snapshot methods return
        lazy iterators. The full graph has ``nodes.shape == (N, C)`` and
        ``edges.shape == (E, 4)``; each snapshot has ``edge_index.shape ==
        (2, E_snapshot)``.

        ``setup()`` 后，``full_graph`` 是 ``TransactionGraph``，``splits`` 是
        ``TransactionGraphSplit``，三个 snapshot 方法返回惰性 iterator。完整
        图的节点/边 shape 分别为 ``(N, C)``、``(E, 4)``；每个 snapshot 的
        ``edge_index.shape == (2, E_snapshot)``。

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
        """Store graph-splitting and snapshot configuration.

        Returns / 返回：
            ``None``. The constructor only validates and stores the input; the
            expensive graph build happens in ``setup()``. Before setup,
            ``full_graph`` and ``splits`` are unavailable and raise
            ``RuntimeError`` when accessed.

            返回 ``None``。构造函数只校验并保存输入，昂贵的建图在 ``setup()``
            中执行。setup 之前访问 ``full_graph`` 或 ``splits`` 会抛出
            ``RuntimeError``。
        """
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

        Returns / 返回：
            ``None``. The method populates the private caches used by
            ``full_graph``, ``splits``, and the snapshot iterators. Repeated
            calls are idempotent and do not rebuild the graph.

            返回 ``None``。方法会填充 ``full_graph``、``splits`` 和 snapshot
            iterator 使用的内部缓存；重复调用不会重新建图。
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
        """Return the complete graph created by ``setup()``.

        Returns / 返回：
            A materialized ``TransactionGraph`` with ``nodes.shape == (N, C)``
            and ``edges.shape == (E, 4)``. Raises ``RuntimeError`` before
            ``setup()``.

            返回已物化的 ``TransactionGraph``，节点和边 shape 分别为
            ``(N, C)`` 和 ``(E, 4)``。在 ``setup()`` 之前访问会抛出
            ``RuntimeError``。
        """
        if self._full_graph is None:
            raise RuntimeError("Call setup() before accessing full_graph")
        return self._full_graph

    @property
    def splits(self) -> TransactionGraphSplit:
        """Return temporal partitions created by ``setup()``.

        Returns / 返回：
            A ``TransactionGraphSplit`` whose ``train``, ``validation``, and
            ``test`` fields are materialized transaction graphs. Raises
            ``RuntimeError`` before ``setup()``.

            返回 ``TransactionGraphSplit``，其中 ``train``、``validation`` 和
            ``test`` 都是已经物化的交易图。在 ``setup()`` 之前访问会抛出
            ``RuntimeError``。
        """
        if self._splits is None:
            raise RuntimeError("Call setup() before accessing splits")
        return self._splits

    def train_snapshots(self) -> Iterator[GraphSnapshot]:
        """Return a lazy iterator over training snapshots.

        Returns / 返回：
            ``Iterator[GraphSnapshot]``. The iterator yields only non-empty
            target windows inside ``[-inf, train_end)``. Each snapshot retains
            ``edge_delta`` history and marks training targets with
            ``target_mask``.

            返回 ``Iterator[GraphSnapshot]``，只 yield 完全位于
            ``[-inf, train_end)`` 且非空的窗口；每个 tensor 的 shape 是
            ``(2, E_window)``。
        """
        return sliding_snapshots(
            self.splits.train,
            window_size=self.window_size,
            stride=self.stride,
            end_time=self.train_end,
            drop_last=self.drop_last,
            context_size=self.edge_delta,
        )

    def validation_snapshots(self) -> Iterator[GraphSnapshot]:
        """Return a lazy iterator over validation snapshots.

        Returns / 返回：
            ``Iterator[GraphSnapshot]`` for the half-open interval
            ``[train_end, validation_end)``. Each yielded snapshot includes
            ``edge_delta`` history and marks only validation nodes with
            ``target_mask``.

            返回验证区间 ``[train_end, validation_end)`` 的
            ``Iterator[GraphSnapshot]``；每个 snapshot 的稀疏
            ``edge_index`` shape 为 ``(2, E_window)``。
        """
        return sliding_snapshots(
            self.full_graph,
            window_size=self.window_size,
            stride=self.stride,
            start_time=self.train_end,
            end_time=self.validation_end,
            drop_last=self.drop_last,
            context_size=self.edge_delta,
        )

    def test_snapshots(self) -> Iterator[GraphSnapshot]:
        """Return a lazy iterator over test snapshots.

        Returns / 返回：
            ``Iterator[GraphSnapshot]`` for
            ``[validation_end, test_end)`` or the graph's inferred end when
            ``test_end`` is ``None``. Each snapshot includes ``edge_delta``
            history and marks only test nodes with ``target_mask``.

            返回测试区间 ``[validation_end, test_end)`` 的
            ``Iterator[GraphSnapshot]``；``test_end`` 为 ``None`` 时使用图的
            推断终点。每个 tensor 的 shape 为 ``(2, E_window)``。
        """
        return sliding_snapshots(
            self.full_graph,
            window_size=self.window_size,
            stride=self.stride,
            start_time=self.validation_end,
            end_time=self.test_end,
            drop_last=self.drop_last,
            context_size=self.edge_delta,
        )


def _build_snapshot(
    graph: TransactionGraph,
    *,
    start_time: datetime,
    end_time: datetime,
    index: int,
    context_size: timedelta | None = None,
) -> GraphSnapshot:
    """Select one window and convert its graph endpoints to local indices.

    中文：先选择窗口及其可选历史 context 节点，再构造诱导子图，最后把稳定的
    交易 ID 映射为当前 snapshot 内的连续整数索引。

    Returns / 返回：
        A ``GraphSnapshot`` whose node table has shape ``(N_window, C)``, edge
        table has shape ``(E_window, 4)``, and ``edge_index`` has shape
        ``(2, E_window)`` with dtype ``torch.long``. ``index`` is preserved as
        the snapshot's one-based identifier.

        返回 ``GraphSnapshot``：节点表 shape 为 ``(N_window, C)``，边表 shape
        为 ``(E_window, 4)``，``edge_index`` shape 为 ``(2, E_window)`` 且
        dtype 为 ``torch.long``；``index`` 会作为 snapshot 的从 1 开始编号保留。
    """
    timestamp = pl.col("timestamp")
    context_start = (
        start_time - context_size if context_size is not None else start_time
    )
    nodes = graph.nodes.filter((timestamp >= context_start) & (timestamp < end_time))
    snapshot_graph = _induced_graph(graph, nodes)
    target_mask = torch.from_numpy((nodes["timestamp"] >= start_time).to_numpy()).to(
        dtype=torch.bool
    )
    return GraphSnapshot(
        graph=snapshot_graph,
        edge_index=_edge_index(snapshot_graph),
        start_time=start_time,
        end_time=end_time,
        index=index,
        target_mask=target_mask,
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

    Returns / 返回：
        A ``TransactionGraph`` with exactly ``nodes`` as its node table and a
        filtered edge table. The node shape is ``(nodes.height, C)``; the edge
        shape is ``(E_induced, 4)``. If ``nodes`` is empty, the edge shape is
        ``(0, 4)`` and the original edge schema is preserved.

        返回 ``TransactionGraph``，节点表就是输入的 ``nodes``，边表是筛选后
        的边。节点 shape 为 ``(nodes.height, C)``，边 shape 为
        ``(E_induced, 4)``。当 ``nodes`` 为空时，边 shape 为 ``(0, 4)``，并
        保留原始边 schema。
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

    Returns / 返回：
        A contiguous ``torch.Tensor`` with dtype ``torch.long`` and shape
        ``(2, graph.num_edges)``. Row 0 contains source node positions and row
        1 contains target node positions. For an empty edge table it returns
        ``torch.empty((2, 0), dtype=torch.long)``.

        返回连续的 ``torch.Tensor``，dtype 为 ``torch.long``，shape 为
        ``(2, graph.num_edges)``。第 0 行是 source 节点位置，第 1 行是 target
        节点位置。空边表返回 ``torch.empty((2, 0), dtype=torch.long)``。
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
    """Resolve optional sampling bounds from explicit values or node times.

    Returns / 返回：
        ``None`` when ``graph.nodes`` is empty; otherwise a tuple
        ``(resolved_start, resolved_end)`` of Python ``datetime`` objects.
        Explicit bounds take precedence. An inferred end is one microsecond
        after the maximum node timestamp so the maximum timestamp is included
        by half-open filtering.

        图为空时返回 ``None``；否则返回由两个 Python ``datetime`` 组成的
        ``(resolved_start, resolved_end)``。显式边界优先；推断终点是最大节点
        时间加一微秒，从而让半开区间筛选包含最大时间戳。
    """
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
    """Validate the graph columns required by temporal sampling.

    Returns / 返回：
        ``None`` when ``graph`` is a ``TransactionGraph`` whose node table
        contains ``transaction_id`` and ``timestamp``. Otherwise it raises
        ``TypeError`` or ``ValueError`` and returns nothing.

        当 ``graph`` 是节点表包含 ``transaction_id``、``timestamp`` 的
        ``TransactionGraph`` 时返回 ``None``；否则抛出 ``TypeError`` 或
        ``ValueError``。
    """
    if not isinstance(graph, TransactionGraph):
        raise TypeError("graph must be a TransactionGraph")
    required = {"transaction_id", "timestamp"}
    missing = required.difference(graph.nodes.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"Transaction graph nodes are missing: {columns}")


def _validate_cutoffs(train_end: datetime, validation_end: datetime) -> None:
    """Validate chronological train/validation cutoffs.

    Returns / 返回：
        ``None`` for two ordered Python ``datetime`` values. Invalid types or
        ``train_end >= validation_end`` raise an exception.

        两个有序的 Python ``datetime`` 合法时返回 ``None``；类型错误或
        ``train_end >= validation_end`` 时抛出异常。
    """
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
    """Validate a duration used by graph construction or window sampling.

    Returns / 返回：
        ``None`` when ``value`` is a valid ``timedelta``. By default it must be
        positive; ``allow_zero=True`` permits zero for ``edge_delta``. Invalid
        types or negative values raise an exception.

        当 ``value`` 是合法 ``timedelta`` 时返回 ``None``。默认要求正数；
        ``allow_zero=True`` 时允许 ``edge_delta`` 为零。类型错误或负数会抛出
        异常。
    """
    if not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a datetime.timedelta")
    minimum = timedelta(0)
    if value < minimum or (value == minimum and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
