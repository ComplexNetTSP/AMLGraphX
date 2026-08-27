"""Graph views built from AMLGraphX transaction tables.

English:
    This module provides two different graph semantics. ``AccountGraph`` uses
    accounts as nodes and transactions as directed edges. ``TransactionGraph``
    uses transactions as nodes and connects a transaction to later feasible
    continuations through the receiver account. They are intentionally
    separate because the node meaning changes the downstream learning task.

中文：
    本模块提供两种不同的图语义。``AccountGraph`` 把账户作为节点、交易作为
    有向边；``TransactionGraph`` 把交易作为节点，并通过收款账户连接到之后
    可能继续流转的交易。两者没有混在一起，因为节点的含义会直接决定下游
    GNN 的任务和标签定义。

Polars quick guide / Polars 快速提示：
    Most Polars calls below build expressions first and apply them to a table.
    For example, ``pl.col("amount") > 0`` means "the amount column is greater
    than zero"; it is not a Python scalar. The expression can then be used as
    ``frame.filter(pl.col("amount") > 0)``. ``with_columns`` adds or replaces
    columns, ``select`` chooses columns, and ``join`` combines tables by keys.
    Example: ``frame.with_columns(pl.col("sender").cast(pl.String))``.

    下面的大多数 Polars 调用都是“先构造表达式，再让表执行表达式”。例如
    ``pl.col("amount") > 0`` 表示“amount 列大于 0”，不是一个 Python 标量；
    它可以交给 ``frame.filter(...)`` 筛选行。``with_columns`` 用来新增或替换
    列，``select`` 用来选择列，``join`` 用键合并表。例如：
    ``frame.with_columns(pl.col("sender").cast(pl.String))``。

Observed smoke-run sizes / 实际 smoke run 尺寸：
    The following numbers came from ``dataset.transactions()`` followed by
    ``build_account_graph`` and a transaction graph with ``delta=1 hour``.
    They document the current data flow; they are not hard-coded limits.

    | dataset | input rows x columns | account nodes x edges | transaction nodes x edges |
    |---|---:|---:|---:|
    | PaySim | 6,362,620 x 16 | 9,073,900 x 6,362,620 | 6,362,620 x 56 |
    | IBM HI-Small | 5,078,345 x 18 | 515,080 x 5,078,345 | 5,078,345 x 2,176,494 |
    | IBM LI-Small | 6,924,049 x 18 | 705,903 x 6,924,049 | 6,924,049 x 3,882,235 |

    实际运行结果来自 ``dataset.transactions()``、``build_account_graph``，以及
    ``delta=1 hour`` 的交易图构建。这些数字用于说明当前数据流，不是代码限制。
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl
from numpy.typing import NDArray

from amlgraphx._native._core import temporal_edge_indices as _temporal_edge_indices

type TransactionTable = pl.DataFrame | pl.LazyFrame

# English: Accept eager and lazy Polars tables at the public boundary.
# 中文：公共 API 同时接受已经计算的 DataFrame 和延迟执行的 LazyFrame。

_SOURCE_ALIASES = (
    "source",
    "sender",
    "sender account",
    "from",
    "from account",
    "account",
    "source id",
    "src",
    "nameorig",
    "origin account",
    "origin",
)
_TARGET_ALIASES = (
    "target",
    "receiver",
    "receiver account",
    "to",
    "to account",
    "account.1",
    "account 1",
    "account duplicated 0",
    "target id",
    "dst",
    "namedest",
    "destination account",
    "destination",
)
_TIMESTAMP_ALIASES = ("timestamp", "datetime", "time", "date")
_TRANSACTION_ID_ALIASES = (
    "transaction_id",
    "transaction id",
    "tx id",
    "id",
)
_ACCOUNT_ID_ALIASES = (
    "node_id",
    "account number",
    "account",
    "account id",
    "account_id",
)

_ROW_INDEX = "__amlgraphx_row_index"
_TIMESTAMP_NS = "__amlgraphx_timestamp_ns"

# These names are temporary implementation columns and must not collide with
# user data. / 下面两个名称是内部临时列，不能与用户输入数据重名。


@dataclass(frozen=True, slots=True)
class AccountGraph:
    """Represent accounts as nodes and transactions as directed edges.

    English:
        Every input transaction remains an edge, including repeated transfers
        between the same pair of accounts. ``nodes`` contains one row per
        account and can optionally be enriched with account metadata.

    中文：
        每一笔输入交易都会保留为一条边，即使同一对账户之间发生多次转账。
        ``nodes`` 每个账户只有一行，也可以通过账户元数据表补充节点属性。

    Output shape / 输出尺寸：
        ``nodes`` is a Polars ``DataFrame`` with shape
        ``(number_of_distinct_accounts, number_of_node_columns)``. Without
        metadata its only column is ``node_id``. ``edges`` is a Polars
        ``DataFrame`` with shape ``(number_of_valid_transactions,
        number_of_transaction_columns)``; every valid transaction contributes
        one row, including repeated transfers.

        ``nodes`` 是 shape 为
        ``(不同账户数量, 节点字段数量)`` 的 Polars ``DataFrame``。没有元数据
        时只有 ``node_id`` 一列。``edges`` 是 shape 为
        ``(有效交易数量, 交易字段数量)`` 的 Polars ``DataFrame``；每一笔有效
        交易保留一行，包括重复转账。

    Smoke-run examples / 实际运行示例：
        With the dataset transaction ``LazyFrame`` as input, the observed
        ``(nodes.height, edges.height)`` values were PaySim ``(9,073,900,
        6,362,620)``, IBM HI-Small ``(515,080, 5,078,345)``, and IBM LI-Small
        ``(705,903, 6,924,049)``.

        使用数据集交易 ``LazyFrame`` 时，实际观察到的
        ``(nodes.height, edges.height)`` 分别为 PaySim ``(9,073,900,
        6,362,620)``、IBM HI-Small ``(515,080, 5,078,345)`` 和 IBM LI-Small
        ``(705,903, 6,924,049)``。

    Args:
        nodes: Account table containing at least ``node_id``.
        edges: Transaction table containing ``source`` and ``target``.
    """

    nodes: pl.DataFrame
    edges: pl.DataFrame

    @property
    def num_nodes(self) -> int:
        """Return ``nodes.height`` as a Python ``int`` / 返回账户节点数量。"""
        return self.nodes.height

    @property
    def num_edges(self) -> int:
        """Return ``edges.height`` as a Python ``int`` / 返回交易边数量。"""
        return self.edges.height

    @classmethod
    def from_transactions(
        cls,
        transactions: TransactionTable,
        *,
        account_metadata: TransactionTable | None = None,
        source_column: str | None = None,
        target_column: str | None = None,
        transaction_id_column: str | None = None,
        account_id_column: str | None = None,
    ) -> "AccountGraph":
        """Build an account graph from a transaction table.

        English:
            The preparation step resolves column aliases, strips account-ID
            whitespace, and creates a stable ``transaction_id``. The node
            table is then made by vertically concatenating all sources and
            targets, deduplicating them, and sorting by ``node_id``.

        中文：
            准备阶段会解析字段别名、清理账户 ID 两侧空白，并生成稳定的
            ``transaction_id``。随后把所有 source 和 target 纵向拼接，去重后
            按 ``node_id`` 排序，得到账户节点表。

        Input / 输入：
            ``transactions`` must be a Polars ``DataFrame`` or ``LazyFrame``.
            It needs one source-like and one target-like column. The optional
            metadata table needs an account-ID column.

            ``transactions`` 必须是 Polars ``DataFrame`` 或 ``LazyFrame``，并且
            至少包含一列 source 别名和一列 target 别名。可选的 metadata 表需要
            一列账户 ID。

        Returns / 返回：
            An ``AccountGraph`` with ``nodes`` shaped as
            ``(unique_account_count, node_column_count)`` and ``edges`` shaped
            as ``(valid_transaction_count, prepared_transaction_column_count)``.
            The returned object is materialized; it is not a ``LazyFrame``.

            返回 ``AccountGraph``；``nodes`` 的 shape 是
            ``(唯一账户数, 节点列数)``，``edges`` 的 shape 是
            ``(有效交易数, 处理后交易列数)``。返回对象已经物化，不是
            ``LazyFrame``。

        Args:
            transactions: Transaction rows as a Polars DataFrame or LazyFrame.
            account_metadata: Optional account table to join onto ``nodes``.
            source_column: Optional sender column override.
            target_column: Optional receiver column override.
            transaction_id_column: Optional transaction ID column override.
            account_id_column: Optional account metadata ID column override.

        Returns:
            An account graph. Repeated transactions remain separate edges.

        Raises:
            ValueError: If source or target columns are missing.
            TypeError: If the input is not a Polars table.
        """
        # Normalize the two endpoint columns before building either table.
        # 先统一两端账户字段，再构建 edges 和 nodes。
        frame, source, target = _prepare_transactions(
            transactions,
            source_column=source_column,
            target_column=target_column,
            transaction_id_column=transaction_id_column,
        )
        return _account_graph_from_prepared(
            frame,
            source=source,
            target=target,
            account_metadata=account_metadata,
            account_id_column=account_id_column,
        )


@dataclass(frozen=True, slots=True)
class TransactionGraph:
    """Represent transactions as nodes linked by temporal money flow.

    English:
        For an earlier transaction ``A -> B`` and a later transaction
        ``B -> C``, an edge is created from the node for the first transaction
        to the node for the second one when the time gap is within ``delta``.
        Transactions with equal timestamps are not ordered and therefore are
        not linked by this implementation.

    中文：
        对于较早的 ``A -> B`` 和较晚的 ``B -> C``，如果两笔交易的时间差不
        超过 ``delta``，就从第一笔交易节点连到第二笔交易节点。时间戳相同
        的交易无法确定先后顺序，因此本实现不会把它们连接起来。

    Output shape / 输出尺寸：
        ``nodes`` is a Polars ``DataFrame`` with one row per valid transaction
        and shape ``(N, C)``. ``edges`` has exactly four columns:
        ``source_transaction_id``, ``target_transaction_id``, ``via_account``,
        and ``time_delta``. Its shape is ``(E, 4)``. The number ``E`` depends on
        account continuity, timestamp ordering, and ``delta``; it is not equal
        to ``N`` in general.

        ``nodes`` 是每笔有效交易一行、shape 为 ``(N, C)`` 的 Polars
        ``DataFrame``。``edges`` 固定包含四列：
        ``source_transaction_id``、``target_transaction_id``、``via_account``
        和 ``time_delta``，shape 为 ``(E, 4)``。``E`` 取决于账户连续性、时间
        顺序和 ``delta``，通常不等于 ``N``。

    Smoke-run examples / 实际运行示例：
        With ``delta=1 hour`` the observed ``(N, E)`` values were PaySim
        ``(6,362,620, 56)``, IBM HI-Small ``(5,078,345, 2,176,494)``, and
        IBM LI-Small ``(6,924,049, 3,882,235)``.

        在 ``delta=1 hour`` 时，实际观察到的 ``(N, E)`` 为 PaySim
        ``(6,362,620, 56)``、IBM HI-Small ``(5,078,345, 2,176,494)`` 和
        IBM LI-Small ``(6,924,049, 3,882,235)``。

    An edge means that a later transaction starts from the earlier
    transaction's receiver within the configured time window. It represents a
    feasible temporal continuation, not proof of exact money tracing.

    Args:
        nodes: Transaction table with canonical graph columns.
        edges: Temporal succession edges between transaction IDs.
    """

    nodes: pl.DataFrame
    edges: pl.DataFrame

    @property
    def num_nodes(self) -> int:
        """Return ``nodes.height`` as a Python ``int`` / 返回交易节点数量。"""
        return self.nodes.height

    @property
    def num_edges(self) -> int:
        """Return ``edges.height`` as a Python ``int`` / 返回时间继承边数量。"""
        return self.edges.height

    @classmethod
    def from_transactions(
        cls,
        transactions: TransactionTable,
        *,
        delta: timedelta,
        source_column: str | None = None,
        target_column: str | None = None,
        timestamp_column: str | None = None,
        transaction_id_column: str | None = None,
    ) -> "TransactionGraph":
        """Build a temporal transaction graph.

        English:
            The algorithm first sorts transactions by timestamp. For each
            receiver account, it stores the later transactions that start from
            that account. Binary search then selects only candidates in
            ``(current_time, current_time + delta]``.

        中文：
            算法先按时间排序。对于每个收款账户，建立“之后从该账户发出”的
            交易索引；再用二分查找只取处于
            ``(当前时间, 当前时间 + delta]`` 的候选交易。

        Input / 输入：
            ``transactions`` is a Polars ``DataFrame`` or ``LazyFrame``. It
            needs source, target, and timestamp columns, either under canonical
            names or one of the supported aliases. ``delta`` is a non-negative
            Python ``timedelta``.

            ``transactions`` 是 Polars ``DataFrame`` 或 ``LazyFrame``，需要
            source、target、timestamp 字段，可以使用统一字段名或支持的别名。
            ``delta`` 是非负的 Python ``timedelta``。

        Returns / 返回：
            A materialized ``TransactionGraph``. Its node table has shape
            ``(N, C)`` and its edge table has shape ``(E, 4)`` with columns
            ``source_transaction_id``, ``target_transaction_id``,
            ``via_account``, and ``time_delta``. ``time_delta`` is a Polars
            duration column with nanosecond resolution.

            返回一个已经物化的 ``TransactionGraph``。节点表 shape 为
            ``(N, C)``，边表 shape 为 ``(E, 4)``，四列为
            ``source_transaction_id``、``target_transaction_id``、
            ``via_account`` 和 ``time_delta``；``time_delta`` 是纳秒精度的
            Polars duration 列。

        Args:
            transactions: Transaction rows as a Polars DataFrame or LazyFrame.
            delta: Inclusive maximum time between two linked transactions.
            source_column: Optional sender column override.
            target_column: Optional receiver column override.
            timestamp_column: Optional timestamp column override.
            transaction_id_column: Optional transaction ID column override.

        Returns:
            A graph whose nodes preserve the input transaction attributes and
            whose edges contain transaction IDs, the via account, and the time
            difference.

        Raises:
            ValueError: If required columns are missing, timestamps are not
                parseable, or ``delta`` is negative.
            TypeError: If the input is not a Polars table.
        """
        if not isinstance(delta, timedelta):
            raise TypeError("delta must be a datetime.timedelta")
        if delta < timedelta(0):
            raise ValueError("delta must be non-negative")

        # ``parse_timestamp=True`` adds the canonical timestamp used by both
        # node filtering and temporal edge construction.
        # ``parse_timestamp=True`` 会生成统一 timestamp，供节点筛选和建边使用。
        frame, source, target = _prepare_transactions(
            transactions,
            source_column=source_column,
            target_column=target_column,
            timestamp_column=timestamp_column,
            transaction_id_column=transaction_id_column,
            parse_timestamp=True,
        )
        if "timestamp" not in frame.columns:
            raise ValueError(
                "Missing required timestamp column; expected one of: "
                + ", ".join(_TIMESTAMP_ALIASES)
            )
        if frame.height and frame["timestamp"].null_count() == frame.height:
            raise ValueError(
                "Timestamp column exists but contains no parseable timestamps"
            )
        # Invalid timestamps cannot participate in a temporal graph. ``filter``
        # keeps only rows whose expression evaluates to true.
        # 无法解析时间的行不能参与时间图；``filter`` 只保留表达式为真的行。
        frame = frame.filter(pl.col("timestamp").is_not_null())

        node_columns = [
            column
            for column in frame.columns
            if column not in {_ROW_INDEX, _TIMESTAMP_NS}
        ]
        # ``select`` materializes the transaction rows as graph nodes while
        # excluding temporary computation columns.
        # ``select`` 把交易行变成图节点，并排除内部计算列。
        nodes = frame.select(node_columns)
        ordered = frame.with_columns(
            _timestamp_nanoseconds(frame).alias(_TIMESTAMP_NS)
        ).sort([_TIMESTAMP_NS, _ROW_INDEX])
        # ``with_columns`` adds the numeric timestamp used for sorting; the
        # original row index breaks ties deterministically.
        # ``with_columns`` 增加用于排序的数值时间戳；原始行号用于稳定处理同一时间。
        delta_ns = _timedelta_to_nanoseconds(delta)

        # One local categorical namespace gives both endpoint columns matching
        # compact u32 codes without Python strings crossing the native boundary.
        categories = pl.Categories.random(namespace="amlgraphx-transaction-graph")
        account_dtype = pl.Categorical(categories)
        coded = ordered.select(
            pl.col(source).cast(account_dtype).to_physical().alias("source_code"),
            pl.col(target).cast(account_dtype).to_physical().alias("target_code"),
        )
        source_positions, target_positions, time_deltas = _temporal_edge_indices(
            coded["source_code"].to_numpy(),
            coded["target_code"].to_numpy(),
            ordered[_TIMESTAMP_NS].to_numpy(),
            delta_ns,
        )
        edges = _transaction_edge_frame(
            ordered,
            source_positions,
            target_positions,
            time_deltas,
        )
        return cls(nodes=nodes, edges=edges)


def build_account_graph(
    transactions: TransactionTable,
    *,
    account_metadata: TransactionTable | None = None,
    source_column: str | None = None,
    target_column: str | None = None,
    transaction_id_column: str | None = None,
    account_id_column: str | None = None,
) -> AccountGraph:
    """Build an account graph from transaction rows.

    English: This is the function-oriented wrapper around
    ``AccountGraph.from_transactions``. It keeps the common public call short
    while the class method owns the actual construction logic.

    中文：这是 ``AccountGraph.from_transactions`` 的函数式包装器。它让常见
    调用更简洁，具体构建逻辑仍集中在类方法中。

    Input / 输入：
        A Polars ``DataFrame`` or ``LazyFrame`` containing source and target
        account columns. Optional account metadata is also accepted.

        输入是包含 source/target 账户列的 Polars ``DataFrame`` 或
        ``LazyFrame``，也可以提供账户元数据。

    Returns / 返回：
        ``AccountGraph``. ``result.nodes`` has one row per unique account;
        ``result.edges`` has one row per valid transaction.

        返回 ``AccountGraph``。``result.nodes`` 每个唯一账户一行；
        ``result.edges`` 每笔有效交易一行。

    Args:
        transactions: Transaction rows as a Polars DataFrame or LazyFrame.
        account_metadata: Optional account metadata table.
        source_column: Optional sender column override.
        target_column: Optional receiver column override.
        transaction_id_column: Optional transaction ID column override.
        account_id_column: Optional account metadata ID column override.

    Returns:
        An account graph with one directed edge per transaction.
    """
    return AccountGraph.from_transactions(
        transactions,
        account_metadata=account_metadata,
        source_column=source_column,
        target_column=target_column,
        transaction_id_column=transaction_id_column,
        account_id_column=account_id_column,
    )


def build_time_aware_account_graph(
    transactions: TransactionTable,
    *,
    account_metadata: TransactionTable | None = None,
    source_column: str | None = None,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    transaction_id_column: str | None = None,
    account_id_column: str | None = None,
) -> AccountGraph:
    """Build an account graph whose transaction edges have canonical time.

    Each valid transaction remains one directed multigraph edge together with
    all input columns, including amount, label, and dataset-specific features.
    The selected timestamp is parsed into the canonical ``timestamp`` column;
    rows with unparseable timestamps are excluded because they cannot belong
    to a temporal graph or event stream.

    中文：每笔有效交易仍是一条独立有向边，并保留金额、标签和数据集特有字段。
    时间字段会解析为统一的 ``timestamp``；无法解析时间的交易不能参与时间图，
    因此会被删除。

    Args:
        transactions: Transaction rows as a Polars DataFrame or LazyFrame.
        account_metadata: Optional attributes joined to account nodes.
        source_column: Optional sender column override.
        target_column: Optional receiver column override.
        timestamp_column: Optional transaction timestamp override.
        transaction_id_column: Optional transaction ID override.
        account_id_column: Optional account metadata ID override.

    Returns:
        An ``AccountGraph`` with typed ``timestamp`` and full edge features.
    """
    frame, source, target = _prepare_transactions(
        transactions,
        source_column=source_column,
        target_column=target_column,
        timestamp_column=timestamp_column,
        transaction_id_column=transaction_id_column,
        parse_timestamp=True,
    )
    if frame.height and frame["timestamp"].null_count() == frame.height:
        raise ValueError("Timestamp column contains no parseable timestamps")
    frame = frame.filter(pl.col("timestamp").is_not_null())
    return _account_graph_from_prepared(
        frame,
        source=source,
        target=target,
        account_metadata=account_metadata,
        account_id_column=account_id_column,
    )


def build_transaction_graph(
    transactions: TransactionTable,
    *,
    delta: timedelta,
    source_column: str | None = None,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    transaction_id_column: str | None = None,
) -> TransactionGraph:
    """Build a temporal transaction graph from transaction rows.

    English: Use this builder when transactions themselves should be the GNN
    nodes. ``delta`` controls how far a later transaction may be connected.

    中文：当你希望“交易本身成为 GNN 节点”时使用这个 builder。``delta`` 控制
    后续交易允许连接的最大时间间隔。

    Input / 输入：
        A Polars ``DataFrame`` or ``LazyFrame`` with source, target, and
        timestamp information, plus a non-negative ``timedelta`` ``delta``.

        输入是包含 source、target、timestamp 信息的 Polars ``DataFrame`` 或
        ``LazyFrame``，以及非负的 ``timedelta`` 类型 ``delta``。

    Returns / 返回：
        ``TransactionGraph`` with ``nodes.shape == (N, C)`` and
        ``edges.shape == (E, 4)``. The exact ``E`` is data- and ``delta``-
        dependent.

        返回 ``TransactionGraph``，其中 ``nodes.shape == (N, C)``，
        ``edges.shape == (E, 4)``；具体 ``E`` 取决于数据和 ``delta``。

    Args:
        transactions: Transaction rows as a Polars DataFrame or LazyFrame.
        delta: Inclusive maximum time between linked transactions.
        source_column: Optional sender column override.
        target_column: Optional receiver column override.
        timestamp_column: Optional timestamp column override.
        transaction_id_column: Optional transaction ID column override.

    Returns:
        A transaction graph with all valid temporal succession edges.
    """
    return TransactionGraph.from_transactions(
        transactions,
        delta=delta,
        source_column=source_column,
        target_column=target_column,
        timestamp_column=timestamp_column,
        transaction_id_column=transaction_id_column,
    )


def _account_graph_from_prepared(
    frame: pl.DataFrame,
    *,
    source: str,
    target: str,
    account_metadata: TransactionTable | None,
    account_id_column: str | None,
) -> AccountGraph:
    """Materialize account nodes and transaction edges from prepared rows."""
    # Keep every transaction column as an edge attribute. Only the temporary
    # row index is implementation detail. / 每个交易字段都是潜在 edge feature；
    # 只有内部行号不属于公开输出。
    edge_columns = [column for column in frame.columns if column != _ROW_INDEX]
    edges = frame.select(edge_columns)
    nodes = (
        pl.concat(
            [
                frame.select(pl.col(source).alias("node_id")),
                frame.select(pl.col(target).alias("node_id")),
            ]
        )
        .unique(subset=["node_id"], maintain_order=True)
        .sort("node_id")
    )
    if account_metadata is not None:
        nodes = _join_account_metadata(
            nodes,
            account_metadata,
            account_id_column=account_id_column,
        )
    return AccountGraph(nodes=nodes, edges=edges)


def _prepare_transactions(
    transactions: TransactionTable,
    *,
    source_column: str | None,
    target_column: str | None,
    timestamp_column: str | None = None,
    transaction_id_column: str | None = None,
    parse_timestamp: bool = False,
) -> tuple[pl.DataFrame, str, str]:
    """Collect and normalize endpoints before either graph is built.

    English:
        This is the shared boundary between raw dataset columns and graph
        logic. It resolves aliases, adds a temporary row number, creates a
        transaction ID, trims endpoint strings, removes invalid endpoints, and
        optionally parses timestamps.

    中文：
        这是原始数据字段进入图逻辑前的共同边界。它会解析别名、添加临时行号、
        创建交易 ID、清理两端账户字符串、删除无效端点，并按需解析时间戳。

    Polars / Polars：
        ``with_columns`` evaluates several ``Expr`` objects and appends or
        replaces columns. ``str.strip_chars`` removes surrounding whitespace;
        ``filter`` keeps rows satisfying a boolean expression. For example,
        ``frame.filter(pl.col("source") != "")`` removes empty senders.
        ``with_columns`` 会计算多个 ``Expr`` 并新增或替换列；
        ``str.strip_chars`` 去除首尾空白；``filter`` 按布尔表达式保留行。
        例如 ``frame.filter(pl.col("source") != "")`` 会去掉空 sender。

    Returns / 返回：
        A tuple ``(frame, "source", "target")``. ``frame`` is an eager Polars
        ``DataFrame`` with one row per transaction that survived endpoint
        cleaning. It keeps raw columns and adds ``__amlgraphx_row_index``,
        ``transaction_id``, canonical ``source`` and ``target``, plus
        ``timestamp`` when ``parse_timestamp=True``.

        返回 ``(frame, "source", "target")``。``frame`` 是已经物化的 Polars
        ``DataFrame``，每行对应一笔通过端点清理的交易。它保留原始列，并增加
        ``__amlgraphx_row_index``、``transaction_id``、统一的 ``source`` 和
        ``target``；当 ``parse_timestamp=True`` 时还会增加 ``timestamp``。
        两个字符串是后续建图使用的统一列名。
    """
    frame = _collect_frame(transactions)
    source = _resolve_required_column(
        frame.columns,
        source_column,
        _SOURCE_ALIASES,
        "source account",
    )
    target = _resolve_required_column(
        frame.columns,
        target_column,
        _TARGET_ALIASES,
        "target account",
    )
    if _ROW_INDEX in frame.columns:
        raise ValueError(f"Input column {_ROW_INDEX!r} is reserved")

    # ``with_row_index`` adds a deterministic integer column without changing
    # the original row order. / ``with_row_index`` 增加稳定行号且不改变原顺序。
    frame = frame.with_row_index(_ROW_INDEX)
    id_column = _resolve_optional_column(
        frame.columns,
        transaction_id_column,
        _TRANSACTION_ID_ALIASES,
    )
    # Cast first, then trim. Casting makes numeric account IDs usable as string
    # graph keys as well. / 先转字符串再清理，保证数字账户也能作为图键。
    frame = frame.with_columns(
        _make_transaction_ids(frame, id_column).alias("transaction_id")
    )

    frame = frame.with_columns(
        pl.col(source).cast(pl.String).str.strip_chars().alias("source"),
        pl.col(target).cast(pl.String).str.strip_chars().alias("target"),
    ).filter(
        pl.col("source").is_not_null()
        & (pl.col("source") != "")
        & pl.col("target").is_not_null()
        & (pl.col("target") != "")
    )

    if parse_timestamp:
        timestamp = _resolve_required_column(
            frame.columns,
            timestamp_column,
            _TIMESTAMP_ALIASES,
            "timestamp",
        )
        frame = frame.with_columns(
            _timestamp_expression(frame, timestamp).alias("timestamp")
        )

    return frame, "source", "target"


def _collect_frame(transactions: TransactionTable) -> pl.DataFrame:
    """Convert either Polars input form to an independent DataFrame.

    English: ``LazyFrame.collect()`` executes the lazy query and materializes
    its result. A DataFrame is cloned so graph construction does not mutate the
    caller's table.

    中文：``LazyFrame.collect()`` 会执行延迟查询并得到实际数据；输入已经是
    DataFrame 时则先 clone，避免图构建修改调用方持有的表。

    Returns / 返回：
        A new eager Polars ``DataFrame`` with the same logical rows and columns
        as the input. A ``LazyFrame`` is collected; a ``DataFrame`` is cloned.
        The returned table is materialized and independent of the caller's
        original DataFrame.

        返回一个新的 eager Polars ``DataFrame``，逻辑上的行列与输入相同。
        ``LazyFrame`` 会被 collect，``DataFrame`` 会被 clone；返回表已经物化，
        且不会修改调用方原始 DataFrame。
    """
    if isinstance(transactions, pl.LazyFrame):
        return transactions.collect()
    if isinstance(transactions, pl.DataFrame):
        return transactions.clone()
    raise TypeError("transactions must be a polars.DataFrame or polars.LazyFrame")


def _make_transaction_ids(
    frame: pl.DataFrame,
    id_column: str | None,
) -> pl.Series:
    """Return stable, non-empty transaction IDs for every prepared row.

    English: Unique non-empty source IDs are retained. Missing or duplicated
    IDs fall back to the original row number, with a suffix if needed.

    中文：唯一且非空的原始交易 ID 会被保留；缺失或重复的 ID 使用原始行号
    生成备用 ID，若仍冲突则追加后缀。

    Returns / 返回：
        A Polars ``Series`` named ``transaction_id`` with length
        ``frame.height`` and dtype ``pl.String``. Every value is non-null and
        unique within this prepared frame.

        返回名为 ``transaction_id`` 的 Polars ``Series``，长度为
        ``frame.height``，dtype 为 ``pl.String``；在当前 frame 内每个值都非空
        且唯一。
    """
    if id_column is None:
        return pl.Series(
            "transaction_id",
            [f"tx_{row_index}" for row_index in frame[_ROW_INDEX].to_list()],
            dtype=pl.String,
        )

    values = frame.get_column(id_column).cast(pl.String).str.strip_chars().to_list()
    counts = Counter(value for value in values if value not in (None, ""))
    used = {
        value for value, count in counts.items() if count == 1 and value is not None
    }
    transaction_ids: list[str] = []
    for row_index, value in zip(frame[_ROW_INDEX].to_list(), values, strict=True):
        if value not in (None, "") and counts[value] == 1:
            transaction_ids.append(value)
            continue

        candidate = f"tx_{row_index}"
        suffix = 1
        while candidate in used:
            candidate = f"tx_{row_index}_{suffix}"
            suffix += 1
        used.add(candidate)
        transaction_ids.append(candidate)

    return pl.Series("transaction_id", transaction_ids, dtype=pl.String)


def _join_account_metadata(
    nodes: pl.DataFrame,
    account_metadata: TransactionTable,
    *,
    account_id_column: str | None,
) -> pl.DataFrame:
    """Clean account metadata and attach it to account nodes.

    Polars / Polars：
        ``unique(subset=["node_id"], maintain_order=True)`` 保留每个 node_id
        的第一行；``join(..., how="left")`` 将右表属性匹配到左表节点，左表
        的所有行都会保留。Example / 示例：
        ``nodes.join(metadata, on="node_id", how="left")``。

    Returns / 返回：
        A Polars ``DataFrame`` with the same height as ``nodes``. It contains
        ``node_id`` and metadata columns. Unmatched metadata values are null;
        account nodes are not removed.

        返回一个 height 与 ``nodes`` 相同的 Polars ``DataFrame``，包含
        ``node_id`` 和 metadata 列。没有匹配到的属性为 null，但账户节点不会
        被删除。
    """
    metadata = _collect_frame(account_metadata)
    id_column = _resolve_required_column(
        metadata.columns,
        account_id_column,
        _ACCOUNT_ID_ALIASES,
        "account metadata ID",
    )
    metadata = metadata.with_columns(
        pl.col(id_column).cast(pl.String).str.strip_chars().alias("node_id")
    ).unique(subset=["node_id"], maintain_order=True)
    if id_column != "node_id":
        metadata = metadata.drop(id_column)
    return nodes.join(metadata, on="node_id", how="left")


def _resolve_required_column(
    columns: Sequence[str],
    requested: str | None,
    aliases: Sequence[str],
    logical_name: str,
) -> str:
    """Resolve a requested column or raise a domain-specific error.

    中文：优先使用用户显式指定的列，否则按别名查找；找不到时抛出包含期望
    字段名的错误，方便定位数据集 schema 问题。

    Returns / 返回：
        The matching original column name as a Python ``str``. This helper
        never returns ``None``; it raises ``ValueError`` when no match exists.

        返回匹配到的原始列名，类型为 Python ``str``；没有匹配时抛出
        ``ValueError``，不会返回 ``None``。
    """
    column = _resolve_optional_column(columns, requested, aliases)
    if column is None:
        expected = ", ".join(aliases)
        raise ValueError(
            f"Missing required {logical_name} column; expected one of: {expected}"
        )
    return column


def _resolve_optional_column(
    columns: Sequence[str],
    requested: str | None,
    aliases: Sequence[str],
) -> str | None:
    """Resolve a column name case-insensitively and punctuation-insensitively.

    Returns / 返回：
        The matching original column name as ``str``, or ``None`` when no
        requested name or alias matches. The original spelling is preserved.

        返回匹配到的原始列名 ``str``；没有匹配时返回 ``None``，并保留原始
        字段的拼写。
    """
    normalized = {_normalize_column(column): column for column in columns}
    if requested is not None:
        if requested in columns:
            return requested
        return normalized.get(_normalize_column(requested))
    for alias in aliases:
        column = normalized.get(_normalize_column(alias))
        if column is not None:
            return column
    return None


def _normalize_column(column: str) -> str:
    """Normalize one column name to a lowercase, space-separated ``str``.

    For example, ``"Account_duplicated_0"`` becomes
    ``"account duplicated 0"``. This scalar helper does not touch a Polars
    table.

    例如 ``"Account_duplicated_0"`` 会变成 ``"account duplicated 0"``。这是
    一个标量 helper，不会修改 Polars 表。
    """
    return " ".join(column.lower().replace("_", " ").replace(".", " ").split())


def _timestamp_expression(frame: pl.DataFrame, column: str) -> pl.Expr:
    """Build a Polars expression that returns a canonical datetime column.

    English: Existing ``Datetime`` values pass through; ``Date`` values are
    cast to midnight; strings are parsed with several accepted formats. If a
    separate date column exists, the time expression is also tried after
    combining the two strings.

    中文：已有的 ``Datetime`` 直接使用；``Date`` 转成当天零点；字符串按多种
    格式尝试解析。如果还有单独的 date 列，也会尝试把 date 和 time 拼起来。

    Polars / Polars：
        ``pl.coalesce(a, b)`` returns the first non-null value row by row.
        ``pl.concat_str([...], separator=" ")`` concatenates string columns
        row by row. Example / 示例：
        ``pl.coalesce(pl.col("parsed"), pl.col("fallback"))``。

    Returns / 返回：
        A Polars ``Expr``. When evaluated under the output name ``timestamp``,
        it produces one datetime value per input row; failed string parses are
        null. This helper itself does not collect or materialize data.

        返回 Polars ``Expr``。以 ``timestamp`` 名称执行后，会得到与输入行数
        相同的 datetime 列；字符串解析失败的值为 null。本函数本身不会 collect
        或物化数据。
    """
    dtype = frame.schema[column]
    expression = pl.col(column)
    if dtype == pl.Datetime:
        return expression
    if dtype == pl.Date:
        return expression.cast(pl.Datetime)
    if dtype == pl.String:
        parsed_timestamp = _parse_datetime_strings(expression)
        date_column = _resolve_optional_column(
            frame.columns,
            None,
            ("date",),
        )
        if date_column is not None and date_column != column:
            # ``str.replace`` removes a time suffix from a date-like string;
            # the regex keeps only the part before ``T`` or a space.
            # ``str.replace`` 去掉日期字符串中的时间后缀，只保留 T 或空格前部分。
            date_only = pl.col(date_column).cast(pl.String).str.replace(r"[T ].*$", "")
            return pl.coalesce(
                parsed_timestamp,
                _parse_datetime_strings(
                    pl.concat_str(
                        [date_only, pl.col(column)],
                        separator=" ",
                    )
                ),
            )
        return parsed_timestamp
    return expression.cast(pl.Datetime, strict=False)


def _timestamp_nanoseconds(frame: pl.DataFrame) -> pl.Expr:
    """Convert Polars datetime values to integer nanoseconds for sorting.

    中文：Polars 的 Datetime 可能使用 ms/us/ns 不同时间单位；先读取单位再缩放
    为统一的纳秒整数，避免直接比较不同精度的内部数值。

    Returns / 返回：
        A Polars ``Expr`` that evaluates to an ``pl.Int64`` column of integer
        nanoseconds with the same height as ``frame``. The expression is lazy
        until a Polars operation evaluates it.

        返回一个 Polars ``Expr``；执行后得到与 ``frame`` 行数相同、dtype 为
        ``pl.Int64`` 的整数纳秒列。表达式在交给 Polars 操作前不会执行。
    """
    time_unit = frame.schema["timestamp"].time_unit
    scale = {"ms": 1_000_000, "us": 1_000, "ns": 1}[time_unit]
    return pl.col("timestamp").cast(pl.Int64) * scale


def _timedelta_to_nanoseconds(value: timedelta) -> int:
    """Convert a Python ``timedelta`` to one integer number of nanoseconds.

    Returns / 返回：
        A scalar Python ``int`` used as the inclusive temporal-search bound;
        this helper does not create a Series or Polars expression.

        返回一个用于时间搜索包含式上界的 Python ``int`` 标量，不会创建 Series
        或 Polars 表达式。
    """
    return (
        value.days * 86_400_000_000_000_000
        + value.seconds * 1_000_000_000
        + value.microseconds * 1_000
    )


def _parse_datetime_strings(expression: pl.Expr) -> pl.Expr:
    """Try the supported datetime string formats without raising on bad rows.

    ``strict=False`` makes a failed parse become null; ``pl.coalesce`` then
    tries the next format. / ``strict=False`` 会把解析失败变成 null，随后由
    ``pl.coalesce`` 继续尝试下一种格式。

    Returns / 返回：
        A Polars ``Expr`` that evaluates to a ``Datetime`` column with the
        input expression's row count. Rows matching no format become null.

        返回一个 Polars ``Expr``，执行后得到与输入表达式行数相同的
        ``Datetime`` 列；所有格式都无法匹配的行变为 null。
    """
    normalized = expression.str.replace("T", " ")
    # pl coalesce 从左到右选方案 选到不是null为止
    return pl.coalesce(
        normalized.str.to_datetime(
            format="%Y-%m-%d %H:%M:%S%.f",
            strict=False,
        ),
        normalized.str.to_datetime(
            format="%Y-%m-%d %H:%M:%S",
            strict=False,
        ),
        normalized.str.to_datetime(format="%Y-%m-%d %H:%M", strict=False),
        normalized.str.to_datetime(format="%Y-%m-%d", strict=False),
    )


def _transaction_edge_frame(
    ordered: pl.DataFrame,
    source_positions: NDArray[np.int64],
    target_positions: NDArray[np.int64],
    time_deltas_ns: NDArray[np.int64],
) -> pl.DataFrame:
    """Create the stable edge schema, including the empty-graph case.

    Polars / Polars：``pl.duration(nanoseconds=...)`` converts the integer time
    gap into a typed duration column. The explicit empty schema ensures that a
    graph with no valid temporal edges still has predictable column names and
    dtypes.

    Returns / 返回：
        A Polars ``DataFrame`` with shape ``(E, 4)`` and columns
        ``source_transaction_id`` (String), ``target_transaction_id`` (String),
        ``via_account`` (String), and ``time_delta`` (Duration[ns]). Empty
        input still returns shape ``(0, 4)`` with the same schema.

        返回 shape 为 ``(E, 4)`` 的 Polars ``DataFrame``，列为
        ``source_transaction_id``、``target_transaction_id``、``via_account``
        （均为 String）和 ``time_delta``（Duration[ns]）。输入为空时仍返回
        shape ``(0, 4)`` 的相同 schema。
    """
    return pl.DataFrame(
        {
            "source_transaction_id": ordered["transaction_id"].gather(source_positions),
            "target_transaction_id": ordered["transaction_id"].gather(target_positions),
            "via_account": ordered["target"].gather(source_positions),
            "time_delta": pl.Series(time_deltas_ns, dtype=pl.Duration("ns")),
        }
    )
