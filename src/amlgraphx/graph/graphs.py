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
"""

from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import polars as pl

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

    Args:
        nodes: Account table containing at least ``node_id``.
        edges: Transaction table containing ``source`` and ``target``.
    """

    nodes: pl.DataFrame
    edges: pl.DataFrame

    @property
    def num_nodes(self) -> int:
        """Return the number of account nodes / 返回账户节点数量。"""
        return self.nodes.height

    @property
    def num_edges(self) -> int:
        """Return the number of transaction edges / 返回交易边数量。"""
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

        # ``select`` keeps the original transaction attributes. The internal
        # row index is only for deterministic ID generation, so it is removed.
        # ``select`` 保留原始交易属性；内部行号只用于生成 ID，因此要删除。
        edge_columns = [column for column in frame.columns if column != _ROW_INDEX]
        edges = frame.select(edge_columns)

        # ``pl.concat`` vertically stacks two one-column tables. ``unique``
        # keeps one row per account, and ``sort`` makes node order stable.
        # ``pl.concat`` 纵向堆叠两张单列表；``unique`` 按账户去重，``sort``
        # 让节点顺序稳定，便于后续把节点映射为整数索引。
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
            # A left join preserves every account observed in transactions,
            # even when metadata is missing. / 左连接保留所有交易中出现的账户，
            # 即使某个账户没有对应的元数据也不会丢失。
            nodes = _join_account_metadata(
                nodes,
                account_metadata,
                account_id_column=account_id_column,
            )

        return cls(nodes=nodes, edges=edges)


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
        """Return the number of transaction nodes / 返回交易节点数量。"""
        return self.nodes.height

    @property
    def num_edges(self) -> int:
        """Return the number of temporal succession edges / 返回时间继承边数量。"""
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
        rows = list(ordered.iter_rows(named=True))
        delta_ns = _timedelta_to_nanoseconds(delta)

        # Build an account -> sorted outgoing transactions index. This turns
        # each later lookup into a binary search instead of a full scan.
        # 建立“账户 -> 已按时间排序的发出交易”索引，后续用二分查找而不是全表扫描。
        outgoing: dict[str, list[tuple[int, int]]] = {}
        outgoing_times: dict[str, list[int]] = {}
        for position, row in enumerate(rows):
            account = row[source]
            timestamp = row[_TIMESTAMP_NS]
            outgoing.setdefault(account, []).append((timestamp, position))
            outgoing_times.setdefault(account, []).append(timestamp)

        edge_records: list[dict[str, object]] = []
        for row in rows:
            candidates = outgoing.get(row[target], [])
            candidate_times = outgoing_times.get(row[target], [])
            timestamp = row[_TIMESTAMP_NS]
            # ``bisect_right`` starts strictly after the current timestamp, so
            # same-time rows are intentionally excluded. The second search is
            # inclusive of ``timestamp + delta_ns``.
            # ``bisect_right`` 从当前时间之后开始，因此同一时间的交易被排除；
            # 第二次查找包含 ``timestamp + delta_ns`` 这一边界。
            start = bisect_right(candidate_times, timestamp)
            end = bisect_right(
                candidate_times,
                timestamp + delta_ns,
            )
            for _, successor_position in candidates[start:end]:
                successor = rows[successor_position]
                edge_records.append(
                    {
                        "source_transaction_id": row["transaction_id"],
                        "target_transaction_id": successor["transaction_id"],
                        "via_account": row[target],
                        "time_delta_ns": successor[_TIMESTAMP_NS] - timestamp,
                    }
                )

        edges = _transaction_edge_frame(edge_records)
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
    """Resolve a column name case-insensitively and punctuation-insensitively."""
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
    """Normalize spaces, underscores, and dots for alias matching."""
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
    """
    time_unit = frame.schema["timestamp"].time_unit
    scale = {"ms": 1_000_000, "us": 1_000, "ns": 1}[time_unit]
    return pl.col("timestamp").cast(pl.Int64) * scale


def _timedelta_to_nanoseconds(value: timedelta) -> int:
    """Convert a Python ``timedelta`` to integer nanoseconds."""
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
    """
    normalized = expression.str.replace("T", " ")
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


def _transaction_edge_frame(records: list[dict[str, object]]) -> pl.DataFrame:
    """Create the stable edge schema, including the empty-graph case.

    Polars / Polars：``pl.duration(nanoseconds=...)`` converts the integer time
    gap into a typed duration column. The explicit empty schema ensures that a
    graph with no valid temporal edges still has predictable column names and
    dtypes.
    """
    if records:
        return (
            pl.DataFrame(records)
            .with_columns(
                pl.duration(nanoseconds=pl.col("time_delta_ns")).alias("time_delta")
            )
            .drop("time_delta_ns")
        )
    return pl.DataFrame(
        {
            "source_transaction_id": pl.Series([], dtype=pl.String),
            "target_transaction_id": pl.Series([], dtype=pl.String),
            "via_account": pl.Series([], dtype=pl.String),
            "time_delta": pl.Series([], dtype=pl.Duration("ns")),
        }
    )
