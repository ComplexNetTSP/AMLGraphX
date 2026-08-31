"""Canonical transaction columns shared by AMLGraphX datasets.

English:
    Dataset adapters may expose different names for the same concept. This
    module keeps the raw columns and adds canonical ``source``, ``target``,
    ``transaction_id``, ``timestamp``, ``amount``, and ``label`` columns when
    they can be found. It operates on a ``LazyFrame`` so loading and cleanup
    can remain part of one Polars query plan.

中文：
    不同数据集可能用不同字段名表达同一个概念。本模块保留原始列，并在能够
    找到对应字段时增加统一的 ``source``、``target``、``transaction_id``、
    ``timestamp``、``amount`` 和 ``label`` 列。输入使用 ``LazyFrame``，因此
    读取和清洗可以保留在同一个 Polars 延迟查询计划中。

Polars quick guide / Polars 快速提示：
    A ``LazyFrame`` describes operations without immediately reading every
    row. ``pl.col("Sender")`` refers to a column expression, ``cast`` changes
    its dtype, ``str.strip_chars`` trims text, and ``alias`` chooses the output
    name. ``with_columns`` evaluates these expressions and adds/replaces the
    resulting columns. For example:

    ``lf.with_columns(pl.col("Sender").cast(pl.String).str.strip_chars().alias("source"))``

    ``LazyFrame`` 描述操作但不会立刻读取全部行。``pl.col("Sender")`` 表示
    一列，``cast`` 改变类型，``str.strip_chars`` 清理字符串首尾空白，
    ``alias`` 指定输出列名；``with_columns`` 执行这些表达式并新增或替换列。
    例如：

    ``lf.with_columns(pl.col("Sender").cast(pl.String).str.strip_chars().alias("source"))``

Observed schema sizes / 实际 schema 尺寸：
    A smoke run called ``dataset.transactions().collect_schema()`` and then
    collected the normalized rows. The returned object from
    ``normalize_transactions`` remained a ``LazyFrame``; the sizes below are
    the sizes after ``collect()``.

    | dataset | rows | columns | canonical columns present |
    |---|---:|---:|---|
    | PaySim | 6,362,620 | 16 | `transaction_id`, `source`, `target`, `timestamp`, `amount`, `label` |
    | IBM HI-Small | 5,078,345 | 18 | `transaction_id`, `source`, `target`, `timestamp`, `amount`, `label` |
    | IBM LI-Small | 6,924,049 | 18 | `transaction_id`, `source`, `target`, `timestamp`, `amount`, `label` |

    实际 smoke run 先检查 ``dataset.transactions().collect_schema()``，再
    ``collect()`` 统一后的行。``normalize_transactions`` 返回的仍然是
    ``LazyFrame``；表格中的 rows/columns 是 collect 之后的尺寸。
"""

from collections.abc import Sequence

import polars as pl

_ROW_INDEX = "__amlgraphx_schema_row"

# Temporary row number used to generate deterministic IDs when the source data
# has no usable transaction ID. / 当原始数据没有可用交易 ID 时，用临时行号生成稳定 ID。

_ALIASES: dict[str, tuple[str, ...]] = {
    "source": (
        "source",
        "sender",
        "sender account",
        "from",
        "from account",
        "source id",
        "src",
        "nameorig",
        "origin account",
        "origin",
    ),
    "target": (
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
    ),
    "timestamp": ("timestamp", "datetime", "time", "date"),
    "amount": ("amount", "amount received", "amount paid", "value"),
    "label": (
        "label",
        "is fraud",
        "isfraud",
        "is laundering",
        "islaundering",
        "fraud",
    ),
    "transaction_id": ("transaction_id", "transaction id", "tx id", "id"),
}


def normalize_transactions(
    frame: pl.LazyFrame,
    *,
    source_column: str | None = None,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    amount_column: str | None = None,
    label_column: str | None = None,
    transaction_id_column: str | None = None,
) -> pl.LazyFrame:
    """Add canonical transaction columns while preserving raw columns.

    English:
        The function is a schema adapter, not a data imputer. It does not
        discard dataset-specific columns and it only adds ``amount``, ``label``,
        or ``timestamp`` when a corresponding input column exists. Call
        ``collect()`` later when the data is needed in memory.

    中文：
        这个函数只负责 schema 适配，不负责通用缺失值填补。它不会删除数据集
        特有字段；只有在原始表中找到对应列时才增加 ``amount``、``label`` 或
        ``timestamp``。真正需要数据时再调用 ``collect()`` 物化。

    Example / 示例：
        ``raw = pl.LazyFrame({"Sender": [" A "], "Receiver": ["B"]})``
        ``clean = normalize_transactions(raw)``
        ``clean.collect().select("source", "target", "transaction_id")``
        produces ``A``, ``B``, and ``tx_0`` while retaining ``Sender`` and
        ``Receiver``.

    Input / 输入：
        A Polars ``LazyFrame``. It must contain source and target columns, or
        columns matching the supported aliases. The optional fields are only
        added when a matching input column exists.

        输入是 Polars ``LazyFrame``，必须包含 source 和 target 字段，或匹配支持
        的字段别名。可选字段只有在原始表存在匹配列时才会增加。

    Returns / 返回：
        A Polars ``LazyFrame`` with the same row count as the input and the
        original columns plus canonical columns that were discoverable. It
        does not collect data and it does not remove rows. After
        ``result.collect()``, the canonical transaction table has shape
        ``(N, C)`` and includes ``transaction_id``, ``source``, ``target`` plus
        any available ``timestamp``, ``amount``, and ``label`` columns.

        返回一个与输入行数相同的 Polars ``LazyFrame``，保留原始列并增加能够
        识别出的统一列。它不会 collect，也不会删除行。执行
        ``result.collect()`` 后，统一交易表 shape 为 ``(N, C)``，包含
        ``transaction_id``、``source``、``target``，以及可用的 ``timestamp``、
        ``amount``、``label``。
    """
    columns = frame.collect_schema().names()
    if _ROW_INDEX in columns:
        raise ValueError(f"Input column {_ROW_INDEX!r} is reserved")

    source = _resolve_column(columns, source_column, _ALIASES["source"])
    target = _resolve_column(columns, target_column, _ALIASES["target"])
    if source is None or target is None:
        raise ValueError("Transactions require source and target columns")

    # ``collect_schema`` inspects column names without collecting all rows.
    # ``with_row_index`` adds a temporary stable row number to the lazy plan.
    # ``collect_schema`` 只读取 schema，不物化全部数据；``with_row_index`` 在
    # 延迟计划中增加稳定的临时行号。
    frame = frame.with_row_index(_ROW_INDEX)
    # Each item is a Polars expression. ``with_columns`` evaluates the list in
    # one plan step and keeps the original raw columns.
    # 每一项都是 Polars 表达式；``with_columns`` 在一个计划步骤中计算它们，
    # 同时保留原始列。
    expressions = [
        pl.col(source).cast(pl.String).str.strip_chars().alias("source"),
        pl.col(target).cast(pl.String).str.strip_chars().alias("target"),
        _transaction_id_expression(
            frame,
            _resolve_column(columns, transaction_id_column, _ALIASES["transaction_id"]),
        ),
    ]
    timestamp = _timestamp_expression(columns, timestamp_column)
    if timestamp is not None:
        expressions.append(timestamp.alias("timestamp"))
    for canonical, requested in (("amount", amount_column), ("label", label_column)):
        column = _resolve_column(columns, requested, _ALIASES[canonical])
        if column is not None:
            expressions.append(pl.col(column).alias(canonical))
    # ``drop`` removes only the implementation column before the lazy frame is
    # returned to the caller. / 返回前用 ``drop`` 删除唯一的内部实现列。
    return frame.with_columns(expressions).drop(_ROW_INDEX)


def _transaction_id_expression(frame: pl.LazyFrame, column: str | None) -> pl.Expr:
    """Build an expression that keeps valid IDs or generates ``tx_<row>``.

    Polars / Polars：
        ``pl.lit("tx_")`` creates a literal expression,
        ``pl.concat_str`` concatenates expressions row by row, and
        ``pl.when(...).then(...).otherwise(...)`` is a vectorized if/else.
        Example / 示例：
        ``pl.when(pl.col("id") != "").then(pl.col("id")).otherwise("generated")``。

    中文：这些操作都是向量化表达式，不需要 Python 逐行循环：``pl.lit`` 创建
    常量，``pl.concat_str`` 按行拼接字符串，``when/then/otherwise`` 表示
    if/else。

    Returns / 返回：
        A Polars ``Expr`` that evaluates to one ``pl.String`` transaction ID per
        input row. If ``column`` is absent, values are ``tx_<row_index>``. If
        it is present, non-empty values are retained and empty/null values use
        the generated fallback. This expression is not evaluated here.

        返回一个 Polars ``Expr``，执行后每个输入行得到一个 ``pl.String`` 交易
        ID。没有 ``column`` 时使用 ``tx_<row_index>``；有该列时保留非空值，
        空值/null 使用备用 ID。本函数不会在这里执行表达式。
    """
    generated = pl.concat_str([pl.lit("tx_"), pl.col(_ROW_INDEX).cast(pl.String)])
    if column is None:
        return generated.alias("transaction_id")

    value = pl.col(column).cast(pl.String).str.strip_chars()
    return (
        pl.when(value.is_not_null() & (value != ""))
        .then(value)
        .otherwise(generated)
        .alias("transaction_id")
    )


def _resolve_column(
    columns: Sequence[str], requested: str | None, aliases: Sequence[str]
) -> str | None:
    """Resolve an explicit column or the first matching alias.

    Returns / 返回：
        The original matching column name as ``str``, or ``None`` if neither
        ``requested`` nor any alias is present. Matching ignores case,
        underscores, dots, and repeated whitespace, but the returned spelling
        is unchanged.

        返回匹配到的原始列名 ``str``；如果 requested 和所有 aliases 都不存在，
        返回 ``None``。匹配时忽略大小写、下划线、点号和重复空格，但返回值
        保留原始拼写。
    """
    normalized = {_normalize_column(column): column for column in columns}
    if requested is not None:
        column = normalized.get(_normalize_column(requested))
        if column is not None:
            return column
    for alias in aliases:
        column = normalized.get(_normalize_column(alias))
        if column is not None:
            return column
    return None


def _timestamp_expression(
    columns: Sequence[str], requested: str | None
) -> pl.Expr | None:
    """Build a lazy expression for the best available timestamp.

    English: Prefer an existing ``timestamp``/``datetime`` column. Otherwise,
    combine separate date and time columns, trying time-only, combined, and
    date-only interpretations in that order.

    中文：优先使用已有的 ``timestamp``/``datetime`` 列；否则组合独立的 date
    和 time 列，并依次尝试仅 time、date+time、仅 date 的解释。

    Polars / Polars：
        ``pl.concat_str([date_text, time_text], separator=" ")`` concatenates
        values row by row. ``pl.coalesce(a, b, c)`` returns the first non-null
        result per row, so it is useful for fallback parsing.
        ``pl.concat_str`` 按行拼接字符串；``pl.coalesce`` 对每一行返回第一个
        非空结果，因此适合实现多种解析方案的 fallback。

    Returns / 返回：
        A Polars ``Expr`` or ``None``. When evaluated as ``timestamp``, the
        expression produces one timestamp value per input row. ``None`` means
        that no timestamp-like column was found, so the caller can decide
        whether a timestamp is required.

        返回 Polars ``Expr`` 或 ``None``。作为 ``timestamp`` 执行时，每个输入
        行产生一个时间值；``None`` 表示没有找到任何时间字段，由调用方决定
        是否报错。
    """
    direct = _resolve_column(columns, requested, ("timestamp", "datetime"))
    if direct is not None:
        return pl.col(direct)

    date = _resolve_column(columns, None, ("date",))
    time = _resolve_column(columns, None, ("time",))
    if date is not None and time is not None:
        # Keep only the date portion before combining it with time.
        # 拼接前先保留 date 字符串中 T 或空格之前的日期部分。
        date_text = pl.col(date).cast(pl.String).str.replace(r"[T ].*$", "")
        time_text = pl.col(time).cast(pl.String)
        combined = pl.concat_str([date_text, time_text], separator=" ")
        return pl.coalesce(
            _parse_datetime_strings(time_text),
            _parse_datetime_strings(combined),
            _parse_datetime_strings(pl.col(date).cast(pl.String)),
        )

    column = date or time
    return pl.col(column) if column is not None else None


def _parse_datetime_strings(expression: pl.Expr) -> pl.Expr:
    """Parse supported string formats and turn failures into nulls.

    ``strict=False`` prevents one malformed row from aborting the whole lazy
    query. ``coalesce`` tries formats from most precise to least precise.
    中文：``strict=False`` 让单个坏值变成 null，而不是让整条 lazy query 失败；
    ``coalesce`` 按“精确到粗略”的顺序尝试多种格式。

    Returns / 返回：
        A Polars ``Expr`` that evaluates to a ``Datetime`` column with the
        same height as the input expression. Failed parses evaluate to null.

        返回一个 Polars ``Expr``，执行后得到与输入表达式行数相同的
        ``Datetime`` 列；解析失败的值为 null。
    """
    normalized = expression.str.replace("T", " ")
    return pl.coalesce(
        normalized.str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False),
        normalized.str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False),
        normalized.str.to_datetime(format="%Y-%m-%d %H:%M", strict=False),
        normalized.str.to_datetime(format="%Y-%m-%d", strict=False),
    )


def _normalize_column(column: str) -> str:
    """Normalize one column name to a lowercase, space-separated ``str``.

    Example / 示例: ``"Account_duplicated_0"`` becomes
    ``"account duplicated 0"``. This scalar helper does not modify a table.
    """
    return " ".join(column.lower().replace("_", " ").replace(".", " ").split())
