"""Canonical transaction columns shared by AMLGraphX datasets."""

from collections.abc import Sequence

import polars as pl

_ROW_INDEX = "__amlgraphx_schema_row"

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
    """Add canonical transaction columns while preserving raw columns."""
    columns = frame.collect_schema().names()
    if _ROW_INDEX in columns:
        raise ValueError(f"Input column {_ROW_INDEX!r} is reserved")

    source = _resolve_column(columns, source_column, _ALIASES["source"])
    target = _resolve_column(columns, target_column, _ALIASES["target"])
    if source is None or target is None:
        raise ValueError("Transactions require source and target columns")

    frame = frame.with_row_index(_ROW_INDEX)
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
    return frame.with_columns(expressions).drop(_ROW_INDEX)


def _transaction_id_expression(frame: pl.LazyFrame, column: str | None) -> pl.Expr:
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
    direct = _resolve_column(columns, requested, ("timestamp", "datetime"))
    if direct is not None:
        return pl.col(direct)

    date = _resolve_column(columns, None, ("date",))
    time = _resolve_column(columns, None, ("time",))
    if date is not None and time is not None:
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
    normalized = expression.str.replace("T", " ")
    return pl.coalesce(
        normalized.str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False),
        normalized.str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False),
        normalized.str.to_datetime(format="%Y-%m-%d %H:%M", strict=False),
        normalized.str.to_datetime(format="%Y-%m-%d", strict=False),
    )


def _normalize_column(column: str) -> str:
    return " ".join(column.lower().replace("_", " ").replace(".", " ").split())
