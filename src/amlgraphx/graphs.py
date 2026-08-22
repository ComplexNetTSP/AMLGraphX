"""Graph views built from AMLGraphX transaction tables."""

from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

type TransactionTable = pl.DataFrame | pl.LazyFrame

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


@dataclass(frozen=True, slots=True)
class AccountGraph:
    """Represent accounts as nodes and transactions as directed edges.

    Args:
        nodes: Account table containing at least ``node_id``.
        edges: Transaction table containing ``source`` and ``target``.
    """

    nodes: pl.DataFrame
    edges: pl.DataFrame

    @property
    def num_nodes(self) -> int:
        """Return the number of account nodes."""
        return self.nodes.height

    @property
    def num_edges(self) -> int:
        """Return the number of transaction edges."""
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
        frame, source, target = _prepare_transactions(
            transactions,
            source_column=source_column,
            target_column=target_column,
            transaction_id_column=transaction_id_column,
        )

        edge_columns = [
            column
            for column in frame.columns
            if column != _ROW_INDEX
        ]
        edges = frame.select(edge_columns)
        nodes = pl.concat(
            [
                frame.select(pl.col(source).alias("node_id")),
                frame.select(pl.col(target).alias("node_id")),
            ]
        ).unique(subset=["node_id"], maintain_order=True).sort("node_id")

        if account_metadata is not None:
            nodes = _join_account_metadata(
                nodes,
                account_metadata,
                account_id_column=account_id_column,
            )

        return cls(nodes=nodes, edges=edges)


@dataclass(frozen=True, slots=True)
class TransactionGraph:
    """Represent transactions as nodes linked by temporal money flow.

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
        """Return the number of transaction nodes."""
        return self.nodes.height

    @property
    def num_edges(self) -> int:
        """Return the number of temporal succession edges."""
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

        frame, source, target = _prepare_transactions(
            transactions,
            source_column=source_column,
            target_column=target_column,
            timestamp_column=timestamp_column,
            transaction_id_column=transaction_id_column,
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
        frame = frame.filter(pl.col("timestamp").is_not_null())

        node_columns = [
            column
            for column in frame.columns
            if column != _ROW_INDEX
        ]
        nodes = frame.select(node_columns)
        ordered = frame.sort(["timestamp", _ROW_INDEX])
        rows = list(ordered.iter_rows(named=True))

        outgoing: dict[str, list[tuple[datetime, int]]] = {}
        outgoing_times: dict[str, list[datetime]] = {}
        for position, row in enumerate(rows):
            account = row[source]
            timestamp = row["timestamp"]
            outgoing.setdefault(account, []).append((timestamp, position))
            outgoing_times.setdefault(account, []).append(timestamp)

        edge_records: list[dict[str, object]] = []
        for row in rows:
            candidates = outgoing.get(row[target], [])
            candidate_times = outgoing_times.get(row[target], [])
            start = bisect_right(candidate_times, row["timestamp"])
            end = bisect_right(
                candidate_times,
                row["timestamp"] + delta,
            )
            for _, successor_position in candidates[start:end]:
                successor = rows[successor_position]
                edge_records.append(
                    {
                        "source_transaction_id": row["transaction_id"],
                        "target_transaction_id": successor["transaction_id"],
                        "via_account": row[target],
                        "time_delta": successor["timestamp"] - row["timestamp"],
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
) -> tuple[pl.DataFrame, str, str]:
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

    frame = frame.with_row_index(_ROW_INDEX)
    id_column = _resolve_optional_column(
        frame.columns,
        transaction_id_column,
        _TRANSACTION_ID_ALIASES,
    )
    frame = frame.with_columns(
        _make_transaction_ids(frame, id_column).alias("transaction_id")
    )

    frame = frame.with_columns(
        pl.col(source)
        .cast(pl.String)
        .str.strip_chars()
        .alias("source"),
        pl.col(target)
        .cast(pl.String)
        .str.strip_chars()
        .alias("target"),
    ).filter(
        pl.col("source").is_not_null()
        & (pl.col("source") != "")
        & pl.col("target").is_not_null()
        & (pl.col("target") != "")
    )

    if timestamp_column is not None or _has_timestamp_column(frame.columns):
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
    if isinstance(transactions, pl.LazyFrame):
        return transactions.collect()
    if isinstance(transactions, pl.DataFrame):
        return transactions.clone()
    raise TypeError("transactions must be a polars.DataFrame or polars.LazyFrame")


def _make_transaction_ids(
    frame: pl.DataFrame,
    id_column: str | None,
) -> pl.Series:
    if id_column is None:
        return pl.Series(
            "transaction_id",
            [f"tx_{row_index}" for row_index in frame[_ROW_INDEX].to_list()],
            dtype=pl.String,
        )

    values = (
        frame.get_column(id_column)
        .cast(pl.String)
        .str.strip_chars()
        .to_list()
    )
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
    metadata = _collect_frame(account_metadata)
    id_column = _resolve_required_column(
        metadata.columns,
        account_id_column,
        _ACCOUNT_ID_ALIASES,
        "account metadata ID",
    )
    metadata = (
        metadata.with_columns(
            pl.col(id_column).cast(pl.String).str.strip_chars().alias("node_id")
        )
        .unique(subset=["node_id"], maintain_order=True)
    )
    if id_column != "node_id":
        metadata = metadata.drop(id_column)
    return nodes.join(metadata, on="node_id", how="left")


def _resolve_required_column(
    columns: Sequence[str],
    requested: str | None,
    aliases: Sequence[str],
    logical_name: str,
) -> str:
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
    return " ".join(column.lower().replace("_", " ").replace(".", " ").split())


def _has_timestamp_column(columns: Sequence[str]) -> bool:
    return _resolve_optional_column(columns, None, _TIMESTAMP_ALIASES) is not None


def _timestamp_expression(frame: pl.DataFrame, column: str) -> pl.Expr:
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
            date_only = pl.col(date_column).cast(pl.String).str.replace(
                r"[T ].*$", ""
            )
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


def _parse_datetime_strings(expression: pl.Expr) -> pl.Expr:
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
    if records:
        return pl.DataFrame(records)
    return pl.DataFrame(
        {
            "source_transaction_id": pl.Series([], dtype=pl.String),
            "target_transaction_id": pl.Series([], dtype=pl.String),
            "via_account": pl.Series([], dtype=pl.String),
            "time_delta": pl.Series([], dtype=pl.Duration("us")),
        }
    )
