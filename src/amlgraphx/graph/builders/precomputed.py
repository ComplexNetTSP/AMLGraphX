"""Import transaction-node graphs whose edges are supplied by a dataset.

This is deliberately separate from :func:`build_transaction_graph`: the latter
derives money-flow successor edges from account transfers, whereas datasets
such as Elliptic already publish their transaction relation.  Re-deriving that
relation would silently change the experiment.
"""

from datetime import timedelta

import polars as pl

from amlgraphx.data.temporal import logical_timestamp_from_step

from ..graphs import TransactionGraph, TransactionTable


def build_precomputed_transaction_graph(
    nodes: TransactionTable,
    edges: TransactionTable,
    *,
    node_id_column: str,
    edge_source_column: str,
    edge_target_column: str,
    time_column: str | None = None,
    step_size: timedelta | None = None,
) -> TransactionGraph:
    """Build a transaction-node graph from a dataset-provided edge list.

    Nodes retain all supplied feature columns.  Edges retain all supplied edge
    attributes and gain canonical endpoint columns.  When ``time_column`` is
    an ordinal value, ``step_size`` explicitly maps it to a logical datetime;
    the raw column remains available to the caller.

    Args:
        nodes: Transaction-node table.
        edges: Directed transaction-to-transaction relation table.
        node_id_column: Unique transaction identifier in ``nodes``.
        edge_source_column: Earlier/source transaction identifier in ``edges``.
        edge_target_column: Later/target transaction identifier in ``edges``.
        time_column: Optional datetime or ordinal node-time column.
        step_size: Required positive duration for an ordinal ``time_column``.

    Returns:
        A ``TransactionGraph`` preserving the supplied relation rather than
        inferring account-continuation edges.
    """
    node_frame = _collect(nodes, "nodes")
    edge_frame = _collect(edges, "edges")
    _require_columns(node_frame, (node_id_column,), "nodes")
    _require_columns(edge_frame, (edge_source_column, edge_target_column), "edges")

    node_frame = node_frame.with_columns(
        pl.col(node_id_column).cast(pl.String).str.strip_chars().alias("transaction_id")
    )
    if (
        node_frame["transaction_id"].null_count()
        or (node_frame["transaction_id"] == "").any()
    ):
        raise ValueError("transaction node IDs must be non-empty")
    if node_frame["transaction_id"].n_unique() != node_frame.height:
        raise ValueError("transaction node IDs must be unique")

    if time_column is not None:
        _require_columns(node_frame, (time_column,), "nodes")
        node_frame = _canonical_node_time(node_frame, time_column, step_size)

    edge_frame = edge_frame.with_columns(
        pl.col(edge_source_column)
        .cast(pl.String)
        .str.strip_chars()
        .alias("source_transaction_id"),
        pl.col(edge_target_column)
        .cast(pl.String)
        .str.strip_chars()
        .alias("target_transaction_id"),
        pl.lit("precomputed").alias("edge_relation"),
    )
    _validate_edge_endpoints(node_frame, edge_frame)
    return TransactionGraph(nodes=node_frame, edges=edge_frame)


def _canonical_node_time(
    nodes: pl.DataFrame,
    column: str,
    step_size: timedelta | None,
) -> pl.DataFrame:
    """Return nodes with a typed timestamp from datetime or ordinal time."""
    dtype = nodes.schema[column]
    if dtype.base_type() == pl.Datetime:
        return nodes.with_columns(pl.col(column).alias("timestamp"))
    if dtype.base_type() == pl.Date:
        return nodes.with_columns(pl.col(column).cast(pl.Datetime).alias("timestamp"))
    if dtype.is_numeric():
        if step_size is None:
            raise ValueError("step_size is required when time_column is ordinal")
        return logical_timestamp_from_step(
            nodes.lazy(), step_column=column, step_size=step_size
        ).collect()
    raise TypeError("time_column must be a Polars Date, Datetime, or numeric step")


def _validate_edge_endpoints(nodes: pl.DataFrame, edges: pl.DataFrame) -> None:
    """Reject an edge list that references transaction nodes not supplied."""
    ids = nodes.select("transaction_id")
    for column in ("source_transaction_id", "target_transaction_id"):
        missing = edges.select(column).join(
            ids.rename({"transaction_id": column}), on=column, how="anti"
        )
        if not missing.is_empty():
            raise ValueError(f"edges reference unknown transaction IDs in {column}")


def _collect(table: TransactionTable, name: str) -> pl.DataFrame:
    """Materialize one supported Polars table without mutating its caller."""
    if isinstance(table, pl.LazyFrame):
        return table.collect()
    if isinstance(table, pl.DataFrame):
        return table.clone()
    raise TypeError(f"{name} must be a polars.DataFrame or polars.LazyFrame")


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...], name: str) -> None:
    """Raise one readable error for a missing public input column."""
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} are missing: {', '.join(sorted(missing))}")


__all__ = ["build_precomputed_transaction_graph"]
