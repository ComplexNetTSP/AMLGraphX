"""Convert AMLGraphX representations to standard PyTorch Geometric objects.

PyG 2.8 ``Data`` accepts arbitrary attributes through ``**kwargs`` and
``TemporalData`` already defines ``src``, ``dst``, ``t``, and ``msg``.
AMLGraphX therefore uses the standard classes directly and stores static graph
timestamps as ``data.edge_time``; no custom PyG subclass is required.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import torch
from torch import Tensor
from torch_geometric.data import Data, TemporalData

from .graphs import AccountGraph, TransactionGraph
from .temporal.event_stream import AccountEventStream

type HomogeneousGraph = AccountGraph | TransactionGraph


def to_pyg_data(
    graph: HomogeneousGraph,
    *,
    node_feature_columns: Sequence[str] = (),
    edge_feature_columns: Sequence[str] = (),
    node_label_column: str | None = None,
    edge_label_column: str | None = None,
) -> Data:
    """Convert either account-node or transaction-node graphs to PyG ``Data``.

    Both graph types use the same output contract: ``edge_index`` is sparse
    COO, selected numerical features become ``x`` and ``edge_attr``, and
    temporal edges expose integer nanoseconds through ``edge_time``. Duration
    edge features such as ``time_delta`` are converted to seconds.

    String and categorical encoding remains an explicit preprocessing choice;
    only requested numerical columns are accepted here.
    """
    _validate_graph_type(graph)
    node_columns, source_column, target_column = _graph_identity_columns(graph)
    indexed_edges = _indexed_edges(
        graph.nodes,
        graph.edges,
        node_id_column=node_columns,
        source_column=source_column,
        target_column=target_column,
    )
    edge_index = _edge_index(indexed_edges)
    node_features = _feature_tensor(
        graph.nodes, node_feature_columns, "node_feature_columns"
    )
    edge_features = _feature_tensor(
        graph.edges, edge_feature_columns, "edge_feature_columns"
    )
    edge_time = _graph_edge_time(graph)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        edge_time=edge_time,
    )
    # Explicit node count preserves isolated nodes even when x is omitted.
    data.num_nodes = graph.num_nodes
    if isinstance(graph, TransactionGraph):
        # A transaction graph is node-temporal. Static-window loaders use this
        # tensor to retain only the configured causal history for each target.
        data.node_time = _time_tensor(graph.nodes["timestamp"])
    if node_label_column is not None:
        data.node_y = _label_tensor(graph.nodes, node_label_column)
    if edge_label_column is not None:
        data.edge_y = _label_tensor(graph.edges, edge_label_column)
    data.validate()
    return data


def to_pyg_temporal_data(
    stream: AccountEventStream,
    *,
    node_feature_columns: Sequence[str] = (),
    message_columns: Sequence[str] = (),
    label_column: str | None = None,
) -> TemporalData:
    """Convert an account event stream to PyG ``TemporalData``.

    ``node_feature_columns`` selects numerical account metadata for ``x``.
    ``message_columns`` selects numerical transaction edge features for
    ``msg``. An empty message selection produces an ``[num_events, 0]`` float
    matrix, allowing models to add their own message encoder later. PyG's
    ``TemporalData`` derives ``num_nodes`` from event endpoints, so AMLGraphX
    does not assign it as a Python integer: that would break PyG's event
    slicing. Account metadata in ``x`` can still contain isolated accounts.
    """
    if not isinstance(stream, AccountEventStream):
        raise TypeError("stream must be an AccountEventStream")

    indexed = _indexed_edges(
        stream.nodes,
        stream.events,
        node_id_column="node_id",
        source_column="source",
        target_column="target",
    )
    message = _feature_tensor(stream.events, message_columns, "message_columns")
    node_features = _feature_tensor(
        stream.nodes, node_feature_columns, "node_feature_columns"
    )
    if message is None:
        message = torch.empty((stream.num_events, 0), dtype=torch.float32)

    kwargs: dict[str, Tensor] = {}
    if label_column is not None:
        kwargs["y"] = _label_tensor(stream.events, label_column)
    data = TemporalData(
        src=torch.tensor(indexed["source_index"].to_numpy(), dtype=torch.long),
        dst=torch.tensor(indexed["target_index"].to_numpy(), dtype=torch.long),
        t=_time_tensor(stream.events["timestamp"]),
        msg=message,
        **kwargs,
    )
    if node_features is not None:
        data.x = node_features
    return data


def _validate_graph_type(graph: HomogeneousGraph) -> None:
    """Reject objects whose endpoint semantics are unknown."""
    if not isinstance(graph, AccountGraph | TransactionGraph):
        raise TypeError("graph must be an AccountGraph or TransactionGraph")


def _graph_identity_columns(graph: HomogeneousGraph) -> tuple[str, str, str]:
    """Return node, source-edge, and target-edge ID columns."""
    if isinstance(graph, AccountGraph):
        return "node_id", "source", "target"
    return "transaction_id", "source_transaction_id", "target_transaction_id"


def _indexed_edges(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    *,
    node_id_column: str,
    source_column: str,
    target_column: str,
) -> pl.DataFrame:
    """Map stable string IDs to contiguous indices while preserving edge order."""
    node_indices = nodes.select(node_id_column).with_row_index("node_index")
    source_indices = node_indices.rename(
        {node_id_column: source_column, "node_index": "source_index"}
    )
    target_indices = node_indices.rename(
        {node_id_column: target_column, "node_index": "target_index"}
    )
    indexed = (
        edges.with_row_index("__edge_order")
        .join(source_indices, on=source_column, how="inner")
        .join(target_indices, on=target_column, how="inner")
        .sort("__edge_order")
    )
    if indexed.height != edges.height:
        raise ValueError("graph edges reference nodes that are not present")
    return indexed


def _edge_index(indexed_edges: pl.DataFrame) -> Tensor:
    """Create a contiguous ``[2, E]`` long tensor."""
    if indexed_edges.is_empty():
        return torch.empty((2, 0), dtype=torch.long)
    values = indexed_edges.select("source_index", "target_index").to_numpy()
    return torch.tensor(values.T, dtype=torch.long).contiguous()


def _feature_tensor(
    frame: pl.DataFrame,
    columns: Sequence[str],
    argument_name: str,
) -> Tensor | None:
    """Convert explicitly selected numerical columns to float model features."""
    if not columns:
        return None
    _require_columns(frame, columns, argument_name)

    expressions: list[pl.Expr] = []
    for column in columns:
        dtype = frame.schema[column]
        if dtype.is_numeric() or dtype == pl.Boolean:
            expressions.append(pl.col(column).cast(pl.Float32))
            continue
        if dtype.base_type() == pl.Duration:
            seconds = pl.col(column).cast(pl.Int64).cast(pl.Float64) / 1_000_000_000
            expressions.append(seconds.cast(pl.Float32).alias(column))
            continue
        raise TypeError(f"{argument_name} column {column!r} must be numerical")

    values = frame.select(expressions).to_numpy()
    return torch.tensor(values, dtype=torch.float32).contiguous()


def _label_tensor(frame: pl.DataFrame, column: str) -> Tensor:
    """Convert one numerical label column without changing integer classes."""
    _require_columns(frame, [column], "label")
    dtype = frame.schema[column]
    if not (dtype.is_numeric() or dtype == pl.Boolean):
        raise TypeError(f"label column {column!r} must be numerical")
    return torch.tensor(frame[column].to_numpy()).contiguous()


def _graph_edge_time(graph: HomogeneousGraph) -> Tensor | None:
    """Return when each graph edge becomes observable, in nanoseconds."""
    if isinstance(graph, AccountGraph):
        if "timestamp" not in graph.edges.columns:
            return None
        return _time_tensor(graph.edges["timestamp"])

    if "timestamp" not in graph.nodes.columns:
        return None
    if graph.edges.is_empty():
        return torch.empty(0, dtype=torch.long)

    target_times = graph.nodes.select(
        pl.col("transaction_id").alias("target_transaction_id"),
        pl.col("timestamp").alias("edge_time"),
    )
    timed_edges = (
        graph.edges.with_row_index("__edge_order")
        .join(target_times, on="target_transaction_id", how="inner")
        .sort("__edge_order")
    )
    return _time_tensor(timed_edges["edge_time"])


def _time_tensor(values: pl.Series) -> Tensor:
    """Convert Polars Datetime or Date values to integer nanoseconds."""
    dtype = values.dtype
    if dtype.base_type() == pl.Date:
        values = values.cast(pl.Datetime("ns"))
        dtype = values.dtype
    if dtype.base_type() != pl.Datetime:
        raise TypeError("temporal values must use Polars Date or Datetime dtype")

    scale = {"ms": 1_000_000, "us": 1_000, "ns": 1}[dtype.time_unit]
    nanoseconds = values.cast(pl.Int64).to_numpy() * scale
    return torch.tensor(nanoseconds, dtype=torch.long).contiguous()


def _require_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
    argument_name: str,
) -> None:
    """Raise one readable error for missing feature columns."""
    missing = set(columns).difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{argument_name} are missing: {names}")


__all__ = ["to_pyg_data", "to_pyg_temporal_data"]
