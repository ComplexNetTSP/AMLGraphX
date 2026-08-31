"""AMLGraphX public package exports."""

from .graph import (
    AccountEventStream,
    AccountGraph,
    GraphBuildSpec,
    GraphNodeType,
    GraphTemporalMode,
    TransactionGraph,
    build_account_event_stream,
    build_account_graph,
    build_account_snapshots,
    build_precomputed_transaction_graph,
    build_time_aware_account_graph,
    build_transaction_graph,
    prepare_graph,
    to_pyg_data,
    to_pyg_temporal_data,
)
from .split import TemporalNodeMasks, TemporalSplit, build_temporal_node_masks


def hello() -> str:
    """Return the package greeting kept for backwards compatibility."""
    return "Hello from amlgraphx!"


__all__ = [
    "AccountEventStream",
    "AccountGraph",
    "GraphBuildSpec",
    "GraphNodeType",
    "GraphTemporalMode",
    "TemporalNodeMasks",
    "TemporalSplit",
    "TransactionGraph",
    "build_account_event_stream",
    "build_account_graph",
    "build_account_snapshots",
    "build_precomputed_transaction_graph",
    "build_temporal_node_masks",
    "build_time_aware_account_graph",
    "build_transaction_graph",
    "hello",
    "prepare_graph",
    "to_pyg_data",
    "to_pyg_temporal_data",
]
