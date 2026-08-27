"""AMLGraphX public package exports."""

from .graph import (
    AccountGraph,
    GraphBuildSpec,
    GraphNodeType,
    GraphTemporalMode,
    TransactionGraph,
    build_account_graph,
    build_transaction_graph,
    prepare_graph,
)
from .split import TemporalNodeMasks, TemporalSplit, build_temporal_node_masks


def hello() -> str:
    """Return the package greeting kept for backwards compatibility."""
    return "Hello from amlgraphx!"


__all__ = [
    "AccountGraph",
    "GraphBuildSpec",
    "GraphNodeType",
    "GraphTemporalMode",
    "TemporalNodeMasks",
    "TemporalSplit",
    "TransactionGraph",
    "build_account_graph",
    "build_temporal_node_masks",
    "build_transaction_graph",
    "hello",
    "prepare_graph",
]
