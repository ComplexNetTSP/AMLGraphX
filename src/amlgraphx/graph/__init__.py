"""Explicit graph representations and preparation APIs for AMLGraphX."""

from .api import (
    GraphBuildSpec,
    GraphFeatureSpec,
    GraphNodeType,
    GraphTemporalMode,
    prepare_graph,
    prepare_pyg_graph,
)
from .builders import (
    AccountGraph,
    TransactionGraph,
    build_account_graph,
    build_precomputed_transaction_graph,
    build_time_aware_account_graph,
    build_transaction_graph,
)
from .pyg import to_pyg_data, to_pyg_temporal_data
from .temporal import (
    AccountEventStream,
    GraphSnapshot,
    build_account_event_stream,
    build_account_snapshots,
    build_transaction_snapshots,
)

__all__ = [
    "AccountEventStream",
    "AccountGraph",
    "GraphBuildSpec",
    "GraphFeatureSpec",
    "GraphNodeType",
    "GraphSnapshot",
    "GraphTemporalMode",
    "TransactionGraph",
    "build_account_event_stream",
    "build_account_graph",
    "build_account_snapshots",
    "build_precomputed_transaction_graph",
    "build_time_aware_account_graph",
    "build_transaction_graph",
    "build_transaction_snapshots",
    "prepare_graph",
    "prepare_pyg_graph",
    "to_pyg_data",
    "to_pyg_temporal_data",
]
