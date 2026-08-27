"""Continuous account-to-account transaction event streams.

Account-as-node has a direct continuous-time interpretation: every original
transaction is one directed event. The event table keeps all transaction
columns, so amount and any private dataset features remain available without
forcing an early tensor encoding.

账户为节点时，一笔原始交易天然就是一个连续时间事件。事件表保留所有交易列，
因此金额和私有数据字段不会在转换成模型输入前丢失。
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..builders.account import AccountGraph, build_time_aware_account_graph
from ..graphs import TransactionTable


@dataclass(frozen=True, slots=True)
class AccountEventStream:
    """Hold timestamp-ordered account interaction events.

    ``nodes`` stores stable account IDs and optional account metadata.
    ``events`` stores one row per transaction with canonical ``source``,
    ``target``, ``transaction_id``, and ``timestamp`` columns plus every input
    edge feature. Events are sorted stably by time.
    """

    nodes: pl.DataFrame
    events: pl.DataFrame

    @classmethod
    def from_graph(cls, graph: AccountGraph) -> AccountEventStream:
        """Create a stream without rebuilding an existing account graph."""
        if not isinstance(graph, AccountGraph):
            raise TypeError("graph must be an AccountGraph")
        if "timestamp" not in graph.edges.columns:
            raise ValueError("account graph edges must contain timestamp")
        if graph.edges.schema["timestamp"].base_type() != pl.Datetime:
            raise TypeError("account graph timestamp must be a Polars Datetime")

        events = graph.edges.sort("timestamp", maintain_order=True)
        return cls(nodes=graph.nodes, events=events)

    @property
    def num_nodes(self) -> int:
        """Return the number of account nodes."""
        return self.nodes.height

    @property
    def num_events(self) -> int:
        """Return the number of transaction events."""
        return self.events.height


def build_account_event_stream(
    transactions: TransactionTable,
    *,
    account_metadata: TransactionTable | None = None,
    source_column: str | None = None,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    transaction_id_column: str | None = None,
    account_id_column: str | None = None,
) -> AccountEventStream:
    """Build a chronological account interaction stream from transactions.

    Returns:
        An ``AccountEventStream`` whose events preserve all transaction edge
        features and use half-open-safe, typed timestamps for later splitting.
    """
    graph = build_time_aware_account_graph(
        transactions,
        account_metadata=account_metadata,
        source_column=source_column,
        target_column=target_column,
        timestamp_column=timestamp_column,
        transaction_id_column=transaction_id_column,
        account_id_column=account_id_column,
    )
    return AccountEventStream.from_graph(graph)


__all__ = ["AccountEventStream", "build_account_event_stream"]
