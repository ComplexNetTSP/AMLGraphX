"""AMLGraphX public package exports."""

from .graphs import (
    AccountGraph,
    TransactionGraph,
    build_account_graph,
    build_transaction_graph,
)


def hello() -> str:
    """Return the package greeting kept for backwards compatibility."""
    return "Hello from amlgraphx!"


__all__ = [
    "AccountGraph",
    "TransactionGraph",
    "build_account_graph",
    "build_transaction_graph",
    "hello",
]
