# Quickstart

This example creates a small transaction table, normalizes it, builds an account graph, and converts the graph to PyTorch Geometric. It uses in-memory data so that the graph semantics are visible without downloading a dataset.

## 1. Create a canonical transaction table

```python
import polars as pl

from amlgraphx.data import normalize_transactions

raw_transactions = pl.DataFrame(
    {
        "sender": ["A", "B", "A"],
        "receiver": ["B", "C", "C"],
        "timestamp": ["2025-01-01 09:00:00", "2025-01-01 10:00:00", "2025-01-01 11:00:00"],
        "amount": [100.0, 80.0, 25.0],
        "is_fraud": [0, 1, 0],
    }
)

transactions = normalize_transactions(raw_transactions.lazy()).collect()
```

`normalize_transactions()` preserves the original columns and adds canonical `transaction_id`, `source`, and `target` columns. It also adds `timestamp`, `amount`, and `label` whenever it can identify matching source columns.

## 2. Build an account graph

```python
from amlgraphx.graph import prepare_graph

graph = prepare_graph(transactions, node_type="account", temporal="static")

print(graph.num_nodes)  # 3
print(graph.num_edges)  # 3
```

In an account graph, accounts are nodes and every transaction remains one directed edge. Parallel transactions are retained: two transfers between the same accounts are still two separate observations.

## 3. Convert explicit numerical features

```python
from amlgraphx.graph import to_pyg_data

data = to_pyg_data(
    graph,
    edge_feature_columns=["amount"],
    edge_label_column="label",
)

print(data.edge_index.shape)  # [2, number_of_transactions]
print(data.edge_attr.shape)   # [number_of_transactions, 1]
```

Feature selection is explicit. AMLGraphX does not silently encode string IDs, categories, or future-aware aggregates before they reach a model.

## Next steps

- Read [Graph and temporal semantics](../concepts/index) before using a transaction graph or time-based evaluation.
- Use a supported public dataset through the [dataset adapters](../datasets/index).
- Follow the [guides](../guides/index) to create snapshots, event streams, strict temporal splits, or tabular graph-feature baselines.
