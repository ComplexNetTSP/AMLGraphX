# Guides

These workflows compose public AMLGraphX components. They do not hide the graph or temporal choices that determine an experiment's meaning.

## Choose a graph representation

Use `prepare_graph()` when you want one explicit dispatcher for node and time semantics:

```python
from datetime import timedelta

from amlgraphx.graph import prepare_graph

transaction_graph = prepare_graph(
    transactions,
    node_type="transaction",
    temporal="static",
    edge_delta=timedelta(hours=4),
)
```

For account nodes, use `temporal="static"`, `"snapshot"`, or `"event_stream"`.
Transaction nodes support `temporal="static"`: their identities do not persist
between windows, so a transaction window is a batching strategy rather than a
snapshot evolution. A transaction-node event stream would require explicit
node-arrival semantics and is not currently implemented.

## Build snapshots or an event stream

```python
from datetime import timedelta

from amlgraphx.graph import prepare_graph

daily_snapshots = prepare_graph(
    transactions,
    node_type="account",
    temporal="snapshot",
    bin_size=timedelta(days=1),
    stride=timedelta(days=1),
)

for snapshot in daily_snapshots:
    print(snapshot.index, snapshot.start_time, snapshot.end_time)
```

Account snapshots are a sequence of graph states with stable account identity.
They are not an event stream. For continuous account interactions, request
`temporal="event_stream"`.

## Use a strict chronological split

```python
from datetime import datetime, timedelta

from amlgraphx.graph import build_transaction_graph
from amlgraphx.split import TemporalSplit, apply_temporal_split

graph = build_transaction_graph(transactions, delta=timedelta(hours=4))
split = TemporalSplit(
    train_end=datetime(2025, 2, 1),
    validation_end=datetime(2025, 3, 1),
)
partitions = apply_temporal_split(graph, split)

train_graph = partitions.train
validation_graph = partitions.validation
test_graph = partitions.test
```

Each result is an induced transaction graph. Cross-partition edges are removed, so a test target cannot use a training-period edge solely because the complete graph was built first. If you intentionally need full-graph transductive structure, use `build_temporal_node_masks()` instead and state that protocol when reporting results.

## Build model-ready features through one API

```python
from amlgraphx.graph import GraphFeatureSpec, prepare_pyg_graph

data = prepare_pyg_graph(
    transactions,
    node_type="transaction",
    temporal="static",
    edge_delta=timedelta(hours=4),
    features=GraphFeatureSpec(
        node_columns=("amount",),
        edge_columns=("time_delta",),
        label_column="label",
    ),
)
```

The facade assigns labels according to graph semantics: transaction labels are
`edge_y` in account graphs and `node_y` in transaction graphs. Transaction
attributes are account-graph edge features but transaction-graph node features;
`time_delta` describes a relation between two transaction nodes. Only numerical
columns are converted directly. Encode categories and fit normalizers using the
training period only; AMLGraphX does not make those potentially leaky decisions
on your behalf.

## Enrich tabular features from transaction history

`GraphFeaturePreprocessor` accepts a numeric matrix whose first four columns are `[edge_id, source_id, target_id, timestamp]`; later columns are numeric transaction features. It appends graph-derived features suitable for a tabular estimator.

```python
from amlgraphx.tabular import GraphFeaturePreprocessor
from amlgraphx.baselines import XGBoostBaseline

preprocessor = GraphFeaturePreprocessor()
preprocessor.fit(history_features)
X_train = preprocessor.transform(train_features)

model = XGBoostBaseline(scale_pos_weight=20)
model.fit(X_train[:, 3:], train_labels)
scores = model.predict_proba(X_train[:, 3:])[:, 1]
```

`transform()` follows batch semantics: rows in one batch can observe each other. Use `transform_causal()` for strict event-by-event features, and ensure events are strictly time ordered. The three identifier columns are graph identities, not estimator features, so remove them before fitting a tabular model.

## Current scope

`amlgraphx.evaluation` now provides labelled binary risk-score metrics: Average
Precision, ROC-AUC, fixed-threshold Precision/Recall/F1, and investigation
budget metrics such as Precision@K and Recall@K. See
`src/amlgraphx/evaluation/metrics.md` for their exact denominators and temporal
evaluation constraints. Torch metrics are independent ``update()`` /
``compute()`` / ``reset()`` objects, so a training engine can receive only the
named metrics needed for an experiment. Neural models, sampling, training,
tracking, and tuning remain under active development.
