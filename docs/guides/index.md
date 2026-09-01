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

For account nodes, use `temporal="static"`, `"snapshot"`, or `"event_stream"`. Transaction nodes support static and snapshot views only.

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

Snapshots are a sequence of separate time windows. They are not an event stream. For a continuous account interaction stream, request `temporal="event_stream"` and convert it with `to_pyg_temporal_data()` when the model expects PyTorch Geometric `TemporalData`.

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

## Convert to PyTorch Geometric deliberately

```python
from amlgraphx.graph import to_pyg_data

data = to_pyg_data(
    transaction_graph,
    node_feature_columns=["amount"],
    node_label_column="label",
)
```

Only numerical columns can be converted as model features. Encode categories and fit normalizers using the training period only; AMLGraphX intentionally does not make those potentially leaky decisions on your behalf.

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
