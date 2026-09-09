# Core concepts

AMLGraphX keeps data, graph construction, temporal representation, splitting, and modelling separate. This avoids silently changing the scientific meaning of an experiment when a different model is introduced.

## Canonical transactions

External AML and fraud datasets use different names for the same idea. Dataset adapters and `normalize_transactions()` preserve raw fields and add discoverable canonical columns:

| Canonical column | Meaning |
| --- | --- |
| `transaction_id` | Stable transaction identity; generated deterministically when needed. |
| `source` | Sending account or origin entity. |
| `target` | Receiving account or destination entity. |
| `timestamp` | Event time, when the source data provides one. |
| `amount` | Transaction value, when available. |
| `label` | Transaction label, when available. |

Canonicalization is schema adaptation, not feature engineering. It does not drop dataset-specific columns or invent missing values.

## Graph node semantics

Choosing a graph changes the learning problem. AMLGraphX does not hide these two meanings behind one ambiguous graph object.

| Representation | Node | Edge | Typical use |
| --- | --- | --- | --- |
| Account graph | Account | A directed transaction from source to target | Account neighbourhoods or transaction-risk models |
| Transaction graph | Transaction | A feasible later transfer after money reaches an account | Transaction flow continuation and temporal motifs |

An account graph retains every input transaction as an edge, including repeated transfers and self-loops. A transaction graph retains every transaction as a node. It adds an edge from transaction `i` to a later transaction `j` when the receiver of `i` is the sender of `j` and `0 < t_j - t_i <= edge_delta`. Simultaneous transactions are not connected.

## Temporal representations

Time-aware graph construction and temporal data representation are distinct choices.

| Mode | Account-as-node | Transaction-as-node |
| --- | --- | --- |
| `static` | Supported: transactions are time-aware edges. | Supported: time constrains directed transaction relations. |
| `snapshot` | Supported: stable accounts recur across graph states. | Not a high-level mode; windows are only a batching utility. |
| `event_stream` | Supported: each transaction is one account-to-account event. | Not implemented; it requires node-arrival and isolated-transaction semantics. |

Account snapshot windows use `[start_time, end_time)` semantics. Equal
`bin_size` and `stride` values create disjoint windows; a smaller stride creates
overlap. Account event streams preserve one event per transaction and are
stably ordered by time.

## Temporal evaluation and leakage

Graph construction is not a split policy. AMLGraphX provides two different chronological protocols, and a research result should name the one it uses.

| Protocol | API | Meaning |
| --- | --- | --- |
| Strict induced subgraphs | `TemporalSplit` + `apply_temporal_split()` | Builds separate train, validation, and test transaction graphs; edges crossing a split boundary are removed. |
| Full graph with chronological masks | `TemporalSplit` + `build_temporal_node_masks()` | Keeps one complete graph and marks node targets by time; structural information remains visible across periods. |

The intervals are `[-∞, train_end)`, `[train_end, validation_end)`, and `[validation_end, +∞)`. Do not describe the full-graph mask protocol as strict causal evaluation. For temporal snapshots that need prior observations, `TransactionGraphDataModule` can preserve explicit lookback context while marking prediction targets.

## From tables to model inputs

AMLGraphX graph objects retain Polars tables with stable IDs and original
features. `GraphFeatureSpec` and `prepare_pyg_graph()` provide the model-ready
high-level path:

| Representation | Node features and labels | Edge features and labels |
| --- | --- | --- |
| Account graph | Account metadata | Transaction attributes and transaction label |
| Transaction graph | Transaction attributes and transaction label | Relation attributes such as `time_delta` |
| Account event stream | Account metadata | Transaction attributes become `msg`; label becomes `y` |

The lower-level `to_pyg_data()` and `to_pyg_temporal_data()` functions remain
available when a prepared graph needs custom conversion. Only explicitly
selected numerical columns are converted. Categorical encoding, normalization,
target masks, and model selection remain explicit research decisions.

## Training batches

Batching preserves the selected temporal representation instead of converting
all data into one generic sequence. A bounded static graph window is an
ordinary PyG `Batch`: each window is a disconnected component and has either a
`target_node_mask` (transaction graph) or `target_edge_mask` (account graph).
An account event stream uses PyG's chronological `TemporalDataLoader`.

An account snapshot batch has one PyG `Batch` at each time position:

```text
context[0] = Batch(Ga[t-5], Gb[t-5], ...)
...
context[4] = Batch(Ga[t-1], Gb[t-1], ...)
target     = Batch(Ga[t],   Gb[t],   ...)
```

The graph components at a single time position execute in parallel. The tuple
keeps the time axis explicit for a researcher-defined temporal model. Static
windows can be shuffled when the model has no cross-window state; event streams
and stateful snapshot models must remain chronological.
