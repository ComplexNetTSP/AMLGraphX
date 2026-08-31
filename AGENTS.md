# AGENTS.md

This file defines the repository-level working rules for AI coding agents such as Codex.

AMLGraphX is a research-oriented open-source Python library for:

- anti-money laundering,
- fraud detection,
- transaction-network analysis,
- graph machine learning,
- temporal graph learning,
- reproducible financial-crime research.

The repository must evolve as a reusable scientific library rather than as a collection of paper-specific scripts.

The highest-level priorities are:

1. scientific correctness,
2. explicit graph and temporal semantics,
3. stable public APIs,
4. clear module responsibility,
5. composability,
6. reproducibility,
7. testability,
8. maintainable Python-first design,
9. evidence-based native acceleration.

---

# 1. Required project context

Before making meaningful code changes, read:

- `memory_bank/lib_design.md`
- `memory_bank/progress.md`

Use `memory_bank/lib_design.md` as the source of truth for:

- long-term architecture,
- public API principles,
- naming conventions,
- modularity,
- compatibility,
- testing philosophy,
- contribution style.

Use `memory_bank/progress.md` to understand:

- what is already implemented,
- current modules,
- known limitations,
- ongoing work,
- migration state,
- planned next steps.

Do not introduce architectural decisions that conflict with `memory_bank/lib_design.md` unless the user explicitly requests such a change.

---

# 2. Current technology stack

The current project is Python-first and uses:

- Python >= 3.12
- PyTorch
- PyTorch Geometric
- Torch Spatiotemporal
- Polars
- PyArrow
- NumPy
- Pydantic
- SnapML
- Hugging Face Hub
- Rust
- PyO3
- Maturin
- uv
- pytest
- Ruff

Do not introduce an alternative framework or dependency when the existing stack already provides the required functionality unless there is a clear technical reason.

In particular, prefer mature existing implementations from:

- Polars,
- PyArrow,
- NumPy,
- PyTorch,
- PyTorch Geometric,
- Torch Spatiotemporal,

before writing custom infrastructure.

---

# 3. Repository layout

The intended repository layout is conceptually:

```text
AMLGraphX/
├── pyproject.toml
├── AGENTS.md
├── memory_bank/
│   ├── lib_design.md
│   └── progress.md
│
├── rust/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── ...
│
├── src/
│   └── amlgraphx/
│       ├── data/
│       ├── datasets/
│       ├── graph/
│       ├── transforms/
│       ├── split/
│       ├── sampling/       # when required
│       ├── nn/
│       ├── baselines/      # when required
│       ├── metrics/
│       ├── evaluation/     # when required
│       ├── features/       # when required
│       ├── _native/
│       ├── utils/          # only for genuine cross-domain utilities
│       └── py.typed
│
└── tests/
```

This is a design direction.

Do not create empty packages or abstraction layers merely to match this tree.

Create modules when real functionality requires them.

---

# 4. Python and Rust source locations

Python source code belongs under:

```text
src/amlgraphx/
```

Rust source code belongs under the repository-level:

```text
rust/
```

The native Python extension is exposed internally as:

```text
amlgraphx._native._core
```

and the compiled extension is placed under:

```text
src/amlgraphx/_native/
```

for example:

```text
src/amlgraphx/_native/_core.cpython-312-x86_64-linux-gnu.so
```

The compiled `.so` file is a generated build artifact.

Never manually edit compiled native artifacts.

The source of truth for native behavior is the Rust source under `rust/`.

---

# 5. CodeGraph-first workflow

Before modifying, adding, deleting, moving, or refactoring code:

1. Use CodeGraph MCP to understand the relevant code structure.
2. Use `codegraph_context` or `codegraph_explore` to inspect relevant modules and symbols.
3. Use `codegraph_callers` and `codegraph_callees` when call relationships matter.
4. Use `codegraph_impact` before changing existing public functions, classes, schemas, dataset interfaces, or import paths.
5. Inspect the relevant source and tests after understanding the dependency graph.
6. Implement the smallest coherent change.
7. Run targeted validation.

Do not begin editing first and investigate dependencies afterward.

Exceptions are allowed when:

- the change is purely documentation or text,
- CodeGraph is unavailable,
- the relevant code is not indexed.

If CodeGraph cannot provide useful information, state that briefly and continue by inspecting the source directly.

---

# 6. CodeGraph and Rust

CodeGraph may not fully capture Rust modules or Python–Rust relationships.

When changing native-backed functionality, supplement CodeGraph analysis by inspecting:

- the public Python wrapper,
- internal Python orchestration,
- `amlgraphx._native`,
- PyO3 registrations,
- the relevant Rust modules,
- Python integration tests,
- Rust tests.

Do not assume native code is isolated merely because CodeGraph shows few Python callers.

Always inspect the Python–Rust boundary.

---

# 7. Understand before editing

Do not modify code based only on:

- a filename,
- a task description,
- a partial snippet,
- an isolated function.

Before changing existing behavior, identify:

- the public API,
- the implementation symbol,
- related callers,
- related callees,
- downstream impact,
- related tests,
- documented behavior,
- graph semantics,
- temporal semantics,
- Python–Rust boundaries where relevant.

Prefer changing internal implementation over breaking public interfaces.

---

# 8. Core AMLGraphX data flow

AMLGraphX should conceptually follow this pipeline:

```text
External Dataset
      ↓
Canonical Transaction Data
      ↓
Transforms / Feature Processing
      ↓
Graph Representation
      ├── Account Graph
      ├── Transaction Graph
      ├── Snapshot Sequence
      └── Temporal Event Stream
      ↓
Split / Evaluation Protocol
      ↓
Loader / Sampler
      ↓
Model / Baseline
      ↓
AML Metrics / Evaluation
```

These stages are different responsibilities.

Do not collapse:

- dataset parsing,
- graph construction,
- temporal splitting,
- model logic,
- training logic,
- evaluation,

into one monolithic pipeline.

---

# 9. Canonical transaction representation

Dataset loading should normally produce a canonical AMLGraphX transaction representation before graph construction.

The preferred conceptual flow is:

```text
raw external data
        ↓
dataset-specific parsing
        ↓
canonical AMLGraphX transaction data
        ↓
graph / temporal conversion
        ↓
PyG / Torch / other backend
```

Avoid:

```text
raw dataset
    ↓
immediately hard-coded into one model-specific graph format
```

Dataset loading and graph modeling are separate concerns.

---

# 10. Responsibility of `datasets`

`amlgraphx.datasets` owns external dataset-specific behavior.

Typical responsibilities include:

- downloading,
- archive handling,
- file discovery,
- dataset-specific parsing,
- schema mapping,
- dataset metadata,
- dataset registry,
- canonical transaction conversion.

Examples include:

- IBM AML,
- PaySim,
- SAML-D,
- Elliptic,
- future AML/Fraud datasets.

Dataset modules should not contain generic:

- GNN models,
- training loops,
- graph algorithms,
- temporal splitting,
- graph sampling,
- evaluation logic.

Dataset-specific knowledge belongs here.

Reusable graph or temporal behavior does not.

---

# 11. Responsibility of `data`

`amlgraphx.data` owns reusable data abstractions and orchestration.

It may contain:

- canonical schemas,
- transaction structures,
- graph-facing data structures,
- temporal representations,
- datamodules,
- loaders,
- batching abstractions,
- validation of shared data contracts.

Do not make `data` a dumping ground for dataset-specific parsing or graph algorithms.

---

# 12. Responsibility of `graph`

`amlgraphx.graph` owns graph representation and graph-construction semantics.

Important AMLGraphX graph paradigms include:

```text
account-as-node
transaction-as-node
heterogeneous graph
snapshot graph sequence
continuous temporal event stream
```

Do not hide fundamentally different semantics behind one ambiguous implementation.

Prefer explicit graph builders and temporal conversion modules.

Conceptually:

```text
graph/
├── builders/
│   ├── account.py
│   ├── transaction.py
│   └── heterogeneous.py
│
└── temporal/
    ├── snapshot.py
    ├── event_stream.py
    └── windows.py
```

Only create modules that are currently needed.

---

# 13. Graph builder rules

Graph builders define graph semantics.

They do not define train/validation/test behavior.

For example, account-as-node may define:

```text
account = node
transaction = directed edge
```

Transaction-as-node may define a relation such as:

```text
transaction = node

receiver(T_i) == sender(T_j)

and

0 <= t_j - t_i <= Δt
```

Graph builders should explicitly define important behavior including:

- node identity,
- edge identity,
- graph direction,
- duplicate edges,
- multigraph behavior,
- self-loops,
- temporal constraints,
- relation semantics,
- ordering assumptions.

Graph construction should be deterministic unless randomness is an explicit part of the algorithm.

---

# 14. Temporal graph paradigms must remain distinct

Do not treat every dynamic graph as the same representation.

AMLGraphX should distinguish at least three temporal paradigms.

## 14.1 Time-aware static graph

One graph whose construction uses timestamps.

Example:

```text
transaction-as-node graph
with money-flow edges only within 4 hours
```

The result may still be represented as a conventional graph.

---

## 14.2 Snapshot sequence

A sequence:

```text
G0, G1, G2, ..., GT
```

where each graph represents a discrete interval.

This is appropriate for models such as EvolveGCN-style architectures.

Snapshot frequency may be:

- hourly,
- daily,
- weekly,
- monthly,
- dataset-defined time steps.

---

## 14.3 Continuous event stream

An ordered event representation such as:

```text
(src, dst, time, message)
```

This is appropriate for TGN, JODIE, TGAT-style workflows.

PyTorch Geometric `TemporalData` is an example backend representation for this paradigm.

---

Do not silently convert:

```text
snapshot graph
```

into:

```text
event stream
```

or vice versa.

APIs and names should communicate their temporal semantics clearly.

---

# 15. Snapshot semantics

Snapshot generation belongs under graph temporal functionality rather than dataset-specific code when the operation is generic.

Examples:

```text
transactions
    ↓
daily snapshots
    ↓
G0, G1, ..., GT
```

A snapshot builder must document whether snapshots are:

```text
disjoint
```

such as:

```text
G_t = events occurring during interval t
```

or cumulative:

```text
G_t = all events occurring up to time t
```

These are scientifically different representations.

Never make this distinction implicit.

---

# 16. Event-stream semantics

Continuous temporal representations should preserve event ordering.

Typical event fields may include:

```text
src
dst
timestamp
message/features
label
transaction id
```

Temporal event APIs must preserve documented ordering semantics.

Do not reorder events in a way that creates future-information leakage.

When converting AML transactions into an event stream, the natural account-level representation is often:

```text
sender account = src
receiver account = dst
transaction timestamp = time
transaction features = message
```

but alternative representations are allowed when explicitly defined.

---

# 17. Responsibility of `split`

Train/validation/test splitting is a first-class research concept.

Reusable splitting should live under:

```text
amlgraphx.split
```

rather than inside:

- dataset loaders,
- graph builders,
- models,
- training scripts.

Potential split protocols include:

- random split,
- chronological split,
- temporal ratio split,
- snapshot split,
- causal split,
- transductive temporal split.

---

# 18. Separate target data from historical context

For temporal research, explicitly distinguish:

```text
prediction targets
historical context
training labels
graph construction context
```

These are not necessarily identical sets.

For example:

```text
[ historical context ][ prediction target ]
```

Historical observations may be required to construct the graph or model state without themselves being prediction targets.

Do not silently discard historical context at split boundaries.

---

# 19. Warm-up and lookback context

Temporal models often require history before the first prediction.

Support explicit concepts such as:

```text
warm-up period
lookback context
history-only observations
prediction period
```

For example, if a snapshot model needs five historical graphs:

```text
G0 G1 G2 G3 G4 → context
G5             → first target
```

Do not pretend that `G0` has five steps of history.

Similarly, if graph construction requires a four-hour lookback, observations before the prediction interval may be used as graph context without becoming prediction targets.

---

# 20. Temporal leakage

Scientific correctness has priority over implementation convenience.

Never use future information for historical predictions unless the protocol explicitly allows transductive access.

Pay particular attention to:

- future transactions,
- future graph edges,
- future account aggregates,
- future node statistics,
- future labels,
- normalization fitted using future observations,
- future neighbors,
- cross-split graph construction,
- temporal sampling,
- timestamp ordering.

Tests for temporal functionality should verify past-only behavior when strict causal evaluation is intended.

If a method intentionally uses transductive information, document that clearly.

Do not describe a transductive protocol as strict causal evaluation.

---

# 21. Responsibility of `transforms`

Generic reusable preprocessing should gradually live under:

```text
amlgraphx.transforms
```

Examples include:

- log amount transformation,
- amount normalization,
- currency encoding,
- payment-format encoding,
- time-of-day features,
- day-of-week features,
- inter-arrival features,
- graph transforms,
- temporal transforms.

Transforms should preferably be:

- composable,
- deterministic,
- independently testable.

Avoid allowing a single `preprocessing.py` to grow into a large collection of unrelated operations.

Dataset-specific cleaning may remain in the corresponding dataset module.

---

# 22. Feature engineering

If reusable AML feature extraction grows substantially, use a dedicated:

```text
amlgraphx.features
```

Examples may include:

```text
transaction features
account features
temporal features
network features
```

Feature extraction should remain separate from the estimator using the features.

For example:

```text
network features
      ↓
XGBoost
```

should not require the network feature implementation to live inside the XGBoost model.

---

# 23. Loading and sampling

Loading and sampling are different from graph construction.

Loaders expose already-defined data structures to models.

Sampling may include:

- temporal neighborhood sampling,
- node neighborhood sampling,
- negative sampling,
- class-imbalance sampling,
- event batching,
- graph batching.

Create:

```text
amlgraphx.sampling
```

when enough functionality exists to justify a dedicated package.

Do not place generic sampling algorithms inside dataset modules.

---

# 24. Neural network organization

Prefer:

```text
amlgraphx.nn
```

for deep-learning components.

The conceptual organization is:

```text
nn/
├── layers/
├── models/
└── functional.py
```

when required.

Complete models may eventually be grouped by paradigm:

```text
nn/models/
├── static/
├── snapshot/
└── temporal/
```

Examples:

```text
static/
    GCN
    GraphSAGE
    GIN

snapshot/
    EvolveGCN

temporal/
    TGN
    TGAT
```

Do not create these directories before real implementations exist.

Reusable neural building blocks belong in `layers`.

Complete trainable architectures belong in `models`.

---

# 25. Classical ML and research baselines

Classical methods do not belong under `nn`.

When enough implementations exist, use:

```text
amlgraphx.baselines
```

Potential baselines include:

- logistic regression,
- random forest,
- XGBoost,
- SnapML,
- Isolation Forest,
- DeepWalk + classifier,
- node2vec + classifier,
- manual network-feature baselines.

Keep feature extraction separate from the estimator.

---

# 26. AML metrics

Reusable metrics should live under:

```text
amlgraphx.metrics
```

AMLGraphX should support metrics appropriate for severe class imbalance and investigation-budget evaluation.

Examples include:

- Precision,
- Recall,
- F1,
- ROC-AUC,
- PR-AUC,
- Precision@K,
- Recall@K,
- Lift@K,
- Precision@0.1%,
- Precision@0.5%,
- Precision@1%.

Metrics must clearly define what the denominator means.

For percentage-based AML metrics, document whether the percentage refers to:

- ranked transactions,
- accounts,
- graph nodes,
- graph edges,
- another investigation population.

Avoid ambiguous evaluation semantics.

---

# 27. Evaluation layer

If evaluation logic becomes sufficiently complex, use:

```text
amlgraphx.evaluation
```

Potential responsibilities include:

- reusable evaluators,
- protocol validation,
- leakage checking,
- investigation-budget evaluation,
- reproducible result reporting.

Evaluation code should not redefine:

- dataset semantics,
- graph semantics,
- temporal splitting.

---

# 28. Paper reproduction

A research paper should not define the repository architecture.

When reproducing a paper:

1. identify reusable data assumptions,
2. identify reusable graph semantics,
3. identify temporal protocol,
4. identify reusable model components,
5. identify evaluation semantics.

Prefer composition such as:

```text
Dataset
   ↓
TransactionGraphBuilder
   ↓
TemporalSplit
   ↓
Model
   ↓
AML Metrics
```

instead of creating isolated structures such as:

```text
paper_name/
├── data.py
├── graph.py
├── train.py
└── evaluate.py
```

Paper-specific defaults may be exposed as configuration or presets when useful.

---

# 29. Public API discipline

Treat documented public import paths as compatibility contracts.

Before changing a public API:

1. inspect usage with CodeGraph,
2. inspect downstream impact,
3. prefer backward-compatible changes,
4. update tests,
5. update documentation.

Do not casually rename, move, or remove:

- public classes,
- public functions,
- public parameters,
- schemas,
- datasets,
- import paths.

Internal implementation may evolve more freely.

---

# 30. Compatibility during module migration

Avoid keeping duplicate implementations.

If an old public path must remain temporarily, use a lightweight compatibility wrapper.

For example:

```text
amlgraphx.graphs
```

may forward to:

```text
amlgraphx.graph
```

during migration.

There should still be only one canonical implementation.

Do not maintain two independent versions of the same graph logic.

---

# 31. Avoid ambiguous duplicate modules

Avoid overlapping names such as:

```text
graphs.py
graph/graphs.py

loader.py
dataloader.py
data_loader.py
```

unless their responsibilities are genuinely different and clearly documented.

Use one canonical concept and one canonical location.

---

# 32. Naming and readability

Follow Python naming conventions:

- modules: `snake_case`
- functions: `snake_case`
- variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_CASE`
- private helpers: leading underscore when appropriate.

Use one canonical term for one concept.

Avoid names such as:

```text
utils2.py
new_utils.py
misc.py
helper_new.py
final_model.py
model_final2.py
```

Prefer domain-oriented names.

Readable code is preferred over clever code.

Do not optimize for minimum line count.

---

# 33. Generic utilities

Do not turn `utils.py` or `utils/` into a dumping ground.

A helper belongs in `utils` only when it is genuinely cross-domain.

If functionality naturally belongs to:

```text
graph
datasets
data
transforms
split
sampling
metrics
evaluation
nn
```

place it there instead.

Prefer semantic locality.

---

# 34. Testing philosophy

New features, bug fixes, and behavior changes should include tests when appropriate.

Tests should focus on observable and scientific behavior rather than private implementation details.

Good tests include:

- known transactions produce expected graph nodes and edges,
- graph direction is correct,
- transaction-as-node relations are correct,
- account-as-node relations are correct,
- temporal lookback is respected,
- future information is excluded,
- snapshot boundaries are correct,
- warm-up context behaves correctly,
- split chronology is preserved,
- dataset parsing produces the canonical schema,
- duplicate transactions follow documented semantics,
- Rust and Python implementations agree.

Avoid tests whose only assertion is:

```text
result is not None
```

unless that behavior is genuinely meaningful.

---

# 35. Test fixtures

For normal CI, prefer:

- small,
- deterministic,
- synthetic,
- easily inspectable fixtures.

Do not require large external AML datasets for ordinary unit tests.

Large real datasets may be used for:

- integration tests,
- manual validation,
- benchmarks,

when appropriate.

---

# 36. Python validation

After meaningful Python changes, run targeted tests first.

Typical workflow:

```text
uv run pytest <relevant tests>
```

Then, when practical:

```text
uv run pytest
```

Run Ruff:

```text
uv run ruff check .
uv run ruff format --check .
```

If formatting is needed:

```text
uv run ruff format .
```

Do not invent formatting rules that conflict with `pyproject.toml`.

Do not introduce unrelated repository-wide formatting changes.

---

# 37. Definition of done for Python work

A meaningful Python code task is normally complete only when:

- project context was read,
- CodeGraph analysis was performed,
- architectural placement was considered,
- implementation is complete,
- relevant tests were added or updated,
- relevant tests pass,
- Ruff passes,
- documentation is updated when necessary,
- `memory_bank/progress.md` is updated for meaningful completed work.

Only then proceed to Git operations.

---

# 38. Progress memory

After completing a meaningful feature or module, update:

```text
memory_bank/progress.md
```

Record concise information about:

- what was implemented,
- important design decisions,
- public APIs added or changed,
- tests added,
- native acceleration added,
- known limitations,
- follow-up work when relevant.

Do not rewrite unrelated historical entries.

The purpose of this file is to allow future agents to quickly reconstruct repository state.

---

# 39. Performance policy

Do not optimize code merely because:

- it contains a Python loop,
- Rust is available,
- parallelism is possible,
- the implementation might theoretically become slow.

Use:

```text
correct implementation
        ↓
tests
        ↓
benchmark
        ↓
profile
        ↓
identify bottleneck
        ↓
optimize
        ↓
verify equivalence
```

Prefer existing optimized libraries first.

---

# 40. Native acceleration philosophy

AMLGraphX may use Rust for performance-critical internal operations.

The preferred architecture is:

```text
Stable Python Public API
          ↓
Python Validation / Orchestration
          ↓
Internal Rust Kernel
```

Rust exists to accelerate AMLGraphX.

AMLGraphX does not exist to expose Rust.

Users should normally interact with:

```python
amlgraphx.graph
amlgraphx.data
amlgraphx.split
amlgraphx.nn
```

rather than:

```python
amlgraphx._native
```

`amlgraphx._native` should normally be treated as private implementation infrastructure.

---

# 41. Native acceleration candidates

Rust should be considered especially for operations whose cost scales strongly with:

- number of transactions,
- number of graph edges,
- graph degree,
- temporal neighborhood size,
- repeated graph indexing.

Likely native acceleration candidates include:

- transaction-as-node edge construction,
- account-as-node adjacency construction,
- temporal edge matching,
- lookback-window matching,
- temporal indexing,
- temporal neighborhood sampling,
- high-degree account expansion,
- sparse graph construction,
- graph traversal,
- connected components,
- motif counting,
- AML laundering-pattern counting,
- repeated graph aggregation,
- large-scale graph transformations,
- CPU-bound transaction processing.

These are candidates, not requirements.

Do not move functionality to Rust without demonstrated benefit.

---

# 42. Example native-backed architecture

The intended pattern is conceptually:

```text
amlgraphx.graph.builders.transaction
                ↓
      Python validation
                ↓
      internal backend call
                ↓
amlgraphx._native._core
                ↓
          Rust kernel
```

Users should call something conceptually like:

```python
TransactionGraphBuilder(...)
```

not:

```python
_native._core.build_edges(...)
```

Low-level bindings should normally remain internal.

---

# 43. Backend independence

Do not expose unnecessary backend details in the public API.

A public AMLGraphX operation should ideally remain stable even if its implementation changes from:

```text
Python
→ Rust
→ C++
→ CUDA
→ external graph engine
```

Do not name public APIs with suffixes such as:

```text
_rust
_native
_fast
_cpp
```

unless backend selection itself is an explicit user-facing feature.

Prefer semantic names.

---

# 44. Python responsibilities

Python normally owns:

- public APIs,
- configuration,
- Pydantic schemas,
- validation,
- dataset interfaces,
- graph abstractions,
- temporal abstractions,
- split protocols,
- orchestration,
- error presentation,
- PyTorch interoperability,
- PyG interoperability,
- Torch Spatiotemporal interoperability,
- Polars workflows,
- PyArrow workflows,
- documentation.

Python should remain the primary user-facing layer.

---

# 45. Rust responsibilities

Rust should normally own narrowly scoped performance kernels such as:

- graph construction,
- temporal matching,
- indexing,
- sampling,
- traversal,
- aggregation,
- motif detection,
- structural graph algorithms.

Avoid moving:

- public configuration,
- high-level experiment logic,
- dataset APIs,
- training loops,
- model orchestration,

into Rust without a strong reason.

Do not allow Rust ownership or lifetime constraints to produce awkward Python APIs.

---

# 46. Rust source structure

Rust source belongs under:

```text
rust/
```

As the native backend grows, prefer a domain-oriented structure such as:

```text
rust/
├── Cargo.toml
└── src/
    ├── lib.rs
    │
    ├── graph/
    │   ├── mod.rs
    │   ├── account.rs
    │   └── transaction.rs
    │
    ├── temporal/
    │   ├── mod.rs
    │   ├── edge_match.rs
    │   └── sampling.rs
    │
    ├── indexing/
    │   ├── mod.rs
    │   └── adjacency.rs
    │
    └── algorithms/
        ├── mod.rs
        └── motifs.rs
```

This is a conceptual direction.

Do not create empty Rust modules before real functionality requires them.

---

# 47. Rust extension entry point

Keep:

```text
rust/src/lib.rs
```

small.

Its primary responsibilities are:

- declaring Rust modules,
- constructing the PyO3 extension,
- registering exposed functions and classes,
- delegating registration to domain modules.

Do not place substantial graph algorithms directly in `lib.rs`.

As native functionality grows, substantial modules should expose small registration functions.

Conceptually:

```rust
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ...
}
```

and `lib.rs` should compose those registrations.

---

# 48. Rust module naming

Prefer domain-oriented modules such as:

```text
graph
temporal
sampling
indexing
aggregation
motifs
```

Avoid native dumping-ground names such as:

```text
utils
misc
helpers
common2
new_core
fast_stuff
```

Small internal helpers may exist, but major functionality should have a clear domain owner.

---

# 49. Python–Rust boundary

Crossing the Python–Rust boundary has overhead.

Prefer:

```text
large collection
      ↓
one native call
      ↓
substantial computation
      ↓
result
```

Avoid:

```text
Python for-loop
      ↓
Rust call for one transaction
      ↓
Python
      ↓
Rust call for next transaction
```

Native APIs should operate on coarse-grained collections when practical.

---

# 50. Polars / PyArrow and Rust

AMLGraphX already depends on Polars and PyArrow.

Do not copy large tables into inefficient Python object structures merely to pass them to Rust.

When designing high-volume native operations, consider interoperability with:

- Arrow-compatible memory,
- NumPy arrays,
- contiguous tensor-like structures,
- efficient integer index arrays.

However, do not introduce complicated zero-copy designs prematurely.

First determine whether Python–Rust transfer is a real bottleneck.

---

# 51. Rust readability

Rust must remain understandable to contributors who primarily work in Python.

Prefer:

- explicit code,
- descriptive names,
- small functions,
- straightforward ownership,
- documented invariants.

Avoid unnecessary:

- macro systems,
- complex generic hierarchies,
- obscure iterator chains,
- advanced lifetime abstractions,
- speculative zero-copy machinery,
- complicated concurrency infrastructure.

Do not optimize for minimum Rust line count.

---

# 52. Unsafe Rust

Avoid `unsafe` unless it is clearly justified.

Before introducing `unsafe`:

1. verify safe Rust is insufficient,
2. document why `unsafe` is required,
3. isolate the unsafe operation,
4. define soundness invariants,
5. add relevant tests.

Do not use `unsafe` for speculative performance improvements.

---

# 53. Semantic equivalence

Performance optimization must not change research semantics.

When replacing Python logic with Rust, verify relevant behavior including:

- node identity,
- edge identity,
- graph direction,
- duplicate transactions,
- duplicate edges,
- multigraph behavior,
- self-loops,
- transaction ordering,
- timestamp ordering,
- temporal boundaries,
- lookback behavior,
- future-information exclusion,
- sampling behavior,
- output type,
- output shape,
- deterministic behavior where required.

For floating-point outputs, use appropriate tolerances.

---

# 54. Python fallback implementations

When useful for scientific clarity or validation, a simple Python implementation may remain as:

- a reference implementation,
- a fallback,
- a correctness oracle for tests.

Do not duplicate complicated production logic indefinitely without reason.

If both implementations exist, clearly establish which is:

```text
reference
```

and which is:

```text
optimized backend
```

Tests should compare them on deterministic small fixtures.

---

# 55. Rust testing

Meaningful native functionality should include Rust tests when appropriate.

Rust-side tests are useful for:

- graph kernels,
- indexing,
- temporal matching,
- data structures,
- invariants,
- boundary conditions,
- parsing,
- native-only algorithms.

Native functionality exposed to Python also requires Python integration tests.

Rust tests alone are not sufficient for user-facing functionality.

---

# 56. Python integration tests for native functionality

Python tests should verify that:

- `amlgraphx._native._core` imports correctly,
- Python inputs are accepted,
- output types are correct,
- output shapes are correct,
- documented semantics are preserved,
- errors become meaningful Python exceptions,
- optimized and reference implementations agree where applicable.

Tests should use small deterministic fixtures whenever possible.

---

# 57. Rust validation workflow

Whenever any of the following changes:

- Rust source files,
- Rust module declarations,
- PyO3 functions,
- PyO3 classes,
- native registration,
- Cargo dependencies,
- Python–Rust bindings,

run native validation.

If the Cargo manifest is under `rust/Cargo.toml`, either run commands from `rust/` or use the manifest explicitly.

Typical commands from the repository root are:

```text
cargo check --manifest-path rust/Cargo.toml
cargo test --manifest-path rust/Cargo.toml
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml
```

If formatting is required:

```text
cargo fmt --manifest-path rust/Cargo.toml
```

Fix relevant warnings introduced by the change.

---

# 58. Rebuild the native Python extension

After meaningful Rust changes, rebuild the PyO3 extension.

Use the repository's Maturin configuration:

```text
uv run maturin develop
```

The configured Python module is:

```text
amlgraphx._native._core
```

and the Python source root is:

```text
src
```

Do not assume Python is using newly edited Rust code until the extension has been rebuilt successfully.

---

# 59. Release native builds

For performance benchmarks or production-like measurements, use an optimized build:

```text
uv run maturin develop --release
```

Do not benchmark Rust performance using only a debug build.

The correct workflow for performance work is:

```text
correctness implementation
        ↓
debug/native build
        ↓
tests
        ↓
release build
        ↓
benchmark
```

---

# 60. Native validation after Maturin

After rebuilding the native extension, run relevant Python tests.

Start with targeted tests:

```text
uv run pytest <relevant native integration tests>
```

Then, when practical:

```text
uv run pytest
```

If Python wrappers changed, also run:

```text
uv run ruff check .
uv run ruff format --check .
```

A successful:

```text
cargo check
```

does not prove that the Python-facing extension works.

---

# 61. Native definition of done

A meaningful Rust-backed task is normally complete only when:

- the intended Python-facing behavior is clear,
- public API compatibility was considered,
- Rust implementation is complete,
- Rust tests were added when appropriate,
- Python integration tests were added when appropriate,
- `cargo check` passes,
- `cargo test` passes,
- `cargo fmt --check` passes,
- `cargo clippy` passes,
- `uv run maturin develop` succeeds,
- relevant Python tests pass,
- Ruff passes for changed Python code,
- benchmark results exist when performance is the purpose,
- documentation is updated when needed,
- `memory_bank/progress.md` is updated.

---

# 62. Benchmark-driven optimization

Do not claim performance improvement merely because functionality was rewritten in Rust.

When performance is the goal, benchmark representative workloads.

Measure where practical:

- runtime,
- memory usage,
- scaling with number of transactions,
- scaling with number of nodes,
- scaling with number of edges,
- scaling with neighborhood size.

Compare implementations with equivalent semantics.

Do not benchmark an optimized Rust implementation against an intentionally inefficient Python implementation when a mature vectorized Python backend already exists.

---

# 63. Parallelism

Do not automatically parallelize Rust code.

Introduce parallelism when:

- workloads are sufficiently large,
- profiling shows meaningful benefit,
- ordering semantics remain correct,
- memory use remains acceptable,
- deterministic behavior remains understood.

Temporal AML code requires special care.

Parallel processing must not violate:

- timestamp ordering,
- past-only constraints,
- stable edge-generation semantics.

---

# 64. Memory and large graph workloads

AMLGraphX may process very large transaction networks.

For large native operations, consider:

- sparse representations,
- unnecessary clones,
- duplicate graph representations,
- integer width,
- temporary allocations,
- intermediate materialization,
- streaming opportunities,
- predictable memory complexity.

Avoid holding multiple full copies of large graph structures without a clear reason.

Prefer predictable memory behavior over small micro-optimizations.

---

# 65. Error handling across Python and Rust

Do not use Rust panics for normal invalid user input.

Expected failures should use Rust `Result` values and become meaningful Python exceptions.

Error messages should explain:

- what input was invalid,
- what condition was violated,
- how the user may correct it.

Avoid exposing irrelevant Rust implementation details.

Internal assertions are acceptable for true internal invariants.

---

# 66. Rust dependencies

Add Rust crates conservatively.

Before adding a dependency:

1. check whether the standard library is sufficient,
2. check existing dependencies,
3. prefer actively maintained crates,
4. avoid large dependency trees for small functionality,
5. avoid duplicate crates with overlapping roles.

Do not expose crate-specific types through the public Python API.

Do not add an entire graph framework solely to avoid implementing a small graph kernel unless the architectural benefit clearly justifies it.

---

# 67. Architecture evolution

AMLGraphX should evolve incrementally.

Do not create every possible future module before it is needed.

At the same time, do not place functionality into an architecturally incorrect location merely because that file already exists.

When a module grows beyond its responsibility, migrate deliberately.

Typical migrations may include:

```text
data/preprocessing.py
        ↓
transforms/

graph/graphs.py
        ↓
graph/builders/

model/dl/
        ↓
nn/

model/ml/
        ↓
baselines/

dataset-specific splitting
        ↓
split/
```

Preserve public compatibility where practical.

---

# 68. Keep changes focused

Avoid mixing unrelated work in one task.

Do not combine:

- a new dataset,
- a new graph builder,
- a model implementation,
- a major public API migration,
- unrelated cleanup,
- repository-wide formatting,

unless explicitly requested.

Prefer small, reviewable, reversible changes.

---

# 69. Do not silently expand scope

If implementing one feature reveals unrelated cleanup opportunities, do not refactor the entire surrounding repository automatically.

Make only changes necessary for:

- correctness,
- integration,
- maintainability of the requested feature.

Large unrelated cleanup should be proposed or handled separately.

---

# 70. Git workflow

After implementation and all relevant validation passes:

1. inspect `git status`,
2. verify only intended files changed,
3. create a focused commit,
4. push to the current working branch.

Do not commit accidental generated or temporary files such as:

```text
__pycache__/
*.pyc
virtual environments
temporary benchmark data
credentials
secrets
local caches
```

Treat compiled native binaries as generated artifacts and follow the repository's explicit tracking policy for them.

Never manually modify compiled native binaries.

---

# 71. Commit messages

Use clear focused commit messages.

Examples:

```text
feat: add transaction graph builder
feat: add snapshot temporal representation
feat: add temporal split protocol
feat: accelerate transaction edge construction
fix: prevent future leakage in temporal sampler
test: add native graph construction equivalence tests
docs: document event stream semantics
refactor: move graph builders into graph package
perf: add native temporal edge matcher
```

Do not commit or push while relevant tests or validation are failing.

---

# 72. Final working rule

When uncertain, prefer the design that maximizes:

- scientific correctness,
- explicit semantics,
- API stability,
- responsibility separation,
- readability,
- composability,
- testability,
- reproducibility,
- backend independence.

Use CodeGraph before changing dependencies or public behavior.

Use:

```text
memory_bank/lib_design.md
```

for architectural direction.

Use:

```text
memory_bank/progress.md
```

for development continuity.

For normal Python work:

```text
Understand
→ CodeGraph analysis
→ Implement
→ Targeted tests
→ Broader tests
→ Ruff
→ Update progress
→ Commit
→ Push
```

For Rust-backed work:

```text
Understand
→ CodeGraph + binding analysis
→ Implement Python/Rust
→ cargo check
→ cargo test
→ cargo fmt --check
→ cargo clippy
→ maturin develop
→ Python integration tests
→ Ruff
→ Update progress
→ Commit
→ Push
```

For performance work:

```text
Correct semantics
→ Tests
→ Benchmark
→ Profile
→ Optimize
→ Release build
→ Benchmark again
→ Verify semantic equivalence
```

AMLGraphX should remain:

> **Python-first, graph-aware, temporal-aware, research-correct, backend-flexible, and selectively Rust-accelerated.**
````

有一个你需要注意的**配置细节**：如果你确定 Rust 的 `Cargo.toml` 也放在：

```text
rust/Cargo.toml
```

而不是项目根目录，那么你的 `pyproject.toml` 最好把 Maturin 明确改成：

```toml
[tool.maturin]
bindings = "pyo3"
module-name = "amlgraphx._native._core"
python-source = "src"
manifest-path = "rust/Cargo.toml"
```

否则 Maturin 默认寻找 Cargo manifest 的位置可能和你新的 `rust/` 布局不一致。

也就是说你最终的物理结构可以很干净：

```text
AMLGraphX/
├── pyproject.toml
│
├── rust/                         # Rust source
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── graph/
│       ├── temporal/
│       ├── sampling/
│       └── ...
│
└── src/
    └── amlgraphx/                # Python source
        ├── graph/
        ├── data/
        ├── datasets/
        ├── split/
        ├── transforms/
        ├── nn/
        ├── metrics/
        └── _native/
            ├── __init__.py
            └── _core.cpython-312-....so
