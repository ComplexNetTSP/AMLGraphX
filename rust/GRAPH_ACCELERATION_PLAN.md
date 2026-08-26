# Rust Acceleration Plan for Graph Construction

Status: proposal only. No implementation is authorized by this document.

Primary Python implementation:
`src/amlgraphx/graph/graphs.py`

Primary target:
`TransactionGraph.from_transactions()` temporal-edge construction.

## 1. Decision Summary

Do not rewrite `graphs.py` in Rust.

Keep the stable Python API, validation, schema handling, Polars orchestration,
and result objects in Python. Move only a measured, CPU-bound native kernel
after profiling proves that the kernel is a material runtime or memory
bottleneck.

The first Rust candidate is the loop that:

1. materializes sorted Polars rows as Python dictionaries;
2. builds per-account outgoing-transaction indexes;
3. searches temporal successor ranges;
4. creates one Python dictionary for every graph edge.

The intended architecture is:

```text
stable Python API
    -> Python validation and Polars preparation
    -> one bulk Rust temporal-edge call
    -> Polars edge-table construction
    -> existing TransactionGraph result
```

## 2. Why This Is the First Candidate

Most expressions in `graphs.py` already execute in Polars' Rust engine:

- `filter`
- `select`
- `with_columns`
- `sort`
- `join`
- `unique`
- string parsing and datetime conversion

Rewriting those expressions in a custom PyO3 extension would duplicate a
mature native implementation and add conversion overhead.

The current temporal-edge loop is different. It returns from native Polars
execution to Python object processing:

```text
Polars DataFrame
    -> iter_rows(named=True)
    -> one Python dict per transaction
    -> Python dict/list indexes
    -> Python binary searches and loops
    -> one Python dict per edge
    -> Polars DataFrame
```

On a graph with millions of transactions and hundreds of millions of edges,
Python object allocation can dominate both runtime and memory.

## 3. Responsibilities That Must Remain in Python

Keep these responsibilities in Python unless later profiling provides strong
contrary evidence:

- public `AccountGraph` and `TransactionGraph` classes;
- public `build_account_graph()` and `build_transaction_graph()` functions;
- argument validation and user-facing errors;
- source, target, timestamp, and transaction-ID alias resolution;
- Polars `DataFrame` and `LazyFrame` handling;
- null filtering and timestamp parsing;
- account-metadata joins;
- final Polars table construction;
- orchestration with the temporal data module;
- interoperability with PyTorch and PyG.

The Python API must not expose Rust-specific types or require users to import
the native extension directly.

## 4. Proposed First Native Kernel

The native kernel should receive prepared, columnar data in one call and
return compact edge arrays in one call.

A conceptual internal contract is:

```text
temporal_edge_indices(
    source_account_codes,
    target_account_codes,
    timestamps_ns,
    delta_ns,
)
    -> source_positions
    -> target_positions
    -> time_deltas_ns
```

This is an internal contract, not a public Python API.

### 4.1 Input Contract

Before calling Rust, Python and Polars must ensure that:

- `source_account_codes` and `target_account_codes` use one shared account-ID
  mapping;
- account codes are contiguous integer arrays;
- timestamps are non-null signed 64-bit nanoseconds;
- rows are stably sorted by `(timestamp_ns, original_row_index)`;
- source and target endpoints are non-null and non-empty;
- `delta_ns` is non-negative;
- all input arrays have the same length.

Do not encode source and target accounts independently. Equal accounts in the
two columns must receive the same integer code.

### 4.2 Output Contract

Returned positions refer to rows in the sorted working table, not directly to
rows in the public `graph.nodes` table.

Python and Polars can reconstruct the existing edge schema as follows:

- `source_transaction_id`: gather the transaction ID at `source_position`;
- `target_transaction_id`: gather the transaction ID at `target_position`;
- `via_account`: gather the target account at `source_position`;
- `time_delta`: convert `time_delta_ns` to `pl.Duration("ns")`.

Rust should not return one Python tuple or dictionary per edge. It should
return bulk integer arrays suitable for direct column construction.

### 4.3 Algorithm

The initial Rust implementation should preserve the current algorithm:

```text
for each sorted transaction position:
    append the position to outgoing[source_account]

for each sorted transaction position as the earlier transaction:
    candidates = outgoing[target_account]
    start = first candidate with timestamp > current_timestamp
    end = first candidate with timestamp > current_timestamp + delta
    emit candidates[start:end]
```

A Rust implementation can store only candidate row positions and use the
shared timestamp array during binary search. It does not need separate copies
of every candidate timestamp.

Expected complexity remains output-sensitive:

```text
time:   O(N + sum(log account_degree) + E)
memory: O(N + E)
```

`N` is the number of transactions and `E` is the number of generated temporal
edges.

## 5. Required Scientific Semantics

The Rust implementation must be observationally equivalent to the Python
reference implementation.

For an earlier transaction `A -> B` and a later transaction `B -> C`, emit a
directed edge only when:

```text
earlier.target == later.source
earlier.timestamp < later.timestamp
later.timestamp <= earlier.timestamp + delta
```

The temporal interval is therefore:

```text
(current_timestamp, current_timestamp + delta]
```

The following rules are mandatory:

- graph direction is earlier transaction to later transaction;
- equal timestamps are not connected;
- the upper `delta` boundary is inclusive;
- backward-time edges are never emitted;
- `delta == 0` produces no temporal edges;
- nanosecond ordering is preserved;
- duplicate transfers remain distinct transactions;
- existing transaction-ID repair semantics remain unchanged;
- empty results retain the documented edge schema and dtypes;
- edge output order is deterministic;
- source transactions are processed in stable sorted order;
- successors for one source transaction are emitted in stable sorted order;
- self-loops and duplicate edges must follow existing Python behavior;
- checked arithmetic must prevent timestamp-plus-delta overflow.

Performance work must not silently change any of these rules.

## 6. Rust Eligibility Standard

A Python section should move to Rust only when all relevant gates below pass.

### Gate 1: Measured Bottleneck

Profiling should show that the candidate consumes at least one of:

- 20% to 30% of total wall-clock time;
- 30% of peak memory;
- enough resources to prevent a representative dataset from completing.

Do not migrate code based only on intuition or line count.

### Gate 2: Python-Level Work

The candidate should contain substantial Python work over large collections,
such as:

- loops over hundreds of thousands or millions of rows;
- per-row dictionaries, tuples, or lists;
- repeated hash-table access;
- per-edge allocation;
- CPU-bound traversal, indexing, or search.

Calling a Polars expression from Python does not by itself satisfy this gate,
because the heavy work already runs in Rust.

### Gate 3: Coarse Boundary

The operation must support:

```text
one bulk input transfer
    -> substantial native computation
    -> one bulk output transfer
```

Never call Rust once per transaction or once per edge.

### Gate 4: Stable and Testable Semantics

Inputs, outputs, ordering, null behavior, temporal boundaries, and exceptions
must be precisely defined before native implementation begins.

### Gate 5: End-to-End Benefit

Benchmark the release build. A reasonable adoption threshold is at least one
of:

- 1.5x faster end-to-end graph construction;
- 30% lower peak memory;
- successful completion of a representative workload that the Python path
  cannot complete within the resource budget.

A fast isolated kernel is not sufficient if Python-to-Rust conversion removes
most of the end-to-end gain.

### Gate 6: Acceptable Maintenance Cost

The native implementation should remain:

- small and domain-focused;
- safe Rust unless unsafe code has a measured and documented justification;
- hidden behind the stable Python API;
- testable from both Rust and Python;
- independent of unnecessary graph frameworks;
- free of speculative parallelism and zero-copy complexity.

## 7. Amdahl's Law Check

Use Amdahl's law before committing to a migration:

```text
overall_speedup = 1 / ((1 - p) + p / s)
```

Where:

- `p` is the fraction of total runtime spent in the candidate;
- `s` is the candidate's native speedup.

Examples:

- If the candidate is 70% of runtime and Rust is 10x faster, total speedup is
  approximately 2.7x.
- If the candidate is 10% of runtime and Rust is 10x faster, total speedup is
  only approximately 1.1x.

Do not add native maintenance burden for negligible end-to-end improvement.

## 8. Candidate Priority

| Candidate | Current execution | Priority | Decision |
|---|---|---:|---|
| Temporal successor search and edge emission | Python loops and objects | 1 | Benchmark, then migrate if gates pass |
| Transaction-ID repair | Python `Counter`, lists, and loop | 2 | Profile; try Polars first |
| Account graph node construction | Polars native engine | Low | Keep in Python/Polars |
| Timestamp parsing | Polars native engine | Low | Keep in Python/Polars |
| Alias resolution and validation | Small Python control flow | None | Keep in Python |
| Edge DataFrame formatting | Mostly Polars | Low | Keep in Python unless boundary data proves costly |

## 9. Benchmark Plan

### 9.1 Build Mode

Always benchmark an optimized native extension:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  uv run maturin develop --release
```

Do not use a debug Rust build for performance conclusions.

### 9.2 Workloads

Use deterministic synthetic workloads with different graph shapes:

1. `NoMatch`: many transactions, almost no temporal edges.
2. `Chain`: approximately one successor edge per transaction.
3. `FanOut`: one account creates many valid successors.
4. `DenseWindow`: large `delta`, so output edge count dominates.
5. `SameTimestamp`: many equal timestamps, which must produce no same-time
   edges.
6. `Nanosecond`: sub-microsecond ordering and boundary cases.

Use real datasets after synthetic correctness and scaling checks:

1. PaySim;
2. IBM AML HI-Small;
3. a larger IBM AML variant only when the smaller runs are stable.

### 9.3 Metrics

Measure at least:

- total graph-construction wall time;
- preparation and sorting time;
- Python-to-Rust conversion time;
- native kernel time;
- native-output-to-Polars conversion time;
- peak resident memory;
- transactions processed per second;
- edges emitted per second;
- node and edge counts;
- deterministic output hash or exact equality on manageable fixtures.

Report both `N` and `E`. Runtime comparisons without output edge counts are
misleading for an output-sensitive algorithm.

### 9.4 Comparison Rules

- Compare identical graph semantics.
- Use the same input ordering and `delta`.
- Warm up imports before timed runs.
- Run multiple repetitions for smaller workloads.
- Report median runtime and peak memory.
- Do not compare release Rust against an intentionally inefficient Python
  baseline with different behavior.

## 10. Implementation Phases

### Phase 0: Baseline

1. Profile current graph construction.
2. Record phase-level runtime and peak memory.
3. Save benchmark configuration and dataset statistics.
4. Confirm that temporal edge generation is a qualifying hotspot.

Stop if the eligibility gates do not pass.

### Phase 1: Define a Python Reference Kernel

Extract the existing temporal-edge loop behind one private internal function
without changing public behavior. This provides a small semantic reference for
native equivalence tests.

Do not add a public backend selector.

### Phase 2: Implement the Rust Kernel

Implement only the prepared-array-to-edge-position operation. Keep the first
version:

- single-threaded;
- safe Rust;
- deterministic;
- free of custom macros;
- free of graph-framework dependencies;
- explicit about integer conversion and overflow.

### Phase 3: Integrate Privately

Call the native kernel from the existing Python builder. Preserve:

```python
build_transaction_graph(transactions, delta=...)
```

Do not expose the low-level PyO3 function as the recommended user API.

### Phase 4: Verify Equivalence

Run exact Python-versus-Rust comparisons on deterministic fixtures and all
existing graph tests. Verify values, dtypes, shapes, ordering, and exceptions.

### Phase 5: Release Benchmark

Build with `--release`, rerun representative benchmarks, and compare total
runtime and peak memory. Adopt the native path only if the end-to-end acceptance
threshold passes.

## 11. Proposed Rust Layout

When implementation is authorized, use the smallest focused layout:

```text
rust/
├── lib.rs
└── graph/
    ├── mod.rs
    └── temporal_edges.rs
```

Responsibilities:

```text
temporal_edges.rs
    algorithm
    internal data validation
    Rust unit tests

graph/mod.rs
    graph-related PyO3 registration

lib.rs
    top-level module composition only
```

`rust/lib.rs` must remain a thin extension entry point. The graph module should
expose a small `register(...)` function and register its own bindings.

## 12. Required Tests

Rust unit tests should cover:

- empty input;
- one transaction;
- one valid chain;
- multiple successors;
- multiple accounts;
- no matching accounts;
- equal timestamps;
- backward ordering;
- inclusive upper `delta` boundary;
- one-nanosecond-outside boundary;
- `delta == 0`;
- deterministic edge ordering;
- timestamp arithmetic overflow;
- mismatched input lengths;
- invalid account codes if the contract permits them.

Python integration tests should verify:

- the native module imports;
- public graph builders keep their signatures;
- Python and Rust edge tables are equal;
- Polars output columns and dtypes are unchanged;
- transaction IDs and `via_account` values are reconstructed correctly;
- existing dataset and temporal-data-module behavior remains unchanged;
- user-facing exceptions remain Python-domain errors rather than Rust panics.

## 13. Validation Workflow

For any future Rust implementation, follow the repository workflow:

```bash
cargo check
cargo test
cargo fmt --check
cargo clippy

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run maturin develop
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run pytest
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run ruff check .
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv run ruff format --check src tests
```

For performance conclusions, rebuild and benchmark with:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  uv run maturin develop --release
```

Update `memory_bank/progress.md` only after a meaningful native feature is
complete and validated.

## 14. Non-Goals for the First Migration

Do not include the following in the first native implementation:

- rewriting `AccountGraph`;
- rewriting Polars schema and timestamp expressions;
- changing the public graph classes;
- introducing a public `backend="python" | "rust"` parameter;
- adding Rayon or other parallel execution;
- adding unsafe Rust;
- adding a graph framework dependency;
- designing a generic native graph engine;
- introducing Arrow zero-copy bindings before conversion is measured;
- changing graph edge semantics;
- redesigning snapshot sampling;
- changing edge storage to CSR or another public representation.

Add one of these only when profiling or a concrete requirement proves it is
necessary.

## 15. Large-Edge-Count Warning

Rust cannot remove the fundamental cost of emitting `E` edges.

If a large `delta` causes hundreds of millions or billions of valid edges, the
next bottleneck may be output storage rather than edge search. In that case,
evaluate a separate design for:

- chunked or streaming edge generation;
- compact integer edge storage;
- COO or CSR representations;
- snapshot-local graph construction;
- bounded temporal neighborhoods;
- maximum successors per transaction.

These changes affect storage or scientific semantics and must not be bundled
silently into the first Rust migration.

## 16. Codex Handoff Checklist

Before implementation:

- [ ] Read `AGENTS.md`.
- [ ] Read `memory_bank/lib_design.md`.
- [ ] Read `memory_bank/progress.md`.
- [ ] Use CodeGraph to inspect the current graph API and callers.
- [ ] Run impact analysis before changing existing public symbols.
- [ ] Preserve unrelated working-tree changes.
- [ ] Establish Python baseline runtime and memory.
- [ ] Confirm the temporal-edge loop passes the Rust eligibility gates.

During implementation:

- [ ] Keep public Python APIs unchanged.
- [ ] Use one coarse-grained native call.
- [ ] Keep `rust/lib.rs` thin.
- [ ] Use safe, deterministic, single-threaded Rust first.
- [ ] Preserve strict-lower/inclusive-upper temporal boundaries.
- [ ] Avoid per-edge Python objects at the native boundary.

Before completion:

- [ ] Run Rust unit tests.
- [ ] Rebuild the extension with Maturin.
- [ ] Run Python equivalence and integration tests.
- [ ] Run Ruff checks.
- [ ] Run release benchmarks.
- [ ] Report `N`, `E`, runtime, and peak memory.
- [ ] Update project progress only after the feature is complete.
- [ ] Commit and push only intended files after all checks pass.

## 17. Final Recommendation

The first native experiment should answer one narrow question:

> Does replacing Python temporal-successor indexing and edge emission with one
> bulk Rust kernel materially improve end-to-end graph construction while
> preserving exact AMLGraphX semantics?

If the answer is yes, adopt the private Rust kernel behind the existing Python
builder. If the answer is no, retain the Polars/Python implementation and avoid
further native complexity until a new measured bottleneck appears.
