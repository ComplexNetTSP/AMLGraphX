# AGENTS.md

This file defines the repository-level working rules for AI coding agents such as Codex.

The goal is to keep development consistent, dependency-aware, testable, and aligned with the long-term library design of AMLGraphX.

---

## 1. Required project context

Before making code changes, read:

- `memory_bank/lib_design.md`
- `memory_bank/progress.md`

Use `memory_bank/lib_design.md` as the source of truth for long-term library design principles, API stability, naming, modularity, testing philosophy, and contribution style.

Use `memory_bank/progress.md` to understand the current implementation state, completed modules, ongoing work, known issues, and next development steps.

Do not make architectural decisions that conflict with `memory_bank/lib_design.md` unless the user explicitly requests such a change.

---

## 2. CodeGraph-first workflow

Before modifying, adding, deleting, or refactoring code:

1. Use CodeGraph MCP first to understand the relevant code structure.
2. Use `codegraph_context` or `codegraph_explore` to inspect the relevant modules and symbols.
3. Use `codegraph_callers` / `codegraph_callees` when call relationships matter.
4. Use `codegraph_impact` before editing existing functions, classes, or public APIs.
5. Only after understanding the dependency graph, inspect source files as needed and make changes.
6. After changes, run the relevant tests and linting checks.

Do not start editing code before performing the CodeGraph analysis unless:

- the change is purely documentation/text,
- CodeGraph is unavailable,
- or the affected code is not indexed.

If CodeGraph is unavailable or the relevant code is not indexed, state this briefly and continue by inspecting the source directly.

---

## 3. Understand before editing

Do not modify code only from a filename or a partial snippet.

Before changing an existing implementation:

- identify the relevant public API,
- identify the implementation symbol,
- inspect its callers and callees when relevant,
- inspect likely downstream impact,
- identify related tests,
- identify whether the change affects documented behavior.

For changes to existing public functions, classes, dataset interfaces, schemas, or other stable APIs, use `codegraph_impact` before editing.

Prefer changing internal implementation without breaking public APIs whenever possible.

---

## 4. Follow the library design principles

AMLGraphX is a research-oriented open-source Python library.

Development should follow the principles defined in `memory_bank/lib_design.md`, especially:

- stable public interfaces,
- flexible internal implementations,
- composable components,
- reproducible research,
- clear responsibility boundaries,
- consistent terminology and naming,
- readable code over clever code,
- behavior-oriented tests,
- evolutionary architecture instead of premature over-engineering.

Do not create unnecessary directories, abstraction layers, registries, factories, or framework machinery unless the current implementation genuinely requires them.

New functionality should integrate with existing project concepts rather than creating isolated paper-specific pipelines.

---

## 5. Public API discipline

Treat public APIs as compatibility contracts.

Before changing a public API:

1. inspect its usage with CodeGraph,
2. evaluate downstream impact,
3. prefer backward-compatible changes,
4. update tests,
5. update documentation if behavior changes.

Do not casually rename, move, or remove public classes, functions, parameters, or import paths.

If a breaking change is necessary, prefer a migration or deprecation path unless the user explicitly requests an immediate breaking change.

Internal code may be reorganized more freely as long as public behavior remains stable.

---

## 6. Naming and readability

Follow standard Python conventions:

- modules: `snake_case`
- functions: `snake_case`
- variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_CASE`
- internal/private helpers: leading underscore when appropriate

Use one canonical term for one concept.

Avoid ambiguous or temporary names such as:

- `utils2.py`
- `new_utils.py`
- `final_model.py`
- `model_final2.py`
- `misc.py`
- `helper_new.py`

Prefer explicit domain-oriented names.

Keep functions and classes focused.

Prefer clear intermediate steps and descriptive helper functions over dense or clever one-liners.

---

## 7. Adding new modules or features

When implementing a new feature or module:

1. inspect existing related abstractions with CodeGraph,
2. reuse existing concepts where appropriate,
3. avoid duplicating existing functionality,
4. preserve naming consistency,
5. keep the implementation focused,
6. add or update tests when needed,
7. run Ruff checks,
8. update relevant documentation when the public behavior needs explanation.

A new feature should fit the existing library rather than create a separate mini-framework.

Do not add abstractions for hypothetical future requirements.

---

## 8. Testing requirements

New features, bug fixes, and behavior changes should include tests when appropriate.

Tests should focus on observable and scientific behavior rather than internal implementation details.

Prefer tests such as:

- known inputs produce known outputs,
- temporal operations do not use future information,
- graph structures have expected nodes and edges,
- dataset loaders produce the expected canonical representation,
- algorithms preserve documented semantics,
- regressions remain fixed.

Avoid tests that only assert that a result is not `None` unless that is genuinely meaningful.

Avoid tightly coupling tests to private implementation details when behavior-level testing is possible.

For large external datasets, prefer small deterministic fixtures for normal CI.

---

## 9. Ruff and code quality

For code changes, run the relevant Ruff checks.

At minimum, use the repository's configured Ruff commands for:

- linting,
- formatting checks or formatting.

Do not manually invent style rules that conflict with `pyproject.toml`.

If Ruff reports issues caused by the change, fix them before considering the task complete.

Do not introduce unrelated large-scale formatting changes unless requested.

---

## 10. Validation before completion

After implementing a code change:

1. run the most relevant targeted tests first,
2. fix failures,
3. run Ruff,
4. when practical, run the broader test suite,
5. verify that the changed public behavior matches the intended design.

Do not claim completion while relevant tests are failing.

If a test cannot be run because of unavailable data, environment limitations, missing services, or unsupported hardware, report that explicitly.

---

## 11. Progress memory

After completing a meaningful module or feature, update:

`memory_bank/progress.md`

The progress update should briefly record:

- what was implemented,
- important design decisions,
- new or changed public APIs,
- tests added or updated,
- known limitations or follow-up work when relevant.

Do not rewrite unrelated historical entries.

Keep `progress.md` concise and useful for future agents.

The purpose of this file is to let future Codex sessions quickly understand the current state of the repository.

---

## 12. Definition of done for code tasks

A code task is normally complete only when:

- relevant CodeGraph analysis was performed,
- the implementation is complete,
- relevant tests were added or updated,
- relevant tests pass,
- Ruff passes for the affected code,
- documentation is updated when needed,
- `memory_bank/progress.md` is updated for a meaningful completed feature or module.

Only then proceed to Git operations.

Pure documentation changes do not require CodeGraph analysis unless they depend on code behavior that must be verified.

---

## 13. Git commit and push

After the implementation is complete and validation passes:

1. inspect `git status`,
2. ensure only intended files are included,
3. create a clear Git commit,
4. push the commit to the current working branch.

Do not commit generated caches, temporary files, local environments, or unrelated changes.

Never include files such as:

- `__pycache__/`
- `*.pyc`
- local virtual environments
- temporary data
- secrets or credentials

Commit messages should describe the completed change clearly.

Examples:

```text
feat: add PaySim dataset loader
fix: prevent future leakage in temporal sampler
test: add transaction graph regression cases
docs: document IBM AML field mapping
refactor: simplify dataset registry
```

Do not create a commit or push if relevant tests or Ruff checks are failing.

If push fails because of authentication, permissions, remote state, or branch protection, report the failure instead of hiding it.

---

## 14. Keep changes focused

Avoid mixing unrelated work in one task.

For example, do not combine:

- a new dataset,
- a model implementation,
- a major API redesign,
- unrelated refactoring,
- repository-wide formatting

unless the user explicitly asks for them together.

Prefer small, reviewable, reversible changes.

This improves:

- CodeGraph impact analysis,
- test coverage,
- review quality,
- Git history,
- future debugging.

---

## 15. Do not silently expand scope

If implementing one feature reveals unrelated cleanup opportunities, do not automatically refactor the entire surrounding codebase.

Make only the changes required for correctness, maintainability, and integration of the requested feature.

Small local cleanup is acceptable when it directly supports the implementation.

Large unrelated refactors should be left for a separate task.

---

## 16. Research correctness has priority

AMLGraphX is intended for AML/Fraud research.

Scientific correctness takes priority over superficial abstraction or optimization.

Pay special attention to:

- temporal leakage,
- future information leakage,
- label semantics,
- timestamp semantics,
- graph direction,
- transaction ordering,
- dataset schema consistency,
- sampling assumptions,
- reproducibility,
- deterministic behavior where appropriate.

When optimizing an implementation, preserve existing semantics and verify them with tests.

---

## 17. Performance work

Do not introduce Rust, C++, parallelism, GPU implementations, or complex acceleration solely because a function may become slow in the future.

Prefer:

1. correct implementation,
2. benchmark,
3. profile,
4. identify the real bottleneck,
5. optimize,
6. verify identical behavior with tests.

Performance backends may change.

The public Python API should remain stable whenever possible.

---

## 18. Final working rule

When uncertain, prefer the solution that maximizes:

- API stability,
- scientific clarity,
- readability,
- composability,
- testability,
- reproducibility,
- future implementation freedom.

Use CodeGraph to understand dependencies before changing code.

Use `memory_bank/lib_design.md` to guide architecture.

Use `memory_bank/progress.md` to preserve development continuity.

For meaningful code changes:

> **Analyze → implement → test → Ruff → update progress → commit → push.**



## Rust and native acceleration

AMLGraphX may use Rust for performance-critical internal implementations.

Rust is an implementation backend, not the primary public interface of the library.

The preferred architecture is:

> stable Python public API → internal Python orchestration → Rust performance kernels

Users should normally interact with Python APIs rather than importing low-level Rust bindings directly.

Do not expose implementation-language details unnecessarily through the public API.

Rust may be used when profiling or benchmarking shows that an operation benefits materially from native execution, especially for workloads involving:

- large graph construction,
- adjacency or sparse structure construction,
- graph traversal,
- temporal neighborhood sampling,
- large-scale aggregation,
- motif or pattern counting,
- connected-component computation,
- repeated graph indexing,
- memory-intensive graph transformations,
- CPU-bound loops,
- high-volume transaction processing,
- parallel graph operations.

Do not rewrite functionality in Rust merely because Rust is available.

Prefer:

1. implement correct semantics,
2. test them,
3. benchmark,
4. profile,
5. identify the bottleneck,
6. move only the performance-critical portion to Rust,
7. verify semantic equivalence.

If an existing optimized library such as Polars, NumPy, PyTorch, PyArrow, or another mature backend already performs the operation efficiently, do not reimplement it in Rust without demonstrated benefit.

---

## Python and Rust responsibility boundaries

Keep responsibilities clear between Python and Rust.

Python should normally remain responsible for:

- public APIs,
- configuration,
- validation,
- dataset interfaces,
- orchestration,
- error presentation,
- interoperability with PyTorch and PyG,
- Polars and PyArrow workflows,
- high-level graph abstractions,
- user-facing documentation.

Rust should normally be responsible for narrowly scoped performance-critical kernels.

Avoid moving high-level application logic into Rust unless there is a strong technical reason.

Do not allow Rust implementation details to dictate awkward Python APIs.

The Python interface should remain idiomatic, readable, and stable even if the native backend changes internally.

---

## Rust module design

Do not place substantial implementations directly in the Rust extension entry point.

Keep the extension entry point thin and primarily responsible for:

- registering functions,
- registering classes,
- exposing native modules,
- connecting Rust implementations to Python bindings.

Organize substantial Rust functionality into focused modules based on domain responsibility.

Prefer domain-oriented modules over generic collections of helpers.

Good conceptual boundaries include:

- graph construction,
- graph storage,
- adjacency structures,
- temporal operations,
- sampling,
- traversal,
- AML-specific structural algorithms,
- aggregation,
- indexing.

Avoid generic modules with unclear responsibility such as:

- `utils`,
- `misc`,
- `helpers`,
- `common2`,
- `new_core`.

Do not create deep Rust module hierarchies before the implementation requires them.

Start simple and reorganize when multiple related native kernels justify a dedicated module.

---

## Rust readability

Rust code must remain understandable to contributors who primarily work in Python.

Prefer straightforward, explicit Rust over highly abstract, macro-heavy, or overly clever implementations.

Avoid unnecessary use of:

- advanced lifetime abstractions,
- custom macros,
- complicated generic hierarchies,
- unsafe Rust,
- obscure iterator constructions,
- premature zero-copy abstractions,
- complex concurrency machinery.

Use descriptive names and small focused functions.

When Rust syntax or ownership requirements force a non-obvious implementation, add a short comment explaining the design reason rather than explaining basic Rust syntax.

Do not optimize for minimum line count.

Optimize for correctness, performance where needed, and maintainability.

---

## Unsafe Rust

Avoid `unsafe` unless it is clearly justified by measurable performance or interoperability requirements.

Before introducing `unsafe`:

1. determine whether safe Rust is sufficient,
2. document why `unsafe` is required,
3. isolate the unsafe operation behind a small safe interface,
4. define the invariants that make the operation sound,
5. add tests covering relevant boundary conditions.

Do not introduce `unsafe` solely to gain speculative performance improvements.

---

## Python–Rust boundary

Crossing the Python–Rust boundary has overhead.

Do not design native APIs that repeatedly transfer individual elements between Python and Rust.

Prefer coarse-grained operations such as:

> pass a collection or array → perform substantial work in Rust → return the result

instead of:

> Python loop → call Rust once per transaction or edge.

Avoid unnecessary copying of large graph or transaction data when practical, but do not introduce complicated zero-copy designs until profiling shows that data transfer is a meaningful bottleneck.

Preserve clear ownership and lifetime semantics at the boundary.

---

## Native API stability

Treat low-level native bindings as internal implementation details unless explicitly designated public.

Prefer Python wrappers around native functions.

For example, the stable API should conceptually be:

> public Python function → internal native implementation

rather than requiring users to depend directly on the native extension.

This allows the implementation to change between:

- Python,
- Rust,
- C++,
- GPU kernels,
- external graph engines,

without unnecessarily breaking users.

If a native function replaces an existing Python implementation, preserve the existing observable behavior unless a behavior change is explicitly requested.

---

## Semantic equivalence for optimized implementations

Performance optimization must not silently change scientific behavior.

When replacing or accelerating an existing implementation with Rust, verify equivalence for relevant cases including:

- graph direction,
- node identity,
- edge identity,
- transaction ordering,
- timestamp ordering,
- duplicate transactions,
- self-loops,
- missing values,
- temporal boundaries,
- future-information exclusion,
- sampling semantics,
- deterministic behavior,
- output types,
- output shapes.

Where floating-point computation is involved, use appropriate numerical tolerances instead of requiring unnecessary bitwise identity.

---

## Rust testing

Meaningful Rust functionality should have Rust-side tests when appropriate.

Use Rust unit tests for:

- internal algorithms,
- data structures,
- edge cases,
- invariants,
- parsing or indexing logic,
- native-only implementation details.

Also add Python-level integration tests for functionality exposed through the Python API.

Do not rely exclusively on Rust unit tests for native functionality that is consumed by Python.

Python integration tests should verify that:

- the native extension can be imported,
- Python inputs are accepted correctly,
- native outputs match documented behavior,
- Python-facing exceptions are appropriate,
- native and Python implementations agree when both exist.

For optimized replacements, use shared deterministic fixtures whenever possible so the Python and Rust implementations can be compared directly.

---

## Rust validation

For meaningful Rust changes, run the relevant native checks.

At minimum, when applicable:

```text
cargo check
cargo test
cargo fmt --check
cargo clippy
```

If the Rust code is exposed to Python, also rebuild or install the native extension through the repository's configured build workflow and run the relevant Python tests.

A Rust change is not complete merely because `cargo check` succeeds.

The Python integration must also work.

If native compilation cannot be performed because of platform, toolchain, hardware, or environment limitations, report that explicitly.

---

## Rust dependencies

Add Rust dependencies conservatively.

Before introducing a new crate:

1. determine whether the standard library or an existing dependency already solves the problem,
2. verify that the crate is actively maintained and appropriate for library use,
3. avoid pulling in large dependency trees for small functionality,
4. avoid duplicate crates serving the same conceptual purpose.

Prefer mature and well-maintained crates.

Do not introduce a graph framework solely to avoid implementing a small graph kernel if doing so creates substantial architectural coupling.

Keep native dependencies implementation-focused and avoid exposing crate-specific types through the Python public API.

---

## Parallelism

Do not automatically parallelize Rust code.

Parallel execution should be introduced when:

- the workload is sufficiently large,
- profiling shows meaningful benefit,
- ordering semantics are preserved where required,
- deterministic behavior remains understood,
- memory usage remains acceptable.

Be especially careful with temporal AML operations where parallel processing may accidentally violate transaction ordering or temporal constraints.

Do not add parallelism that makes scientific behavior harder to reason about for negligible performance benefit.

---

## Memory and large graph workloads

AMLGraphX may operate on graphs substantially larger than typical in-memory research examples.

For native graph implementations, consider:

- memory complexity,
- unnecessary cloning,
- duplicate representations,
- integer width,
- sparse storage,
- allocation behavior,
- streaming opportunities,
- intermediate materialization.

Avoid holding multiple full copies of large graph structures unless necessary.

Prefer predictable memory behavior over micro-optimizations.

When changing graph representations for performance reasons, preserve the Python-level semantics and document important limitations.

---

## Error handling across Rust and Python

Do not use panics for normal user-facing errors.

Convert expected failure conditions into appropriate Rust results and expose meaningful Python exceptions.

Error messages should describe the user's problem rather than Rust implementation details.

Do not expose messages such as internal indexing failures or low-level parsing state unless they are genuinely useful for debugging.

Unexpected internal invariants may use stronger assertions where appropriate, but user input validation should produce controlled errors.

---

## Benchmark-driven optimization

Performance claims must be supported by measurement.

When implementing or replacing a performance-sensitive kernel:

- benchmark representative workloads,
- include realistic graph sizes when practical,
- distinguish debug and optimized native builds,
- avoid comparing optimized Rust builds against artificially inefficient Python baselines,
- preserve identical semantics between implementations.

Prefer measuring:

- runtime,
- peak or approximate memory usage,
- scaling with nodes or edges,
- scaling with transaction volume.

Do not claim that an implementation is faster merely because it is written in Rust.

---

## Rust changes and CodeGraph

Use CodeGraph for Python-side dependency and API analysis before introducing or changing native-backed behavior.

When CodeGraph does not index Rust or cannot represent Python–Rust relationships completely, supplement CodeGraph analysis by inspecting:

- Python wrappers,
- native binding registrations,
- relevant Rust modules,
- associated tests.

Do not assume that a native implementation is isolated merely because CodeGraph shows few Python dependencies.

Always inspect the Python binding boundary when changing Rust-backed behavior.

---

## Completion requirements for native changes

A meaningful Rust-backed code task is normally complete only when:

- the intended Python-facing behavior is clear,
- existing API compatibility has been considered,
- the Rust implementation is complete,
- Rust tests are added or updated when appropriate,
- Python integration tests are added or updated when appropriate,
- `cargo check` passes,
- `cargo test` passes,
- `cargo fmt --check` passes,
- `cargo clippy` passes,
- relevant Python tests pass,
- Ruff passes for affected Python code,
- performance-sensitive changes are benchmarked when performance is the purpose of the change,
- documentation is updated when public behavior changes,
- progress memory is updated for meaningful completed work.

Performance optimization is not complete until correctness has been verified.

---

## Rust development principle

Rust exists to accelerate AMLGraphX, not to redefine AMLGraphX around Rust.

Prefer:

> Python-first public design, native-backed performance.

Keep the boundary replaceable.

Keep native kernels focused.

Keep semantics testable.

Keep optimization evidence-based.

When uncertain whether functionality belongs in Python or Rust, implement or preserve it in Python unless profiling, scale, memory behavior, or algorithmic requirements provide a clear reason for native implementation.


### Rust extension entry point

Keep `src/lib.rs` as a thin PyO3 extension entry point.

Its primary responsibilities are:

- declaring Rust modules,
- registering Python-exposed functions and classes,
- constructing the native Python module,
- wiring Rust implementations to Python bindings.

Do not place substantial graph algorithms, sampling logic, data structures, or performance kernels directly in `src/lib.rs`.

Implement substantial functionality in dedicated Rust modules and register those functions or classes through `src/lib.rs`.

As the Rust backend grows, `src/lib.rs` should remain small and easy to inspect.




## Rust build and validation workflow

Whenever Rust source code is added, modified, moved, or deleted, rebuild the native extension before considering the task complete.

Use the following workflow:

1. Check Rust compilation:

```text
cargo check
```

2. Run Rust tests:

```text
cargo test
```

3. Check Rust formatting:

```text
cargo fmt --check
```

If formatting fails, run:

```text
cargo fmt
```

and check again.

4. Run Rust linting:

```text
cargo clippy
```

Fix warnings introduced by the change when they are relevant to the modified code.

5. Rebuild the Python native extension:

```text
uv run maturin develop
```

This step is required after Rust code changes so that Python uses the newly compiled native implementation.

Do not assume that Python is using the latest Rust code until `uv run maturin develop` has completed successfully.

6. If the task concerns performance, benchmarking, or production-like native execution, also build the optimized version:

```text
uv run maturin develop --release
```

Use release builds for performance measurements. Do not benchmark Rust performance using only the default debug build.

7. Verify Python import and integration.

Run the relevant Python tests after rebuilding the native extension:

```text
uv run pytest
```

For focused changes, run the most relevant targeted tests first, then run the broader suite when practical.

8. Run Python linting and formatting checks when Python wrappers or interfaces were changed:

```text
uv run ruff check .
uv run ruff format --check .
```

If formatting is required:

```text
uv run ruff format .
```

9. Update `memory_bank/progress.md` after a meaningful Rust-backed module or feature is completed.

10. Only after all relevant validation passes should Git commit and push operations proceed.

The normal Rust-backed development sequence is:

> **CodeGraph analysis → edit Rust/Python → cargo check → cargo test → cargo fmt --check → cargo clippy → maturin develop → Python tests → Ruff → update progress → commit → push**

If only Python code changed and no Rust source, Rust module declaration, native binding, Cargo dependency, or PyO3 registration changed, rebuilding with Maturin is not required.

If any of the following changes, rebuild with Maturin:

- Rust source files,
- Rust module declarations,
- PyO3 functions or classes,
- native module registration,
- Cargo dependencies affecting the extension,
- Python–Rust binding definitions.

For performance-related work, use:

> **correctness build first → validate → release build → benchmark**

Never treat a successful `cargo check` alone as proof that the Python-facing Rust extension works.


As the native backend grows, do not centralize all PyO3 registrations in `src/lib.rs`.

Each substantial Rust module should expose a small `register(...)` function responsible for registering its own Python-exposed functions or classes.

`src/lib.rs` should delegate registration to module-level registration functions and remain a thin top-level composition layer.