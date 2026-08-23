
# Library Design Principles

This document summarizes the software design and open-source development principles that AMLGraphX adopts from mature scientific Python libraries such as **NetworkX** and **PyTorch Geometric (PyG)**.

The goal is not to reproduce their current directory structures. Large projects evolve over many years, and their repository layouts naturally contain historical decisions.

Instead, AMLGraphX should learn from the principles that allow these projects to remain stable, extensible, understandable, and maintainable as features and contributors increase.

---

## 1. Core Philosophy

AMLGraphX follows four fundamental principles:

> **Stable interfaces, flexible internals, composable components, reproducible research.**

In practice:

- Public APIs should change slowly.
- Internal implementations should be free to evolve.
- Components should have clear responsibilities.
- Research workflows should remain reproducible.
- New functionality should integrate with existing abstractions.
- Architecture should evolve from actual project requirements rather than speculative future needs.

The goal is long-term maintainability rather than short-term convenience.

---

# 2. Stable Public API, Flexible Internals

The most important long-term contract of a library is its **public API**, not its file structure.

For example:

```python
from amlgraphx.datasets import IBMAmlDataset
```

should ideally remain stable even if the internal implementation changes from:

```text
datasets/ibm_aml.py
```

to:

```text
datasets/
    ibm/
        dataset.py
        parser.py
        schema.py
```

Users should depend on stable concepts, not internal file locations.

Therefore AMLGraphX should distinguish between:

- **Public API**
- **Internal implementation**

Internal code may be reorganized, optimized, or rewritten without unnecessarily breaking user code.

A useful rule is:

> Internal structure may evolve frequently. Public concepts should evolve slowly.

---

# 3. Architecture Should Evolve Naturally

Do not create complex directory structures before the project actually requires them.

For example:

```text
graphs.py
```

is perfectly reasonable while graph functionality is still small.

If it eventually grows to contain independent concepts such as:

```text
transaction graphs
account graphs
temporal graphs
heterogeneous graphs
PyG conversion
NetworkX conversion
```

then it may naturally become:

```text
graphs/
```

The decision should come from real complexity.

A useful rule is:

> Split modules because the current code is difficult to understand, not because future complexity is imagined.

Avoid premature architecture.

---

# 4. Organize Around Concepts, Not Papers

A research library should not become a collection of independent paper repositories.

Avoid structures where every research method separately implements:

```text
dataset loading
preprocessing
splitting
sampling
features
evaluation
```

Instead, organize functionality around reusable concepts:

```text
dataset
schema
preprocessing
split
sampling
graph
feature
model
metric
evaluation
```

A research method should be a composition of reusable components.

For example:

```python
dataset = IBMAmlDataset(...)
transactions = dataset.load()

split = temporal_split(transactions)

graph = build_transaction_graph(split.train)

model = SomeModel(...)
```

New research methods should adapt to AMLGraphX abstractions rather than creating new isolated workflows.

---

# 5. Clear Responsibility Boundaries

Each component should have a limited and understandable responsibility.

For example, a dataset component may reasonably handle:

```text
metadata
download
integrity checking
parsing
schema normalization
cache management
```

but it should generally not also perform:

```text
graph algorithms
model training
evaluation
visualization
feature engineering
```

Avoid large objects such as:

```python
dataset.download()
dataset.split()
dataset.build_graph()
dataset.compute_features()
dataset.train()
dataset.evaluate()
dataset.plot()
```

Prefer composable operations:

```python
transactions = dataset.load()

split = temporal_split(transactions)

graph = build_graph(split.train)

features = compute_features(graph)
```

Small components are easier to understand, test, replace, and optimize.

---

# 6. Prefer Functions Unless State Is Necessary

Not everything needs to be implemented as a class.

Classes are useful when the abstraction contains persistent state, for example:

```text
Dataset
Model
Sampler
Configuration
Cache
Registry
```

Stateless operations should usually remain functions:

```python
temporal_split(...)
build_transaction_graph(...)
compute_fan_in(...)
lift_at_k(...)
```

Avoid unnecessary abstraction layers.

Prefer:

```python
compute_fan_in(graph)
```

over unnecessarily complex patterns such as:

```python
FanInDegreeCalculatorFactoryManager(...)
```

Simple APIs are easier to understand and maintain.

---

# 7. Canonical Vocabulary

A mature library should use one consistent term for one concept.

Avoid using many names for approximately the same abstraction:

```text
loader
reader
parser
dataloader
data_reader
```

unless they actually mean different things.

AMLGraphX should gradually establish a canonical vocabulary such as:

```text
Dataset
Loader
Sampler
Split
Transform
Graph
Feature
Model
Metric
Evaluator
Schema
```

Contributors should not need to guess which word to use.

Naming consistency becomes increasingly important as the contributor community grows.

---

# 8. Naming Conventions

Follow standard Python conventions consistently.

## Modules

Use `snake_case`.

```text
temporal_split.py
graph_builder.py
ibm_aml.py
```

## Functions and variables

Use `snake_case`.

```python
temporal_split()
build_transaction_graph()

edge_timestamp
source_account
```

## Classes

Use `PascalCase`.

```python
IBMAmlDataset
TemporalNeighborSampler
TransactionSchema
```

## Constants

Use `UPPER_CASE`.

```python
DEFAULT_CACHE_DIR
SUPPORTED_DATASETS
```

## Internal APIs

Use a leading underscore where appropriate.

```python
_prepare_data()
_validate_schema()
```

Avoid names such as:

```text
utils2.py
new_utils.py
model_final.py
model_final2.py
misc.py
helper_new.py
```

Generic modules such as `utils.py` should also remain limited.

If a function clearly belongs to a domain concept, it should live close to that concept.

---

# 9. Readability Over Cleverness

Scientific software should optimize for readability rather than minimal line count.

Complex one-line expressions may be convenient during experiments but difficult to maintain.

Prefer:

```python
def collect_past_transactions(...):
    ...

def compute_neighbor_counts(...):
    ...

def compute_fan_in(...):
    ...
```

over deeply nested transformations that combine several concepts at once.

A useful question is:

> Can another researcher understand why this code is correct six months later?

AML/Fraud research already contains substantial domain complexity:

```text
temporal leakage
dynamic graph sampling
transaction semantics
motifs
entity relationships
anomaly patterns
```

Implementation style should reduce additional cognitive complexity.

---

# 10. Canonical Data Representation

AML datasets frequently represent identical concepts using incompatible field names.

Examples include:

```text
Sender / Receiver
nameOrig / nameDest
from_id / to_id
account_from / account_to
```

AMLGraphX should gradually establish a canonical representation for common concepts.

For example:

```text
source
target
timestamp
amount
currency
label
```

Conceptually:

```text
Raw dataset
     ↓
Dataset-specific parser
     ↓
AMLGraphX canonical representation
     ↓
Split / Graph / Features / Models / Evaluation
```

This common representation enables downstream algorithms to work across multiple datasets.

A stable data abstraction may ultimately provide more value to researchers than any individual model implementation.

---

# 11. Dataset Metadata and Reproducibility

Research reproducibility requires more than simply loading CSV files.

Datasets should make it possible to track information such as:

```text
dataset name
source
version
checksum
license
citation
schema version
processing version
label definition
timestamp semantics
dataset variant
```

The intended relationship is:

```text
raw source
    ↓
processing version
    ↓
canonical representation
    ↓
experiment
    ↓
published result
```

Researchers should eventually be able to reproduce the exact data processing pipeline used by a previous AMLGraphX release.

---

# 12. Splitting Is Research Infrastructure

Train/test splitting is especially important in AML and Fraud research.

Poor evaluation design may introduce:

```text
future leakage
temporal leakage
entity leakage
duplicate information
unrealistic evaluation conditions
```

Therefore splitting strategies should be treated as first-class research components rather than small convenience utilities.

For example:

```python
split = temporal_split(
    transactions,
    train_ratio=0.7,
    val_ratio=0.1,
)
```

The behavior should clearly define:

```text
timestamp boundaries
future-information restrictions
entity overlap
ordering assumptions
```

A standardized evaluation protocol can provide substantial long-term value to the research community.

---

# 13. Models Are Replaceable Components

Models change quickly.

Datasets, schemas, evaluation protocols, and metrics usually have much longer lifetimes.

Therefore the library should conceptually distinguish between:

## Stable infrastructure

```text
Dataset
Schema
Split
Graph
Sampling
Features
Metrics
Evaluation
```

and:

## Rapidly changing research methods

```text
Models
Baselines
Experimental algorithms
```

New models should integrate with existing library concepts.

Avoid model-specific pipelines such as:

```python
model.prepare_ibm_data()
model.perform_temporal_split()
model.compute_features()
model.train_special_pipeline()
```

Prefer:

```python
dataset = IBMAmlDataset(...)
split = temporal_split(dataset.load())

model = NewModel(...)
output = model(...)
```

The model should adapt to the framework rather than redefining the framework.

---

# 14. Preserve Ecosystem Interfaces

When AMLGraphX uses an established ecosystem, follow its conventions when possible.

For example, PyTorch models should normally remain standard:

```python
torch.nn.Module
```

and support expected behavior such as:

```python
forward()
state_dict()
parameters()
train()
eval()
```

Avoid inventing incompatible interfaces without a strong technical reason.

Following ecosystem conventions improves interoperability and reduces the learning burden for users.

---

# 15. Backend Implementation Should Be Replaceable

Public interfaces should hide implementation choices whenever practical.

For example:

```python
compute_fan_in(transactions)
```

might initially use Python or Polars.

Later the same functionality might use:

```text
Rust
C++
parallel execution
GPU acceleration
```

The researcher should ideally not need to rewrite experiment code.

Conceptually:

```text
Public Python API
       ↓
Internal implementation
       ↓
Python / Polars / Rust / C++ / GPU
```

Performance implementations may change.

The user-facing abstraction should remain stable.

---

# 16. Tests Are a Safety Net for Future Development

Tests are not only for detecting bugs in newly written code.

Their larger purpose in an open-source library is to allow future contributors to modify implementations safely.

A contributor should be able to refactor code and run:

```bash
pytest
```

to determine whether established behavior has been broken.

Tests therefore act as a contract between:

```text
existing behavior
      ↕
future contributors
```

Without this safety net, maintainers eventually become afraid to refactor their own project.

---

# 17. Test Behavior, Not Implementation

Tests should focus on observable behavior rather than internal implementation details.

Avoid:

```python
assert dataset._internal_buffer == expected
```

Prefer:

```python
data = dataset.load()

assert set(data.columns) == expected_columns
assert len(data) == expected_size
```

This allows internal implementations to change without invalidating unrelated tests.

For example:

```text
Pandas → Polars
Python → Rust
single-thread → parallel
```

should ideally preserve the same behavioral test suite.

---

# 18. Scientific Correctness Is More Important Than Coverage Numbers

Code coverage is useful, but high coverage does not necessarily mean scientifically correct software.

Tests should verify domain properties.

Examples:

```python
def test_temporal_split_does_not_use_future_transactions():
    ...
```

```python
def test_fan_in_counts_known_incoming_edges():
    ...
```

```python
def test_cycle_detector_finds_known_cycle():
    ...
```

```python
def test_temporal_sampler_uses_only_past_edges():
    ...
```

Tests should encode known mathematical, graph, temporal, and AML properties.

A test such as:

```python
assert result is not None
```

provides relatively little protection.

---

# 19. Small Deterministic Test Data

Continuous integration should not depend on large external AML datasets.

Use small deterministic fixtures containing intentionally constructed structures such as:

```text
fan-in
fan-out
cycles
layering
burst activity
repeated transfers
rapid movement
cross-border interactions
```

These fixtures should be:

```text
small
fast
deterministic
understandable
independent of external services
```

Large datasets can instead be used for:

```text
integration tests
benchmarks
reproduction experiments
```

---

# 20. Documentation Is Part of the API

A public feature is not complete simply because the implementation works.

Users should be able to understand:

```text
what the function does
expected inputs
returned outputs
assumptions
limitations
important edge cases
relevant citations
```

without reading internal source code.

Documentation also acts as an API design test.

If a function is extremely difficult to explain, the API itself may be unnecessarily complicated.

---

# 21. Examples and Tests Have Different Purposes

Examples demonstrate how researchers are expected to use the library.

Tests verify correctness.

For example:

```text
examples/load_real_datasets.py
```

should demonstrate normal usage.

Tests should remain in the test suite:

```text
tests/test_dataset_loading.py
```

Avoid mixing example scripts and pytest-style test files unless there is a deliberate reason.

---

# 22. Small Pull Requests

Pull requests should solve one coherent problem whenever possible.

Avoid PRs such as:

```text
Add dataset
+ graph conversion
+ sampler
+ model
+ API refactor
+ documentation rewrite
```

Prefer:

```text
PR 1 — Add PaySim dataset support
PR 2 — Add temporal splitting
PR 3 — Add graph conversion
PR 4 — Add model integration
```

Small PRs improve:

```text
review quality
debugging
rollback safety
Git history
contributor feedback
maintainability
```

This is especially important once multiple people begin contributing.

---

# 23. Review Contributions from Four Perspectives

Every significant feature should be reviewed from four perspectives.

## API

Is the public interface understandable and consistent?

## Correctness

Is the implementation scientifically and technically correct?

For AMLGraphX this may include:

```text
temporal leakage
timestamp interpretation
label mapping
future information usage
transaction semantics
graph construction correctness
```

## Tests

Will future modifications reveal if the behavior is broken?

## Documentation

Can another researcher use the feature without reading the implementation?

A feature is not truly complete until all four dimensions are addressed.

---

# 24. Contribution Workflow

Large features should preferably begin with design discussion rather than immediately appearing as thousands of lines of code.

A healthy workflow is:

```text
Issue / Discussion
        ↓
Design agreement
        ↓
Implementation
        ↓
Tests
        ↓
Documentation
        ↓
Pull Request
        ↓
CI
        ↓
Review
        ↓
Merge
```

Small bug fixes do not necessarily require prior discussion.

Large architectural changes generally should.

Examples include:

```text
changing canonical data representation
adding a mandatory dependency
redesigning dataset APIs
changing schema
introducing backend dispatch
changing model interfaces
```

Architectural decisions should not accidentally be introduced inside unrelated PRs.

---

# 25. Automated Style Enforcement

Formatting and basic style should be automated.

Human reviewers should spend their time evaluating:

```text
API design
scientific correctness
maintainability
architecture
documentation
```

rather than:

```text
spacing
import order
line formatting
```

Tooling may change over time, but automated checks may include:

```text
formatting
linting
type checking
unit tests
package build
documentation build
```

Contributors should not need to guess project style.

---

# 26. Type Hints for Public Interfaces

Public interfaces should use type annotations where they improve clarity.

Prefer:

```python
def temporal_split(
    transactions: TransactionFrame,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> DatasetSplit:
    ...
```

over:

```python
def split(data, ratio, mode=None):
    ...
```

Keyword arguments are especially useful when a function contains multiple configuration options.

Readable APIs reduce misuse.

---

# 27. Deprecation Instead of Sudden Breakage

Stable libraries still need to improve their APIs.

API mistakes should therefore not mean that old interfaces must remain forever.

Instead, use a deprecation lifecycle.

For example:

```text
Version N
Old API works normally

Version N+1
Old API still works but emits a deprecation warning

Version N+2
Old API may be removed
```

Example:

```python
warnings.warn(
    "`load_ibm()` is deprecated; use `IBMAmlDataset.load()`.",
    DeprecationWarning,
    stacklevel=2,
)
```

The important principle is:

> Users should receive migration time before public APIs disappear.

---

# 28. Versioning Represents Compatibility

Versions should communicate changes in compatibility and functionality.

During early development:

```text
0.x.y
```

allows the project to evolve rapidly.

As the core abstractions become stable, AMLGraphX may eventually establish:

```text
1.0.0
```

as its first stable API contract.

Afterward, releases should distinguish approximately between:

```text
bug fixes
backward-compatible features
breaking changes
```

Researchers should eventually be able to write:

```text
Experiments were conducted using AMLGraphX vX.Y.Z.
```

and reproduce the same behavior.

---

# 29. Releases Should Be Reproducible Events

A release should be more than simply uploading the current repository state.

A release should ideally correspond to:

```text
known source commit
version number
Git tag
changelog
package artifact
documentation state
```

This creates a stable reference for published research.

Reproducibility requires both:

```text
data version
software version
```

---

# 30. Experimental Features Should Not Immediately Become Stable APIs

Research moves quickly.

A newly implemented algorithm may still have uncertain:

```text
API design
correctness
maintenance cost
community interest
performance
```

Experimental functionality should therefore be allowed to mature before becoming part of the long-term public contract.

Conceptually:

```text
experimental
    ↓
validation
    ↓
tests
    ↓
documentation
    ↓
API stabilization
    ↓
core library
```

This allows innovation without making every experimental decision permanent.

---

# 31. Dependencies Have Maintenance Cost

Adding a dependency is not free.

Every dependency introduces possible issues involving:

```text
installation
version compatibility
platform compatibility
security
maintenance
CI complexity
```

Therefore mandatory dependencies should only be introduced when they provide substantial value.

Optional capabilities can often use optional dependencies.

The core installation should remain reasonably lightweight whenever possible.

---

# 32. Maintainers Protect Consistency

As the project grows, the maintainer's role changes.

A contributor often asks:

> Does my feature work?

The maintainer must additionally ask:

```text
Does the API fit existing conventions?

Will this still make sense in several years?

Does this create unnecessary maintenance burden?

Is a new dependency justified?

Can the implementation be replaced later?

Can another researcher understand it?

Can future contributors safely modify it?
```

Maintainers therefore protect the conceptual consistency of the library.

Maintaining architecture becomes increasingly important as contribution volume grows.

---

# 33. Prefer Composable Primitives

A research library should expose small building blocks that can be combined in different ways.

Prefer components such as:

```text
load dataset
normalize schema
split transactions
construct graph
sample neighborhood
compute feature
run model
evaluate prediction
```

rather than one large workflow:

```python
run_complete_aml_pipeline(...)
```

Convenience pipelines may exist, but they should usually be built on top of reusable primitives.

Composable primitives make the library useful for research because researchers frequently need to replace individual steps.

---

# 34. Separate Mechanism from Policy

Whenever possible, distinguish between:

- **mechanism** — what the library is capable of doing
- **policy** — a particular experimental choice

For example:

```text
temporal splitting mechanism
```

is different from:

```text
70/10/20 temporal split used in one paper
```

Similarly:

```text
temporal neighborhood sampler
```

is different from:

```text
sample exactly 500 past edges with a specific decay parameter
```

Libraries should provide flexible mechanisms.

Experiment configurations determine policy.

This separation makes research workflows easier to reproduce and extend.

---

# 35. Avoid Hidden Behavior

Research software should prefer explicit behavior.

Avoid functions whose important behavior depends on hidden global state or undocumented defaults.

For example, important decisions such as:

```text
time ordering
sampling direction
random seed
label mapping
normalization
graph direction
```

should be visible through configuration, documentation, or metadata.

Hidden decisions make scientific results difficult to reproduce.

---

# 36. Determinism Should Be Possible

Where algorithms involve randomness, researchers should be able to control it.

For example:

```python
sampler = SomeSampler(seed=42)
```

or equivalent reproducibility mechanisms.

Not every algorithm must always be deterministic, but deterministic execution should be possible where practical.

Randomness should not silently change experimental results between runs.

---

# 37. Performance Optimization Should Follow Profiling

Do not prematurely rewrite functionality in Rust, C++, CUDA, or parallel frameworks simply because performance may eventually matter.

Start with a clear and correct implementation.

Then:

```text
benchmark
profile
identify bottleneck
optimize bottleneck
verify correctness
```

Only optimize the parts that actually matter.

A fast incorrect AML algorithm is less valuable than a clear correct implementation.

Correctness and API design should precede low-level optimization.

---

# 38. Optimization Must Preserve Semantics

When replacing an implementation with a faster backend, existing behavioral tests should still pass.

For example:

```text
Python implementation
        ↓
optimized Rust implementation
```

should return scientifically equivalent results.

Performance improvements must not silently alter:

```text
time semantics
edge ordering
sampling behavior
floating-point assumptions
label handling
```

Benchmarks measure speed.

Tests protect meaning.

---

# 39. Research Reproduction Should Be Separated from Core APIs

Exact paper reproduction may require:

```text
specific hyperparameters
specific data split
specific random seeds
specific preprocessing
training scripts
experiment configuration
```

These details should not necessarily become permanent public APIs.

The library should provide stable primitives.

Reproduction configurations can compose those primitives.

Conceptually:

```text
AMLGraphX core components
        ↓
experiment configuration
        ↓
paper reproduction
```

This prevents paper-specific details from polluting the general library design.

---

# 40. Design for Future Contributors

Every new abstraction should be understandable without requiring knowledge of the entire repository.

A contributor working on one dataset should not need to understand every model.

A contributor working on one metric should not need to understand the download subsystem.

Good module boundaries reduce the amount of context required to contribute.

A useful question when reviewing architecture is:

> How much of the repository must someone understand before they can safely modify this component?

Lower is generally better.

---

# 41. Do Not Over-Engineer Early

AMLGraphX is expected to evolve continuously.

Therefore avoid introducing complexity solely because mature libraries currently contain it.

Do not copy mechanisms such as:

```text
complex plugin architectures
large registry systems
multiple abstraction layers
advanced governance
backend dispatch
deep class hierarchies
```

until real project requirements justify them.

NetworkX and PyG should be studied for their **development philosophy**, not mechanically copied.

A young project should remain simple.

---

# 42. Preferred Evolution Strategy

When adding functionality:

```text
1. Implement the simplest correct version.
2. Define a clear public interface.
3. Add behavioral tests.
4. Document assumptions.
5. Observe real usage.
6. Refactor when patterns become obvious.
7. Optimize when benchmarks reveal bottlenecks.
8. Stabilize APIs only after sufficient experience.
```

This is preferable to attempting to predict the final architecture at the beginning of the project.

---

# 43. Definition of Done

For a meaningful public feature, "done" should normally mean more than:

> The code runs on the author's machine.

A feature should ideally satisfy:

```text
clear responsibility
consistent API
readable implementation
type information where useful
scientific correctness
tests
documentation
appropriate error handling
compatibility with project conventions
```

For research algorithms, also consider:

```text
paper citation
reference implementation
expected inputs
expected outputs
known limitations
reproducibility information
```

---

# 44. AMLGraphX Design Rule

When uncertain about a design decision, prefer the option that preserves:

```text
stable user concepts
clear scientific semantics
composable components
easy testing
future implementation freedom
```

The internal implementation may change dramatically over time.

AMLGraphX may eventually change:

```text
Pandas → Polars
Python → Rust
CPU → GPU
single-process → distributed
custom graph code → external backend
```

without forcing researchers to rewrite their experiments.

That is the purpose of a well-designed library abstraction.

---

# Final Principle

The most important lesson from NetworkX and PyTorch Geometric is not their current repository structure.

It is that successful scientific libraries develop a **stable conceptual language** around their problem domain.

Directories change.

Files move.

Implementations are rewritten.

Models become obsolete.

Dependencies change.

Performance backends change.

But concepts such as:

```text
Dataset
Graph
Sampler
Transform
Model
Metric
```

remain understandable for many years.

AMLGraphX should aim for the same property in the AML/Fraud research domain.

> **Keep the concepts stable, keep the components composable, and allow the implementation to evolve.**
