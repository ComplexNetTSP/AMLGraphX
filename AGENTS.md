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
