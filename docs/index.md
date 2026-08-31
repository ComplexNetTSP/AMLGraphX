# AMLGraphX

**Graph machine learning for anti-money laundering and transaction-fraud research.**

AMLGraphX is a Python library for turning financial transactions into explicit graph and temporal representations with clearly stated data and time semantics.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Start with data
:link: getting_started/quickstart
:link-type: doc

Load a supported dataset or normalize a Polars transaction table, then build your first graph.
:::

:::{grid-item-card} Understand the semantics
:link: concepts/index
:link-type: doc

Choose between account and transaction nodes, snapshots and event streams, and strict or transductive time protocols.
:::

:::{grid-item-card} Find the Python API
:link: api/index
:link-type: doc

Browse the supported public modules for datasets, graph preparation, splitting, features, and baselines.
:::
::::

## What AMLGraphX provides

- Dataset adapters that produce a canonical transaction representation while retaining the original columns.
- Explicit account-node and transaction-node graph builders.
- Separate static, snapshot, and continuous account-event representations.
- Time-aware split utilities that make the evaluation protocol visible.
- Interoperability with Polars, PyTorch, and PyTorch Geometric.

The public interface is Python-first. Implementation details are deliberately not part of the user workflow: use documented `amlgraphx` APIs rather than private modules.

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
concepts/index
datasets/index
guides/index
api/index
```
