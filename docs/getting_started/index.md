# Getting started

AMLGraphX is organised around a research pipeline rather than a single model:

```text
transactions → canonical table → graph representation → temporal protocol → model or baseline → evaluation
```

Start with the shortest path below.

1. [Install AMLGraphX](installation) and verify the environment.
2. Follow the [quickstart](quickstart) with a small in-memory transaction table.
3. Read the [core concepts](../concepts/index) before designing an experiment.
4. Use the [dataset overview](../datasets/index) or [workflow guides](../guides/index) for a real AML research task.

## What you need to know

AMLGraphX uses Polars tables at its data boundary. Dataset adapters return `polars.LazyFrame` objects, so downloading a dataset does not immediately load the complete transaction table into memory. Graph builders accept either a Polars `DataFrame` or `LazyFrame` and return AMLGraphX graph objects. Convert to PyTorch Geometric only when a model needs tensors.

Before a realistic experiment, make three explicit decisions:

- What a node represents: an account or a transaction.
- Which temporal representation is required: static, snapshots, or an event stream.
- Whether the evaluation protocol is strict causal or transductive.

```{toctree}
:hidden:

installation
quickstart
```
