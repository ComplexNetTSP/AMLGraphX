# Datasets and canonical transactions

`amlgraphx.datasets` owns external dataset behaviour: downloading, archive handling, file discovery, dataset-specific parsing, metadata, and conversion to the canonical transaction representation. It does not build graphs, split data, train models, or calculate metrics.

## Available adapters

| Adapter | Dataset | Supported scope | License recorded by the adapter |
| --- | --- | --- | --- |
| `IBMAML` | IBM AML | `hi-small`, `hi-medium`, `hi-large`, `li-small`, `li-medium`, `li-large` | CDLA-Sharing-1.0 |
| `PaySim` | PaySim | Transaction classification | CC-BY-SA-4.0 |
| `SAML` | SAML-D | Transaction or edge files | CC-BY-NC-SA-4.0 |

The adapters use third-party Hugging Face mirrors. Read each dataset's source and licence terms before redistributing data or publishing results. Dataset metadata is available through each adapter's `metadata` property.

## Load a dataset lazily

```python
from amlgraphx.datasets import IBMAML

dataset = IBMAML("hi-small")
transactions = dataset.transactions()

print(transactions.collect_schema())
```

`transactions` is a Polars `LazyFrame`. It is not materialized until you call an operation such as `collect()`, or pass it to a graph builder that needs a materialized representation. IBM AML also provides `accounts()` and `patterns()` methods.

Use `load_dataset()` when the dataset name is selected through configuration:

```python
from amlgraphx.datasets import load_dataset

dataset = load_dataset("ibm-aml", variant="hi-small")
transactions = dataset.transactions()
```

By default, downloads are cached under the AMLGraphX cache directory. Pass a dedicated `cache_dir` or `local_dir` when an experiment must control where data is stored.

## Canonical transaction contract

Every adapter returns original dataset fields together with canonical columns whenever they can be derived. At minimum, graph construction needs `source` and `target`; temporal graph operations additionally need a valid `timestamp`.

PaySim's `step` is a simulated hour rather than a wall-clock datetime. The adapter retains `step` and adds a logical timestamp anchored at the Unix epoch solely to give the sequence a deterministic temporal order. Interpret results accordingly.

If you provide your own Polars table, use `normalize_transactions()` before generic graph construction. Explicit column arguments are available whenever a source schema cannot be resolved through the supported aliases.

## Reproducibility checklist

- Dataset adapter and variant.
- Dataset repository revision and local preprocessing choices.
- Label definition and class prevalence after filtering.
- Timestamp interpretation, timezone, and any synthetic time mapping.
- Graph type, temporal settings, and split protocol.

These details are part of the experimental method, not incidental loader configuration.
