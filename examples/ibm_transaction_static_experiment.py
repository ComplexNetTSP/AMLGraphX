"""Train a small transaction-node risk model on IBM AML HI-Small.

This example shows the complete static-graph path:

``dataset -> canonical transactions -> PyG graph -> split loaders -> Experiment``.

Use ``--mode full`` when the graph fits in accelerator memory. Use
``--mode windows`` to keep the same time-aware transaction graph semantics but
score one chronological target window at a time with causal lookback context.
The temporary cache is deleted when the process exits.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from amlgraphx.data import StaticGraphWindowDataset
from amlgraphx.datasets import load_dataset
from amlgraphx.evaluation import Precision, Recall
from amlgraphx.experiments import BinaryRiskTask, Experiment
from amlgraphx.graph import GraphFeatureSpec, prepare_pyg_graph


class TransactionRiskMLP(nn.Module):
    """A deliberately simple researcher-owned transaction-node classifier."""

    def __init__(self, feature_dim: int) -> None:
        """Create a two-layer MLP that returns one logit per transaction node."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, batch: object) -> torch.Tensor:
        """Return raw binary logits; AMLGraphX validates the output shape."""
        features = torch.log1p(batch.x.abs())  # type: ignore[union-attr]
        return self.network(features).squeeze(-1)


def load_transactions(cache_dir: Path, limit: int):
    """Download, lazily load, and materialize a bounded chronological sample."""
    dataset = load_dataset("ibm-aml", variant="hi-small", cache_dir=cache_dir)
    transactions = dataset.transactions().sort("timestamp")
    return temporal_sample(transactions, limit)


def temporal_sample(transactions: pl.LazyFrame, limit: int) -> pl.DataFrame:
    """Sample six time bands so every train/validation/test period is populated."""
    minimum, maximum = (
        transactions.select(
            pl.col("timestamp").min().alias("minimum"),
            pl.col("timestamp").max().alias("maximum"),
        )
        .collect()
        .row(0)
    )
    per_band = max(1, limit // 6)
    pieces: list[pl.DataFrame] = []
    for index in range(6):
        start = minimum + (maximum - minimum) * index / 6
        end = minimum + (maximum - minimum) * (index + 1) / 6
        pieces.append(
            transactions.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
            )
            .limit(per_band)
            .collect()
        )
    return pl.concat(pieces, how="vertical_relaxed").sort("timestamp")


def add_full_graph_masks(data: object) -> None:
    """Attach chronological train/validation/test masks to a full PyG graph."""
    time = data.node_time  # type: ignore[union-attr]
    first = int(time.min())
    last = int(time.max())
    train_end = first + 3 * (last - first) // 5
    validation_end = first + 4 * (last - first) // 5
    data.train_mask = time < train_end  # type: ignore[union-attr]
    data.validation_mask = (time >= train_end) & (time < validation_end)  # type: ignore[union-attr]
    data.test_mask = time >= validation_end  # type: ignore[union-attr]


def window_loader(data: object, start: int, end: int, *, batch_size: int):
    """Return windows whose targets fall in one split while retaining past context."""
    windows = StaticGraphWindowDataset(
        data, window_size=timedelta(days=1), lookback=timedelta(hours=4)
    )
    selected = [
        index
        for index, window_start in enumerate(windows.window_starts)
        if start <= window_start < end
    ]
    if not selected:
        raise ValueError(
            "the selected temporal split contains no complete target windows"
        )
    return DataLoader(Subset(windows, selected), batch_size=batch_size, shuffle=False)


def make_loaders(data: object, mode: str, batch_size: int):
    """Choose full-graph masks or bounded time windows without changing graph meaning."""
    add_full_graph_masks(data)
    if mode == "full":
        loader = DataLoader([data], batch_size=1)
        return loader, loader, loader, BinaryRiskTask("static", "node")

    time = data.node_time  # type: ignore[union-attr]
    first = int(time.min())
    last = int(time.max()) + 1
    train_end = first + 3 * (last - first) // 5
    validation_end = first + 4 * (last - first) // 5
    task = BinaryRiskTask("static", "node", target_mask_attr="target_node_mask")
    return (
        window_loader(data, first, train_end, batch_size=batch_size),
        window_loader(data, train_end, validation_end, batch_size=batch_size),
        window_loader(data, validation_end, last, batch_size=batch_size),
        task,
    )


def run(args: argparse.Namespace) -> None:
    """Build IBM inputs, run Experiment, and print frozen test-split metrics."""
    with TemporaryDirectory(prefix="amlgraphx-ibm-static-") as directory:
        transactions = load_transactions(Path(directory), args.limit)
        data = prepare_pyg_graph(
            transactions,
            node_type="transaction",
            temporal="static",
            edge_delta=timedelta(hours=4),
            features=GraphFeatureSpec(
                node_columns=("amount",),
                edge_columns=("time_delta",),
                label_column="label",
            ),
        )
        train_loader, validation_loader, test_loader, task = make_loaders(
            data, args.mode, args.batch_size
        )
        model = TransactionRiskMLP(data.x.shape[1])
        experiment = Experiment(
            model,
            task=task,
            metrics={
                "precision": Precision(),
                "recall": Recall(),
            },
            trainer_kwargs={"max_epochs": args.epochs, "accelerator": "auto"},
            evaluation_kwargs={"top_fractions": (0.01,)},
        )
        result = experiment.run(
            train_dataloaders=train_loader,
            validation_dataloaders=validation_loader,
            test_dataloaders=test_loader,
        )
    print("Lightning test metrics:", result.test_metrics)
    print("AML risk metrics:", result.risk_metrics)


def parse_args() -> argparse.Namespace:
    """Parse the intentionally small, inspectable example configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mode", choices=("full", "windows"), default="full")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
