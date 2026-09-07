"""Run a lightweight JODIE-style AML event experiment on IBM HI-Small.

The example is intentionally an educational adaptation, not a claim of a
paper-faithful JODIE reproduction. It keeps the central JODIE idea: account
embeddings are projected through elapsed time and updated after every observed
transaction. AMLGraphX owns data loading, event construction, chronological
batching, model-contract validation, training, and risk-score metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import torch
from torch import Tensor, nn
from torch_geometric.data import TemporalData

from amlgraphx.data import event_stream_loader
from amlgraphx.datasets import load_dataset
from amlgraphx.evaluation import Precision, Recall
from amlgraphx.experiments import BinaryRiskTask, Experiment
from amlgraphx.graph import GraphFeatureSpec, prepare_pyg_graph


class JODIEStyleRiskModel(nn.Module):
    """Small dynamic account-memory model inspired by JODIE's projection/update path."""

    def __init__(
        self, num_accounts: int, message_dim: int, embedding_dim: int = 32
    ) -> None:
        """Create account memory, a time projection, and a transaction scorer."""
        super().__init__()
        self.register_buffer("memory", torch.zeros(num_accounts, embedding_dim))
        self.register_buffer("last_time", torch.zeros(num_accounts))
        self.time_projection = nn.Linear(1, embedding_dim)
        self.update = nn.GRUCell(embedding_dim * 2 + message_dim, embedding_dim)
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2 + message_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, batch: TemporalData) -> Tensor:
        """Score each event before its account memories are updated."""
        source_memory = self.memory[batch.src]
        destination_memory = self.memory[batch.dst]
        current_time = batch.t.to(dtype=torch.float32) / 1e9
        elapsed_hours = ((current_time - self.last_time[batch.src]) / 3600).clamp(0, 24)
        projected_source = source_memory + self.time_projection(elapsed_hours[:, None])
        message = torch.log1p(batch.msg.abs())
        features = torch.cat((projected_source, destination_memory, message), dim=-1)
        return self.scorer(features).squeeze(-1)

    def update_state(self, batch: TemporalData) -> None:
        """Apply the post-prediction JODIE-style update required by AMLGraphX."""
        with torch.no_grad():
            source = self.memory[batch.src]
            destination = self.memory[batch.dst]
            message = torch.log1p(batch.msg.abs())
            source_input = torch.cat((source, destination, message), dim=-1)
            destination_input = torch.cat((destination, source, message), dim=-1)
            self.memory.index_copy_(0, batch.src, self.update(source_input, source))
            self.memory.index_copy_(
                0, batch.dst, self.update(destination_input, destination)
            )
            event_time = batch.t.to(dtype=torch.float32) / 1e9
            self.last_time.index_copy_(0, batch.src, event_time)
            self.last_time.index_copy_(0, batch.dst, event_time)

    def reset_state(self) -> None:
        """Start each AMLGraphX split sequence with an explicit empty history."""
        self.memory.zero_()
        self.last_time.zero_()


def load_events(cache_dir: Path, limit: int) -> TemporalData:
    """Download IBM, retain a bounded chronological sample, and build account events."""
    dataset = load_dataset("ibm-aml", variant="hi-small", cache_dir=cache_dir)
    transactions = temporal_sample(dataset.transactions().sort("timestamp"), limit)
    return prepare_pyg_graph(
        transactions,
        node_type="account",
        temporal="event_stream",
        features=GraphFeatureSpec(edge_columns=("amount",), label_column="label"),
    )


def temporal_sample(transactions: pl.LazyFrame, limit: int) -> pl.DataFrame:
    """Keep six time bands so chronological train/validation/test all have events."""
    minimum, maximum = (
        transactions.select(
            pl.col("timestamp").min().alias("minimum"),
            pl.col("timestamp").max().alias("maximum"),
        )
        .collect()
        .row(0)
    )
    pieces: list[pl.DataFrame] = []
    for index in range(6):
        start = minimum + (maximum - minimum) * index / 6
        end = minimum + (maximum - minimum) * (index + 1) / 6
        pieces.append(
            transactions.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
            )
            .limit(max(1, limit // 6))
            .collect()
        )
    return pl.concat(pieces, how="vertical_relaxed").sort("timestamp")


def slice_events(events: TemporalData, start: int, end: int) -> TemporalData:
    """Slice an ordered event interval while preserving the full account namespace."""
    result = TemporalData(
        src=events.src[start:end],
        dst=events.dst[start:end],
        t=events.t[start:end],
        msg=events.msg[start:end],
        y=events.y[start:end],
    )
    if getattr(events, "x", None) is not None:
        result.x = events.x
    return result


def split_events(
    events: TemporalData,
) -> tuple[TemporalData, TemporalData, TemporalData]:
    """Use chronological 60/20/20 event intervals for train/validation/test."""
    count = events.num_events
    train_end = 3 * count // 5
    validation_end = 4 * count // 5
    return (
        slice_events(events, 0, train_end),
        slice_events(events, train_end, validation_end),
        slice_events(events, validation_end, count),
    )


def run(args: argparse.Namespace) -> None:
    """Create event loaders, run Experiment, and print held-out AML metrics."""
    with TemporaryDirectory(prefix="amlgraphx-ibm-jodie-") as directory:
        events = load_events(Path(directory), args.limit)
        train, validation, test = split_events(events)
        model = JODIEStyleRiskModel(events.num_nodes, events.msg.shape[1])
        experiment = Experiment(
            model,
            task=BinaryRiskTask("event_stream", "event"),
            metrics={"precision": Precision(), "recall": Recall()},
            trainer_kwargs={"max_epochs": args.epochs, "accelerator": "auto"},
            evaluation_kwargs={"top_fractions": (0.01,)},
        )
        result = experiment.run(
            train_dataloaders=event_stream_loader(train, batch_size=args.batch_size),
            validation_dataloaders=event_stream_loader(
                validation, batch_size=args.batch_size
            ),
            test_dataloaders=event_stream_loader(test, batch_size=args.batch_size),
        )
    print("Lightning test metrics:", result.test_metrics)
    print("AML risk metrics:", result.risk_metrics)


def parse_args() -> argparse.Namespace:
    """Parse a small default workload suitable for a laptop smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
