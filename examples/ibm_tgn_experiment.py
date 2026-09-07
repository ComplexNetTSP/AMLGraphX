"""Run a lightweight TGN-style AML event experiment on IBM HI-Small.

This is a compact, educational TGN adaptation rather than a paper-faithful
benchmark. It illustrates a memory, time encoding, message aggregation, and
post-prediction memory updates through the same AMLGraphX event Experiment API
used by a researcher-defined temporal model.
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


class TGNStyleRiskModel(nn.Module):
    """Small memory-and-message account model inspired by a temporal graph network."""

    def __init__(
        self, num_accounts: int, message_dim: int, memory_dim: int = 32
    ) -> None:
        """Create stateful account memory, time encoder, message updater, and scorer."""
        super().__init__()
        self.register_buffer("memory", torch.zeros(num_accounts, memory_dim))
        self.register_buffer("last_time", torch.zeros(num_accounts))
        self.time_encoder = nn.Sequential(nn.Linear(1, memory_dim), nn.Tanh())
        self.message_function = nn.Linear(memory_dim * 2 + message_dim, memory_dim)
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)
        self.scorer = nn.Linear(memory_dim * 2 + message_dim, 1)

    def forward(self, batch: TemporalData) -> Tensor:
        """Score events from current memory and elapsed-time encodings."""
        source_memory = self.memory[batch.src]
        destination_memory = self.memory[batch.dst]
        current_time = batch.t.to(dtype=torch.float32) / 1e9
        elapsed_hours = ((current_time - self.last_time[batch.src]) / 3600).clamp(0, 24)
        temporal_source = source_memory + self.time_encoder(elapsed_hours[:, None])
        message = torch.log1p(batch.msg.abs())
        return self.scorer(
            torch.cat((temporal_source, destination_memory, message), -1)
        ).squeeze(-1)

    def update_state(self, batch: TemporalData) -> None:
        """Aggregate transaction messages into the two endpoint memory states."""
        with torch.no_grad():
            source = self.memory[batch.src]
            destination = self.memory[batch.dst]
            message = torch.log1p(batch.msg.abs())
            source_message = self.message_function(
                torch.cat((source, destination, message), -1)
            )
            destination_message = self.message_function(
                torch.cat((destination, source, message), -1)
            )
            self.memory.index_copy_(
                0, batch.src, self.memory_updater(source_message, source)
            )
            self.memory.index_copy_(
                0, batch.dst, self.memory_updater(destination_message, destination)
            )
            event_time = batch.t.to(dtype=torch.float32) / 1e9
            self.last_time.index_copy_(0, batch.src, event_time)
            self.last_time.index_copy_(0, batch.dst, event_time)

    def reset_state(self) -> None:
        """Reset memory at each explicitly independent train/validation/test run."""
        self.memory.zero_()
        self.last_time.zero_()


def load_events(cache_dir: Path, limit: int) -> TemporalData:
    """Load canonical IBM transactions and convert them into account event data."""
    dataset = load_dataset("ibm-aml", variant="hi-small", cache_dir=cache_dir)
    transactions = temporal_sample(dataset.transactions().sort("timestamp"), limit)
    return prepare_pyg_graph(
        transactions,
        node_type="account",
        temporal="event_stream",
        features=GraphFeatureSpec(edge_columns=("amount",), label_column="label"),
    )


def temporal_sample(transactions: pl.LazyFrame, limit: int) -> pl.DataFrame:
    """Draw an equal bounded sample from six chronological IBM time bands."""
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
    """Keep chronological events while retaining account indices used by memory."""
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
    """Make chronological 60/20/20 event splits without shuffling interactions."""
    train_end = 3 * events.num_events // 5
    validation_end = 4 * events.num_events // 5
    return (
        slice_events(events, 0, train_end),
        slice_events(events, train_end, validation_end),
        slice_events(events, validation_end, events.num_events),
    )


def run(args: argparse.Namespace) -> None:
    """Train the lightweight temporal model and report test risk scores."""
    with TemporaryDirectory(prefix="amlgraphx-ibm-tgn-") as directory:
        events = load_events(Path(directory), args.limit)
        train, validation, test = split_events(events)
        model = TGNStyleRiskModel(events.num_nodes, events.msg.shape[1])
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
    """Parse a compact default workload and optional training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
