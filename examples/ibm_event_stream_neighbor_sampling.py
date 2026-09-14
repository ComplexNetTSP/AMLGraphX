"""Run stateless strict-past event-neighborhood sampling on IBM HI-Small.

The example deliberately supplies a small researcher-defined model rather than
an AMLGraphX model. Every target transaction receives its own local graph of
strictly earlier account interactions. The model combines the target amount
with the mean message of that target's sampled historical edges.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import torch
from torch import Tensor, nn
from torch_geometric.data import Data, TemporalData

from amlgraphx.datasets import load_dataset
from amlgraphx.evaluation import Precision, Recall
from amlgraphx.experiments import BinaryRiskTask, Experiment
from amlgraphx.graph import GraphFeatureSpec, prepare_pyg_graph
from amlgraphx.sampling import causal_event_neighbor_loader


class LocalHistoryRiskModel(nn.Module):
    """Score one event from its own message and sampled strict-past context."""

    def __init__(self, message_dim: int) -> None:
        """Create a scorer over target and same-sample historical messages."""
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(message_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, batch: Data) -> Tensor:
        """Return one raw binary logit for every ``event_y`` target event."""
        history = _mean_history_message(batch)
        features = torch.cat((torch.log1p(batch.event_msg.abs()), history), dim=-1)
        return self.scorer(features).squeeze(-1)


def _mean_history_message(batch: Data) -> Tensor:
    """Pool context edges separately for every disjoint PyG target sample."""
    count = batch.event_y.numel()
    width = batch.event_msg.shape[1]
    result = batch.event_msg.new_zeros((count, width))
    if batch.num_edges == 0:
        return result
    sample_id = batch.batch[batch.edge_index[0]]
    result.index_add_(0, sample_id, torch.log1p(batch.edge_attr.abs()))
    denominator = torch.bincount(sample_id, minlength=count).clamp_min(1)
    return result / denominator[:, None]


def load_events(cache_dir: Path, limit: int) -> TemporalData:
    """Load a bounded chronological IBM account event stream."""
    dataset = load_dataset("ibm-aml", variant="hi-small", cache_dir=cache_dir)
    transactions = temporal_sample(dataset.transactions().sort("timestamp"), limit)
    return prepare_pyg_graph(
        transactions,
        node_type="account",
        temporal="event_stream",
        features=GraphFeatureSpec(edge_columns=("amount",), label_column="label"),
    )


def temporal_sample(transactions: pl.LazyFrame, limit: int) -> pl.DataFrame:
    """Keep six chronological bands so every split has representative events."""
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


def add_chronological_masks(events: TemporalData) -> None:
    """Attach strict 60/20/20 masks without splitting equal-time observations."""
    count = events.num_events
    train_end = _next_timestamp_boundary(events.t, 3 * count // 5)
    validation_end = _next_timestamp_boundary(events.t, 4 * count // 5)
    if not 0 < train_end < validation_end < count:
        raise ValueError(
            "event stream needs three non-empty timestamp-separated splits"
        )
    events.train_mask = _event_interval_mask(count, 0, train_end)
    events.validation_mask = _event_interval_mask(count, train_end, validation_end)
    events.test_mask = _event_interval_mask(count, validation_end, count)


def _next_timestamp_boundary(event_time: Tensor, index: int) -> int:
    """Move a nominal split end past its complete equal-timestamp group."""
    while (
        index < event_time.numel()
        and index > 0
        and event_time[index] == event_time[index - 1]
    ):
        index += 1
    return index


def _event_interval_mask(count: int, start: int, end: int) -> Tensor:
    """Return one boolean event selector for a half-open chronological interval."""
    mask = torch.zeros(count, dtype=torch.bool)
    mask[start:end] = True
    return mask


def run(args: argparse.Namespace) -> None:
    """Train and evaluate a stateless sampled-event model on frozen IBM targets."""
    with TemporaryDirectory(prefix="amlgraphx-ibm-event-neighbors-") as directory:
        events = load_events(Path(directory), args.limit)
        add_chronological_masks(events)
        loader_options = {
            "num_neighbors": [args.fanout, args.fanout],
            "batch_size": args.batch_size,
        }
        experiment = Experiment(
            LocalHistoryRiskModel(events.msg.shape[1]),
            task=BinaryRiskTask(
                "event_stream",
                "event",
                label_attr="event_y",
                target_mask_attr="target_event_mask",
            ),
            predictor_kwargs={"timestamp_attr": "event_time"},
            metrics={"precision": Precision(), "recall": Recall()},
            trainer_kwargs={"max_epochs": args.epochs, "accelerator": "auto"},
            evaluation_kwargs={"top_fractions": (0.01,)},
        )
        result = experiment.run(
            train_dataloaders=causal_event_neighbor_loader(
                events, target_mask_attr="train_mask", **loader_options
            ),
            validation_dataloaders=causal_event_neighbor_loader(
                events, target_mask_attr="validation_mask", **loader_options
            ),
            test_dataloaders=causal_event_neighbor_loader(
                events, target_mask_attr="test_mask", **loader_options
            ),
        )
    print("Lightning test metrics:", result.test_metrics)
    print("AML risk metrics:", result.risk_metrics)


def parse_args() -> argparse.Namespace:
    """Parse a compact real-data sampling workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=12_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fanout", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
