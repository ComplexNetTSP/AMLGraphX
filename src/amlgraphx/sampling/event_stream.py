"""Causal history delivery for continuous account event streams.

Event-stream models use history in two distinct ways. Stateful models update
their own memory after each observed timestamp group; use
``causal_event_stream_loader`` to receive those groups and optional warm-up
events. Stateless models rebuild a strict-past local graph for every target
event; use ``causal_event_neighbor_loader`` for native PyG link batches.

The module does not provide a temporal model. Researchers keep ownership of
their model state and consume the documented native PyG fields.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import Data, TemporalData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader


def causal_event_stream_loader(
    targets: TemporalData,
    *,
    history: TemporalData | None = None,
    target_mask_attr: str = "target_event_mask",
) -> DataLoader[TemporalData]:
    """Yield strict timestamp groups for a stateful event-stream model.

    Every yielded batch contains exactly one timestamp. Batches from
    ``history`` are marked false in ``target_mask_attr`` and are therefore
    warm-up observations: they update researcher-owned state but contribute no
    loss, metric, or held-out score. Batches from ``targets`` are marked true.
    This guarantees that equal-timestamp events are predicted before any of
    them can become context for a later timestamp.

    ``history`` must end strictly before ``targets`` starts. For validation use
    the training events as history; for test use training plus validation
    events. Pair the result with ``EventStreamBinaryPredictor`` and
    ``BinaryRiskTask(..., target_mask_attr="target_event_mask")``.

    Args:
        targets: Ordered events to predict, with ``src``, ``dst``, and ``t``.
        history: Optional ordered, history-only events preceding every target.
        target_mask_attr: Boolean event field written on every yielded batch.

    Returns:
        A regular PyTorch loader yielding native PyG ``TemporalData`` batches.
    """
    _validate_event_stream(targets, "targets")
    if not isinstance(target_mask_attr, str) or not target_mask_attr:
        raise ValueError("target_mask_attr must be a non-empty string")
    if history is not None:
        _validate_event_stream(history, "history", allow_empty=True)
        _validate_history_boundary(history, targets)
    return DataLoader(
        _TimestampEventBatches(history, targets, target_mask_attr), batch_size=None
    )


def causal_event_neighbor_loader(
    data: TemporalData,
    *,
    num_neighbors: Sequence[int],
    batch_size: int,
    target_mask_attr: str,
    label_attr: str = "y",
    message_attr: str = "msg",
    temporal_strategy: Literal["uniform", "last"] = "last",
    **kwargs: object,
) -> LinkNeighborLoader:
    """Create strict-past local graph batches for stateless event models.

    The complete event stream remains the sampling graph while
    ``target_mask_attr`` selects the events to predict. PyG receives one
    nanosecond before each target time as its inclusive cutoff, so neither the
    target event nor equal-time/future events can enter message-passing
    context. ``temporal_strategy="last"`` selects the newest eligible history
    at every hop.

    Batches are native PyG ``Data`` objects. Their sampled historical graph is
    stored in ``edge_index``, ``edge_time``, and optional ``edge_attr``. Target
    events are stored in ``edge_label_index`` and AMLGraphX event fields:
    ``event_y``, ``event_time``, optional ``event_msg``, ``event_id``, and an
    all-true ``target_event_mask``. A stateless researcher model must return
    one logit per ``event_y`` value. Use this with
    ``BinaryRiskTask("event_stream", "event", label_attr="event_y",
    target_mask_attr="target_event_mask")`` and pass
    ``predictor_kwargs={"timestamp_attr": "event_time"}`` to ``Experiment``.

    Args:
        data: Complete chronological account event stream.
        num_neighbors: Per-hop history fanout, for example ``[25, 10]``.
        batch_size: Number of target events per PyG sampled batch.
        target_mask_attr: Boolean event mask selecting one prediction split.
        label_attr: Event-aligned binary label field, normally ``"y"``.
        message_attr: Optional event feature field copied to ``event_msg`` and
            used as historical ``edge_attr``.
        temporal_strategy: PyG choice among eligible historical interactions.
        **kwargs: Additional ``LinkNeighborLoader`` options.

    Returns:
        A native PyG ``LinkNeighborLoader`` over strict-past event histories.
    """
    _validate_event_stream(data, "data")
    _validate_neighbor_arguments(num_neighbors, batch_size, temporal_strategy)
    _reject_loader_overrides(kwargs)
    event_time = _integer_time(data.t, "data.t")
    event_label = _event_tensor(data, label_attr, data.t.numel())
    target_mask = _event_mask(data, target_mask_attr, data.t.numel())
    event_message = _optional_event_tensor(data, message_attr, data.t.numel())
    graph = _event_graph(data, event_time, event_message)
    transform = _MarkEventTargets(event_time, event_label, event_message)
    return LinkNeighborLoader(
        graph,
        num_neighbors=list(num_neighbors),
        edge_label_index=graph.edge_index[:, target_mask],
        edge_label=event_label[target_mask],
        edge_label_time=_strictly_before(event_time[target_mask]),
        input_id=torch.arange(data.t.numel(), device=data.t.device)[target_mask],
        batch_size=batch_size,
        shuffle=False,
        replace=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr="edge_time",
        transform=transform,
        **kwargs,
    )


def recent_event_neighbors(
    data: TemporalData,
    *,
    size: int,
    device: torch.device | str | None = None,
) -> LastNeighborLoader:
    """Return PyG's mutable recent-neighbor index for a stateful model.

    Query it before a timestamp group is predicted and call ``insert(src, dst)``
    from the model's post-prediction ``update_state(batch)`` hook. Pair it with
    :func:`causal_event_stream_loader`; calling ``insert`` earlier would expose
    the current event to its own prediction. The returned object is PyG's
    ``LastNeighborLoader``, not an AMLGraphX wrapper.
    """
    _validate_event_stream(data, "data")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    if data.num_nodes is None or int(data.num_nodes) < 1:
        raise ValueError("data must define at least one account node")
    return LastNeighborLoader(int(data.num_nodes), size, device=device)


class _TimestampEventBatches(IterableDataset[TemporalData]):
    """Expose history and targets as immutable equal-timestamp PyG batches."""

    def __init__(
        self,
        history: TemporalData | None,
        targets: TemporalData,
        target_mask_attr: str,
    ) -> None:
        """Store already validated event intervals without copying them."""
        self.history = history
        self.targets = targets
        self.target_mask_attr = target_mask_attr

    def __iter__(self) -> Iterator[TemporalData]:
        """Yield history first, then target timestamp groups."""
        if self.history is not None:
            yield from _timestamp_batches(self.history, False, self.target_mask_attr)
        yield from _timestamp_batches(self.targets, True, self.target_mask_attr)

    def __len__(self) -> int:
        """Return the number of timestamp groups consumed in one sequence."""
        history_count = _timestamp_group_count(self.history)
        return history_count + _timestamp_group_count(self.targets)


@dataclass(frozen=True, slots=True)
class _MarkEventTargets:
    """Attach target-event fields after PyG has sampled one local graph."""

    event_time: Tensor
    event_label: Tensor
    event_message: Tensor | None

    def __call__(self, batch: Data) -> Data:
        """Map PyG target input IDs back to original event-aligned fields."""
        event_id = batch.input_id
        batch.event_id = event_id
        batch.event_y = self.event_label[event_id]
        batch.event_time = self.event_time[event_id]
        if self.event_message is not None:
            batch.event_msg = self.event_message[event_id]
        batch.target_event_mask = torch.ones(
            batch.event_y.numel(), dtype=torch.bool, device=batch.event_y.device
        )
        return batch


def _timestamp_batches(
    data: TemporalData, is_target: bool, target_mask_attr: str
) -> Iterator[TemporalData]:
    """Slice an ordered stream into whole timestamp groups without mutation."""
    _, counts = torch.unique_consecutive(data.t, return_counts=True)
    start = 0
    for count in counts.tolist():
        end = start + count
        index = torch.arange(start, end, device=data.t.device)
        batch = data.index_select(index)
        setattr(
            batch,
            target_mask_attr,
            torch.full((count,), is_target, dtype=torch.bool, device=data.t.device),
        )
        yield batch
        start = end


def _timestamp_group_count(data: TemporalData | None) -> int:
    """Count distinct consecutive timestamps, treating absent history as empty."""
    if data is None:
        return 0
    return int(torch.unique_consecutive(data.t).numel())


def _event_graph(
    data: TemporalData, event_time: Tensor, event_message: Tensor | None
) -> Data:
    """Represent account events as time-stamped edges for PyG sampling."""
    graph = Data(
        num_nodes=int(data.num_nodes),
        edge_index=torch.stack((data.src, data.dst)),
        edge_time=event_time,
    )
    node_features = getattr(data, "x", None)
    if isinstance(node_features, Tensor):
        graph.x = node_features
    if event_message is not None:
        graph.edge_attr = event_message
    return graph


def _validate_event_stream(
    data: TemporalData, name: str, *, allow_empty: bool = False
) -> None:
    """Require a chronological PyG event stream with aligned endpoint fields."""
    if not isinstance(data, TemporalData):
        raise TypeError(f"{name} must be a torch_geometric.data.TemporalData object")
    for attr in ("src", "dst", "t"):
        value = getattr(data, attr, None)
        if not isinstance(value, Tensor) or value.ndim != 1:
            raise ValueError(f"{name}.{attr} must be a one-dimensional torch.Tensor")
    count = data.t.numel()
    if not allow_empty and count == 0:
        raise ValueError(f"{name} must contain at least one event")
    if data.src.numel() != count or data.dst.numel() != count:
        raise ValueError(f"{name}.src, {name}.dst, and {name}.t must be aligned")
    if count > 1 and bool(torch.any(data.t[1:] < data.t[:-1])):
        raise ValueError(f"{name}.t must be sorted in non-decreasing order")


def _validate_history_boundary(history: TemporalData, targets: TemporalData) -> None:
    """Forbid equal-time split context under strict timestamp causality."""
    if history.t.numel() and history.t[-1] >= targets.t[0]:
        raise ValueError("history must end strictly before the first target timestamp")


def _event_mask(data: TemporalData, attr: str, count: int) -> Tensor:
    """Read one non-empty boolean event selector from a complete stream."""
    value = getattr(data, attr, None)
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{attr} must be a boolean torch.Tensor")
    if value.ndim != 1 or value.numel() != count:
        raise ValueError(f"{attr} must have one value per event")
    if not bool(value.any()):
        raise ValueError(f"{attr} must select at least one target event")
    return value


def _event_tensor(data: TemporalData, attr: str, count: int) -> Tensor:
    """Read a required event-aligned tensor such as a binary label."""
    value = getattr(data, attr, None)
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() != count:
        raise ValueError(f"{attr} must be a one-dimensional tensor per event")
    return value


def _optional_event_tensor(data: TemporalData, attr: str, count: int) -> Tensor | None:
    """Read an optional event feature tensor while preserving its feature width."""
    value = getattr(data, attr, None)
    if value is None:
        return None
    if not isinstance(value, Tensor) or value.ndim < 1 or value.shape[0] != count:
        raise ValueError(f"{attr} must have one leading row per event")
    return value


def _integer_time(value: Tensor, name: str) -> Tensor:
    """Require integer nanoseconds so one tick can express strict past."""
    if value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must use integer nanoseconds for causal sampling")
    return value


def _strictly_before(times: Tensor) -> Tensor:
    """Turn PyG's inclusive temporal cutoff into an exact strict-past cutoff."""
    minimum = torch.iinfo(times.dtype).min
    if bool(torch.any(times == minimum)):
        raise ValueError("timestamps cannot use the minimum integer value")
    return times - 1


def _validate_neighbor_arguments(
    num_neighbors: Sequence[int], batch_size: int, temporal_strategy: str
) -> None:
    """Reject malformed PyG temporal-neighborhood settings early."""
    if isinstance(num_neighbors, str) or not isinstance(num_neighbors, Sequence):
        raise TypeError("num_neighbors must be a sequence of integers")
    if not num_neighbors or any(
        isinstance(value, bool) or not isinstance(value, int) or value < -1
        for value in num_neighbors
    ):
        raise ValueError(
            "num_neighbors must contain integers greater than or equal to -1"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if temporal_strategy not in {"uniform", "last"}:
        raise ValueError("temporal_strategy must be 'uniform' or 'last'")


def _reject_loader_overrides(kwargs: dict[str, object]) -> None:
    """Keep causal target and temporal filtering controls owned by AMLGraphX."""
    protected = {
        "batch_size",
        "disjoint",
        "edge_label",
        "edge_label_index",
        "edge_label_time",
        "input_id",
        "replace",
        "shuffle",
        "temporal_strategy",
        "time_attr",
        "transform",
    }
    overlap = protected.intersection(kwargs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise TypeError(f"sampling options are controlled by AMLGraphX: {names}")


__all__ = [
    "causal_event_neighbor_loader",
    "causal_event_stream_loader",
    "recent_event_neighbors",
]
