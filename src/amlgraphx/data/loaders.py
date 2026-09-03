"""Mini-batch loaders for AMLGraphX graph representations.

The loaders keep graph semantics explicit. Static graph windows use ordinary
PyG ``DataLoader`` batching, account event streams use PyG's
``TemporalDataLoader``, and snapshot windows batch each time position with
``Batch.from_data_list``. No model architecture is assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data, TemporalData
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.loader import TemporalDataLoader

_SNAPSHOT_METADATA = ("snapshot_index", "start_time", "end_time")


class StaticGraphWindowDataset(Dataset[Data]):
    """Split one time-aware static PyG graph into disjoint target windows.

    Transaction-node graphs need ``node_time`` and produce ``target_node_mask``.
    Account-node graphs need ``edge_time`` and produce ``target_edge_mask``.
    ``lookback`` retains earlier context in every subgraph; for a causal
    transaction graph it should equal the edge-construction ``delta``. Target
    intervals remain non-overlapping even though their context can overlap.

    The dataset builds one sparse local subgraph only when it is indexed. This
    avoids materializing all windows before a PyG ``DataLoader`` starts work.
    """

    def __init__(
        self,
        data: Data,
        *,
        window_size: timedelta,
        lookback: timedelta = timedelta(0),
        drop_last: bool = False,
    ) -> None:
        """Store one static graph and derive its non-empty target intervals."""
        if not isinstance(data, Data):
            raise TypeError("data must be a torch_geometric.data.Data object")
        _validate_duration(window_size, "window_size", allow_zero=False)
        _validate_duration(lookback, "lookback", allow_zero=True)
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a bool")

        target_kind, target_time = _target_time(data)
        self.data = data
        self.target_kind = target_kind
        self.window_ns = _timedelta_ns(window_size)
        self.lookback_ns = _timedelta_ns(lookback)
        self.window_starts = _nonempty_window_starts(
            target_time, self.window_ns, drop_last
        )

    def __len__(self) -> int:
        """Return the number of non-empty target windows."""
        return len(self.window_starts)

    def __getitem__(self, index: int) -> Data:
        """Return one local PyG subgraph and its node or edge target mask."""
        start = self.window_starts[index]
        end = start + self.window_ns
        if self.target_kind == "node":
            return _node_window(self.data, start, end, self.lookback_ns, index)
        return _edge_window(self.data, start, end, self.lookback_ns, index)


def static_graph_loader(
    data: Data,
    *,
    window_size: timedelta,
    lookback: timedelta = timedelta(0),
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    **kwargs: object,
) -> PyGDataLoader:
    """Create a PyG loader over bounded-memory static graph windows.

    The returned batches are ordinary PyG ``Batch`` objects. For transaction
    graphs pass ``lookback=edge_delta`` so every current transaction can retain
    the predecessor nodes allowed by the causal graph construction.
    """
    dataset = StaticGraphWindowDataset(
        data,
        window_size=window_size,
        lookback=lookback,
        drop_last=drop_last,
    )
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


@dataclass(frozen=True, slots=True)
class SnapshotWindow:
    """One ordered context sequence and its current target snapshot."""

    context: tuple[Data, ...]
    target: Data


@dataclass(frozen=True, slots=True)
class SnapshotBatch:
    """Parallel PyG batches for one batch of equally long snapshot windows.

    Each item in ``context`` and ``target`` is a normal PyG ``Batch``. At a
    fixed time position, PyG concatenates graphs into disconnected components
    and offsets their ``edge_index`` values. The tuple preserves temporal order.
    """

    context: tuple[Batch, ...]
    target: Batch

    def to(self, *args: object, **kwargs: object) -> SnapshotBatch:
        """Move every PyG batch to a device using PyG's standard ``to`` method."""
        context = tuple(snapshot.to(*args, **kwargs) for snapshot in self.context)
        return SnapshotBatch(context=context, target=self.target.to(*args, **kwargs))


class SnapshotWindowDataset(Dataset[SnapshotWindow]):
    """Create fixed-length context/target samples from account snapshots.

    ``context_size=5`` makes samples ``(G[t-5], ..., G[t-1]) -> G[t]``. The
    dataset never changes a snapshot's edges or labels. It validates optional
    ``snapshot_index`` metadata so an accidental time reversal is rejected at
    construction time.
    """

    def __init__(self, snapshots: Sequence[Data], *, context_size: int) -> None:
        """Store an ordered snapshot sequence and validate its public contract."""
        if isinstance(context_size, bool) or not isinstance(context_size, int):
            raise TypeError("context_size must be an integer")
        if context_size < 1:
            raise ValueError("context_size must be at least 1")
        self.snapshots = tuple(snapshots)
        if len(self.snapshots) <= context_size:
            raise ValueError("snapshots must contain at least one target snapshot")
        _validate_snapshot_sequence(self.snapshots)
        self.context_size = context_size

    def __len__(self) -> int:
        """Return the number of available current-target snapshots."""
        return len(self.snapshots) - self.context_size

    def __getitem__(self, index: int) -> SnapshotWindow:
        """Return one ordered history and the snapshot immediately after it."""
        end = index + self.context_size
        return SnapshotWindow(
            context=self.snapshots[index:end], target=self.snapshots[end]
        )


class SnapshotDataLoader(DataLoader[SnapshotBatch]):
    """Batch snapshot windows with PyG's disconnected-graph mechanism.

    Batch size is the number of temporal windows processed in parallel. The
    loader may shuffle independent windows only when a model keeps no state
    between calls; it never rearranges time positions inside a window.
    """

    def __init__(
        self,
        dataset: Dataset[SnapshotWindow] | Sequence[SnapshotWindow],
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs: object,
    ) -> None:
        """Create a PyTorch loader with the AMLGraphX snapshot collator."""
        if "collate_fn" in kwargs:
            raise TypeError("SnapshotDataLoader defines its own collate_fn")
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=snapshot_collate,
            **kwargs,
        )


def snapshot_collate(samples: Sequence[SnapshotWindow]) -> SnapshotBatch:
    """Batch every common time position with ``Batch.from_data_list``."""
    if not samples:
        raise ValueError("cannot collate an empty snapshot batch")
    context_size = len(samples[0].context)
    if any(len(sample.context) != context_size for sample in samples):
        raise ValueError("all snapshot windows must have the same context size")

    context = tuple(
        _batch_snapshots([sample.context[position] for sample in samples])
        for position in range(context_size)
    )
    target = _batch_snapshots([sample.target for sample in samples])
    return SnapshotBatch(context=context, target=target)


def event_stream_loader(
    data: TemporalData,
    *,
    batch_size: int,
    neg_sampling_ratio: float = 0.0,
    **kwargs: object,
) -> TemporalDataLoader:
    """Return PyG's chronological ``TemporalDataLoader`` for account events."""
    if not isinstance(data, TemporalData):
        raise TypeError("data must be a torch_geometric.data.TemporalData object")
    if data.t is None:
        raise ValueError("TemporalData must define t")
    if data.t.numel() > 1 and bool(torch.any(data.t[1:] < data.t[:-1])):
        raise ValueError("TemporalData.t must be sorted in non-decreasing order")
    return TemporalDataLoader(
        data,
        batch_size=batch_size,
        neg_sampling_ratio=neg_sampling_ratio,
        **kwargs,
    )


def _target_time(data: Data) -> tuple[str, Tensor]:
    """Choose transaction nodes first, otherwise account transaction edges."""
    node_time = getattr(data, "node_time", None)
    if isinstance(node_time, Tensor):
        _validate_time_tensor(node_time, "node_time", data.num_nodes)
        return "node", node_time
    edge_time = getattr(data, "edge_time", None)
    if isinstance(edge_time, Tensor):
        _validate_time_tensor(edge_time, "edge_time", data.num_edges)
        return "edge", edge_time
    raise ValueError("data needs node_time or edge_time for static window batching")


def _nonempty_window_starts(
    times: Tensor, window_ns: int, drop_last: bool
) -> list[int]:
    """Return disjoint intervals that contain at least one prediction target."""
    if times.numel() == 0:
        return []
    first = int(times.min())
    final = int(times.max()) + 1
    starts: list[int] = []
    start = first
    while start < final:
        end = start + window_ns
        if end > final and drop_last:
            break
        if bool(torch.any((times >= start) & (times < end))):
            starts.append(start)
        start = end
    return starts


def _node_window(data: Data, start: int, end: int, lookback: int, index: int) -> Data:
    """Select transaction nodes and their bounded causal context."""
    node_time = data.node_time
    included = (node_time >= start - lookback) & (node_time < end)
    window = data.subgraph(included)
    window.target_node_mask = (window.node_time >= start) & (window.node_time < end)
    window.window_id = torch.tensor(index, dtype=torch.long)
    return window


def _edge_window(data: Data, start: int, end: int, lookback: int, index: int) -> Data:
    """Select account transaction edges, then compact their active account nodes."""
    edge_time = data.edge_time
    included = (edge_time >= start - lookback) & (edge_time < end)
    edge_window = data.edge_subgraph(included)
    active_nodes = edge_window.edge_index.flatten().unique(sorted=True)
    window = edge_window.subgraph(active_nodes)
    window.target_edge_mask = (window.edge_time >= start) & (window.edge_time < end)
    window.window_id = torch.tensor(index, dtype=torch.long)
    return window


def _batch_snapshots(snapshots: Sequence[Data]) -> Batch:
    """Use PyG batching while dropping non-tensor time metadata safely."""
    return Batch.from_data_list(list(snapshots), exclude_keys=list(_SNAPSHOT_METADATA))


def _validate_snapshot_sequence(snapshots: Sequence[Data]) -> None:
    """Require PyG snapshots and verify optional strictly increasing indices."""
    previous: int | None = None
    for snapshot in snapshots:
        if not isinstance(snapshot, Data):
            raise TypeError("snapshots must contain torch_geometric.data.Data")
        value = getattr(snapshot, "snapshot_index", None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("snapshot_index must be an integer when provided")
        if previous is not None and value <= previous:
            raise ValueError("snapshot_index values must increase strictly")
        previous = value


def _validate_time_tensor(value: Tensor, name: str, expected_size: int | None) -> None:
    """Validate a one-dimensional integer timestamp tensor."""
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_size is not None and value.numel() != expected_size:
        raise ValueError(f"{name} must have one value per prediction entity")
    if value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must use integer nanoseconds")


def _validate_duration(value: timedelta, name: str, *, allow_zero: bool) -> None:
    """Validate a positive duration, optionally accepting zero lookback."""
    if not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a datetime.timedelta")
    if value < timedelta(0) or (value == timedelta(0) and not allow_zero):
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")


def _timedelta_ns(value: timedelta) -> int:
    """Convert a timedelta to exact integer nanoseconds without float rounding."""
    return (
        value.days * 86_400 + value.seconds
    ) * 1_000_000_000 + value.microseconds * 1_000


__all__ = [
    "SnapshotBatch",
    "SnapshotDataLoader",
    "SnapshotWindow",
    "SnapshotWindowDataset",
    "StaticGraphWindowDataset",
    "event_stream_loader",
    "snapshot_collate",
    "static_graph_loader",
]
