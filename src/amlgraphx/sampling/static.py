"""Strictly causal neighborhood loaders for time-aware static graphs.

The graph builders own graph meaning and ``amlgraphx.split`` owns which
entities belong to each research split. This module only limits model input:
it samples a target's local graph neighborhood without allowing future or
same-timestamp observations into its history.

Both public functions return ordinary PyG ``Data`` batches. Node batches carry
``target_node_mask`` and edge batches carry ``edge_label`` plus
``target_edge_mask``. They therefore work with ``BinaryRiskTask`` and
``Experiment`` without a sampler-specific model contract.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import timedelta
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from amlgraphx.data import StaticGraphWindowDataset

_TargetKind = Literal["node", "edge"]


def causal_static_node_loader(
    data: Data,
    *,
    num_neighbors: Sequence[int],
    batch_size: int,
    target_mask_attr: str | None = None,
    window_size: timedelta | None = None,
    lookback: timedelta = timedelta(0),
    shuffle: bool = False,
    temporal_strategy: Literal["uniform", "last"] = "last",
    **kwargs: object,
) -> NeighborLoader | DataLoader[Data]:
    """Create strict-past node-neighborhood batches for transaction graphs.

    ``data`` must expose ``node_time`` in integer nanoseconds. A seed transaction
    at time ``t`` can receive its own features, but every sampled context node
    must have time strictly earlier than ``t``. Equal timestamps are treated as
    one prediction group and are excluded from one another's context.

    With ``window_size=None``, ``target_mask_attr`` selects seed nodes from the
    physical full graph. With a window size, target windows are produced by
    :class:`amlgraphx.data.StaticGraphWindowDataset`; an optional named split
    mask is intersected with every window's ``target_node_mask``. In both cases
    the output is an ordinary PyG batch whose first ``batch_size`` nodes are
    marked by ``target_node_mask``.

    Args:
        data: Transaction-as-node PyG graph with ``node_time`` and usually
            ``node_y``.
        num_neighbors: Per-hop PyG fanout, such as ``[25, 10]``. ``-1`` keeps
            all eligible neighbors at that hop.
        batch_size: Number of target transactions per sampled batch.
        target_mask_attr: Boolean graph attribute selecting a split, such as
            ``"train_mask"``. Required for full-graph training and optional
            for windowed training, where all window targets are selected by
            default.
        window_size: Optional outer prediction-window duration.
        lookback: Context duration retained in each outer window.
        shuffle: Shuffle seed targets within each loader. Keep ``False`` for
            validation and test reproducibility.
        temporal_strategy: PyG neighbor choice among eligible history nodes.
        **kwargs: Additional PyG ``NeighborLoader`` options.

    Returns:
        A PyG ``NeighborLoader`` for full graphs, or a standard PyTorch
        ``DataLoader`` yielding the same PyG batches for sliding windows.
    """
    _validate_common_arguments(data, num_neighbors, batch_size, temporal_strategy)
    _reject_loader_overrides(kwargs)
    if window_size is None:
        mask = _full_target_mask(data, target_mask_attr, "node")
        return _node_neighbor_loader(
            data,
            mask,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            temporal_strategy=temporal_strategy,
            loader_kwargs=kwargs,
        )
    return _windowed_loader(
        data,
        target_kind="node",
        target_mask_attr=target_mask_attr,
        label_attr=None,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        window_size=window_size,
        lookback=lookback,
        shuffle=shuffle,
        temporal_strategy=temporal_strategy,
        loader_kwargs=kwargs,
    )


def causal_static_edge_loader(
    data: Data,
    *,
    num_neighbors: Sequence[int],
    batch_size: int,
    target_mask_attr: str | None = None,
    label_attr: str = "edge_y",
    window_size: timedelta | None = None,
    lookback: timedelta = timedelta(0),
    shuffle: bool = False,
    temporal_strategy: Literal["uniform", "last"] = "last",
    **kwargs: object,
) -> LinkNeighborLoader | DataLoader[Data]:
    """Create strict-past edge-neighborhood batches for account graphs.

    Account-node graphs keep transactions as graph edges. This loader puts the
    selected transaction labels in PyG's ``edge_label`` field and retains the
    target transaction endpoints in ``edge_label_index``. The target edge is
    queried for prediction but is excluded from message-passing context because
    sampling uses one nanosecond before its ``edge_time`` as the cutoff.

    For :class:`amlgraphx.experiments.BinaryRiskTask`, use
    ``label_attr="edge_label"`` and ``target_mask_attr="target_edge_mask"``.
    The returned batch also exposes ``target_edge_time`` with the original
    transaction timestamp; PyG's ``edge_label_time`` remains the strict-past
    sampling cutoff.

    Args:
        data: Account-as-node PyG graph with ``edge_time`` and edge labels.
        num_neighbors: Per-hop PyG fanout, such as ``[25, 10]``.
        batch_size: Number of target transactions per sampled batch.
        target_mask_attr: Boolean edge mask selecting a split. Required for
            full-graph training and optional for windowed training.
        label_attr: Edge attribute containing binary transaction labels.
        window_size: Optional outer prediction-window duration.
        lookback: Context duration retained in each outer window.
        shuffle: Shuffle target transactions within each loader.
        temporal_strategy: PyG neighbor choice among eligible history edges.
        **kwargs: Additional PyG ``LinkNeighborLoader`` options.

    Returns:
        A PyG ``LinkNeighborLoader`` for full graphs, or a standard PyTorch
        ``DataLoader`` yielding the same PyG batches for sliding windows.
    """
    _validate_common_arguments(data, num_neighbors, batch_size, temporal_strategy)
    _reject_loader_overrides(kwargs)
    if window_size is None:
        mask = _full_target_mask(data, target_mask_attr, "edge")
        return _edge_neighbor_loader(
            data,
            mask,
            label_attr=label_attr,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            temporal_strategy=temporal_strategy,
            loader_kwargs=kwargs,
        )
    return _windowed_loader(
        data,
        target_kind="edge",
        target_mask_attr=target_mask_attr,
        label_attr=label_attr,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        window_size=window_size,
        lookback=lookback,
        shuffle=shuffle,
        temporal_strategy=temporal_strategy,
        loader_kwargs=kwargs,
    )


class _WindowedStaticSamples(IterableDataset[Data]):
    """Yield sampled PyG batches from chronological outer target windows."""

    def __init__(
        self,
        data: Data,
        *,
        target_kind: _TargetKind,
        target_mask_attr: str | None,
        label_attr: str | None,
        num_neighbors: Sequence[int],
        batch_size: int,
        window_size: timedelta,
        lookback: timedelta,
        shuffle: bool,
        temporal_strategy: Literal["uniform", "last"],
        loader_kwargs: dict[str, object],
    ) -> None:
        """Store the outer windows and their common PyG sampling settings."""
        self.windows = StaticGraphWindowDataset(
            data,
            window_size=window_size,
            lookback=lookback,
        )
        self.target_kind = target_kind
        self.target_mask_attr = target_mask_attr
        self.label_attr = label_attr
        self.num_neighbors = tuple(num_neighbors)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.temporal_strategy = temporal_strategy
        self.loader_kwargs = loader_kwargs

    def __iter__(self) -> Iterator[Data]:
        """Sample each chronological target window without materializing all batches."""
        for window in self.windows:
            target_mask = _window_target_mask(
                window, self.target_kind, self.target_mask_attr
            )
            if not bool(target_mask.any()):
                continue
            yield from self._sample_window(window, target_mask)

    def _sample_window(self, window: Data, target_mask: Tensor) -> Iterator[Data]:
        """Return sampled batches for one already bounded local graph."""
        if self.target_kind == "node":
            return iter(
                _node_neighbor_loader(
                    window,
                    target_mask,
                    num_neighbors=self.num_neighbors,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    temporal_strategy=self.temporal_strategy,
                    loader_kwargs=self.loader_kwargs,
                )
            )
        return iter(
            _edge_neighbor_loader(
                window,
                target_mask,
                label_attr=self.label_attr or "edge_y",
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                temporal_strategy=self.temporal_strategy,
                loader_kwargs=self.loader_kwargs,
            )
        )


def _windowed_loader(
    data: Data,
    *,
    target_kind: _TargetKind,
    target_mask_attr: str | None,
    label_attr: str | None,
    num_neighbors: Sequence[int],
    batch_size: int,
    window_size: timedelta,
    lookback: timedelta,
    shuffle: bool,
    temporal_strategy: Literal["uniform", "last"],
    loader_kwargs: dict[str, object],
) -> DataLoader[Data]:
    """Wrap lazy window sampling in a regular DataLoader accepted by Lightning."""
    dataset = _WindowedStaticSamples(
        data,
        target_kind=target_kind,
        target_mask_attr=target_mask_attr,
        label_attr=label_attr,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        window_size=window_size,
        lookback=lookback,
        shuffle=shuffle,
        temporal_strategy=temporal_strategy,
        loader_kwargs=loader_kwargs,
    )
    return DataLoader(dataset, batch_size=None)


def _node_neighbor_loader(
    data: Data,
    target_mask: Tensor,
    *,
    num_neighbors: Sequence[int],
    batch_size: int,
    shuffle: bool,
    temporal_strategy: Literal["uniform", "last"],
    loader_kwargs: dict[str, object],
) -> NeighborLoader:
    """Build one temporal PyG node loader with a strict-past seed cutoff."""
    node_time = _time_tensor(data, "node_time", data.num_nodes)
    seed_time = _strictly_before(node_time[target_mask])
    return NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        input_nodes=target_mask,
        input_time=seed_time,
        batch_size=batch_size,
        shuffle=shuffle,
        replace=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr="node_time",
        transform=_mark_node_targets,
        **loader_kwargs,
    )


def _edge_neighbor_loader(
    data: Data,
    target_mask: Tensor,
    *,
    label_attr: str,
    num_neighbors: Sequence[int],
    batch_size: int,
    shuffle: bool,
    temporal_strategy: Literal["uniform", "last"],
    loader_kwargs: dict[str, object],
) -> LinkNeighborLoader:
    """Build one temporal PyG link loader with target edges outside context."""
    edge_time = _time_tensor(data, "edge_time", data.num_edges)
    edge_label = _edge_labels(data, label_attr)
    return LinkNeighborLoader(
        data,
        num_neighbors=list(num_neighbors),
        edge_label_index=data.edge_index[:, target_mask],
        edge_label=edge_label[target_mask],
        edge_label_time=_strictly_before(edge_time[target_mask]),
        batch_size=batch_size,
        shuffle=shuffle,
        replace=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr="edge_time",
        transform=_mark_edge_targets,
        **loader_kwargs,
    )


def _mark_node_targets(batch: Data) -> Data:
    """Mark PyG's leading seed nodes as the only prediction targets."""
    batch.target_node_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
    batch.target_node_mask[: batch.batch_size] = True
    return batch


def _mark_edge_targets(batch: Data) -> Data:
    """Mark all PyG link labels and preserve their original transaction times."""
    batch.target_edge_mask = torch.ones(
        batch.edge_label.numel(), dtype=torch.bool, device=batch.edge_label.device
    )
    batch.target_edge_time = batch.edge_label_time + 1
    return batch


def _full_target_mask(data: Data, attr: str | None, target_kind: _TargetKind) -> Tensor:
    """Read a required split mask from a complete graph."""
    if attr is None:
        raise ValueError("target_mask_attr is required when window_size is None")
    return _boolean_mask(data, attr, _target_count(data, target_kind))


def _window_target_mask(
    window: Data, target_kind: _TargetKind, attr: str | None
) -> Tensor:
    """Intersect one window's targets with an optional split mask."""
    base_attr = "target_node_mask" if target_kind == "node" else "target_edge_mask"
    base = _boolean_mask(window, base_attr, _target_count(window, target_kind))
    if attr is None or attr == base_attr:
        return base
    return base & _boolean_mask(window, attr, base.numel(), allow_empty=True)


def _boolean_mask(
    data: Data, attr: str, expected_size: int, *, allow_empty: bool = False
) -> Tensor:
    """Read a non-empty boolean target selector with a clear public error."""
    value = getattr(data, attr, None)
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{attr} must be a boolean torch.Tensor")
    if value.ndim != 1 or value.numel() != expected_size:
        raise ValueError(f"{attr} must have one value per prediction target")
    if not allow_empty and not bool(value.any()):
        raise ValueError(f"{attr} must select at least one prediction target")
    return value


def _edge_labels(data: Data, label_attr: str) -> Tensor:
    """Validate that selected account transactions have one edge-aligned label."""
    value = getattr(data, label_attr, None)
    if not isinstance(value, Tensor):
        raise TypeError(f"{label_attr} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() != data.num_edges:
        raise ValueError(f"{label_attr} must have one value per graph edge")
    return value


def _time_tensor(data: Data, attr: str, expected_size: int) -> Tensor:
    """Read graph timestamps stored as one integer nanosecond value per entity."""
    value = getattr(data, attr, None)
    if not isinstance(value, Tensor):
        raise TypeError(f"data must define {attr} for causal sampling")
    if value.ndim != 1 or value.numel() != expected_size:
        raise ValueError(f"{attr} must have one value per prediction entity")
    if value.is_floating_point() or value.is_complex():
        raise TypeError(f"{attr} must use integer nanoseconds")
    return value


def _strictly_before(times: Tensor) -> Tensor:
    """Turn PyG's inclusive time filter into a strict-past cutoff."""
    minimum = torch.iinfo(times.dtype).min
    if bool(torch.any(times == minimum)):
        raise ValueError("timestamps cannot use the minimum integer value")
    return times - 1


def _target_count(data: Data, target_kind: _TargetKind) -> int:
    """Return node or transaction-edge count for a homogeneous PyG graph."""
    return int(data.num_nodes) if target_kind == "node" else int(data.num_edges)


def _validate_common_arguments(
    data: Data,
    num_neighbors: Sequence[int],
    batch_size: int,
    temporal_strategy: str,
) -> None:
    """Reject malformed sampler settings before PyG constructs an index."""
    if not isinstance(data, Data):
        raise TypeError("data must be a torch_geometric.data.Data object")
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
    """Keep public causal invariants from being silently overridden in kwargs."""
    protected = {
        "batch_size",
        "disjoint",
        "edge_label",
        "edge_label_index",
        "edge_label_time",
        "input_nodes",
        "input_time",
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


__all__ = ["causal_static_edge_loader", "causal_static_node_loader"]
