"""Temporal train/validation/test protocol anchors.

Splitting is deliberately separate from graph construction. A split describes
time boundaries; a graph builder decides how nodes and edges are represented.
Historical context and prediction targets can therefore be added here without
putting evaluation policy into a dataset adapter or graph builder.

中文：
    时间切分与图构建分离。split 只描述时间边界，builder 决定节点和边的语义。
    未来可以在这里加入历史 context 与 prediction target，而不把评估协议塞进
    dataset adapter 或 graph builder。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from amlgraphx.data.datamodule import TransactionGraphSplit
    from amlgraphx.graph.graphs import TransactionGraph


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Describe chronological train, validation, and test boundaries.

    The intervals are ``[-inf, train_end)``,
    ``[train_end, validation_end)``, and ``[validation_end, +inf)``. The
    object stores protocol configuration only; it does not load data or build
    graphs until :func:`apply_temporal_split` or
    :func:`build_temporal_node_masks` is called.

    中文：
        三个区间分别是 ``[-inf, train_end)``、
        ``[train_end, validation_end)`` 和 ``[validation_end, +inf)``。该对象
        只保存协议配置，调用 ``apply_temporal_split`` 或
        ``build_temporal_node_masks`` 后才作用于图。
    """

    train_end: datetime
    validation_end: datetime

    def __post_init__(self) -> None:
        """Validate ordered Python datetime cutoffs."""
        if not isinstance(self.train_end, datetime):
            raise TypeError("train_end must be a datetime")
        if not isinstance(self.validation_end, datetime):
            raise TypeError("validation_end must be a datetime")
        if self.train_end >= self.validation_end:
            raise ValueError("train_end must be earlier than validation_end")


@dataclass(frozen=True, slots=True)
class TemporalNodeMasks:
    """Hold chronological masks for one complete transaction graph.

    The masks partition the nodes of one graph into train, validation, and
    test intervals. They do not remove nodes or edges from the graph, so this
    is the transductive ``full graph + chronological masks`` protocol.

    中文：
        这三个 mask 把一张完整交易图的节点按时间划分为 train、validation
        和 test。它们不会删除节点或边，因此对应 transductive 的“完整图 +
        时间 mask”协议。
    """

    train_mask: Tensor
    validation_mask: Tensor
    test_mask: Tensor

    def __post_init__(self) -> None:
        """Validate one-dimensional, boolean, exhaustive node masks."""
        masks = (self.train_mask, self.validation_mask, self.test_mask)
        if any(not isinstance(mask, Tensor) for mask in masks):
            raise TypeError("temporal masks must be torch.Tensor objects")
        if any(mask.dtype != torch.bool or mask.ndim != 1 for mask in masks):
            raise ValueError("temporal masks must be one-dimensional bool tensors")
        if len({mask.numel() for mask in masks}) != 1:
            raise ValueError("temporal masks must have the same length")

        covered = self.train_mask | self.validation_mask | self.test_mask
        if not torch.all(covered):
            raise ValueError("temporal masks must cover every graph node")
        if torch.any(self.train_mask & self.validation_mask):
            raise ValueError("train and validation masks overlap")
        if torch.any(self.train_mask & self.test_mask):
            raise ValueError("train and test masks overlap")
        if torch.any(self.validation_mask & self.test_mask):
            raise ValueError("validation and test masks overlap")

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes covered by the masks."""
        return self.train_mask.numel()


def build_temporal_node_masks(
    graph: TransactionGraph,
    split: TemporalSplit,
) -> TemporalNodeMasks:
    """Build chronological node masks without changing the graph.

    The intervals are ``[-inf, train_end)``,
    ``[train_end, validation_end)``, and ``[validation_end, +inf)``. The
    returned masks have the same order as ``graph.nodes`` and are suitable for
    PyTorch Geometric-style node classification.

    中文：
        按照半开区间生成时间节点 mask，但不修改原图。返回 mask 与
        ``graph.nodes`` 行顺序一致，可直接用于 PyTorch Geometric 风格的
        节点分类。

    Unknown labels are intentionally not filtered here. If a dataset marks
    unknown labels explicitly, intersect the temporal mask with a separate
    known-label mask before calculating loss or metrics.

    Args:
        graph: Complete transaction graph whose nodes contain ``timestamp``.
        split: Chronological train/validation/test boundaries.

    Returns:
        A ``TemporalNodeMasks`` object. The input graph and its edge table are
        left unchanged.

    Raises:
        ValueError: If the graph has no timestamp column.
    """
    if "timestamp" not in graph.nodes.columns:
        raise ValueError("graph nodes must contain a timestamp column")

    timestamp = graph.nodes["timestamp"]
    train_mask = torch.from_numpy((timestamp < split.train_end).to_numpy()).to(
        dtype=torch.bool
    )
    validation_mask = torch.from_numpy(
        ((timestamp >= split.train_end) & (timestamp < split.validation_end)).to_numpy()
    ).to(dtype=torch.bool)
    test_mask = torch.from_numpy((timestamp >= split.validation_end).to_numpy()).to(
        dtype=torch.bool
    )

    return TemporalNodeMasks(
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
    )


def apply_temporal_split(
    graph: TransactionGraph,
    split: TemporalSplit,
) -> TransactionGraphSplit:
    """Apply a temporal split to a transaction graph.

    The existing induced-subgraph implementation is called lazily here to
    preserve one source of truth during migration. Cross-partition edges are
    removed by that implementation.

    中文：
        当前阶段延迟调用已有的诱导子图实现，以保持只有一份算法。跨分区边由
        该实现删除。
    """
    from amlgraphx.data.datamodule import split_transaction_graph

    return split_transaction_graph(
        graph,
        train_end=split.train_end,
        validation_end=split.validation_end,
    )


__all__ = [
    "TemporalNodeMasks",
    "TemporalSplit",
    "apply_temporal_split",
    "build_temporal_node_masks",
]
