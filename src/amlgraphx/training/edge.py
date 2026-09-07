"""Lightning orchestration for binary transaction-edge risk scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor, nn
from torchmetrics import Metric

from .static import StaticBinaryNodePredictor, _batch_num_edges, _validate_logits


class StaticBinaryEdgePredictor(StaticBinaryNodePredictor):
    """Train a researcher model that returns one binary logit per graph edge.

    Account-as-node graphs retain transactions as directed edges. This predictor
    keeps the standard Lightning lifecycle while placing labels and split masks
    on those transaction edges instead of account nodes.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[Tensor, Tensor], Tensor],
        *,
        metrics: Mapping[str, Metric] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an edge-risk predictor with AMLGraphX edge-field defaults."""
        super().__init__(
            model,
            loss,
            metrics=metrics,
            target_attr=kwargs.pop("target_attr", "edge_y"),
            train_mask_attr=kwargs.pop("train_mask_attr", "train_edge_mask"),
            validation_mask_attr=kwargs.pop(
                "validation_mask_attr", "validation_edge_mask"
            ),
            test_mask_attr=kwargs.pop("test_mask_attr", "test_edge_mask"),
            **kwargs,
        )

    def forward(self, batch: Any) -> Tensor:
        """Return validated raw logits for every directed transaction edge."""
        return _validate_logits(
            self.model(batch), _batch_num_edges(batch), item_name="edge"
        )


__all__ = ["StaticBinaryEdgePredictor"]
