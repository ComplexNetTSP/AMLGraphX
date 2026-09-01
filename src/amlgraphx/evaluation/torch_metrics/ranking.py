"""Global ranking Torch metrics for binary AML and fraud risk scores."""

from __future__ import annotations

from torch import Tensor
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)

from .base import BinaryRankingMetric


class AveragePrecision(BinaryRankingMetric):
    """Compute exact global Average Precision (the discrete PR-AUC variant)."""

    def compute(self) -> Tensor:
        """Return Average Precision over every accumulated prediction target."""
        self._require_two_classes()
        return binary_average_precision(self.risk_scores, self.targets)


class RocAuc(BinaryRankingMetric):
    """Compute exact global ROC-AUC as a supplementary ranking metric."""

    def compute(self) -> Tensor:
        """Return ROC-AUC over every accumulated prediction target."""
        self._require_two_classes()
        return binary_auroc(self.risk_scores, self.targets)


__all__ = ["AveragePrecision", "RocAuc"]
