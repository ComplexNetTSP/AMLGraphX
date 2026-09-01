"""Investigation-budget Torch metrics for ranked AML and fraud alerts."""

from __future__ import annotations

from torch import Tensor

from .base import BinaryRankingMetric, f1_from_precision_recall, safe_divide


class InvestigationMetric(BinaryRankingMetric):
    """Base class for one fixed highest-risk investigation budget ``k``."""

    def __init__(self, k: int, **kwargs: object) -> None:
        """Create a global ranking metric for exactly ``k`` records.

        Args:
            k: Number of top-risk records available for investigation. It is
                validated against the accumulated split at :meth:`compute`.
        """
        super().__init__(**kwargs)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    def _counts(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return positives at K, all positives, and global sample count."""
        self._require_two_classes()
        if self.k > self.targets.numel():
            raise ValueError(
                f"k must be at most the accumulated sample count {self.targets.numel()}"
            )
        order = self.risk_scores.argsort(descending=True, stable=True)
        return (
            self.targets[order[: self.k]].sum(),
            self.targets.sum(),
            self.targets.new_tensor(self.targets.numel()),
        )


class PrecisionAtK(InvestigationMetric):
    """Compute the confirmed-positive rate in the top ``k`` risk alerts."""

    def compute(self) -> Tensor:
        """Return ``positives_in_top_k / k`` over the complete split."""
        positives_at_k, _, _ = self._counts()
        return safe_divide(positives_at_k, positives_at_k.new_tensor(self.k))


class RecallAtK(InvestigationMetric):
    """Compute the share of all known positives captured in the top ``k``."""

    def compute(self) -> Tensor:
        """Return ``positives_in_top_k / all_positives``."""
        positives_at_k, positive_count, _ = self._counts()
        return safe_divide(positives_at_k, positive_count)


class F1AtK(InvestigationMetric):
    """Compute F1 from Precision@K and Recall@K on the complete split."""

    def compute(self) -> Tensor:
        """Return the harmonic mean of the fixed-budget precision and recall."""
        positives_at_k, positive_count, _ = self._counts()
        precision = safe_divide(positives_at_k, positives_at_k.new_tensor(self.k))
        recall = safe_divide(positives_at_k, positive_count)
        return f1_from_precision_recall(precision, recall)


class LiftAtK(InvestigationMetric):
    """Compute top-``k`` precision relative to positive prevalence."""

    def compute(self) -> Tensor:
        """Return ``Precision@K / (all_positives / all_records)``."""
        positives_at_k, positive_count, sample_count = self._counts()
        precision = safe_divide(positives_at_k, positives_at_k.new_tensor(self.k))
        prevalence = safe_divide(positive_count, sample_count)
        return safe_divide(precision, prevalence)


__all__ = [
    "F1AtK",
    "InvestigationMetric",
    "LiftAtK",
    "PrecisionAtK",
    "RecallAtK",
]
