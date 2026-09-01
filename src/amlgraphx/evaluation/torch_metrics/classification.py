"""Threshold-dependent Torch metrics for binary AML and fraud alerts."""

from __future__ import annotations

from torch import Tensor

from .base import BinaryThresholdMetric, f1_from_precision_recall, safe_divide


class Precision(BinaryThresholdMetric):
    """Compute alert precision from globally accumulated confusion counts."""

    higher_is_better = True

    def compute(self) -> Tensor:
        """Return ``TP / (TP + FP)`` or zero when no alert was emitted."""
        return safe_divide(
            self.true_positives, self.true_positives + self.false_positives
        )


class Recall(BinaryThresholdMetric):
    """Compute AML/fraud positive-class recall from globally accumulated counts."""

    higher_is_better = True

    def compute(self) -> Tensor:
        """Return ``TP / (TP + FN)`` or zero when no positives were observed."""
        return safe_divide(
            self.true_positives, self.true_positives + self.false_negatives
        )


class F1(BinaryThresholdMetric):
    """Compute positive-class F1 at one fixed risk-score threshold."""

    higher_is_better = True

    def compute(self) -> Tensor:
        """Return the harmonic mean of global precision and recall."""
        precision = safe_divide(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = safe_divide(
            self.true_positives, self.true_positives + self.false_negatives
        )
        return f1_from_precision_recall(precision, recall)


class Accuracy(BinaryThresholdMetric):
    """Compute accuracy as a supplementary, not primary, imbalanced metric."""

    higher_is_better = True

    def compute(self) -> Tensor:
        """Return ``(TP + TN) / (TP + FP + FN + TN)``."""
        total = (
            self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives
        )
        return safe_divide(self.true_positives + self.true_negatives, total)


class FalsePositiveRate(BinaryThresholdMetric):
    """Compute the fraction of known-negative records that became alerts."""

    higher_is_better = False

    def compute(self) -> Tensor:
        """Return ``FP / (FP + TN)`` or zero when no negatives were observed."""
        return safe_divide(
            self.false_positives, self.false_positives + self.true_negatives
        )


__all__ = ["F1", "Accuracy", "FalsePositiveRate", "Precision", "Recall"]
