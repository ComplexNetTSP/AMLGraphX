"""NumPy metrics for binary fraud and AML risk scores.

The functions in this module evaluate one fully materialized, labelled split.
They do not choose a threshold, create a data split, or fit a model. For a
temporal experiment, pass only the prediction targets for the split being
reported; do not include lookback context rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Metrics produced by one fixed risk-score threshold.

    The threshold is an experimental policy. Select it on a validation split
    or set it before observing test labels; this class does not tune it.
    """

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    false_positive_rate: float

    def as_dict(self) -> dict[str, float | int]:
        """Return named scalar values suitable for a result table or logger."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InvestigationMetrics:
    """Metrics at one fixed, highest-risk investigation budget."""

    k: int
    positive_count: int
    precision: float
    recall: float
    f1: float
    lift: float

    def as_dict(self) -> dict[str, float | int]:
        """Return named scalar values suitable for a result table or logger."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BinaryRiskMetrics:
    """Threshold, ranking, and investigation-budget results for one split."""

    sample_count: int
    positive_count: int
    negative_count: int
    average_precision: float
    roc_auc: float
    threshold: ThresholdMetrics | None
    investigation: tuple[InvestigationMetrics, ...]

    def as_dict(self) -> dict[str, float | int]:
        """Flatten scalar results for a CSV row or experiment tracker.

        Investigation fields use names such as ``precision_at_100``. The
        caller retains the unit meaning of ``K`` (transactions, accounts, or
        graph nodes) in experiment metadata rather than in these scalar names.
        """
        values: dict[str, float | int] = {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
        }
        if self.threshold is not None:
            values.update(
                {
                    f"threshold_{name}": value
                    for name, value in self.threshold.as_dict().items()
                }
            )
        for budget in self.investigation:
            values.update(
                {
                    f"{name}_at_{budget.k}": value
                    for name, value in budget.as_dict().items()
                    if name != "k"
                }
            )
        return values


def threshold_metrics(
    y_true: ArrayLike,
    risk_score: ArrayLike,
    *,
    threshold: float = 0.5,
) -> ThresholdMetrics:
    """Calculate binary precision, recall, F1, and confusion counts.

    Args:
        y_true: One-dimensional binary labels where ``1`` is fraud / AML.
        risk_score: One-dimensional risk scores aligned with ``y_true``.
            Scores need not be calibrated probabilities.
        threshold: Score at or above which a record enters the alert queue.

    Returns:
        Confusion counts plus threshold-dependent metrics. A zero denominator
        produces ``0.0`` rather than a warning.

    Raises:
        ValueError: If inputs are empty, misaligned, non-finite, or non-binary.
    """
    y, score = _validate_binary_inputs(y_true, risk_score)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    predicted = score >= threshold
    positives = y == 1
    true_positives = int(np.count_nonzero(predicted & positives))
    false_positives = int(np.count_nonzero(predicted & ~positives))
    false_negatives = int(np.count_nonzero(~predicted & positives))
    true_negatives = int(np.count_nonzero(~predicted & ~positives))
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)

    return ThresholdMetrics(
        threshold=float(threshold),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        accuracy=_safe_divide(true_positives + true_negatives, y.size),
        false_positive_rate=_safe_divide(
            false_positives, false_positives + true_negatives
        ),
    )


def investigation_metrics(
    y_true: ArrayLike,
    risk_score: ArrayLike,
    *,
    k: int,
) -> InvestigationMetrics:
    """Calculate Precision@K, Recall@K, F1@K, and Lift@K.

    Higher scores are investigated first. Equal scores preserve their input
    order, so callers needing cross-run reproducibility should pre-sort input
    rows with a stable identifier such as ``transaction_id``.

    Args:
        y_true: One-dimensional binary labels where ``1`` is fraud / AML.
        risk_score: One-dimensional risk scores aligned with ``y_true``.
        k: Number of highest-risk records available for investigation.

    Returns:
        Investigation-budget metrics. Recall's denominator is every positive
        in ``y_true``, not only positives in the alert queue.

    Raises:
        ValueError: If ``k`` is outside ``[1, len(y_true)]``, the inputs are
            invalid, or the evaluation split has no positive labels.
    """
    y, score = _validate_binary_inputs(y_true, risk_score)
    _validate_k(k, y.size)
    positive_count = int(np.count_nonzero(y))
    if positive_count == 0:
        raise ValueError("investigation metrics require at least one positive label")

    order = np.lexsort((np.arange(y.size), -score))
    positives_at_k = int(np.count_nonzero(y[order[:k]]))
    precision = positives_at_k / k
    recall = positives_at_k / positive_count
    prevalence = positive_count / y.size
    return InvestigationMetrics(
        k=k,
        positive_count=positives_at_k,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        lift=precision / prevalence,
    )


def evaluate_binary_risk_scores(
    y_true: ArrayLike,
    risk_score: ArrayLike,
    *,
    threshold: float | None = None,
    top_k: Sequence[int] = (),
    top_fractions: Sequence[float] = (),
) -> BinaryRiskMetrics:
    """Evaluate labelled AML or fraud risk scores on one frozen split.

    ``average_precision`` is scikit-learn's discrete Average Precision, not a
    trapezoidal approximation of the PR curve. It is the AMLGraphX numeric
    implementation of the PR-AUC-like ranking metric described in
    ``evaluation/metrics.md``. ``roc_auc`` is included for comparison, but
    Average Precision and investigation budgets should normally be primary on
    severely imbalanced data.

    Args:
        y_true: One-dimensional binary labels where ``1`` is fraud / AML.
        risk_score: One-dimensional risk scores aligned with ``y_true``.
        threshold: Optional preselected alert threshold. ``None`` omits
            threshold-dependent results instead of choosing from test labels.
        top_k: Fixed investigation capacities, such as ``(100, 1_000)``.
        top_fractions: Investigation proportions in ``(0, 1]``. Each becomes
            ``ceil(fraction * sample_count)``; for example ``0.01`` is top 1%.

    Returns:
        Average Precision, ROC-AUC, optional fixed-threshold results, and one
        result for each requested investigation budget.

    Raises:
        ValueError: If inputs are invalid, one class is absent, a budget is
            invalid, or equivalent fixed/fractional budgets are repeated.
    """
    y, score = _validate_binary_inputs(y_true, risk_score)
    positive_count = int(np.count_nonzero(y))
    negative_count = y.size - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Average Precision and ROC-AUC require both positive and negative labels"
        )

    for value in top_k:
        _validate_k(value, y.size)
    budgets = [int(value) for value in top_k]
    budgets.extend(_k_from_fraction(value, y.size) for value in top_fractions)
    if len(set(budgets)) != len(budgets):
        raise ValueError("each investigation budget must be unique")
    for k in budgets:
        _validate_k(k, y.size)

    return BinaryRiskMetrics(
        sample_count=int(y.size),
        positive_count=positive_count,
        negative_count=negative_count,
        average_precision=float(average_precision_score(y, score)),
        roc_auc=float(roc_auc_score(y, score)),
        threshold=(
            threshold_metrics(y, score, threshold=threshold)
            if threshold is not None
            else None
        ),
        investigation=tuple(investigation_metrics(y, score, k=k) for k in budgets),
    )


def _validate_binary_inputs(
    y_true: ArrayLike,
    risk_score: ArrayLike,
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Return validated one-dimensional binary labels and finite scores."""
    y = np.asarray(y_true)
    score = np.asarray(risk_score, dtype=np.float64)
    if y.ndim != 1 or score.ndim != 1:
        raise ValueError("y_true and risk_score must be one-dimensional")
    if y.size == 0 or score.size == 0:
        raise ValueError("y_true and risk_score must be non-empty")
    if y.size != score.size:
        raise ValueError("y_true and risk_score must have the same length")
    if not np.isfinite(score).all():
        raise ValueError("risk_score must contain only finite values")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    return y.astype(np.int8, copy=False), score


def _validate_k(k: int, sample_count: int) -> None:
    """Validate that one fixed investigation budget fits the split."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if not 1 <= k <= sample_count:
        raise ValueError(f"k must be between 1 and {sample_count}")


def _k_from_fraction(fraction: float, sample_count: int) -> int:
    """Convert one explicit investigation fraction into a non-zero budget."""
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("top_fractions values must be finite and in (0, 1]")
    return ceil(fraction * sample_count)


def _safe_divide(numerator: int, denominator: int) -> float:
    """Return zero for undefined empty-alert rates without emitting warnings."""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """Return the harmonic mean, defining the all-zero case as zero."""
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )


__all__ = [
    "BinaryRiskMetrics",
    "InvestigationMetrics",
    "ThresholdMetrics",
    "evaluate_binary_risk_scores",
    "investigation_metrics",
    "threshold_metrics",
]
