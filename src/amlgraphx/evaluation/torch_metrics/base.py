"""Shared state and validation for binary AML/Fraud Torch metrics."""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric


class BinaryThresholdMetric(Metric):
    """Accumulate confusion counts for one fixed binary alert threshold.

    Subclasses only implement :meth:`compute`; all counts are reduced across
    devices by TorchMetrics. Scores at or above ``threshold`` are alerts.
    """

    full_state_update = False
    higher_is_better: bool | None = None

    def __init__(self, *, threshold: float = 0.5, **kwargs: object) -> None:
        """Create a metric with an explicit, preselected alert threshold."""
        super().__init__(**kwargs)
        if not torch.isfinite(torch.tensor(threshold)).item():
            raise ValueError("threshold must be finite")
        self.threshold = float(threshold)
        self.add_state(
            "true_positives",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_positives",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_negatives",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "true_negatives",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, risk_score: Tensor, target: Tensor) -> None:
        """Add one labelled batch of risk scores and binary targets."""
        score, y = validate_binary_tensors(risk_score, target)
        predicted = score >= self.threshold
        positive = y == 1
        self.true_positives += (predicted & positive).sum()
        self.false_positives += (predicted & ~positive).sum()
        self.false_negatives += (~predicted & positive).sum()
        self.true_negatives += (~predicted & ~positive).sum()


class BinaryRankingMetric(Metric):
    """Store one labelled split exactly for a global ranking calculation.

    PR-AUC, ROC-AUC and Top-K cannot be averaged batch by batch. This base
    retains detached score/target tensors and uses TorchMetrics' ``cat``
    reduction so subclasses calculate one exact result at :meth:`compute`.
    """

    full_state_update = False
    higher_is_better = True

    def __init__(self, **kwargs: object) -> None:
        """Create an empty metric state."""
        super().__init__(**kwargs)
        self.add_state("risk_scores", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state(
            "targets", default=torch.empty(0, dtype=torch.long), dist_reduce_fx="cat"
        )

    def update(self, risk_score: Tensor, target: Tensor) -> None:
        """Store one labelled batch without retaining its autograd graph."""
        score, y = validate_binary_tensors(risk_score, target)
        score = score.detach()
        y = y.detach()
        if self.risk_scores.numel() == 0:
            self.risk_scores = score.clone()
            self.targets = y.clone()
        else:
            self.risk_scores = torch.cat((self.risk_scores, score))
            self.targets = torch.cat((self.targets, y))

    def _require_two_classes(self) -> None:
        """Reject undefined ranking metrics before delegating to a backend."""
        if self.targets.numel() == 0:
            raise ValueError("metric has no accumulated targets")
        positive_count = int(self.targets.sum().item())
        if positive_count == 0 or positive_count == self.targets.numel():
            raise ValueError(
                "ranking metrics require both positive and negative targets"
            )


def validate_binary_tensors(
    risk_score: Tensor, target: Tensor
) -> tuple[Tensor, Tensor]:
    """Validate aligned one-dimensional scores and binary labels.

    ``risk_score`` may contain arbitrary finite ranking scores, not only
    calibrated probabilities. ``target=1`` denotes fraud or AML positive.
    """
    if not isinstance(risk_score, Tensor) or not isinstance(target, Tensor):
        raise TypeError("risk_score and target must be torch.Tensor objects")
    if risk_score.ndim != 1 or target.ndim != 1:
        raise ValueError("risk_score and target must be one-dimensional")
    if risk_score.numel() == 0 or target.numel() == 0:
        raise ValueError("risk_score and target must be non-empty")
    if risk_score.shape != target.shape:
        raise ValueError("risk_score and target must have the same shape")
    if risk_score.device != target.device:
        raise ValueError("risk_score and target must be on the same device")
    if not risk_score.is_floating_point():
        raise TypeError("risk_score must use a floating-point dtype")
    if not torch.isfinite(risk_score).all().item():
        raise ValueError("risk_score must contain only finite values")
    if not torch.isin(target, torch.tensor((0, 1), device=target.device)).all().item():
        raise ValueError("target must contain only binary labels 0 and 1")
    return risk_score, target.to(dtype=torch.long)


def safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Divide count tensors, returning scalar zero for an empty denominator."""
    numerator = numerator.to(dtype=torch.get_default_dtype())
    denominator = denominator.to(dtype=torch.get_default_dtype())
    return torch.where(
        denominator != 0,
        numerator / denominator,
        torch.zeros((), dtype=numerator.dtype, device=numerator.device),
    )


def f1_from_precision_recall(precision: Tensor, recall: Tensor) -> Tensor:
    """Return the harmonic mean, defining the all-zero case as zero."""
    return torch.where(
        precision + recall != 0,
        2 * precision * recall / (precision + recall),
        torch.zeros((), dtype=precision.dtype, device=precision.device),
    )


__all__ = [
    "BinaryRankingMetric",
    "BinaryThresholdMetric",
    "f1_from_precision_recall",
    "safe_divide",
    "validate_binary_tensors",
]
