"""Composable TorchMetrics classes for binary AML and fraud evaluation.

Each object follows the standard ``update(risk_score, target)`` /
``compute()`` / ``reset()`` lifecycle. Classes can be instantiated separately
and stored in a training engine's metric dictionary.
"""

from .classification import F1, Accuracy, FalsePositiveRate, Precision, Recall
from .investigation import F1AtK, InvestigationMetric, LiftAtK, PrecisionAtK, RecallAtK
from .ranking import AveragePrecision, RocAuc

__all__ = [
    "F1",
    "Accuracy",
    "AveragePrecision",
    "F1AtK",
    "FalsePositiveRate",
    "InvestigationMetric",
    "LiftAtK",
    "Precision",
    "PrecisionAtK",
    "Recall",
    "RecallAtK",
    "RocAuc",
]
