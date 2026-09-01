"""Tests for AML/fraud risk-score metric semantics."""

import numpy as np
import pytest
import torch

from amlgraphx.evaluation import (
    F1,
    Accuracy,
    AveragePrecision,
    F1AtK,
    FalsePositiveRate,
    LiftAtK,
    Precision,
    PrecisionAtK,
    Recall,
    RecallAtK,
    RocAuc,
    evaluate_binary_risk_scores,
    investigation_metrics,
    threshold_metrics,
)


def test_binary_risk_evaluation_reports_ranking_threshold_and_budget_metrics() -> None:
    """One labelled split produces documented global and top-K values."""
    y_true = np.asarray([0, 1, 0, 1], dtype=np.int64)
    score = np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float64)

    result = evaluate_binary_risk_scores(
        y_true,
        score,
        threshold=0.5,
        top_k=(2,),
        top_fractions=(0.25,),
    )

    assert result.sample_count == 4
    assert result.positive_count == 2
    assert result.negative_count == 2
    assert result.average_precision == pytest.approx(5 / 6)
    assert result.roc_auc == pytest.approx(0.75)
    assert result.threshold is not None
    assert result.threshold.precision == pytest.approx(0.5)
    assert result.threshold.recall == pytest.approx(0.5)
    assert result.threshold.f1 == pytest.approx(0.5)
    assert result.threshold.false_positive_rate == pytest.approx(0.5)
    assert result.investigation[0].k == 2
    assert result.investigation[0].precision == pytest.approx(0.5)
    assert result.investigation[0].recall == pytest.approx(0.5)
    assert result.investigation[0].lift == pytest.approx(1.0)
    assert result.investigation[1].k == 1
    assert result.investigation[1].precision == pytest.approx(1.0)
    assert result.as_dict()["precision_at_2"] == pytest.approx(0.5)


def test_investigation_metrics_use_stable_input_order_for_equal_scores() -> None:
    """A deterministic input tie-breaker makes Precision@K reproducible."""
    result = investigation_metrics([1, 0], [0.5, 0.5], k=1)

    assert result.positive_count == 1
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_threshold_metrics_define_an_empty_alert_queue_as_zero_precision() -> None:
    """Fixed thresholds never emit an undefined-value warning for no alerts."""
    result = threshold_metrics([0, 1], [0.1, 0.2], threshold=0.5)

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.accuracy == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("y_true", "score", "match"),
    [
        ([0, 1], [0.1], "same length"),
        ([0, 2], [0.1, 0.2], "binary labels"),
        ([0, 1], [0.1, np.nan], "finite"),
        ([0, 0], [0.1, 0.2], "both positive and negative"),
    ],
)
def test_binary_risk_evaluation_rejects_invalid_or_unlabelled_splits(
    y_true: list[int], score: list[float], match: str
) -> None:
    """A metric result is never silently invented for an invalid split."""
    with pytest.raises(ValueError, match=match):
        evaluate_binary_risk_scores(y_true, score)


def test_binary_risk_evaluation_rejects_duplicate_budgets() -> None:
    """Equivalent fixed and fractional budgets cannot overwrite a result key."""
    with pytest.raises(ValueError, match="unique"):
        evaluate_binary_risk_scores(
            [0, 1, 0, 1],
            [0.1, 0.9, 0.8, 0.2],
            top_k=(1,),
            top_fractions=(0.25,),
        )


@pytest.mark.parametrize("budget", [(1.9,), (True,)])
def test_binary_risk_evaluation_rejects_invalid_budget_types(
    budget: tuple[object],
) -> None:
    """Invalid fixed budgets are rejected before normalization."""
    with pytest.raises(TypeError, match="integer"):
        evaluate_binary_risk_scores(
            [0, 1],
            [0.1, 0.9],
            top_k=budget,  # type: ignore[arg-type]
        )


def test_metrics_are_public_from_canonical_namespace() -> None:
    """Reusable metrics are available from amlgraphx.metrics."""
    from amlgraphx.metrics import evaluate_binary_risk_scores as canonical

    assert canonical is evaluate_binary_risk_scores


def test_torch_metric_instances_are_composable_in_a_metric_dictionary() -> None:
    """Independent metrics update over batches and compute full-split values."""
    score = torch.tensor([0.1, 0.9, 0.8, 0.2])
    target = torch.tensor([0, 1, 0, 1])
    metrics = {
        "precision": Precision(threshold=0.5),
        "recall": Recall(threshold=0.5),
        "f1": F1(threshold=0.5),
        "accuracy": Accuracy(threshold=0.5),
        "false_positive_rate": FalsePositiveRate(threshold=0.5),
        "average_precision": AveragePrecision(),
        "roc_auc": RocAuc(),
        "precision_at_2": PrecisionAtK(2),
        "recall_at_2": RecallAtK(2),
        "f1_at_2": F1AtK(2),
        "lift_at_2": LiftAtK(2),
    }

    for metric in metrics.values():
        metric.update(score[:2], target[:2])
        metric.update(score[2:], target[2:])
    values = {name: metric.compute().item() for name, metric in metrics.items()}

    assert values["precision"] == pytest.approx(0.5)
    assert values["recall"] == pytest.approx(0.5)
    assert values["f1"] == pytest.approx(0.5)
    assert values["accuracy"] == pytest.approx(0.5)
    assert values["false_positive_rate"] == pytest.approx(0.5)
    assert values["average_precision"] == pytest.approx(5 / 6)
    assert values["roc_auc"] == pytest.approx(0.75)
    assert values["precision_at_2"] == pytest.approx(0.5)
    assert values["recall_at_2"] == pytest.approx(0.5)
    assert values["f1_at_2"] == pytest.approx(0.5)
    assert values["lift_at_2"] == pytest.approx(1.0)


def test_torch_ranking_metrics_validate_global_budget_at_compute_time() -> None:
    """Top-K state is global, so an oversized budget fails only at compute."""
    metric = PrecisionAtK(3)
    metric.update(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))

    with pytest.raises(ValueError, match="at most"):
        metric.compute()
