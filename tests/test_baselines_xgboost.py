"""Tests for the thin XGBoost baseline integration."""

import numpy as np
import pytest

from amlgraphx.baselines import XGBoostBaseline


def test_xgboost_baseline_delegates_training_and_probability_prediction() -> None:
    """The wrapper keeps native parameters and supports AML sample weights."""
    X = np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)

    model = XGBoostBaseline(
        n_estimators=4,
        max_depth=2,
        learning_rate=0.3,
        tree_method="hist",
        n_jobs=1,
        random_state=0,
    )
    model.fit(X, y, sample_weight=np.ones_like(y, dtype=np.float32))

    probabilities = model.predict_proba(X)

    assert model.get_params()["eval_metric"] == "aucpr"
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_xgboost_baseline_can_return_shap_explanations() -> None:
    """The opt-in SHAP path returns a labelled native explanation."""
    X = np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    model = XGBoostBaseline(
        use_shap=True,
        n_estimators=4,
        max_depth=2,
        tree_method="hist",
        n_jobs=1,
        random_state=0,
    ).fit(X, y)

    explanation = model.explain(X, feature_names=("amount",))

    assert explanation.values.shape == (4, 1)
    assert tuple(explanation.feature_names) == ("amount",)


def test_xgboost_baseline_requires_explicit_shap_opt_in() -> None:
    """SHAP stays disabled unless the caller explicitly enables it."""
    model = XGBoostBaseline()

    with pytest.raises(RuntimeError, match="SHAP is disabled"):
        model.explain(np.zeros((1, 1), dtype=np.float32))
