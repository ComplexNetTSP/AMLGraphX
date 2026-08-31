"""Tests for the CatBoost and LightGBM baseline integrations."""

from pathlib import Path

import numpy as np
import pytest

from amlgraphx.baselines import CatBoostBaseline, LightGBMBaseline

X = np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
y = np.asarray([0, 0, 1, 1], dtype=np.int64)


@pytest.mark.parametrize(
    ("model", "metric"),
    [
        (
            CatBoostBaseline(
                use_shap=True,
                iterations=4,
                depth=2,
                learning_rate=0.3,
                thread_count=1,
                random_seed=0,
            ),
            "PRAUC",
        ),
        (
            LightGBMBaseline(
                use_shap=True,
                n_estimators=4,
                num_leaves=4,
                learning_rate=0.3,
                n_jobs=1,
                random_state=0,
            ),
            "average_precision",
        ),
    ],
)
def test_tree_baseline_trains_and_explains(
    model: CatBoostBaseline | LightGBMBaseline,
    metric: str,
) -> None:
    """Both wrappers expose the same AML-facing prediction and SHAP flow."""
    model.fit(X, y, sample_weight=np.ones_like(y, dtype=np.float32))
    probabilities = model.predict_proba(X)
    explanation = model.explain(X, feature_names=("amount",))

    assert model.get_params()["eval_metric"] == metric
    assert probabilities.shape == (4, 2)
    assert explanation.values.shape == (4, 1)
    assert tuple(explanation.feature_names) == ("amount",)


def test_lightgbm_baseline_loads_native_booster(tmp_path: Path) -> None:
    """A saved LightGBM booster remains usable through the wrapper."""
    model = LightGBMBaseline(
        n_estimators=4,
        num_leaves=4,
        learning_rate=0.3,
        n_jobs=1,
        random_state=0,
    ).fit(X, y)
    expected = model.predict_proba(X)
    path = tmp_path / "lightgbm.txt"
    model.save_model(path)

    restored = LightGBMBaseline().load_model(path)

    assert np.allclose(restored.predict_proba(X), expected)
