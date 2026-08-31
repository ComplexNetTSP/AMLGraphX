"""Small SHAP helpers shared by AMLGraphX model integrations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import shap


def explain_tree_model(
    model: Any,
    X: Any,
    *,
    feature_names: Sequence[str] | None = None,
) -> shap.Explanation:
    """Explain a fitted tree model with SHAP's native tree explainer.

    Args:
        model: A fitted XGBoost, LightGBM, CatBoost, or compatible tree model.
        X: The samples to explain. Use the same feature order used for fitting.
        feature_names: Optional feature names for readable downstream output.

    Returns:
        A native ``shap.Explanation`` containing SHAP values and feature data.

    Raises:
        ValueError: If ``feature_names`` does not match the feature count.

    Notes:
        The default SHAP output explains the model's native tree output. This
        helper intentionally does not sample rows or alter the model output;
        callers can choose the explanation population and post-processing.
    """
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")

    explanation = shap.TreeExplainer(model)(X)
    if feature_names is not None:
        names = tuple(feature_names)
        if len(names) != values.shape[1]:
            raise ValueError(
                "feature_names must have one name per input feature "
                f"({values.shape[1]} expected, got {len(names)})"
            )
        explanation.feature_names = names
    return explanation


__all__ = ["explain_tree_model"]
