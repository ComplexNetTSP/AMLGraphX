"""Thin LightGBM integration for tabular AML baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import Booster, LGBMClassifier


class LightGBMBaseline:
    """Keep LightGBM's estimator behind a small AML-friendly interface.

    LightGBM remains responsible for model training. This adapter only adds
    the shared AMLGraphX entry points and optional SHAP explanations.
    """

    def __init__(self, *, use_shap: bool = False, **params: Any) -> None:
        """Create a LightGBM classifier with AML-oriented defaults.

        Args:
            use_shap: Enable :meth:`explain`; SHAP runs only when requested.
            **params: Any parameter accepted by ``LGBMClassifier``.
        """
        self.use_shap = use_shap
        self.eval_metric = params.pop("eval_metric", "average_precision")
        params.setdefault("verbosity", -1)
        self.estimator = LGBMClassifier(**params)
        self._booster: Booster | None = None

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        eval_set: Any | None = None,
    ) -> LightGBMBaseline:
        """Fit LightGBM and return this baseline for method chaining."""
        self._booster = None
        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
            fit_params["eval_metric"] = self.eval_metric
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: Any) -> Any:
        """Predict binary labels using LightGBM."""
        if self._booster is not None:
            return (self._booster.predict(X) >= 0.5).astype(int)
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Return class probabilities for AML ranking."""
        if self._booster is not None:
            positive = self._booster.predict(X)
            return np.column_stack((1.0 - positive, positive))
        return self.estimator.predict_proba(X)

    def explain(self, X: Any, *, feature_names: Any | None = None) -> Any:
        """Return SHAP values when ``use_shap=True``."""
        if not self.use_shap:
            raise RuntimeError(
                "SHAP is disabled; construct LightGBMBaseline(use_shap=True)"
            )

        from amlgraphx.explain.shap import explain_tree_model

        model = self._booster if self._booster is not None else self.estimator
        return explain_tree_model(
            model,
            X,
            feature_names=feature_names,
        )

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        """Return native LightGBM parameters and the SHAP switch."""
        params = self.estimator.get_params(deep=deep)
        params["use_shap"] = self.use_shap
        params["eval_metric"] = self.eval_metric
        return params

    def set_params(self, **params: Any) -> LightGBMBaseline:
        """Update native LightGBM parameters or the SHAP switch."""
        if "use_shap" in params:
            self.use_shap = bool(params.pop("use_shap"))
        if "eval_metric" in params:
            self.eval_metric = params.pop("eval_metric")
        self.estimator.set_params(**params)
        return self

    def save_model(self, path: str | Path) -> None:
        """Save the native LightGBM model to ``path``."""
        booster = self._booster or self.estimator.booster_
        booster.save_model(str(path))

    def load_model(self, path: str | Path) -> LightGBMBaseline:
        """Load a LightGBM model from ``path``."""
        self._booster = Booster(model_file=str(path))
        return self


__all__ = ["LightGBMBaseline"]
