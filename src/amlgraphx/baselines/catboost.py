"""Thin CatBoost integration for tabular AML baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier


class CatBoostBaseline:
    """Keep CatBoost's estimator behind a small AML-friendly interface.

    CatBoost remains responsible for model training. This adapter only adds
    the shared AMLGraphX entry points and optional SHAP explanations.
    """

    def __init__(self, *, use_shap: bool = False, **params: Any) -> None:
        """Create a CatBoost classifier with AML-oriented defaults.

        Args:
            use_shap: Enable :meth:`explain`; SHAP runs only when requested.
            **params: Any parameter accepted by ``CatBoostClassifier``.
        """
        self.use_shap = use_shap
        params.setdefault("eval_metric", "PRAUC")
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        self.estimator = CatBoostClassifier(**params)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        eval_set: Any | None = None,
        verbose: bool = False,
    ) -> CatBoostBaseline:
        """Fit CatBoost and return this baseline for method chaining."""
        fit_params: dict[str, Any] = {"verbose": verbose}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: Any) -> Any:
        """Predict binary labels using CatBoost."""
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Return class probabilities for AML ranking."""
        return self.estimator.predict_proba(X)

    def explain(self, X: Any, *, feature_names: Any | None = None) -> Any:
        """Return SHAP values when ``use_shap=True``."""
        if not self.use_shap:
            raise RuntimeError(
                "SHAP is disabled; construct CatBoostBaseline(use_shap=True)"
            )

        from amlgraphx.explain.shap import explain_tree_model

        return explain_tree_model(
            self.estimator,
            X,
            feature_names=feature_names,
        )

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        """Return native CatBoost parameters and the SHAP switch."""
        params = self.estimator.get_params(deep=deep)
        params["use_shap"] = self.use_shap
        return params

    def set_params(self, **params: Any) -> CatBoostBaseline:
        """Update native CatBoost parameters or the SHAP switch."""
        if "use_shap" in params:
            self.use_shap = bool(params.pop("use_shap"))
        self.estimator.set_params(**params)
        return self

    def save_model(self, path: str | Path) -> None:
        """Save the native CatBoost model to ``path``."""
        self.estimator.save_model(str(path))

    def load_model(self, path: str | Path) -> CatBoostBaseline:
        """Load a native CatBoost model from ``path``."""
        self.estimator.load_model(str(path))
        return self


__all__ = ["CatBoostBaseline"]
