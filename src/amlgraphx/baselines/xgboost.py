"""Thin XGBoost integration for tabular AML baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xgboost import XGBClassifier


class XGBoostBaseline:
    """Keep XGBoost's estimator while exposing an AML-friendly entry point.

    This class does not implement gradient boosting. It only supplies the
    small integration layer needed by AMLGraphX: PR-AUC as the default
    training metric, optional sample weights for class imbalance, and a stable
    fit/predict/save interface. All constructor parameters are passed to the
    native ``XGBClassifier``.
    """

    def __init__(self, *, use_shap: bool = False, **params: Any) -> None:
        """Create an XGBoost classifier with AML-oriented defaults.

        Args:
            use_shap: Enable the :meth:`explain` convenience method. SHAP is
                evaluated only when that method is called.
            **params: Any parameter accepted by ``xgboost.XGBClassifier``.
                For example, ``scale_pos_weight``, ``max_depth`` and
                ``n_estimators`` remain native XGBoost parameters.
        """
        self.use_shap = use_shap
        params.setdefault("eval_metric", "aucpr")
        self.estimator = XGBClassifier(**params)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        eval_set: Any | None = None,
        verbose: bool = False,
    ) -> XGBoostBaseline:
        """Fit the native estimator and return this baseline.

        ``sample_weight`` is deliberately forwarded to XGBoost instead of
        performing resampling here. SMOTE and temporal sampling belong in
        ``amlgraphx.sampling``.
        """
        fit_params: dict[str, Any] = {"verbose": verbose}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: Any) -> Any:
        """Predict binary labels using the native estimator."""
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Return class probabilities for AML ranking and evaluation."""
        return self.estimator.predict_proba(X)

    def explain(self, X: Any, *, feature_names: Any | None = None) -> Any:
        """Return SHAP values for ``X`` when SHAP is enabled.

        Args:
            X: The same feature representation used by the fitted estimator.
            feature_names: Optional names assigned to the returned SHAP
                explanation, such as ``["amount", "fan_in"]``.

        Raises:
            RuntimeError: If ``use_shap`` is ``False``.
        """
        if not self.use_shap:
            raise RuntimeError(
                "SHAP is disabled; construct XGBoostBaseline(use_shap=True)"
            )

        from amlgraphx.explain.shap import explain_tree_model

        return explain_tree_model(
            self.estimator,
            X,
            feature_names=feature_names,
        )

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        """Return native parameters plus the AMLGraphX SHAP switch."""
        params = self.estimator.get_params(deep=deep)
        params["use_shap"] = self.use_shap
        return params

    def set_params(self, **params: Any) -> XGBoostBaseline:
        """Update native parameters or the SHAP switch."""
        if "use_shap" in params:
            self.use_shap = bool(params.pop("use_shap"))
        self.estimator.set_params(**params)
        return self

    def save_model(self, path: str | Path) -> None:
        """Save the native XGBoost model to ``path``."""
        self.estimator.save_model(str(path))

    def load_model(self, path: str | Path) -> XGBoostBaseline:
        """Load a native XGBoost model from ``path``."""
        self.estimator.load_model(str(path))
        return self


__all__ = ["XGBoostBaseline"]
