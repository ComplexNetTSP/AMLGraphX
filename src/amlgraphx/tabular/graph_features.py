"""Streaming graph features for tabular AML classifiers.

The public API intentionally follows SnapML's ``GraphFeaturePreprocessor``:
transactions are directed temporal edges with the required layout
``[edge_id, source_id, target_id, timestamp, ...numeric features]``. The
native implementation maintains a bounded dynamic multigraph and appends
pattern and account-statistic features suitable for XGBoost or LightGBM.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from amlgraphx._native._core import NativeGraphFeaturePreprocessor
except ImportError as error:  # pragma: no cover - exercised by package build.
    message = (
        "AMLGraphX native tabular features are unavailable. Build the package "
        "with `uv run maturin develop --release --manifest-path rust/Cargo.toml`."
    )
    raise ImportError(message) from error


class GraphFeaturePreprocessor:
    """Extract stateful graph features from a stream of temporal transactions.

    ``fit`` replaces the dynamic graph, ``partial_fit`` only inserts rows, and
    ``transform`` inserts an entire batch before producing its features. This
    matches SnapML's batch semantics: rows in the same transform batch can see
    each other. Use :meth:`transform_causal` when every row must only use
    strictly earlier rows.

    The appended columns follow this fixed order:

    ``fan-in, fan-out, degree-in, degree-out, scatter-gather, temporal-cycle,
    length-constrained-cycle, source-out stats, source-in stats, target-out
    stats, target-in stats``.

    All input values are converted to contiguous ``float64``. Account and edge
    IDs are graph identifiers only; drop those three raw ID columns before
    fitting a downstream tabular estimator.
    """

    def __init__(self) -> None:
        """Create an empty preprocessor with SnapML-compatible defaults."""
        self.params = _default_params()
        self._native = _build_native(self.params)

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        """Return the current SnapML-compatible parameter dictionary.

        Args:
            deep: Return an independent copy when ``True``.
        """
        return deepcopy(self.params) if deep else self.params

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Update parameters and clear all in-memory transaction history.

        Args:
            params: A subset of the documented SnapML GFP configuration keys.

        Raises:
            KeyError: If a parameter name is not supported.
            ValueError: If a value is incompatible with graph feature semantics.
        """
        updated = deepcopy(self.params)
        for key, value in params.items():
            if key not in updated:
                raise KeyError(f"Unsupported key: {key}")
            updated[key] = value
        self._native = _build_native(updated)
        self.params = updated

    def fit(self, features: ArrayLike, y: ArrayLike | None = None) -> None:
        """Replace the in-memory graph with ``features`` without emitting rows.

        Args:
            features: Matrix shaped ``(n_edges, n_raw_features)`` with columns
                ``[edge_id, source_id, target_id, timestamp, ...]``.
            y: Accepted only for scikit-learn compatibility and ignored.
        """
        del y
        self._native.fit(_as_feature_matrix(features))

    def partial_fit(self, features: ArrayLike) -> None:
        """Insert ``features`` into the dynamic graph without emitting rows."""
        self._native.partial_fit(_as_feature_matrix(features))

    def transform(self, features: ArrayLike) -> NDArray[np.float64]:
        """Insert a batch and return raw plus graph-derived transaction features.

        The graph is updated before output rows are computed, matching SnapML.
        This is fast and reproduces the paper's 128-row batch protocol, but is
        not strict event-by-event causal within a single batch.
        """
        matrix = _as_feature_matrix(features)
        if matrix.shape[0] == 0:
            return np.empty((0, matrix.shape[1] + _engineered_width(self.params)))
        return self._native.transform(matrix)

    def fit_transform(
        self, features: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Clear state, insert ``features``, and enrich the same batch.

        Args:
            features: Matrix shaped ``(n_edges, n_raw_features)``.
            y: Accepted only for scikit-learn compatibility and ignored.
        """
        del y
        matrix = _as_feature_matrix(features)
        if matrix.shape[0] == 0:
            self._native.fit(matrix)
            return np.empty((0, matrix.shape[1] + _engineered_width(self.params)))
        return self._native.fit_transform(matrix)

    def transform_causal(self, features: ArrayLike) -> NDArray[np.float64]:
        """Enrich timestamp-ordered rows without within-batch future visibility.

        This method intentionally transforms one row at a time, after checking
        non-decreasing timestamps. It is the strict temporal protocol for
        leakage-sensitive evaluation; use ordinary :meth:`transform` for the
        higher-throughput SnapML-compatible batch protocol.
        """
        matrix = _as_feature_matrix(features)
        if matrix.shape[0] > 1 and np.any(np.diff(matrix[:, 3]) < 0.0):
            raise ValueError("transform_causal requires non-decreasing timestamps")
        if matrix.shape[0] == 0:
            return self.transform(matrix)
        return np.vstack(
            [
                self.transform(matrix[index : index + 1])
                for index in range(matrix.shape[0])
            ]
        )

    @property
    def active_edge_count(self) -> int:
        """Return the number of transactions currently retained in memory."""
        return int(self._native.active_edge_count())


def _as_feature_matrix(features: ArrayLike) -> NDArray[np.float64]:
    """Validate and normalize a two-dimensional numerical transaction matrix."""
    matrix = np.ascontiguousarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 4:
        raise ValueError(
            "features must be a 2D float64 matrix with edge_id, source, target, and timestamp columns"
        )
    if not np.isfinite(matrix[:, :4]).all():
        raise ValueError("edge_id, source, target, and timestamp must be finite")
    return matrix


def _build_native(params: Mapping[str, Any]) -> NativeGraphFeaturePreprocessor:
    """Translate public parameter names into the compact native JSON contract."""
    payload = {
        "num_threads": int(params["num_threads"]),
        "time_window": float(params["time_window"]),
        "max_no_edges": int(params["max_no_edges"]),
        "vertex_stats": bool(params["vertex_stats"]),
        "vertex_stats_tw": float(params["vertex_stats_tw"]),
        "vertex_stats_cols": [int(value) for value in params["vertex_stats_cols"]],
        "vertex_stats_feats": [int(value) for value in params["vertex_stats_feats"]],
        "fan": _pattern_payload(params, "fan"),
        "degree": _pattern_payload(params, "degree"),
        "scatter_gather": _pattern_payload(params, "scatter-gather"),
        "temp_cycle": _pattern_payload(params, "temp-cycle"),
        "lc_cycle": _pattern_payload(params, "lc-cycle"),
        "lc_cycle_len": int(params["lc-cycle_len"]),
    }
    return NativeGraphFeaturePreprocessor(json.dumps(payload))


def _pattern_payload(params: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Map one public pattern configuration into Rust's snake-case schema."""
    return {
        "enabled": bool(params[name]),
        "time_window": float(params[f"{name}_tw"]),
        "bins": [int(value) for value in params[f"{name}_bins"]],
    }


def _engineered_width(params: Mapping[str, Any]) -> int:
    """Return the fixed number of appended columns for a configuration."""
    width = sum(
        2 * len(params[f"{name}_bins"]) for name in ("fan", "degree") if params[name]
    )
    width += sum(
        len(params[f"{name}_bins"])
        for name in ("scatter-gather", "temp-cycle", "lc-cycle")
        if params[name]
    )
    if params["vertex_stats"]:
        structural = sum(
            feature in {0, 1, 2} for feature in params["vertex_stats_feats"]
        )
        numeric = sum(
            feature in range(3, 11) for feature in params["vertex_stats_feats"]
        )
        width += 4 * (structural + len(params["vertex_stats_cols"]) * numeric)
    return width


def _default_params() -> dict[str, Any]:
    """Return the GraphFeaturePreprocessor defaults shipped by SnapML 1.17.2."""
    bins = list(range(2, 31))
    return {
        "num_threads": 12,
        "time_window": -1,
        "max_no_edges": -1,
        "vertex_stats": True,
        "vertex_stats_tw": 480 * 3600,
        "vertex_stats_cols": [3],
        "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],
        "fan": True,
        "fan_tw": 12 * 3600,
        "fan_bins": bins,
        "degree": False,
        "degree_tw": 12 * 3600,
        "degree_bins": bins.copy(),
        "scatter-gather": False,
        "scatter-gather_tw": 120 * 3600,
        "scatter-gather_bins": bins.copy(),
        "temp-cycle": False,
        "temp-cycle_tw": 480 * 3600,
        "temp-cycle_bins": bins.copy(),
        "lc-cycle": False,
        "lc-cycle_tw": 240 * 3600,
        "lc-cycle_len": 10,
        "lc-cycle_bins": list(range(2, 11)),
    }
