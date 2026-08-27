"""Behavioural tests for the native tabular Graph Feature Preprocessor."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from snapml import GraphFeaturePreprocessor as SnapMLGraphFeaturePreprocessor

from amlgraphx.tabular import GraphFeaturePreprocessor


def _params(**updates: object) -> dict[str, object]:
    """Return a compact all-feature configuration used by reference tests."""
    params: dict[str, object] = {
        "num_threads": 1,
        "time_window": 20,
        "max_no_edges": -1,
        "vertex_stats": True,
        "vertex_stats_tw": 15,
        "vertex_stats_cols": [4],
        "vertex_stats_feats": list(range(11)),
        "fan": True,
        "fan_tw": 10,
        "fan_bins": [2, 3, 4, 5],
        "degree": True,
        "degree_tw": 10,
        "degree_bins": [2, 3, 4, 5],
        "scatter-gather": True,
        "scatter-gather_tw": 10,
        "scatter-gather_bins": [2, 3, 4, 5],
        "temp-cycle": True,
        "temp-cycle_tw": 15,
        "temp-cycle_bins": [2, 3, 4, 5],
        "lc-cycle": True,
        "lc-cycle_tw": 15,
        "lc-cycle_len": 5,
        "lc-cycle_bins": [2, 3, 4, 5],
    }
    params.update(updates)
    return params


def _transform_with(
    preprocessor_type: type[object], features: np.ndarray, params: dict[str, object]
) -> np.ndarray:
    """Configure either implementation and return its enriched matrix."""
    preprocessor = preprocessor_type()
    preprocessor.set_params(params)  # type: ignore[attr-defined]
    return preprocessor.transform(features)  # type: ignore[no-any-return, attr-defined]


def test_matches_snapml_for_every_feature_family() -> None:
    """All five patterns and 11 vertex statistics match SnapML on a small graph."""
    features = np.array(
        [
            [1, 1, 2, 1, 10],
            [2, 2, 3, 2, 20],
            [3, 3, 1, 3, 30],
            [4, 1, 4, 4, 40],
            [5, 4, 3, 5, 50],
            [6, 1, 5, 6, 60],
            [7, 5, 3, 7, 70],
        ],
        dtype=np.float64,
    )
    params = _params()

    expected = _transform_with(SnapMLGraphFeaturePreprocessor, features, params)
    actual = _transform_with(GraphFeaturePreprocessor, features, params)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_matches_snapml_for_random_simple_multigraph_free_batches() -> None:
    """Reference parity covers overlapping stars, cycles, and scatter-gathers."""
    params = _params()
    for seed in range(20):
        generator = np.random.default_rng(seed)
        pairs: list[tuple[int, int]] = []
        while len(pairs) < 9:
            source, target = generator.integers(1, 6, size=2)
            pair = int(source), int(target)
            if source != target and pair not in pairs:
                pairs.append(pair)
        features = np.array(
            [
                [index + 1, source, target, index + 1, generator.integers(1, 20)]
                for index, (source, target) in enumerate(pairs)
            ],
            dtype=np.float64,
        )

        expected = _transform_with(SnapMLGraphFeaturePreprocessor, features, params)
        actual = _transform_with(GraphFeaturePreprocessor, features, params)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_matches_snapml_for_parallel_edges_with_equal_timestamps() -> None:
    """Concurrent multigraph events use SnapML's timestamp-group semantics."""
    features = np.array(
        [
            [1, 1, 9, 1, 10],
            [2, 2, 9, 2, 20],
            [3, 1, 9, 2, 30],
            [4, 3, 9, 2, 40],
        ],
        dtype=np.float64,
    )
    params = _params(
        **{
            "scatter-gather": False,
            "temp-cycle": False,
            "lc-cycle": False,
        }
    )

    expected = _transform_with(SnapMLGraphFeaturePreprocessor, features, params)
    actual = _transform_with(GraphFeaturePreprocessor, features, params)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_matches_snapml_for_per_event_pattern_windows() -> None:
    """Pattern windows follow each event, not the largest batch timestamp."""
    features = np.array(
        [[1, 1, 9, 1], [2, 2, 9, 9], [3, 3, 9, 11], [4, 4, 9, 19]],
        dtype=np.float64,
    )
    params = _params(
        vertex_stats=False,
        fan_tw=10,
        degree_tw=10,
        **{
            "scatter-gather": False,
            "temp-cycle": False,
            "lc-cycle": False,
        },
    )

    expected = _transform_with(SnapMLGraphFeaturePreprocessor, features, params)
    actual = _transform_with(GraphFeaturePreprocessor, features, params)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_matches_snapml_for_equal_timestamps_across_batches() -> None:
    """Earlier batches contribute graph state but not a second concurrent emission."""
    history = np.array([[1, 2, 6, 13], [2, 5, 2, 14]], dtype=np.float64)
    batch = np.array([[3, 2, 5, 13], [4, 3, 4, 15]], dtype=np.float64)
    params = _params(
        vertex_stats=False,
        **{
            "scatter-gather": False,
            "temp-cycle": False,
            "lc-cycle": False,
        },
    )

    expected = SnapMLGraphFeaturePreprocessor()
    actual = GraphFeaturePreprocessor()
    expected.set_params(params)
    actual.set_params(params)
    expected.transform(history)
    actual.transform(history)

    np.testing.assert_allclose(
        actual.transform(batch), expected.transform(batch), rtol=1e-12, atol=1e-12
    )


def test_transform_causal_excludes_later_rows_in_its_input_batch() -> None:
    """Strict causal mode never lets the first star edge observe later rows."""
    features = np.array([[1, 1, 2, 1], [2, 1, 3, 2], [3, 1, 4, 3]], dtype=np.float64)
    params = _params(
        vertex_stats=False,
        degree=False,
        **{"scatter-gather": False, "temp-cycle": False, "lc-cycle": False},
    )

    batch = GraphFeaturePreprocessor()
    batch.set_params(params)
    batch_output = batch.transform(features)

    causal = GraphFeaturePreprocessor()
    causal.set_params(params)
    causal_output = causal.transform_causal(features)

    assert np.count_nonzero(batch_output[0, 4:]) > 0
    assert np.count_nonzero(causal_output[0, 4:]) == 0
    assert np.count_nonzero(causal_output[1, 4:]) == 1


def test_causal_mode_rejects_retained_future_history() -> None:
    """Strict causal mode cannot evaluate an event before retained state."""
    preprocessor = GraphFeaturePreprocessor()
    preprocessor.partial_fit(np.array([[1, 1, 2, 10]], dtype=float))

    with pytest.raises(ValueError, match="later than retained history"):
        preprocessor.transform_causal(np.array([[2, 2, 3, 9]], dtype=float))


def test_vertex_statistics_respect_their_own_time_window() -> None:
    """A longer fan window must not keep stale account statistics alive."""
    preprocessor = GraphFeaturePreprocessor()
    preprocessor.set_params(
        _params(
            vertex_stats_cols=[],
            vertex_stats_feats=[1],
            vertex_stats_tw=10,
            time_window=-1,
            fan=True,
            fan_tw=100,
            fan_bins=[2],
            degree=False,
            **{"scatter-gather": False, "temp-cycle": False, "lc-cycle": False},
        )
    )

    output = preprocessor.transform(
        np.array([[1, 1, 2, 0], [2, 1, 3, 50]], dtype=float)
    )

    assert output[0, -4] == 2
    assert output[1, -4] == 1


def test_duplicate_vertex_statistic_features_are_rejected() -> None:
    """Output width and native feature emission stay one-to-one."""
    preprocessor = GraphFeaturePreprocessor()

    with pytest.raises(ValueError, match="must not contain duplicates"):
        preprocessor.set_params(_params(vertex_stats_feats=[0, 0]))


def test_time_window_maximum_edge_count_and_duplicate_edge_ids() -> None:
    """Dynamic retention removes the inclusive lower boundary and duplicates."""
    preprocessor = GraphFeaturePreprocessor()
    preprocessor.set_params(
        _params(
            time_window=10,
            max_no_edges=2,
            vertex_stats=False,
            fan=False,
            degree=False,
            **{"scatter-gather": False, "temp-cycle": False, "lc-cycle": False},
        )
    )
    preprocessor.partial_fit(np.array([[1, 1, 2, 1], [2, 2, 3, 2]], dtype=float))
    preprocessor.partial_fit(np.array([[1, 9, 9, 3]], dtype=float))
    assert preprocessor.active_edge_count == 2

    preprocessor.partial_fit(np.array([[3, 3, 4, 11]], dtype=float))
    assert preprocessor.active_edge_count == 2


def test_parallel_workers_and_python_threads_are_deterministic() -> None:
    """Independent instances have stable results with no shared mutable state."""
    features = np.array(
        [
            [index, index % 7, (index * 3) % 7, index, index % 11]
            for index in range(1, 129)
        ],
        dtype=np.float64,
    )
    params = _params(num_threads=4)

    def run() -> np.ndarray:
        return _transform_with(GraphFeaturePreprocessor, features, params)

    expected = run()
    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(executor.map(lambda _: run(), range(2)))

    for actual in concurrent:
        np.testing.assert_array_equal(actual, expected)


def test_serial_and_parallel_feature_extraction_match_exactly() -> None:
    """Read-only Rayon workers do not alter row order or floating-point results."""
    features = np.array(
        [
            [index, index % 11, (index * 5) % 11, index, index % 17]
            for index in range(1, 129)
        ],
        dtype=np.float64,
    )

    serial = _transform_with(GraphFeaturePreprocessor, features, _params(num_threads=1))
    parallel = _transform_with(
        GraphFeaturePreprocessor, features, _params(num_threads=4)
    )

    np.testing.assert_array_equal(parallel, serial)


def test_causal_mode_rejects_non_strict_timestamps() -> None:
    """Causal processing fails early instead of silently creating future leakage."""
    preprocessor = GraphFeaturePreprocessor()
    with pytest.raises(ValueError, match="strictly increasing"):
        preprocessor.transform_causal(
            np.array([[1, 1, 2, 2], [2, 2, 3, 1]], dtype=float)
        )


def test_empty_batches_preserve_snapml_output_width() -> None:
    """Empty transforms still expose the configured downstream schema."""
    params = _params(
        vertex_stats=False,
        degree=False,
        **{"scatter-gather": False, "temp-cycle": False, "lc-cycle": False},
    )
    features = np.empty((0, 4), dtype=np.float64)

    expected = _transform_with(SnapMLGraphFeaturePreprocessor, features, params)
    actual = _transform_with(GraphFeaturePreprocessor, features, params)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
