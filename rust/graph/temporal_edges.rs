//! Parallel temporal successor matching for transaction-as-node graphs.

use std::panic::{AssertUnwindSafe, catch_unwind};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use rayon::prelude::*;

type PyEdgeArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

#[derive(Default)]
struct EdgeArrays {
    sources: Vec<i64>,
    targets: Vec<i64>,
    deltas: Vec<i64>,
}

impl EdgeArrays {
    fn append(&mut self, other: &mut Self) {
        self.sources.append(&mut other.sources);
        self.targets.append(&mut other.targets);
        self.deltas.append(&mut other.deltas);
    }
}

/// Return sorted-row positions for every valid temporal continuation edge.
///
/// Inputs must be sorted by timestamp and source/target codes must share one
/// contiguous categorical namespace. The interval is `(timestamp, timestamp +
/// delta]`: equal timestamps are excluded and the upper boundary is included.
#[pyfunction]
pub(crate) fn temporal_edge_indices<'py>(
    py: Python<'py>,
    source_codes: PyReadonlyArray1<'_, u32>,
    target_codes: PyReadonlyArray1<'_, u32>,
    timestamps_ns: PyReadonlyArray1<'_, i64>,
    delta_ns: i128,
) -> PyResult<PyEdgeArrays<'py>> {
    let sources = contiguous_copy(source_codes, "source_codes")?;
    let targets = contiguous_copy(target_codes, "target_codes")?;
    let timestamps = contiguous_copy(timestamps_ns, "timestamps_ns")?;
    validate(&sources, &targets, &timestamps, delta_ns).map_err(PyValueError::new_err)?;

    let result = py.allow_threads(move || {
        catch_unwind(AssertUnwindSafe(|| {
            build_edges(&sources, &targets, &timestamps, delta_ns)
        }))
        .map_err(|_| "native temporal edge worker panicked".to_owned())
    });
    let result = result.map_err(PyRuntimeError::new_err)?;
    let result = result.map_err(PyValueError::new_err)?;

    Ok((
        PyArray1::from_vec(py, result.sources),
        PyArray1::from_vec(py, result.targets),
        PyArray1::from_vec(py, result.deltas),
    ))
}

fn contiguous_copy<T: numpy::Element + Copy>(
    values: PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<Vec<T>> {
    values
        .as_slice()
        .map(<[T]>::to_vec)
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

fn validate(
    sources: &[u32],
    targets: &[u32],
    timestamps: &[i64],
    delta_ns: i128,
) -> Result<(), String> {
    if sources.len() != targets.len() || sources.len() != timestamps.len() {
        return Err("source_codes, target_codes, and timestamps_ns must have equal lengths".into());
    }
    if delta_ns < 0 {
        return Err("delta_ns must be non-negative".into());
    }
    if timestamps.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("timestamps_ns must be sorted in non-decreasing order".into());
    }
    if let Some(maximum) = sources.iter().chain(targets).copied().max() {
        let maximum = maximum as usize;
        let code_limit = sources.len().saturating_mul(2);
        if maximum >= code_limit {
            return Err("account codes must use one compact shared namespace".into());
        }
    }
    Ok(())
}

fn build_edges(
    sources: &[u32],
    targets: &[u32],
    timestamps: &[i64],
    delta_ns: i128,
) -> Result<EdgeArrays, String> {
    if sources.is_empty() || delta_ns == 0 {
        return Ok(EdgeArrays::default());
    }

    let account_count = sources
        .iter()
        .chain(targets)
        .copied()
        .max()
        .map_or(0, |maximum| maximum as usize + 1);
    let mut outgoing = vec![Vec::new(); account_count];
    for (position, &account) in sources.iter().enumerate() {
        outgoing[account as usize].push(position);
    }

    // Small inputs stay serial; larger inputs use bounded Rayon workers. Each
    // chunk owns its vectors, so there are no locks or shared output writes.
    if sources.len() < 4_096 {
        return build_chunk(0, targets, &outgoing, timestamps, delta_ns);
    }
    let worker_count = rayon::current_num_threads().max(1);
    let task_count = worker_count.saturating_mul(4).max(1);
    let chunk_size = sources.len().div_ceil(task_count).max(4_096);
    let chunks = targets
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, target_chunk)| {
            build_chunk(
                chunk_index * chunk_size,
                target_chunk,
                &outgoing,
                timestamps,
                delta_ns,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let edge_count = chunks.iter().map(|chunk| chunk.sources.len()).sum();
    let mut result = EdgeArrays {
        sources: Vec::with_capacity(edge_count),
        targets: Vec::with_capacity(edge_count),
        deltas: Vec::with_capacity(edge_count),
    };
    for mut chunk in chunks {
        result.append(&mut chunk);
    }
    Ok(result)
}

fn build_chunk(
    offset: usize,
    target_codes: &[u32],
    outgoing: &[Vec<usize>],
    timestamps: &[i64],
    delta_ns: i128,
) -> Result<EdgeArrays, String> {
    let mut result = EdgeArrays::default();
    for (local_position, &target_code) in target_codes.iter().enumerate() {
        let source_position = offset + local_position;
        let timestamp = timestamps[source_position];
        let upper = (timestamp as i128)
            .saturating_add(delta_ns)
            .min(i64::MAX as i128) as i64;
        let candidates = &outgoing[target_code as usize];
        let start = candidates.partition_point(|&position| timestamps[position] <= timestamp);
        let end = candidates.partition_point(|&position| timestamps[position] <= upper);

        result.sources.reserve(end - start);
        result.targets.reserve(end - start);
        result.deltas.reserve(end - start);
        for &target_position in &candidates[start..end] {
            result.sources.push(source_position as i64);
            result.targets.push(target_position as i64);
            let duration = timestamps[target_position]
                .checked_sub(timestamp)
                .ok_or("time delta exceeds the supported int64 nanosecond range")?;
            result.deltas.push(duration);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges(
        sources: &[u32],
        targets: &[u32],
        timestamps: &[i64],
        delta: i128,
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        validate(sources, targets, timestamps, delta).unwrap();
        let result = build_edges(sources, targets, timestamps, delta).unwrap();
        (result.sources, result.targets, result.deltas)
    }

    #[test]
    fn temporal_boundaries_and_order_are_exact() {
        assert_eq!(edges(&[], &[], &[], 10), (vec![], vec![], vec![]));
        assert_eq!(edges(&[0], &[1], &[0], 10), (vec![], vec![], vec![]));
        assert_eq!(
            edges(&[0, 1, 1], &[1, 2, 3], &[0, 5, 10], 10),
            (vec![0, 0], vec![1, 2], vec![5, 10],)
        );
        assert_eq!(
            edges(&[0, 1], &[1, 2], &[0, 11], 10),
            (vec![], vec![], vec![])
        );
        assert_eq!(
            edges(&[0, 1], &[1, 2], &[5, 5], 10),
            (vec![], vec![], vec![])
        );
        assert_eq!(
            edges(&[0, 1], &[1, 2], &[0, 1], 0),
            (vec![], vec![], vec![])
        );
    }

    #[test]
    fn accounts_duplicates_and_overflow_are_preserved() {
        assert_eq!(
            edges(&[0, 2, 1], &[1, 3, 4], &[0, 2, 3], 10),
            (vec![0], vec![2], vec![3],)
        );
        assert_eq!(
            edges(&[0, 1, 1], &[1, 2, 2], &[0, 1, 1], 2),
            (vec![0, 0], vec![1, 2], vec![1, 1],)
        );
        assert_eq!(
            edges(&[0, 1], &[1, 2], &[i64::MAX - 1, i64::MAX], i128::MAX),
            (vec![0], vec![1], vec![1]),
        );
    }

    #[test]
    fn unrepresentable_duration_is_rejected() {
        assert_eq!(
            build_edges(&[0, 1], &[1, 2], &[i64::MIN, i64::MAX], i128::MAX,)
                .err()
                .as_deref(),
            Some("time delta exceeds the supported int64 nanosecond range"),
        );
    }

    #[test]
    fn invalid_inputs_are_rejected_without_panics() {
        assert!(validate(&[0], &[], &[0], 1).is_err());
        assert!(validate(&[0, 1], &[1, 2], &[1, 0], 1).is_err());
        assert!(validate(&[0], &[1], &[0], -1).is_err());
        assert!(validate(&[99], &[0], &[0], 1).is_err());
    }

    #[test]
    fn parallel_runs_are_deterministic() {
        let timestamps = (0..20_000).map(i64::from).collect::<Vec<_>>();
        let sources = (0..20_000)
            .map(|value| (value % 8) as u32)
            .collect::<Vec<_>>();
        let targets = (0..20_000)
            .map(|value| ((value + 1) % 8) as u32)
            .collect::<Vec<_>>();
        let expected = edges(&sources, &targets, &timestamps, 25);
        for _ in 0..4 {
            assert_eq!(edges(&sources, &targets, &timestamps, 25), expected);
        }
    }
}
