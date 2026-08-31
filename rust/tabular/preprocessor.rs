//! PyO3 bridge for the private, thread-safe tabular graph feature engine.

use std::panic::{AssertUnwindSafe, catch_unwind};

use numpy::{PyArray2, PyReadonlyArray2, ndarray::Array2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};

use super::{config::Config, features::engineered_features, state::GraphState};

/// Stateful native backend used by `amlgraphx.tabular.GraphFeaturePreprocessor`.
///
/// Inserts and eviction are serial and exclusively owned by this object. After
/// a `transform` batch has finished updating the graph, each row is evaluated
/// independently in a private Rayon pool against immutable state. No mutexes,
/// channels, global worker pool, or shared mutable output buffer are used.
#[pyclass(module = "amlgraphx._native._core", unsendable)]
pub(crate) struct NativeGraphFeaturePreprocessor {
    config: Config,
    state: GraphState,
    pool: ThreadPool,
}

#[pymethods]
impl NativeGraphFeaturePreprocessor {
    /// Construct a new empty preprocessor from validated Python JSON settings.
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let config = Config::from_json(config_json).map_err(PyValueError::new_err)?;
        let pool = build_pool(&config)?;
        Ok(Self {
            config,
            state: GraphState::default(),
            pool,
        })
    }

    /// Replace graph state with edges from `features` without generating output.
    fn fit(&mut self, features: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let rows = copy_rows(features)?;
        self.state.clear();
        self.state
            .insert_batch(rows, &self.config)
            .map_err(PyValueError::new_err)
    }

    /// Append `features` to the in-memory graph without generating output.
    fn partial_fit(&mut self, features: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let rows = copy_rows(features)?;
        self.state
            .insert_batch(rows, &self.config)
            .map_err(PyValueError::new_err)
    }

    /// Insert a batch, then append graph features to every input transaction.
    fn transform<'py>(
        &mut self,
        py: Python<'py>,
        features: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = copy_rows(features)?;
        let batch_start_sequence = self.state.next_sequence();
        self.state
            .insert_batch(rows.iter().cloned(), &self.config)
            .map_err(PyValueError::new_err)?;
        let width = rows.first().map_or(0, Vec::len);
        let output = py.allow_threads(|| self.parallel_transform(&rows, batch_start_sequence));
        let output = output.map_err(PyRuntimeError::new_err)?;
        to_numpy(py, output, width)
    }

    /// Clear state, insert `features`, and enrich the same batch.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        features: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = copy_rows(features)?;
        self.state.clear();
        let batch_start_sequence = self.state.next_sequence();
        self.state
            .insert_batch(rows.iter().cloned(), &self.config)
            .map_err(PyValueError::new_err)?;
        let width = rows.first().map_or(0, Vec::len);
        let output = py.allow_threads(|| self.parallel_transform(&rows, batch_start_sequence));
        let output = output.map_err(PyRuntimeError::new_err)?;
        to_numpy(py, output, width)
    }

    /// Return the active dynamic-graph edge count for diagnostics and tests.
    fn active_edge_count(&self) -> usize {
        self.state.edge_count()
    }

    /// Return the newest retained event time for strict causal validation.
    fn latest_timestamp(&self) -> Option<f64> {
        self.state.latest_timestamp()
    }
}

impl NativeGraphFeaturePreprocessor {
    fn parallel_transform(
        &self,
        rows: &[Vec<f64>],
        batch_start_sequence: u64,
    ) -> Result<Vec<Vec<f64>>, String> {
        let computation = catch_unwind(AssertUnwindSafe(|| {
            self.pool.install(|| {
                rows.par_iter()
                    .map(|row| {
                        let mut enriched = row.clone();
                        enriched.extend(engineered_features(
                            &self.state,
                            row,
                            batch_start_sequence,
                            &self.config,
                        )?);
                        Ok(enriched)
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
        }));
        computation.map_err(|_| "native graph feature worker panicked".to_owned())?
    }
}

/// Copy a NumPy matrix before releasing the GIL for native parallel work.
fn copy_rows(features: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vec<f64>>> {
    let view = features.as_array();
    if view.ncols() < 4 {
        return Err(PyValueError::new_err(
            "features must have [edge_id, source, target, timestamp] columns",
        ));
    }
    Ok(view.outer_iter().map(|row| row.to_vec()).collect())
}

/// Create a contiguous two-dimensional NumPy result without shared writes.
fn to_numpy<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f64>>,
    raw_width: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let row_count = rows.len();
    let width = rows.first().map_or(raw_width, Vec::len);
    let values: Vec<f64> = rows.into_iter().flatten().collect();
    let matrix = Array2::from_shape_vec((row_count, width), values)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, matrix))
}

/// Build one bounded pool per preprocessor, avoiding Rayon global-pool contention.
fn build_pool(config: &Config) -> PyResult<ThreadPool> {
    ThreadPoolBuilder::new()
        .num_threads(config.worker_count())
        .build()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
