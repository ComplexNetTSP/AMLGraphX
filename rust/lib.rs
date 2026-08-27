use pyo3::prelude::*;

mod graph;
mod tabular;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    graph::register(m)?;
    tabular::register(m)
}
