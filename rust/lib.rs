use pyo3::prelude::*;

mod tabular;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    tabular::register(m)
}
