//! Native graph-construction kernels.

mod temporal_edges;

use pyo3::prelude::*;

/// Register private graph functions with the AMLGraphX extension module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        temporal_edges::temporal_edge_indices,
        module
    )?)?;
    Ok(())
}
