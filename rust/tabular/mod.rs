//! Native kernels for tabular AML feature extraction.

mod config;
mod features;
mod preprocessor;
mod state;

use pyo3::prelude::*;

/// Register the tabular native API with the private extension module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<preprocessor::NativeGraphFeaturePreprocessor>()?;
    Ok(())
}
