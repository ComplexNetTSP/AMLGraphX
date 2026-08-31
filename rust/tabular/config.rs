//! Configuration and validation for the Graph Feature Preprocessor kernel.

use serde::Deserialize;

/// Pattern configuration shared by the five supported pattern families.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct PatternConfig {
    pub(crate) enabled: bool,
    pub(crate) time_window: f64,
    pub(crate) bins: Vec<usize>,
}

/// Serializable configuration passed from the Python compatibility wrapper.
///
/// The field names intentionally match SnapML's public parameter names after
/// Python converts hyphenated names into this structured representation.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    pub(crate) num_threads: usize,
    pub(crate) time_window: f64,
    pub(crate) max_no_edges: isize,
    pub(crate) vertex_stats: bool,
    pub(crate) vertex_stats_tw: f64,
    pub(crate) vertex_stats_cols: Vec<usize>,
    pub(crate) vertex_stats_feats: Vec<usize>,
    pub(crate) fan: PatternConfig,
    pub(crate) degree: PatternConfig,
    pub(crate) scatter_gather: PatternConfig,
    pub(crate) temp_cycle: PatternConfig,
    pub(crate) lc_cycle: PatternConfig,
    pub(crate) lc_cycle_len: usize,
}

impl Config {
    /// Parse a Python-owned JSON configuration and reject unsafe values early.
    pub(crate) fn from_json(value: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(value)
            .map_err(|error| format!("invalid graph feature configuration: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Return the retention window for the dynamic graph.
    pub(crate) fn graph_time_window(&self) -> f64 {
        if self.time_window >= 0.0 {
            return self.time_window;
        }

        let mut maximum: f64 = 0.0;
        for pattern in [
            &self.fan,
            &self.degree,
            &self.scatter_gather,
            &self.temp_cycle,
            &self.lc_cycle,
        ] {
            if pattern.enabled {
                maximum = maximum.max(pattern.time_window);
            }
        }
        if self.vertex_stats {
            maximum = maximum.max(self.vertex_stats_tw);
        }
        maximum
    }

    /// Bound the private Rayon pool to the machine's available CPUs.
    pub(crate) fn worker_count(&self) -> usize {
        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        self.num_threads.max(1).min(available)
    }

    /// Validate numerical windows, histogram bins, and feature selections.
    fn validate(&self) -> Result<(), String> {
        if !self.time_window.is_finite() || self.time_window < -1.0 {
            return Err("time_window must be -1 or a finite non-negative value".into());
        }
        if self.max_no_edges == 0 || self.max_no_edges < -1 {
            return Err("max_no_edges must be -1 or a positive integer".into());
        }
        if !self.vertex_stats_tw.is_finite() || self.vertex_stats_tw < 0.0 {
            return Err("vertex_stats_tw must be finite and non-negative".into());
        }
        if self.vertex_stats_feats.iter().any(|feature| *feature > 10) {
            return Err("vertex_stats_feats values must be in [0, 10]".into());
        }
        if self
            .vertex_stats_feats
            .iter()
            .enumerate()
            .any(|(index, feature)| self.vertex_stats_feats[index + 1..].contains(feature))
        {
            return Err("vertex_stats_feats must not contain duplicates".into());
        }
        for (name, pattern) in [
            ("fan", &self.fan),
            ("degree", &self.degree),
            ("scatter-gather", &self.scatter_gather),
            ("temp-cycle", &self.temp_cycle),
            ("lc-cycle", &self.lc_cycle),
        ] {
            if !pattern.time_window.is_finite() || pattern.time_window < 0.0 {
                return Err(format!("{name}_tw must be finite and non-negative"));
            }
            if pattern.enabled {
                validate_bins(name, &pattern.bins)?;
            }
        }
        if self.lc_cycle.enabled && self.lc_cycle_len < 2 {
            return Err("lc_cycle_len must be at least 2 when lc-cycle is enabled".into());
        }
        Ok(())
    }
}

/// Ensure histogram buckets have a deterministic, non-overlapping meaning.
fn validate_bins(name: &str, bins: &[usize]) -> Result<(), String> {
    if bins.is_empty() {
        return Err(format!(
            "{name}_bins must not be empty when the pattern is enabled"
        ));
    }
    if bins.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(format!("{name}_bins must be strictly increasing"));
    }
    Ok(())
}
