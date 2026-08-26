//! Read-only graph pattern and account-statistic feature extraction.

use std::collections::HashSet;

use super::{
    config::{Config, PatternConfig},
    state::{GraphState, VertexId},
};

/// Produce every engineered value for one transaction from an immutable graph.
pub(crate) fn engineered_features(
    state: &GraphState,
    row: &[f64],
    config: &Config,
) -> Result<Vec<f64>, String> {
    if row.len() < 4 {
        return Err("features must have [edge_id, source, target, timestamp] columns".into());
    }
    if row[..4].iter().any(|value| !value.is_finite()) {
        return Err("edge ID, source, target, and timestamp must be finite float64 values".into());
    }

    let source = row[1].to_bits();
    let target = row[2].to_bits();
    let edge_id = row[0].to_bits();
    let latest = state.latest_timestamp().unwrap_or(row[3]);
    let mut output = Vec::new();

    if config.fan.enabled {
        append_fan_features(
            &mut output,
            state,
            edge_id,
            source,
            target,
            latest,
            &config.fan,
        );
    }
    if config.degree.enabled {
        append_degree_features(
            &mut output,
            state,
            edge_id,
            source,
            target,
            latest,
            &config.degree,
        );
    }
    if config.scatter_gather.enabled {
        output.extend(scatter_gather_features(
            state,
            source,
            target,
            latest,
            &config.scatter_gather,
        ));
    }
    if config.temp_cycle.enabled {
        output.extend(cycle_features(
            state,
            source,
            target,
            row[3],
            latest,
            &config.temp_cycle,
            None,
            true,
        ));
    }
    if config.lc_cycle.enabled {
        output.extend(cycle_features(
            state,
            source,
            target,
            row[3],
            latest,
            &config.lc_cycle,
            Some(config.lc_cycle_len),
            false,
        ));
    }
    if config.vertex_stats {
        append_vertex_statistics(&mut output, state, row, source, target, latest, config)?;
    }
    Ok(output)
}

/// Append target fan-in and source fan-out as bucketed one-hot features.
fn append_fan_features(
    output: &mut Vec<f64>,
    state: &GraphState,
    edge_id: u64,
    source: VertexId,
    target: VertexId,
    latest: f64,
    config: &PatternConfig,
) {
    let cutoff = latest - config.time_window;
    output.extend(prefix_pattern_histogram(
        state,
        edge_id,
        target,
        false,
        cutoff,
        &config.bins,
        true,
    ));
    output.extend(prefix_pattern_histogram(
        state,
        edge_id,
        source,
        true,
        cutoff,
        &config.bins,
        true,
    ));
}

/// Append target in-degree and source out-degree histogram buckets.
fn append_degree_features(
    output: &mut Vec<f64>,
    state: &GraphState,
    edge_id: u64,
    source: VertexId,
    target: VertexId,
    latest: f64,
    config: &PatternConfig,
) {
    let cutoff = latest - config.time_window;
    output.extend(prefix_pattern_histogram(
        state,
        edge_id,
        target,
        false,
        cutoff,
        &config.bins,
        false,
    ));
    output.extend(prefix_pattern_histogram(
        state,
        edge_id,
        source,
        true,
        cutoff,
        &config.bins,
        false,
    ));
}

/// Encode the growing fan/degree patterns that contain the current edge.
///
/// SnapML records one pattern for every chronological prefix of a vertex's
/// incident edge sequence. A transaction therefore belongs to all prefixes
/// ending at or after its own position, rather than only one final degree.
/// Fans use the first transaction to each distinct neighbour; degrees use every
/// parallel edge. This detail matters for multi-row batches and multigraphs.
fn prefix_pattern_histogram(
    state: &GraphState,
    edge_id: u64,
    vertex: VertexId,
    outgoing: bool,
    cutoff: f64,
    bins: &[usize],
    fan: bool,
) -> Vec<f64> {
    let incident = state.incident_edges(vertex, outgoing, cutoff);
    let mut histogram = vec![0.0; bins.len()];
    if incident.is_empty() {
        return histogram;
    }

    let (current_rank, pattern_count) = if fan {
        fan_rank_and_count(&incident, edge_id)
    } else {
        degree_rank_and_count(&incident, edge_id)
    };
    let Some(rank) = current_rank else {
        return histogram;
    };
    for size in rank.max(2)..=pattern_count {
        increment_bucket(&mut histogram, size, bins);
    }
    histogram
}

/// Return the physical edge rank and degree-pattern size for one edge.
fn degree_rank_and_count(
    incident: &[super::state::IncidentEdge],
    edge_id: u64,
) -> (Option<usize>, usize) {
    (
        incident
            .iter()
            .position(|edge| edge.id == edge_id)
            .map(|index| index + 1),
        incident.len(),
    )
}

/// Return the neighbour-prefix rank and fan-pattern size for one edge.
fn fan_rank_and_count(
    incident: &[super::state::IncidentEdge],
    edge_id: u64,
) -> (Option<usize>, usize) {
    let current = incident.iter().find(|edge| edge.id == edge_id);
    let Some(current) = current else {
        return (None, 0);
    };
    let mut neighbours = HashSet::new();
    let mut rank = None;
    for edge in incident {
        if neighbours.insert(edge.neighbour) && edge.neighbour == current.neighbour {
            rank = Some(neighbours.len());
        }
    }
    (rank, neighbours.len())
}

/// Enumerate scatter-gather patterns in both roles of the current edge.
///
/// The first phase treats `source -> target` as the first leg of
/// `source -> intermediates -> terminal`; the second treats it as the final
/// leg. A valid pattern has at least two distinct intermediate accounts.
fn scatter_gather_features(
    state: &GraphState,
    source: VertexId,
    target: VertexId,
    latest: f64,
    config: &PatternConfig,
) -> Vec<f64> {
    let cutoff = latest - config.time_window;
    let source_out = state.outgoing_neighbors(source, cutoff);
    let target_out = state.outgoing_neighbors(target, cutoff);
    let source_in = state.incoming_neighbors(source, cutoff);
    let target_in = state.incoming_neighbors(target, cutoff);
    let mut histogram = vec![0.0; config.bins.len()];

    for terminal in target_out {
        if terminal == source {
            continue;
        }
        let terminal_in = state.incoming_neighbors(terminal, cutoff);
        record_intersection(&mut histogram, &source_out, &terminal_in, &config.bins);
    }
    for origin in source_in {
        if origin == target {
            continue;
        }
        let origin_out = state.outgoing_neighbors(origin, cutoff);
        record_intersection(&mut histogram, &target_in, &origin_out, &config.bins);
    }
    histogram
}

/// Record one scatter-gather pattern when two neighbor sets share >=2 vertices.
fn record_intersection(
    histogram: &mut [f64],
    left: &[VertexId],
    right: &[VertexId],
    bins: &[usize],
) {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut shared = 0;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                shared += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    if shared >= 2 {
        increment_bucket(histogram, shared, bins);
    }
}

/// Count simple or temporal directed cycles that contain the current edge.
///
/// A simple cycle is anchored on the current `source -> target` transaction so
/// each vertex cycle is visited once. For a temporal cycle, the edge timestamps
/// must be strictly increasing after some rotation of the cycle; equivalently,
/// the circular timestamp sequence has exactly one descent.
#[allow(clippy::too_many_arguments)]
fn cycle_features(
    state: &GraphState,
    source: VertexId,
    target: VertexId,
    edge_timestamp: f64,
    latest: f64,
    config: &PatternConfig,
    maximum_length: Option<usize>,
    temporal: bool,
) -> Vec<f64> {
    if source == target {
        return vec![0.0; config.bins.len()];
    }
    let cutoff = latest - config.time_window;
    let mut histogram = vec![0.0; config.bins.len()];
    let mut path = vec![source, target];
    let mut visited = HashSet::from([source, target]);
    let mut edge_pairs = Vec::new();
    search_cycles(
        state,
        source,
        target,
        edge_timestamp,
        cutoff,
        config.bins.as_slice(),
        maximum_length,
        temporal,
        &mut path,
        &mut visited,
        &mut edge_pairs,
        &mut histogram,
    );
    histogram
}

#[allow(clippy::too_many_arguments)]
fn search_cycles(
    state: &GraphState,
    source: VertexId,
    current: VertexId,
    edge_timestamp: f64,
    cutoff: f64,
    bins: &[usize],
    maximum_length: Option<usize>,
    temporal: bool,
    path: &mut Vec<VertexId>,
    visited: &mut HashSet<VertexId>,
    edge_pairs: &mut Vec<(VertexId, VertexId)>,
    histogram: &mut [f64],
) {
    let current_length = path.len() - 1;
    if maximum_length.is_some_and(|limit| current_length >= limit) {
        return;
    }

    for next in state.outgoing_neighbors(current, cutoff) {
        if next == source {
            let cycle_length = current_length + 1;
            if cycle_length < 2 || maximum_length.is_some_and(|limit| cycle_length > limit) {
                continue;
            }
            edge_pairs.push((current, source));
            let is_temporal =
                !temporal || temporal_timestamps_exist(state, edge_timestamp, edge_pairs, cutoff);
            if is_temporal {
                increment_bucket(histogram, cycle_length, bins);
            }
            edge_pairs.pop();
            continue;
        }
        if visited.contains(&next) {
            continue;
        }

        path.push(next);
        visited.insert(next);
        edge_pairs.push((current, next));
        search_cycles(
            state,
            source,
            next,
            edge_timestamp,
            cutoff,
            bins,
            maximum_length,
            temporal,
            path,
            visited,
            edge_pairs,
            histogram,
        );
        edge_pairs.pop();
        visited.remove(&next);
        path.pop();
    }
}

/// Check whether any choice of parallel-edge timestamps makes a cycle temporal.
fn temporal_timestamps_exist(
    state: &GraphState,
    first_timestamp: f64,
    edge_pairs: &[(VertexId, VertexId)],
    cutoff: f64,
) -> bool {
    let timestamp_options: Vec<_> = edge_pairs
        .iter()
        .map(|(source, target)| state.pair_timestamps(*source, *target, cutoff))
        .collect();
    if timestamp_options.iter().any(Vec::is_empty) {
        return false;
    }
    has_one_circular_descent(&timestamp_options, 0, first_timestamp, first_timestamp, 0)
}

/// Recursively choose one timestamp per edge while allowing exactly one descent.
fn has_one_circular_descent(
    options: &[Vec<f64>],
    index: usize,
    first: f64,
    previous: f64,
    descents: usize,
) -> bool {
    if index == options.len() {
        return descents + usize::from(first <= previous) == 1;
    }
    options[index].iter().copied().any(|timestamp| {
        let next_descents = descents + usize::from(timestamp <= previous);
        next_descents <= 1
            && has_one_circular_descent(options, index + 1, first, timestamp, next_descents)
    })
}

/// Append source/target outgoing/incoming account statistics in SnapML order.
fn append_vertex_statistics(
    output: &mut Vec<f64>,
    state: &GraphState,
    row: &[f64],
    source: VertexId,
    target: VertexId,
    latest: f64,
    config: &Config,
) -> Result<(), String> {
    for column in &config.vertex_stats_cols {
        if *column >= row.len() {
            return Err(format!(
                "vertex_stats_cols contains out-of-range column {column}"
            ));
        }
    }
    let cutoff = latest - config.vertex_stats_tw;
    for (vertex, outgoing) in [
        (source, true),
        (source, false),
        (target, true),
        (target, false),
    ] {
        append_one_vertex_statistics(output, state, vertex, outgoing, cutoff, config);
    }
    Ok(())
}

/// Append requested statistics for one endpoint/direction pair.
fn append_one_vertex_statistics(
    output: &mut Vec<f64>,
    state: &GraphState,
    vertex: VertexId,
    outgoing: bool,
    cutoff: f64,
    config: &Config,
) {
    let (fan, degree) = state.fan_and_degree(vertex, outgoing, cutoff);
    let ratio = if degree == 0 {
        0.0
    } else {
        fan as f64 / degree as f64
    };
    append_selected_structural_statistics(output, &config.vertex_stats_feats, fan, degree, ratio);

    for column in &config.vertex_stats_cols {
        let values = state.incident_values(vertex, outgoing, cutoff, *column);
        let stats = NumericStatistics::from_values(&values);
        append_selected_numeric_statistics(output, &config.vertex_stats_feats, stats);
    }
}

/// Add selected fan, degree, and ratio values in their fixed public order.
fn append_selected_structural_statistics(
    output: &mut Vec<f64>,
    requested: &[usize],
    fan: usize,
    degree: usize,
    ratio: f64,
) {
    for (feature, value) in [(0, fan as f64), (1, degree as f64), (2, ratio)] {
        if requested.contains(&feature) {
            output.push(value);
        }
    }
}

/// Add selected numeric statistics in the same order as the SnapML wrapper.
fn append_selected_numeric_statistics(
    output: &mut Vec<f64>,
    requested: &[usize],
    stats: NumericStatistics,
) {
    for (feature, value) in [
        (3, stats.mean),
        (4, stats.sum),
        (5, stats.minimum),
        (6, stats.maximum),
        (7, stats.median),
        (8, stats.variance),
        (9, stats.skew),
        (10, stats.kurtosis),
    ] {
        if requested.contains(&feature) {
            output.push(value);
        }
    }
}

/// Compute population moments; zero variance has zero skew and kurtosis.
#[derive(Clone, Copy)]
struct NumericStatistics {
    sum: f64,
    mean: f64,
    minimum: f64,
    maximum: f64,
    median: f64,
    variance: f64,
    skew: f64,
    kurtosis: f64,
}

impl NumericStatistics {
    fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::zero();
        }
        let count = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let mean = sum / count;
        let (second, third, fourth) = values.iter().fold((0.0, 0.0, 0.0), |moments, value| {
            let centered = value - mean;
            (
                moments.0 + centered.powi(2),
                moments.1 + centered.powi(3),
                moments.2 + centered.powi(4),
            )
        });
        let variance = second / count;
        let skew = if variance == 0.0 {
            0.0
        } else {
            (third / count) / variance.powf(1.5)
        };
        let kurtosis = if variance == 0.0 {
            0.0
        } else {
            (fourth / count) / variance.powi(2)
        };
        let mut ordered = values.to_vec();
        ordered.sort_by(f64::total_cmp);
        let middle = ordered.len() / 2;
        let median = if ordered.len() % 2 == 0 {
            (ordered[middle - 1] + ordered[middle]) / 2.0
        } else {
            ordered[middle]
        };
        Self {
            sum,
            mean,
            minimum: ordered[0],
            maximum: ordered[ordered.len() - 1],
            median,
            variance,
            skew,
            kurtosis,
        }
    }

    fn zero() -> Self {
        Self {
            sum: 0.0,
            mean: 0.0,
            minimum: 0.0,
            maximum: 0.0,
            median: 0.0,
            variance: 0.0,
            skew: 0.0,
            kurtosis: 0.0,
        }
    }
}

/// Return a one-hot histogram bucket, with zero below the first bucket.
#[cfg(test)]
fn bucket_one(value: usize, bins: &[usize]) -> Vec<f64> {
    let mut histogram = vec![0.0; bins.len()];
    increment_bucket(&mut histogram, value, bins);
    histogram
}

/// Increment the bucket containing `value`; the final bucket is open-ended.
fn increment_bucket(histogram: &mut [f64], value: usize, bins: &[usize]) {
    let Some(index) = bins.iter().rposition(|lower_bound| value >= *lower_bound) else {
        return;
    };
    histogram[index] += 1.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_uses_an_open_ended_final_bucket() {
        assert_eq!(bucket_one(1, &[2, 3, 4]), vec![0.0, 0.0, 0.0]);
        assert_eq!(bucket_one(2, &[2, 3, 4]), vec![1.0, 0.0, 0.0]);
        assert_eq!(bucket_one(5, &[2, 3, 4]), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn temporal_cycles_require_exactly_one_circular_descent() {
        assert!(has_one_circular_descent(
            &[vec![2.0], vec![3.0]],
            0,
            1.0,
            1.0,
            0
        ));
        assert!(!has_one_circular_descent(
            &[vec![3.0], vec![2.0]],
            0,
            1.0,
            1.0,
            0
        ));
    }

    #[test]
    fn numeric_statistics_match_population_moments() {
        let stats = NumericStatistics::from_values(&[1.0, 2.0, 3.0]);
        assert_eq!(stats.sum, 6.0);
        assert_eq!(stats.mean, 2.0);
        assert_eq!(stats.variance, 2.0 / 3.0);
        assert_eq!(stats.kurtosis, 1.5);
    }
}
