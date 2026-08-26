//! Deterministic dynamic multigraph storage used by the native GFP kernel.

use std::collections::{BTreeMap, HashMap, VecDeque};

use ordered_float::OrderedFloat;

use super::config::Config;

pub(crate) type EdgeId = u64;
pub(crate) type VertexId = u64;

/// One transaction kept in the active dynamic graph.
#[derive(Clone, Debug)]
pub(crate) struct Edge {
    pub(crate) id: EdgeId,
    pub(crate) source: VertexId,
    pub(crate) target: VertexId,
    pub(crate) timestamp: f64,
    sequence: u64,
    pub(crate) values: Vec<f64>,
}

/// One edge incident to a vertex, ordered deterministically by transaction time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IncidentEdge {
    pub(crate) id: EdgeId,
    pub(crate) neighbour: VertexId,
    pub(crate) timestamp: f64,
    pub(crate) sequence: u64,
}

type NeighbourEdges = HashMap<VertexId, VecDeque<EdgeId>>;

/// A mutable graph used only while an API call updates its transaction state.
///
/// Feature extraction never mutates this type. `transform` first completes all
/// inserts and removals, then lets Rayon read this graph concurrently. That
/// ownership boundary removes the need for locks and makes output ordering
/// deterministic.
#[derive(Default)]
pub(crate) struct GraphState {
    edges: HashMap<EdgeId, Edge>,
    outgoing: HashMap<VertexId, NeighbourEdges>,
    incoming: HashMap<VertexId, NeighbourEdges>,
    by_time: BTreeMap<OrderedFloat<f64>, Vec<EdgeId>>,
    insertion_order: VecDeque<EdgeId>,
    next_sequence: u64,
}

impl GraphState {
    /// Clear all graph state before a new `fit` call or parameter update.
    pub(crate) fn clear(&mut self) {
        self.edges.clear();
        self.outgoing.clear();
        self.incoming.clear();
        self.by_time.clear();
        self.insertion_order.clear();
        self.next_sequence = 0;
    }

    /// Insert one raw transaction unless its active edge ID already exists.
    pub(crate) fn insert(&mut self, values: Vec<f64>, config: &Config) -> Result<(), String> {
        let mut edge = Edge::from_values(values)?;
        if self.edges.contains_key(&edge.id) {
            return Ok(());
        }

        edge.sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        let timestamp = edge.timestamp;
        self.add_adjacency(&edge);
        self.by_time
            .entry(OrderedFloat(timestamp))
            .or_default()
            .push(edge.id);
        self.insertion_order.push_back(edge.id);
        self.edges.insert(edge.id, edge);

        self.prune_time(timestamp, config.graph_time_window());
        self.prune_count(config.max_no_edges);
        Ok(())
    }

    /// Insert an ordered batch. State mutation deliberately remains serial.
    pub(crate) fn insert_batch(
        &mut self,
        rows: impl IntoIterator<Item = Vec<f64>>,
        config: &Config,
    ) -> Result<(), String> {
        for row in rows {
            self.insert(row, config)?;
        }
        Ok(())
    }

    /// Return the largest timestamp among active edges.
    pub(crate) fn latest_timestamp(&self) -> Option<f64> {
        self.by_time.iter().rev().find_map(|(time, ids)| {
            ids.iter()
                .any(|id| self.edges.contains_key(id))
                .then_some(time.into_inner())
        })
    }

    /// Return active outgoing neighbors with an edge inside a feature window.
    pub(crate) fn outgoing_neighbors(&self, vertex: VertexId, cutoff: f64) -> Vec<VertexId> {
        self.neighbors(&self.outgoing, vertex, cutoff)
    }

    /// Return active incoming neighbors with an edge inside a feature window.
    pub(crate) fn incoming_neighbors(&self, vertex: VertexId, cutoff: f64) -> Vec<VertexId> {
        self.neighbors(&self.incoming, vertex, cutoff)
    }

    /// Return values of all qualifying incident edges for vertex statistics.
    pub(crate) fn incident_values(
        &self,
        vertex: VertexId,
        outgoing: bool,
        cutoff: f64,
        column: usize,
    ) -> Vec<f64> {
        let adjacency = if outgoing {
            &self.outgoing
        } else {
            &self.incoming
        };
        let Some(neighbours) = adjacency.get(&vertex) else {
            return Vec::new();
        };

        neighbours
            .values()
            .flat_map(|ids| ids.iter())
            .filter_map(|id| self.edges.get(id))
            .filter(|edge| edge.timestamp >= cutoff)
            .map(|edge| edge.values[column])
            .collect()
    }

    /// Return fan and degree values inside a feature-specific time window.
    pub(crate) fn fan_and_degree(
        &self,
        vertex: VertexId,
        outgoing: bool,
        cutoff: f64,
    ) -> (usize, usize) {
        let adjacency = if outgoing {
            &self.outgoing
        } else {
            &self.incoming
        };
        let Some(neighbours) = adjacency.get(&vertex) else {
            return (0, 0);
        };

        let mut fan = 0;
        let mut degree = 0;
        for ids in neighbours.values() {
            let count = ids
                .iter()
                .filter_map(|id| self.edges.get(id))
                .filter(|edge| edge.timestamp >= cutoff)
                .count();
            if count > 0 {
                fan += 1;
                degree += count;
            }
        }
        (fan, degree)
    }

    /// Return physical incident edges sorted by timestamp and insertion order.
    pub(crate) fn incident_edges(
        &self,
        vertex: VertexId,
        outgoing: bool,
        cutoff: f64,
    ) -> Vec<IncidentEdge> {
        let adjacency = if outgoing {
            &self.outgoing
        } else {
            &self.incoming
        };
        let Some(neighbours) = adjacency.get(&vertex) else {
            return Vec::new();
        };
        let mut result: Vec<_> = neighbours
            .iter()
            .flat_map(|(neighbour, ids)| {
                ids.iter().filter_map(|id| {
                    self.edges.get(id).and_then(|edge| {
                        (edge.timestamp >= cutoff).then_some(IncidentEdge {
                            id: edge.id,
                            neighbour: *neighbour,
                            timestamp: edge.timestamp,
                            sequence: edge.sequence,
                        })
                    })
                })
            })
            .collect();
        result.sort_by(|left, right| {
            left.timestamp
                .total_cmp(&right.timestamp)
                .then(left.sequence.cmp(&right.sequence))
        });
        result
    }

    /// Return edge timestamps for one directed vertex pair in the feature window.
    pub(crate) fn pair_timestamps(
        &self,
        source: VertexId,
        target: VertexId,
        cutoff: f64,
    ) -> Vec<f64> {
        let Some(neighbours) = self.outgoing.get(&source) else {
            return Vec::new();
        };
        let Some(ids) = neighbours.get(&target) else {
            return Vec::new();
        };

        let mut timestamps: Vec<_> = ids
            .iter()
            .filter_map(|id| self.edges.get(id))
            .filter(|edge| edge.timestamp >= cutoff)
            .map(|edge| edge.timestamp)
            .collect();
        timestamps.sort_by(f64::total_cmp);
        timestamps.dedup_by(|left, right| left.total_cmp(right).is_eq());
        timestamps
    }

    /// Return the number of active edges without exposing mutable internals.
    pub(crate) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return the insertion sequence assigned to the next batch edge.
    pub(crate) fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    fn neighbors(
        &self,
        adjacency: &HashMap<VertexId, NeighbourEdges>,
        vertex: VertexId,
        cutoff: f64,
    ) -> Vec<VertexId> {
        let Some(neighbours) = adjacency.get(&vertex) else {
            return Vec::new();
        };
        let mut result: Vec<_> = neighbours
            .iter()
            .filter(|(_, ids)| self.has_edge_after(ids, cutoff))
            .map(|(neighbour, _)| *neighbour)
            .collect();
        result.sort_unstable();
        result
    }

    fn has_edge_after(&self, ids: &VecDeque<EdgeId>, cutoff: f64) -> bool {
        ids.iter()
            .filter_map(|id| self.edges.get(id))
            .any(|edge| edge.timestamp >= cutoff)
    }

    fn add_adjacency(&mut self, edge: &Edge) {
        self.outgoing
            .entry(edge.source)
            .or_default()
            .entry(edge.target)
            .or_default()
            .push_back(edge.id);
        self.incoming
            .entry(edge.target)
            .or_default()
            .entry(edge.source)
            .or_default()
            .push_back(edge.id);
    }

    fn prune_time(&mut self, now: f64, time_window: f64) {
        let cutoff = OrderedFloat(now - time_window);
        let expired_times: Vec<_> = self
            .by_time
            .range(..=cutoff)
            .map(|(time, _)| *time)
            .collect();
        for time in expired_times {
            let ids = self.by_time.remove(&time).unwrap_or_default();
            for id in ids {
                self.remove_edge(id, false);
            }
        }
        self.drop_stale_insertion_prefix();
    }

    fn prune_count(&mut self, max_no_edges: isize) {
        if max_no_edges < 0 {
            return;
        }
        let limit = max_no_edges as usize;
        self.drop_stale_insertion_prefix();
        while self.edges.len() > limit {
            let Some(id) = self.insertion_order.pop_front() else {
                break;
            };
            self.remove_edge(id, true);
            self.drop_stale_insertion_prefix();
        }
    }

    fn drop_stale_insertion_prefix(&mut self) {
        while self
            .insertion_order
            .front()
            .is_some_and(|id| !self.edges.contains_key(id))
        {
            self.insertion_order.pop_front();
        }
    }

    fn remove_edge(&mut self, id: EdgeId, remove_time_index: bool) {
        let Some(edge) = self.edges.remove(&id) else {
            return;
        };
        self.remove_adjacency(&edge, true);
        self.remove_adjacency(&edge, false);
        if remove_time_index {
            let time = OrderedFloat(edge.timestamp);
            let remove_key = if let Some(ids) = self.by_time.get_mut(&time) {
                ids.retain(|candidate| *candidate != id);
                ids.is_empty()
            } else {
                false
            };
            if remove_key {
                self.by_time.remove(&time);
            }
        }
    }

    fn remove_adjacency(&mut self, edge: &Edge, outgoing: bool) {
        let (adjacency, vertex, neighbour) = if outgoing {
            (&mut self.outgoing, edge.source, edge.target)
        } else {
            (&mut self.incoming, edge.target, edge.source)
        };
        let mut remove_vertex = false;
        if let Some(neighbours) = adjacency.get_mut(&vertex) {
            let mut remove_neighbour = false;
            if let Some(ids) = neighbours.get_mut(&neighbour) {
                ids.retain(|candidate| *candidate != edge.id);
                remove_neighbour = ids.is_empty();
            }
            if remove_neighbour {
                neighbours.remove(&neighbour);
            }
            remove_vertex = neighbours.is_empty();
        }
        if remove_vertex {
            adjacency.remove(&vertex);
        }
    }
}

impl Edge {
    fn from_values(values: Vec<f64>) -> Result<Self, String> {
        if values.len() < 4 {
            return Err("features must have [edge_id, source, target, timestamp] columns".into());
        }
        if values[..4].iter().any(|value| !value.is_finite()) {
            return Err(
                "edge ID, source, target, and timestamp must be finite float64 values".into(),
            );
        }
        Ok(Self {
            id: values[0].to_bits(),
            source: values[1].to_bits(),
            target: values[2].to_bits(),
            timestamp: values[3],
            sequence: 0,
            values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tabular::config::{Config, PatternConfig};

    fn config() -> Config {
        Config {
            num_threads: 1,
            time_window: 10.0,
            max_no_edges: -1,
            vertex_stats: false,
            vertex_stats_tw: 10.0,
            vertex_stats_cols: vec![],
            vertex_stats_feats: vec![],
            fan: pattern(),
            degree: pattern(),
            scatter_gather: pattern(),
            temp_cycle: pattern(),
            lc_cycle: pattern(),
            lc_cycle_len: 2,
        }
    }

    fn pattern() -> PatternConfig {
        PatternConfig {
            enabled: false,
            time_window: 10.0,
            bins: vec![2],
        }
    }

    #[test]
    fn time_window_evicts_the_inclusive_lower_boundary() {
        let mut state = GraphState::default();
        let config = config();
        state.insert(vec![1.0, 1.0, 2.0, 1.0], &config).unwrap();
        state.insert(vec![2.0, 2.0, 3.0, 11.0], &config).unwrap();
        assert_eq!(state.edge_count(), 1);
        assert_eq!(state.latest_timestamp(), Some(11.0));
    }

    #[test]
    fn duplicate_active_edge_id_is_ignored() {
        let mut state = GraphState::default();
        let config = config();
        state.insert(vec![1.0, 1.0, 2.0, 1.0], &config).unwrap();
        state.insert(vec![1.0, 2.0, 3.0, 2.0], &config).unwrap();
        assert_eq!(state.edge_count(), 1);
        assert_eq!(
            state.outgoing_neighbors(1.0_f64.to_bits(), 0.0),
            vec![2.0_f64.to_bits()]
        );
    }
}
