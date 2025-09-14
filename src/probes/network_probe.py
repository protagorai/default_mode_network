"""
Network-level probes for monitoring connectivity and global dynamics.

This module provides probes for recording network-wide activity patterns,
connectivity changes, and emergent network properties like small-world
characteristics and global synchronization.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from collections import defaultdict
import networkx as nx

from .base_probe import BaseProbe, ProbeType


class NetworkActivityProbe(BaseProbe):
    """
    Probe for monitoring network-wide activity patterns and dynamics.
    
    Records global network metrics including total activity, connectivity
    utilization, and network-wide synchronization measures.
    """
    
    def __init__(self, probe_id: str, network_id: str,
                 sampling_interval: float = 10.0,
                 track_connectivity: bool = True,
                 track_synchrony: bool = True,
                 track_criticality: bool = False):
        """
        Initialize network activity probe.
        
        Args:
            probe_id: Unique identifier for probe
            network_id: ID of target network
            sampling_interval: Time between samples (ms)
            track_connectivity: Monitor connectivity changes
            track_synchrony: Monitor network synchronization
            track_criticality: Monitor critical dynamics indicators
        """
        super().__init__(probe_id, ProbeType.NETWORK_ACTIVITY, [network_id], sampling_interval)
        
        self.network_id = network_id
        self.track_connectivity = track_connectivity
        self.track_synchrony = track_synchrony
        self.track_criticality = track_criticality
        
        # Network object reference
        self.network = None
        self.neuron_objects: Dict[str, Any] = {}
        self.connection_objects: Dict[str, Any] = {}
        
        # Activity tracking
        self.global_activity: List[float] = []
        self.active_connections: List[int] = []
        self.network_synchrony: List[float] = []
        
        # Connectivity metrics
        self.clustering_coefficients: List[float] = []
        self.path_lengths: List[float] = []
        self.small_world_indices: List[float] = []
        
        # Criticality indicators
        self.avalanche_sizes: List[int] = []
        self.avalanche_durations: List[float] = []
        self.branching_parameters: List[float] = []
        
        # Network statistics
        self.network_stats = {
            'total_neurons': 0,
            'total_connections': 0,
            'max_activity': 0.0,
            'mean_clustering': 0.0,
            'mean_path_length': 0.0
        }
    
    def register_network(self, network: Any) -> None:
        """
        Register network object for monitoring.
        
        Args:
            network: Network object to monitor
        """
        self.network = network
        
        # Extract network components
        if hasattr(network, 'neurons'):
            self.neuron_objects = network.neurons
        if hasattr(network, 'connections') or hasattr(network, 'synapses'):
            self.connection_objects = getattr(network, 'connections', 
                                            getattr(network, 'synapses', {}))
        
        # Update network stats
        self.network_stats['total_neurons'] = len(self.neuron_objects)
        self.network_stats['total_connections'] = len(self.connection_objects)
    
    def record(self, current_time: float) -> None:
        """
        Record network activity at current time.
        
        Args:
            current_time: Current simulation time (ms)
        """
        self.current_time = current_time
        
        if not self.should_record(current_time):
            return
        
        network_data = {}
        
        # Calculate global activity
        global_activity = self._calculate_global_activity()
        self.global_activity.append(global_activity)
        network_data['global_activity'] = global_activity
        
        # Track connectivity if enabled
        if self.track_connectivity:
            active_conn = self._count_active_connections()
            self.active_connections.append(active_conn)
            network_data['active_connections'] = active_conn
            
            # Calculate network topology metrics (less frequently)
            if len(self.data.timestamps) % 10 == 0:  # Every 10 samples
                topology_metrics = self._calculate_topology_metrics()
                network_data.update(topology_metrics)
        
        # Track network synchrony if enabled
        if self.track_synchrony:
            synchrony = self._calculate_network_synchrony()
            self.network_synchrony.append(synchrony)
            network_data['network_synchrony'] = synchrony
        
        # Track criticality indicators if enabled
        if self.track_criticality:
            criticality_metrics = self._analyze_criticality()
            network_data.update(criticality_metrics)
        
        self._record_sample(current_time, network_data)
        self._update_statistics(network_data)
    
    def _calculate_global_activity(self) -> float:
        """Calculate global network activity level."""
        if not self.neuron_objects:
            return 0.0
        
        total_activity = 0.0
        active_neurons = 0
        
        for neuron in self.neuron_objects.values():
            if hasattr(neuron, 'has_spiked') and neuron.has_spiked():
                active_neurons += 1
                total_activity += 1.0
            elif hasattr(neuron, 'get_membrane_potential'):
                # Use membrane potential as activity indicator
                v_mem = neuron.get_membrane_potential()
                v_rest = getattr(neuron.parameters, 'v_rest', -70.0)
                activity = max(0.0, (v_mem - v_rest) / 50.0)  # Normalized activity
                total_activity += activity
        
        return total_activity / len(self.neuron_objects) if self.neuron_objects else 0.0
    
    def _count_active_connections(self) -> int:
        """Count currently active synaptic connections."""
        if not self.connection_objects:
            return 0
        
        active_count = 0
        for connection in self.connection_objects.values():
            if hasattr(connection, 'get_current'):
                current = abs(connection.get_current())
                if current > 1e-6:  # Threshold for "active"
                    active_count += 1
            elif hasattr(connection, 'get_conductance'):
                conductance = connection.get_conductance()
                if conductance > 1e-6:
                    active_count += 1
        
        return active_count
    
    def _calculate_network_synchrony(self) -> float:
        """Calculate global network synchronization."""
        if not self.neuron_objects:
            return 0.0
        
        # Collect recent spike times
        recent_spikes = []
        time_window = 50.0  # ms
        current_time = self.current_time
        
        for neuron in self.neuron_objects.values():
            if hasattr(neuron, 'get_spike_times'):
                spike_times = neuron.get_spike_times()
                window_spikes = [t for t in spike_times 
                               if current_time - time_window <= t <= current_time]
                recent_spikes.extend(window_spikes)
        
        if len(recent_spikes) < 2:
            return 0.0
        
        # Calculate synchrony as inverse of spike time variance
        spike_array = np.array(recent_spikes)
        if len(spike_array) > 1:
            spike_variance = np.var(spike_array)
            # Normalize by time window
            normalized_variance = spike_variance / (time_window ** 2)
            synchrony = 1.0 / (1.0 + normalized_variance)
            return min(1.0, synchrony)
        
        return 0.0
    
    def _calculate_topology_metrics(self) -> Dict[str, float]:
        """Calculate network topology metrics."""
        if not self.neuron_objects or not self.connection_objects:
            return {}
        
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for neuron_id in self.neuron_objects.keys():
                G.add_node(neuron_id)
            
            # Add edges from connections
            for connection in self.connection_objects.values():
                if hasattr(connection, 'presynaptic_neuron_id') and hasattr(connection, 'postsynaptic_neuron_id'):
                    pre_id = connection.presynaptic_neuron_id
                    post_id = connection.postsynaptic_neuron_id
                    
                    # Only add if both neurons exist
                    if pre_id in self.neuron_objects and post_id in self.neuron_objects:
                        weight = 1.0
                        if hasattr(connection, 'get_weight'):
                            weight = abs(connection.get_weight())
                        G.add_edge(pre_id, post_id, weight=weight)
            
            if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
                return {}
            
            # Calculate metrics
            metrics = {}
            
            # Clustering coefficient
            clustering = nx.average_clustering(G)
            self.clustering_coefficients.append(clustering)
            metrics['clustering_coefficient'] = clustering
            
            # Average path length (for connected components)
            if nx.is_connected(G):
                path_length = nx.average_shortest_path_length(G)
                self.path_lengths.append(path_length)
                metrics['average_path_length'] = path_length
                
                # Small-world index
                # Compare with random network of same size and degree
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
                avg_degree = 2 * n_edges / n_nodes
                
                if avg_degree > 1:
                    # Expected clustering for random network
                    random_clustering = avg_degree / n_nodes
                    # Expected path length for random network
                    random_path_length = np.log(n_nodes) / np.log(avg_degree)
                    
                    if random_clustering > 0 and random_path_length > 0:
                        small_world = (clustering / random_clustering) / (path_length / random_path_length)
                        self.small_world_indices.append(small_world)
                        metrics['small_world_index'] = small_world
            
            return metrics
            
        except Exception:
            # Return empty dict if topology calculation fails
            return {}
    
    def _analyze_criticality(self) -> Dict[str, Any]:
        """Analyze indicators of critical dynamics."""
        criticality_data = {}
        
        # Detect neuronal avalanches
        avalanche_metrics = self._detect_avalanches()
        if avalanche_metrics:
            criticality_data.update(avalanche_metrics)
        
        # Calculate branching parameter
        branching_param = self._calculate_branching_parameter()
        if branching_param is not None:
            self.branching_parameters.append(branching_param)
            criticality_data['branching_parameter'] = branching_param
        
        return criticality_data
    
    def _detect_avalanches(self) -> Dict[str, Any]:
        """Detect and analyze neuronal avalanches."""
        if not self.neuron_objects:
            return {}
        
        # Simple avalanche detection based on activity propagation
        time_bin = 1.0  # ms
        current_time = self.current_time
        
        # Count active neurons in current bin
        active_neurons = 0
        for neuron in self.neuron_objects.values():
            if hasattr(neuron, 'has_spiked') and neuron.has_spiked():
                active_neurons += 1
        
        # Track avalanche state
        if not hasattr(self, '_avalanche_state'):
            self._avalanche_state = {
                'in_avalanche': False,
                'avalanche_start': None,
                'avalanche_size': 0
            }
        
        state = self._avalanche_state
        
        if active_neurons > 0:
            if not state['in_avalanche']:
                # Start new avalanche
                state['in_avalanche'] = True
                state['avalanche_start'] = current_time
                state['avalanche_size'] = active_neurons
            else:
                # Continue avalanche
                state['avalanche_size'] += active_neurons
        else:
            if state['in_avalanche']:
                # End avalanche
                duration = current_time - state['avalanche_start']
                
                self.avalanche_sizes.append(state['avalanche_size'])
                self.avalanche_durations.append(duration)
                
                # Reset state
                state['in_avalanche'] = False
                state['avalanche_start'] = None
                state['avalanche_size'] = 0
                
                return {
                    'avalanche_size': state['avalanche_size'],
                    'avalanche_duration': duration
                }
        
        return {}
    
    def _calculate_branching_parameter(self) -> Optional[float]:
        """Calculate branching parameter for criticality assessment."""
        if not self.neuron_objects:
            return None
        
        # Simplified branching parameter: ratio of offspring to parent events
        time_window = 10.0  # ms
        current_time = self.current_time
        
        # Count spikes in current and next time window
        current_spikes = 0
        next_spikes = 0
        
        for neuron in self.neuron_objects.values():
            if hasattr(neuron, 'get_spike_times'):
                spike_times = neuron.get_spike_times()
                
                # Current window
                current_window_spikes = [t for t in spike_times 
                                       if current_time - time_window <= t <= current_time]
                current_spikes += len(current_window_spikes)
                
                # Next window (estimated)
                if hasattr(neuron, 'get_membrane_potential'):
                    v_mem = neuron.get_membrane_potential()
                    v_thresh = getattr(neuron.parameters, 'v_thresh', -50.0)
                    if v_mem > v_thresh - 10.0:  # Close to threshold
                        next_spikes += 1
        
        if current_spikes > 0:
            return next_spikes / current_spikes
        
        return 1.0  # Neutral branching
    
    def _update_statistics(self, network_data: Dict[str, Any]) -> None:
        """Update network statistics."""
        if 'global_activity' in network_data:
            activity = network_data['global_activity']
            if activity > self.network_stats['max_activity']:
                self.network_stats['max_activity'] = activity
        
        if 'clustering_coefficient' in network_data:
            self.network_stats['mean_clustering'] = np.mean(self.clustering_coefficients)
        
        if 'average_path_length' in network_data:
            self.network_stats['mean_path_length'] = np.mean(self.path_lengths)
    
    def get_measurement_value(self, target_id: str) -> Any:
        """Get current global activity level."""
        if self.global_activity:
            return self.global_activity[-1]
        return 0.0
    
    def get_activity_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get global activity time series."""
        times = self.data.get_time_array()
        activity = np.array(self.global_activity)
        return times, activity
    
    def get_synchrony_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get network synchrony time series."""
        if not self.track_synchrony:
            return np.array([]), np.array([])
        
        times = self.data.get_time_array()
        synchrony = np.array(self.network_synchrony)
        return times, synchrony
    
    def get_topology_statistics(self) -> Dict[str, float]:
        """Get network topology statistics."""
        stats = {}
        
        if self.clustering_coefficients:
            stats['mean_clustering'] = np.mean(self.clustering_coefficients)
            stats['std_clustering'] = np.std(self.clustering_coefficients)
        
        if self.path_lengths:
            stats['mean_path_length'] = np.mean(self.path_lengths)
            stats['std_path_length'] = np.std(self.path_lengths)
        
        if self.small_world_indices:
            stats['mean_small_world'] = np.mean(self.small_world_indices)
            stats['std_small_world'] = np.std(self.small_world_indices)
        
        return stats
    
    def get_criticality_analysis(self) -> Dict[str, Any]:
        """Get criticality analysis results."""
        if not self.track_criticality:
            return {}
        
        analysis = {}
        
        if self.avalanche_sizes:
            sizes = np.array(self.avalanche_sizes)
            analysis['avalanche_stats'] = {
                'mean_size': np.mean(sizes),
                'std_size': np.std(sizes),
                'max_size': np.max(sizes),
                'total_avalanches': len(sizes)
            }
        
        if self.avalanche_durations:
            durations = np.array(self.avalanche_durations)
            analysis['duration_stats'] = {
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'max_duration': np.max(durations)
            }
        
        if self.branching_parameters:
            branching = np.array(self.branching_parameters)
            analysis['branching_stats'] = {
                'mean_branching': np.mean(branching),
                'std_branching': np.std(branching),
                'criticality_index': abs(np.mean(branching) - 1.0)  # Distance from critical value
            }
        
        return analysis
    
    def __str__(self) -> str:
        """String representation."""
        return (f"NetworkActivityProbe(id={self.probe_id}, "
                f"network={self.network_id}, "
                f"neurons={self.network_stats['total_neurons']}, "
                f"connections={self.network_stats['total_connections']}, "
                f"samples={self.samples_collected})")


class ConnectivityProbe(BaseProbe):
    """
    Probe for monitoring synaptic connectivity changes over time.
    
    Tracks synaptic weight changes, connection formation/elimination,
    and plasticity-driven network reorganization.
    """
    
    def __init__(self, probe_id: str, target_connections: List[str],
                 sampling_interval: float = 100.0,
                 track_weights: bool = True,
                 track_formation: bool = True):
        """
        Initialize connectivity probe.
        
        Args:
            probe_id: Unique identifier
            target_connections: List of connection/synapse IDs to monitor
            sampling_interval: Time between samples (ms)
            track_weights: Monitor synaptic weight changes
            track_formation: Monitor connection formation/elimination
        """
        super().__init__(probe_id, ProbeType.CONNECTIVITY, target_connections, sampling_interval)
        
        self.track_weights = track_weights
        self.track_formation = track_formation
        
        # Connection object references
        self.connection_objects: Dict[str, Any] = {}
        
        # Weight tracking
        self.weight_histories: Dict[str, List[float]] = defaultdict(list)
        self.weight_changes: List[Dict[str, float]] = []
        
        # Formation/elimination tracking
        self.active_connections: Dict[str, bool] = {}
        self.formation_events: List[Dict[str, Any]] = []
        self.elimination_events: List[Dict[str, Any]] = []
        
        # Statistics
        self.connectivity_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'mean_weight': 0.0,
            'weight_range': 0.0
        }
    
    def register_connection_objects(self, connection_objects: Dict[str, Any]) -> None:
        """
        Register connection objects for monitoring.
        
        Args:
            connection_objects: Dictionary of connection objects
        """
        for conn_id in self.target_ids:
            if conn_id in connection_objects:
                self.connection_objects[conn_id] = connection_objects[conn_id]
                self.active_connections[conn_id] = True
        
        self.connectivity_stats['total_connections'] = len(self.connection_objects)
    
    def record(self, current_time: float) -> None:
        """
        Record connectivity data at current time.
        
        Args:
            current_time: Current simulation time (ms)
        """
        self.current_time = current_time
        
        if not self.should_record(current_time):
            return
        
        connectivity_data = {}
        
        if self.track_weights:
            weight_data = self._record_weights()
            connectivity_data['weights'] = weight_data
        
        if self.track_formation:
            formation_data = self._check_formation_elimination()
            connectivity_data.update(formation_data)
        
        self._record_sample(current_time, connectivity_data)
        self._update_statistics(connectivity_data)
    
    def _record_weights(self) -> Dict[str, float]:
        """Record current synaptic weights."""
        current_weights = {}
        
        for conn_id, connection in self.connection_objects.items():
            if hasattr(connection, 'get_weight'):
                weight = connection.get_weight()
                current_weights[conn_id] = weight
                self.weight_histories[conn_id].append(weight)
        
        return current_weights
    
    def _check_formation_elimination(self) -> Dict[str, Any]:
        """Check for connection formation/elimination events."""
        formation_data = {}
        
        # Check for newly formed connections (simplified)
        active_count = 0
        for conn_id, connection in self.connection_objects.items():
            is_active = False
            
            if hasattr(connection, 'get_weight'):
                weight = connection.get_weight()
                is_active = abs(weight) > 1e-6
            elif hasattr(connection, 'get_conductance'):
                conductance = connection.get_conductance()
                is_active = conductance > 1e-6
            
            # Check for state changes
            was_active = self.active_connections.get(conn_id, False)
            
            if is_active and not was_active:
                # Connection formed/reactivated
                event = {
                    'connection_id': conn_id,
                    'event_time': self.current_time,
                    'event_type': 'formation'
                }
                self.formation_events.append(event)
            elif not is_active and was_active:
                # Connection eliminated/deactivated
                event = {
                    'connection_id': conn_id,
                    'event_time': self.current_time,
                    'event_type': 'elimination'
                }
                self.elimination_events.append(event)
            
            self.active_connections[conn_id] = is_active
            if is_active:
                active_count += 1
        
        formation_data['active_connections'] = active_count
        formation_data['formation_events'] = len(self.formation_events)
        formation_data['elimination_events'] = len(self.elimination_events)
        
        return formation_data
    
    def _update_statistics(self, connectivity_data: Dict[str, Any]) -> None:
        """Update connectivity statistics."""
        if 'active_connections' in connectivity_data:
            self.connectivity_stats['active_connections'] = connectivity_data['active_connections']
        
        if 'weights' in connectivity_data:
            weights = list(connectivity_data['weights'].values())
            if weights:
                self.connectivity_stats['mean_weight'] = np.mean(weights)
                self.connectivity_stats['weight_range'] = np.max(weights) - np.min(weights)
    
    def get_measurement_value(self, target_id: str) -> Any:
        """Get current weight of target connection."""
        if target_id in self.connection_objects:
            connection = self.connection_objects[target_id]
            if hasattr(connection, 'get_weight'):
                return connection.get_weight()
        return 0.0
    
    def get_weight_history(self, connection_id: str) -> List[float]:
        """Get weight history for specific connection."""
        return self.weight_histories.get(connection_id, []).copy()
    
    def get_connectivity_statistics(self) -> Dict[str, Any]:
        """Get connectivity statistics."""
        return self.connectivity_stats.copy()
    
    def __str__(self) -> str:
        """String representation."""
        return (f"ConnectivityProbe(id={self.probe_id}, "
                f"connections={len(self.target_ids)}, "
                f"active={self.connectivity_stats['active_connections']}, "
                f"samples={self.samples_collected})")
