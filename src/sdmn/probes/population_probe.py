"""
Population activity probes for monitoring aggregate neural dynamics.

This module provides probes for recording population-level activity,
including population firing rates, local field potentials (LFP),
and synchronization measures.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from collections import defaultdict

from sdmn.probes.base_probe import BaseProbe, ProbeType
from sdmn.neurons.base_neuron import BaseNeuron


class PopulationActivityProbe(BaseProbe):
    """
    Probe for monitoring population-level neural activity.
    
    Records aggregate measures of population activity including
    instantaneous firing rates, spike count histograms, and
    synchronization indices.
    """
    
    def __init__(self, probe_id: str, target_population: str,
                 target_neurons: List[str],
                 bin_size: float = 10.0,
                 sliding_window: float = 100.0,
                 record_synchrony: bool = True):
        """
        Initialize population activity probe.
        
        Args:
            probe_id: Unique identifier for probe
            target_population: Name of target population
            target_neurons: List of neuron IDs in population
            bin_size: Time bin size for spike counting (ms)
            sliding_window: Window for rate calculations (ms)
            record_synchrony: Whether to calculate synchrony measures
        """
        super().__init__(probe_id, ProbeType.POPULATION_ACTIVITY, target_neurons, bin_size)
        
        self.target_population = target_population
        self.bin_size = bin_size
        self.sliding_window = sliding_window
        self.record_synchrony = record_synchrony
        
        # Population activity data
        self.population_rates: List[float] = []
        self.spike_counts: List[int] = []
        self.active_neuron_counts: List[int] = []
        
        # Synchrony measures
        self.synchrony_indices: List[float] = []
        self.correlation_coefficients: List[float] = []
        
        # Spike time tracking for synchrony calculations
        self.recent_spike_times: Dict[str, List[float]] = defaultdict(list)
        self.spike_history_window = max(sliding_window, 200.0)  # Keep longer history
        
        # Reference to neuron objects
        self.neuron_objects: Dict[str, BaseNeuron] = {}
        
        # Statistics
        self.population_stats = {
            'max_rate': 0.0,
            'mean_rate': 0.0,
            'total_spikes': 0,
            'peak_synchrony': 0.0
        }
    
    def register_neuron_objects(self, neuron_objects: Dict[str, BaseNeuron]) -> None:
        """
        Register neuron objects for the population.
        
        Args:
            neuron_objects: Dictionary mapping neuron IDs to neuron objects
        """
        for neuron_id in self.target_ids:
            if neuron_id in neuron_objects:
                self.neuron_objects[neuron_id] = neuron_objects[neuron_id]
    
    def record(self, current_time: float) -> None:
        """
        Record population activity at current time.
        
        Args:
            current_time: Current simulation time (ms)
        """
        self.current_time = current_time
        
        if not self.should_record(current_time):
            return
        
        # Count spikes in current time bin
        spike_count = self._count_recent_spikes(current_time, self.bin_size)
        
        # Calculate instantaneous population firing rate
        population_size = len(self.target_ids)
        if population_size > 0:
            instantaneous_rate = (spike_count * 1000.0) / (self.bin_size * population_size)
        else:
            instantaneous_rate = 0.0
        
        # Count active neurons
        active_neurons = self._count_active_neurons(current_time, self.bin_size)
        
        # Calculate synchrony if enabled
        synchrony_index = 0.0
        correlation_coeff = 0.0
        if self.record_synchrony:
            synchrony_index = self._calculate_synchrony_index(current_time)
            correlation_coeff = self._calculate_population_correlation(current_time)
        
        # Store measurements
        self.population_rates.append(instantaneous_rate)
        self.spike_counts.append(spike_count)
        self.active_neuron_counts.append(active_neurons)
        
        if self.record_synchrony:
            self.synchrony_indices.append(synchrony_index)
            self.correlation_coefficients.append(correlation_coeff)
        
        # Record as structured data
        activity_data = {
            'population_rate': instantaneous_rate,
            'spike_count': spike_count,
            'active_neurons': active_neurons,
            'synchrony_index': synchrony_index if self.record_synchrony else None,
            'correlation': correlation_coeff if self.record_synchrony else None
        }
        
        self._record_sample(current_time, activity_data)
        self._update_statistics(activity_data)
        
        # Update spike history
        self._update_spike_history(current_time)
    
    def _count_recent_spikes(self, current_time: float, window: float) -> int:
        """Count spikes in recent time window."""
        spike_count = 0
        window_start = current_time - window
        
        for neuron_id in self.target_ids:
            if neuron_id in self.neuron_objects:
                neuron = self.neuron_objects[neuron_id]
                if hasattr(neuron, 'get_spike_times'):
                    spike_times = neuron.get_spike_times()
                    recent_spikes = [t for t in spike_times if t > window_start]
                    spike_count += len(recent_spikes)
        
        return spike_count
    
    def _count_active_neurons(self, current_time: float, window: float) -> int:
        """Count neurons that spiked in recent window."""
        active_count = 0
        window_start = current_time - window
        
        for neuron_id in self.target_ids:
            if neuron_id in self.neuron_objects:
                neuron = self.neuron_objects[neuron_id]
                if hasattr(neuron, 'get_spike_times'):
                    spike_times = neuron.get_spike_times()
                    recent_spikes = [t for t in spike_times if t > window_start]
                    if len(recent_spikes) > 0:
                        active_count += 1
        
        return active_count
    
    def _calculate_synchrony_index(self, current_time: float) -> float:
        """
        Calculate population synchrony index.
        
        Uses the variance-based synchrony measure:
        χ = (σ²_pop - <σ²_ind>) / (σ²_pop + <σ²_ind>)
        """
        window_start = current_time - self.sliding_window
        
        # Get spike trains for synchrony window
        spike_trains = {}
        for neuron_id in self.target_ids:
            if neuron_id in self.neuron_objects:
                neuron = self.neuron_objects[neuron_id]
                if hasattr(neuron, 'get_spike_times'):
                    spike_times = neuron.get_spike_times()
                    window_spikes = [t for t in spike_times 
                                   if window_start <= t <= current_time]
                    spike_trains[neuron_id] = window_spikes
        
        if len(spike_trains) < 2:
            return 0.0
        
        # Bin spike trains
        bin_edges = np.arange(window_start, current_time + self.bin_size, self.bin_size)
        if len(bin_edges) < 2:
            return 0.0
        
        binned_trains = []
        for spike_times in spike_trains.values():
            if spike_times:
                counts, _ = np.histogram(spike_times, bins=bin_edges)
                binned_trains.append(counts)
            else:
                binned_trains.append(np.zeros(len(bin_edges) - 1))
        
        if not binned_trains:
            return 0.0
        
        binned_array = np.array(binned_trains)
        
        # Calculate population sum variance
        pop_sum = np.sum(binned_array, axis=0)
        pop_variance = np.var(pop_sum)
        
        # Calculate mean individual variance
        individual_variances = np.var(binned_array, axis=1)
        mean_individual_variance = np.mean(individual_variances)
        
        # Synchrony index
        denominator = pop_variance + mean_individual_variance
        if denominator > 0:
            synchrony = (pop_variance - mean_individual_variance) / denominator
            return max(0.0, min(1.0, synchrony))  # Clamp to [0,1]
        
        return 0.0
    
    def _calculate_population_correlation(self, current_time: float) -> float:
        """Calculate mean pairwise correlation coefficient."""
        window_start = current_time - self.sliding_window
        
        # Get binned spike trains
        spike_trains = []
        for neuron_id in self.target_ids:
            if neuron_id in self.neuron_objects:
                neuron = self.neuron_objects[neuron_id]
                if hasattr(neuron, 'get_spike_times'):
                    spike_times = neuron.get_spike_times()
                    window_spikes = [t for t in spike_times 
                                   if window_start <= t <= current_time]
                    
                    # Bin spikes
                    bin_edges = np.arange(window_start, current_time + self.bin_size, 
                                        self.bin_size)
                    if len(bin_edges) >= 2:
                        counts, _ = np.histogram(window_spikes, bins=bin_edges)
                        spike_trains.append(counts)
        
        if len(spike_trains) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        n_neurons = len(spike_trains)
        
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                corr = np.corrcoef(spike_trains[i], spike_trains[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _update_spike_history(self, current_time: float) -> None:
        """Update recent spike history for calculations."""
        cutoff_time = current_time - self.spike_history_window
        
        # Clean old spikes and add new ones
        for neuron_id in self.target_ids:
            if neuron_id in self.neuron_objects:
                neuron = self.neuron_objects[neuron_id]
                if hasattr(neuron, 'get_spike_times'):
                    all_spikes = neuron.get_spike_times()
                    recent_spikes = [t for t in all_spikes if t > cutoff_time]
                    self.recent_spike_times[neuron_id] = recent_spikes
    
    def _update_statistics(self, activity_data: Dict[str, Any]) -> None:
        """Update population statistics."""
        rate = activity_data['population_rate']
        self.population_stats['total_spikes'] += activity_data['spike_count']
        
        if rate > self.population_stats['max_rate']:
            self.population_stats['max_rate'] = rate
        
        # Update running mean
        n_samples = len(self.population_rates)
        if n_samples > 0:
            old_mean = self.population_stats['mean_rate']
            self.population_stats['mean_rate'] = (old_mean * (n_samples - 1) + rate) / n_samples
        
        if self.record_synchrony and activity_data['synchrony_index']:
            sync_idx = activity_data['synchrony_index']
            if sync_idx > self.population_stats['peak_synchrony']:
                self.population_stats['peak_synchrony'] = sync_idx
    
    def get_measurement_value(self, target_id: str) -> Any:
        """Get current population rate."""
        if self.population_rates:
            return self.population_rates[-1]
        return 0.0
    
    def get_population_rate_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get time series of population firing rate."""
        times = self.data.get_time_array()
        rates = np.array(self.population_rates)
        return times, rates
    
    def get_synchrony_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get time series of synchrony index."""
        if not self.record_synchrony:
            return np.array([]), np.array([])
        
        times = self.data.get_time_array()
        synchrony = np.array(self.synchrony_indices)
        return times, synchrony
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get population activity statistics."""
        if not self.population_rates:
            return self.population_stats
        
        rates = np.array(self.population_rates)
        stats = self.population_stats.copy()
        
        stats.update({
            'mean_rate_calculated': np.mean(rates),
            'std_rate': np.std(rates),
            'median_rate': np.median(rates),
            'rate_range': np.max(rates) - np.min(rates),
            'cv_rate': np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0.0
        })
        
        if self.record_synchrony and self.synchrony_indices:
            sync_array = np.array(self.synchrony_indices)
            stats.update({
                'mean_synchrony': np.mean(sync_array),
                'std_synchrony': np.std(sync_array),
                'median_synchrony': np.median(sync_array)
            })
        
        return stats
    
    def detect_population_bursts(self, rate_threshold: float = None,
                               min_duration: float = 50.0) -> List[Dict[str, Any]]:
        """
        Detect population burst events.
        
        Args:
            rate_threshold: Rate threshold for burst detection (uses mean + 2*std if None)
            min_duration: Minimum burst duration (ms)
            
        Returns:
            List of detected bursts
        """
        if not self.population_rates:
            return []
        
        rates = np.array(self.population_rates)
        times = self.data.get_time_array()
        
        if rate_threshold is None:
            rate_threshold = np.mean(rates) + 2 * np.std(rates)
        
        # Find periods above threshold
        above_threshold = rates > rate_threshold
        
        bursts = []
        in_burst = False
        burst_start = None
        burst_indices = []
        
        for i, (is_above, time_point) in enumerate(zip(above_threshold, times)):
            if is_above and not in_burst:
                # Start of burst
                in_burst = True
                burst_start = time_point
                burst_indices = [i]
            elif is_above and in_burst:
                # Continue burst
                burst_indices.append(i)
            elif not is_above and in_burst:
                # End of burst
                burst_end = times[burst_indices[-1]]
                duration = burst_end - burst_start
                
                if duration >= min_duration:
                    burst_rates = rates[burst_indices]
                    bursts.append({
                        'start_time': burst_start,
                        'end_time': burst_end,
                        'duration': duration,
                        'peak_rate': np.max(burst_rates),
                        'mean_rate': np.mean(burst_rates),
                        'total_spikes': np.sum([self.spike_counts[j] for j in burst_indices])
                    })
                
                in_burst = False
                burst_start = None
                burst_indices = []
        
        return bursts
    
    def __str__(self) -> str:
        """String representation."""
        return (f"PopulationActivityProbe(id={self.probe_id}, "
                f"population={self.target_population}, "
                f"neurons={len(self.target_ids)}, "
                f"bin_size={self.bin_size}ms, "
                f"samples={self.samples_collected})")


class LFPProbe(BaseProbe):
    """
    Local Field Potential (LFP) probe for recording extracellular potentials.
    
    Approximates LFP by summing weighted contributions from nearby neurons
    based on their membrane potentials and distances.
    """
    
    def __init__(self, probe_id: str, target_neurons: List[str],
                 probe_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 sampling_interval: float = 1.0,
                 distance_weights: bool = True,
                 max_distance: float = 500.0):  # micrometers
        """
        Initialize LFP probe.
        
        Args:
            probe_id: Unique identifier
            target_neurons: List of neuron IDs to include
            probe_position: 3D position of recording electrode (μm)
            sampling_interval: Sampling interval (ms)
            distance_weights: Weight contributions by distance
            max_distance: Maximum distance for contributions (μm)
        """
        super().__init__(probe_id, ProbeType.LFP, target_neurons, sampling_interval)
        
        self.probe_position = np.array(probe_position)
        self.distance_weights = distance_weights
        self.max_distance = max_distance
        
        # Neuron positions and weights
        self.neuron_positions: Dict[str, np.ndarray] = {}
        self.neuron_weights: Dict[str, float] = {}
        self.neuron_objects: Dict[str, BaseNeuron] = {}
        
        # LFP data
        self.lfp_values: List[float] = []
        
        # Filtering (optional)
        self.enable_filtering = False
        self.filter_params = {'low': 1.0, 'high': 300.0}  # Hz
    
    def register_neuron_objects(self, neuron_objects: Dict[str, BaseNeuron],
                              neuron_positions: Optional[Dict[str, Tuple[float, float, float]]] = None) -> None:
        """
        Register neuron objects and positions.
        
        Args:
            neuron_objects: Dictionary of neuron objects
            neuron_positions: Dictionary of neuron 3D positions (μm)
        """
        for neuron_id in self.target_ids:
            if neuron_id in neuron_objects:
                self.neuron_objects[neuron_id] = neuron_objects[neuron_id]
                
                # Set position (random if not specified)
                if neuron_positions and neuron_id in neuron_positions:
                    self.neuron_positions[neuron_id] = np.array(neuron_positions[neuron_id])
                else:
                    # Random position within max_distance
                    random_pos = np.random.randn(3) * self.max_distance / 3
                    self.neuron_positions[neuron_id] = self.probe_position + random_pos
                
                # Calculate distance weight
                if self.distance_weights:
                    distance = np.linalg.norm(self.neuron_positions[neuron_id] - self.probe_position)
                    if distance <= self.max_distance:
                        # 1/r² weighting with minimum distance to avoid division by zero
                        weight = 1.0 / max(distance, 10.0) ** 2
                        self.neuron_weights[neuron_id] = weight
                    else:
                        self.neuron_weights[neuron_id] = 0.0
                else:
                    self.neuron_weights[neuron_id] = 1.0
    
    def record(self, current_time: float) -> None:
        """
        Record LFP at current time.
        
        Args:
            current_time: Current simulation time (ms)
        """
        self.current_time = current_time
        
        if not self.should_record(current_time):
            return
        
        # Calculate LFP as weighted sum of membrane potentials
        lfp_value = self._calculate_lfp()
        self.lfp_values.append(lfp_value)
        
        self._record_sample(current_time, lfp_value)
    
    def _calculate_lfp(self) -> float:
        """Calculate LFP value from neuron contributions."""
        lfp_sum = 0.0
        total_weight = 0.0
        
        for neuron_id in self.target_ids:
            if (neuron_id in self.neuron_objects and 
                neuron_id in self.neuron_weights):
                
                neuron = self.neuron_objects[neuron_id]
                weight = self.neuron_weights[neuron_id]
                
                if weight > 0:
                    # Use membrane potential as contribution
                    voltage = neuron.get_membrane_potential()
                    lfp_sum += weight * voltage
                    total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return lfp_sum / total_weight
        else:
            return 0.0
    
    def get_measurement_value(self, target_id: str) -> Any:
        """Get current LFP value."""
        if self.lfp_values:
            return self.lfp_values[-1]
        return 0.0
    
    def get_lfp_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get LFP time series."""
        times = self.data.get_time_array()
        lfp = np.array(self.lfp_values)
        return times, lfp
    
    def calculate_lfp_power_spectrum(self, window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectrum of LFP signal.
        
        Args:
            window_size: Window size for FFT (uses full signal if None)
            
        Returns:
            Frequencies and power spectral density
        """
        if len(self.lfp_values) < 10:
            return np.array([]), np.array([])
        
        lfp_signal = np.array(self.lfp_values)
        if window_size is not None:
            lfp_signal = lfp_signal[-window_size:]
        
        # Calculate power spectrum
        fft_values = np.fft.fft(lfp_signal)
        power = np.abs(fft_values) ** 2
        
        # Frequency array
        dt = self.sampling_interval / 1000.0  # Convert to seconds
        freqs = np.fft.fftfreq(len(lfp_signal), dt)
        
        # Return positive frequencies only
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        return positive_freqs, positive_power
    
    def __str__(self) -> str:
        """String representation."""
        return (f"LFPProbe(id={self.probe_id}, "
                f"neurons={len(self.target_ids)}, "
                f"position={self.probe_position}, "
                f"samples={self.samples_collected})")
