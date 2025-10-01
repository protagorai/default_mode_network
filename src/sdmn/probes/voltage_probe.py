"""
Voltage probe for recording membrane potentials from neurons.

This module provides voltage monitoring capabilities for tracking
membrane potential changes over time in individual neurons or
groups of neurons.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from sdmn.probes.base_probe import BaseProbe, ProbeType
from sdmn.neurons.base_neuron import BaseNeuron


class VoltageProbe(BaseProbe):
    """
    Probe for recording membrane potential from neurons.
    
    Records voltage traces from one or more target neurons with
    configurable sampling rates and optional filtering.
    """
    
    def __init__(self, probe_id: str, target_neurons: Union[str, List[str]], 
                 sampling_interval: float = 0.1, 
                 voltage_range: Optional[tuple] = None,
                 enable_filtering: bool = False,
                 filter_cutoff: float = 100.0):
        """
        Initialize voltage probe.
        
        Args:
            probe_id: Unique identifier for probe
            target_neurons: Neuron ID(s) to monitor
            sampling_interval: Time between samples (ms)
            voltage_range: Optional (min, max) voltage range for clipping
            enable_filtering: Enable low-pass filtering
            filter_cutoff: Filter cutoff frequency (Hz)
        """
        super().__init__(probe_id, ProbeType.VOLTAGE, target_neurons, sampling_interval)
        
        # Voltage-specific settings
        self.voltage_range = voltage_range
        self.enable_filtering = enable_filtering
        self.filter_cutoff = filter_cutoff
        
        # Filtering state (simple first-order low-pass)
        self.filter_alpha = self._calculate_filter_alpha()
        self.filtered_voltages: Dict[str, float] = {}
        
        # Reference to neuron objects (populated during simulation)
        self.neuron_objects: Dict[str, BaseNeuron] = {}
        
        # Statistics
        self.voltage_stats: Dict[str, Dict[str, float]] = {}
        
    def _calculate_filter_alpha(self) -> float:
        """Calculate filter coefficient for low-pass filter."""
        if not self.enable_filtering or self.filter_cutoff <= 0:
            return 1.0
        
        # First-order low-pass filter coefficient
        dt_sec = self.sampling_interval / 1000.0  # Convert ms to seconds
        rc = 1.0 / (2 * np.pi * self.filter_cutoff)
        return dt_sec / (rc + dt_sec)
    
    def register_neuron_object(self, neuron_id: str, neuron: BaseNeuron) -> None:
        """
        Register neuron object for direct access.
        
        Args:
            neuron_id: ID of the neuron
            neuron: Neuron object reference
        """
        self.neuron_objects[neuron_id] = neuron
        
        # Initialize filtering state
        if self.enable_filtering:
            initial_voltage = neuron.get_membrane_potential()
            self.filtered_voltages[neuron_id] = initial_voltage
    
    def record(self, current_time: float) -> None:
        """
        Record voltage data at current time.
        
        Args:
            current_time: Current simulation time (ms)
        """
        self.current_time = current_time
        
        if not self.should_record(current_time):
            return
        
        # Record from all target neurons
        voltages = {}
        for neuron_id in self.target_ids:
            voltage = self.get_measurement_value(neuron_id)
            if voltage is not None:
                voltages[neuron_id] = voltage
        
        if voltages:
            self._record_sample(current_time, voltages)
            self._update_statistics(voltages)
    
    def get_measurement_value(self, neuron_id: str) -> Optional[float]:
        """
        Get membrane potential from target neuron.
        
        Args:
            neuron_id: ID of target neuron
            
        Returns:
            Current membrane potential or None if unavailable
        """
        if neuron_id not in self.neuron_objects:
            return None
        
        neuron = self.neuron_objects[neuron_id]
        raw_voltage = neuron.get_membrane_potential()
        
        # Apply voltage range clipping if specified
        if self.voltage_range is not None:
            raw_voltage = np.clip(raw_voltage, self.voltage_range[0], self.voltage_range[1])
        
        # Apply filtering if enabled
        if self.enable_filtering:
            if neuron_id in self.filtered_voltages:
                # First-order low-pass filter: y[n] = α*x[n] + (1-α)*y[n-1]
                filtered_voltage = (self.filter_alpha * raw_voltage + 
                                  (1 - self.filter_alpha) * self.filtered_voltages[neuron_id])
                self.filtered_voltages[neuron_id] = filtered_voltage
                return filtered_voltage
            else:
                # Initialize filter
                self.filtered_voltages[neuron_id] = raw_voltage
                return raw_voltage
        
        return raw_voltage
    
    def _update_statistics(self, voltages: Dict[str, float]) -> None:
        """Update voltage statistics."""
        for neuron_id, voltage in voltages.items():
            if neuron_id not in self.voltage_stats:
                self.voltage_stats[neuron_id] = {
                    'min': voltage,
                    'max': voltage,
                    'mean': voltage,
                    'count': 1,
                    'sum': voltage,
                    'sum_sq': voltage * voltage
                }
            else:
                stats = self.voltage_stats[neuron_id]
                stats['min'] = min(stats['min'], voltage)
                stats['max'] = max(stats['max'], voltage)
                stats['count'] += 1
                stats['sum'] += voltage
                stats['sum_sq'] += voltage * voltage
                stats['mean'] = stats['sum'] / stats['count']
    
    def get_voltage_traces(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get voltage traces for all neurons.
        
        Returns:
            Dictionary mapping neuron IDs to time/voltage arrays
        """
        traces = {}
        times = self.data.get_time_array()
        
        for neuron_id in self.target_ids:
            # Extract voltage values for this neuron
            neuron_voltages = []
            for value_dict in self.data.values:
                if isinstance(value_dict, dict) and neuron_id in value_dict:
                    neuron_voltages.append(value_dict[neuron_id])
                else:
                    neuron_voltages.append(np.nan)  # Missing data
            
            traces[neuron_id] = {
                'time': times,
                'voltage': np.array(neuron_voltages)
            }
        
        return traces
    
    def get_voltage_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get voltage statistics for all monitored neurons.
        
        Returns:
            Dictionary with voltage statistics per neuron
        """
        stats = {}
        for neuron_id, raw_stats in self.voltage_stats.items():
            if raw_stats['count'] > 0:
                variance = (raw_stats['sum_sq'] / raw_stats['count'] - 
                           (raw_stats['mean']) ** 2)
                std_dev = np.sqrt(max(0, variance))
                
                stats[neuron_id] = {
                    'mean': raw_stats['mean'],
                    'std': std_dev,
                    'min': raw_stats['min'],
                    'max': raw_stats['max'],
                    'range': raw_stats['max'] - raw_stats['min'],
                    'count': raw_stats['count']
                }
            else:
                stats[neuron_id] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 
                    'max': 0.0, 'range': 0.0, 'count': 0
                }
        
        return stats
    
    def detect_spikes_from_voltage(self, neuron_id: str, 
                                  threshold: float = -30.0,
                                  min_interval: float = 2.0) -> List[float]:
        """
        Detect spikes from voltage trace using threshold crossing.
        
        Args:
            neuron_id: ID of target neuron
            threshold: Spike detection threshold (mV)
            min_interval: Minimum interval between spikes (ms)
            
        Returns:
            List of spike times
        """
        traces = self.get_voltage_traces()
        if neuron_id not in traces:
            return []
        
        times = traces[neuron_id]['time']
        voltages = traces[neuron_id]['voltage']
        
        if len(times) < 2:
            return []
        
        spike_times = []
        last_spike_time = -min_interval
        
        # Find threshold crossings with positive slope
        for i in range(1, len(voltages)):
            if (voltages[i-1] < threshold and voltages[i] >= threshold and
                times[i] - last_spike_time >= min_interval):
                spike_times.append(times[i])
                last_spike_time = times[i]
        
        return spike_times
    
    def calculate_voltage_derivative(self, neuron_id: str) -> Dict[str, np.ndarray]:
        """
        Calculate time derivative of membrane potential.
        
        Args:
            neuron_id: ID of target neuron
            
        Returns:
            Dictionary with time and dV/dt arrays
        """
        traces = self.get_voltage_traces()
        if neuron_id not in traces:
            return {'time': np.array([]), 'dvdt': np.array([])}
        
        times = traces[neuron_id]['time']
        voltages = traces[neuron_id]['voltage']
        
        if len(times) < 2:
            return {'time': np.array([]), 'dvdt': np.array([])}
        
        # Calculate derivative using central differences
        dt = np.diff(times)
        dv = np.diff(voltages)
        dvdt = dv / dt
        
        # Use midpoint times for derivative
        mid_times = times[:-1] + dt / 2
        
        return {'time': mid_times, 'dvdt': dvdt}
    
    def get_resting_potential(self, neuron_id: str, 
                             percentile: float = 10.0) -> Optional[float]:
        """
        Estimate resting potential as low percentile of voltage distribution.
        
        Args:
            neuron_id: ID of target neuron
            percentile: Percentile to use for resting potential estimate
            
        Returns:
            Estimated resting potential
        """
        traces = self.get_voltage_traces()
        if neuron_id not in traces:
            return None
        
        voltages = traces[neuron_id]['voltage']
        voltages = voltages[~np.isnan(voltages)]  # Remove NaN values
        
        if len(voltages) == 0:
            return None
        
        return np.percentile(voltages, percentile)
    
    def set_voltage_range(self, min_voltage: float, max_voltage: float) -> None:
        """Set voltage clipping range."""
        self.voltage_range = (min_voltage, max_voltage)
    
    def set_filtering(self, enable: bool, cutoff_freq: Optional[float] = None) -> None:
        """
        Enable/disable voltage filtering.
        
        Args:
            enable: Enable filtering
            cutoff_freq: Filter cutoff frequency (Hz)
        """
        self.enable_filtering = enable
        if cutoff_freq is not None:
            self.filter_cutoff = cutoff_freq
            self.filter_alpha = self._calculate_filter_alpha()
        
        # Reset filter state
        self.filtered_voltages.clear()
    
    def export_voltage_data(self, format: str = 'dict') -> Union[Dict, np.ndarray]:
        """
        Export voltage data in specified format.
        
        Args:
            format: Export format ('dict', 'array', 'traces')
            
        Returns:
            Voltage data in requested format
        """
        if format == 'traces':
            return self.get_voltage_traces()
        elif format == 'array':
            traces = self.get_voltage_traces()
            # Stack all neuron voltages into single array
            voltage_arrays = [traces[nid]['voltage'] for nid in self.target_ids 
                            if nid in traces]
            if voltage_arrays:
                return np.column_stack([self.data.get_time_array()] + voltage_arrays)
            else:
                return np.array([])
        else:
            return super().export_data(format)
    
    def __str__(self) -> str:
        """String representation."""
        filter_str = f", filtered@{self.filter_cutoff}Hz" if self.enable_filtering else ""
        return (f"VoltageProbe(id={self.probe_id}, "
                f"neurons={len(self.target_ids)}, "
                f"rate={1000.0/self.sampling_interval:.1f}Hz"
                f"{filter_str}, "
                f"samples={self.samples_collected})")
