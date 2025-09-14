"""
Spike detection and recording probes for neural activity monitoring.

This module provides spike detection capabilities for monitoring
action potential generation and analyzing spike timing patterns
in neural networks.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from collections import defaultdict

from .base_probe import BaseProbe, ProbeType, ProbeData
from ..neurons.base_neuron import BaseNeuron


class SpikeProbe(BaseProbe):
    """
    Probe for detecting and recording spike events from neurons.
    
    Records precise spike timing and provides analysis capabilities
    for spike train statistics and firing rate calculations.
    """
    
    def __init__(self, probe_id: str, target_neurons: Union[str, List[str]],
                 detection_threshold: float = -30.0,
                 min_spike_interval: float = 1.0,
                 record_waveforms: bool = False,
                 waveform_window: float = 5.0):
        """
        Initialize spike probe.
        
        Args:
            probe_id: Unique identifier for probe
            target_neurons: Neuron ID(s) to monitor
            detection_threshold: Spike detection threshold (mV)
            min_spike_interval: Minimum time between spikes (ms)
            record_waveforms: Whether to record spike waveforms
            waveform_window: Window around spike for waveform (ms)
        """
        # Use high sampling rate for spike detection
        super().__init__(probe_id, ProbeType.SPIKE, target_neurons, sampling_interval=0.1)
        
        # Spike detection parameters
        self.detection_threshold = detection_threshold
        self.min_spike_interval = min_spike_interval
        self.record_waveforms = record_waveforms
        self.waveform_window = waveform_window
        
        # Spike data storage
        self.spike_times: Dict[str, List[float]] = defaultdict(list)
        self.spike_waveforms: Dict[str, List[np.ndarray]] = defaultdict(list)
        
        # Detection state for each neuron
        self.last_spike_times: Dict[str, float] = {}
        self.previous_voltages: Dict[str, float] = {}
        
        # Reference to neuron objects
        self.neuron_objects: Dict[str, BaseNeuron] = {}
        
        # Statistics
        self.spike_counts: Dict[str, int] = defaultdict(int)
        self.detection_stats = {
            'total_spikes': 0,
            'false_positives': 0,
            'missed_spikes': 0
        }
    
    def register_neuron_object(self, neuron_id: str, neuron: BaseNeuron) -> None:
        """
        Register neuron object for spike detection.
        
        Args:
            neuron_id: ID of the neuron
            neuron: Neuron object reference
        """
        self.neuron_objects[neuron_id] = neuron
        self.last_spike_times[neuron_id] = -self.min_spike_interval
        
        # Initialize voltage history
        if neuron_id not in self.previous_voltages:
            self.previous_voltages[neuron_id] = neuron.get_membrane_potential()
    
    def record(self, current_time: float) -> None:
        """
        Detect and record spike events at current time.
        
        Args:
            current_time: Current simulation time (ms)
        """
        self.current_time = current_time
        
        # Check each target neuron for spikes
        detected_spikes = {}
        for neuron_id in self.target_ids:
            spike_info = self._detect_spike(neuron_id, current_time)
            if spike_info is not None:
                detected_spikes[neuron_id] = spike_info
        
        # Record detected spikes
        if detected_spikes:
            self._record_sample(current_time, detected_spikes)
    
    def _detect_spike(self, neuron_id: str, current_time: float) -> Optional[Dict[str, Any]]:
        """
        Detect spike in target neuron.
        
        Args:
            neuron_id: ID of target neuron
            current_time: Current simulation time
            
        Returns:
            Spike information if detected, None otherwise
        """
        if neuron_id not in self.neuron_objects:
            return None
        
        neuron = self.neuron_objects[neuron_id]
        current_voltage = neuron.get_membrane_potential()
        
        # Check if neuron has built-in spike detection
        if hasattr(neuron, 'has_spiked') and neuron.has_spiked():
            spike_time = neuron.get_last_spike_time()
            if spike_time is not None and spike_time >= self.last_spike_times[neuron_id]:
                return self._record_detected_spike(neuron_id, spike_time, current_voltage)
        
        # Fallback to threshold-based detection
        previous_voltage = self.previous_voltages.get(neuron_id, current_voltage)
        
        # Detect threshold crossing with positive slope
        if (previous_voltage < self.detection_threshold and 
            current_voltage >= self.detection_threshold and
            current_time - self.last_spike_times[neuron_id] >= self.min_spike_interval):
            
            return self._record_detected_spike(neuron_id, current_time, current_voltage)
        
        # Update voltage history
        self.previous_voltages[neuron_id] = current_voltage
        return None
    
    def _record_detected_spike(self, neuron_id: str, spike_time: float, 
                             voltage: float) -> Dict[str, Any]:
        """
        Record a detected spike.
        
        Args:
            neuron_id: ID of spiking neuron
            spike_time: Time of spike
            voltage: Voltage at spike detection
            
        Returns:
            Spike information dictionary
        """
        # Update spike records
        self.spike_times[neuron_id].append(spike_time)
        self.spike_counts[neuron_id] += 1
        self.last_spike_times[neuron_id] = spike_time
        self.detection_stats['total_spikes'] += 1
        
        spike_info = {
            'neuron_id': neuron_id,
            'spike_time': spike_time,
            'peak_voltage': voltage,
            'isi': (spike_time - self.spike_times[neuron_id][-2] 
                   if len(self.spike_times[neuron_id]) > 1 else None)
        }
        
        # Record waveform if enabled
        if self.record_waveforms:
            waveform = self._extract_waveform(neuron_id, spike_time)
            if waveform is not None:
                self.spike_waveforms[neuron_id].append(waveform)
                spike_info['waveform'] = waveform
        
        return spike_info
    
    def _extract_waveform(self, neuron_id: str, spike_time: float) -> Optional[np.ndarray]:
        """
        Extract spike waveform around spike time.
        
        Args:
            neuron_id: ID of spiking neuron
            spike_time: Time of spike
            
        Returns:
            Waveform array or None if unavailable
        """
        # This would require access to recent voltage history
        # For now, return placeholder - would need voltage probe integration
        return np.array([])
    
    def get_measurement_value(self, target_id: str) -> Any:
        """Get spike detection status."""
        if target_id in self.spike_times:
            recent_spikes = [t for t in self.spike_times[target_id] 
                           if t > self.current_time - 1.0]  # Last 1ms
            return len(recent_spikes) > 0
        return False
    
    def get_spike_times(self, neuron_id: Optional[str] = None) -> Union[Dict[str, List[float]], List[float]]:
        """
        Get spike times for neuron(s).
        
        Args:
            neuron_id: Specific neuron ID (returns all if None)
            
        Returns:
            Spike times for specified neuron or all neurons
        """
        if neuron_id is not None:
            return self.spike_times.get(neuron_id, []).copy()
        else:
            return {nid: times.copy() for nid, times in self.spike_times.items()}
    
    def get_spike_counts(self) -> Dict[str, int]:
        """Get spike counts for all neurons."""
        return dict(self.spike_counts)
    
    def calculate_firing_rates(self, time_window: Optional[float] = None,
                              sliding_window: bool = False,
                              window_step: float = 100.0) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate firing rates for all monitored neurons.
        
        Args:
            time_window: Time window for rate calculation (ms, uses full recording if None)
            sliding_window: Calculate sliding window rates
            window_step: Step size for sliding window (ms)
            
        Returns:
            Firing rates (Hz) for each neuron
        """
        rates = {}
        
        for neuron_id, spike_times in self.spike_times.items():
            if not spike_times:
                rates[neuron_id] = 0.0 if not sliding_window else np.array([])
                continue
            
            if time_window is None:
                # Use full recording duration
                if len(spike_times) > 1:
                    duration = spike_times[-1] - spike_times[0]
                    rate = (len(spike_times) - 1) * 1000.0 / duration if duration > 0 else 0.0
                else:
                    rate = 0.0
                rates[neuron_id] = rate
            elif not sliding_window:
                # Fixed time window from end
                recent_time = self.current_time - time_window
                recent_spikes = [t for t in spike_times if t > recent_time]
                rate = len(recent_spikes) * 1000.0 / time_window
                rates[neuron_id] = rate
            else:
                # Sliding window analysis
                if not spike_times:
                    rates[neuron_id] = np.array([])
                    continue
                
                start_time = spike_times[0]
                end_time = spike_times[-1]
                
                if end_time - start_time < time_window:
                    rates[neuron_id] = np.array([])
                    continue
                
                # Calculate sliding window rates
                window_centers = np.arange(start_time + time_window/2, 
                                         end_time - time_window/2 + window_step, 
                                         window_step)
                window_rates = []
                
                for center in window_centers:
                    window_start = center - time_window/2
                    window_end = center + time_window/2
                    window_spikes = [t for t in spike_times 
                                   if window_start <= t <= window_end]
                    rate = len(window_spikes) * 1000.0 / time_window
                    window_rates.append(rate)
                
                rates[neuron_id] = np.array(window_rates)
        
        return rates
    
    def calculate_isi_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate inter-spike interval statistics.
        
        Returns:
            ISI statistics for each neuron
        """
        isi_stats = {}
        
        for neuron_id, spike_times in self.spike_times.items():
            if len(spike_times) < 2:
                isi_stats[neuron_id] = {
                    'mean_isi': 0.0, 'std_isi': 0.0, 'cv_isi': 0.0, 
                    'min_isi': 0.0, 'max_isi': 0.0, 'count': 0
                }
                continue
            
            isis = np.diff(spike_times)
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            cv_isi = std_isi / mean_isi if mean_isi > 0 else 0.0
            
            isi_stats[neuron_id] = {
                'mean_isi': mean_isi,
                'std_isi': std_isi,
                'cv_isi': cv_isi,
                'min_isi': np.min(isis),
                'max_isi': np.max(isis),
                'count': len(isis)
            }
        
        return isi_stats
    
    def detect_bursts(self, neuron_id: str, burst_threshold: float = 100.0,
                     min_burst_duration: float = 10.0,
                     max_intraburst_isi: float = 10.0) -> List[Dict[str, Any]]:
        """
        Detect burst events in spike train.
        
        Args:
            neuron_id: Target neuron ID
            burst_threshold: Minimum firing rate for burst (Hz)
            min_burst_duration: Minimum burst duration (ms)
            max_intraburst_isi: Maximum ISI within burst (ms)
            
        Returns:
            List of detected bursts with timing and statistics
        """
        if neuron_id not in self.spike_times or len(self.spike_times[neuron_id]) < 3:
            return []
        
        spike_times = self.spike_times[neuron_id]
        isis = np.diff(spike_times)
        
        bursts = []
        burst_start = None
        burst_spikes = []
        
        for i, (spike_time, isi) in enumerate(zip(spike_times[:-1], isis)):
            if isi <= max_intraburst_isi:
                if burst_start is None:
                    burst_start = spike_time
                    burst_spikes = [spike_time]
                burst_spikes.append(spike_times[i + 1])
            else:
                # End of potential burst
                if burst_start is not None and len(burst_spikes) >= 3:
                    burst_end = burst_spikes[-1]
                    duration = burst_end - burst_start
                    
                    if duration >= min_burst_duration:
                        rate = (len(burst_spikes) - 1) * 1000.0 / duration
                        if rate >= burst_threshold:
                            bursts.append({
                                'start_time': burst_start,
                                'end_time': burst_end,
                                'duration': duration,
                                'spike_count': len(burst_spikes),
                                'firing_rate': rate,
                                'spike_times': burst_spikes.copy()
                            })
                
                burst_start = None
                burst_spikes = []
        
        return bursts
    
    def get_spike_waveforms(self, neuron_id: str) -> List[np.ndarray]:
        """Get recorded spike waveforms for neuron."""
        return self.spike_waveforms.get(neuron_id, []).copy()
    
    def set_detection_parameters(self, threshold: Optional[float] = None,
                               min_interval: Optional[float] = None) -> None:
        """
        Update spike detection parameters.
        
        Args:
            threshold: New detection threshold (mV)
            min_interval: New minimum spike interval (ms)
        """
        if threshold is not None:
            self.detection_threshold = threshold
        if min_interval is not None:
            self.min_spike_interval = min_interval
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get spike detection performance statistics."""
        total_neurons = len(self.target_ids)
        active_neurons = len([nid for nid in self.target_ids if self.spike_counts[nid] > 0])
        
        return {
            'total_spikes_detected': self.detection_stats['total_spikes'],
            'neurons_monitored': total_neurons,
            'active_neurons': active_neurons,
            'average_spikes_per_neuron': (self.detection_stats['total_spikes'] / total_neurons 
                                        if total_neurons > 0 else 0.0),
            'detection_threshold': self.detection_threshold,
            'min_spike_interval': self.min_spike_interval
        }
    
    def export_spike_data(self, format: str = 'dict') -> Union[Dict, List]:
        """
        Export spike data in specified format.
        
        Args:
            format: Export format ('dict', 'list', 'raster')
            
        Returns:
            Spike data in requested format
        """
        if format == 'raster':
            # Return list of (neuron_id, spike_time) tuples
            raster = []
            for neuron_id, spike_times in self.spike_times.items():
                for spike_time in spike_times:
                    raster.append((neuron_id, spike_time))
            return sorted(raster, key=lambda x: x[1])  # Sort by time
        elif format == 'list':
            return list(self.spike_times.items())
        else:
            return dict(self.spike_times)
    
    def __str__(self) -> str:
        """String representation."""
        total_spikes = sum(self.spike_counts.values())
        return (f"SpikeProbe(id={self.probe_id}, "
                f"neurons={len(self.target_ids)}, "
                f"threshold={self.detection_threshold}mV, "
                f"total_spikes={total_spikes})")


class SpikeDetector(SpikeProbe):
    """
    Specialized spike detector with advanced detection algorithms.
    
    Provides enhanced spike detection capabilities with multiple
    detection methods and adaptive thresholding.
    """
    
    def __init__(self, probe_id: str, target_neurons: Union[str, List[str]],
                 detection_method: str = 'threshold',
                 adaptive_threshold: bool = False,
                 **kwargs):
        """
        Initialize advanced spike detector.
        
        Args:
            probe_id: Unique identifier
            target_neurons: Neurons to monitor
            detection_method: Detection algorithm ('threshold', 'derivative', 'template')
            adaptive_threshold: Use adaptive threshold
        """
        super().__init__(probe_id, target_neurons, **kwargs)
        
        self.detection_method = detection_method
        self.adaptive_threshold = adaptive_threshold
        
        # Adaptive threshold parameters
        self.threshold_adaptation_rate = 0.01
        self.adaptive_thresholds: Dict[str, float] = {}
        
        # Template matching (if using template method)
        self.spike_templates: Dict[str, np.ndarray] = {}
    
    def _detect_spike(self, neuron_id: str, current_time: float) -> Optional[Dict[str, Any]]:
        """Enhanced spike detection with multiple methods."""
        if self.detection_method == 'template':
            return self._detect_spike_template(neuron_id, current_time)
        elif self.detection_method == 'derivative':
            return self._detect_spike_derivative(neuron_id, current_time)
        else:
            # Use base threshold detection with optional adaptation
            if self.adaptive_threshold:
                self._update_adaptive_threshold(neuron_id)
            return super()._detect_spike(neuron_id, current_time)
    
    def _detect_spike_derivative(self, neuron_id: str, current_time: float) -> Optional[Dict[str, Any]]:
        """Detect spikes using voltage derivative."""
        # Implementation would require voltage history
        # Placeholder for derivative-based detection
        return super()._detect_spike(neuron_id, current_time)
    
    def _detect_spike_template(self, neuron_id: str, current_time: float) -> Optional[Dict[str, Any]]:
        """Detect spikes using template matching."""
        # Implementation would require template correlation
        # Placeholder for template-based detection
        return super()._detect_spike(neuron_id, current_time)
    
    def _update_adaptive_threshold(self, neuron_id: str) -> None:
        """Update adaptive detection threshold."""
        if neuron_id not in self.neuron_objects:
            return
        
        current_voltage = self.neuron_objects[neuron_id].get_membrane_potential()
        
        if neuron_id not in self.adaptive_thresholds:
            self.adaptive_thresholds[neuron_id] = self.detection_threshold
        
        # Simple adaptation: move threshold toward current voltage
        current_threshold = self.adaptive_thresholds[neuron_id]
        target_threshold = current_voltage + (self.detection_threshold - current_voltage) * 0.1
        
        self.adaptive_thresholds[neuron_id] = (
            current_threshold + 
            (target_threshold - current_threshold) * self.threshold_adaptation_rate
        )
