"""
Base probe interface and common functionality for the monitoring system.

This module defines the abstract base class and interfaces that all
probe types must implement for consistent data collection and analysis.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time

from sdmn.core.state_manager import StateSerializable


class ProbeType(Enum):
    """Types of probes supported by the framework."""
    VOLTAGE = "voltage"
    SPIKE = "spike"
    CURRENT = "current"
    CONDUCTANCE = "conductance"
    POPULATION_ACTIVITY = "population_activity"
    LFP = "lfp"  # Local Field Potential
    NETWORK_ACTIVITY = "network_activity"
    CONNECTIVITY = "connectivity"
    CUSTOM = "custom"


@dataclass
class ProbeData:
    """Container for probe recorded data."""
    probe_id: str
    probe_type: ProbeType
    target_ids: List[str]
    timestamps: List[float] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_sample(self, timestamp: float, value: Any) -> None:
        """Add a data sample."""
        self.timestamps.append(timestamp)
        self.values.append(value)
    
    def get_data_array(self) -> np.ndarray:
        """Get values as numpy array."""
        return np.array(self.values)
    
    def get_time_array(self) -> np.ndarray:
        """Get timestamps as numpy array."""
        return np.array(self.timestamps)
    
    def get_sample_count(self) -> int:
        """Get number of recorded samples."""
        return len(self.timestamps)
    
    def get_time_range(self) -> tuple:
        """Get time range of recordings."""
        if not self.timestamps:
            return (0.0, 0.0)
        return (min(self.timestamps), max(self.timestamps))
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self.timestamps.clear()
        self.values.clear()


class BaseProbe(StateSerializable, ABC):
    """
    Abstract base class for all monitoring probes.
    
    Defines the standard interface that all probe types must implement
    for consistent data collection and analysis within the simulation
    framework.
    """
    
    def __init__(self, probe_id: str, probe_type: ProbeType, 
                 target_ids: Union[str, List[str]], 
                 sampling_interval: float = 1.0):
        """
        Initialize the base probe.
        
        Args:
            probe_id: Unique identifier for this probe
            probe_type: Type of probe
            target_ids: ID(s) of target objects to monitor
            sampling_interval: Time interval between samples (ms)
        """
        self.probe_id = probe_id
        self.probe_type = probe_type
        self.target_ids = [target_ids] if isinstance(target_ids, str) else target_ids
        self.sampling_interval = sampling_interval
        
        # Recording state
        self.is_recording = False
        self.last_sample_time = -float('inf')
        self.current_time = 0.0
        
        # Data storage
        self.data = ProbeData(probe_id, probe_type, self.target_ids)
        
        # Callbacks for real-time processing
        self.data_callbacks: List[Callable[[ProbeData, float, Any], None]] = []
        
        # Configuration
        self.max_samples = None  # No limit by default
        self.buffer_mode = 'circular'  # 'circular' or 'growing'
        
        # Statistics tracking
        self.samples_collected = 0
        self.recording_start_time = None
        self.recording_wall_time_start = None
    
    @abstractmethod
    def record(self, current_time: float) -> None:
        """
        Record data at the current simulation time.
        
        Args:
            current_time: Current simulation time (ms)
        """
        pass
    
    @abstractmethod
    def get_measurement_value(self, target_id: str) -> Any:
        """
        Get the current measurement value from a target.
        
        Args:
            target_id: ID of target to measure
            
        Returns:
            Current measurement value
        """
        pass
    
    def should_record(self, current_time: float) -> bool:
        """
        Check if probe should record at current time.
        
        Args:
            current_time: Current simulation time (ms)
            
        Returns:
            True if should record, False otherwise
        """
        if not self.is_recording:
            return False
        
        return (current_time - self.last_sample_time) >= self.sampling_interval
    
    def start_recording(self, start_time: Optional[float] = None) -> None:
        """
        Start recording data.
        
        Args:
            start_time: Optional start time (uses current time if None)
        """
        self.is_recording = True
        self.recording_start_time = start_time or self.current_time
        self.recording_wall_time_start = time.time()
        self.last_sample_time = self.recording_start_time - self.sampling_interval
        
        # Clear existing data if starting new recording
        if start_time is not None:
            self.data.clear()
            self.samples_collected = 0
    
    def stop_recording(self) -> None:
        """Stop recording data."""
        self.is_recording = False
    
    def pause_recording(self) -> None:
        """Pause recording (can be resumed)."""
        self.is_recording = False
    
    def resume_recording(self) -> None:
        """Resume paused recording.""" 
        self.is_recording = True
    
    def is_recording_active(self) -> bool:
        """Check if probe is currently recording."""
        return self.is_recording
    
    def add_data_callback(self, callback: Callable[[ProbeData, float, Any], None]) -> None:
        """
        Add callback for real-time data processing.
        
        Args:
            callback: Function called when new data is recorded
        """
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable[[ProbeData, float, Any], None]) -> None:
        """Remove data callback."""
        try:
            self.data_callbacks.remove(callback)
        except ValueError:
            pass  # Callback not found
    
    def _record_sample(self, timestamp: float, value: Any) -> None:
        """
        Internal method to record a sample.
        
        Args:
            timestamp: Sample timestamp
            value: Sample value
        """
        # Handle buffer limits
        if self.max_samples is not None:
            if self.buffer_mode == 'circular' and len(self.data.values) >= self.max_samples:
                # Remove oldest sample
                self.data.timestamps.pop(0)
                self.data.values.pop(0)
            elif self.buffer_mode == 'growing' and len(self.data.values) >= self.max_samples:
                # Stop recording when limit reached
                self.stop_recording()
                return
        
        # Add new sample
        self.data.add_sample(timestamp, value)
        self.samples_collected += 1
        self.last_sample_time = timestamp
        
        # Call registered callbacks
        for callback in self.data_callbacks:
            callback(self.data, timestamp, value)
    
    def get_data(self) -> ProbeData:
        """Get recorded data."""
        return self.data
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """
        Get recording statistics.
        
        Returns:
            Dictionary with recording statistics
        """
        time_range = self.data.get_time_range()
        wall_time_elapsed = (time.time() - self.recording_wall_time_start 
                           if self.recording_wall_time_start else 0.0)
        
        return {
            'probe_id': self.probe_id,
            'probe_type': self.probe_type.value,
            'is_recording': self.is_recording,
            'samples_collected': self.samples_collected,
            'recording_duration': time_range[1] - time_range[0],
            'wall_time_elapsed': wall_time_elapsed,
            'sampling_rate_hz': 1000.0 / self.sampling_interval if self.sampling_interval > 0 else 0.0,
            'actual_samples_per_second': (self.samples_collected / wall_time_elapsed 
                                        if wall_time_elapsed > 0 else 0.0),
            'time_range': time_range,
            'target_count': len(self.target_ids),
            'buffer_usage': len(self.data.values)
        }
    
    def set_sampling_interval(self, interval: float) -> None:
        """
        Set sampling interval.
        
        Args:
            interval: New sampling interval (ms)
        """
        if interval <= 0:
            raise ValueError("Sampling interval must be positive")
        self.sampling_interval = interval
    
    def set_buffer_limits(self, max_samples: Optional[int], mode: str = 'circular') -> None:
        """
        Set buffer limits and mode.
        
        Args:
            max_samples: Maximum samples to store (None for unlimited)
            mode: 'circular' (overwrite old) or 'growing' (stop when full)
        """
        if mode not in ['circular', 'growing']:
            raise ValueError("Buffer mode must be 'circular' or 'growing'")
        
        self.max_samples = max_samples
        self.buffer_mode = mode
    
    def export_data(self, format: str = 'dict') -> Union[Dict, np.ndarray]:
        """
        Export recorded data in specified format.
        
        Args:
            format: Export format ('dict', 'array', 'structured_array')
            
        Returns:
            Data in requested format
        """
        if format == 'dict':
            return {
                'probe_id': self.probe_id,
                'probe_type': self.probe_type.value,
                'target_ids': self.target_ids,
                'timestamps': self.data.get_time_array(),
                'values': self.data.get_data_array(),
                'metadata': self.data.metadata
            }
        elif format == 'array':
            return np.column_stack([self.data.get_time_array(), self.data.get_data_array()])
        elif format == 'structured_array':
            return np.array(
                list(zip(self.data.timestamps, self.data.values)),
                dtype=[('time', 'f8'), ('value', 'f8')]
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_data(self) -> None:
        """Clear all recorded data."""
        self.data.clear()
        self.samples_collected = 0
    
    def get_target_count(self) -> int:
        """Get number of monitoring targets."""
        return len(self.target_ids)
    
    def add_target(self, target_id: str) -> None:
        """Add monitoring target."""
        if target_id not in self.target_ids:
            self.target_ids.append(target_id)
            self.data.target_ids.append(target_id)
    
    def remove_target(self, target_id: str) -> None:
        """Remove monitoring target."""
        try:
            self.target_ids.remove(target_id)
            self.data.target_ids.remove(target_id)
        except ValueError:
            pass  # Target not found
    
    # StateSerializable interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get probe state for serialization."""
        return {
            'probe_id': self.probe_id,
            'probe_type': self.probe_type.value,
            'target_ids': self.target_ids.copy(),
            'sampling_interval': self.sampling_interval,
            'is_recording': self.is_recording,
            'current_time': self.current_time,
            'last_sample_time': self.last_sample_time,
            'samples_collected': self.samples_collected,
            'max_samples': self.max_samples,
            'buffer_mode': self.buffer_mode,
            'data': {
                'timestamps': self.data.timestamps.copy(),
                'values': self.data.values.copy(), 
                'metadata': self.data.metadata.copy()
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set probe state from serialization."""
        self.probe_id = state['probe_id']
        self.probe_type = ProbeType(state['probe_type'])
        self.target_ids = state['target_ids'].copy()
        self.sampling_interval = state['sampling_interval']
        self.is_recording = state['is_recording']
        self.current_time = state['current_time']
        self.last_sample_time = state['last_sample_time']
        self.samples_collected = state['samples_collected']
        self.max_samples = state.get('max_samples')
        self.buffer_mode = state.get('buffer_mode', 'circular')
        
        # Restore data
        if 'data' in state:
            data_state = state['data']
            self.data.timestamps = data_state['timestamps'].copy()
            self.data.values = data_state['values'].copy()
            self.data.metadata = data_state['metadata'].copy()
    
    def get_state_version(self) -> str:
        """Get state format version."""
        return "1.0"
    
    def __str__(self) -> str:
        """String representation of probe."""
        return (f"{self.__class__.__name__}(id={self.probe_id}, "
                f"targets={len(self.target_ids)}, "
                f"samples={self.samples_collected}, "
                f"recording={self.is_recording})")
    
    def __repr__(self) -> str:
        """Detailed representation of probe."""
        return (f"{self.__class__.__name__}("
                f"probe_id='{self.probe_id}', "
                f"probe_type={self.probe_type}, "
                f"target_ids={self.target_ids}, "
                f"sampling_interval={self.sampling_interval})")


class ProbeManager:
    """
    Manages multiple probes and coordinates their operation.
    
    Provides centralized control for starting/stopping recording,
    data collection, and probe lifecycle management.
    """
    
    def __init__(self):
        self.probes: Dict[str, BaseProbe] = {}
        self.probe_groups: Dict[str, List[str]] = {}
        self.global_recording = False
    
    def add_probe(self, probe: BaseProbe) -> None:
        """Add probe to manager."""
        self.probes[probe.probe_id] = probe
    
    def remove_probe(self, probe_id: str) -> None:
        """Remove probe from manager."""
        self.probes.pop(probe_id, None)
        
        # Remove from groups
        for group_probes in self.probe_groups.values():
            if probe_id in group_probes:
                group_probes.remove(probe_id)
    
    def get_probe(self, probe_id: str) -> Optional[BaseProbe]:
        """Get probe by ID."""
        return self.probes.get(probe_id)
    
    def create_probe_group(self, group_name: str, probe_ids: List[str]) -> None:
        """Create named group of probes."""
        self.probe_groups[group_name] = probe_ids.copy()
    
    def start_recording_all(self, start_time: Optional[float] = None) -> None:
        """Start recording on all probes."""
        self.global_recording = True
        for probe in self.probes.values():
            probe.start_recording(start_time)
    
    def stop_recording_all(self) -> None:
        """Stop recording on all probes.""" 
        self.global_recording = False
        for probe in self.probes.values():
            probe.stop_recording()
    
    def start_recording_group(self, group_name: str, start_time: Optional[float] = None) -> None:
        """Start recording on probe group."""
        if group_name in self.probe_groups:
            for probe_id in self.probe_groups[group_name]:
                if probe_id in self.probes:
                    self.probes[probe_id].start_recording(start_time)
    
    def get_all_data(self) -> Dict[str, ProbeData]:
        """Get data from all probes."""
        return {probe_id: probe.get_data() for probe_id, probe in self.probes.items()}
    
    def get_recording_summary(self) -> Dict[str, Any]:
        """Get summary of recording status."""
        active_probes = sum(1 for p in self.probes.values() if p.is_recording_active())
        total_samples = sum(p.samples_collected for p in self.probes.values())
        
        return {
            'total_probes': len(self.probes),
            'active_probes': active_probes,
            'total_samples': total_samples,
            'global_recording': self.global_recording,
            'probe_groups': len(self.probe_groups)
        }
