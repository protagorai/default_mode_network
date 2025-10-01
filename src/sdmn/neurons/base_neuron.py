"""
Base neuron interface and common functionality.

This module defines the abstract base class and interfaces that all
neuron models must implement, ensuring consistent behavior and
interchangeability across different neuron types.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from sdmn.core.state_manager import StateSerializable


class NeuronType(Enum):
    """Types of neurons supported by the framework."""
    INTEGRATE_FIRE = "integrate_fire"
    LEAKY_INTEGRATE_FIRE = "leaky_integrate_fire"
    HODGKIN_HUXLEY = "hodgkin_huxley"
    IZHIKEVICH = "izhikevich"
    CUSTOM = "custom"


@dataclass
class NeuronParameters:
    """Base parameters for neuron models."""
    neuron_type: NeuronType
    dt: float = 0.1  # Integration time step (ms)
    
    # Common biophysical parameters
    v_rest: float = -70.0        # Resting potential (mV)
    v_thresh: float = -50.0      # Spike threshold (mV)
    v_reset: float = -80.0       # Reset potential after spike (mV)
    refractory_period: float = 2.0  # Absolute refractory period (ms)
    
    # Additional parameters stored as dictionary for flexibility
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a custom parameter value."""
        return self.custom_params.get(name, default)
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set a custom parameter value."""
        self.custom_params[name] = value


@dataclass
class NeuronState:
    """Represents the current state of a neuron."""
    membrane_potential: float   # Current membrane potential (mV)
    spike_time: Optional[float] = None  # Time of last spike (ms)
    refractory_until: float = 0.0  # Time until refractory period ends (ms)
    
    # Additional state variables stored as dictionary
    state_vars: Dict[str, float] = field(default_factory=dict)
    
    def get_state_var(self, name: str, default: float = 0.0) -> float:
        """Get a state variable value."""
        return self.state_vars.get(name, default)
    
    def set_state_var(self, name: str, value: float) -> None:
        """Set a state variable value."""
        self.state_vars[name] = value
    
    def is_refractory(self, current_time: float) -> bool:
        """Check if neuron is in refractory period."""
        return current_time < self.refractory_until


class BaseNeuron(StateSerializable, ABC):
    """
    Abstract base class for all neuron models.
    
    Defines the standard interface that all neuron models must implement
    for consistent behavior and interchangeability within the simulation
    framework.
    """
    
    def __init__(self, neuron_id: str, parameters: NeuronParameters):
        """
        Initialize the base neuron.
        
        Args:
            neuron_id: Unique identifier for this neuron
            parameters: Neuron parameters
        """
        self.neuron_id = neuron_id
        self.parameters = parameters
        self.current_time = 0.0
        
        # Initialize state
        self.state = NeuronState(membrane_potential=parameters.v_rest)
        
        # Input management
        self.synaptic_inputs: List[float] = []
        self.external_input: float = 0.0
        
        # Recording/monitoring
        self.spike_times: List[float] = []
        self.voltage_history: List[Tuple[float, float]] = []  # (time, voltage)
        
        # Connections
        self.presynaptic_connections: List['Synapse'] = []
        self.postsynaptic_connections: List['Synapse'] = []
    
    @abstractmethod
    def update(self, dt: float, inputs: Optional[List[float]] = None) -> None:
        """
        Update neuron state for one time step.
        
        Args:
            dt: Time step size (ms)
            inputs: Optional list of synaptic inputs
        """
        pass
    
    @abstractmethod
    def integrate_membrane_equation(self, dt: float, total_input: float) -> float:
        """
        Integrate the membrane equation for one time step.
        
        Args:
            dt: Time step size (ms)
            total_input: Total input current
            
        Returns:
            New membrane potential
        """
        pass
    
    def get_membrane_potential(self) -> float:
        """Get current membrane potential."""
        return self.state.membrane_potential
    
    def set_membrane_potential(self, voltage: float) -> None:
        """Set membrane potential."""
        self.state.membrane_potential = voltage
    
    def has_spiked(self) -> bool:
        """Check if neuron has spiked in current time step."""
        return (self.state.spike_time is not None and 
                abs(self.state.spike_time - self.current_time) < self.parameters.dt / 2)
    
    def get_last_spike_time(self) -> Optional[float]:
        """Get time of last spike."""
        return self.state.spike_time
    
    def get_spike_times(self) -> List[float]:
        """Get all recorded spike times."""
        return self.spike_times.copy()
    
    def add_synaptic_input(self, input_current: float) -> None:
        """Add synaptic input for current time step."""
        self.synaptic_inputs.append(input_current)
    
    def set_external_input(self, input_current: float) -> None:
        """Set external input current."""
        self.external_input = input_current
    
    def get_total_input(self) -> float:
        """Calculate total input current."""
        synaptic_total = sum(self.synaptic_inputs)
        return synaptic_total + self.external_input
    
    def clear_inputs(self) -> None:
        """Clear all inputs for next time step."""
        self.synaptic_inputs.clear()
        self.external_input = 0.0
    
    def _check_spike_condition(self) -> bool:
        """Check if spike threshold is reached."""
        return (self.state.membrane_potential >= self.parameters.v_thresh and
                not self.state.is_refractory(self.current_time))
    
    def _generate_spike(self) -> None:
        """Generate action potential."""
        self.state.spike_time = self.current_time
        self.spike_times.append(self.current_time)
        self.state.membrane_potential = self.parameters.v_reset
        self.state.refractory_until = self.current_time + self.parameters.refractory_period
        
        # Propagate spike to postsynaptic connections
        self._propagate_spike()
    
    def _propagate_spike(self) -> None:
        """Propagate spike to connected synapses."""
        for synapse in self.postsynaptic_connections:
            synapse.receive_spike(self.current_time)
    
    def add_presynaptic_connection(self, synapse: 'Synapse') -> None:
        """Add presynaptic connection."""
        self.presynaptic_connections.append(synapse)
    
    def add_postsynaptic_connection(self, synapse: 'Synapse') -> None:
        """Add postsynaptic connection."""
        self.postsynaptic_connections.append(synapse)
    
    def remove_presynaptic_connection(self, synapse: 'Synapse') -> None:
        """Remove presynaptic connection."""
        try:
            self.presynaptic_connections.remove(synapse)
        except ValueError:
            pass
    
    def remove_postsynaptic_connection(self, synapse: 'Synapse') -> None:
        """Remove postsynaptic connection."""
        try:
            self.postsynaptic_connections.remove(synapse)
        except ValueError:
            pass
    
    def get_connection_count(self) -> Tuple[int, int]:
        """Get number of pre/postsynaptic connections."""
        return len(self.presynaptic_connections), len(self.postsynaptic_connections)
    
    def record_voltage(self) -> None:
        """Record current voltage with timestamp."""
        self.voltage_history.append((self.current_time, self.state.membrane_potential))
    
    def get_voltage_history(self) -> List[Tuple[float, float]]:
        """Get voltage recording history."""
        return self.voltage_history.copy()
    
    def clear_history(self) -> None:
        """Clear all recorded history."""
        self.spike_times.clear()
        self.voltage_history.clear()
    
    def get_firing_rate(self, time_window: float) -> float:
        """
        Calculate firing rate over specified time window.
        
        Args:
            time_window: Time window for rate calculation (ms)
            
        Returns:
            Firing rate in Hz
        """
        if not self.spike_times or time_window <= 0:
            return 0.0
        
        # Count spikes in recent time window
        recent_spikes = [t for t in self.spike_times 
                        if t > self.current_time - time_window]
        
        return len(recent_spikes) * 1000.0 / time_window  # Convert ms to Hz
    
    def get_isi_statistics(self) -> Dict[str, float]:
        """
        Get inter-spike interval statistics.
        
        Returns:
            Dictionary with ISI statistics
        """
        if len(self.spike_times) < 2:
            return {'mean_isi': 0.0, 'std_isi': 0.0, 'cv_isi': 0.0}
        
        isis = np.diff(self.spike_times)
        mean_isi = np.mean(isis)
        std_isi = np.std(isis)
        cv_isi = std_isi / mean_isi if mean_isi > 0 else 0.0
        
        return {
            'mean_isi': mean_isi,
            'std_isi': std_isi,
            'cv_isi': cv_isi,
            'count': len(isis)
        }
    
    def reset_neuron(self) -> None:
        """Reset neuron to initial state."""
        self.state.membrane_potential = self.parameters.v_rest
        self.state.spike_time = None
        self.state.refractory_until = 0.0
        self.state.state_vars.clear()
        self.clear_inputs()
        self.clear_history()
        self.current_time = 0.0
    
    # StateSerializable interface implementation
    def get_state(self) -> Dict[str, Any]:
        """Get neuron state for serialization."""
        return {
            'neuron_id': self.neuron_id,
            'neuron_type': self.parameters.neuron_type.value,
            'current_time': self.current_time,
            'membrane_potential': self.state.membrane_potential,
            'spike_time': self.state.spike_time,
            'refractory_until': self.state.refractory_until,
            'state_vars': self.state.state_vars.copy(),
            'spike_times': self.spike_times.copy(),
            'parameters': {
                'v_rest': self.parameters.v_rest,
                'v_thresh': self.parameters.v_thresh,
                'v_reset': self.parameters.v_reset,
                'refractory_period': self.parameters.refractory_period,
                'custom_params': self.parameters.custom_params.copy()
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set neuron state from serialization."""
        self.neuron_id = state['neuron_id']
        self.current_time = state['current_time']
        self.state.membrane_potential = state['membrane_potential']
        self.state.spike_time = state.get('spike_time')
        self.state.refractory_until = state.get('refractory_until', 0.0)
        self.state.state_vars = state.get('state_vars', {}).copy()
        self.spike_times = state.get('spike_times', []).copy()
        
        # Update parameters
        if 'parameters' in state:
            params = state['parameters']
            self.parameters.v_rest = params.get('v_rest', -70.0)
            self.parameters.v_thresh = params.get('v_thresh', -50.0)
            self.parameters.v_reset = params.get('v_reset', -80.0)
            self.parameters.refractory_period = params.get('refractory_period', 2.0)
            self.parameters.custom_params = params.get('custom_params', {}).copy()
    
    def get_state_version(self) -> str:
        """Get state format version."""
        return "1.0"
    
    def __str__(self) -> str:
        """String representation of neuron."""
        return (f"{self.__class__.__name__}(id={self.neuron_id}, "
                f"V_mem={self.state.membrane_potential:.1f}mV, "
                f"spikes={len(self.spike_times)})")
    
    def __repr__(self) -> str:
        """Detailed representation of neuron."""
        return (f"{self.__class__.__name__}("
                f"neuron_id='{self.neuron_id}', "
                f"membrane_potential={self.state.membrane_potential:.2f}, "
                f"parameters={self.parameters})")


class NeuronFactory:
    """Factory for creating different types of neurons."""
    
    _neuron_classes = {}
    
    @classmethod
    def register_neuron_class(cls, neuron_type: NeuronType, neuron_class: type) -> None:
        """Register a neuron class for factory creation."""
        cls._neuron_classes[neuron_type] = neuron_class
    
    @classmethod
    def create_neuron(cls, neuron_type: NeuronType, neuron_id: str, 
                     parameters: NeuronParameters) -> BaseNeuron:
        """Create a neuron of the specified type."""
        if neuron_type not in cls._neuron_classes:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        
        neuron_class = cls._neuron_classes[neuron_type]
        return neuron_class(neuron_id, parameters)
    
    @classmethod
    def get_available_types(cls) -> List[NeuronType]:
        """Get list of available neuron types."""
        return list(cls._neuron_classes.keys())
