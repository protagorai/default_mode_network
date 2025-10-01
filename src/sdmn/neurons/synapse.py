"""
Synaptic connection models for neural networks.

This module implements various types of synaptic connections between
neurons, including different dynamics, plasticity mechanisms, and
neurotransmitter types.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from sdmn.core.state_manager import StateSerializable


class SynapseType(Enum):
    """Types of synaptic connections."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory" 
    ELECTRICAL = "electrical"
    MODULATORY = "modulatory"


class NeurotransmitterType(Enum):
    """Types of neurotransmitters."""
    GLUTAMATE = "glutamate"        # Primary excitatory
    GABA = "gaba"                  # Primary inhibitory
    DOPAMINE = "dopamine"          # Modulatory
    SEROTONIN = "serotonin"        # Modulatory
    ACETYLCHOLINE = "acetylcholine" # Modulatory
    CUSTOM = "custom"


@dataclass
class SynapticParameters:
    """Parameters for synaptic connections."""
    synapse_type: SynapseType
    neurotransmitter: NeurotransmitterType = NeurotransmitterType.GLUTAMATE
    
    # Basic synaptic parameters
    weight: float = 1.0              # Synaptic strength
    delay: float = 1.0               # Synaptic delay (ms)
    tau_rise: float = 0.5            # Rise time constant (ms)
    tau_decay: float = 5.0           # Decay time constant (ms)
    
    # Plasticity parameters
    enable_plasticity: bool = False
    learning_rate: float = 0.01
    tau_pre: float = 20.0            # Pre-synaptic trace time constant (ms)
    tau_post: float = 20.0           # Post-synaptic trace time constant (ms)
    
    # Receptor parameters
    reversal_potential: float = 0.0   # Reversal potential (mV)
    max_conductance: float = 1.0      # Maximum conductance (nS)
    
    # Additional parameters
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}
        
        # Set default reversal potentials based on synapse type
        if self.reversal_potential == 0.0:  # Default not set
            if self.synapse_type == SynapseType.EXCITATORY:
                self.reversal_potential = 0.0    # Excitatory
            elif self.synapse_type == SynapseType.INHIBITORY:
                self.reversal_potential = -80.0  # Inhibitory
            else:
                self.reversal_potential = 0.0


class BaseSynapse(StateSerializable, ABC):
    """Abstract base class for synaptic connections."""
    
    def __init__(self, synapse_id: str, presynaptic_neuron_id: str, 
                 postsynaptic_neuron_id: str, parameters: SynapticParameters):
        """
        Initialize synapse.
        
        Args:
            synapse_id: Unique identifier for synapse
            presynaptic_neuron_id: ID of presynaptic neuron
            postsynaptic_neuron_id: ID of postsynaptic neuron
            parameters: Synaptic parameters
        """
        self.synapse_id = synapse_id
        self.presynaptic_neuron_id = presynaptic_neuron_id
        self.postsynaptic_neuron_id = postsynaptic_neuron_id
        self.parameters = parameters
        
        # State variables
        self.current_time = 0.0
        self.conductance = 0.0
        self.current = 0.0
        
        # Spike timing
        self.last_presynaptic_spike = None
        self.last_postsynaptic_spike = None
        
        # Plasticity traces
        self.presynaptic_trace = 0.0
        self.postsynaptic_trace = 0.0
        
        # History for analysis
        self.spike_times: List[float] = []
        self.weight_history: List[tuple] = []  # (time, weight)
    
    @abstractmethod
    def update(self, dt: float) -> None:
        """Update synapse state for one time step."""
        pass
    
    @abstractmethod
    def receive_spike(self, spike_time: float) -> None:
        """Process incoming spike from presynaptic neuron."""
        pass
    
    def get_current(self) -> float:
        """Get synaptic current."""
        return self.current
    
    def get_conductance(self) -> float:
        """Get synaptic conductance."""
        return self.conductance
    
    def get_weight(self) -> float:
        """Get current synaptic weight."""
        return self.parameters.weight
    
    def set_weight(self, weight: float) -> None:
        """Set synaptic weight."""
        old_weight = self.parameters.weight
        self.parameters.weight = weight
        self.weight_history.append((self.current_time, weight))
    
    def is_excitatory(self) -> bool:
        """Check if synapse is excitatory."""
        return self.parameters.synapse_type == SynapseType.EXCITATORY
    
    def is_inhibitory(self) -> bool:
        """Check if synapse is inhibitory."""
        return self.parameters.synapse_type == SynapseType.INHIBITORY


class Synapse(BaseSynapse):
    """
    Standard synaptic connection with exponential dynamics.
    
    Implements double-exponential synaptic current model and
    optional spike-timing dependent plasticity (STDP).
    """
    
    def __init__(self, synapse_id: str, presynaptic_neuron_id: str,
                 postsynaptic_neuron_id: str, parameters: SynapticParameters):
        super().__init__(synapse_id, presynaptic_neuron_id, postsynaptic_neuron_id, parameters)
        
        # Double exponential model state
        self.g_rise = 0.0    # Rising conductance component
        self.g_decay = 0.0   # Decaying conductance component
        
        # Normalization factor for double exponential
        self.norm_factor = self._calculate_normalization_factor()
    
    def _calculate_normalization_factor(self) -> float:
        """Calculate normalization factor for double exponential."""
        tau_rise = self.parameters.tau_rise
        tau_decay = self.parameters.tau_decay
        
        if tau_rise == tau_decay:
            return 1.0
        
        # Peak time
        t_peak = (tau_rise * tau_decay) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        
        # Peak value
        peak_val = (np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise))
        
        return 1.0 / peak_val if peak_val > 0 else 1.0
    
    def update(self, dt: float) -> None:
        """Update synapse state for one time step."""
        self.current_time += dt
        
        # Update double exponential conductance
        self.g_rise *= np.exp(-dt / self.parameters.tau_rise)
        self.g_decay *= np.exp(-dt / self.parameters.tau_decay)
        
        # Total conductance
        self.conductance = (self.g_decay - self.g_rise) * self.norm_factor
        self.conductance = max(0.0, self.conductance)  # Ensure non-negative
        
        # Update plasticity traces
        if self.parameters.enable_plasticity:
            self.presynaptic_trace *= np.exp(-dt / self.parameters.tau_pre)
            self.postsynaptic_trace *= np.exp(-dt / self.parameters.tau_post)
    
    def receive_spike(self, spike_time: float) -> None:
        """Process incoming spike from presynaptic neuron."""
        self.last_presynaptic_spike = spike_time
        self.spike_times.append(spike_time)
        
        # Add to conductance
        increment = self.parameters.weight * self.parameters.max_conductance
        self.g_rise += increment
        self.g_decay += increment
        
        # Update plasticity trace
        if self.parameters.enable_plasticity:
            self.presynaptic_trace += 1.0
            self._apply_plasticity_rule()
    
    def receive_postsynaptic_spike(self, spike_time: float) -> None:
        """Process spike from postsynaptic neuron for plasticity."""
        if not self.parameters.enable_plasticity:
            return
        
        self.last_postsynaptic_spike = spike_time
        self.postsynaptic_trace += 1.0
        self._apply_plasticity_rule()
    
    def _apply_plasticity_rule(self) -> None:
        """Apply spike-timing dependent plasticity (STDP)."""
        if (self.last_presynaptic_spike is None or 
            self.last_postsynaptic_spike is None):
            return
        
        # Time difference (post - pre)
        dt_spike = self.last_postsynaptic_spike - self.last_presynaptic_spike
        
        # STDP rule
        if dt_spike > 0:  # Post after pre - potentiation
            weight_change = self.parameters.learning_rate * self.presynaptic_trace
        else:  # Pre after post - depression
            weight_change = -self.parameters.learning_rate * self.postsynaptic_trace
        
        # Update weight with bounds
        new_weight = self.parameters.weight + weight_change
        new_weight = max(0.0, min(2.0, new_weight))  # Clamp between 0 and 2
        
        self.set_weight(new_weight)
    
    def calculate_current(self, postsynaptic_voltage: float) -> float:
        """
        Calculate synaptic current based on postsynaptic voltage.
        
        Args:
            postsynaptic_voltage: Membrane potential of postsynaptic neuron
            
        Returns:
            Synaptic current
        """
        # I_syn = g_syn * (V_post - E_rev)
        self.current = (self.conductance * 
                       (postsynaptic_voltage - self.parameters.reversal_potential))
        
        # Apply sign based on synapse type
        if self.parameters.synapse_type == SynapseType.INHIBITORY:
            self.current = -abs(self.current)
        else:
            self.current = abs(self.current)
        
        return self.current
    
    def get_efficacy(self) -> float:
        """Get current synaptic efficacy (weight * max_conductance)."""
        return self.parameters.weight * self.parameters.max_conductance
    
    def reset_synapse(self) -> None:
        """Reset synapse to initial state."""
        self.conductance = 0.0
        self.current = 0.0
        self.g_rise = 0.0
        self.g_decay = 0.0
        self.presynaptic_trace = 0.0
        self.postsynaptic_trace = 0.0
        self.last_presynaptic_spike = None
        self.last_postsynaptic_spike = None
        self.spike_times.clear()
        self.weight_history.clear()
        self.current_time = 0.0
    
    # StateSerializable interface
    def get_state(self) -> Dict[str, Any]:
        """Get synapse state for serialization."""
        return {
            'synapse_id': self.synapse_id,
            'presynaptic_neuron_id': self.presynaptic_neuron_id,
            'postsynaptic_neuron_id': self.postsynaptic_neuron_id,
            'current_time': self.current_time,
            'conductance': self.conductance,
            'current': self.current,
            'g_rise': self.g_rise,
            'g_decay': self.g_decay,
            'presynaptic_trace': self.presynaptic_trace,
            'postsynaptic_trace': self.postsynaptic_trace,
            'last_presynaptic_spike': self.last_presynaptic_spike,
            'last_postsynaptic_spike': self.last_postsynaptic_spike,
            'spike_times': self.spike_times.copy(),
            'parameters': {
                'synapse_type': self.parameters.synapse_type.value,
                'neurotransmitter': self.parameters.neurotransmitter.value,
                'weight': self.parameters.weight,
                'delay': self.parameters.delay,
                'tau_rise': self.parameters.tau_rise,
                'tau_decay': self.parameters.tau_decay,
                'enable_plasticity': self.parameters.enable_plasticity,
                'learning_rate': self.parameters.learning_rate,
                'tau_pre': self.parameters.tau_pre,
                'tau_post': self.parameters.tau_post,
                'reversal_potential': self.parameters.reversal_potential,
                'max_conductance': self.parameters.max_conductance,
                'custom_params': self.parameters.custom_params.copy()
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set synapse state from serialization."""
        self.synapse_id = state['synapse_id']
        self.presynaptic_neuron_id = state['presynaptic_neuron_id']
        self.postsynaptic_neuron_id = state['postsynaptic_neuron_id']
        self.current_time = state['current_time']
        self.conductance = state['conductance']
        self.current = state['current']
        self.g_rise = state['g_rise']
        self.g_decay = state['g_decay']
        self.presynaptic_trace = state['presynaptic_trace']
        self.postsynaptic_trace = state['postsynaptic_trace']
        self.last_presynaptic_spike = state['last_presynaptic_spike']
        self.last_postsynaptic_spike = state['last_postsynaptic_spike']
        self.spike_times = state['spike_times'].copy()
        
        # Update parameters
        if 'parameters' in state:
            params = state['parameters']
            self.parameters.synapse_type = SynapseType(params['synapse_type'])
            self.parameters.neurotransmitter = NeurotransmitterType(params['neurotransmitter'])
            self.parameters.weight = params['weight']
            self.parameters.delay = params['delay']
            self.parameters.tau_rise = params['tau_rise']
            self.parameters.tau_decay = params['tau_decay']
            self.parameters.enable_plasticity = params['enable_plasticity']
            self.parameters.learning_rate = params['learning_rate']
            self.parameters.tau_pre = params['tau_pre']
            self.parameters.tau_post = params['tau_post']
            self.parameters.reversal_potential = params['reversal_potential']
            self.parameters.max_conductance = params['max_conductance']
            self.parameters.custom_params = params['custom_params'].copy()
    
    def get_state_version(self) -> str:
        """Get state format version."""
        return "1.0"


class SynapseFactory:
    """Factory for creating different types of synapses."""
    
    @staticmethod
    def create_excitatory_synapse(synapse_id: str, pre_id: str, post_id: str,
                                 weight: float = 1.0, delay: float = 1.0) -> Synapse:
        """Create standard excitatory synapse."""
        params = SynapticParameters(
            synapse_type=SynapseType.EXCITATORY,
            neurotransmitter=NeurotransmitterType.GLUTAMATE,
            weight=weight,
            delay=delay,
            reversal_potential=0.0
        )
        return Synapse(synapse_id, pre_id, post_id, params)
    
    @staticmethod
    def create_inhibitory_synapse(synapse_id: str, pre_id: str, post_id: str,
                                weight: float = 1.0, delay: float = 1.0) -> Synapse:
        """Create standard inhibitory synapse."""
        params = SynapticParameters(
            synapse_type=SynapseType.INHIBITORY,
            neurotransmitter=NeurotransmitterType.GABA,
            weight=weight,
            delay=delay,
            reversal_potential=-80.0
        )
        return Synapse(synapse_id, pre_id, post_id, params)
    
    @staticmethod
    def create_plastic_synapse(synapse_id: str, pre_id: str, post_id: str,
                             synapse_type: SynapseType = SynapseType.EXCITATORY,
                             weight: float = 1.0, learning_rate: float = 0.01) -> Synapse:
        """Create synapse with plasticity enabled."""
        params = SynapticParameters(
            synapse_type=synapse_type,
            weight=weight,
            enable_plasticity=True,
            learning_rate=learning_rate
        )
        return Synapse(synapse_id, pre_id, post_id, params)
