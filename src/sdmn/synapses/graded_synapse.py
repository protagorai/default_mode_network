"""
Graded chemical synapse for C. elegans graded potential neurons.

Implements voltage-dependent neurotransmitter release without spike threshold.
Release is proportional to presynaptic voltage via sigmoid function.

References:
    Goodman et al. (1998), Neuron, 20(4), 763-772.
    Liu et al. (2018), Cell, 175(1), 57-70.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

from sdmn.neurons.synapse import BaseSynapse, SynapseType


@dataclass
class GradedSynapseParameters:
    """
    Parameters for graded chemical synapses.
    
    All conductances in nanosiemens (nS), voltages in millivolts (mV),
    times in milliseconds (ms).
    """
    
    synapse_type: SynapseType = SynapseType.EXCITATORY
    
    # Synaptic strength
    weight: float = 1.0                 # Synaptic weight/strength (nS)
    
    # Graded release parameters
    V_thresh: float = -40.0             # Threshold voltage for release (mV)
    k_release: float = 5.0              # Sigmoid slope for release (mV)
    
    # Synaptic kinetics
    tau_rise: float = 1.0               # Rise time constant (ms)
    tau_decay: float = 5.0              # Decay time constant (ms)
    
    # Reversal potential
    E_syn: float = 0.0                  # Reversal potential (mV)
    
    # Synaptic delay
    delay: float = 0.5                  # Synaptic delay (ms)
    
    # Noise (optional)
    release_noise_std: float = 0.0      # Std dev of release variability
    
    def __post_init__(self):
        """Set default reversal potentials based on synapse type."""
        if self.E_syn == 0.0 and self.synapse_type == SynapseType.INHIBITORY:
            self.E_syn = -75.0
        
        # Validate parameters
        if self.weight < 0:
            raise ValueError("Synaptic weight must be non-negative")
        if self.tau_rise <= 0 or self.tau_decay <= 0:
            raise ValueError("Time constants must be positive")
        if self.delay < 0:
            raise ValueError("Delay must be non-negative")


class GradedChemicalSynapse(BaseSynapse):
    """
    Graded chemical synapse with voltage-dependent release.
    
    Unlike spike-triggered synapses, release is continuous and proportional
    to presynaptic membrane voltage via a sigmoid function.
    
    Mathematical Model:
        release(V_pre) = 1 / (1 + exp(-(V_pre - V_thresh) / k))
        
        Synaptic conductance follows alpha function or dual exponential:
        dg_syn/dt = release(V_pre) * weight * (1/tau_rise) - g_syn/tau_decay
        
        Synaptic current:
        I_syn = g_syn * (V_post - E_syn)
    """
    
    def __init__(self, synapse_id: str, presynaptic_neuron, postsynaptic_neuron,
                 parameters: GradedSynapseParameters):
        """
        Initialize graded chemical synapse.
        
        Args:
            synapse_id: Unique identifier
            presynaptic_neuron: Presynaptic graded neuron
            postsynaptic_neuron: Postsynaptic graded neuron
            parameters: Synapse parameters
        """
        # Create compatible BaseSynapse parameters
        from sdmn.neurons.synapse import SynapticParameters
        base_params = SynapticParameters(
            synapse_type=parameters.synapse_type,
            weight=parameters.weight,
            delay=parameters.delay,
            tau_rise=parameters.tau_rise,
            tau_decay=parameters.tau_decay,
            reversal_potential=parameters.E_syn
        )
        
        super().__init__(
            synapse_id,
            presynaptic_neuron.neuron_id if hasattr(presynaptic_neuron, 'neuron_id') else str(presynaptic_neuron),
            postsynaptic_neuron.neuron_id if hasattr(postsynaptic_neuron, 'neuron_id') else str(postsynaptic_neuron),
            base_params
        )
        
        # Store neuron references
        self.pre_neuron = presynaptic_neuron
        self.post_neuron = postsynaptic_neuron
        
        # Store graded-specific parameters
        self.graded_params = parameters
        
        # Synaptic state
        self.g_rise = 0.0                # Rising component
        self.g_decay = 0.0               # Decaying component
        self.g_syn = 0.0                 # Total conductance (nS)
        self.I_syn = 0.0                 # Synaptic current (pA)
        self.release_prob = 0.0          # Current release probability
        
        # Delay buffer (stores release values)
        if self.graded_params.delay > 0:
            # Calculate buffer size based on timestep (assume 0.01 ms default)
            buffer_size = max(1, int(self.graded_params.delay / 0.01))
            self.delay_buffer = deque([0.0] * buffer_size, maxlen=buffer_size)
        else:
            self.delay_buffer = None
        
        # Random number generator for noise
        self.rng = np.random.default_rng()
        
        # Recording
        self.conductance_history = []
        self.current_history = []
    
    def _compute_release(self, V_pre: float) -> float:
        """
        Compute graded neurotransmitter release as function of presynaptic voltage.
        
        Args:
            V_pre: Presynaptic voltage (mV)
            
        Returns:
            Release probability in [0, 1]
        """
        # Sigmoid release function
        release = 1.0 / (1.0 + np.exp(
            -(V_pre - self.graded_params.V_thresh) / self.graded_params.k_release
        ))
        
        # Add noise if specified
        if self.graded_params.release_noise_std > 0:
            noise = self.rng.normal(0, self.graded_params.release_noise_std)
            release += noise
            release = np.clip(release, 0.0, 1.0)
        
        return release
    
    def update(self, dt: float) -> None:
        """
        Update synapse state for one time step.
        
        Args:
            dt: Time step (ms)
        """
        self.current_time += dt
        
        # Get presynaptic voltage
        if hasattr(self.pre_neuron, 'get_voltage'):
            V_pre = self.pre_neuron.get_voltage()
        elif hasattr(self.pre_neuron, 'voltage'):
            V_pre = self.pre_neuron.voltage
        else:
            V_pre = self.graded_params.V_thresh  # Default if can't access
        
        # Compute release probability
        release_current = self._compute_release(V_pre)
        
        # Handle synaptic delay
        if self.delay_buffer is not None:
            self.delay_buffer.append(release_current)
            release_delayed = self.delay_buffer[0]
        else:
            release_delayed = release_current
        
        self.release_prob = release_delayed
        
        # Update conductance using dual exponential
        # Decay existing conductance
        self.g_rise *= np.exp(-dt / self.graded_params.tau_rise)
        self.g_decay *= np.exp(-dt / self.graded_params.tau_decay)
        
        # Add new release
        increment = release_delayed * self.graded_params.weight
        self.g_rise += increment
        self.g_decay += increment
        
        # Compute normalized conductance
        # Normalization factor ensures peak = weight
        norm_factor = 1.0
        if self.graded_params.tau_rise != self.graded_params.tau_decay:
            t_peak = (self.graded_params.tau_rise * self.graded_params.tau_decay) / \
                     (self.graded_params.tau_decay - self.graded_params.tau_rise) * \
                     np.log(self.graded_params.tau_decay / self.graded_params.tau_rise)
            peak_val = np.exp(-t_peak / self.graded_params.tau_decay) - \
                      np.exp(-t_peak / self.graded_params.tau_rise)
            norm_factor = 1.0 / peak_val if peak_val > 0 else 1.0
        
        self.g_syn = (self.g_decay - self.g_rise) * norm_factor
        self.g_syn = max(0.0, self.g_syn)
        
        # Compute synaptic current
        if hasattr(self.post_neuron, 'get_voltage'):
            V_post = self.post_neuron.get_voltage()
        elif hasattr(self.post_neuron, 'voltage'):
            V_post = self.post_neuron.voltage
        else:
            V_post = self.graded_params.E_syn  # Default
        
        # I_syn = g_syn * (V_post - E_syn)
        # For excitatory: negative current = depolarizing (inward)
        # For inhibitory: positive current = hyperpolarizing (outward)
        self.I_syn = self.g_syn * (V_post - self.graded_params.E_syn)
        
        # Apply sign convention
        if self.graded_params.synapse_type == SynapseType.EXCITATORY:
            self.I_syn = -abs(self.I_syn)  # Inward (depolarizing)
        else:  # Inhibitory
            self.I_syn = abs(self.I_syn)   # Outward (hyperpolarizing)
        
        # Deliver current to postsynaptic neuron
        if hasattr(self.post_neuron, 'add_synaptic_current'):
            self.post_neuron.add_synaptic_current(self.I_syn)
    
    def get_conductance(self) -> float:
        """Get current synaptic conductance (nS)."""
        return self.g_syn
    
    def get_current(self) -> float:
        """Get current synaptic current (pA)."""
        return self.I_syn
    
    def get_release_probability(self) -> float:
        """Get current release probability."""
        return self.release_prob
    
    def receive_spike(self, spike_time: float) -> None:
        """
        Not applicable for graded synapses.
        
        Graded synapses respond continuously to voltage, not discrete spikes.
        """
        pass
    
    def reset(self) -> None:
        """Reset synapse to initial state."""
        self.g_rise = 0.0
        self.g_decay = 0.0
        self.g_syn = 0.0
        self.I_syn = 0.0
        self.release_prob = 0.0
        
        if self.delay_buffer is not None:
            self.delay_buffer.clear()
            buffer_size = self.delay_buffer.maxlen
            self.delay_buffer.extend([0.0] * buffer_size)
        
        self.conductance_history.clear()
        self.current_history.clear()
        self.current_time = 0.0
    
    def record_state(self) -> None:
        """Record current state for analysis."""
        self.conductance_history.append((self.current_time, self.g_syn))
        self.current_history.append((self.current_time, self.I_syn))
    
    def get_state(self) -> Dict[str, Any]:
        """Get synapse state for serialization."""
        return {
            'synapse_id': self.synapse_id,
            'presynaptic_neuron_id': self.presynaptic_neuron_id,
            'postsynaptic_neuron_id': self.postsynaptic_neuron_id,
            'current_time': self.current_time,
            'g_rise': self.g_rise,
            'g_decay': self.g_decay,
            'g_syn': self.g_syn,
            'I_syn': self.I_syn,
            'release_prob': self.release_prob,
            'parameters': {
                'synapse_type': self.graded_params.synapse_type.value,
                'weight': self.graded_params.weight,
                'V_thresh': self.graded_params.V_thresh,
                'k_release': self.graded_params.k_release,
                'tau_rise': self.graded_params.tau_rise,
                'tau_decay': self.graded_params.tau_decay,
                'E_syn': self.graded_params.E_syn,
                'delay': self.graded_params.delay,
                'release_noise_std': self.graded_params.release_noise_std,
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set synapse state from serialization."""
        self.synapse_id = state['synapse_id']
        self.presynaptic_neuron_id = state['presynaptic_neuron_id']
        self.postsynaptic_neuron_id = state['postsynaptic_neuron_id']
        self.current_time = state['current_time']
        self.g_rise = state['g_rise']
        self.g_decay = state['g_decay']
        self.g_syn = state['g_syn']
        self.I_syn = state['I_syn']
        self.release_prob = state['release_prob']
    
    def get_state_version(self) -> str:
        """Get state format version."""
        return "1.0"

