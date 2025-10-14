"""
Gap junction (electrical synapse) for C. elegans networks.

Implements bidirectional electrical coupling between neurons via gap junctions.
Current flow is ohmic and proportional to voltage difference.

References:
    White et al. (1986), Phil. Trans. R. Soc. B, 314(1165), 1-340.
    Varshney et al. (2011), PLOS Comput. Biol., 7(2), e1001066.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from sdmn.neurons.synapse import BaseSynapse, SynapseType, SynapticParameters


@dataclass
class GapJunctionParameters:
    """
    Parameters for gap junctions (electrical synapses).
    
    Conductance in nanosiemens (nS).
    """
    
    # Coupling strength
    conductance: float = 0.5            # Gap junction conductance (nS)
    
    # Optional heterogeneity
    conductance_min: float = 0.1        # Minimum conductance (nS)
    conductance_max: float = 2.0        # Maximum conductance (nS)
    
    def __post_init__(self):
        """Validate parameters."""
        if self.conductance < 0:
            raise ValueError("Gap junction conductance must be non-negative")
        if self.conductance < self.conductance_min or self.conductance > self.conductance_max:
            raise ValueError(f"Conductance {self.conductance} outside valid range "
                           f"[{self.conductance_min}, {self.conductance_max}]")


class GapJunction(BaseSynapse):
    """
    Gap junction (electrical synapse) with bidirectional coupling.
    
    Provides direct electrical coupling between neurons via ohmic conductance.
    Unlike chemical synapses, gap junctions:
    - Are bidirectional (symmetric)
    - Have no delay
    - Have linear (ohmic) current-voltage relationship
    - Allow rapid synchronization
    
    Mathematical Model:
        I_gap,i = g_gap * (V_j - V_i)
        I_gap,j = g_gap * (V_i - V_j) = -I_gap,i
        
    where g_gap is the coupling conductance.
    """
    
    def __init__(self, synapse_id: str, neuron_a, neuron_b,
                 parameters: GapJunctionParameters):
        """
        Initialize gap junction between two neurons.
        
        Args:
            synapse_id: Unique identifier
            neuron_a: First neuron
            neuron_b: Second neuron
            parameters: Gap junction parameters
        """
        # Create compatible BaseSynapse parameters
        base_params = SynapticParameters(
            synapse_type=SynapseType.ELECTRICAL,
            weight=parameters.conductance,
            delay=0.0  # No delay for gap junctions
        )
        
        super().__init__(
            synapse_id,
            neuron_a.neuron_id if hasattr(neuron_a, 'neuron_id') else str(neuron_a),
            neuron_b.neuron_id if hasattr(neuron_b, 'neuron_id') else str(neuron_b),
            base_params
        )
        
        # Store neuron references (bidirectional)
        self.neuron_a = neuron_a
        self.neuron_b = neuron_b
        
        # Store gap junction parameters
        self.gap_params = parameters
        
        # Current state
        self.I_gap_a = 0.0  # Current into neuron_a (pA)
        self.I_gap_b = 0.0  # Current into neuron_b (pA)
        self.V_diff = 0.0   # Voltage difference (mV)
        
        # Recording
        self.current_history = []
        self.voltage_diff_history = []
    
    def update(self, dt: float) -> None:
        """
        Update gap junction state for one time step.
        
        Args:
            dt: Time step (ms)
        """
        self.current_time += dt
        
        # Get voltages from both neurons
        if hasattr(self.neuron_a, 'get_voltage'):
            V_a = self.neuron_a.get_voltage()
        elif hasattr(self.neuron_a, 'voltage'):
            V_a = self.neuron_a.voltage
        else:
            V_a = -65.0  # Default resting potential
        
        if hasattr(self.neuron_b, 'get_voltage'):
            V_b = self.neuron_b.get_voltage()
        elif hasattr(self.neuron_b, 'voltage'):
            V_b = self.neuron_b.voltage
        else:
            V_b = -65.0  # Default resting potential
        
        # Compute voltage difference
        self.V_diff = V_b - V_a
        
        # Compute ohmic currents
        # I_gap = g_gap * (V_other - V_self)
        # Sign convention: positive current = depolarizing (inward)
        
        # Current into neuron_a from neuron_b
        # If V_b > V_a, positive current flows into a (depolarizing)
        self.I_gap_a = self.gap_params.conductance * (V_b - V_a)
        
        # Current into neuron_b from neuron_a
        # If V_a > V_b, positive current flows into b (depolarizing)
        self.I_gap_b = self.gap_params.conductance * (V_a - V_b)
        
        # Note: I_gap_a = -I_gap_b (conservation of current)
        
        # Deliver currents to neurons
        if hasattr(self.neuron_a, 'add_gap_junction_current'):
            self.neuron_a.add_gap_junction_current(self.I_gap_a)
        
        if hasattr(self.neuron_b, 'add_gap_junction_current'):
            self.neuron_b.add_gap_junction_current(self.I_gap_b)
    
    def get_conductance(self) -> float:
        """
        Get gap junction conductance (nS).
        
        Returns:
            Coupling conductance
        """
        return self.gap_params.conductance
    
    def set_conductance(self, conductance: float) -> None:
        """
        Set gap junction conductance.
        
        Args:
            conductance: New conductance value (nS)
        """
        # Validate
        if conductance < self.gap_params.conductance_min:
            conductance = self.gap_params.conductance_min
        if conductance > self.gap_params.conductance_max:
            conductance = self.gap_params.conductance_max
        
        self.gap_params.conductance = conductance
    
    def get_currents(self) -> tuple[float, float]:
        """
        Get currents flowing into both neurons.
        
        Returns:
            Tuple of (I_gap_a, I_gap_b) in pA
        """
        return (self.I_gap_a, self.I_gap_b)
    
    def get_voltage_difference(self) -> float:
        """
        Get voltage difference between neurons.
        
        Returns:
            V_b - V_a in mV
        """
        return self.V_diff
    
    def get_coupling_coefficient(self) -> float:
        """
        Get coupling coefficient (dimensionless).
        
        Coupling coefficient = g_gap / g_leak
        Estimates how strong the electrical coupling is relative to leak.
        
        Returns:
            Coupling coefficient (if leak conductance available)
        """
        # Try to get leak conductance from neurons
        g_leak = 0.3  # Default value (nS)
        
        if hasattr(self.neuron_a, 'params') and hasattr(self.neuron_a.params, 'g_leak'):
            g_leak = self.neuron_a.params.g_leak
        elif hasattr(self.neuron_a, 'celegans_params'):
            g_leak = self.neuron_a.celegans_params.g_leak
        
        return self.gap_params.conductance / g_leak if g_leak > 0 else 0.0
    
    def is_bidirectional(self) -> bool:
        """
        Check if gap junction is bidirectional.
        
        Returns:
            Always True for gap junctions
        """
        return True
    
    def receive_spike(self, spike_time: float) -> None:
        """
        Not applicable for gap junctions.
        
        Gap junctions respond continuously to voltage, not discrete spikes.
        """
        pass
    
    def reset(self) -> None:
        """Reset gap junction to initial state."""
        self.I_gap_a = 0.0
        self.I_gap_b = 0.0
        self.V_diff = 0.0
        self.current_history.clear()
        self.voltage_diff_history.clear()
        self.current_time = 0.0
    
    def record_state(self) -> None:
        """Record current state for analysis."""
        self.current_history.append((self.current_time, self.I_gap_a, self.I_gap_b))
        self.voltage_diff_history.append((self.current_time, self.V_diff))
    
    def get_state(self) -> Dict[str, Any]:
        """Get gap junction state for serialization."""
        return {
            'synapse_id': self.synapse_id,
            'neuron_a_id': self.presynaptic_neuron_id,
            'neuron_b_id': self.postsynaptic_neuron_id,
            'current_time': self.current_time,
            'I_gap_a': self.I_gap_a,
            'I_gap_b': self.I_gap_b,
            'V_diff': self.V_diff,
            'parameters': {
                'conductance': self.gap_params.conductance,
                'conductance_min': self.gap_params.conductance_min,
                'conductance_max': self.gap_params.conductance_max,
            }
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set gap junction state from serialization."""
        self.synapse_id = state['synapse_id']
        self.presynaptic_neuron_id = state['neuron_a_id']
        self.postsynaptic_neuron_id = state['neuron_b_id']
        self.current_time = state['current_time']
        self.I_gap_a = state['I_gap_a']
        self.I_gap_b = state['I_gap_b']
        self.V_diff = state['V_diff']
    
    def get_state_version(self) -> str:
        """Get state format version."""
        return "1.0"
    
    def __str__(self) -> str:
        """String representation."""
        return (f"GapJunction(id={self.synapse_id}, "
                f"{self.presynaptic_neuron_id}<->{self.postsynaptic_neuron_id}, "
                f"g={self.gap_params.conductance:.2f}nS, "
                f"I_a={self.I_gap_a:.2f}pA, I_b={self.I_gap_b:.2f}pA)")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"GapJunction(synapse_id='{self.synapse_id}', "
                f"neuron_a_id='{self.presynaptic_neuron_id}', "
                f"neuron_b_id='{self.postsynaptic_neuron_id}', "
                f"conductance={self.gap_params.conductance}, "
                f"V_diff={self.V_diff:.2f})")

