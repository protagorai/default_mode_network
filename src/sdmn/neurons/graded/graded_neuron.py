"""
Base graded potential neuron model.

Implements continuous, analog voltage dynamics without action potentials.
Based on conductance-based models with voltage-gated ion channels.

Reference:
    Lockery & Goodman (2009), "The quest for action potentials in C. elegans 
    neurons hits a plateau." Nature Neuroscience, 12(4), 377-378.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from sdmn.neurons.base_neuron import BaseNeuron, NeuronParameters, NeuronType


@dataclass
class GradedNeuronParameters(NeuronParameters):
    """
    Parameters for graded potential neuron models.
    
    All conductances in nanosiemens (nS), voltages in millivolts (mV),
    times in milliseconds (ms), capacitance in picofarads (pF).
    """
    
    # Override base neuron type
    neuron_type: NeuronType = NeuronType.CUSTOM
    
    # Membrane properties
    C_m: float = 3.0                    # Membrane capacitance (pF)
    g_leak: float = 0.3                 # Leak conductance (nS)
    E_leak: float = -65.0               # Leak reversal potential (mV)
    
    # Numerical integration
    dt: float = 0.01                    # Time step (ms) - 10 μs default
    integration_method: str = "RK4"     # "RK4" or "Euler"
    
    # Voltage clipping (for stability)
    V_min: float = -100.0               # Minimum voltage (mV)
    V_max: float = 100.0                # Maximum voltage (mV)
    
    # Noise parameters (optional)
    voltage_noise_std: float = 0.0      # Voltage noise std (mV/√ms)
    current_noise_std: float = 0.0      # Background current noise std (pA)
    
    # Custom parameters dictionary
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.C_m <= 0:
            raise ValueError("Membrane capacitance must be positive")
        if self.g_leak < 0:
            raise ValueError("Leak conductance must be non-negative")
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if self.integration_method not in ["RK4", "Euler"]:
            raise ValueError("Integration method must be 'RK4' or 'Euler'")


class GradedNeuron(BaseNeuron):
    """
    Base class for graded potential neurons.
    
    Implements continuous voltage dynamics without spiking threshold.
    Subclasses should override _compute_ionic_currents() to define
    specific ion channel dynamics.
    
    Mathematical Model:
        C_m * dV/dt = -I_leak + I_ionic + I_ext + I_syn + I_gap
        
    where:
        I_leak = g_leak * (V - E_leak)
        I_ionic = ion channel currents (defined by subclass)
        I_ext = external injected current
        I_syn = synaptic current
        I_gap = gap junction current
    """
    
    def __init__(self, neuron_id: str, parameters: GradedNeuronParameters):
        """
        Initialize graded potential neuron.
        
        Args:
            neuron_id: Unique identifier for this neuron
            parameters: Neuron parameters
        """
        # Initialize base neuron
        super().__init__(neuron_id, parameters)
        
        # Cast parameters to correct type
        self.params: GradedNeuronParameters = parameters
        
        # Initialize voltage to leak reversal (resting state)
        self.state.membrane_potential = self.params.E_leak
        
        # Current accumulators
        self.I_syn_total = 0.0      # Total synaptic current (pA)
        self.I_gap_total = 0.0      # Total gap junction current (pA)
        self.I_ext_current = 0.0    # External current (pA)
        
        # For RK4 integration
        self.rng = np.random.default_rng()
    
    def _compute_leak_current(self, V: float) -> float:
        """
        Compute leak current.
        
        Args:
            V: Membrane voltage (mV)
            
        Returns:
            Leak current (pA)
        """
        # I_leak = g_leak * (V - E_leak)
        # Convert nS * mV = pA
        return self.params.g_leak * (V - self.params.E_leak)
    
    def _compute_ionic_currents(self, V: float) -> float:
        """
        Compute voltage-gated ionic currents.
        
        This method should be overridden by subclasses to implement
        specific ion channel dynamics (e.g., Ca2+, K+, etc.).
        
        Args:
            V: Membrane voltage (mV)
            
        Returns:
            Total ionic current (pA)
        """
        # Base implementation: no voltage-gated channels
        return 0.0
    
    def _compute_dVdt(self, V: float) -> float:
        """
        Compute time derivative of membrane voltage.
        
        Args:
            V: Current membrane voltage (mV)
            
        Returns:
            dV/dt in mV/ms
        """
        # Compute all currents
        I_leak = self._compute_leak_current(V)
        I_ionic = self._compute_ionic_currents(V)
        I_total = I_ionic + self.I_ext_current + self.I_syn_total + self.I_gap_total
        
        # Add noise if specified
        if self.params.current_noise_std > 0:
            noise = self.rng.normal(0, self.params.current_noise_std)
            I_total += noise
        
        # C_m * dV/dt = -I_leak + I_total
        # dV/dt = (-I_leak + I_total) / C_m
        # Convert: pA / pF = mV/ms
        dVdt = (-I_leak + I_total) / self.params.C_m
        
        return dVdt
    
    def _integrate_voltage_rk4(self, dt: float) -> float:
        """
        Integrate voltage using 4th-order Runge-Kutta method.
        
        Args:
            dt: Time step (ms)
            
        Returns:
            New voltage (mV)
        """
        V = self.state.membrane_potential
        
        # RK4 integration
        k1 = self._compute_dVdt(V)
        k2 = self._compute_dVdt(V + 0.5 * dt * k1)
        k3 = self._compute_dVdt(V + 0.5 * dt * k2)
        k4 = self._compute_dVdt(V + dt * k3)
        
        V_new = V + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return V_new
    
    def _integrate_voltage_euler(self, dt: float) -> float:
        """
        Integrate voltage using forward Euler method.
        
        Args:
            dt: Time step (ms)
            
        Returns:
            New voltage (mV)
        """
        V = self.state.membrane_potential
        dVdt = self._compute_dVdt(V)
        V_new = V + dt * dVdt
        
        return V_new
    
    def _update_gating_variables(self, dt: float) -> None:
        """
        Update ion channel gating variables.
        
        This method should be overridden by subclasses that have
        gating variables (e.g., m, h, n for Hodgkin-Huxley-type models).
        
        Args:
            dt: Time step (ms)
        """
        pass
    
    def integrate_membrane_equation(self, dt: float, total_input: float) -> float:
        """
        Integrate the membrane equation for one time step.
        
        Note: For graded neurons, we override the update() method instead
        and this is here for BaseNeuron interface compatibility.
        
        Args:
            dt: Time step size (ms)
            total_input: Total input current (pA)
            
        Returns:
            New membrane potential (mV)
        """
        # Set external current
        self.I_ext_current = total_input
        
        # Integrate based on method
        if self.params.integration_method == "RK4":
            V_new = self._integrate_voltage_rk4(dt)
        else:  # Euler
            V_new = self._integrate_voltage_euler(dt)
        
        # Clip voltage for stability
        V_new = np.clip(V_new, self.params.V_min, self.params.V_max)
        
        # Add voltage noise if specified
        if self.params.voltage_noise_std > 0:
            noise = self.rng.normal(0, self.params.voltage_noise_std * np.sqrt(dt))
            V_new += noise
            V_new = np.clip(V_new, self.params.V_min, self.params.V_max)
        
        return V_new
    
    def update(self, dt: Optional[float] = None, inputs: Optional[list] = None) -> None:
        """
        Update neuron state for one time step.
        
        Args:
            dt: Time step size (ms). If None, uses self.params.dt
            inputs: Optional list of synaptic inputs (for compatibility)
        """
        if dt is None:
            dt = self.params.dt
        
        # Accumulate inputs
        if inputs:
            for inp in inputs:
                self.add_synaptic_input(inp)
        
        # Get total external input
        total_external = self.get_total_input()
        
        # Update gating variables first (if any)
        self._update_gating_variables(dt)
        
        # Integrate voltage
        V_new = self.integrate_membrane_equation(dt, total_external)
        self.state.membrane_potential = V_new
        
        # Update time
        self.current_time += dt
        
        # Clear inputs for next step
        self.clear_inputs()
        
        # Reset current accumulators
        self.I_syn_total = 0.0
        self.I_gap_total = 0.0
    
    def add_synaptic_current(self, current: float) -> None:
        """
        Add synaptic current for this time step.
        
        Args:
            current: Synaptic current (pA)
        """
        self.I_syn_total += current
    
    def add_gap_junction_current(self, current: float) -> None:
        """
        Add gap junction current for this time step.
        
        Args:
            current: Gap junction current (pA)
        """
        self.I_gap_total += current
    
    def set_external_current(self, current: float) -> None:
        """
        Set external injected current.
        
        Args:
            current: External current (pA)
        """
        self.I_ext_current = current
    
    def get_voltage(self) -> float:
        """
        Get current membrane voltage.
        
        Returns:
            Voltage (mV)
        """
        return self.state.membrane_potential
    
    @property
    def voltage(self) -> float:
        """Membrane voltage property."""
        return self.state.membrane_potential
    
    def has_spiked(self) -> bool:
        """
        Graded neurons don't spike.
        
        Returns:
            Always False
        """
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get neuron state for serialization."""
        state = super().get_state()
        state.update({
            'I_syn_total': self.I_syn_total,
            'I_gap_total': self.I_gap_total,
            'I_ext_current': self.I_ext_current,
        })
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set neuron state from serialization."""
        super().set_state(state)
        self.I_syn_total = state.get('I_syn_total', 0.0)
        self.I_gap_total = state.get('I_gap_total', 0.0)
        self.I_ext_current = state.get('I_ext_current', 0.0)

