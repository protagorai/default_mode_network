"""
Leaky Integrate-and-Fire neuron model implementation.

This module provides a biologically-inspired LIF neuron that implements
the classic leaky integrator model with exponential decay towards
resting potential.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from sdmn.neurons.base_neuron import BaseNeuron, NeuronParameters, NeuronType


@dataclass
class LIFParameters(NeuronParameters):
    """Parameters specific to Leaky Integrate-and-Fire neurons."""
    
    def __init__(self, 
                 tau_m: float = 20.0,         # Membrane time constant (ms)
                 r_mem: float = 10.0,         # Membrane resistance (MΩ)
                 c_mem: float = 2.0,          # Membrane capacitance (nF)
                 i_offset: float = 0.0,       # Offset current (nA)
                 v_rest: float = -70.0,       # Resting potential (mV)
                 v_thresh: float = -50.0,     # Spike threshold (mV)  
                 v_reset: float = -80.0,      # Reset potential (mV)
                 refractory_period: float = 2.0,  # Refractory period (ms)
                 dt: float = 0.1):            # Integration time step (ms)
        
        super().__init__(
            neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE,
            dt=dt,
            v_rest=v_rest,
            v_thresh=v_thresh,
            v_reset=v_reset,
            refractory_period=refractory_period
        )
        
        # LIF-specific parameters
        self.tau_m = tau_m
        self.r_mem = r_mem  
        self.c_mem = c_mem
        self.i_offset = i_offset
        
        # Store in custom params for serialization
        self.custom_params.update({
            'tau_m': tau_m,
            'r_mem': r_mem,
            'c_mem': c_mem,
            'i_offset': i_offset
        })


class LIFNeuron(BaseNeuron):
    """
    Leaky Integrate-and-Fire neuron implementation.
    
    The LIF model is described by the differential equation:
    τ_m * dV/dt = (V_rest - V) + R_m * I
    
    Where:
    - τ_m is the membrane time constant
    - V is the membrane potential
    - V_rest is the resting potential
    - R_m is the membrane resistance
    - I is the input current
    """
    
    def __init__(self, neuron_id: str, parameters: Optional[LIFParameters] = None):
        """
        Initialize LIF neuron.
        
        Args:
            neuron_id: Unique identifier for this neuron
            parameters: LIF-specific parameters (uses defaults if None)
        """
        if parameters is None:
            parameters = LIFParameters()
        
        if not isinstance(parameters, LIFParameters):
            # Convert generic parameters to LIF parameters
            lif_params = LIFParameters()
            lif_params.v_rest = parameters.v_rest
            lif_params.v_thresh = parameters.v_thresh
            lif_params.v_reset = parameters.v_reset
            lif_params.refractory_period = parameters.refractory_period
            lif_params.dt = parameters.dt
            parameters = lif_params
        
        super().__init__(neuron_id, parameters)
        
        # Cache frequently used parameters
        self.tau_m = parameters.tau_m
        self.r_mem = parameters.r_mem
        self.c_mem = parameters.c_mem
        self.i_offset = parameters.i_offset
        
        # Pre-calculate exponential decay factor
        self.alpha = np.exp(-parameters.dt / self.tau_m)
        
        # Integration method
        self.integration_method = 'exponential_euler'  # or 'euler'
    
    def update(self, dt: Optional[float] = None, inputs: Optional[List[float]] = None) -> None:
        """
        Update neuron state for one time step.
        
        Args:
            dt: Time step size (uses parameter default if None)
            inputs: Optional list of synaptic inputs
        """
        if dt is None:
            dt = self.parameters.dt
        
        self.current_time += dt
        
        # Add any provided inputs
        if inputs:
            for inp in inputs:
                self.add_synaptic_input(inp)
        
        # Skip update if in refractory period
        if self.state.is_refractory(self.current_time):
            self.clear_inputs()
            return
        
        # Calculate total input current
        total_input = self.get_total_input() + self.i_offset
        
        # Integrate membrane equation
        new_voltage = self.integrate_membrane_equation(dt, total_input)
        self.state.membrane_potential = new_voltage
        
        # Check for spike
        if self._check_spike_condition():
            self._generate_spike()
        
        # Record voltage if enabled
        self.record_voltage()
        
        # Clear inputs for next time step
        self.clear_inputs()
    
    def integrate_membrane_equation(self, dt: float, total_input: float) -> float:
        """
        Integrate the LIF membrane equation.
        
        Args:
            dt: Time step size (ms)
            total_input: Total input current (nA)
            
        Returns:
            New membrane potential (mV)
        """
        v_current = self.state.membrane_potential
        
        if self.integration_method == 'exponential_euler':
            # Exact solution for exponential Euler method
            # V(t+dt) = V_rest + (V(t) - V_rest + R*I) * exp(-dt/τ) - R*I
            v_inf = self.parameters.v_rest + self.r_mem * total_input
            v_new = v_inf + (v_current - v_inf) * self.alpha
            
        else:  # Standard Euler method
            # dV/dt = (V_rest - V + R*I) / τ
            dv_dt = (self.parameters.v_rest - v_current + self.r_mem * total_input) / self.tau_m
            v_new = v_current + dv_dt * dt
        
        return v_new
    
    def set_integration_method(self, method: str) -> None:
        """
        Set numerical integration method.
        
        Args:
            method: 'exponential_euler' or 'euler'
        """
        if method not in ['exponential_euler', 'euler']:
            raise ValueError("Integration method must be 'exponential_euler' or 'euler'")
        
        self.integration_method = method
        
        # Update alpha for exponential method
        if method == 'exponential_euler':
            self.alpha = np.exp(-self.parameters.dt / self.tau_m)
    
    def get_membrane_time_constant(self) -> float:
        """Get membrane time constant."""
        return self.tau_m
    
    def get_membrane_resistance(self) -> float:
        """Get membrane resistance.""" 
        return self.r_mem
    
    def get_membrane_capacitance(self) -> float:
        """Get membrane capacitance."""
        return self.c_mem
    
    def set_offset_current(self, current: float) -> None:
        """Set constant offset current."""
        self.i_offset = current
    
    def get_offset_current(self) -> float:
        """Get constant offset current."""
        return self.i_offset
    
    def get_input_resistance(self) -> float:
        """Calculate input resistance at rest."""
        return self.r_mem
    
    def get_rheobase_current(self) -> float:
        """
        Calculate rheobase (minimum current for spiking).
        
        Returns:
            Rheobase current in nA
        """
        return (self.parameters.v_thresh - self.parameters.v_rest) / self.r_mem
    
    def calculate_f_i_curve(self, currents: List[float], duration: float = 1000.0) -> List[float]:
        """
        Calculate frequency-current (f-I) relationship.
        
        Args:
            currents: List of input currents to test (nA)
            duration: Simulation duration for each current (ms)
            
        Returns:
            List of firing rates (Hz) corresponding to each current
        """
        firing_rates = []
        
        # Store original state
        original_state = self.get_state()
        
        for current in currents:
            # Reset neuron
            self.reset_neuron()
            
            # Simulate with constant current
            time = 0.0
            spike_count = 0
            
            while time < duration:
                self.set_external_input(current)
                self.update()
                
                if self.has_spiked():
                    spike_count += 1
                
                time += self.parameters.dt
            
            # Calculate firing rate
            firing_rate = spike_count * 1000.0 / duration  # Convert to Hz
            firing_rates.append(firing_rate)
        
        # Restore original state
        self.set_state(original_state)
        
        return firing_rates
    
    def get_analytical_firing_rate(self, input_current: float) -> float:
        """
        Calculate analytical firing rate for constant input.
        
        Args:
            input_current: Constant input current (nA)
            
        Returns:
            Theoretical firing rate (Hz)
        """
        # Steady-state voltage
        v_ss = self.parameters.v_rest + self.r_mem * input_current
        
        # If below threshold, no firing
        if v_ss <= self.parameters.v_thresh:
            return 0.0
        
        # Time to reach threshold from reset
        # V(t) = V_rest + (V_reset - V_rest + R*I) * (1 - exp(-t/τ))
        # Solve for t when V(t) = V_thresh
        
        numerator = self.parameters.v_thresh - self.parameters.v_rest
        denominator = self.parameters.v_reset - self.parameters.v_rest + self.r_mem * input_current
        
        if denominator <= 0:
            return 0.0
        
        ratio = numerator / denominator
        if ratio >= 1:
            return 0.0
        
        time_to_threshold = -self.tau_m * np.log(1 - ratio)
        
        # Total period includes refractory time
        period = time_to_threshold + self.parameters.refractory_period
        
        return 1000.0 / period  # Convert to Hz
    
    def __str__(self) -> str:
        """String representation."""
        return (f"LIFNeuron(id={self.neuron_id}, "
                f"V_mem={self.state.membrane_potential:.1f}mV, "
                f"τ_m={self.tau_m}ms, "
                f"spikes={len(self.spike_times)})")


# Register LIF neuron with factory
from sdmn.neurons.base_neuron import NeuronFactory
NeuronFactory.register_neuron_class(NeuronType.LEAKY_INTEGRATE_FIRE, LIFNeuron)
