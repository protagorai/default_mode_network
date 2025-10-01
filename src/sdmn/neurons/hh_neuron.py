"""
Hodgkin-Huxley neuron model implementation.

This module provides a detailed biophysical neuron model based on the
classic Hodgkin-Huxley equations with voltage-gated sodium and 
potassium channels.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from sdmn.neurons.base_neuron import BaseNeuron, NeuronParameters, NeuronType


@dataclass
class HHParameters(NeuronParameters):
    """Parameters specific to Hodgkin-Huxley neurons."""
    
    def __init__(self,
                 # Membrane parameters
                 c_mem: float = 1.0,          # Membrane capacitance (μF/cm²)
                 
                 # Sodium channel parameters  
                 g_na_max: float = 120.0,     # Max sodium conductance (mS/cm²)
                 e_na: float = 50.0,          # Sodium reversal potential (mV)
                 
                 # Potassium channel parameters
                 g_k_max: float = 36.0,       # Max potassium conductance (mS/cm²) 
                 e_k: float = -77.0,          # Potassium reversal potential (mV)
                 
                 # Leak channel parameters
                 g_leak: float = 0.3,         # Leak conductance (mS/cm²)
                 e_leak: float = -54.4,       # Leak reversal potential (mV)
                 
                 # Integration parameters
                 dt: float = 0.01,            # Small time step for HH (ms)
                 temperature: float = 6.3):   # Temperature (°C)
        
        super().__init__(
            neuron_type=NeuronType.HODGKIN_HUXLEY,
            dt=dt,
            v_rest=e_leak,  # Resting potential approximately equals leak potential
            v_thresh=-30.0,  # Approximate spike threshold
            v_reset=e_leak,
            refractory_period=1.0
        )
        
        # HH-specific parameters
        self.c_mem = c_mem
        self.g_na_max = g_na_max
        self.e_na = e_na
        self.g_k_max = g_k_max
        self.e_k = e_k
        self.g_leak = g_leak
        self.e_leak = e_leak
        self.temperature = temperature
        
        # Temperature correction factor (Q10 = 3)
        self.temp_factor = 3.0 ** ((temperature - 6.3) / 10.0)
        
        # Store in custom params for serialization
        self.custom_params.update({
            'c_mem': c_mem,
            'g_na_max': g_na_max,
            'e_na': e_na,
            'g_k_max': g_k_max,
            'e_k': e_k,
            'g_leak': g_leak,
            'e_leak': e_leak,
            'temperature': temperature,
            'temp_factor': self.temp_factor
        })


class HHNeuron(BaseNeuron):
    """
    Hodgkin-Huxley neuron implementation.
    
    The HH model is described by:
    C_m * dV/dt = -g_Na*m³*h*(V-E_Na) - g_K*n⁴*(V-E_K) - g_leak*(V-E_leak) + I
    
    With gating variables:
    dm/dt = α_m(V)*(1-m) - β_m(V)*m
    dh/dt = α_h(V)*(1-h) - β_h(V)*h  
    dn/dt = α_n(V)*(1-n) - β_n(V)*n
    """
    
    def __init__(self, neuron_id: str, parameters: Optional[HHParameters] = None):
        """
        Initialize HH neuron.
        
        Args:
            neuron_id: Unique identifier for this neuron
            parameters: HH-specific parameters (uses defaults if None)
        """
        if parameters is None:
            parameters = HHParameters()
        
        if not isinstance(parameters, HHParameters):
            # Convert generic parameters to HH parameters
            hh_params = HHParameters()
            hh_params.dt = parameters.dt
            parameters = hh_params
        
        super().__init__(neuron_id, parameters)
        
        # Cache frequently used parameters
        self.c_mem = parameters.c_mem
        self.g_na_max = parameters.g_na_max
        self.e_na = parameters.e_na
        self.g_k_max = parameters.g_k_max
        self.e_k = parameters.e_k
        self.g_leak = parameters.g_leak
        self.e_leak = parameters.e_leak
        self.temp_factor = parameters.temp_factor
        
        # Initialize gating variables at resting potential
        v_init = self.e_leak
        self.m = self._m_inf(v_init)  # Sodium activation
        self.h = self._h_inf(v_init)  # Sodium inactivation  
        self.n = self._n_inf(v_init)  # Potassium activation
        
        # Store gating variables in state
        self.state.set_state_var('m', self.m)
        self.state.set_state_var('h', self.h)
        self.state.set_state_var('n', self.n)
        
        # Integration method
        self.integration_method = 'runge_kutta_4'  # or 'euler'
        
        # For spike detection
        self.v_previous = v_init
        self.spike_detected_this_step = False
    
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
        
        # Calculate total input current
        total_input = self.get_total_input()
        
        # Integrate HH equations
        if self.integration_method == 'runge_kutta_4':
            self._rk4_step(dt, total_input)
        else:
            self._euler_step(dt, total_input)
        
        # Detect spike (crossing of threshold with positive slope)
        self._detect_spike()
        
        # Record voltage
        self.record_voltage()
        
        # Clear inputs for next time step
        self.clear_inputs()
    
    def integrate_membrane_equation(self, dt: float, total_input: float) -> float:
        """
        Integrate membrane equation (called by base class).
        
        Note: For HH model, the full update is done in update() method.
        This method is provided for interface compatibility.
        """
        # This is a simplified version - full dynamics are in _rk4_step
        v = self.state.membrane_potential
        
        # Calculate ionic currents
        i_na = self._calculate_sodium_current(v)
        i_k = self._calculate_potassium_current(v)
        i_leak = self._calculate_leak_current(v)
        
        # Total ionic current
        i_ionic = i_na + i_k + i_leak
        
        # Membrane equation
        dv_dt = (-i_ionic + total_input) / self.c_mem
        
        return v + dv_dt * dt
    
    def _rk4_step(self, dt: float, input_current: float) -> None:
        """Integrate using 4th-order Runge-Kutta method."""
        # Current state
        v = self.state.membrane_potential
        m = self.state.get_state_var('m')
        h = self.state.get_state_var('h') 
        n = self.state.get_state_var('n')
        
        # k1
        k1_v = self._dv_dt(v, m, h, n, input_current)
        k1_m = self._dm_dt(v, m)
        k1_h = self._dh_dt(v, h)
        k1_n = self._dn_dt(v, n)
        
        # k2
        v2 = v + 0.5 * dt * k1_v
        m2 = m + 0.5 * dt * k1_m
        h2 = h + 0.5 * dt * k1_h
        n2 = n + 0.5 * dt * k1_n
        
        k2_v = self._dv_dt(v2, m2, h2, n2, input_current)
        k2_m = self._dm_dt(v2, m2)
        k2_h = self._dh_dt(v2, h2)
        k2_n = self._dn_dt(v2, n2)
        
        # k3
        v3 = v + 0.5 * dt * k2_v
        m3 = m + 0.5 * dt * k2_m
        h3 = h + 0.5 * dt * k2_h
        n3 = n + 0.5 * dt * k2_n
        
        k3_v = self._dv_dt(v3, m3, h3, n3, input_current)
        k3_m = self._dm_dt(v3, m3)
        k3_h = self._dh_dt(v3, h3)
        k3_n = self._dn_dt(v3, n3)
        
        # k4
        v4 = v + dt * k3_v
        m4 = m + dt * k3_m
        h4 = h + dt * k3_h
        n4 = n + dt * k3_n
        
        k4_v = self._dv_dt(v4, m4, h4, n4, input_current)
        k4_m = self._dm_dt(v4, m4)
        k4_h = self._dh_dt(v4, h4)
        k4_n = self._dn_dt(v4, n4)
        
        # Update state
        self.state.membrane_potential = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.m = m + (dt / 6.0) * (k1_m + 2*k2_m + 2*k3_m + k4_m)
        self.h = h + (dt / 6.0) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
        self.n = n + (dt / 6.0) * (k1_n + 2*k2_n + 2*k3_n + k4_n)
        
        # Clamp gating variables to [0,1]
        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0) 
        self.n = np.clip(self.n, 0.0, 1.0)
        
        # Update state variables
        self.state.set_state_var('m', self.m)
        self.state.set_state_var('h', self.h)
        self.state.set_state_var('n', self.n)
    
    def _euler_step(self, dt: float, input_current: float) -> None:
        """Integrate using forward Euler method."""
        v = self.state.membrane_potential
        m = self.state.get_state_var('m')
        h = self.state.get_state_var('h')
        n = self.state.get_state_var('n')
        
        # Update state variables
        self.state.membrane_potential = v + dt * self._dv_dt(v, m, h, n, input_current)
        self.m = m + dt * self._dm_dt(v, m)
        self.h = h + dt * self._dh_dt(v, h)
        self.n = n + dt * self._dn_dt(v, n)
        
        # Clamp gating variables
        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)
        
        # Update state variables
        self.state.set_state_var('m', self.m)
        self.state.set_state_var('h', self.h)
        self.state.set_state_var('n', self.n)
    
    def _dv_dt(self, v: float, m: float, h: float, n: float, i_ext: float) -> float:
        """Calculate dV/dt."""
        i_na = self.g_na_max * (m**3) * h * (v - self.e_na)
        i_k = self.g_k_max * (n**4) * (v - self.e_k)
        i_leak = self.g_leak * (v - self.e_leak)
        
        return (-i_na - i_k - i_leak + i_ext) / self.c_mem
    
    def _dm_dt(self, v: float, m: float) -> float:
        """Calculate dm/dt."""
        alpha_m = self._alpha_m(v)
        beta_m = self._beta_m(v)
        return self.temp_factor * (alpha_m * (1 - m) - beta_m * m)
    
    def _dh_dt(self, v: float, h: float) -> float:
        """Calculate dh/dt."""
        alpha_h = self._alpha_h(v)
        beta_h = self._beta_h(v)
        return self.temp_factor * (alpha_h * (1 - h) - beta_h * h)
    
    def _dn_dt(self, v: float, n: float) -> float:
        """Calculate dn/dt."""
        alpha_n = self._alpha_n(v)
        beta_n = self._beta_n(v)
        return self.temp_factor * (alpha_n * (1 - n) - beta_n * n)
    
    # Rate functions for sodium activation (m)
    def _alpha_m(self, v: float) -> float:
        """Sodium activation rate α_m."""
        if abs(v + 40.0) < 1e-6:
            return 1.0  # L'Hopital's rule limit
        return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
    
    def _beta_m(self, v: float) -> float:
        """Sodium activation rate β_m."""
        return 4.0 * np.exp(-(v + 65.0) / 18.0)
    
    # Rate functions for sodium inactivation (h)
    def _alpha_h(self, v: float) -> float:
        """Sodium inactivation rate α_h."""
        return 0.07 * np.exp(-(v + 65.0) / 20.0)
    
    def _beta_h(self, v: float) -> float:
        """Sodium inactivation rate β_h."""
        return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    
    # Rate functions for potassium activation (n)
    def _alpha_n(self, v: float) -> float:
        """Potassium activation rate α_n."""
        if abs(v + 55.0) < 1e-6:
            return 0.1  # L'Hopital's rule limit
        return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
    
    def _beta_n(self, v: float) -> float:
        """Potassium activation rate β_n."""
        return 0.125 * np.exp(-(v + 65.0) / 80.0)
    
    # Steady-state values
    def _m_inf(self, v: float) -> float:
        """Steady-state sodium activation."""
        alpha = self._alpha_m(v)
        beta = self._beta_m(v)
        return alpha / (alpha + beta)
    
    def _h_inf(self, v: float) -> float:
        """Steady-state sodium inactivation."""
        alpha = self._alpha_h(v)
        beta = self._beta_h(v)
        return alpha / (alpha + beta)
    
    def _n_inf(self, v: float) -> float:
        """Steady-state potassium activation."""
        alpha = self._alpha_n(v)
        beta = self._beta_n(v)
        return alpha / (alpha + beta)
    
    def _calculate_sodium_current(self, v: float) -> float:
        """Calculate sodium current."""
        return self.g_na_max * (self.m**3) * self.h * (v - self.e_na)
    
    def _calculate_potassium_current(self, v: float) -> float:
        """Calculate potassium current.""" 
        return self.g_k_max * (self.n**4) * (v - self.e_k)
    
    def _calculate_leak_current(self, v: float) -> float:
        """Calculate leak current."""
        return self.g_leak * (v - self.e_leak)
    
    def _detect_spike(self) -> None:
        """Detect action potential by threshold crossing with positive slope."""
        v_current = self.state.membrane_potential
        
        # Check for upward threshold crossing
        if (self.v_previous < self.parameters.v_thresh and 
            v_current >= self.parameters.v_thresh and
            not self.spike_detected_this_step):
            
            # Generate spike
            self.state.spike_time = self.current_time
            self.spike_times.append(self.current_time)
            self.spike_detected_this_step = True
            
            # Propagate spike
            self._propagate_spike()
        else:
            self.spike_detected_this_step = False
        
        self.v_previous = v_current
    
    def get_gating_variables(self) -> Tuple[float, float, float]:
        """Get current gating variable values."""
        return self.m, self.h, self.n
    
    def get_ionic_currents(self) -> Dict[str, float]:
        """Get current ionic current values."""
        v = self.state.membrane_potential
        return {
            'i_na': self._calculate_sodium_current(v),
            'i_k': self._calculate_potassium_current(v),
            'i_leak': self._calculate_leak_current(v)
        }
    
    def set_integration_method(self, method: str) -> None:
        """Set numerical integration method."""
        if method not in ['runge_kutta_4', 'euler']:
            raise ValueError("Integration method must be 'runge_kutta_4' or 'euler'")
        self.integration_method = method
    
    def reset_neuron(self) -> None:
        """Reset neuron to resting state."""
        super().reset_neuron()
        
        # Reset gating variables to steady state
        v_rest = self.e_leak
        self.m = self._m_inf(v_rest)
        self.h = self._h_inf(v_rest)
        self.n = self._n_inf(v_rest)
        
        self.state.set_state_var('m', self.m)
        self.state.set_state_var('h', self.h) 
        self.state.set_state_var('n', self.n)
        
        self.v_previous = v_rest
        self.spike_detected_this_step = False
    
    def __str__(self) -> str:
        """String representation."""
        return (f"HHNeuron(id={self.neuron_id}, "
                f"V_mem={self.state.membrane_potential:.1f}mV, "
                f"m={self.m:.3f}, h={self.h:.3f}, n={self.n:.3f}, "
                f"spikes={len(self.spike_times)})")


# Register HH neuron with factory
from sdmn.neurons.base_neuron import NeuronFactory
NeuronFactory.register_neuron_class(NeuronType.HODGKIN_HUXLEY, HHNeuron)
