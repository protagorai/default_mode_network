"""
C. elegans specific graded potential neuron model.

Implements biologically-accurate ion channel dynamics based on empirical
data from C. elegans neurons, including Ca2+, K+, and Ca-dependent K+ channels.

References:
    Goodman et al. (1998), "Active currents regulate sensitivity and dynamic
    range in C. elegans neurons." Neuron, 20(4), 763-772.
    
    Liu et al. (2018), "C. elegans AWA olfactory neurons fire calcium-mediated 
    all-or-none action potentials." Cell, 175(1), 57-70.
    
    Lockery & Goodman (2009), Nature Neuroscience, 12(4), 377-378.
"""

from typing import Dict, Any
from dataclasses import dataclass
import numpy as np

from .graded_neuron import GradedNeuron, GradedNeuronParameters


# Faraday constant for calcium dynamics
FARADAY_CONSTANT = 96485.0  # C/mol


@dataclass
class CElegansParameters(GradedNeuronParameters):
    """
    Parameters for C. elegans graded potential neurons.
    
    All values based on published electrophysiological recordings.
    Default parameters represent a typical interneuron.
    """
    
    # Calcium channel parameters
    g_Ca: float = 0.8                   # Max Ca conductance (nS)
    E_Ca: float = 50.0                  # Ca reversal potential (mV)
    V_half_Ca: float = -20.0            # Half-activation voltage (mV)
    k_Ca: float = 5.0                   # Slope factor (mV)
    tau_Ca_min: float = 0.5             # Min time constant (ms)
    tau_Ca_max: float = 5.0             # Max time constant (ms)
    V_half_tau_Ca: float = -30.0        # Voltage of half-max tau (mV)
    k_tau_Ca: float = 10.0              # Slope for tau voltage-dependence (mV)
    Ca_power: int = 2                   # Power for Ca activation (m^2)
    
    # Potassium channel parameters  
    g_K: float = 1.5                    # Max K conductance (nS)
    E_K: float = -80.0                  # K reversal potential (mV)
    V_half_K: float = -25.0             # Half-activation voltage (mV)
    k_K: float = 10.0                   # Slope factor (mV)
    tau_K_min: float = 1.0              # Min time constant (ms)
    tau_K_max: float = 10.0             # Max time constant (ms)
    V_half_tau_K: float = -30.0         # Voltage of half-max tau (mV)
    k_tau_K: float = 10.0               # Slope for tau voltage-dependence (mV)
    K_power: int = 4                    # Power for K activation (m^4)
    
    # Calcium-dependent potassium channel parameters
    g_KCa: float = 0.5                  # Max Ca-dependent K conductance (nS)
    Ca_half: float = 100.0              # Half-activation [Ca] (nM)
    tau_KCa: float = 50.0               # Time constant (ms)
    
    # Intracellular calcium dynamics
    Ca_rest: float = 50.0               # Resting [Ca] (nM)
    tau_Ca_removal: float = 100.0       # Ca removal time constant (ms)
    f_Ca: float = 0.01                  # Fraction of free calcium
    cell_volume: float = 1.0            # Cell volume (pL)
    
    def __post_init__(self):
        """Validate parameters."""
        super().__post_init__()
        
        # Validate conductances
        if self.g_Ca < 0 or self.g_K < 0 or self.g_KCa < 0:
            raise ValueError("Conductances must be non-negative")
        
        # Validate time constants
        if self.tau_Ca_min <= 0 or self.tau_K_min <= 0 or self.tau_KCa <= 0:
            raise ValueError("Time constants must be positive")
        if self.tau_Ca_min > self.tau_Ca_max:
            raise ValueError("tau_Ca_min must be <= tau_Ca_max")
        if self.tau_K_min > self.tau_K_max:
            raise ValueError("tau_K_min must be <= tau_K_max")


class CElegansNeuron(GradedNeuron):
    """
    C. elegans graded potential neuron with realistic ion channel dynamics.
    
    Mathematical Model:
        C_m * dV/dt = -I_leak - I_Ca - I_K - I_KCa + I_ext + I_syn + I_gap
        
    where:
        I_leak = g_leak * (V - E_leak)
        I_Ca = g_Ca * m_Ca^2 * (V - E_Ca)
        I_K = g_K * m_K^4 * (V - E_K)
        I_KCa = g_KCa * m_KCa * (V - E_K)
        
    Gating variables follow first-order kinetics:
        dm/dt = (m_inf(V) - m) / tau_m(V)
        
    Intracellular calcium:
        d[Ca]/dt = -f * (I_Ca / (2*F*vol)) - ([Ca] - [Ca]_rest) / tau_removal
    """
    
    def __init__(self, neuron_id: str, parameters: CElegansParameters):
        """
        Initialize C. elegans neuron.
        
        Args:
            neuron_id: Unique identifier
            parameters: C. elegans-specific parameters
        """
        super().__init__(neuron_id, parameters)
        
        # Cast to correct parameter type
        self.celegans_params: CElegansParameters = parameters
        
        # Initialize gating variables at resting potential
        V_init = self.state.membrane_potential
        self.m_Ca = self._compute_m_inf(
            V_init, self.celegans_params.V_half_Ca, self.celegans_params.k_Ca
        )
        self.m_K = self._compute_m_inf(
            V_init, self.celegans_params.V_half_K, self.celegans_params.k_K
        )
        self.m_KCa = 0.0  # Initialize at 0 (no Ca yet)
        
        # Initialize calcium concentration
        self.Ca_internal = self.celegans_params.Ca_rest  # nM
        
    def _sigmoid(self, V: float, V_half: float, k: float) -> float:
        """
        Boltzmann sigmoid activation function.
        
        Args:
            V: Membrane voltage (mV)
            V_half: Half-activation voltage (mV)
            k: Slope factor (mV)
            
        Returns:
            Activation value in [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-(V - V_half) / k))
    
    def _compute_m_inf(self, V: float, V_half: float, k: float) -> float:
        """
        Compute steady-state activation.
        
        Args:
            V: Membrane voltage (mV)
            V_half: Half-activation voltage (mV)
            k: Slope factor (mV)
            
        Returns:
            Steady-state activation
        """
        return self._sigmoid(V, V_half, k)
    
    def _compute_tau_m(self, V: float, tau_min: float, tau_max: float,
                       V_half_tau: float, k_tau: float) -> float:
        """
        Compute voltage-dependent time constant.
        
        Args:
            V: Membrane voltage (mV)
            tau_min: Minimum time constant (ms)
            tau_max: Maximum time constant (ms)
            V_half_tau: Voltage of half-maximal tau (mV)
            k_tau: Slope factor for tau (mV)
            
        Returns:
            Time constant (ms)
        """
        # tau(V) = tau_min + (tau_max - tau_min) / (1 + exp((V - V_half) / k))
        tau = tau_min + (tau_max - tau_min) / (1.0 + np.exp((V - V_half_tau) / k_tau))
        return max(tau, 0.001)  # Prevent division by zero
    
    def _compute_ionic_currents(self, V: float) -> float:
        """
        Compute all voltage-gated ionic currents.
        
        Args:
            V: Membrane voltage (mV)
            
        Returns:
            Total ionic current (pA)
        """
        # Calcium current (depolarizing)
        I_Ca = (self.celegans_params.g_Ca * 
                (self.m_Ca ** self.celegans_params.Ca_power) * 
                (V - self.celegans_params.E_Ca))
        
        # Potassium current (repolarizing)
        I_K = (self.celegans_params.g_K * 
               (self.m_K ** self.celegans_params.K_power) * 
               (V - self.celegans_params.E_K))
        
        # Calcium-dependent potassium current (adaptation)
        I_KCa = (self.celegans_params.g_KCa * 
                 self.m_KCa * 
                 (V - self.celegans_params.E_K))
        
        # Total (Ca is depolarizing, K and KCa are hyperpolarizing)
        # Sign convention: positive current = depolarizing
        I_total = -I_Ca - I_K - I_KCa
        
        return I_total
    
    def _update_gating_variables(self, dt: float) -> None:
        """
        Update all gating variables using first-order kinetics.
        
        Args:
            dt: Time step (ms)
        """
        V = self.state.membrane_potential
        
        # Update calcium channel activation
        m_Ca_inf = self._compute_m_inf(
            V, self.celegans_params.V_half_Ca, self.celegans_params.k_Ca
        )
        tau_Ca = self._compute_tau_m(
            V, self.celegans_params.tau_Ca_min, self.celegans_params.tau_Ca_max,
            self.celegans_params.V_half_tau_Ca, self.celegans_params.k_tau_Ca
        )
        self.m_Ca += dt * (m_Ca_inf - self.m_Ca) / tau_Ca
        self.m_Ca = np.clip(self.m_Ca, 0.0, 1.0)
        
        # Update potassium channel activation
        m_K_inf = self._compute_m_inf(
            V, self.celegans_params.V_half_K, self.celegans_params.k_K
        )
        tau_K = self._compute_tau_m(
            V, self.celegans_params.tau_K_min, self.celegans_params.tau_K_max,
            self.celegans_params.V_half_tau_K, self.celegans_params.k_tau_K
        )
        self.m_K += dt * (m_K_inf - self.m_K) / tau_K
        self.m_K = np.clip(self.m_K, 0.0, 1.0)
        
        # Update Ca-dependent K channel activation
        # Depends on intracellular calcium
        m_KCa_inf = self.Ca_internal / (self.Ca_internal + self.celegans_params.Ca_half)
        tau_KCa = self.celegans_params.tau_KCa
        self.m_KCa += dt * (m_KCa_inf - self.m_KCa) / tau_KCa
        self.m_KCa = np.clip(self.m_KCa, 0.0, 1.0)
        
        # Update intracellular calcium
        # Calcium influx from Ca current
        I_Ca = (self.celegans_params.g_Ca * 
                (self.m_Ca ** self.celegans_params.Ca_power) * 
                (V - self.celegans_params.E_Ca))
        
        # Convert current to concentration change
        # d[Ca]/dt = -f * (I_Ca / (2*F*vol)) - ([Ca] - [Ca]_rest) / tau_removal
        # I_Ca in pA, F in C/mol, vol in pL
        # pA / (C/mol * pL) = pA / (pC/nmol) = nmol/s / pL = nM/ms
        
        Ca_influx = -self.celegans_params.f_Ca * I_Ca / (
            2.0 * FARADAY_CONSTANT * self.celegans_params.cell_volume
        )
        Ca_removal = (self.Ca_internal - self.celegans_params.Ca_rest) / self.celegans_params.tau_Ca_removal
        
        self.Ca_internal += dt * (Ca_influx - Ca_removal)
        self.Ca_internal = np.clip(self.Ca_internal, 0.0, 10000.0)  # 0-10 μM range
    
    def get_channel_states(self) -> Dict[str, float]:
        """
        Get current state of all ion channels.
        
        Returns:
            Dictionary with channel activations and calcium concentration
        """
        return {
            'm_Ca': self.m_Ca,
            'm_K': self.m_K,
            'm_KCa': self.m_KCa,
            'Ca_internal': self.Ca_internal,
        }
    
    def get_currents(self) -> Dict[str, float]:
        """
        Get individual ionic currents.
        
        Returns:
            Dictionary with all currents (pA)
        """
        V = self.state.membrane_potential
        
        I_leak = self._compute_leak_current(V)
        
        I_Ca = (self.celegans_params.g_Ca * 
                (self.m_Ca ** self.celegans_params.Ca_power) * 
                (V - self.celegans_params.E_Ca))
        
        I_K = (self.celegans_params.g_K * 
               (self.m_K ** self.celegans_params.K_power) * 
               (V - self.celegans_params.E_K))
        
        I_KCa = (self.celegans_params.g_KCa * 
                 self.m_KCa * 
                 (V - self.celegans_params.E_K))
        
        return {
            'I_leak': I_leak,
            'I_Ca': I_Ca,
            'I_K': I_K,
            'I_KCa': I_KCa,
            'I_syn': self.I_syn_total,
            'I_gap': self.I_gap_total,
            'I_ext': self.I_ext_current,
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete neuron state for serialization."""
        state = super().get_state()
        state.update({
            'm_Ca': self.m_Ca,
            'm_K': self.m_K,
            'm_KCa': self.m_KCa,
            'Ca_internal': self.Ca_internal,
        })
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set complete neuron state from serialization."""
        super().set_state(state)
        self.m_Ca = state.get('m_Ca', 0.0)
        self.m_K = state.get('m_K', 0.0)
        self.m_KCa = state.get('m_KCa', 0.0)
        self.Ca_internal = state.get('Ca_internal', self.celegans_params.Ca_rest)
    
    def reset_neuron(self) -> None:
        """Reset neuron to resting state."""
        super().reset_neuron()
        
        # Reset gating variables
        V_init = self.state.membrane_potential
        self.m_Ca = self._compute_m_inf(
            V_init, self.celegans_params.V_half_Ca, self.celegans_params.k_Ca
        )
        self.m_K = self._compute_m_inf(
            V_init, self.celegans_params.V_half_K, self.celegans_params.k_K
        )
        self.m_KCa = 0.0
        self.Ca_internal = self.celegans_params.Ca_rest
