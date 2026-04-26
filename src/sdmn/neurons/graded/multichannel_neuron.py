"""
Multi-channel C. elegans neuron model based on Nicoletti et al. (2019).

Supports per-neuron-type ion channel complements with individually parameterized
channels extracted from published electrophysiology. This model goes beyond the
generic Ca/K/K(Ca) triplet in celegans_neuron.py by implementing the actual
ion channel families expressed in C. elegans:

Voltage-gated K+ channels:
    SHL-1 (Shaker-like, fast transient)
    SHK-1 (Shaw-like, delayed rectifier)
    KVS-1 (voltage-sensitive, rapidly inactivating)
    KQT-3 (KCNQ family, M-current)
    EGL-2 (EAG-like, non-inactivating)
    EGL-36 (Shaw-type, triple kinetics)
    IRK (inward rectifier)

Voltage-gated Ca2+ channels:
    EGL-19 (L-type, sustained)
    UNC-2 (N/P/Q-type, high-voltage activated)
    CCA-1 (T-type, low-voltage transient)

Ca2+-activated K+ channels:
    SLO-1 (BK, large-conductance, Ca+V dependent)
    SLO-2 (related to SLO-1, Na/Ca activated)
    KCNL (SK, small-conductance, Ca-dependent)

References:
    Nicoletti M, et al. (2019) PLOS ONE 14(7):e0218738
    Goodman MB, et al. (1998) Neuron 20(4):763-772
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

from .graded_neuron import GradedNeuron, GradedNeuronParameters


FARADAY = 96485.0  # C/mol


@dataclass
class ChannelParams:
    """Parameters for a single ion channel type."""
    name: str
    g_max: float = 0.0          # nS
    E_rev: float = 0.0          # mV (key to which reversal: ek, eca, etc.)
    V_half_act: float = 0.0     # mV
    k_act: float = 10.0         # mV, slope
    V_half_inact: float = 0.0   # mV
    k_inact: float = 10.0       # mV
    tau_act_min: float = 0.5    # ms
    tau_act_max: float = 10.0   # ms
    tau_inact: float = 100.0    # ms
    act_power: int = 1
    has_inactivation: bool = False
    reversal_key: str = "ek"    # 'ek', 'eca', 'eleak', 'ena'


@dataclass
class MultiChannelParameters(GradedNeuronParameters):
    """
    Parameters for a multi-channel C. elegans neuron.

    Instead of a fixed Ca/K/K(Ca) triplet, this allows arbitrary combinations
    of ion channels, each with their own kinetics and conductances.
    """
    E_K: float = -80.0
    E_Ca: float = 60.0
    E_Na: float = 30.0

    channels: List[ChannelParams] = field(default_factory=list)

    # Ca dynamics
    Ca_rest: float = 50e-6      # mM (50 nM as in Nicoletti)
    tau_Ca_removal: float = 50.0  # ms
    f_Ca: float = 0.001
    cell_volume: float = 31.16  # um^3 (AWC default from NeuroMorpho)

    # SK channel Ca half-activation
    sk_Ca_half: float = 0.33e-3  # mM (from Nicoletti KCNL k_sk2)
    sk_tau: float = 6.3          # ms

    def __post_init__(self):
        super().__post_init__()


class IonChannelState:
    """Runtime state for a single ion channel."""
    __slots__ = ('params', 'm', 'h', 'current')

    def __init__(self, params: ChannelParams, V_init: float):
        self.params = params
        self.m = _boltzmann(V_init, params.V_half_act, params.k_act)
        self.h = 1.0 - _boltzmann(V_init, params.V_half_inact, params.k_inact) if params.has_inactivation else 1.0
        self.current = 0.0

    def steady_state_act(self, V: float) -> float:
        return _boltzmann(V, self.params.V_half_act, self.params.k_act)

    def steady_state_inact(self, V: float) -> float:
        return 1.0 - _boltzmann(V, self.params.V_half_inact, self.params.k_inact)

    def tau_activation(self, V: float) -> float:
        p = self.params
        frac = _boltzmann(V, (p.V_half_act + p.V_half_inact) / 2, p.k_act)
        tau = p.tau_act_min + (p.tau_act_max - p.tau_act_min) * (1.0 - frac)
        return max(tau, 0.01)


def _boltzmann(V: float, V_half: float, k: float) -> float:
    x = (V - V_half) / k
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class MultiChannelNeuron(GradedNeuron):
    """
    C. elegans neuron with arbitrary ion channel complement.

    Mathematical model:
        C_m * dV/dt = -I_leak - sum(I_channels) + I_ext + I_syn + I_gap

    Each channel i:
        I_i = g_i * m_i^p * h_i * (V - E_i)
        dm_i/dt = (m_inf(V) - m_i) / tau_m(V)
        dh_i/dt = (h_inf(V) - h_i) / tau_h

    Intracellular calcium:
        d[Ca]/dt = -f * sum(I_Ca) / (2*F*vol) - ([Ca] - [Ca]_rest) / tau_removal
    """

    def __init__(self, neuron_id: str, parameters: MultiChannelParameters):
        super().__init__(neuron_id, parameters)
        self.mc_params: MultiChannelParameters = parameters

        V_init = self.state.membrane_potential

        self.channel_states: List[IonChannelState] = [
            IonChannelState(cp, V_init) for cp in parameters.channels
        ]

        self.Ca_internal = parameters.Ca_rest
        self.sk_m = 0.0

    def _get_reversal(self, key: str) -> float:
        if key == "ek":
            return self.mc_params.E_K
        elif key == "eca":
            return self.mc_params.E_Ca
        elif key == "ena":
            return self.mc_params.E_Na
        elif key == "eleak":
            return self.mc_params.E_leak
        return 0.0

    def _compute_ionic_currents(self, V: float) -> float:
        total = 0.0

        for cs in self.channel_states:
            E_rev = self._get_reversal(cs.params.reversal_key)
            g = cs.params.g_max * (cs.m ** cs.params.act_power)
            if cs.params.has_inactivation:
                g *= cs.h
            cs.current = g * (V - E_rev)
            total += cs.current

        # SK (Ca-dependent K) channel
        sk_g = 0.0
        for cs in self.channel_states:
            if cs.params.name == "KCNL":
                sk_g = cs.params.g_max * self.sk_m
                cs.current = sk_g * (V - self.mc_params.E_K)
                total += cs.current - cs.params.g_max * (cs.m ** cs.params.act_power) * (V - self._get_reversal(cs.params.reversal_key))

        return -total

    def _update_gating_variables(self, dt: float) -> None:
        V = self.state.membrane_potential

        for cs in self.channel_states:
            if cs.params.name == "KCNL":
                continue

            m_inf = cs.steady_state_act(V)
            tau_m = cs.tau_activation(V)
            cs.m += dt * (m_inf - cs.m) / tau_m
            cs.m = np.clip(cs.m, 0.0, 1.0)

            if cs.params.has_inactivation:
                h_inf = cs.steady_state_inact(V)
                cs.h += dt * (h_inf - cs.h) / cs.params.tau_inact
                cs.h = np.clip(cs.h, 0.0, 1.0)

        # SK gating via intracellular calcium
        sk_inf = self.Ca_internal / (self.Ca_internal + self.mc_params.sk_Ca_half)
        self.sk_m += dt * (sk_inf - self.sk_m) / self.mc_params.sk_tau
        self.sk_m = np.clip(self.sk_m, 0.0, 1.0)

        # Calcium dynamics
        I_Ca_total = sum(
            cs.current for cs in self.channel_states
            if cs.params.reversal_key == "eca"
        )
        alpha_ca = 1.0 / (2.0 * self.mc_params.cell_volume * FARADAY * 1e-9)
        Ca_influx = -self.mc_params.f_Ca * alpha_ca * I_Ca_total if I_Ca_total < 0 else 0.0
        Ca_removal = (self.Ca_internal - self.mc_params.Ca_rest) / self.mc_params.tau_Ca_removal
        self.Ca_internal += dt * (Ca_influx - Ca_removal)
        self.Ca_internal = np.clip(self.Ca_internal, 0.0, 0.01)  # 0–10 mM max

    def get_channel_states(self) -> Dict[str, float]:
        result = {'Ca_internal': self.Ca_internal, 'sk_m': self.sk_m}
        for cs in self.channel_states:
            result[f'm_{cs.params.name}'] = cs.m
            if cs.params.has_inactivation:
                result[f'h_{cs.params.name}'] = cs.h
        return result

    def get_currents(self) -> Dict[str, float]:
        V = self.state.membrane_potential
        result = {
            'I_leak': self._compute_leak_current(V),
            'I_syn': self.I_syn_total,
            'I_gap': self.I_gap_total,
            'I_ext': self.I_ext_current,
        }
        for cs in self.channel_states:
            result[f'I_{cs.params.name}'] = cs.current
        return result

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['Ca_internal'] = self.Ca_internal
        state['sk_m'] = self.sk_m
        state['channels'] = {
            cs.params.name: {'m': cs.m, 'h': cs.h, 'I': cs.current}
            for cs in self.channel_states
        }
        return state

    def reset_neuron(self) -> None:
        super().reset_neuron()
        V_init = self.state.membrane_potential
        for cs in self.channel_states:
            cs.m = cs.steady_state_act(V_init)
            cs.h = cs.steady_state_inact(V_init) if cs.params.has_inactivation else 1.0
            cs.current = 0.0
        self.Ca_internal = self.mc_params.Ca_rest
        self.sk_m = 0.0


# ---------------------------------------------------------------
# Factory functions for specific neuron types
# ---------------------------------------------------------------

def create_awc_neuron(neuron_id: str, **overrides) -> MultiChannelNeuron:
    """Create AWCon chemosensory neuron with Nicoletti et al. (2019) parameters."""
    channels = [
        ChannelParams("SHL-1", g_max=2.9, V_half_act=-6.8, k_act=14.1,
                       V_half_inact=-15.1, k_inact=8.3,
                       tau_act_min=0.19, tau_act_max=2.0, tau_inact=50.0,
                       act_power=3, has_inactivation=True, reversal_key="ek"),
        ChannelParams("KVS-1", g_max=0.8, V_half_act=27.1, k_act=25.0,
                       V_half_inact=47.3, k_inact=11.1,
                       tau_act_min=0.1, tau_act_max=3.0, tau_inact=10.0,
                       act_power=1, has_inactivation=True, reversal_key="ek"),
        ChannelParams("SHK-1", g_max=0.1, V_half_act=20.4, k_act=7.7,
                       V_half_inact=-6.95, k_inact=5.8,
                       tau_act_min=2.0, tau_act_max=28.0, tau_inact=1400.0,
                       act_power=1, has_inactivation=True, reversal_key="ek"),
        ChannelParams("KQT-3", g_max=0.55, V_half_act=-2.67, k_act=15.8,
                       tau_act_min=1.0, tau_act_max=50.0,
                       act_power=1, has_inactivation=False, reversal_key="ek"),
        ChannelParams("EGL-2", g_max=0.85, V_half_act=-6.86, k_act=14.9,
                       tau_act_min=8.0, tau_act_max=17.0,
                       act_power=1, has_inactivation=False, reversal_key="ek"),
        ChannelParams("IRK", g_max=0.65, V_half_act=-22.0, k_act=-13.0,
                       tau_act_min=3.8, tau_act_max=20.0,
                       act_power=1, has_inactivation=False, reversal_key="ek"),
        ChannelParams("CCA-1", g_max=0.7, V_half_act=-27.65, k_act=2.38,
                       V_half_inact=-43.0, k_inact=8.05,
                       tau_act_min=0.7, tau_act_max=20.0, tau_inact=25.0,
                       act_power=2, has_inactivation=True, reversal_key="eca"),
        ChannelParams("UNC-2", g_max=1.0, V_half_act=12.83, k_act=3.97,
                       V_half_inact=-27.47, k_inact=5.6,
                       tau_act_min=0.1, tau_act_max=4.5, tau_inact=100.0,
                       act_power=1, has_inactivation=True, reversal_key="eca"),
        ChannelParams("EGL-19", g_max=1.55, V_half_act=15.6, k_act=7.5,
                       V_half_inact=20.0, k_inact=10.0,
                       tau_act_min=2.3, tau_act_max=5.0, tau_inact=30.0,
                       act_power=1, has_inactivation=True, reversal_key="eca"),
        ChannelParams("KCNL", g_max=0.06, reversal_key="ek"),
    ]

    params = MultiChannelParameters(
        C_m=1.4, g_leak=0.27, E_leak=-80.0,
        E_K=-80.0, E_Ca=60.0, E_Na=30.0,
        channels=channels,
        Ca_rest=50e-6, tau_Ca_removal=50.0, f_Ca=0.001,
        cell_volume=31.16,
        sk_Ca_half=0.33e-3, sk_tau=6.3,
        integration_method="RK4", dt=0.01,
    )
    return MultiChannelNeuron(neuron_id, params)


def create_rmd_neuron(neuron_id: str, **overrides) -> MultiChannelNeuron:
    """Create RMD motor neuron with Nicoletti et al. (2019) parameters."""
    channels = [
        ChannelParams("SHL-1", g_max=2.48, V_half_act=-6.8, k_act=14.1,
                       V_half_inact=-15.1, k_inact=8.3,
                       tau_act_min=0.19, tau_act_max=2.0, tau_inact=50.0,
                       act_power=3, has_inactivation=True, reversal_key="ek"),
        ChannelParams("SHK-1", g_max=1.1, V_half_act=20.4, k_act=7.7,
                       V_half_inact=-6.95, k_inact=5.8,
                       tau_act_min=2.0, tau_act_max=28.0, tau_inact=1400.0,
                       act_power=1, has_inactivation=True, reversal_key="ek"),
        ChannelParams("EGL-36", g_max=1.3, V_half_act=63.0, k_act=28.5,
                       tau_act_min=13.0, tau_act_max=355.0,
                       act_power=1, has_inactivation=False, reversal_key="ek"),
        ChannelParams("IRK", g_max=0.2, V_half_act=-22.0, k_act=-13.0,
                       tau_act_min=3.8, tau_act_max=20.0,
                       act_power=1, has_inactivation=False, reversal_key="ek"),
        ChannelParams("CCA-1", g_max=3.1, V_half_act=-27.65, k_act=2.38,
                       V_half_inact=-43.0, k_inact=8.05,
                       tau_act_min=0.7, tau_act_max=20.0, tau_inact=25.0,
                       act_power=2, has_inactivation=True, reversal_key="eca"),
        ChannelParams("UNC-2", g_max=0.9, V_half_act=12.83, k_act=3.97,
                       V_half_inact=-27.47, k_inact=5.6,
                       tau_act_min=0.1, tau_act_max=4.5, tau_inact=100.0,
                       act_power=1, has_inactivation=True, reversal_key="eca"),
        ChannelParams("EGL-19", g_max=0.99, V_half_act=15.6, k_act=7.5,
                       V_half_inact=20.0, k_inact=10.0,
                       tau_act_min=2.3, tau_act_max=5.0, tau_inact=30.0,
                       act_power=1, has_inactivation=True, reversal_key="eca"),
        ChannelParams("SLO-1", g_max=0.6, V_half_act=-20.0, k_act=15.0,
                       tau_act_min=0.5, tau_act_max=5.0,
                       act_power=1, has_inactivation=False, reversal_key="ek"),
        ChannelParams("SLO-2", g_max=0.6, V_half_act=-10.0, k_act=20.0,
                       tau_act_min=1.0, tau_act_max=10.0,
                       act_power=1, has_inactivation=False, reversal_key="ek"),
        ChannelParams("KCNL", g_max=0.06, reversal_key="ek"),
    ]

    params = MultiChannelParameters(
        C_m=1.2, g_leak=0.4, E_leak=-80.0,
        E_K=-80.0, E_Ca=60.0, E_Na=30.0,
        channels=channels,
        Ca_rest=50e-6, tau_Ca_removal=50.0, f_Ca=0.001,
        cell_volume=5.65,
        sk_Ca_half=0.33e-3, sk_tau=6.3,
        integration_method="RK4", dt=0.01,
    )
    return MultiChannelNeuron(neuron_id, params)
