"""
Neuromodulation system for C. elegans simulation.

Models three neuromodulatory systems using real neuron identities:
  - **Dopamine** (DA): reward signal, gates learning, food detection
  - **Serotonin** (5-HT): satiety, feeding state, slows locomotion
  - **Octopamine** (OA): stress/starvation, aversive signal, speeds locomotion

Each modulator is a slow signal (tau ~500-2000 ms) that diffuses globally
and modulates synapse weights, neuron excitability, and learning rates.

Neuron identities from Varshney et al. (2011) / WormAtlas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

# Real C. elegans neuromodulatory neurons
DA_NEURONS = {"CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER"}
SEROTONIN_NEURONS = {"NSML", "NSMR"}
OCTOPAMINE_NEURONS = {"RICL", "RICR"}  # RIC produces octopamine/tyramine
TYRAMINE_NEURONS = {"RIML", "RIMR"}  # RIM produces tyramine

# ASH: main nociceptive neurons
NOCICEPTIVE_NEURONS = {"ASHL", "ASHR"}

# Chemosensory neurons that detect food
FOOD_SENSORY = {"AWCL", "AWCR", "ASEL", "ASER", "AWAL", "AWAR"}


@dataclass
class NeuromodConfig:
    """Configuration for the neuromodulation system."""
    enable_dopamine: bool = True
    enable_serotonin: bool = True
    enable_octopamine: bool = True
    enable_nociception: bool = True
    enable_sensory_adaptation: bool = True
    enable_reward_learning: bool = True

    da_tau: float = 1000.0          # Dopamine decay time constant (ms)
    da_release_gain: float = 1.0    # How much DA neurons release per activation
    da_excitability_mod: float = 0.3  # How much DA modulates target excitability

    serotonin_tau: float = 2000.0   # Serotonin decay (ms), very slow
    serotonin_release_gain: float = 0.8
    serotonin_speed_mod: float = -0.5  # Negative = slows locomotion

    octopamine_tau: float = 800.0   # Octopamine decay (ms)
    octopamine_release_gain: float = 1.2
    octopamine_speed_mod: float = 0.5  # Positive = speeds locomotion

    sensory_adapt_tau: float = 50.0  # Adaptation time constant (ms)
    sensory_adapt_strength: float = 0.8  # How much adaptation reduces response

    reward_learning_rate: float = 0.001  # Three-factor Hebbian rate
    reward_eligibility_tau: float = 100.0  # Eligibility trace decay (ms)


class NeuromodulationSystem:
    """
    Manages neuromodulatory signals and their effects on the network.

    Call ``step()`` each simulation batch with current voltages.
    Read ``get_modulatory_currents()`` for per-neuron excitability modulation.
    Read ``get_reward_signal()`` for the three-factor learning gate.
    """

    def __init__(self, config: NeuromodConfig, neuron_ids: List[str],
                 id_to_idx: Dict[str, int]):
        self.config = config
        self.neuron_ids = neuron_ids
        self.id_to_idx = id_to_idx
        self.n = len(neuron_ids)

        # Global modulator levels (0 to 1 scale)
        self.dopamine: float = 0.0
        self.serotonin: float = 0.0
        self.octopamine: float = 0.0

        # Identify modulator neuron indices
        self.da_indices = [id_to_idx[n] for n in DA_NEURONS if n in id_to_idx]
        self.serotonin_indices = [id_to_idx[n] for n in SEROTONIN_NEURONS if n in id_to_idx]
        self.octopamine_indices = [id_to_idx[n] for n in OCTOPAMINE_NEURONS if n in id_to_idx]
        self.nociceptive_indices = [id_to_idx[n] for n in NOCICEPTIVE_NEURONS if n in id_to_idx]
        self.food_sensory_indices = [id_to_idx[n] for n in FOOD_SENSORY if n in id_to_idx]

        # Sensory adaptation state (per-neuron previous input for derivative)
        self.prev_input = np.zeros(self.n)
        self.adaptation = np.zeros(self.n)

        # Reward-modulated STDP eligibility traces (per-synapse)
        self.eligibility_traces: Optional[np.ndarray] = None

        # Food/pain state
        self.food_consumed: float = 0.0
        self.pain_level: float = 0.0

    def init_eligibility(self, n_synapses: int) -> None:
        """Initialize eligibility traces for reward-modulated STDP."""
        self.eligibility_traces = np.zeros(n_synapses)

    def step(self, V: np.ndarray, I_ext: np.ndarray, dt_batch: float) -> None:
        """
        Update neuromodulator levels based on neuron activity.

        Args:
            V: Current voltages (n_neurons,)
            I_ext: External current array (modified in-place for nociception)
            dt_batch: Total simulation time for this batch (ms)
        """
        activation = np.clip((V - (-65.0)) / 25.0, 0.0, 1.0)

        # --- Dopamine: released by DA neurons when they're active ---
        if self.config.enable_dopamine and self.da_indices:
            da_activity = np.mean(activation[self.da_indices])
            da_release = da_activity * self.config.da_release_gain
            decay = np.exp(-dt_batch / self.config.da_tau)
            self.dopamine = self.dopamine * decay + da_release * (1 - decay)
            self.dopamine = min(1.0, self.dopamine)

        # --- Serotonin: released by NSM neurons ---
        if self.config.enable_serotonin and self.serotonin_indices:
            sht_activity = np.mean(activation[self.serotonin_indices])
            sht_release = sht_activity * self.config.serotonin_release_gain
            decay = np.exp(-dt_batch / self.config.serotonin_tau)
            self.serotonin = self.serotonin * decay + sht_release * (1 - decay)
            self.serotonin = min(1.0, self.serotonin)

        # --- Octopamine: released by RIC neurons (stress/starvation) ---
        if self.config.enable_octopamine and self.octopamine_indices:
            oa_activity = np.mean(activation[self.octopamine_indices])
            oa_release = oa_activity * self.config.octopamine_release_gain
            decay = np.exp(-dt_batch / self.config.octopamine_tau)
            self.octopamine = self.octopamine * decay + oa_release * (1 - decay)
            self.octopamine = min(1.0, self.octopamine)

        # --- Sensory adaptation: respond to CHANGE, not absolute level ---
        if self.config.enable_sensory_adaptation:
            current_input = I_ext.copy()
            delta = current_input - self.prev_input
            self.adaptation += (np.abs(delta) - self.adaptation) * (dt_batch / self.config.sensory_adapt_tau)
            self.adaptation = np.clip(self.adaptation, 0, 10)
            self.prev_input = current_input

    def get_modulatory_currents(self) -> np.ndarray:
        """
        Return per-neuron excitability modulation current (pA).

        Dopamine increases excitability of food-sensory and motor neurons.
        Serotonin decreases general excitability (calming).
        Octopamine increases excitability broadly (arousal).
        """
        mod = np.zeros(self.n)

        if self.config.enable_dopamine:
            da_boost = self.dopamine * self.config.da_excitability_mod * 20.0
            for idx in self.food_sensory_indices:
                mod[idx] += da_boost

        if self.config.enable_serotonin:
            sht_suppress = self.serotonin * 5.0
            mod -= sht_suppress  # global calming

        if self.config.enable_octopamine:
            oa_boost = self.octopamine * 8.0
            mod += oa_boost  # global arousal

        return mod

    def get_reward_signal(self) -> float:
        """
        Combined reward signal for three-factor learning.

        Positive reward = dopamine (food found).
        Negative reward = octopamine (pain/stress).
        Serotonin modulates learning rate (satiety reduces learning).
        """
        reward = self.dopamine - self.octopamine * 0.5
        satiety_gate = 1.0 - self.serotonin * 0.5
        return reward * satiety_gate

    def get_speed_modulation(self) -> float:
        """
        Locomotion speed modulation factor.

        Serotonin slows (feeding/satiety). Octopamine speeds (escape/search).
        Returns multiplier around 1.0.
        """
        mod = 1.0
        if self.config.enable_serotonin:
            mod += self.serotonin * self.config.serotonin_speed_mod
        if self.config.enable_octopamine:
            mod += self.octopamine * self.config.octopamine_speed_mod
        return max(0.2, min(3.0, mod))

    def update_eligibility(self, pre_trace: np.ndarray, post_trace: np.ndarray,
                           syn_pre: np.ndarray, syn_post: np.ndarray,
                           dt_batch: float) -> None:
        """Update per-synapse eligibility traces for three-factor learning."""
        if self.eligibility_traces is None or not self.config.enable_reward_learning:
            return
        decay = np.exp(-dt_batch / self.config.reward_eligibility_tau)
        hebbian = pre_trace[syn_pre] * post_trace[syn_post]
        self.eligibility_traces = self.eligibility_traces * decay + hebbian

    def apply_reward_learning(self, syn_weight: np.ndarray,
                              syn_weight_initial: np.ndarray,
                              w_min: float, w_max_factor: float) -> None:
        """Apply three-factor weight update: dw = learning_rate * eligibility * reward."""
        if self.eligibility_traces is None or not self.config.enable_reward_learning:
            return
        reward = self.get_reward_signal()
        if abs(reward) < 0.01:
            return
        dw = self.config.reward_learning_rate * self.eligibility_traces * reward
        syn_weight += dw
        w_max = syn_weight_initial * w_max_factor
        np.clip(syn_weight, w_min, w_max, out=syn_weight)

    def to_json(self) -> dict:
        """Serialize state for WebSocket."""
        return {
            "dopamine": round(self.dopamine, 4),
            "serotonin": round(self.serotonin, 4),
            "octopamine": round(self.octopamine, 4),
            "reward": round(self.get_reward_signal(), 4),
            "speed_mod": round(self.get_speed_modulation(), 3),
            "food_consumed": round(self.food_consumed, 2),
            "pain_level": round(self.pain_level, 3),
        }

    def reset(self) -> None:
        self.dopamine = 0.0
        self.serotonin = 0.0
        self.octopamine = 0.0
        self.prev_input[:] = 0
        self.adaptation[:] = 0
        self.food_consumed = 0.0
        self.pain_level = 0.0
        if self.eligibility_traces is not None:
            self.eligibility_traces[:] = 0
