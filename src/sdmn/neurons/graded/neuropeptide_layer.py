"""
Neuropeptide signaling layer for C. elegans network.

Models extrasynaptic neuropeptide and monoamine signaling that operates
in parallel to the synaptic connectome. This addresses the critical gap
identified by Randi et al. (2023) where functional signal propagation
differs substantially from anatomical connectivity predictions.

Architecture:
    - NeuropeptideSignal: a slow, diffuse modulatory signal from a source
      neuron to target neurons expressing the cognate receptor
    - MonoamineSignal: similar but for serotonin, dopamine, tyramine, octopamine
    - ExtrasynapticLayer: manages all non-synaptic signaling for a network

References:
    Randi F, et al. (2023) Nature 623:406-414
    Bentley B, et al. (2016) PLOS Comp Biol 12(12):e1005283
    Beets I, et al. (2023) Cell Reports 42(9):113058
    Ripoll-Sánchez L, et al. (2023) Neuron 111(22):3570-3589
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NeuropeptideParams:
    """Parameters for a single neuropeptide signaling pathway."""
    peptide_name: str = ""
    receptor_name: str = ""

    source_neurons: List[str] = field(default_factory=list)
    target_neurons: List[str] = field(default_factory=list)

    release_threshold: float = -30.0  # mV — voltage threshold for release
    release_slope: float = 8.0        # mV — shallower than synaptic release (slower)
    release_rate: float = 0.005       # per ms — slow release kinetics
    clearance_rate: float = 0.001     # per ms — slow clearance (seconds timescale)

    # Modulatory effect on targets
    effect_type: str = "conductance_mod"  # 'conductance_mod', 'leak_shift', 'threshold_shift'
    effect_magnitude: float = 0.1     # fractional change (0.1 = 10% modulation)
    effect_sign: float = 1.0          # +1 for facilitation, -1 for suppression

    ec50: float = 0.5                 # half-maximal effect concentration (arbitrary units)

    def __post_init__(self):
        if self.clearance_rate <= 0:
            raise ValueError("clearance_rate must be positive")


@dataclass
class MonoamineParams:
    """Parameters for monoamine (5-HT, DA, TA, OA) signaling."""
    amine_name: str = ""
    source_neurons: List[str] = field(default_factory=list)
    target_neurons: List[str] = field(default_factory=list)

    release_threshold: float = -35.0
    release_slope: float = 5.0
    release_rate: float = 0.01
    clearance_rate: float = 0.005

    effect_type: str = "excitability_mod"
    effect_magnitude: float = 0.15
    effect_sign: float = 1.0
    ec50: float = 0.5


class ExtrasynapticSignal:
    """Runtime state for a single extrasynaptic signaling pathway."""

    def __init__(self, params):
        self.params = params
        self.concentration = 0.0
        self.release_level = 0.0

    def update_release(self, source_voltages: Dict[str, float], dt: float) -> None:
        """Compute release based on source neuron voltages."""
        total_release = 0.0
        for name in self.params.source_neurons:
            if name in source_voltages:
                V = source_voltages[name]
                release = 1.0 / (1.0 + np.exp(
                    -(V - self.params.release_threshold) / self.params.release_slope
                ))
                total_release += release

        if self.params.source_neurons:
            total_release /= len(self.params.source_neurons)

        self.release_level = total_release

        d_conc = (self.params.release_rate * total_release -
                  self.params.clearance_rate * self.concentration)
        self.concentration += dt * d_conc
        self.concentration = np.clip(self.concentration, 0.0, 10.0)

    def get_effect(self) -> float:
        """Hill-function modulation based on current concentration."""
        return (self.params.effect_sign * self.params.effect_magnitude *
                self.concentration / (self.concentration + self.params.ec50))


class ExtrasynapticLayer:
    """
    Manages all neuropeptide and monoamine signaling for a network.

    Usage:
        layer = ExtrasynapticLayer()
        layer.add_neuropeptide_pathway(params)
        layer.add_monoamine_pathway(params)

        # Each simulation step:
        layer.update(voltages_dict, dt)
        modulations = layer.get_target_modulations()
        # Apply modulations to neurons before voltage integration
    """

    def __init__(self):
        self.neuropeptide_signals: List[ExtrasynapticSignal] = []
        self.monoamine_signals: List[ExtrasynapticSignal] = []

    def add_neuropeptide_pathway(self, params: NeuropeptideParams) -> None:
        self.neuropeptide_signals.append(ExtrasynapticSignal(params))

    def add_monoamine_pathway(self, params: MonoamineParams) -> None:
        self.monoamine_signals.append(ExtrasynapticSignal(params))

    def update(self, neuron_voltages: Dict[str, float], dt: float) -> None:
        for sig in self.neuropeptide_signals:
            sig.update_release(neuron_voltages, dt)
        for sig in self.monoamine_signals:
            sig.update_release(neuron_voltages, dt)

    def get_target_modulations(self) -> Dict[str, float]:
        """
        Returns a dict {neuron_name: total_modulation_factor} for all targets.

        Modulation > 0 means facilitation (increased excitability).
        Modulation < 0 means suppression (decreased excitability).
        """
        modulations: Dict[str, float] = {}

        for sig in self.neuropeptide_signals + self.monoamine_signals:
            effect = sig.get_effect()
            for target in sig.params.target_neurons:
                modulations[target] = modulations.get(target, 0.0) + effect

        return modulations

    def get_state(self) -> Dict[str, Any]:
        return {
            'neuropeptides': [
                {'name': s.params.peptide_name if hasattr(s.params, 'peptide_name') else s.params.amine_name,
                 'concentration': s.concentration, 'release': s.release_level}
                for s in self.neuropeptide_signals
            ],
            'monoamines': [
                {'name': s.params.amine_name if hasattr(s.params, 'amine_name') else '',
                 'concentration': s.concentration, 'release': s.release_level}
                for s in self.monoamine_signals
            ],
        }

    def reset(self) -> None:
        for sig in self.neuropeptide_signals + self.monoamine_signals:
            sig.concentration = 0.0
            sig.release_level = 0.0


# ---------------------------------------------------------------
# Predefined monoamine pathways from C. elegans literature
# ---------------------------------------------------------------

def create_serotonin_pathway() -> MonoamineParams:
    """
    Serotonin (5-HT) signaling pathway.

    Sources: NSM, HSN, ADF, AIM, RIH (Sze et al. 2000)
    Targets: Broadly modulatory, affects locomotion, feeding, egg-laying
    """
    return MonoamineParams(
        amine_name="serotonin",
        source_neurons=["NSML", "NSMR", "HSNL", "HSNR", "ADFL", "ADFR", "AIML", "AIMR", "RIH"],
        target_neurons=[],  # Populated from receptor expression data
        release_threshold=-35.0,
        release_slope=5.0,
        release_rate=0.008,
        clearance_rate=0.003,
        effect_type="excitability_mod",
        effect_magnitude=0.2,
        effect_sign=1.0,
    )


def create_dopamine_pathway() -> MonoamineParams:
    """
    Dopamine signaling pathway.

    Sources: CEP (4), ADE (2), PDE (2) (Sulston et al. 1975)
    Targets: Modulates locomotion speed, area-restricted search
    """
    return MonoamineParams(
        amine_name="dopamine",
        source_neurons=["CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER"],
        target_neurons=[],
        release_threshold=-35.0,
        release_slope=5.0,
        release_rate=0.01,
        clearance_rate=0.005,
        effect_type="excitability_mod",
        effect_magnitude=0.15,
        effect_sign=-1.0,  # Generally slows locomotion
    )


def create_tyramine_pathway() -> MonoamineParams:
    """
    Tyramine signaling pathway.

    Sources: RIM (Alkema et al. 2005)
    Targets: Suppresses head oscillations during reversals
    """
    return MonoamineParams(
        amine_name="tyramine",
        source_neurons=["RIML", "RIMR"],
        target_neurons=[],
        release_threshold=-30.0,
        release_slope=5.0,
        release_rate=0.01,
        clearance_rate=0.005,
        effect_type="excitability_mod",
        effect_magnitude=0.15,
        effect_sign=-1.0,
    )
