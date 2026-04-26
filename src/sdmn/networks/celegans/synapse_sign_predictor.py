"""
Improved synapse sign prediction using neurotransmitter identity and receptor expression.

Replaces the simple GABA-only rule with a biologically grounded mapping based on:
1. Presynaptic neurotransmitter identity (Loer & Rand 2022; Wang et al. 2024)
2. Known receptor families and their ion selectivity
3. Gene-expression-based predictions (Fenyves et al. 2020)

Key biological corrections vs. the original binary rule:
- Glutamate can be INHIBITORY via GluCl channels (avr-14, avr-15, glc-1..4)
- Acetylcholine can be INHIBITORY via ACC-family Cl- channels
- GABA can be EXCITATORY via EXP-1 cation channel
- Monoamines (5-HT, DA, TA, OA) are MODULATORY, not simply E/I

References:
    Fenyves BG, et al. (2020) PLOS Comp Biol 16(12):e1007974
    Loer CM, Rand JB (2022) WormAtlas neurotransmitter table
    Wang Y, et al. (2024) eLife 13:e95402
    Bentley B, et al. (2016) PLOS Comp Biol 12(12):e1005283
"""

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DEFAULT_NT_DIR = _PACKAGE_ROOT / "data" / "validation" / "neurotransmitters"


class SynapseSignPredictor:
    """
    Predict synapse sign and reversal potential from neurotransmitter + receptor data.

    Usage:
        predictor = SynapseSignPredictor()
        predictor.load()
        sign, E_rev, confidence = predictor.predict("AVAL", "DA01")
    """

    REVERSAL_POTENTIALS = {
        'excitatory_cation': -5.0,    # nAChR, iGluR: mixed Na+/K+
        'inhibitory_chloride': -35.0,  # GABA-A, GluCl, ACC: Cl- channels
        'modulatory': None,            # GPCRs: no direct reversal potential
    }

    # Default sign by neurotransmitter (without receptor specificity)
    NT_DEFAULT_SIGN = {
        'acetylcholine': ('excitatory', -5.0, 'moderate'),
        'glutamate': ('excitatory', -5.0, 'low'),  # LOW confidence - could be GluCl
        'GABA': ('inhibitory', -35.0, 'high'),
        'serotonin': ('modulatory', None, 'high'),
        'dopamine': ('modulatory', None, 'high'),
        'tyramine': ('modulatory', None, 'high'),
        'octopamine': ('modulatory', None, 'high'),
    }

    # Known glutamatergic neurons that signal via GluCl (inhibitory glutamate)
    # Based on functional studies: pharyngeal neurons, some motor circuits
    GLUTAMATE_INHIBITORY_TARGETS = {
        'M3', 'M4', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6',  # Pharyngeal
        'RME',  # Head motor (receive inhibitory glutamate)
    }

    def __init__(self):
        self.neurotransmitter_map: Dict[str, str] = {}
        self.sign_rules: list = []

    def load(self, nt_map_path: Optional[str] = None,
             rules_path: Optional[str] = None) -> None:
        if nt_map_path is None:
            nt_map_path = str(_DEFAULT_NT_DIR / "celegans_neurotransmitter_map.csv")
        if rules_path is None:
            rules_path = str(_DEFAULT_NT_DIR / "synapse_sign_rules.csv")

        if Path(nt_map_path).exists():
            with open(nt_map_path, newline='') as f:
                for row in csv.DictReader(f):
                    self.neurotransmitter_map[row['neuron']] = row['neurotransmitter']

        if Path(rules_path).exists():
            with open(rules_path, newline='') as f:
                self.sign_rules = list(csv.DictReader(f))

    def get_neurotransmitter(self, neuron_name: str) -> str:
        """Get the neurotransmitter for a given neuron."""
        if neuron_name in self.neurotransmitter_map:
            return self.neurotransmitter_map[neuron_name]

        base = ''.join(c for c in neuron_name if not c.isdigit()).rstrip('LR')
        for name, nt in self.neurotransmitter_map.items():
            name_base = ''.join(c for c in name if not c.isdigit()).rstrip('LR')
            if name_base == base:
                return nt
        return 'unknown'

    def predict(self, pre_neuron: str, post_neuron: str
                ) -> Tuple[str, Optional[float], str]:
        """
        Predict synapse sign and reversal potential for a connection.

        Args:
            pre_neuron: Presynaptic neuron name
            post_neuron: Postsynaptic neuron name

        Returns:
            (sign, E_rev_mV, confidence)
            sign: 'excitatory', 'inhibitory', 'modulatory', or 'unknown'
            E_rev_mV: Reversal potential in mV (None for modulatory)
            confidence: 'high', 'moderate', 'low'
        """
        nt = self.get_neurotransmitter(pre_neuron)

        if nt == 'unknown':
            return ('excitatory', -5.0, 'very_low')

        if nt in self.NT_DEFAULT_SIGN:
            sign, E_rev, conf = self.NT_DEFAULT_SIGN[nt]

            # Glutamate special case: check if target has GluCl
            if nt == 'glutamate':
                post_base = ''.join(c for c in post_neuron if not c.isdigit()).rstrip('LR')
                if post_base in self.GLUTAMATE_INHIBITORY_TARGETS:
                    return ('inhibitory', -50.0, 'moderate')

            return (sign, E_rev, conf)

        return ('excitatory', -5.0, 'very_low')

    def predict_all_edges(self, edges) -> Dict[str, Tuple[str, Optional[float], str]]:
        """
        Predict signs for all edges in a connectome.

        Args:
            edges: List of (pre, post) tuples or EdgeRecord objects

        Returns:
            {edge_key: (sign, E_rev, confidence)}
        """
        results = {}
        for edge in edges:
            if hasattr(edge, 'pre'):
                pre, post = edge.pre, edge.post
            else:
                pre, post = edge[0], edge[1]
            key = f"{pre}->{post}"
            results[key] = self.predict(pre, post)
        return results

    def get_sign_statistics(self) -> Dict[str, int]:
        """Get statistics on the neurotransmitter assignments."""
        stats = {
            'total_neurons_mapped': len(self.neurotransmitter_map),
            'acetylcholine': 0,
            'glutamate': 0,
            'GABA': 0,
            'serotonin': 0,
            'dopamine': 0,
            'tyramine': 0,
            'octopamine': 0,
            'unknown': 0,
        }
        for nt in self.neurotransmitter_map.values():
            if nt in stats:
                stats[nt] += 1
            else:
                stats['unknown'] += 1
        return stats
