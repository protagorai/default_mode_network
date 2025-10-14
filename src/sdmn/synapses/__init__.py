"""
Synaptic connections for neural networks.

This module implements various types of synaptic connections including:
- Spike-triggered synapses (for LIF/HH neurons)
- Graded chemical synapses (for graded potential neurons)
- Gap junctions (electrical synapses)
"""

from sdmn.neurons.synapse import (
    BaseSynapse,
    Synapse,
    SynapseType,
    SynapticParameters,
    NeurotransmitterType,
    SynapseFactory
)

from .graded_synapse import (
    GradedChemicalSynapse,
    GradedSynapseParameters
)

from .gap_junction import (
    GapJunction,
    GapJunctionParameters
)

__all__ = [
    # Base classes
    'BaseSynapse',
    'SynapseType',
    'NeurotransmitterType',
    
    # Spike-triggered synapses (existing)
    'Synapse',
    'SynapticParameters',
    'SynapseFactory',
    
    # Graded synapses (new)
    'GradedChemicalSynapse',
    'GradedSynapseParameters',
    
    # Gap junctions (new)
    'GapJunction',
    'GapJunctionParameters',
]

