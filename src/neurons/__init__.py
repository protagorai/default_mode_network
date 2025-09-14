"""
Neuron models and interfaces for the Synthetic Default Mode Network Framework.

This module provides biologically-inspired neuron models with standardized
interfaces for interchangeable use in neural network simulations.
"""

from .base_neuron import BaseNeuron, NeuronParameters, NeuronState
from .lif_neuron import LIFNeuron, LIFParameters
from .hh_neuron import HHNeuron, HHParameters
from .synapse import Synapse, SynapseType, SynapticParameters

__all__ = [
    'BaseNeuron',
    'NeuronParameters', 
    'NeuronState',
    'LIFNeuron',
    'LIFParameters',
    'HHNeuron', 
    'HHParameters',
    'Synapse',
    'SynapseType',
    'SynapticParameters'
]
