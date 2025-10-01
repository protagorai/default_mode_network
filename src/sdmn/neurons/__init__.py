"""
Neuron models and interfaces for the Synthetic Default Mode Network Framework.

This module provides biologically-inspired neuron models with standardized
interfaces for interchangeable use in neural network simulations.
"""

from sdmn.neurons.base_neuron import (
    BaseNeuron, 
    NeuronParameters, 
    NeuronState, 
    NeuronType,
    NeuronFactory
)
from sdmn.neurons.lif_neuron import LIFNeuron, LIFParameters
from sdmn.neurons.hh_neuron import HHNeuron, HHParameters
from sdmn.neurons.synapse import (
    Synapse, 
    BaseSynapse,
    SynapseType, 
    SynapticParameters,
    SynapseFactory,
    NeurotransmitterType
)

__all__ = [
    'BaseNeuron',
    'NeuronParameters', 
    'NeuronState',
    'NeuronType',
    'NeuronFactory',
    'LIFNeuron',
    'LIFParameters',
    'HHNeuron', 
    'HHParameters',
    'Synapse',
    'BaseSynapse',
    'SynapseType',
    'SynapticParameters',
    'SynapseFactory',
    'NeurotransmitterType'
]
