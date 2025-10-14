"""
Graded potential neuron models for C. elegans simulations.

This module implements graded (non-spiking) neuron models that use continuous,
analog voltage changes rather than discrete action potentials. These models are
based on empirical data from C. elegans neurons.
"""

from .graded_neuron import GradedNeuron, GradedNeuronParameters
from .celegans_neuron import CElegansNeuron, CElegansParameters
from .neuron_classes import (
    SensoryNeuron,
    Interneuron,
    MotorNeuron,
    CElegansNeuronClass
)

__all__ = [
    'GradedNeuron',
    'GradedNeuronParameters',
    'CElegansNeuron',
    'CElegansParameters',
    'SensoryNeuron',
    'Interneuron',
    'MotorNeuron',
    'CElegansNeuronClass',
]

