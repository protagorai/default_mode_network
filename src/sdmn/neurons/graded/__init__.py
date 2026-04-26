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
from .multichannel_neuron import (
    MultiChannelNeuron,
    MultiChannelParameters,
    ChannelParams,
    create_awc_neuron,
    create_rmd_neuron,
)
from .neuropeptide_layer import (
    ExtrasynapticLayer,
    NeuropeptideParams,
    MonoamineParams,
    create_serotonin_pathway,
    create_dopamine_pathway,
    create_tyramine_pathway,
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
    'MultiChannelNeuron',
    'MultiChannelParameters',
    'ChannelParams',
    'create_awc_neuron',
    'create_rmd_neuron',
    'ExtrasynapticLayer',
    'NeuropeptideParams',
    'MonoamineParams',
    'create_serotonin_pathway',
    'create_dopamine_pathway',
    'create_tyramine_pathway',
]

