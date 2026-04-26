"""
Validation module for comparing SDMN simulation outputs against
published C. elegans electrophysiology and behavioral data.
"""

from .single_neuron import SingleNeuronValidator
from .network_validation import NetworkValidator
