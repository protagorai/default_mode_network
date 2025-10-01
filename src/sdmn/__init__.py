"""
Synthetic Default Mode Network Framework
======================================

A comprehensive framework for simulating biologically-inspired neural networks
with focus on default mode network dynamics and emergent properties.

This package provides:
- Core simulation engine with time management and event handling
- Various neuron models (LIF, Hodgkin-Huxley, etc.)
- Network topology builders and connectivity patterns
- Comprehensive monitoring and data collection probes
- State management and checkpointing capabilities
"""

from sdmn.version import __version__

# Core simulation components
from sdmn.core import (
    SimulationEngine,
    SimulationResults,
    Event,
    EventQueue,
    EventType,
    StateManager,
    TimeManager,
)

# Neuron models and interfaces
from sdmn.neurons import (
    BaseNeuron,
    NeuronParameters,
    NeuronState,
    LIFNeuron,
    LIFParameters,
    HHNeuron,
    HHParameters,
    Synapse,
    SynapseType,
    SynapticParameters,
)

# Network building and analysis
from sdmn.networks import NetworkBuilder, NetworkTopology

# Monitoring and data collection
from sdmn.probes import (
    BaseProbe,
    ProbeType,
    ProbeData,
    VoltageProbe,
    SpikeProbe,
    PopulationActivityProbe,
    NetworkActivityProbe,
)

__all__ = [
    "__version__",
    # Core components
    "SimulationEngine",
    "SimulationResults",
    "Event",
    "EventQueue", 
    "EventType",
    "StateManager",
    "TimeManager",
    # Neuron models
    "BaseNeuron",
    "NeuronParameters",
    "NeuronState",
    "LIFNeuron",
    "LIFParameters",
    "HHNeuron",
    "HHParameters",
    "Synapse",
    "SynapseType",
    "SynapticParameters",
    # Network building
    "NetworkBuilder",
    "NetworkTopology",
    # Monitoring
    "BaseProbe",
    "ProbeType",
    "ProbeData",
    "VoltageProbe",
    "SpikeProbe", 
    "PopulationActivityProbe",
    "NetworkActivityProbe",
]


def get_version() -> str:
    """Get the current version of the package."""
    return __version__


def get_info() -> dict:
    """Get package information."""
    return {
        "name": "synthetic-default-mode-network",
        "version": __version__,
        "description": "A framework for simulating synthetic default mode networks",
        "author": "SDMN Team",
        "license": "MIT",
    }
