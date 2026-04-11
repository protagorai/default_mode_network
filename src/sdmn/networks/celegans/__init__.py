"""
C. elegans network construction and management tools.

This module provides utilities for building and simulating C. elegans
neural networks with graded potential neurons.
"""

from .network_manager import (
    CElegansNetwork, SimulationState, SimulationConfig, SimulationFrame,
    PlasticityConfig, DMNConfig,
)
from .topology_builders import (
    TopologyParameters,
    BaseTopologyBuilder,
    UniformTopologyBuilder,
    SmallWorldBuilder,
    RandomTopologyBuilder,
    ScaleFreeBuilder,
    build_uniform_network,
    build_small_world_network,
    build_random_network,
    build_scale_free_network,
)
from .connectome_loader import (
    ConnectomeLoader,
    ConnectomeData,
    NeuronRecord,
    EdgeRecord,
    build_connectome_network,
)

__all__ = [
    'CElegansNetwork',
    'SimulationState',
    'SimulationConfig',
    'SimulationFrame',
    'PlasticityConfig',
    'DMNConfig',
    'TopologyParameters',
    'BaseTopologyBuilder',
    'UniformTopologyBuilder',
    'SmallWorldBuilder',
    'RandomTopologyBuilder',
    'ScaleFreeBuilder',
    'ConnectomeLoader',
    'ConnectomeData',
    'NeuronRecord',
    'EdgeRecord',
    'build_uniform_network',
    'build_small_world_network',
    'build_random_network',
    'build_scale_free_network',
    'build_connectome_network',
]


