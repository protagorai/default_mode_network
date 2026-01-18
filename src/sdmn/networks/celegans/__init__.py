"""
C. elegans network construction and management tools.

This module provides utilities for building and simulating C. elegans
neural networks with graded potential neurons.
"""

from .network_manager import CElegansNetwork, SimulationState, SimulationConfig
from .topology_builders import (
    TopologyParameters,
    UniformTopologyBuilder,
    SmallWorldBuilder,
    RandomTopologyBuilder,
    build_uniform_network,
    build_small_world_network,
    build_random_network
)

__all__ = [
    'CElegansNetwork',
    'SimulationState',
    'SimulationConfig',
    'TopologyParameters',
    'UniformTopologyBuilder',
    'SmallWorldBuilder',
    'RandomTopologyBuilder',
    'build_uniform_network',
    'build_small_world_network',
    'build_random_network',
]


