"""
C. elegans network construction and management tools.

This module provides utilities for building and simulating C. elegans
neural networks with graded potential neurons.
"""

from .network_manager import CElegansNetwork, SimulationState

__all__ = [
    'CElegansNetwork',
    'SimulationState',
]


