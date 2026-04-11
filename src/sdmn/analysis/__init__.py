"""
Analysis tools for C. elegans neural network simulations.

Provides topology metrics, dynamical analysis, behavioral detection,
and visualization utilities.
"""

from sdmn.analysis.topology import TopologyAnalyzer
from sdmn.analysis.dynamics import DynamicsAnalyzer
from sdmn.analysis.behavior import BehaviorDetector, LocomotionEvent, ActivityState
from sdmn.analysis.visualization import NetworkVisualizer

__all__ = [
    'TopologyAnalyzer',
    'DynamicsAnalyzer',
    'BehaviorDetector',
    'LocomotionEvent',
    'ActivityState',
    'NetworkVisualizer',
]
