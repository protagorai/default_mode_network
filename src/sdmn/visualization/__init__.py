"""
Live real-time visualization for C. elegans neural network simulations.

Provides browser-based (Starlette + WebSocket + HTML5 Canvas) and
matplotlib-based backends that display neuron activations, worm body
movement in a 2D arena, population activity timeline, DMN baseline,
and interactive stimulus injection.
"""

from sdmn.visualization.live_visualizer import LiveVisualizer, VisualizationFrame, brightness
from sdmn.visualization.layouts import grid_layout, graph_layout, anatomical_layout, compute_layout
from sdmn.visualization.browser_backend import BrowserVisualizer
from sdmn.visualization.matplotlib_backend import MatplotlibVisualizer
from sdmn.visualization.environment import WormEnvironment, WormState, ChemicalSource

__all__ = [
    "LiveVisualizer",
    "VisualizationFrame",
    "BrowserVisualizer",
    "MatplotlibVisualizer",
    "WormEnvironment",
    "WormState",
    "ChemicalSource",
    "brightness",
    "grid_layout",
    "graph_layout",
    "anatomical_layout",
    "compute_layout",
]
