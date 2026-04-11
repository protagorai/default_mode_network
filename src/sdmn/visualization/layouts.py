"""
Layout computation for neuron positions in live visualizations.

Provides grid (3-band by class) and graph (NetworkX spring/Kamada-Kawai)
layout strategies that map neuron IDs to normalised (x, y) coordinates.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from sdmn.networks.celegans.network_manager import CElegansNetwork


def _neuron_class(network: CElegansNetwork, nid: str) -> str:
    neuron = network.neurons.get(nid)
    if neuron and hasattr(neuron, "neuron_class"):
        return neuron.neuron_class.value
    return "unknown"


def grid_layout(network: CElegansNetwork,
                columns: int = 14,
                padding: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """
    Arrange neurons in three horizontal bands (sensory / interneuron / motor).

    Returns normalised positions in [0, 1] x [0, 1].
    """
    groups: Dict[str, List[str]] = {"sensory": [], "interneuron": [], "motor": [], "unknown": []}
    for nid in sorted(network.neurons.keys()):
        cls = _neuron_class(network, nid)
        groups.get(cls, groups["unknown"]).append(nid)

    band_order = ["sensory", "interneuron", "motor"]
    if groups["unknown"]:
        band_order.append("unknown")

    positions: Dict[str, Tuple[float, float]] = {}
    n_bands = len(band_order)

    for band_idx, cls in enumerate(band_order):
        ids = groups[cls]
        if not ids:
            continue
        n = len(ids)
        rows = max(1, math.ceil(n / columns))
        band_height = (1.0 - padding * (n_bands + 1)) / n_bands
        y_base = padding + band_idx * (band_height + padding)

        for i, nid in enumerate(ids):
            col = i % columns
            row = i // columns
            x = padding + (col + 0.5) * ((1.0 - 2 * padding) / columns)
            y = y_base + (row + 0.5) * (band_height / rows)
            positions[nid] = (x, y)

    return positions


def graph_layout(network: CElegansNetwork,
                 algorithm: str = "spring",
                 seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Compute positions using a NetworkX graph-layout algorithm.

    Supported algorithms: ``"spring"``, ``"kamada_kawai"``, ``"circular"``.
    Returns normalised positions in [0, 1] x [0, 1].
    """
    g = nx.Graph()
    for nid in network.neurons:
        g.add_node(nid)
    for syn in network.chemical_synapses.values():
        g.add_edge(syn.presynaptic_neuron_id, syn.postsynaptic_neuron_id)
    for gap in network.gap_junctions.values():
        g.add_edge(gap.presynaptic_neuron_id, gap.postsynaptic_neuron_id)

    if algorithm == "kamada_kawai":
        raw = nx.kamada_kawai_layout(g)
    elif algorithm == "circular":
        raw = nx.circular_layout(g)
    else:
        raw = nx.spring_layout(g, seed=seed, k=1.5 / max(1, g.number_of_nodes() ** 0.5))

    if not raw:
        return {}

    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    margin = 0.05
    positions: Dict[str, Tuple[float, float]] = {}
    for nid, (px, py) in raw.items():
        positions[nid] = (
            margin + (px - x_min) / x_range * (1.0 - 2 * margin),
            margin + (py - y_min) / y_range * (1.0 - 2 * margin),
        )
    return positions


_HEAD_NEURONS = {
    "AWCL", "AWCR", "AWAL", "AWAR", "AWBL", "AWBR", "ASEL", "ASER",
    "ASHL", "ASHR", "ADFL", "ADFR", "ADLL", "ADLR", "AFDL", "AFDR",
    "ASGL", "ASGR", "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
    "IL1DL", "IL1DR", "IL1L", "IL1R", "IL1VL", "IL1VR",
    "IL2DL", "IL2DR", "IL2L", "IL2R", "IL2VL", "IL2VR",
    "OLLL", "OLLR", "OLQDL", "OLQDR", "OLQVL", "OLQVR",
    "CEPDL", "CEPDR", "CEPVL", "CEPVR", "URXL", "URXR",
    "BAGL", "BAGR", "FLPL", "FLPR", "AQR",
}
_NERVE_RING = {
    "AIAL", "AIAR", "AIBL", "AIBR", "AIML", "AIMR",
    "AINL", "AINR", "AIYL", "AIYR", "AIZL", "AIZR",
    "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR",
    "AVEL", "AVER", "AVFL", "AVFR", "AVG", "AVHL", "AVHR",
    "AVJL", "AVJR", "AVKL", "AVKR", "AVL", "AVM",
    "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR",
    "RIFL", "RIFR", "RIGL", "RIGR", "RIH", "RIML", "RIMR",
    "RIPL", "RIPR", "RIR", "RIS", "RIVL", "RIVR",
    "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR",
    "RMED", "RMEL", "RMER", "RMEV", "RMFL", "RMFR",
    "RMGL", "RMGR", "RMHL", "RMHR",
    "SAADL", "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR",
    "SIADL", "SIADR", "SIAVL", "SIAVR",
    "SIBDL", "SIBDR", "SIBVL", "SIBVR",
    "SMBDL", "SMBDR", "SMBVL", "SMBVR",
    "SMDDL", "SMDDR", "SMDVL", "SMDVR",
    "URADL", "URADR", "URAVL", "URAVR",
    "URBL", "URBR", "URYDL", "URYDR", "URYVL", "URYVR",
    "ADAL", "ADAR", "AUAL", "AUAR", "BDUL", "BDUR",
    "DVA", "DVB", "DVC", "ALA", "PVT",
    "SDQL", "SDQR", "LUAL", "LUAR",
}
_TAIL_NEURONS = {
    "PVCL", "PVCR", "PVDL", "PVDR", "PVM", "PVNL", "PVNR",
    "PVPL", "PVPR", "PVQL", "PVQR", "PVR", "PVWL", "PVWR",
    "PLML", "PLMR", "PLNL", "PLNR", "PQR",
    "PHAL", "PHAR", "PHBL", "PHBR", "PHCL", "PHCR",
    "PDEL", "PDER", "ALNL", "ALNR", "ALML", "ALMR",
    "HSNL", "HSNR", "PDA", "PDB",
}


def anatomical_layout(network: CElegansNetwork,
                      padding: float = 0.06) -> Dict[str, Tuple[float, float]]:
    """
    Arrange neurons in a worm-shaped layout: head left, tail right.

    Regions (left to right):
      0.06-0.25  Head sensory neurons
      0.22-0.55  Nerve ring / command interneurons
      0.50-0.95  Ventral cord motor neurons + tail
    Dorsal neurons are placed above centre, ventral below.
    """
    positions: Dict[str, Tuple[float, float]] = {}
    all_ids = sorted(network.neurons.keys())

    head_ids = [n for n in all_ids if n in _HEAD_NEURONS]
    ring_ids = [n for n in all_ids if n in _NERVE_RING]
    tail_ids = [n for n in all_ids if n in _TAIL_NEURONS]
    placed = set(head_ids + ring_ids + tail_ids)

    motor_ids = []
    other_ids = []
    for nid in all_ids:
        if nid in placed:
            continue
        cls = _neuron_class(network, nid)
        if cls == "motor":
            motor_ids.append(nid)
        else:
            other_ids.append(nid)

    def _place_group(ids, x_min, x_max, y_center, y_spread):
        n = len(ids)
        if n == 0:
            return
        cols = max(1, int(math.sqrt(n * 2)))
        rows = max(1, math.ceil(n / cols))
        for i, nid in enumerate(ids):
            c = i % cols
            r = i // cols
            x = x_min + (c + 0.5) / cols * (x_max - x_min)
            y = y_center - y_spread / 2 + (r + 0.5) / rows * y_spread
            is_dorsal = "D" in nid or "L" in nid[-1:]
            if is_dorsal:
                y = y_center - abs(y - y_center) - 0.02
            positions[nid] = (x, y)

    _place_group(head_ids,  0.04, 0.24, 0.45, 0.55)
    _place_group(ring_ids,  0.22, 0.54, 0.50, 0.70)
    _place_group(motor_ids, 0.50, 0.92, 0.50, 0.60)
    _place_group(tail_ids,  0.82, 0.97, 0.50, 0.50)
    _place_group(other_ids, 0.35, 0.70, 0.50, 0.40)

    for nid in positions:
        x, y = positions[nid]
        positions[nid] = (
            max(padding, min(1 - padding, x)),
            max(padding, min(1 - padding, y)),
        )

    return positions


def compute_layout(network: CElegansNetwork,
                   mode: str = "grid",
                   **kwargs) -> Dict[str, Tuple[float, float]]:
    """Dispatch to the requested layout strategy."""
    if mode == "graph":
        return graph_layout(network, **kwargs)
    if mode == "anatomical":
        return anatomical_layout(network, **kwargs)
    return grid_layout(network, **kwargs)
