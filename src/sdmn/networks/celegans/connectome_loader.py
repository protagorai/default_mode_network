"""
Loader for real C. elegans connectome data.

Parses curated CSV files (neuron metadata + synaptic connections) and
builds a fully wired ``CElegansNetwork`` with biologically accurate
neuron classes, chemical synapses, and gap junctions.

Data sources:
    Varshney et al. (2011) PLOS Comp Biol 7(2):e1001066
    Cook et al. (2019) Nature 571:63-71
    OpenWorm project  https://github.com/openworm/c302
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from sdmn.networks.celegans.network_manager import CElegansNetwork
from sdmn.synapses import SynapseType

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DEFAULT_DATA_DIR = _PACKAGE_ROOT / "data" / "connectome"


@dataclass
class NeuronRecord:
    """Single neuron entry from the metadata CSV."""
    name: str
    neuron_class: str  # sensory | interneuron | motor
    organism_type: str = "hermaphrodite"


@dataclass
class EdgeRecord:
    """Single connection entry from the connectome CSV."""
    pre: str
    post: str
    conn_type: str  # chemical | gap
    weight: int = 1
    neurotransmitter: str = ""


@dataclass
class ConnectomeData:
    """Parsed connectome dataset ready for analysis or network construction."""
    neurons: Dict[str, NeuronRecord] = field(default_factory=dict)
    edges: List[EdgeRecord] = field(default_factory=list)

    @property
    def neuron_names(self) -> List[str]:
        return sorted(self.neurons.keys())

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_chemical(self) -> int:
        return sum(1 for e in self.edges if e.conn_type == "chemical")

    @property
    def n_gap(self) -> int:
        return sum(1 for e in self.edges if e.conn_type == "gap")


class ConnectomeLoader:
    """
    Load and manipulate C. elegans connectome data.

    Example::

        loader = ConnectomeLoader()
        loader.load()                        # uses bundled CSVs
        network = loader.build_network()     # -> CElegansNetwork
        adj = loader.get_adjacency_matrix()  # -> numpy array
    """

    def __init__(self):
        self.data: Optional[ConnectomeData] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self,
             connectome_path: Optional[str] = None,
             neurons_path: Optional[str] = None) -> ConnectomeData:
        """
        Parse connectome and neuron-metadata CSVs.

        Args:
            connectome_path: Path to edges CSV (default: bundled data).
            neurons_path:    Path to neuron metadata CSV (default: bundled data).

        Returns:
            Parsed ``ConnectomeData``.
        """
        if connectome_path is None:
            connectome_path = str(_DEFAULT_DATA_DIR / "celegans_connectome.csv")
        if neurons_path is None:
            neurons_path = str(_DEFAULT_DATA_DIR / "celegans_neurons.csv")

        neurons = self._parse_neurons(neurons_path)
        edges = self._parse_edges(connectome_path)

        self.data = ConnectomeData(neurons=neurons, edges=edges)
        return self.data

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def build_network(self,
                      weight_scale: float = 1.0,
                      gap_conductance_scale: float = 0.5) -> CElegansNetwork:
        """
        Construct a ``CElegansNetwork`` from loaded connectome data.

        Args:
            weight_scale:          Multiplier applied to CSV weights for
                                   chemical synapse conductance (nS).
            gap_conductance_scale: Multiplier applied to CSV weights for
                                   gap junction conductance (nS).

        Returns:
            Fully wired ``CElegansNetwork``.
        """
        if self.data is None:
            raise RuntimeError("Call load() before build_network()")

        network = CElegansNetwork()

        for name, rec in sorted(self.data.neurons.items()):
            if rec.neuron_class == "sensory":
                network.add_sensory_neuron(name)
            elif rec.neuron_class == "motor":
                network.add_motor_neuron(name)
            else:
                network.add_interneuron(name)

        added_gaps: Set[str] = set()

        for edge in self.data.edges:
            if edge.pre not in self.data.neurons or edge.post not in self.data.neurons:
                continue

            if edge.conn_type == "gap":
                pair_key = "_".join(sorted([edge.pre, edge.post]))
                if pair_key in added_gaps:
                    continue
                added_gaps.add(pair_key)

                conductance = edge.weight * gap_conductance_scale
                network.add_gap_junction(
                    edge.pre, edge.post,
                    conductance=max(0.1, conductance),
                )

            elif edge.conn_type == "chemical":
                nt = edge.neurotransmitter.lower()
                if nt == "gaba":
                    syn_type = SynapseType.INHIBITORY
                else:
                    syn_type = SynapseType.EXCITATORY

                weight = edge.weight * weight_scale
                network.add_graded_synapse(
                    edge.pre, edge.post,
                    weight=max(0.1, weight),
                    synapse_type=syn_type,
                )

        return network

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_neuron_metadata(self) -> Dict[str, str]:
        """Return ``{name: class}`` mapping for every neuron."""
        if self.data is None:
            raise RuntimeError("Call load() first")
        return {n: r.neuron_class for n, r in self.data.neurons.items()}

    def get_adjacency_matrix(self, conn_type: str = "all"
                             ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a dense adjacency matrix.

        Args:
            conn_type: ``"chemical"``, ``"gap"``, or ``"all"``.

        Returns:
            ``(matrix, neuron_names)`` where *matrix[i][j]* is the total
            weight from neuron *i* to neuron *j*.
        """
        if self.data is None:
            raise RuntimeError("Call load() first")

        names = self.data.neuron_names
        idx = {n: i for i, n in enumerate(names)}
        n = len(names)
        mat = np.zeros((n, n), dtype=float)

        for edge in self.data.edges:
            if edge.pre not in idx or edge.post not in idx:
                continue
            if conn_type != "all" and edge.conn_type != conn_type:
                continue
            mat[idx[edge.pre], idx[edge.post]] += edge.weight

        return mat, names

    def filter(self,
               neuron_names: Optional[List[str]] = None,
               min_weight: int = 1,
               conn_type: Optional[str] = None) -> ConnectomeData:
        """
        Return a filtered copy of the loaded connectome.

        Args:
            neuron_names: Keep only these neurons (and edges between them).
            min_weight:   Drop edges below this weight.
            conn_type:    Keep only ``"chemical"`` or ``"gap"`` edges.

        Returns:
            New ``ConnectomeData`` with the filters applied.
        """
        if self.data is None:
            raise RuntimeError("Call load() first")

        keep_names: Optional[Set[str]] = None
        if neuron_names is not None:
            keep_names = set(neuron_names)

        neurons = {}
        for name, rec in self.data.neurons.items():
            if keep_names is not None and name not in keep_names:
                continue
            neurons[name] = rec

        edges = []
        for e in self.data.edges:
            if keep_names is not None and (e.pre not in keep_names or e.post not in keep_names):
                continue
            if e.weight < min_weight:
                continue
            if conn_type is not None and e.conn_type != conn_type:
                continue
            edges.append(e)

        return ConnectomeData(neurons=neurons, edges=edges)

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_neurons(path: str) -> Dict[str, NeuronRecord]:
        neurons: Dict[str, NeuronRecord] = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                neurons[row["name"]] = NeuronRecord(
                    name=row["name"],
                    neuron_class=row["class"],
                    organism_type=row.get("type", "hermaphrodite"),
                )
        return neurons

    @staticmethod
    def _parse_edges(path: str) -> List[EdgeRecord]:
        edges: List[EdgeRecord] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                edges.append(EdgeRecord(
                    pre=row["pre"],
                    post=row["post"],
                    conn_type=row["type"],
                    weight=int(row["weight"]),
                    neurotransmitter=row.get("neurotransmitter", ""),
                ))
        return edges


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def build_connectome_network(connectome_path: Optional[str] = None,
                             neurons_path: Optional[str] = None,
                             weight_scale: float = 1.0,
                             gap_conductance_scale: float = 0.5,
                             ) -> CElegansNetwork:
    """
    One-liner to load the real connectome and return a ready-to-simulate network.

    Args:
        connectome_path:       Path to edges CSV (default: bundled data).
        neurons_path:          Path to neuron metadata CSV (default: bundled data).
        weight_scale:          Multiplier for chemical synapse weights.
        gap_conductance_scale: Multiplier for gap junction conductances.

    Returns:
        Fully wired ``CElegansNetwork``.

    Example::

        network = build_connectome_network()
        network.simulate(duration=500)
    """
    loader = ConnectomeLoader()
    loader.load(connectome_path, neurons_path)
    return loader.build_network(
        weight_scale=weight_scale,
        gap_conductance_scale=gap_conductance_scale,
    )
