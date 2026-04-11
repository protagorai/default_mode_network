"""
Topology analysis tools for C. elegans neural networks.

Computes graph-theoretic metrics (degree distribution, clustering,
path length, small-world coefficient, modularity, rich-club) using
``networkx`` on the connectivity extracted from a ``CElegansNetwork``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from sdmn.networks.celegans.network_manager import CElegansNetwork


class TopologyAnalyzer:
    """
    Compute structural metrics on a C. elegans network.

    Usage::

        analyzer = TopologyAnalyzer.from_network(network)
        report = analyzer.full_report()
    """

    def __init__(self, graph: nx.DiGraph, neuron_classes: Dict[str, str]):
        """
        Args:
            graph:          Directed graph with neurons as nodes.
            neuron_classes: ``{neuron_id: class}`` mapping.
        """
        self.graph = graph
        self.neuron_classes = neuron_classes
        self._undirected: Optional[nx.Graph] = None

    @classmethod
    def from_network(cls, network: CElegansNetwork) -> "TopologyAnalyzer":
        """Build analyser from a live ``CElegansNetwork``."""
        g = nx.DiGraph()

        for nid in network.neurons:
            g.add_node(nid)

        for syn in network.chemical_synapses.values():
            pre = syn.presynaptic_neuron_id
            post = syn.postsynaptic_neuron_id
            if g.has_edge(pre, post):
                g[pre][post]["weight"] += 1
            else:
                g.add_edge(pre, post, weight=1, edge_type="chemical")

        for gap in network.gap_junctions.values():
            a = gap.presynaptic_neuron_id
            b = gap.postsynaptic_neuron_id
            for u, v in [(a, b), (b, a)]:
                if g.has_edge(u, v):
                    g[u][v]["weight"] += 1
                else:
                    g.add_edge(u, v, weight=1, edge_type="gap")

        neuron_classes: Dict[str, str] = {}
        for nid, neuron in network.neurons.items():
            if hasattr(neuron, "neuron_class"):
                neuron_classes[nid] = neuron.neuron_class.value
            else:
                neuron_classes[nid] = "unknown"

        return cls(g, neuron_classes)

    # ------------------------------------------------------------------
    # Degree metrics
    # ------------------------------------------------------------------

    def degree_distribution(self) -> Dict[str, np.ndarray]:
        """Total, in, and out degree arrays."""
        in_deg = np.array([d for _, d in self.graph.in_degree()])
        out_deg = np.array([d for _, d in self.graph.out_degree()])
        return {"total": in_deg + out_deg, "in": in_deg, "out": out_deg}

    def in_degree_distribution(self) -> np.ndarray:
        return np.array([d for _, d in self.graph.in_degree()])

    def out_degree_distribution(self) -> np.ndarray:
        return np.array([d for _, d in self.graph.out_degree()])

    def hub_neurons(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Return the *top_n* highest-degree neurons."""
        degs = dict(self.graph.degree())
        ranked = sorted(degs.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def clustering_coefficient(self) -> float:
        """Average clustering coefficient (undirected view)."""
        return nx.average_clustering(self._get_undirected())

    def clustering_distribution(self) -> Dict[str, float]:
        """Per-node clustering coefficient."""
        return nx.clustering(self._get_undirected())

    # ------------------------------------------------------------------
    # Path length
    # ------------------------------------------------------------------

    def average_path_length(self) -> float:
        """Average shortest-path length over the largest connected component."""
        ug = self._get_undirected()
        if not nx.is_connected(ug):
            largest_cc = max(nx.connected_components(ug), key=len)
            ug = ug.subgraph(largest_cc).copy()
        return nx.average_shortest_path_length(ug)

    def diameter(self) -> int:
        """Diameter of the largest connected component."""
        ug = self._get_undirected()
        if not nx.is_connected(ug):
            largest_cc = max(nx.connected_components(ug), key=len)
            ug = ug.subgraph(largest_cc).copy()
        return nx.diameter(ug)

    # ------------------------------------------------------------------
    # Small-world coefficient
    # ------------------------------------------------------------------

    def small_world_coefficient(self, n_random: int = 5) -> float:
        """
        Compute sigma = (C/C_rand) / (L/L_rand).

        Values > 1 indicate small-world topology.
        """
        ug = self._get_undirected()
        if not nx.is_connected(ug):
            largest_cc = max(nx.connected_components(ug), key=len)
            ug = ug.subgraph(largest_cc).copy()

        C = nx.average_clustering(ug)
        L = nx.average_shortest_path_length(ug)

        n = ug.number_of_nodes()
        m = ug.number_of_edges()

        C_rand_list, L_rand_list = [], []
        for _ in range(n_random):
            rg = nx.gnm_random_graph(n, m)
            if not nx.is_connected(rg):
                largest = max(nx.connected_components(rg), key=len)
                rg = rg.subgraph(largest).copy()
            C_rand_list.append(nx.average_clustering(rg))
            L_rand_list.append(nx.average_shortest_path_length(rg))

        C_rand = float(np.mean(C_rand_list))
        L_rand = float(np.mean(L_rand_list))

        if C_rand == 0 or L_rand == 0:
            return 0.0

        sigma = (C / C_rand) / (L / L_rand)
        return sigma

    # ------------------------------------------------------------------
    # Modularity / community detection
    # ------------------------------------------------------------------

    def detect_communities(self) -> Dict[str, int]:
        """Greedy modularity community detection; returns ``{node: community_id}``."""
        ug = self._get_undirected()
        communities = nx.community.greedy_modularity_communities(ug)
        mapping: Dict[str, int] = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                mapping[node] = cid
        return mapping

    def modularity(self) -> float:
        """Newman modularity score for detected communities."""
        ug = self._get_undirected()
        communities = nx.community.greedy_modularity_communities(ug)
        return nx.community.modularity(ug, communities)

    # ------------------------------------------------------------------
    # Rich-club
    # ------------------------------------------------------------------

    def rich_club_coefficient(self, k: Optional[int] = None) -> Dict[int, float]:
        """
        Rich-club coefficient phi(k) for each degree threshold.

        Args:
            k: If given, return only for that threshold.
        """
        ug = self._get_undirected()
        rc = nx.rich_club_coefficient(ug, normalized=False)
        if k is not None:
            return {k: rc.get(k, 0.0)}
        return dict(rc)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def full_report(self) -> Dict[str, Any]:
        """Compute and return all topology metrics in one call."""
        deg = self.degree_distribution()
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "mean_degree": float(deg["total"].mean()),
            "std_degree": float(deg["total"].std()),
            "max_degree": int(deg["total"].max()),
            "clustering_coefficient": self.clustering_coefficient(),
            "average_path_length": self.average_path_length(),
            "diameter": self.diameter(),
            "small_world_sigma": self.small_world_coefficient(),
            "modularity": self.modularity(),
            "hub_neurons": self.hub_neurons(10),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_undirected(self) -> nx.Graph:
        if self._undirected is None:
            self._undirected = self.graph.to_undirected()
        return self._undirected
