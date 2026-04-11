"""
Visualization utilities for C. elegans network analysis.

All functions return ``matplotlib.figure.Figure`` objects so callers can
choose to show, save, or further customise them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sdmn.analysis.dynamics import DynamicsAnalyzer
from sdmn.analysis.topology import TopologyAnalyzer
from sdmn.networks.celegans.network_manager import CElegansNetwork


_CLASS_COLORS = {
    "sensory": "#2196F3",
    "interneuron": "#4CAF50",
    "motor": "#FF9800",
    "unknown": "#9E9E9E",
}


class NetworkVisualizer:
    """Static collection of plotting helpers."""

    # ------------------------------------------------------------------
    # Network graph
    # ------------------------------------------------------------------

    @staticmethod
    def plot_network_graph(
        analyzer: TopologyAnalyzer,
        layout: str = "spring",
        color_by: str = "class",
        node_size_factor: float = 15.0,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        Draw the network as a node-link diagram.

        Args:
            analyzer:          ``TopologyAnalyzer`` instance.
            layout:            ``"spring"`` | ``"circular"`` | ``"kamada_kawai"``.
            color_by:          ``"class"`` or ``"degree"``.
            node_size_factor:  Scale node size by degree * this factor.
        """
        g = analyzer.graph.to_undirected()

        if layout == "spring":
            pos = nx.spring_layout(g, seed=42, k=1.5 / (g.number_of_nodes() ** 0.5))
        elif layout == "circular":
            pos = nx.circular_layout(g)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(g)
        else:
            pos = nx.spring_layout(g, seed=42)

        fig, ax = plt.subplots(figsize=figsize)

        degrees = dict(g.degree())
        sizes = [max(10, degrees.get(n, 1) * node_size_factor) for n in g.nodes()]

        if color_by == "class":
            colors = [_CLASS_COLORS.get(analyzer.neuron_classes.get(n, "unknown"), "#9E9E9E")
                      for n in g.nodes()]
        else:
            deg_arr = np.array([degrees.get(n, 0) for n in g.nodes()])
            colors = deg_arr

        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.15, width=0.3)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes,
                               node_color=colors, alpha=0.8)
        ax.set_title("Network Graph")
        ax.axis("off")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Degree distribution
    # ------------------------------------------------------------------

    @staticmethod
    def plot_degree_distribution(
        analyzer: TopologyAnalyzer,
        figsize: Tuple[int, int] = (8, 5),
    ) -> plt.Figure:
        deg = analyzer.degree_distribution()
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(deg["total"], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
        ax.set_title("Degree Distribution")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Voltage raster
    # ------------------------------------------------------------------

    @staticmethod
    def plot_voltage_raster(
        dyn: DynamicsAnalyzer,
        neuron_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        Heatmap of voltage traces over time (neurons on y-axis).
        """
        ids = neuron_ids or dyn.neuron_ids
        indices = [dyn._id_index[n] for n in ids if n in dyn._id_index]
        times = dyn.times
        data = dyn.voltages[indices]

        if time_range is not None:
            t_mask = (times >= time_range[0]) & (times <= time_range[1])
            times = times[t_mask]
            data = data[:, t_mask]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[times[0], times[-1], 0, len(indices)],
                       cmap="RdBu_r")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron Index")
        ax.set_title("Voltage Raster")
        fig.colorbar(im, ax=ax, label="Voltage (mV)")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Voltage traces
    # ------------------------------------------------------------------

    @staticmethod
    def plot_voltage_traces(
        dyn: DynamicsAnalyzer,
        neuron_ids: List[str],
        time_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        times = dyn.times
        fig, ax = plt.subplots(figsize=figsize)

        for nid in neuron_ids:
            if nid not in dyn._id_index:
                continue
            idx = dyn._id_index[nid]
            v = dyn.voltages[idx]
            if time_range is not None:
                t_mask = (times >= time_range[0]) & (times <= time_range[1])
                ax.plot(times[t_mask], v[t_mask], label=nid, linewidth=0.8)
            else:
                ax.plot(times, v, label=nid, linewidth=0.8)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("Voltage Traces")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def plot_correlation_matrix(
        dyn: DynamicsAnalyzer,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        corr = dyn.pairwise_correlation_matrix()
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title("Pairwise Correlation Matrix")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Neuron")
        fig.colorbar(im, ax=ax, label="Pearson r")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # PCA trajectory
    # ------------------------------------------------------------------

    @staticmethod
    def plot_pca_trajectory(
        dyn: DynamicsAnalyzer,
        components: Tuple[int, int] = (0, 1),
        figsize: Tuple[int, int] = (8, 7),
    ) -> plt.Figure:
        traj, ev = dyn.pca_trajectories(max(components) + 1)
        c0, c1 = components

        fig, ax = plt.subplots(figsize=figsize)
        colors = dyn.times[:traj.shape[1]]
        sc = ax.scatter(traj[c0], traj[c1], c=colors, s=2, cmap="viridis")
        ax.set_xlabel(f"PC{c0 + 1} ({ev[c0]*100:.1f}%)")
        ax.set_ylabel(f"PC{c1 + 1} ({ev[c1]*100:.1f}%)")
        ax.set_title("PCA Trajectory")
        fig.colorbar(sc, ax=ax, label="Time (ms)")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Power spectrum
    # ------------------------------------------------------------------

    @staticmethod
    def plot_power_spectrum(
        dyn: DynamicsAnalyzer,
        neuron_ids: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 5),
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)

        if neuron_ids:
            for nid in neuron_ids:
                freqs, psd = dyn.power_spectrum(nid)
                ax.semilogy(freqs, psd, label=nid, linewidth=0.8)
            ax.legend(fontsize=7, ncol=2)
        else:
            freqs, psd = dyn.population_spectrum()
            ax.semilogy(freqs, psd, linewidth=1.0)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (mV^2/Hz)")
        ax.set_title("Power Spectrum")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Topology comparison
    # ------------------------------------------------------------------

    @staticmethod
    def plot_topology_comparison(
        reports: Dict[str, Dict[str, Any]],
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Side-by-side bar chart comparing topology metrics.

        Args:
            reports: ``{label: full_report_dict}`` from ``TopologyAnalyzer``.
        """
        labels = list(reports.keys())
        metrics = ["mean_degree", "clustering_coefficient",
                    "average_path_length", "modularity"]

        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        for ax, metric in zip(axes, metrics):
            vals = [reports[l].get(metric, 0) for l in labels]
            ax.bar(labels, vals, edgecolor="black", alpha=0.8)
            ax.set_title(metric.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=30)

        fig.suptitle("Topology Comparison", fontsize=14)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Community structure
    # ------------------------------------------------------------------

    @staticmethod
    def plot_community_structure(
        analyzer: TopologyAnalyzer,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        g = analyzer.graph.to_undirected()
        communities = analyzer.detect_communities()
        pos = nx.spring_layout(g, seed=42)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.colormaps.get_cmap("tab20")
        n_communities = max(communities.values()) + 1 if communities else 1
        colors = [cmap(communities.get(n, 0) / max(n_communities, 1))
                  for n in g.nodes()]

        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.1, width=0.3)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=30,
                               node_color=colors, alpha=0.8)
        ax.set_title(f"Community Structure ({n_communities} communities)")
        ax.axis("off")
        fig.tight_layout()
        return fig
