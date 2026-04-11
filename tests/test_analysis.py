"""
Tests for the analysis module: TopologyAnalyzer, DynamicsAnalyzer,
BehaviorDetector, and NetworkVisualizer.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from sdmn.networks.celegans import (
    build_small_world_network,
    build_uniform_network,
    build_scale_free_network,
    CElegansNetwork,
)
from sdmn.analysis.topology import TopologyAnalyzer
from sdmn.analysis.dynamics import DynamicsAnalyzer
from sdmn.analysis.behavior import BehaviorDetector
from sdmn.analysis.visualization import NetworkVisualizer


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_sw_network():
    """20-neuron small-world network (built once per module)."""
    return build_small_world_network(n_neurons=20, k_neighbors=4,
                                     rewire_prob=0.3, random_seed=42)


@pytest.fixture(scope="module")
def small_uniform_network():
    return build_uniform_network(n_neurons=20, k_neighbors=4,
                                 connection_variance=0.0, random_seed=42)


@pytest.fixture(scope="module")
def simulated_network():
    """20-neuron network that has been simulated for 50 ms."""
    net = build_small_world_network(n_neurons=20, k_neighbors=4,
                                    rewire_prob=0.3, random_seed=42)
    net.set_external_current("N0", 50.0)
    net.set_external_current("N1", 30.0)
    net.simulate(duration=50, progress=False)
    return net


# ------------------------------------------------------------------
# TopologyAnalyzer
# ------------------------------------------------------------------

class TestTopologyAnalyzer:

    def test_from_network(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        assert analyzer.graph.number_of_nodes() == 20
        assert analyzer.graph.number_of_edges() > 0

    def test_degree_distribution(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        deg = analyzer.degree_distribution()
        assert len(deg["total"]) == 20
        assert deg["total"].sum() > 0

    def test_hub_neurons(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        hubs = analyzer.hub_neurons(top_n=3)
        assert len(hubs) == 3
        assert all(isinstance(h[1], int) for h in hubs)

    def test_clustering_coefficient(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        cc = analyzer.clustering_coefficient()
        assert 0.0 <= cc <= 1.0

    def test_average_path_length(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        apl = analyzer.average_path_length()
        assert apl > 0

    def test_diameter(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        d = analyzer.diameter()
        assert isinstance(d, int)
        assert d >= 1

    def test_small_world_coefficient(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        sigma = analyzer.small_world_coefficient(n_random=2)
        assert isinstance(sigma, float)

    def test_modularity(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        q = analyzer.modularity()
        assert -1.0 <= q <= 1.0

    def test_detect_communities(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        comms = analyzer.detect_communities()
        assert len(comms) == 20

    def test_rich_club(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        rc = analyzer.rich_club_coefficient()
        assert isinstance(rc, dict)

    def test_full_report(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        report = analyzer.full_report()
        assert "n_nodes" in report
        assert "clustering_coefficient" in report
        assert "average_path_length" in report
        assert "modularity" in report

    def test_degree_distribution_scale_free_vs_uniform(self):
        sf = build_scale_free_network(n_neurons=60, m=3, random_seed=42)
        uf = build_uniform_network(n_neurons=60, k_neighbors=6,
                                   connection_variance=0.0, random_seed=42)
        sf_std = TopologyAnalyzer.from_network(sf).degree_distribution()["total"].std()
        uf_std = TopologyAnalyzer.from_network(uf).degree_distribution()["total"].std()
        assert sf_std > uf_std


# ------------------------------------------------------------------
# DynamicsAnalyzer
# ------------------------------------------------------------------

class TestDynamicsAnalyzer:

    def test_from_simulation(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        assert dyn.n_neurons == 20
        assert dyn.n_timepoints > 0

    def test_correlation_matrix_shape(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        corr = dyn.pairwise_correlation_matrix()
        assert corr.shape == (20, 20)

    def test_mean_synchronization(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        sync = dyn.mean_synchronization()
        assert 0.0 <= sync <= 1.0

    def test_propagation_delay(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        delay = dyn.propagation_delay("N0", "N5")
        assert isinstance(delay, float)

    def test_power_spectrum(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        freqs, psd = dyn.power_spectrum("N0")
        assert len(freqs) == len(psd)
        assert len(freqs) > 0

    def test_dominant_frequency(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        freq = dyn.dominant_frequency("N0")
        assert freq >= 0

    def test_population_spectrum(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        freqs, psd = dyn.population_spectrum()
        assert len(freqs) == len(psd)

    def test_pca_trajectories(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        traj, ev = dyn.pca_trajectories(n_components=3)
        assert traj.shape[0] == 3
        assert len(ev) == 3

    def test_transfer_entropy(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        te = dyn.transfer_entropy("N0", "N1", lag=1)
        assert te >= 0.0

    def test_mutual_information(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        mi = dyn.mutual_information("N0", "N1")
        assert mi >= 0.0

    def test_mean_voltage_trace(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        trace = dyn.mean_voltage_trace()
        assert len(trace) == dyn.n_timepoints

    def test_active_neuron_count(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        counts = dyn.active_neuron_count(threshold=-50.0)
        assert len(counts) == dyn.n_timepoints


# ------------------------------------------------------------------
# BehaviorDetector
# ------------------------------------------------------------------

class TestBehaviorDetector:

    def test_from_simulation(self, simulated_network):
        det = BehaviorDetector.from_simulation(simulated_network)
        assert det.n_timepoints > 0

    @property
    def n_timepoints(self):
        """Helper — BehaviorDetector stores times directly."""
        return 0  # not used, just for IDE

    def test_classify_activity_state(self, simulated_network):
        det = BehaviorDetector.from_simulation(simulated_network)
        states = det.classify_activity_state(window_ms=10.0)
        assert len(states) > 0
        for s in states:
            assert s.state in ("quiescent", "active", "bursting")

    def test_behavioral_report(self, simulated_network):
        det = BehaviorDetector.from_simulation(simulated_network)
        report = det.behavioral_report()
        assert "n_forward_bouts" in report
        assert "n_reversal_bouts" in report
        assert "activity_state_counts" in report

    def test_stimulus_response_latency(self):
        """Build a tiny network with named motor neurons and apply stimulus."""
        net = CElegansNetwork()
        net.add_sensory_neuron("AWCL")
        net.add_interneuron("AIAL")
        net.add_motor_neuron("DA01")
        net.add_graded_synapse("AWCL", "AIAL", weight=3.0)
        net.add_graded_synapse("AIAL", "DA01", weight=3.0)
        net.set_external_current("AWCL", 80.0)
        net.simulate(duration=50, progress=False)

        det = BehaviorDetector.from_simulation(net)
        latency = det.stimulus_response_latency(0.0, ["DA01"], threshold=-60.0)
        assert isinstance(latency, float)


# ------------------------------------------------------------------
# NetworkVisualizer (no-crash tests — matplotlib Agg backend)
# ------------------------------------------------------------------

class TestNetworkVisualizer:

    def test_plot_network_graph(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        fig = NetworkVisualizer.plot_network_graph(analyzer)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_degree_distribution(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        fig = NetworkVisualizer.plot_degree_distribution(analyzer)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_voltage_raster(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        fig = NetworkVisualizer.plot_voltage_raster(dyn)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_voltage_traces(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        fig = NetworkVisualizer.plot_voltage_traces(dyn, neuron_ids=["N0", "N1"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_matrix(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        fig = NetworkVisualizer.plot_correlation_matrix(dyn)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pca_trajectory(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        fig = NetworkVisualizer.plot_pca_trajectory(dyn)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_power_spectrum(self, simulated_network):
        dyn = DynamicsAnalyzer.from_simulation(simulated_network)
        fig = NetworkVisualizer.plot_power_spectrum(dyn, neuron_ids=["N0"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_topology_comparison(self, small_sw_network, small_uniform_network):
        sw_a = TopologyAnalyzer.from_network(small_sw_network)
        uf_a = TopologyAnalyzer.from_network(small_uniform_network)
        reports = {
            "small-world": sw_a.full_report(),
            "uniform": uf_a.full_report(),
        }
        fig = NetworkVisualizer.plot_topology_comparison(reports)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_community_structure(self, small_sw_network):
        analyzer = TopologyAnalyzer.from_network(small_sw_network)
        fig = NetworkVisualizer.plot_community_structure(analyzer)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
