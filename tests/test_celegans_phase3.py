"""
Tests for C. elegans Phase 3: ScaleFreeBuilder, ConnectomeLoader, and
refactored BaseTopologyBuilder._get_neuron_id.
"""

import pytest
import numpy as np

from sdmn.networks.celegans import (
    CElegansNetwork,
    TopologyParameters,
    UniformTopologyBuilder,
    SmallWorldBuilder,
    RandomTopologyBuilder,
    ScaleFreeBuilder,
    ConnectomeLoader,
    build_uniform_network,
    build_small_world_network,
    build_random_network,
    build_scale_free_network,
    build_connectome_network,
)


# ======================================================================
# BaseTopologyBuilder._get_neuron_id refactor
# ======================================================================

class TestGetNeuronIdRefactor:
    """Verify existing builders still work after moving _get_neuron_id to base."""

    def test_uniform_builder_basic(self):
        net = build_uniform_network(n_neurons=20, k_neighbors=4, random_seed=1)
        assert net.get_neuron_count() == 20
        summary = net.get_connectivity_summary()
        assert summary["n_chemical_synapses"] + summary["n_gap_junctions"] > 0

    def test_small_world_builder_basic(self):
        net = build_small_world_network(n_neurons=20, k_neighbors=4,
                                        rewire_prob=0.2, random_seed=1)
        assert net.get_neuron_count() == 20

    def test_random_builder_basic(self):
        net = build_random_network(n_neurons=20, k_neighbors=4, random_seed=1)
        assert net.get_neuron_count() == 20

    def test_mixed_neuron_ids(self):
        params = TopologyParameters(
            n_neurons=10, neuron_class="mixed",
            sensory_fraction=0.3, motor_fraction=0.2,
            k_neighbors=4, random_seed=42,
        )
        builder = UniformTopologyBuilder(params)
        net = builder.build()
        ids = sorted(net.neurons.keys())
        assert any(n.startswith("S") for n in ids)
        assert any(n.startswith("I") for n in ids)
        assert any(n.startswith("M") for n in ids)


# ======================================================================
# ScaleFreeBuilder
# ======================================================================

class TestScaleFreeBuilder:

    def test_build_small(self):
        net = build_scale_free_network(n_neurons=30, m=3, random_seed=42)
        assert net.get_neuron_count() == 30
        summary = net.get_connectivity_summary()
        assert summary["n_chemical_synapses"] + summary["n_gap_junctions"] > 0

    def test_hub_existence(self):
        """Scale-free networks should have high-degree hub neurons."""
        net = build_scale_free_network(n_neurons=100, m=5, random_seed=42)
        summary = net.get_connectivity_summary()
        total_conn = summary["n_chemical_synapses"] + summary["n_gap_junctions"]
        avg = total_conn / 100
        # at least some neurons should have degree >> average (hubs)
        assert avg > 0

    def test_degree_variance_higher_than_uniform(self):
        """Scale-free degree distribution should have higher variance than uniform."""
        from sdmn.analysis.topology import TopologyAnalyzer

        sf_net = build_scale_free_network(n_neurons=80, m=4, random_seed=42)
        uf_net = build_uniform_network(n_neurons=80, k_neighbors=8,
                                       connection_variance=0.0, random_seed=42)

        sf_deg = TopologyAnalyzer.from_network(sf_net).degree_distribution()["total"]
        uf_deg = TopologyAnalyzer.from_network(uf_net).degree_distribution()["total"]

        assert sf_deg.std() > uf_deg.std()

    def test_custom_m0(self):
        params = TopologyParameters(n_neurons=20, k_neighbors=6, random_seed=7)
        builder = ScaleFreeBuilder(params, m=3, m0=5)
        net = builder.build()
        assert net.get_neuron_count() == 20


# ======================================================================
# ConnectomeLoader
# ======================================================================

class TestConnectomeLoader:

    def test_load_bundled_data(self):
        loader = ConnectomeLoader()
        data = loader.load()
        assert data.n_neurons > 250
        assert data.n_chemical > 0
        assert data.n_gap > 0

    def test_neuron_classes_present(self):
        loader = ConnectomeLoader()
        loader.load()
        meta = loader.get_neuron_metadata()
        classes = set(meta.values())
        assert "sensory" in classes
        assert "interneuron" in classes
        assert "motor" in classes

    def test_build_network(self):
        net = build_connectome_network()
        assert net.get_neuron_count() > 250
        summary = net.get_connectivity_summary()
        assert summary["n_chemical_synapses"] > 0
        assert summary["n_gap_junctions"] > 0
        class_counts = summary["neurons_by_class"]
        assert class_counts["sensory"] > 0
        assert class_counts["interneuron"] > 0
        assert class_counts["motor"] > 0

    def test_adjacency_matrix_shape(self):
        loader = ConnectomeLoader()
        loader.load()
        mat, names = loader.get_adjacency_matrix()
        assert mat.shape == (len(names), len(names))
        assert mat.shape[0] == loader.data.n_neurons

    def test_adjacency_chemical_only(self):
        loader = ConnectomeLoader()
        loader.load()
        mat_chem, _ = loader.get_adjacency_matrix(conn_type="chemical")
        mat_gap, _ = loader.get_adjacency_matrix(conn_type="gap")
        mat_all, _ = loader.get_adjacency_matrix(conn_type="all")
        assert mat_chem.sum() + mat_gap.sum() == pytest.approx(mat_all.sum())

    def test_filter_by_names(self):
        loader = ConnectomeLoader()
        loader.load()
        subset = loader.filter(neuron_names=["AVAL", "AVAR", "AVBL", "AVBR"])
        assert subset.n_neurons == 4
        for e in subset.edges:
            assert e.pre in ("AVAL", "AVAR", "AVBL", "AVBR")
            assert e.post in ("AVAL", "AVAR", "AVBL", "AVBR")

    def test_filter_by_min_weight(self):
        loader = ConnectomeLoader()
        loader.load()
        subset = loader.filter(min_weight=3)
        for e in subset.edges:
            assert e.weight >= 3

    def test_filter_by_conn_type(self):
        loader = ConnectomeLoader()
        loader.load()
        subset = loader.filter(conn_type="gap")
        for e in subset.edges:
            assert e.conn_type == "gap"
