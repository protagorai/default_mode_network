"""Tests for network building and topology generation."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from sdmn.networks import (
    NetworkBuilder,
    NetworkTopology, 
    NetworkConfiguration,
    Network
)
from sdmn.neurons import NeuronType, LIFParameters


class TestNetworkConfiguration:
    """Test NetworkConfiguration class."""

    @pytest.mark.unit
    def test_default_configuration(self):
        """Test default network configuration."""
        config = NetworkConfiguration(
            name="test_net",
            n_neurons=50,
            topology=NetworkTopology.RANDOM
        )
        
        assert config.name == "test_net"
        assert config.n_neurons == 50
        assert config.topology == NetworkTopology.RANDOM
        assert config.neuron_type == NeuronType.LEAKY_INTEGRATE_FIRE
        assert config.connection_probability == 0.1
        assert config.excitatory_ratio == 0.8
        assert config.enable_plasticity is False

    @pytest.mark.unit
    def test_custom_configuration(self):
        """Test custom network configuration.""" 
        config = NetworkConfiguration(
            name="custom_net",
            n_neurons=100,
            topology=NetworkTopology.SMALL_WORLD,
            neuron_type=NeuronType.HODGKIN_HUXLEY,
            connection_probability=0.15,
            weight_range=(1.0, 3.0),
            delay_range=(0.5, 8.0),
            excitatory_ratio=0.75,
            enable_plasticity=True
        )
        
        assert config.neuron_type == NeuronType.HODGKIN_HUXLEY
        assert config.connection_probability == 0.15
        assert config.weight_range == (1.0, 3.0)
        assert config.delay_range == (0.5, 8.0)
        assert config.excitatory_ratio == 0.75
        assert config.enable_plasticity is True

    @pytest.mark.unit
    def test_custom_params_initialization(self):
        """Test custom parameters initialization."""
        config = NetworkConfiguration(
            name="test",
            n_neurons=10,
            topology=NetworkTopology.RING,
            custom_params={'special_param': 42}
        )
        
        assert config.custom_params['special_param'] == 42
        
        # Test None initialization
        config2 = NetworkConfiguration(
            name="test2",
            n_neurons=10,
            topology=NetworkTopology.RING
        )
        assert config2.custom_params == {}


class TestNetworkBuilder:
    """Test NetworkBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create NetworkBuilder instance."""
        return NetworkBuilder()

    @pytest.mark.unit
    def test_builder_initialization(self, builder):
        """Test network builder initialization."""
        assert isinstance(builder.networks, dict)
        assert isinstance(builder.configurations, dict)
        assert len(builder.networks) == 0

    @pytest.mark.integration
    def test_create_small_random_network(self, builder, small_network_config):
        """Test creating small random network."""
        network = builder.create_network(small_network_config)
        
        assert isinstance(network, Network)
        assert network.name == "test_network"
        assert len(network.neurons) == 10
        assert network in builder.networks.values()
        
        # Check network statistics
        stats = network.get_network_statistics()
        assert stats['total_neurons'] == 10
        assert stats['total_synapses'] > 0
        assert 0 <= stats['excitatory_ratio'] <= 1

    @pytest.mark.integration
    def test_create_ring_network(self, builder):
        """Test creating ring topology network."""
        config = NetworkConfiguration(
            name="ring_net",
            n_neurons=8,
            topology=NetworkTopology.RING,
            connection_probability=0.5  # Not used in ring
        )
        
        network = builder.create_network(config)
        
        assert len(network.neurons) == 8
        # Ring topology should have exactly n connections
        assert len(network.synapses) == 8

    @pytest.mark.integration
    def test_create_small_world_network(self, builder):
        """Test creating small-world topology network."""
        config = NetworkConfiguration(
            name="sw_net",
            n_neurons=20,
            topology=NetworkTopology.SMALL_WORLD,
            connection_probability=0.1
        )
        
        network = builder.create_network(config)
        
        assert len(network.neurons) == 20
        assert len(network.synapses) > 20  # Should have more than ring connections
        
        # Should have both local and random connections
        stats = network.get_network_statistics()
        assert stats['mean_in_degree'] > 1.0  # More than simple ring

    @pytest.mark.integration 
    def test_create_grid_network(self, builder):
        """Test creating 2D grid topology network."""
        config = NetworkConfiguration(
            name="grid_net",
            n_neurons=16,  # 4x4 grid
            topology=NetworkTopology.GRID_2D
        )
        
        network = builder.create_network(config)
        
        assert len(network.neurons) == 16
        # Grid should have local connections
        assert len(network.synapses) > 0
        
    @pytest.mark.unit
    def test_parameter_variability(self, builder):
        """Test that neurons have parameter variability."""
        config = NetworkConfiguration(
            name="var_test",
            n_neurons=5,
            topology=NetworkTopology.RANDOM,
            neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE
        )
        
        network = builder.create_network(config)
        
        # Check that neurons have slightly different parameters
        neuron_list = list(network.neurons.values())
        v_rest_values = [n.parameters.v_rest for n in neuron_list]
        
        # Should have some variability (not all exactly the same)
        assert len(set(v_rest_values)) > 1 or len(v_rest_values) == 1  # Allow for small networks

    @pytest.mark.unit
    def test_plasticity_configuration(self, builder):
        """Test plasticity configuration in network."""
        config = NetworkConfiguration(
            name="plastic_net",
            n_neurons=5,
            topology=NetworkTopology.RANDOM,
            connection_probability=0.8,
            enable_plasticity=True
        )
        
        network = builder.create_network(config)
        
        # Check that some synapses have plasticity enabled
        plastic_synapses = [s for s in network.synapses.values() 
                          if s.parameters.enable_plasticity]
        assert len(plastic_synapses) > 0

    @pytest.mark.unit
    def test_excitatory_inhibitory_ratio(self, builder):
        """Test excitatory/inhibitory synapse ratio."""
        config = NetworkConfiguration(
            name="ratio_test",
            n_neurons=10,
            topology=NetworkTopology.RANDOM,
            connection_probability=0.5,
            excitatory_ratio=0.7
        )
        
        network = builder.create_network(config)
        
        if len(network.synapses) > 0:
            stats = network.get_network_statistics()
            # Should be approximately 0.7 (within reasonable tolerance)
            assert 0.5 <= stats['excitatory_ratio'] <= 0.9

    @pytest.mark.unit
    def test_get_network(self, builder, small_network_config):
        """Test getting network by name."""
        network = builder.create_network(small_network_config)
        
        retrieved = builder.get_network("test_network")
        assert retrieved == network
        
        assert builder.get_network("nonexistent") is None

    @pytest.mark.unit
    def test_list_networks(self, builder):
        """Test listing network names."""
        config1 = NetworkConfiguration("net1", 5, NetworkTopology.RANDOM)
        config2 = NetworkConfiguration("net2", 5, NetworkTopology.RING)
        
        builder.create_network(config1)
        builder.create_network(config2)
        
        networks = builder.list_networks()
        assert "net1" in networks
        assert "net2" in networks
        assert len(networks) == 2

    @pytest.mark.unit
    def test_unsupported_topology(self, builder):
        """Test error handling for unsupported topology."""
        # This would require modifying the enum or using a mock
        # For now, test with a topology that should raise NotImplementedError
        config = NetworkConfiguration(
            name="unsupported",
            n_neurons=5,
            topology=NetworkTopology.HIERARCHICAL  # Not implemented
        )
        
        with pytest.raises(NotImplementedError):
            builder.create_network(config)


class TestNetwork:
    """Test Network class."""

    @pytest.fixture
    def simple_network(self, network_builder):
        """Create simple network for testing."""
        config = NetworkConfiguration(
            name="simple_test",
            n_neurons=5,
            topology=NetworkTopology.RANDOM,
            connection_probability=0.4
        )
        return network_builder.create_network(config)

    @pytest.mark.unit
    def test_network_initialization(self, simple_network):
        """Test network initialization."""
        assert simple_network.name == "simple_test"
        assert len(simple_network.neurons) == 5
        assert simple_network.current_time == 0.0
        assert isinstance(simple_network.synapses, dict)

    @pytest.mark.unit
    def test_network_update(self, simple_network):
        """Test network update method."""
        initial_time = simple_network.current_time
        dt = 0.1
        
        simple_network.update(dt)
        
        assert simple_network.current_time == initial_time + dt
        
        # All neurons should have been updated
        for neuron in simple_network.neurons.values():
            assert neuron.current_time >= dt

    @pytest.mark.unit
    def test_network_statistics(self, simple_network):
        """Test network statistics calculation."""
        stats = simple_network.get_network_statistics()
        
        required_keys = [
            'name', 'total_neurons', 'total_synapses',
            'excitatory_synapses', 'inhibitory_synapses',
            'excitatory_ratio', 'connection_density',
            'mean_in_degree', 'mean_out_degree'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert stats['name'] == "simple_test"
        assert stats['total_neurons'] == 5
        assert stats['excitatory_ratio'] >= 0
        assert stats['connection_density'] >= 0

    @pytest.mark.unit
    def test_network_reset(self, simple_network):
        """Test network reset functionality."""
        # Modify network state
        simple_network.update(1.0)
        
        # Add some artificial state to neurons
        for neuron in simple_network.neurons.values():
            neuron.spike_times.append(0.5)
        
        simple_network.reset()
        
        assert simple_network.current_time == 0.0
        
        # All neurons should be reset
        for neuron in simple_network.neurons.values():
            assert neuron.current_time == 0.0
            assert len(neuron.spike_times) == 0

    @pytest.mark.integration
    def test_network_with_stimulus(self, simple_network):
        """Test network behavior with external stimulus."""
        # Add external input to first neuron
        first_neuron = next(iter(simple_network.neurons.values()))
        first_neuron.set_external_input(2.0)  # Strong input
        
        initial_spikes = sum(len(n.spike_times) for n in simple_network.neurons.values())
        
        # Run network for several steps
        for _ in range(100):  # 10ms
            simple_network.update(0.1)
        
        final_spikes = sum(len(n.spike_times) for n in simple_network.neurons.values())
        
        # Should generate some spikes
        assert final_spikes > initial_spikes

    @pytest.mark.integration
    def test_network_connectivity(self, simple_network):
        """Test network connectivity properties."""
        # Test that neurons are properly connected to synapses
        neuron_ids = set(simple_network.neurons.keys())
        
        for synapse in simple_network.synapses.values():
            # Pre and post neurons should exist
            assert synapse.presynaptic_neuron_id in neuron_ids
            assert synapse.postsynaptic_neuron_id in neuron_ids
            
            # Neurons should have references to synapses
            pre_neuron = simple_network.neurons[synapse.presynaptic_neuron_id]
            post_neuron = simple_network.neurons[synapse.postsynaptic_neuron_id]
            
            assert synapse in pre_neuron.postsynaptic_connections
            assert synapse in post_neuron.presynaptic_connections

    @pytest.mark.unit
    def test_network_string_representation(self, simple_network):
        """Test network string representations."""
        str_repr = str(simple_network)
        assert "simple_test" in str_repr
        assert "5" in str_repr  # neuron count
        
        repr_str = repr(simple_network)
        assert "Network" in repr_str
        assert "simple_test" in repr_str

    @pytest.mark.integration
    def test_synaptic_transmission(self, simple_network):
        """Test synaptic transmission in network."""
        # Find a synapse and its connected neurons
        if len(simple_network.synapses) == 0:
            pytest.skip("No synapses in test network")
        
        synapse = next(iter(simple_network.synapses.values()))
        pre_neuron = simple_network.neurons[synapse.presynaptic_neuron_id]
        post_neuron = simple_network.neurons[synapse.postsynaptic_neuron_id]
        
        # Force pre-synaptic neuron to spike
        pre_neuron.set_external_input(10.0)
        initial_post_v = post_neuron.get_membrane_potential()
        
        # Run network to allow spike propagation
        for _ in range(50):  # 5ms
            simple_network.update(0.1)
            if pre_neuron.has_spiked():
                break
        
        # Continue running to see effect on post-synaptic neuron
        for _ in range(100):  # 10ms more
            simple_network.update(0.1)
        
        final_post_v = post_neuron.get_membrane_potential()
        
        # Post-synaptic potential should have changed
        # (exact change depends on synapse type and timing)
        assert final_post_v != initial_post_v
