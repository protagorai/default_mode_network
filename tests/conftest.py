"""Pytest configuration and shared fixtures for SDMN tests."""

import pytest
import numpy as np
from typing import Dict, Any
from pathlib import Path

from sdmn.core import SimulationEngine, SimulationConfig
from sdmn.networks import NetworkBuilder, NetworkConfiguration, NetworkTopology
from sdmn.neurons import (
    NeuronType, 
    LIFNeuron, 
    LIFParameters,
    HHNeuron, 
    HHParameters,
    SynapseFactory
)
from sdmn.probes import VoltageProbe, SpikeProbe, PopulationActivityProbe


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("sdmn_test_data")


@pytest.fixture
def basic_sim_config():
    """Basic simulation configuration for testing."""
    return SimulationConfig(
        dt=0.1,
        max_time=100.0,
        checkpoint_interval=100,
        enable_logging=False,
        random_seed=42
    )


@pytest.fixture
def simulation_engine(basic_sim_config):
    """Basic simulation engine for testing."""
    return SimulationEngine(basic_sim_config)


@pytest.fixture
def lif_neuron():
    """Create a basic LIF neuron for testing."""
    params = LIFParameters(
        tau_m=20.0,
        v_rest=-70.0,
        v_thresh=-50.0,
        v_reset=-80.0,
        refractory_period=2.0
    )
    return LIFNeuron("test_lif", params)


@pytest.fixture
def hh_neuron():
    """Create a basic HH neuron for testing."""
    params = HHParameters()
    return HHNeuron("test_hh", params)


@pytest.fixture
def small_network_config():
    """Configuration for small test network."""
    return NetworkConfiguration(
        name="test_network",
        n_neurons=10,
        topology=NetworkTopology.RANDOM,
        neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE,
        connection_probability=0.2,
        weight_range=(0.5, 1.5),
        delay_range=(1.0, 5.0),
        excitatory_ratio=0.8
    )


@pytest.fixture
def network_builder():
    """Network builder instance."""
    return NetworkBuilder()


@pytest.fixture
def small_network(network_builder, small_network_config):
    """Small test network."""
    return network_builder.create_network(small_network_config)


@pytest.fixture
def voltage_probe():
    """Basic voltage probe."""
    return VoltageProbe("test_voltage", ["neuron_1", "neuron_2"], sampling_interval=0.5)


@pytest.fixture
def spike_probe():
    """Basic spike probe.""" 
    return SpikeProbe("test_spike", ["neuron_1", "neuron_2"])


@pytest.fixture
def population_probe():
    """Basic population activity probe."""
    return PopulationActivityProbe(
        "test_population", 
        "test_pop", 
        ["neuron_1", "neuron_2", "neuron_3"],
        bin_size=5.0
    )


@pytest.fixture
def excitatory_synapse():
    """Basic excitatory synapse."""
    return SynapseFactory.create_excitatory_synapse(
        "syn_exc", "pre_neuron", "post_neuron", weight=1.0, delay=2.0
    )


@pytest.fixture
def inhibitory_synapse():
    """Basic inhibitory synapse."""
    return SynapseFactory.create_inhibitory_synapse(
        "syn_inh", "pre_neuron", "post_neuron", weight=1.0, delay=2.0
    )


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_close(actual, expected, tolerance=1e-6):
        """Assert two values are close within tolerance."""
        if isinstance(actual, (list, np.ndarray)):
            actual = np.array(actual)
            expected = np.array(expected)
            assert np.allclose(actual, expected, atol=tolerance), \
                f"Arrays not close: {actual} vs {expected}"
        else:
            assert abs(actual - expected) < tolerance, \
                f"Values not close: {actual} vs {expected}"
    
    @staticmethod
    def run_simulation_steps(engine, n_steps=10):
        """Run simulation for specified number of steps."""
        for _ in range(n_steps):
            engine._execute_step()
    
    @staticmethod
    def create_spike_train(times, neuron_id="test_neuron"):
        """Create artificial spike train for testing."""
        spikes = []
        for t in times:
            spikes.append({
                'neuron_id': neuron_id,
                'spike_time': t,
                'peak_voltage': -30.0
            })
        return spikes


@pytest.fixture
def test_utils():
    """Test utilities."""
    return TestUtils


# Markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"  
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "network: mark test as network-related"
    )
    config.addinivalue_line(
        "markers", "neuron: mark test as neuron-related"
    )
    config.addinivalue_line(
        "markers", "probe: mark test as probe-related"
    )
