"""Integration tests for the SDMN framework."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from sdmn.core import SimulationEngine, SimulationConfig
from sdmn.networks import NetworkBuilder, NetworkConfiguration, NetworkTopology
from sdmn.neurons import NeuronType, LIFParameters
from sdmn.probes import VoltageProbe, SpikeProbe, PopulationActivityProbe


class TestBasicSimulation:
    """Test basic simulation functionality."""

    @pytest.mark.integration
    def test_simple_lif_simulation(self):
        """Test simple LIF network simulation."""
        # Create simulation configuration
        config = SimulationConfig(
            dt=0.1,
            max_time=50.0,  # Short simulation
            enable_logging=False,
            random_seed=42
        )
        
        # Create engine
        engine = SimulationEngine(config)
        
        # Create small network
        network_config = NetworkConfiguration(
            name="test_network",
            n_neurons=5,
            topology=NetworkTopology.RANDOM,
            neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE,
            connection_probability=0.3
        )
        
        builder = NetworkBuilder()
        network = builder.create_network(network_config)
        engine.add_network("main", network)
        
        # Add probes
        voltage_probe = VoltageProbe("voltage", list(network.neurons.keys())[:2])
        spike_probe = SpikeProbe("spikes", list(network.neurons.keys()))
        
        # Register neurons with probes
        for neuron_id, neuron in network.neurons.items():
            voltage_probe.register_neuron_object(neuron_id, neuron)
            spike_probe.register_neuron_object(neuron_id, neuron)
        
        engine.add_probe("voltage", voltage_probe)
        engine.add_probe("spikes", spike_probe)
        
        # Run simulation
        results = engine.run()
        
        # Check results
        assert results.success is True
        assert results.total_steps > 0
        assert results.simulation_time == 50.0
        
        # Check probe data
        voltage_data = voltage_probe.get_data()
        spike_data = spike_probe.get_data()
        
        assert len(voltage_data.timestamps) > 0
        assert len(spike_data.timestamps) >= 0  # May or may not have spikes

    @pytest.mark.integration  
    def test_stimulated_network_simulation(self):
        """Test network simulation with external stimulation."""
        config = SimulationConfig(
            dt=0.1,
            max_time=100.0,
            enable_logging=False,
            random_seed=123
        )
        
        engine = SimulationEngine(config)
        
        # Create network
        network_config = NetworkConfiguration(
            name="stimulated_network",
            n_neurons=10,
            topology=NetworkTopology.SMALL_WORLD,
            neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE,
            connection_probability=0.2
        )
        
        builder = NetworkBuilder()
        network = builder.create_network(network_config)
        
        # Add strong stimulus to first neuron
        first_neuron = next(iter(network.neurons.values()))
        first_neuron.set_external_input(3.0)  # Strong current
        
        engine.add_network("main", network)
        
        # Add population probe
        pop_probe = PopulationActivityProbe(
            "population",
            "main_pop",
            list(network.neurons.keys()),
            bin_size=10.0
        )
        pop_probe.register_neuron_objects(network.neurons)
        engine.add_probe("population", pop_probe)
        
        # Run simulation
        results = engine.run()
        
        assert results.success is True
        
        # Should generate some population activity
        pop_data = pop_probe.get_data()
        assert len(pop_data.timestamps) > 0
        
        # Check if there was any spiking activity
        total_spikes = sum(len(n.spike_times) for n in network.neurons.values())
        assert total_spikes > 0  # Should have some spikes with stimulation

    @pytest.mark.integration
    def test_network_with_different_topologies(self):
        """Test networks with different topologies."""
        topologies = [
            NetworkTopology.RANDOM,
            NetworkTopology.RING,
            NetworkTopology.SMALL_WORLD,
            NetworkTopology.GRID_2D
        ]
        
        builder = NetworkBuilder()
        networks = []
        
        for topology in topologies:
            config = NetworkConfiguration(
                name=f"{topology.value}_network",
                n_neurons=8,  # Small for fast testing
                topology=topology,
                connection_probability=0.3
            )
            
            network = builder.create_network(config)
            networks.append(network)
            
            # Basic sanity checks
            assert len(network.neurons) == 8
            assert len(network.synapses) >= 0
        
        # Check that different topologies produce different connectivity
        stats = [net.get_network_statistics() for net in networks]
        
        # Ring should have exactly n synapses
        ring_stats = stats[1]  # Ring is second in list
        assert ring_stats['total_synapses'] == 8

    @pytest.mark.integration
    def test_checkpointing_and_restore(self):
        """Test checkpointing and restoration functionality.""" 
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SimulationConfig(
                dt=0.1,
                max_time=30.0,
                checkpoint_interval=100,  # Checkpoint every 100 steps
                enable_logging=False,
                checkpoint_dir=Path(tmp_dir)
            )
            
            engine = SimulationEngine(config)
            
            # Create simple network
            network_config = NetworkConfiguration(
                name="checkpoint_test",
                n_neurons=5,
                topology=NetworkTopology.RANDOM,
                connection_probability=0.2
            )
            
            builder = NetworkBuilder()
            network = builder.create_network(network_config)
            engine.add_network("main", network)
            
            # Add stimulus
            first_neuron = next(iter(network.neurons.values()))
            first_neuron.set_external_input(2.0)
            
            # Run simulation partway
            engine.time_manager.max_time = 15.0  # Run to halfway
            partial_results = engine.run()
            
            assert partial_results.success is True
            
            # Create manual checkpoint
            checkpoint = engine.create_manual_checkpoint({"test": "checkpoint"})
            assert checkpoint.step_number > 0
            
            # Continue simulation
            engine.time_manager.max_time = 30.0
            engine.is_running = True  # Reset running state
            final_results = engine.run()
            
            assert final_results.success is True
            assert final_results.simulation_time == 30.0


class TestProbeIntegration:
    """Test probe integration with simulation."""

    @pytest.mark.integration
    def test_multi_probe_simulation(self):
        """Test simulation with multiple probe types."""
        config = SimulationConfig(
            dt=0.1,
            max_time=200.0,
            enable_logging=False,
            random_seed=456
        )
        
        engine = SimulationEngine(config)
        
        # Create network
        network_config = NetworkConfiguration(
            name="multi_probe_net",
            n_neurons=15,
            topology=NetworkTopology.SMALL_WORLD,
            neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE,
            connection_probability=0.15
        )
        
        builder = NetworkBuilder()
        network = builder.create_network(network_config)
        engine.add_network("main", network)
        
        # Add stimulation to multiple neurons
        neuron_list = list(network.neurons.values())
        neuron_list[0].set_external_input(1.5)
        neuron_list[1].set_external_input(1.0)
        
        # Create multiple probes
        neuron_ids = list(network.neurons.keys())
        
        # Voltage probe on subset
        voltage_probe = VoltageProbe("voltage", neuron_ids[:5], sampling_interval=1.0)
        for nid in neuron_ids[:5]:
            voltage_probe.register_neuron_object(nid, network.neurons[nid])
        
        # Spike probe on all neurons
        spike_probe = SpikeProbe("spikes", neuron_ids)
        for nid, neuron in network.neurons.items():
            spike_probe.register_neuron_object(nid, neuron)
        
        # Population probe
        pop_probe = PopulationActivityProbe(
            "population", "main_pop", neuron_ids,
            bin_size=20.0, record_synchrony=True
        )
        pop_probe.register_neuron_objects(network.neurons)
        
        # Add probes to engine
        engine.add_probe("voltage", voltage_probe)
        engine.add_probe("spikes", spike_probe)
        engine.add_probe("population", pop_probe)
        
        # Run simulation
        results = engine.run()
        
        assert results.success is True
        
        # Check all probes collected data
        v_data = voltage_probe.get_data()
        s_data = spike_probe.get_data()
        p_data = pop_probe.get_data()
        
        assert len(v_data.timestamps) > 0
        assert len(s_data.timestamps) >= 0  # May have spikes
        assert len(p_data.timestamps) > 0
        
        # Check voltage statistics
        v_stats = voltage_probe.get_voltage_statistics()
        assert len(v_stats) == 5  # 5 monitored neurons
        
        # Check spike statistics  
        spike_counts = spike_probe.get_spike_counts()
        total_spikes = sum(spike_counts.values())
        
        # With stimulation, should have some spikes
        assert total_spikes > 0
        
        # Check population statistics
        pop_stats = pop_probe.get_population_statistics()
        assert 'mean_rate' in pop_stats
        assert pop_stats['total_spikes'] >= 0

    @pytest.mark.integration
    def test_probe_data_consistency(self):
        """Test consistency between different probe measurements."""
        config = SimulationConfig(
            dt=0.1,
            max_time=100.0,
            enable_logging=False,
            random_seed=789
        )
        
        engine = SimulationEngine(config)
        
        # Create small network for precise monitoring
        network_config = NetworkConfiguration(
            name="consistency_test",
            n_neurons=3,
            topology=NetworkTopology.RING,  # Predictable connectivity
            neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE
        )
        
        builder = NetworkBuilder()
        network = builder.create_network(network_config)
        engine.add_network("main", network)
        
        # Strong stimulation to ensure spikes
        first_neuron = list(network.neurons.values())[0]
        first_neuron.set_external_input(5.0)
        
        neuron_ids = list(network.neurons.keys())
        
        # High-resolution voltage probe
        voltage_probe = VoltageProbe("voltage", neuron_ids, sampling_interval=0.1)
        for nid, neuron in network.neurons.items():
            voltage_probe.register_neuron_object(nid, neuron)
        
        # Spike detector
        spike_probe = SpikeProbe("spikes", neuron_ids, detection_threshold=-40.0)
        for nid, neuron in network.neurons.items():
            spike_probe.register_neuron_object(nid, neuron)
        
        engine.add_probe("voltage", voltage_probe)
        engine.add_probe("spikes", spike_probe)
        
        # Run simulation
        results = engine.run()
        assert results.success is True
        
        # Analyze consistency
        voltage_traces = voltage_probe.get_voltage_traces()
        spike_times = spike_probe.get_spike_times()
        
        # For each neuron that spiked, check voltage traces
        for neuron_id, spikes in spike_times.items():
            if len(spikes) > 0 and neuron_id in voltage_traces:
                trace = voltage_traces[neuron_id]
                
                # Check that spikes correspond to high voltage values
                for spike_time in spikes:
                    # Find closest voltage measurement
                    time_diffs = np.abs(trace['time'] - spike_time)
                    closest_idx = np.argmin(time_diffs)
                    
                    if time_diffs[closest_idx] < 1.0:  # Within 1ms
                        voltage_at_spike = trace['voltage'][closest_idx]
                        # Should be near or above threshold
                        assert voltage_at_spike > -60.0  # Well above resting


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_medium_network_simulation(self):
        """Test simulation with medium-sized network.""" 
        config = SimulationConfig(
            dt=0.1,
            max_time=500.0,  # Longer simulation
            enable_logging=False,
            random_seed=999
        )
        
        engine = SimulationEngine(config)
        
        # Medium-sized network
        network_config = NetworkConfiguration(
            name="medium_network",
            n_neurons=100,
            topology=NetworkTopology.SMALL_WORLD,
            neuron_type=NeuronType.LEAKY_INTEGRATE_FIRE,
            connection_probability=0.05,  # Sparse connectivity
            enable_plasticity=True
        )
        
        builder = NetworkBuilder()
        network = builder.create_network(network_config)
        
        # Add distributed stimulation
        neurons = list(network.neurons.values())
        for i in range(0, len(neurons), 10):  # Every 10th neuron
            neurons[i].set_external_input(np.random.uniform(0.5, 2.0))
        
        engine.add_network("main", network)
        
        # Add population-level monitoring only (for performance)
        pop_probe = PopulationActivityProbe(
            "population",
            "main_pop", 
            list(network.neurons.keys()),
            bin_size=50.0
        )
        pop_probe.register_neuron_objects(network.neurons)
        engine.add_probe("population", pop_probe)
        
        # Run simulation and measure performance
        import time
        start_time = time.time()
        results = engine.run()
        end_time = time.time()
        
        assert results.success is True
        
        # Basic performance check (should complete in reasonable time)
        simulation_time = end_time - start_time
        assert simulation_time < 60.0  # Should complete within 1 minute
        
        # Check network generated reasonable activity
        network_stats = network.get_network_statistics()
        assert network_stats['total_neurons'] == 100
        assert network_stats['total_synapses'] > 0
        
        # Check population activity
        pop_stats = pop_probe.get_population_statistics()
        assert pop_stats['total_spikes'] >= 0

    @pytest.mark.integration
    def test_different_neuron_types(self):
        """Test simulation with different neuron types."""
        # Test both LIF and HH neurons in separate simulations
        neuron_types = [
            NeuronType.LEAKY_INTEGRATE_FIRE,
            NeuronType.HODGKIN_HUXLEY
        ]
        
        for neuron_type in neuron_types:
            config = SimulationConfig(
                dt=0.01 if neuron_type == NeuronType.HODGKIN_HUXLEY else 0.1,
                max_time=50.0,
                enable_logging=False,
                random_seed=111
            )
            
            engine = SimulationEngine(config)
            
            network_config = NetworkConfiguration(
                name=f"{neuron_type.value}_network",
                n_neurons=5,
                topology=NetworkTopology.RANDOM,
                neuron_type=neuron_type,
                connection_probability=0.3
            )
            
            builder = NetworkBuilder()
            network = builder.create_network(network_config)
            
            # Add stimulation
            first_neuron = list(network.neurons.values())[0]
            if neuron_type == NeuronType.HODGKIN_HUXLEY:
                first_neuron.set_external_input(20.0)  # HH needs more current
            else:
                first_neuron.set_external_input(2.0)
            
            engine.add_network("main", network)
            
            # Run simulation
            results = engine.run()
            
            assert results.success is True
            assert results.total_steps > 0
            
            # Verify neuron type
            for neuron in network.neurons.values():
                assert neuron.parameters.neuron_type == neuron_type


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness."""

    @pytest.mark.integration
    def test_simulation_with_no_network(self):
        """Test simulation behavior with no networks added."""
        config = SimulationConfig(
            dt=0.1,
            max_time=10.0,
            enable_logging=False
        )
        
        engine = SimulationEngine(config)
        
        # Run simulation without adding any networks
        results = engine.run()
        
        # Should complete successfully (empty simulation)
        assert results.success is True
        assert results.total_steps > 0

    @pytest.mark.integration
    def test_simulation_interruption(self):
        """Test simulation interruption and cleanup."""
        config = SimulationConfig(
            dt=0.1,
            max_time=100.0,
            enable_logging=False
        )
        
        engine = SimulationEngine(config)
        
        # Create network
        network_config = NetworkConfiguration(
            name="interrupt_test",
            n_neurons=10,
            topology=NetworkTopology.RANDOM
        )
        
        builder = NetworkBuilder()
        network = builder.create_network(network_config)
        engine.add_network("main", network)
        
        # Start simulation in separate thread and interrupt
        import threading
        import time
        
        results = [None]
        
        def run_simulation():
            results[0] = engine.run()
        
        sim_thread = threading.Thread(target=run_simulation)
        sim_thread.start()
        
        # Let it run briefly then stop
        time.sleep(0.1)  
        engine.stop()
        
        sim_thread.join(timeout=5.0)
        
        # Should have stopped gracefully
        assert results[0] is not None
        assert results[0].total_steps > 0
