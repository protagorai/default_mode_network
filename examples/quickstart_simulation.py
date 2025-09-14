#!/usr/bin/env python3
"""
Quickstart simulation example for SDMN Framework.

This script demonstrates the basic usage of the framework by creating
a simple neural network with feedback loops and monitoring it with probes
to generate synthetic brain wave patterns.

Run with:
    python examples/quickstart_simulation.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import SDMN Framework components
from neurons import LIFNeuron, LIFParameters, SynapseFactory
from probes import VoltageProbe, SpikeProbe, PopulationActivityProbe
from core import SimulationEngine, SimulationConfig

class SimpleNetwork:
    """Simple network implementation for the quickstart example."""
    
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
    
    def update(self, dt):
        """Update all network components for one time step."""
        # Update all synapses first
        for synapse in self.synapses.values():
            synapse.update(dt)
        
        # Update all neurons
        for neuron in self.neurons.values():
            # Calculate synaptic inputs
            for synapse in neuron.presynaptic_connections:
                if synapse.synapse_id in self.synapses:
                    current = synapse.calculate_current(
                        neuron.get_membrane_potential()
                    )
                    neuron.add_synaptic_input(current)
            
            # Update neuron state
            neuron.update(dt)

def create_ring_network(n_neurons=10, add_feedback=True):
    """
    Create a ring network with optional long-range feedback connections.
    
    Args:
        n_neurons: Number of neurons in the ring
        add_feedback: Whether to add long-range feedback connections
        
    Returns:
        SimpleNetwork instance
    """
    print(f"Creating ring network with {n_neurons} neurons...")
    
    neurons = {}
    synapses = {}
    
    # Create neurons with slight parameter variations for heterogeneity
    for i in range(n_neurons):
        neuron_id = f"neuron_{i:03d}"
        params = LIFParameters(
            tau_m=20.0 + np.random.normal(0, 1.0),      # Vary time constant
            v_thresh=-50.0 + np.random.normal(0, 1.0),  # Vary threshold
            r_mem=10.0 + np.random.normal(0, 0.5),      # Vary resistance
            refractory_period=2.0
        )
        neurons[neuron_id] = LIFNeuron(neuron_id, params)
    
    # Create ring connections (each neuron connects to the next)
    for i in range(n_neurons):
        next_i = (i + 1) % n_neurons
        syn_id = f"ring_syn_{i}_to_{next_i}"
        
        # Create excitatory synapse with delay
        synapse = SynapseFactory.create_excitatory_synapse(
            syn_id,
            f"neuron_{i:03d}",
            f"neuron_{next_i:03d}",
            weight=1.2,
            delay=3.0
        )
        synapses[syn_id] = synapse
        
        # Register connections with neurons
        neurons[f"neuron_{i:03d}"].add_postsynaptic_connection(synapse)
        neurons[f"neuron_{next_i:03d}"].add_presynaptic_connection(synapse)
    
    # Add long-range feedback connections for default mode network behavior
    if add_feedback:
        for i in range(0, n_neurons, 3):  # Every 3rd neuron
            target_i = (i + n_neurons//2) % n_neurons  # Connect to opposite side
            syn_id = f"feedback_syn_{i}_to_{target_i}"
            
            synapse = SynapseFactory.create_excitatory_synapse(
                syn_id,
                f"neuron_{i:03d}",
                f"neuron_{target_i:03d}",
                weight=0.8,
                delay=8.0
            )
            synapses[syn_id] = synapse
            
            # Register feedback connections
            neurons[f"neuron_{i:03d}"].add_postsynaptic_connection(synapse)
            neurons[f"neuron_{target_i:03d}"].add_presynaptic_connection(synapse)
    
    print(f"Created {len(neurons)} neurons and {len(synapses)} synapses")
    return SimpleNetwork(neurons, synapses)

def setup_monitoring(network):
    """
    Set up probes to monitor network activity.
    
    Args:
        network: SimpleNetwork instance
        
    Returns:
        Dictionary of probe objects
    """
    print("Setting up monitoring probes...")
    
    neuron_ids = list(network.neurons.keys())
    
    # Voltage probe for membrane potential traces (synthetic EEG)
    voltage_probe = VoltageProbe(
        probe_id="voltage_monitor",
        target_neurons=neuron_ids[:5],  # Monitor first 5 neurons
        sampling_interval=1.0,          # Sample every 1 ms
        enable_filtering=True,
        filter_cutoff=100.0            # 100 Hz low-pass filter
    )
    
    # Register neuron objects
    for neuron_id in voltage_probe.target_ids:
        voltage_probe.register_neuron_object(neuron_id, network.neurons[neuron_id])
    
    # Spike probe for precise spike timing
    spike_probe = SpikeProbe(
        probe_id="spike_monitor",
        target_neurons=neuron_ids,
        detection_threshold=-35.0
    )
    
    # Register neurons with spike probe
    for neuron_id, neuron in network.neurons.items():
        spike_probe.register_neuron_object(neuron_id, neuron)
    
    # Population activity probe for synthetic brain waves
    population_probe = PopulationActivityProbe(
        probe_id="population_monitor",
        target_population="ring_network",
        target_neurons=neuron_ids,
        bin_size=5.0,              # 5 ms bins
        sliding_window=50.0,       # 50 ms window
        record_synchrony=True
    )
    
    population_probe.register_neuron_objects(network.neurons)
    
    probes = {
        'voltage': voltage_probe,
        'spike': spike_probe,
        'population': population_probe
    }
    
    print(f"Set up {len(probes)} monitoring probes")
    return probes

def run_simulation(network, probes, duration=1000.0):
    """
    Run the neural network simulation.
    
    Args:
        network: SimpleNetwork instance
        probes: Dictionary of probe objects
        duration: Simulation duration in ms
        
    Returns:
        SimulationResults object
    """
    print(f"Starting simulation for {duration} ms...")
    
    # Create simulation configuration
    config = SimulationConfig(
        dt=0.1,                    # 0.1 ms time steps
        max_time=duration,
        checkpoint_interval=5000,   # Checkpoint every 500 ms
        enable_logging=True,
        log_level="INFO"
    )
    
    # Create simulation engine
    engine = SimulationEngine(config)
    
    # Add network to engine
    engine.add_network("ring_network", network)
    
    # Add probes to engine
    for probe_name, probe in probes.items():
        engine.add_probe(probe_name, probe)
    
    # Set up stimulus pattern
    def stimulus_callback(step, time):
        """Apply periodic stimulation to maintain network activity."""
        # Apply stimulus every 100 ms with some variability
        if step % 1000 == 0:  # Every 100 ms (1000 steps * 0.1 ms)
            # Random stimulus to first few neurons
            for i in range(3):
                neuron_id = f"neuron_{i:03d}"
                stimulus_current = np.random.normal(2.5, 0.5)  # 2.5 ± 0.5 nA
                network.neurons[neuron_id].set_external_input(stimulus_current)
        else:
            # Small baseline current to maintain excitability
            for i in range(3):
                neuron_id = f"neuron_{i:03d}"
                network.neurons[neuron_id].set_external_input(0.2)
    
    engine.register_step_callback(stimulus_callback)
    
    # Start recording
    for probe in probes.values():
        probe.start_recording()
    
    # Run simulation
    results = engine.run()
    
    # Stop recording
    for probe in probes.values():
        probe.stop_recording()
    
    if results.success:
        print(f"Simulation completed successfully!")
        print(f"Simulated {results.simulation_time:.1f} ms in {results.wall_time:.2f} seconds")
        speed_ratio = results.simulation_time / (results.wall_time * 1000)
        print(f"Speed: {speed_ratio:.1f}x real-time")
    else:
        print(f"Simulation failed: {results.error_message}")
    
    return results

def analyze_results(probes):
    """
    Analyze and visualize simulation results.
    
    Args:
        probes: Dictionary of probe objects with recorded data
    """
    print("Analyzing results and generating plots...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('SDMN Framework - Synthetic Default Mode Network Activity', fontsize=16)
    
    # 1. Voltage traces (synthetic EEG)
    voltage_traces = probes['voltage'].get_voltage_traces()
    colors = plt.cm.viridis(np.linspace(0, 1, len(voltage_traces)))
    
    for i, (neuron_id, trace) in enumerate(voltage_traces.items()):
        if len(trace['time']) > 0:
            # Offset voltages for better visualization
            offset = i * 20  # 20 mV offset between traces
            axes[0].plot(trace['time'], trace['voltage'] + offset, 
                        label=f'{neuron_id}', color=colors[i], linewidth=1)
    
    axes[0].set_ylabel('Membrane Potential (mV)')
    axes[0].set_title('Synthetic "EEG" - Neural Membrane Potentials')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Spike raster plot
    spike_times = probes['spike'].get_spike_times()
    neuron_indices = {nid: i for i, nid in enumerate(spike_times.keys())}
    
    for neuron_id, spikes in spike_times.items():
        if spikes:
            y_positions = [neuron_indices[neuron_id]] * len(spikes)
            axes[1].scatter(spikes, y_positions, s=1, c='black', alpha=0.6)
    
    axes[1].set_ylabel('Neuron Index')
    axes[1].set_title('Network Spike Raster Plot')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Population firing rate (synthetic brain wave)
    times, rates = probes['population'].get_population_rate_trace()
    if len(rates) > 0:
        axes[2].plot(times, rates, 'b-', linewidth=2, label='Population Rate')
        axes[2].fill_between(times, rates, alpha=0.3)
        axes[2].set_ylabel('Firing Rate (Hz)')
        axes[2].set_title('Synthetic Brain Wave - Population Activity')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    # 4. Network synchrony
    times, synchrony = probes['population'].get_synchrony_trace()
    if len(synchrony) > 0:
        axes[3].plot(times, synchrony, 'r-', linewidth=2, label='Synchrony Index')
        axes[3].fill_between(times, synchrony, alpha=0.3, color='red')
        axes[3].set_ylabel('Synchrony Index')
        axes[3].set_title('Network Synchronization')
        axes[3].set_xlabel('Time (ms)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "sdmn_quickstart_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Results plot saved to: {output_file}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        print("Could not display plot (running in non-interactive mode)")
    
    # Print statistics
    print("\n--- Simulation Statistics ---")
    
    # Spike statistics
    total_spikes = sum(len(spikes) for spikes in spike_times.values())
    active_neurons = sum(1 for spikes in spike_times.values() if len(spikes) > 0)
    print(f"Total spikes: {total_spikes}")
    print(f"Active neurons: {active_neurons}/{len(spike_times)}")
    print(f"Average spikes per active neuron: {total_spikes/max(active_neurons, 1):.1f}")
    
    # Population statistics
    pop_stats = probes['population'].get_population_statistics()
    print(f"Mean population firing rate: {pop_stats['mean_rate']:.2f} Hz")
    print(f"Max population firing rate: {pop_stats['max_rate']:.2f} Hz")
    
    if 'mean_synchrony' in pop_stats:
        print(f"Mean network synchrony: {pop_stats['mean_synchrony']:.3f}")
    
    print(f"\nResults saved to {output_dir}/")

def main():
    """Main function to run the quickstart simulation."""
    print("="*60)
    print("SDMN Framework - Quickstart Simulation")
    print("="*60)
    print()
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create network
        network = create_ring_network(n_neurons=15, add_feedback=True)
        
        # Set up monitoring
        probes = setup_monitoring(network)
        
        # Run simulation
        results = run_simulation(network, probes, duration=2000.0)  # 2 seconds
        
        if results.success:
            # Analyze results
            analyze_results(probes)
            
            print("\n--- SUCCESS ---")
            print("Quickstart simulation completed successfully!")
            print("\nThis simulation demonstrated:")
            print("• Creating spiking neural networks with feedback loops")
            print("• Monitoring neural activity with multiple probe types")
            print("• Generating synthetic 'brain wave' patterns")
            print("• Analyzing network synchronization and dynamics")
            print("\nNext steps:")
            print("• Try modifying network parameters in the code")
            print("• Experiment with different neuron models")
            print("• Add more complex connectivity patterns")
            print("• Implement learning and plasticity")
        else:
            print(f"\n--- SIMULATION FAILED ---")
            print(f"Error: {results.error_message}")
            
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
