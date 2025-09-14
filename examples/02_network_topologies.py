#!/usr/bin/env python3
"""
Network Topologies Demonstration - SDMN Framework Example 02

This example demonstrates different network topologies and their properties,
including random, small-world, and default mode network architectures.

Run with:
    python examples/02_network_topologies.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from networks import NetworkBuilder, NetworkConfiguration, NetworkTopology
from neurons.base_neuron import NeuronType
from core import SimulationEngine, SimulationConfig
from probes import SpikeProbe, PopulationActivityProbe

def create_sample_networks():
    """Create different network topologies for comparison."""
    builder = NetworkBuilder()
    networks = {}
    
    # Common parameters
    n_neurons = 50
    sim_duration = 2000.0  # ms
    
    print("Creating sample networks...")
    
    # 1. Random Network
    random_config = NetworkConfiguration(
        name="random_network",
        n_neurons=n_neurons,
        topology=NetworkTopology.RANDOM,
        connection_probability=0.1,
        weight_range=(0.8, 1.5),
        delay_range=(1.0, 5.0),
        excitatory_ratio=0.8
    )
    networks["random"] = builder.create_network(random_config)
    
    # 2. Ring Network (for comparison)
    ring_config = NetworkConfiguration(
        name="ring_network", 
        n_neurons=n_neurons,
        topology=NetworkTopology.RING,
        weight_range=(1.0, 1.5),
        delay_range=(2.0, 6.0),
        excitatory_ratio=1.0  # All excitatory for ring
    )
    networks["ring"] = builder.create_network(ring_config)
    
    # 3. Small-World Network
    small_world_config = NetworkConfiguration(
        name="small_world_network",
        n_neurons=n_neurons,
        topology=NetworkTopology.SMALL_WORLD,
        connection_probability=0.15,
        weight_range=(0.9, 1.4),
        delay_range=(1.0, 8.0),
        excitatory_ratio=0.8
    )
    networks["small_world"] = builder.create_network(small_world_config)
    
    # 4. 2D Grid Network
    grid_config = NetworkConfiguration(
        name="grid_network",
        n_neurons=49,  # 7x7 grid
        topology=NetworkTopology.GRID_2D,
        weight_range=(1.0, 1.3),
        delay_range=(1.0, 3.0),
        excitatory_ratio=0.85
    )
    networks["grid"] = builder.create_network(grid_config)
    
    return networks

def analyze_network_properties(networks):
    """Analyze structural properties of different networks."""
    print("\n=== Network Analysis ===")
    
    analysis = {}
    
    for name, network in networks.items():
        stats = network.get_network_statistics()
        
        print(f"\n{name.upper()} NETWORK:")
        print(f"  Neurons: {stats['total_neurons']}")
        print(f"  Synapses: {stats['total_synapses']}")
        print(f"  Connection density: {stats['connection_density']*100:.2f}%")
        print(f"  E/I ratio: {stats['excitatory_ratio']:.2f}")
        print(f"  Mean in-degree: {stats['mean_in_degree']:.1f} ± {stats['std_in_degree']:.1f}")
        print(f"  Mean out-degree: {stats['mean_out_degree']:.1f} ± {stats['std_out_degree']:.1f}")
        
        analysis[name] = stats
    
    return analysis

def simulate_network_dynamics(networks):
    """Simulate dynamics of different network topologies."""
    print("\n=== Network Dynamics Simulation ===")
    
    results = {}
    
    for name, network in networks.items():
        print(f"\nSimulating {name} network...")
        
        # Create simulation configuration
        config = SimulationConfig(
            dt=0.1,
            max_time=1000.0,  # 1 second
            enable_logging=False
        )
        
        # Create simulation engine
        engine = SimulationEngine(config)
        engine.add_network(name, network)
        
        # Setup monitoring
        spike_probe = SpikeProbe(
            f"{name}_spikes",
            list(network.neurons.keys())
        )
        
        for neuron_id, neuron in network.neurons.items():
            spike_probe.register_neuron_object(neuron_id, neuron)
        
        pop_probe = PopulationActivityProbe(
            f"{name}_population",
            name,
            list(network.neurons.keys()),
            bin_size=10.0,
            record_synchrony=True
        )
        
        pop_probe.register_neuron_objects(network.neurons)
        
        engine.add_probe("spikes", spike_probe)
        engine.add_probe("population", pop_probe)
        
        # Add stimulus - stimulate a few random neurons
        stimulated_neurons = np.random.choice(
            list(network.neurons.keys()), 
            size=min(5, len(network.neurons)), 
            replace=False
        )
        
        def stimulus_callback(step, time):
            # Periodic stimulation
            if step % 1000 == 0:  # Every 100ms
                for neuron_id in stimulated_neurons:
                    current = np.random.normal(2.0, 0.3)
                    network.neurons[neuron_id].set_external_input(current)
            else:
                # Small baseline to maintain excitability
                for neuron_id in stimulated_neurons:
                    network.neurons[neuron_id].set_external_input(0.1)
        
        engine.register_step_callback(stimulus_callback)
        
        # Start recording
        spike_probe.start_recording()
        pop_probe.start_recording()
        
        # Run simulation
        sim_results = engine.run()
        
        if sim_results.success:
            # Collect results
            spike_times = spike_probe.get_spike_times()
            times, rates = pop_probe.get_population_rate_trace()
            times_sync, synchrony = pop_probe.get_synchrony_trace()
            
            total_spikes = sum(len(spikes) for spikes in spike_times.values())
            active_neurons = sum(1 for spikes in spike_times.values() if len(spikes) > 0)
            
            results[name] = {
                'spike_times': spike_times,
                'population_rates': (times, rates),
                'synchrony': (times_sync, synchrony),
                'total_spikes': total_spikes,
                'active_neurons': active_neurons,
                'mean_rate': np.mean(rates) if len(rates) > 0 else 0,
                'mean_synchrony': np.mean(synchrony) if len(synchrony) > 0 else 0
            }
            
            print(f"  Total spikes: {total_spikes}")
            print(f"  Active neurons: {active_neurons}/{len(network.neurons)}")
            print(f"  Mean population rate: {np.mean(rates):.2f} Hz")
            print(f"  Mean synchrony: {np.mean(synchrony):.3f}")
        else:
            print(f"  Simulation failed: {sim_results.error_message}")
            results[name] = None
    
    return results

def plot_network_comparison(networks, analysis, dynamics):
    """Plot comparison of network properties and dynamics."""
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Network Topology Comparison', fontsize=16)
    
    # Extract network names and colors
    network_names = list(networks.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(network_names)))
    
    # 1. Connection Statistics
    densities = [analysis[name]['connection_density']*100 for name in network_names]
    mean_degrees = [analysis[name]['mean_in_degree'] for name in network_names]
    
    x = np.arange(len(network_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, densities, width, label='Connection Density (%)', color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('Connection Density (%)')
    axes[0, 0].set_title('Network Connectivity')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([n.replace('_', '\n') for n in network_names])
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Degree Distribution for one network (small-world)
    if 'small_world' in networks:
        sw_network = networks['small_world']
        in_degrees = []
        out_degrees = []
        
        for neuron in sw_network.neurons.values():
            in_deg, out_deg = neuron.get_connection_count()
            in_degrees.append(in_deg)
            out_degrees.append(out_deg)
        
        axes[0, 1].hist(in_degrees, bins=10, alpha=0.7, label='In-degree', color='blue')
        axes[0, 1].hist(out_degrees, bins=10, alpha=0.7, label='Out-degree', color='red')
        axes[0, 1].set_xlabel('Degree')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Degree Distribution (Small-World)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Network Activity Summary
    if dynamics:
        total_spikes = [dynamics[name]['total_spikes'] if dynamics[name] else 0 for name in network_names]
        mean_rates = [dynamics[name]['mean_rate'] if dynamics[name] else 0 for name in network_names]
        
        axes[0, 2].bar(x, total_spikes, color=colors, alpha=0.7)
        axes[0, 2].set_ylabel('Total Spikes')
        axes[0, 2].set_title('Network Activity Level')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels([n.replace('_', '\n') for n in network_names])
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4-6. Population Rate Traces
    for i, name in enumerate(network_names[:3]):  # Show first 3 networks
        if dynamics and dynamics[name]:
            times, rates = dynamics[name]['population_rates']
            axes[1, i].plot(times, rates, color=colors[i], linewidth=2)
            axes[1, i].set_xlabel('Time (ms)')
            axes[1, i].set_ylabel('Population Rate (Hz)')
            axes[1, i].set_title(f'{name.replace("_", " ").title()} Activity')
            axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "02_network_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_file}")
    
    # Create raster plots
    plot_raster_comparison(dynamics, output_dir)
    
    try:
        plt.show()
    except:
        print("Could not display plot (running in non-interactive mode)")

def plot_raster_comparison(dynamics, output_dir):
    """Create raster plot comparison."""
    if not dynamics:
        return
        
    fig, axes = plt.subplots(len(dynamics), 1, figsize=(12, 8))
    fig.suptitle('Spike Raster Plot Comparison', fontsize=16)
    
    if len(dynamics) == 1:
        axes = [axes]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(dynamics)))
    
    for i, (name, data) in enumerate(dynamics.items()):
        if data and 'spike_times' in data:
            spike_times = data['spike_times']
            
            y_pos = 0
            for neuron_id, spikes in spike_times.items():
                if spikes:
                    axes[i].scatter(spikes, [y_pos] * len(spikes), 
                                   s=1, c=colors[i], alpha=0.7)
                y_pos += 1
            
            axes[i].set_ylabel('Neuron Index')
            axes[i].set_title(f'{name.replace("_", " ").title()} Network')
            axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    
    # Save raster plot
    raster_file = output_dir / "02_raster_comparison.png"
    plt.savefig(raster_file, dpi=300, bbox_inches='tight')
    print(f"Raster comparison saved to: {raster_file}")

def create_default_mode_network():
    """Create a simplified default mode network architecture."""
    print("\n=== Creating Default Mode Network ===")
    
    from neurons import LIFNeuron, LIFParameters, SynapseFactory
    
    # Define DMN regions
    regions = {
        'PCC': {'neurons': 20, 'params': LIFParameters(tau_m=25.0, v_thresh=-45.0)},  # Posterior Cingulate
        'mPFC': {'neurons': 15, 'params': LIFParameters(tau_m=22.0, v_thresh=-48.0)}, # medial Prefrontal
        'AG': {'neurons': 12, 'params': LIFParameters(tau_m=20.0, v_thresh=-52.0)},    # Angular Gyrus
    }
    
    neurons = {}
    synapses = {}
    
    # Create neurons for each region
    for region, info in regions.items():
        for i in range(info['neurons']):
            neuron_id = f"{region}_{i:03d}"
            neurons[neuron_id] = LIFNeuron(neuron_id, info['params'])
    
    # Intra-regional connections (local)
    for region, info in regions.items():
        n_neurons = info['neurons']
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                if np.random.random() < 0.3:  # 30% connection probability
                    # Bidirectional connections
                    for direction in [(i, j), (j, i)]:
                        syn_id = f"intra_{region}_{direction[0]}_{direction[1]}"
                        synapse = SynapseFactory.create_excitatory_synapse(
                            syn_id,
                            f"{region}_{direction[0]:03d}",
                            f"{region}_{direction[1]:03d}",
                            weight=0.9,
                            delay=2.0
                        )
                        synapses[syn_id] = synapse
                        
                        # Register connections
                        neurons[f"{region}_{direction[0]:03d}"].add_postsynaptic_connection(synapse)
                        neurons[f"{region}_{direction[1]:03d}"].add_presynaptic_connection(synapse)
    
    # Inter-regional connections (long-range)
    inter_connections = [
        ('PCC', 'mPFC', 0.4, 15.0),  # Strong PCC-mPFC connection
        ('PCC', 'AG', 0.3, 12.0),    # PCC-Angular Gyrus
        ('mPFC', 'AG', 0.25, 18.0),  # mPFC-Angular Gyrus
    ]
    
    for region1, region2, prob, delay in inter_connections:
        n1, n2 = regions[region1]['neurons'], regions[region2]['neurons']
        for i in range(n1):
            for j in range(n2):
                if np.random.random() < prob:
                    # Create bidirectional connection
                    for direction in [(region1, i, region2, j), (region2, j, region1, i)]:
                        syn_id = f"inter_{direction[0]}_{direction[1]}_{direction[2]}_{direction[3]}"
                        synapse = SynapseFactory.create_excitatory_synapse(
                            syn_id,
                            f"{direction[0]}_{direction[1]:03d}",
                            f"{direction[2]}_{direction[3]:03d}",
                            weight=1.2,
                            delay=delay
                        )
                        synapses[syn_id] = synapse
                        
                        # Register connections
                        neurons[f"{direction[0]}_{direction[1]:03d}"].add_postsynaptic_connection(synapse)
                        neurons[f"{direction[2]}_{direction[3]:03d}"].add_presynaptic_connection(synapse)
    
    # Create simple network container
    class DMNNetwork:
        def __init__(self, neurons, synapses, regions):
            self.neurons = neurons
            self.synapses = synapses
            self.regions = regions
            
        def update(self, dt):
            # Update synapses
            for synapse in self.synapses.values():
                synapse.update(dt)
            
            # Update neurons
            for neuron in self.neurons.values():
                # Calculate synaptic inputs
                for synapse in neuron.presynaptic_connections:
                    if synapse.synapse_id in self.synapses:
                        current = synapse.calculate_current(
                            neuron.get_membrane_potential()
                        )
                        neuron.add_synaptic_input(current)
                
                neuron.update(dt)
    
    dmn = DMNNetwork(neurons, synapses, regions)
    
    print(f"Created DMN with {len(neurons)} neurons and {len(synapses)} synapses")
    print("Regions:", {r: info['neurons'] for r, info in regions.items()})
    
    return dmn

def main():
    """Main function."""
    print("SDMN Framework - Network Topologies Demonstration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Create and analyze networks
        networks = create_sample_networks()
        analysis = analyze_network_properties(networks)
        dynamics = simulate_network_dynamics(networks)
        
        # Create plots
        plot_network_comparison(networks, analysis, dynamics)
        
        # Create DMN example
        dmn = create_default_mode_network()
        
        print("\n=== Summary ===")
        print("✓ Random networks: Good for general connectivity studies")
        print("✓ Ring networks: Simple, predictable dynamics")
        print("✓ Small-world networks: Balance of local and global connectivity")
        print("✓ Grid networks: Spatial organization, local processing")
        print("✓ DMN networks: Brain-inspired regional architecture")
        
        print("\nKey findings:")
        for name in networks.keys():
            if dynamics[name]:
                print(f"• {name}: {dynamics[name]['total_spikes']} spikes, "
                      f"{dynamics[name]['mean_synchrony']:.3f} synchrony")
        
        print("\nNext steps:")
        print("• Try different connection probabilities")
        print("• Add inhibitory neurons for stability")
        print("• Implement plasticity for learning")
        print("• Run example 03 for probe demonstrations")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
