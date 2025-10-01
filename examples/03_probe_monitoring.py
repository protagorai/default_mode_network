#!/usr/bin/env python3
"""
Probe Monitoring Demonstration - SDMN Framework Example 03

This example demonstrates the comprehensive probe system for monitoring
neural activity, including voltage traces, spike detection, population
dynamics, and synthetic EEG generation.

Run with:
    python examples/03_probe_monitoring.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from the SDMN package
import sdmn
from sdmn.neurons import LIFNeuron, LIFParameters, SynapseFactory
from sdmn.probes import VoltageProbe, SpikeProbe, PopulationActivityProbe, LFPProbe
from sdmn.core import SimulationEngine, SimulationConfig

def create_monitored_network():
    """Create a small network specifically designed for monitoring demonstration."""
    print("Creating monitored neural network...")
    
    # Create diverse neurons
    neurons = {}
    neuron_configs = [
        ("fast", LIFParameters(tau_m=15.0, v_thresh=-45.0, r_mem=8.0)),
        ("medium", LIFParameters(tau_m=20.0, v_thresh=-50.0, r_mem=10.0)),
        ("slow", LIFParameters(tau_m=30.0, v_thresh=-55.0, r_mem=12.0))
    ]
    
    # Create 5 neurons of each type
    for neuron_type, params in neuron_configs:
        for i in range(5):
            neuron_id = f"{neuron_type}_{i:03d}"
            # Add some variability
            varied_params = LIFParameters(
                tau_m=params.tau_m + np.random.normal(0, 2.0),
                v_thresh=params.v_thresh + np.random.normal(0, 1.0),
                r_mem=params.r_mem + np.random.normal(0, 0.5),
                refractory_period=2.0
            )
            neurons[neuron_id] = LIFNeuron(neuron_id, varied_params)
    
    # Create connections
    synapses = {}
    neuron_ids = list(neurons.keys())
    
    # Connect neurons with some probability
    for i, pre_id in enumerate(neuron_ids):
        for j, post_id in enumerate(neuron_ids):
            if i != j and np.random.random() < 0.15:  # 15% connection probability
                syn_id = f"syn_{pre_id}_to_{post_id}"
                
                # Mostly excitatory with some inhibitory
                if np.random.random() < 0.8:
                    synapse = SynapseFactory.create_excitatory_synapse(
                        syn_id, pre_id, post_id,
                        weight=np.random.uniform(0.8, 1.5),
                        delay=np.random.uniform(1.0, 8.0)
                    )
                else:
                    synapse = SynapseFactory.create_inhibitory_synapse(
                        syn_id, pre_id, post_id,
                        weight=np.random.uniform(0.5, 1.2),
                        delay=np.random.uniform(1.0, 5.0)
                    )
                
                synapses[syn_id] = synapse
                
                # Register connections
                neurons[pre_id].add_postsynaptic_connection(synapse)
                neurons[post_id].add_presynaptic_connection(synapse)
    
    print(f"Created network with {len(neurons)} neurons and {len(synapses)} synapses")
    
    # Simple network class
    class MonitoredNetwork:
        def __init__(self, neurons, synapses):
            self.neurons = neurons
            self.synapses = synapses
        
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
    
    return MonitoredNetwork(neurons, synapses)

def setup_comprehensive_probes(network):
    """Setup various types of probes to monitor the network."""
    print("Setting up comprehensive probe system...")
    
    probes = {}
    neuron_ids = list(network.neurons.keys())
    
    # 1. Voltage Probe - High resolution membrane potential recording
    voltage_probe = VoltageProbe(
        probe_id="voltage_monitor",
        target_neurons=neuron_ids[:6],  # Monitor first 6 neurons
        sampling_interval=0.5,          # High resolution: sample every 0.5 ms
        enable_filtering=True,
        filter_cutoff=300.0            # 300 Hz low-pass filter
    )
    
    # Register neurons
    for neuron_id in voltage_probe.target_ids:
        voltage_probe.register_neuron_object(neuron_id, network.neurons[neuron_id])
    
    probes['voltage'] = voltage_probe
    
    # 2. Spike Probe - Precise spike timing detection
    spike_probe = SpikeProbe(
        probe_id="spike_monitor",
        target_neurons=neuron_ids,
        detection_threshold=-35.0,
        min_spike_interval=0.8,        # Minimum time between spikes
        record_waveforms=False
    )
    
    # Register neurons
    for neuron_id, neuron in network.neurons.items():
        spike_probe.register_neuron_object(neuron_id, neuron)
    
    probes['spike'] = spike_probe
    
    # 3. Population Activity Probe - Synthetic brain waves
    population_probe = PopulationActivityProbe(
        probe_id="population_monitor",
        target_population="monitored_network",
        target_neurons=neuron_ids,
        bin_size=5.0,                  # 5 ms bins
        sliding_window=100.0,          # 100 ms window for synchrony
        record_synchrony=True
    )
    
    population_probe.register_neuron_objects(network.neurons)
    probes['population'] = population_probe
    
    # 4. LFP Probe - Local Field Potential simulation
    # Create random positions for neurons
    neuron_positions = {}
    for neuron_id in neuron_ids:
        # Random 3D positions within 1000μm radius
        pos = np.random.randn(3) * 300  # Standard deviation of 300μm
        neuron_positions[neuron_id] = tuple(pos)
    
    lfp_probe = LFPProbe(
        probe_id="lfp_monitor",
        target_neurons=neuron_ids,
        probe_position=(0.0, 0.0, 0.0),  # Probe at center
        sampling_interval=2.0,            # 2 ms sampling
        distance_weights=True,
        max_distance=800.0               # 800μm maximum distance
    )
    
    lfp_probe.register_neuron_objects(network.neurons, neuron_positions)
    probes['lfp'] = lfp_probe
    
    print(f"Setup {len(probes)} different probe types")
    return probes

def run_monitored_simulation(network, probes):
    """Run simulation with comprehensive monitoring."""
    print("Running monitored simulation...")
    
    # Create simulation
    config = SimulationConfig(
        dt=0.1,
        max_time=3000.0,  # 3 seconds
        enable_logging=False,
        checkpoint_interval=10000
    )
    
    engine = SimulationEngine(config)
    engine.add_network("monitored_network", network)
    
    # Add probes to engine
    for probe_name, probe in probes.items():
        engine.add_probe(probe_name, probe)
    
    # Create stimulation pattern
    neuron_ids = list(network.neurons.keys())
    
    # Select neurons for different stimulation patterns
    fast_neurons = [nid for nid in neuron_ids if nid.startswith('fast')]
    medium_neurons = [nid for nid in neuron_ids if nid.startswith('medium')]
    slow_neurons = [nid for nid in neuron_ids if nid.startswith('slow')]
    
    def complex_stimulus(step, time):
        """Complex stimulation pattern to generate interesting activity."""
        
        # Periodic stimulation of fast neurons (high frequency)
        if step % 500 == 0:  # Every 50 ms
            for neuron_id in fast_neurons[:2]:
                current = np.random.normal(2.5, 0.3)
                network.neurons[neuron_id].set_external_input(current)
        
        # Less frequent stimulation of medium neurons
        if step % 1000 == 0:  # Every 100 ms
            for neuron_id in medium_neurons[:2]:
                current = np.random.normal(2.0, 0.4)
                network.neurons[neuron_id].set_external_input(current)
        
        # Occasional stimulation of slow neurons
        if step % 2000 == 0:  # Every 200 ms
            for neuron_id in slow_neurons[:1]:
                current = np.random.normal(1.8, 0.2)
                network.neurons[neuron_id].set_external_input(current)
        
        # Background noise for all neurons
        if step % 100 == 0:  # Every 10 ms
            for neuron_id in neuron_ids:
                noise = np.random.normal(0.0, 0.05)  # Small noise
                current_input = network.neurons[neuron_id].external_input
                network.neurons[neuron_id].set_external_input(current_input + noise)
    
    engine.register_step_callback(complex_stimulus)
    
    # Start all probes
    for probe in probes.values():
        probe.start_recording()
    
    # Run simulation
    results = engine.run()
    
    # Stop recording
    for probe in probes.values():
        probe.stop_recording()
    
    return results

def analyze_probe_data(probes):
    """Comprehensive analysis of probe data."""
    print("\n=== Probe Data Analysis ===")
    
    analysis = {}
    
    # 1. Voltage Analysis
    voltage_probe = probes['voltage']
    voltage_traces = voltage_probe.get_voltage_traces()
    voltage_stats = voltage_probe.get_voltage_statistics()
    
    print("\nVOLTAGE ANALYSIS:")
    for neuron_id, stats in voltage_stats.items():
        print(f"  {neuron_id}: {stats['mean']:.1f}±{stats['std']:.1f} mV, "
              f"range: {stats['range']:.1f} mV")
    
    analysis['voltage'] = {
        'traces': voltage_traces,
        'statistics': voltage_stats
    }
    
    # 2. Spike Analysis
    spike_probe = probes['spike']
    spike_times = spike_probe.get_spike_times()
    spike_counts = spike_probe.get_spike_counts()
    firing_rates = spike_probe.calculate_firing_rates(time_window=1000.0)  # 1 second window
    isi_stats = spike_probe.calculate_isi_statistics()
    
    print("\nSPIKE ANALYSIS:")
    total_spikes = sum(spike_counts.values())
    active_neurons = len([n for n, count in spike_counts.items() if count > 0])
    print(f"  Total spikes: {total_spikes}")
    print(f"  Active neurons: {active_neurons}/{len(spike_times)}")
    
    print("  Firing rates (Hz):")
    for neuron_id, rate in list(firing_rates.items())[:5]:  # Show first 5
        print(f"    {neuron_id}: {rate:.2f}")
    
    analysis['spike'] = {
        'spike_times': spike_times,
        'firing_rates': firing_rates,
        'isi_statistics': isi_stats
    }
    
    # 3. Population Analysis
    population_probe = probes['population']
    times, rates = population_probe.get_population_rate_trace()
    times_sync, synchrony = population_probe.get_synchrony_trace()
    pop_stats = population_probe.get_population_statistics()
    
    print("\nPOPULATION ANALYSIS:")
    print(f"  Mean population rate: {pop_stats['mean_rate']:.2f} Hz")
    print(f"  Max population rate: {pop_stats['max_rate']:.2f} Hz")
    if 'mean_synchrony' in pop_stats:
        print(f"  Mean synchrony: {pop_stats['mean_synchrony']:.3f}")
    
    analysis['population'] = {
        'rates': (times, rates),
        'synchrony': (times_sync, synchrony),
        'statistics': pop_stats
    }
    
    # 4. LFP Analysis
    lfp_probe = probes['lfp']
    lfp_times, lfp_signal = lfp_probe.get_lfp_trace()
    
    if len(lfp_signal) > 100:  # Only analyze if we have enough data
        # Frequency analysis
        try:
            freqs, power = lfp_probe.calculate_lfp_power_spectrum()
            
            print("\nLFP ANALYSIS:")
            print(f"  Signal length: {len(lfp_signal)} samples")
            print(f"  Signal range: {np.min(lfp_signal):.2f} to {np.max(lfp_signal):.2f} mV")
            
            if len(freqs) > 0:
                # Find dominant frequency
                dominant_freq_idx = np.argmax(power[1:]) + 1  # Skip DC component
                dominant_freq = freqs[dominant_freq_idx]
                print(f"  Dominant frequency: {dominant_freq:.1f} Hz")
            
            analysis['lfp'] = {
                'signal': (lfp_times, lfp_signal),
                'spectrum': (freqs, power) if len(freqs) > 0 else None
            }
        except:
            print("\nLFP ANALYSIS: Could not compute frequency spectrum")
            analysis['lfp'] = {
                'signal': (lfp_times, lfp_signal),
                'spectrum': None
            }
    
    return analysis

def plot_comprehensive_results(analysis):
    """Create comprehensive visualization of all probe data."""
    print("\nGenerating comprehensive plots...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create large figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Neural Activity Monitoring', fontsize=16)
    
    # 1. Voltage traces (synthetic EEG)
    if 'voltage' in analysis:
        voltage_traces = analysis['voltage']['traces']
        colors = plt.cm.viridis(np.linspace(0, 1, len(voltage_traces)))
        
        for i, (neuron_id, trace) in enumerate(list(voltage_traces.items())[:4]):
            if len(trace['time']) > 0:
                offset = i * 25  # Offset for visualization
                axes[0, 0].plot(trace['time'], trace['voltage'] + offset, 
                               color=colors[i], linewidth=1, label=neuron_id)
        
        axes[0, 0].set_ylabel('Membrane Potential (mV)')
        axes[0, 0].set_title('Voltage Traces (Synthetic EEG)')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spike raster
    if 'spike' in analysis:
        spike_times = analysis['spike']['spike_times']
        neuron_indices = {nid: i for i, nid in enumerate(spike_times.keys())}
        
        for neuron_id, spikes in spike_times.items():
            if spikes:
                y_pos = neuron_indices[neuron_id]
                axes[0, 1].scatter(spikes, [y_pos] * len(spikes), 
                                  s=1, c='black', alpha=0.6)
        
        axes[0, 1].set_ylabel('Neuron Index')
        axes[0, 1].set_title('Spike Raster Plot')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Firing rates
    if 'spike' in analysis:
        firing_rates = analysis['spike']['firing_rates']
        neuron_ids = list(firing_rates.keys())
        rates = list(firing_rates.values())
        
        axes[0, 2].bar(range(len(rates)), rates, alpha=0.7, color='orange')
        axes[0, 2].set_ylabel('Firing Rate (Hz)')
        axes[0, 2].set_title('Individual Firing Rates')
        axes[0, 2].set_xticks(range(0, len(neuron_ids), 3))
        axes[0, 2].set_xticklabels([neuron_ids[i][:8] for i in range(0, len(neuron_ids), 3)], 
                                  rotation=45, fontsize='small')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Population rate (synthetic brain wave)
    if 'population' in analysis:
        times, rates = analysis['population']['rates']
        axes[1, 0].plot(times, rates, 'b-', linewidth=2, label='Population Rate')
        axes[1, 0].fill_between(times, rates, alpha=0.3)
        axes[1, 0].set_ylabel('Population Rate (Hz)')
        axes[1, 0].set_title('Synthetic Brain Wave')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # 5. Synchrony
    if 'population' in analysis:
        times_sync, synchrony = analysis['population']['synchrony']
        axes[1, 1].plot(times_sync, synchrony, 'r-', linewidth=2)
        axes[1, 1].fill_between(times_sync, synchrony, alpha=0.3, color='red')
        axes[1, 1].set_ylabel('Synchrony Index')
        axes[1, 1].set_title('Network Synchronization')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. LFP signal
    if 'lfp' in analysis and analysis['lfp']['signal']:
        lfp_times, lfp_signal = analysis['lfp']['signal']
        axes[1, 2].plot(lfp_times, lfp_signal, 'purple', linewidth=1.5)
        axes[1, 2].set_ylabel('LFP (mV)')
        axes[1, 2].set_title('Local Field Potential')
        axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Voltage statistics
    if 'voltage' in analysis:
        voltage_stats = analysis['voltage']['statistics']
        neuron_names = list(voltage_stats.keys())
        means = [stats['mean'] for stats in voltage_stats.values()]
        stds = [stats['std'] for stats in voltage_stats.values()]
        
        axes[2, 0].bar(range(len(means)), means, yerr=stds, 
                      alpha=0.7, color='green', capsize=5)
        axes[2, 0].set_ylabel('Mean Voltage (mV)')
        axes[2, 0].set_title('Voltage Statistics')
        axes[2, 0].set_xticks(range(len(neuron_names)))
        axes[2, 0].set_xticklabels([n[:8] for n in neuron_names], 
                                  rotation=45, fontsize='small')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 8. ISI distribution
    if 'spike' in analysis:
        isi_stats = analysis['spike']['isi_statistics']
        # Get ISIs from one active neuron
        active_neuron = None
        for neuron_id, stats in isi_stats.items():
            if stats['count'] > 5:  # At least 5 ISIs
                active_neuron = neuron_id
                break
        
        if active_neuron:
            # Calculate ISIs for plotting
            spike_times_neuron = analysis['spike']['spike_times'][active_neuron]
            if len(spike_times_neuron) > 1:
                isis = np.diff(spike_times_neuron)
                axes[2, 1].hist(isis, bins=15, alpha=0.7, color='cyan')
                axes[2, 1].axvline(np.mean(isis), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(isis):.1f}ms')
                axes[2, 1].set_xlabel('ISI (ms)')
                axes[2, 1].set_ylabel('Count')
                axes[2, 1].set_title(f'ISI Distribution ({active_neuron[:8]})')
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)
    
    # 9. LFP frequency spectrum
    if 'lfp' in analysis and analysis['lfp']['spectrum']:
        freqs, power = analysis['lfp']['spectrum']
        axes[2, 2].semilogy(freqs, power, 'purple', linewidth=2)
        axes[2, 2].set_xlabel('Frequency (Hz)')
        axes[2, 2].set_ylabel('Power')
        axes[2, 2].set_title('LFP Power Spectrum')
        axes[2, 2].set_xlim(0, 100)  # Focus on 0-100 Hz
        axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comprehensive plot
    output_file = output_dir / "03_comprehensive_monitoring.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {output_file}")
    
    try:
        plt.show()
    except:
        print("Could not display plot (running in non-interactive mode)")

def main():
    """Main function."""
    print("SDMN Framework - Probe Monitoring Demonstration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Create network
        network = create_monitored_network()
        
        # Setup probes
        probes = setup_comprehensive_probes(network)
        
        # Run simulation
        results = run_monitored_simulation(network, probes)
        
        if results.success:
            # Analyze data
            analysis = analyze_probe_data(probes)
            
            # Create visualizations
            plot_comprehensive_results(analysis)
            
            print("\n=== Summary ===")
            print("✓ Voltage probes: Record membrane potentials (synthetic EEG)")
            print("✓ Spike probes: Detect and analyze action potentials")
            print("✓ Population probes: Monitor collective network activity")
            print("✓ LFP probes: Simulate local field potential recordings")
            
            print("\nKey capabilities demonstrated:")
            print("• High-resolution voltage monitoring")
            print("• Precise spike timing detection")
            print("• Population-level brain wave generation")
            print("• Frequency domain analysis")
            print("• Network synchronization measurement")
            print("• Local field potential simulation")
            
            print("\nNext steps:")
            print("• Experiment with different probe configurations")
            print("• Try real-time visualization capabilities")
            print("• Implement custom probe types")
            print("• Run example 04 for advanced network analysis")
        
        else:
            print(f"Simulation failed: {results.error_message}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
