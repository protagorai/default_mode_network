#!/usr/bin/env python3
"""
Working Neuron Network Demo - SDMN Framework

This example demonstrates actual working neural network simulation 
with guaranteed spiking activity and rich visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from the SDMN package
import sdmn
from sdmn.neurons import LIFNeuron, LIFParameters


def create_working_network(n_neurons=10):
    """Create a simple working network that will definitely spike."""
    print(f"Creating working network with {n_neurons} neurons...")
    
    # Create neurons with parameters that will spike
    neurons = {}
    for i in range(n_neurons):
        params = LIFParameters(
            tau_m=20.0,           # Membrane time constant
            v_rest=-70.0,         # Resting potential
            v_thresh=-50.0,       # Spike threshold (easier to reach)
            v_reset=-80.0,        # Reset potential
            r_mem=10.0,           # Membrane resistance
            refractory_period=2.0  # Refractory period
        )
        neurons[f"neuron_{i:03d}"] = LIFNeuron(f"neuron_{i:03d}", params)
    
    print(f"[OK] Created {len(neurons)} LIF neurons")
    return neurons


def run_working_simulation(neurons, duration=1000.0, dt=0.1):
    """Run a working simulation with guaranteed activity."""
    print(f"Running simulation for {duration} ms...")
    
    steps = int(duration / dt)
    
    # Data storage
    times = []
    all_voltages = {nid: [] for nid in neurons.keys()}
    all_spikes = {nid: [] for nid in neurons.keys()}
    
    # Apply stimulation to ensure activity
    stimulus_neurons = ['neuron_000', 'neuron_002', 'neuron_005']
    
    print(f"Applying stimulation to: {stimulus_neurons}")
    
    # Run simulation manually
    for step in range(steps):
        time = step * dt
        times.append(time)
        
        # Apply time-varying stimulation
        for i, neuron_id in enumerate(stimulus_neurons):
            if neuron_id in neurons:
                # Periodic stimulation with different phases
                base_current = 2.5  # Base current above threshold
                phase_offset = i * np.pi / 3  # Different phases
                stimulus = base_current + 0.5 * np.sin(2 * np.pi * time / 200 + phase_offset)
                
                # Add some randomness
                stimulus += np.random.normal(0, 0.2)
                
                neurons[neuron_id].set_external_input(max(0, stimulus))
        
        # Update all neurons
        for neuron_id, neuron in neurons.items():
            neuron.update(dt)
            
            # Record voltage
            all_voltages[neuron_id].append(neuron.get_membrane_potential())
            
            # Record spikes
            if neuron.has_spiked():
                all_spikes[neuron_id].append(time)
        
        # Progress reporting
        if step % 5000 == 0:
            progress = (step / steps) * 100
            total_spikes = sum(len(spikes) for spikes in all_spikes.values())
            print(f"  Progress: {progress:.0f}%, Total spikes: {total_spikes}")
    
    # Final statistics
    total_spikes = sum(len(spikes) for spikes in all_spikes.values())
    active_neurons = sum(1 for spikes in all_spikes.values() if len(spikes) > 0)
    
    print(f"[OK] Simulation completed!")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Active neurons: {active_neurons}/{len(neurons)}")
    
    if total_spikes > 0:
        mean_rate = total_spikes * 1000 / duration  # Convert to Hz
        print(f"  Mean firing rate: {mean_rate:.1f} Hz")
    
    return times, all_voltages, all_spikes


def create_rich_visualization(times, voltages, spikes, duration=1000.0, dt=0.1):
    """Create rich visualization showing neural activity."""
    print("Creating comprehensive visualization...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('SDMN Framework - Working Neural Network Activity', fontsize=16)
    
    # Convert times to numpy array
    times = np.array(times)
    
    # Plot 1: Voltage traces for first 5 neurons
    neuron_ids = list(voltages.keys())[:5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(neuron_ids)))
    
    for i, (neuron_id, color) in enumerate(zip(neuron_ids, colors)):
        v_trace = np.array(voltages[neuron_id])
        axes[0, 0].plot(times, v_trace, color=color, linewidth=1.5, 
                       label=neuron_id, alpha=0.8)
        
        # Mark spikes
        for spike_time in spikes[neuron_id]:
            axes[0, 0].axvline(x=spike_time, color=color, alpha=0.6, linewidth=0.8)
    
    axes[0, 0].axhline(y=-50, color='red', linestyle='--', alpha=0.7, label='Threshold')
    axes[0, 0].set_ylabel('Membrane Potential (mV)')
    axes[0, 0].set_title('Neural Membrane Potentials')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spike raster plot
    for i, (neuron_id, spike_times) in enumerate(spikes.items()):
        if len(spike_times) > 0:
            axes[0, 1].scatter(spike_times, [i] * len(spike_times), 
                             s=10, c='black', alpha=0.7)
    
    axes[0, 1].set_ylabel('Neuron Index')
    axes[0, 1].set_title('Network Spike Raster')
    axes[0, 1].set_ylim(-0.5, len(voltages) - 0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Population firing rate over time
    bin_size = 50.0  # ms
    time_bins = np.arange(0, times[-1], bin_size)
    pop_rates = []
    
    for t_start in time_bins:
        t_end = t_start + bin_size
        spikes_in_bin = 0
        
        for spike_times in spikes.values():
            spikes_in_bin += sum(1 for t in spike_times if t_start <= t < t_end)
        
        # Convert to Hz (spikes per second)
        rate = spikes_in_bin * 1000 / bin_size / len(voltages)
        pop_rates.append(rate)
    
    axes[1, 0].plot(time_bins, pop_rates, 'darkblue', linewidth=2)
    axes[1, 0].set_ylabel('Population Rate (Hz)')
    axes[1, 0].set_title('Population Firing Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Inter-spike interval distribution
    all_isis = []
    for spike_times in spikes.values():
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            all_isis.extend(isis)
    
    if len(all_isis) > 0:
        axes[1, 1].hist(all_isis, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Inter-Spike Interval (ms)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('ISI Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_isi = np.mean(all_isis)
        std_isi = np.std(all_isis)
        axes[1, 1].axvline(x=mean_isi, color='red', linestyle='--', 
                          label=f'Mean: {mean_isi:.1f}±{std_isi:.1f} ms')
        axes[1, 1].legend()
    
    # Plot 5: Firing rate per neuron
    firing_rates = []
    neuron_labels = []
    
    for neuron_id, spike_times in spikes.items():
        rate = len(spike_times) * 1000 / duration  # Hz
        firing_rates.append(rate)
        neuron_labels.append(neuron_id)
    
    axes[2, 0].bar(range(len(firing_rates)), firing_rates, 
                   color='steelblue', alpha=0.7)
    axes[2, 0].set_xlabel('Neuron Index')
    axes[2, 0].set_ylabel('Firing Rate (Hz)')
    axes[2, 0].set_title('Individual Neuron Firing Rates')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Cross-correlation between neurons (connectivity inference)
    if len(neuron_ids) >= 2 and any(len(voltages[nid]) > 100 for nid in neuron_ids[:2]):
        v1 = np.array(voltages[neuron_ids[0]])
        v2 = np.array(voltages[neuron_ids[1]])
        
        # Compute cross-correlation
        correlation = np.correlate(v1 - np.mean(v1), v2 - np.mean(v2), mode='full')
        lags = np.arange(-len(v2) + 1, len(v1)) * dt
        
        # Focus on reasonable lag range
        max_lag_ms = 50
        valid_indices = np.abs(lags) <= max_lag_ms
        
        axes[2, 1].plot(lags[valid_indices], correlation[valid_indices], 
                       'purple', linewidth=2)
        axes[2, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[2, 1].set_xlabel('Lag (ms)')
        axes[2, 1].set_ylabel('Cross-Correlation')
        axes[2, 1].set_title(f'Cross-Correlation: {neuron_ids[0]} vs {neuron_ids[1]}')
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "working_neuron_demo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Comprehensive plot saved to: {output_file}")
    
    # Show plot if possible
    try:
        plt.show()
    except:
        print("Display not available - plot saved to file")
    
    return output_file


def main():
    """Main demonstration function."""
    print("SDMN Framework - Working Neural Network Demo")
    print("=" * 60)
    print("This demo ensures actual spiking activity and rich visualizations")
    print("")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Create working network
        neurons = create_working_network(n_neurons=10)
        
        # Run simulation
        times, voltages, spikes = run_working_simulation(neurons, duration=1000.0)
        
        # Create visualization
        plot_file = create_rich_visualization(times, voltages, spikes, duration=1000.0, dt=0.1)
        
        # Summary statistics
        total_spikes = sum(len(spike_times) for spike_times in spikes.values())
        active_neurons = sum(1 for spike_times in spikes.values() if len(spike_times) > 0)
        
        print(f"\n=== Final Results ===")
        print(f"[OK] Total spikes generated: {total_spikes}")
        print(f"[OK] Active neurons: {active_neurons}/{len(neurons)}")
        
        if total_spikes > 0:
            mean_rate = total_spikes * 1000 / 1000.0  # Hz
            print(f"[OK] Mean network firing rate: {mean_rate:.1f} Hz")
            
            # ISI analysis
            all_isis = []
            for spike_times in spikes.values():
                if len(spike_times) > 1:
                    all_isis.extend(np.diff(spike_times))
            
            if all_isis:
                print(f"[OK] Mean ISI: {np.mean(all_isis):.1f} ± {np.std(all_isis):.1f} ms")
                print(f"[OK] ISI regularity (CV): {np.std(all_isis)/np.mean(all_isis):.2f}")
        
        print(f"\n[OK] Visualization saved to: {plot_file}")
        print("\nThis demo shows what neural network simulations should produce:")
        print("  • Rich spiking activity from stimulated neurons")
        print("  • Membrane potential traces with clear dynamics")  
        print("  • Population-level firing rate patterns")
        print("  • Inter-spike interval distributions")
        print("  • Cross-correlation between neurons")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
