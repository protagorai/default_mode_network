#!/usr/bin/env python3
"""
Basic Neuron Demonstration - SDMN Framework Example 01

This example demonstrates the basic usage of individual neuron models,
showing how to create, stimulate, and analyze different types of neurons.

Run with:
    python examples/01_basic_neuron_demo.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from the SDMN package
import sdmn
from sdmn.neurons import LIFNeuron, LIFParameters, HHNeuron, HHParameters, NeuronType

def demo_lif_neuron():
    """Demonstrate Leaky Integrate-and-Fire neuron behavior."""
    print("=== LIF Neuron Demonstration ===")
    
    # Create neuron with custom parameters
    params = LIFParameters(
        tau_m=20.0,           # Membrane time constant (ms)
        v_rest=-70.0,         # Resting potential (mV)
        v_thresh=-50.0,       # Spike threshold (mV)
        v_reset=-80.0,        # Reset potential (mV)
        r_mem=10.0,           # Membrane resistance (MΩ)
        refractory_period=2.0 # Refractory period (ms)
    )
    
    neuron = LIFNeuron("demo_lif", params)
    
    # Simulation parameters
    dt = 0.1  # ms
    duration = 200.0  # ms
    steps = int(duration / dt)
    
    # Data storage
    times = []
    voltages = []
    input_currents = []
    spikes = []
    
    print(f"Simulating LIF neuron for {duration} ms...")
    
    # Run simulation
    for step in range(steps):
        time = step * dt
        
        # Apply step current at t=50ms
        if 50 <= time <= 150:
            input_current = 3.0  # nA - above threshold
        else:
            input_current = 0.0
        
        # Update neuron
        neuron.set_external_input(input_current)
        neuron.update(dt)
        
        # Record data
        times.append(time)
        voltages.append(neuron.get_membrane_potential())
        input_currents.append(input_current)
        
        if neuron.has_spiked():
            spikes.append(time)
    
    # Calculate rheobase
    rheobase = neuron.get_rheobase_current()
    
    print(f"Neuron fired {len(spikes)} spikes")
    print(f"Rheobase current: {rheobase:.2f} nA")
    if spikes:
        isis = np.diff(spikes)
        print(f"Inter-spike intervals: {isis} ms")
        print(f"Mean firing rate: {len(spikes) * 1000 / duration:.1f} Hz")
    
    return times, voltages, input_currents, spikes, rheobase

def demo_hh_neuron():
    """Demonstrate Hodgkin-Huxley neuron behavior."""
    print("\n=== Hodgkin-Huxley Neuron Demonstration ===")
    
    # Create HH neuron
    params = HHParameters()
    neuron = HHNeuron("demo_hh", params)
    
    # Simulation parameters
    dt = 0.01  # Smaller time step for HH model
    duration = 50.0  # ms
    steps = int(duration / dt)
    
    # Data storage
    times = []
    voltages = []
    input_currents = []
    spikes = []
    gating_vars = []
    
    print(f"Simulating HH neuron for {duration} ms...")
    
    # Run simulation
    for step in range(steps):
        time = step * dt
        
        # Apply current pulse
        if 10 <= time <= 40:
            input_current = 10.0  # μA/cm²
        else:
            input_current = 0.0
        
        # Update neuron
        neuron.set_external_input(input_current)
        neuron.update(dt)
        
        # Record data
        times.append(time)
        voltages.append(neuron.get_membrane_potential())
        input_currents.append(input_current)
        
        # Record gating variables
        m, h, n = neuron.get_gating_variables()
        gating_vars.append((m, h, n))
        
        if neuron.has_spiked():
            spikes.append(time)
    
    print(f"HH neuron fired {len(spikes)} spikes")
    
    return times, voltages, input_currents, spikes, gating_vars

def plot_results(lif_data, hh_data):
    """Plot comparison of LIF and HH neuron responses."""
    lif_times, lif_voltages, lif_currents, lif_spikes, rheobase = lif_data
    hh_times, hh_voltages, hh_currents, hh_spikes, hh_gating = hh_data
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Neuron Model Comparison: LIF vs Hodgkin-Huxley', fontsize=16)
    
    # LIF Neuron plots
    # Voltage trace
    axes[0, 0].plot(lif_times, lif_voltages, 'b-', linewidth=2, label='Membrane Potential')
    axes[0, 0].axhline(y=-50, color='r', linestyle='--', alpha=0.7, label='Threshold')
    axes[0, 0].axhline(y=-70, color='g', linestyle='--', alpha=0.7, label='Resting')
    
    # Mark spikes
    for spike_time in lif_spikes:
        axes[0, 0].axvline(x=spike_time, color='r', alpha=0.5)
    
    axes[0, 0].set_ylabel('Voltage (mV)')
    axes[0, 0].set_title('LIF Neuron - Membrane Potential')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Input current
    axes[1, 0].plot(lif_times, lif_currents, 'orange', linewidth=2)
    axes[1, 0].axhline(y=rheobase, color='r', linestyle=':', label=f'Rheobase ({rheobase:.1f} nA)')
    axes[1, 0].set_ylabel('Current (nA)')
    axes[1, 0].set_title('Input Current')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F-I curve for LIF
    currents = np.linspace(0, 6, 20)
    firing_rates = []
    
    for current in currents:
        # Quick simulation for each current
        test_neuron = LIFNeuron("test", LIFParameters())
        spike_count = 0
        test_duration = 1000  # ms
        
        for t in range(int(test_duration / 0.1)):
            test_neuron.set_external_input(current)
            test_neuron.update(0.1)
            if test_neuron.has_spiked():
                spike_count += 1
        
        firing_rates.append(spike_count)  # Hz (spikes per second)
    
    axes[2, 0].plot(currents, firing_rates, 'b-', linewidth=2, label='Simulated')
    
    # Analytical F-I curve
    analytical_rates = []
    lif_params = LIFParameters()
    temp_neuron = LIFNeuron("temp", lif_params)
    for current in currents:
        rate = temp_neuron.get_analytical_firing_rate(current)
        analytical_rates.append(rate)
    
    axes[2, 0].plot(currents, analytical_rates, 'r--', linewidth=2, label='Analytical')
    axes[2, 0].axvline(x=rheobase, color='orange', linestyle=':', alpha=0.7)
    axes[2, 0].set_xlabel('Input Current (nA)')
    axes[2, 0].set_ylabel('Firing Rate (Hz)')
    axes[2, 0].set_title('F-I Curve')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # HH Neuron plots
    # Voltage trace
    axes[0, 1].plot(hh_times, hh_voltages, 'purple', linewidth=2)
    
    # Mark spikes
    for spike_time in hh_spikes:
        axes[0, 1].axvline(x=spike_time, color='r', alpha=0.5)
    
    axes[0, 1].set_ylabel('Voltage (mV)')
    axes[0, 1].set_title('Hodgkin-Huxley Neuron - Action Potential')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Input current
    axes[1, 1].plot(hh_times, hh_currents, 'orange', linewidth=2)
    axes[1, 1].set_ylabel('Current (μA/cm²)')
    axes[1, 1].set_title('Input Current')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Gating variables
    if hh_gating:
        m_vals = [g[0] for g in hh_gating]
        h_vals = [g[1] for g in hh_gating]
        n_vals = [g[2] for g in hh_gating]
        
        axes[2, 1].plot(hh_times, m_vals, 'r-', linewidth=2, label='m (Na+ activation)')
        axes[2, 1].plot(hh_times, h_vals, 'b-', linewidth=2, label='h (Na+ inactivation)')
        axes[2, 1].plot(hh_times, n_vals, 'g-', linewidth=2, label='n (K+ activation)')
    
    axes[2, 1].set_xlabel('Time (ms)')
    axes[2, 1].set_ylabel('Gating Variable')
    axes[2, 1].set_title('Gating Variables')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "01_neuron_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    try:
        plt.show()
    except:
        print("Could not display plot (running in non-interactive mode)")

def main():
    """Main function."""
    print("SDMN Framework - Basic Neuron Demonstration")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run demonstrations
        lif_data = demo_lif_neuron()
        hh_data = demo_hh_neuron()
        
        # Create plots
        plot_results(lif_data, hh_data)
        
        print("\n=== Summary ===")
        print("[OK] LIF neuron: Simple, computationally efficient")
        print("[OK] HH neuron: Biophysically detailed, shows action potential shape")
        print("[OK] Both models show spiking behavior with current input")
        print("[OK] LIF suitable for large networks, HH for detailed biophysics")
        
        print("\nNext steps:")
        print("• Try adjusting neuron parameters")
        print("• Test different input patterns")
        print("• Run example 02 for network simulations")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
