"""
Example: C. elegans Network Manager

Demonstrates the high-level NetworkManager for easy network construction,
simulation control, and connectivity management.

Shows:
- Easy network building
- Simulation control (run, pause, extend, reset)
- Dynamic connectivity changes
- Data access and visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directory
os.makedirs('output', exist_ok=True)

from sdmn.networks.celegans import CElegansNetwork, SimulationState
from sdmn.synapses import SynapseType


def example1_basic_usage():
    """Example 1: Basic network creation and simulation."""
    print("=" * 70)
    print("Example 1: Basic Network Manager Usage")
    print("=" * 70)
    
    # Create network
    network = CElegansNetwork()
    
    # Add neurons easily
    n1 = network.add_sensory_neuron("AWC")
    n2 = network.add_interneuron("AVA")
    n3 = network.add_motor_neuron("DA01")
    
    # Add connections
    network.add_graded_synapse(n1, n2, weight=2.0, synapse_type=SynapseType.EXCITATORY)
    network.add_graded_synapse(n2, n3, weight=1.5, synapse_type=SynapseType.EXCITATORY)
    network.add_gap_junction(n2, n3, conductance=0.8)
    
    # Print network info
    summary = network.get_connectivity_summary()
    print(f"\nNetwork created:")
    print(f"  Neurons: {summary['n_neurons']}")
    print(f"  Chemical synapses: {summary['n_chemical_synapses']}")
    print(f"  Gap junctions: {summary['n_gap_junctions']}")
    print(f"  By class: {summary['neurons_by_class']}")
    
    # Set stimulation
    network.set_external_current("AWC", 60.0)  # Stimulate sensory neuron
    
    # Run simulation
    network.simulate(duration=500.0, progress=True)
    
    # Get results
    voltages = network.get_current_voltages()
    print(f"\nFinal voltages:")
    for nid, v in voltages.items():
        print(f"  {nid}: {v:.2f} mV")
    
    return network


def example2_simulation_control():
    """Example 2: Simulation control - pause, resume, extend."""
    print("\n" + "=" * 70)
    print("Example 2: Simulation Control")
    print("=" * 70)
    
    # Create simple network
    network = CElegansNetwork()
    n1 = network.add_interneuron("N1")
    n2 = network.add_interneuron("N2")
    network.add_gap_junction(n1, n2, conductance=1.0)
    
    # Stimulate N1
    network.set_external_current("N1", 50.0)
    
    # Run first part
    print("\n[1] Running 200 ms...")
    network.simulate(duration=200.0, progress=False)
    v1 = network.get_neuron("N1").voltage
    v2 = network.get_neuron("N2").voltage
    print(f"  After 200 ms: N1={v1:.2f} mV, N2={v2:.2f} mV")
    
    # Pause
    network.pause()
    print(f"\n[2] Paused at t={network.current_time:.1f} ms")
    
    # Resume and extend
    print("\n[3] Extending simulation by 300 ms...")
    network.resume(duration=300.0)
    v1 = network.get_neuron("N1").voltage
    v2 = network.get_neuron("N2").voltage
    print(f"  After 500 ms total: N1={v1:.2f} mV, N2={v2:.2f} mV")
    
    # Run even more
    print("\n[4] Extending again by 500 ms...")
    network.simulate(duration=500.0, progress=False)
    v1 = network.get_neuron("N1").voltage
    v2 = network.get_neuron("N2").voltage
    print(f"  After 1000 ms total: N1={v1:.2f} mV, N2={v2:.2f} mV")
    print(f"  Total simulation time: {network.current_time:.1f} ms")
    
    # Reset
    print("\n[5] Resetting network...")
    network.reset()
    v1 = network.get_neuron("N1").voltage
    v2 = network.get_neuron("N2").voltage
    print(f"  After reset: N1={v1:.2f} mV, N2={v2:.2f} mV")
    print(f"  Time: {network.current_time:.1f} ms")
    
    return network


def example3_dynamic_connectivity():
    """Example 3: Modify connectivity during experiment."""
    print("\n" + "=" * 70)
    print("Example 3: Dynamic Connectivity Changes")
    print("=" * 70)
    
    # Create network
    network = CElegansNetwork()
    n1 = network.add_sensory_neuron("S1")
    n2 = network.add_interneuron("I1")
    n3 = network.add_motor_neuron("M1")
    
    # Initial connectivity
    syn1 = network.add_graded_synapse(n1, n2, weight=1.0)
    syn2 = network.add_graded_synapse(n2, n3, weight=1.0)
    
    print("\n[1] Initial network: S1 -> I1 -> M1")
    print(f"    Syn1 weight: 1.0 nS, Syn2 weight: 1.0 nS")
    
    # Simulate
    network.set_external_current("S1", 60.0)
    network.simulate(duration=200.0, progress=False)
    
    v_m1_initial = network.get_neuron("M1").voltage
    print(f"    M1 voltage after 200 ms: {v_m1_initial:.2f} mV")
    
    # Modify synaptic weights
    print("\n[2] Increasing synaptic weights to 3.0 nS...")
    network.set_synapse_weight(syn1, 3.0)
    network.set_synapse_weight(syn2, 3.0)
    
    # Continue simulation
    network.simulate(duration=300.0, progress=False)
    v_m1_after = network.get_neuron("M1").voltage
    print(f"    M1 voltage after 500 ms total: {v_m1_after:.2f} mV")
    print(f"    Change: {v_m1_after - v_m1_initial:+.2f} mV")
    
    # Add new connection
    print("\n[3] Adding direct S1 -> M1 connection...")
    syn3 = network.add_graded_synapse(n1, n3, weight=2.5)
    
    # Continue
    network.simulate(duration=200.0, progress=False)
    v_m1_final = network.get_neuron("M1").voltage
    print(f"    M1 voltage after 700 ms total: {v_m1_final:.2f} mV")
    print(f"    Total change: {v_m1_final - v_m1_initial:+.2f} mV")
    
    summary = network.get_connectivity_summary()
    print(f"\n[4] Final network: {summary['n_chemical_synapses']} synapses")
    
    return network


def example4_larger_network():
    """Example 4: Larger network with visualization."""
    print("\n" + "=" * 70)
    print("Example 4: Larger Network Simulation")
    print("=" * 70)
    
    # Create network with 10 neurons
    network = CElegansNetwork()
    
    # Add neurons
    print("\nBuilding network...")
    neurons = []
    for i in range(10):
        if i < 2:
            nid = network.add_sensory_neuron(f"S{i}")
        elif i < 7:
            nid = network.add_interneuron(f"I{i-2}")
        else:
            nid = network.add_motor_neuron(f"M{i-7}")
        neurons.append(nid)
    
    # Add connections (small-world-like)
    # Local connections
    for i in range(len(neurons) - 1):
        network.add_graded_synapse(
            neurons[i], neurons[i+1], 
            weight=np.random.uniform(1.0, 2.5)
        )
    
    # Some long-range connections
    network.add_graded_synapse(neurons[0], neurons[5], weight=2.0)
    network.add_graded_synapse(neurons[1], neurons[7], weight=1.8)
    
    # Gap junctions between interneurons
    network.add_gap_junction(neurons[2], neurons[3], conductance=0.8)
    network.add_gap_junction(neurons[4], neurons[5], conductance=0.8)
    
    summary = network.get_connectivity_summary()
    print(f"  Neurons: {summary['n_neurons']}")
    print(f"  Synapses: {summary['n_chemical_synapses']}")
    print(f"  Gap junctions: {summary['n_gap_junctions']}")
    print(f"  By class: {summary['neurons_by_class']}")
    
    # Stimulate sensory neurons
    network.set_external_currents({
        "S0": 70.0,
        "S1": 60.0
    })
    
    # Simulate
    network.simulate(duration=800.0, progress=True)
    
    # Plot results
    times, voltages = network.get_voltages_array()
    
    if len(times) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, neuron_id in enumerate(sorted(network.neurons.keys())):
            ax.plot(times, voltages[i], label=neuron_id, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Voltage (mV)', fontsize=12)
        ax.set_title('10-Neuron C. elegans Network Activity', 
                     fontsize=14, fontweight='bold')
        ax.legend(ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/07_network_manager_large.png', dpi=300, bbox_inches='tight')
        print(f"\nFigure saved: output/07_network_manager_large.png")
        plt.close()
    
    return network


def main():
    """Run all network manager examples."""
    print("\n" + "=" * 70)
    print("C. ELEGANS NETWORK MANAGER - DEMONSTRATION")
    print("=" * 70)
    print("\nHigh-level interface for easy network management:")
    print("  * Easy neuron and connection creation")
    print("  * Simulation control (run, pause, extend, reset)")
    print("  * Dynamic connectivity modification")
    print("  * Simple data access")
    print("=" * 70)
    
    # Run examples
    net1 = example1_basic_usage()
    net2 = example2_simulation_control()
    net3 = example3_dynamic_connectivity()
    net4 = example4_larger_network()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n[OK] Example 1: Basic usage - easy network creation")
    print("[OK] Example 2: Simulation control - pause/resume/extend/reset")
    print("[OK] Example 3: Dynamic connectivity - modify during simulation")
    print("[OK] Example 4: Larger network - 10 neurons with visualization")
    print("\nThe NetworkManager makes it easy to:")
    print("  • Build networks quickly")
    print("  • Control simulations")
    print("  • Modify connectivity on the fly")
    print("  • Access results easily")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


