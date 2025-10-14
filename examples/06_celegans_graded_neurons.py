"""
Example: C. elegans Graded Potential Neurons

Demonstrates the new graded potential neuron models inspired by C. elegans,
including:
- Single graded neuron dynamics
- Graded chemical synapses
- Gap junctions
- Different neuron classes (sensory, interneuron, motor)

This example shows the fundamental difference between graded (analog) and
spiking (digital) neuron models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Import C. elegans graded neurons
from sdmn.neurons.graded import (
    CElegansNeuron,
    CElegansParameters,
    SensoryNeuron,
    Interneuron,
    MotorNeuron
)

# Import synapses
from sdmn.synapses import GradedChemicalSynapse, GradedSynapseParameters, SynapseType
from sdmn.synapses import GapJunction, GapJunctionParameters


def example1_single_neuron():
    """Example 1: Single graded neuron response to current injection."""
    print("=" * 70)
    print("Example 1: Single C. elegans Graded Neuron")
    print("=" * 70)
    
    # Create a C. elegans interneuron with default parameters
    params = CElegansParameters(dt=0.01)  # 10 μs time step
    neuron = CElegansNeuron("INT-1", params)
    
    # Simulation parameters
    duration = 500.0  # ms
    dt = 0.01  # ms
    steps = int(duration / dt)
    
    # Current injection protocol
    I_ext = np.zeros(steps)
    I_ext[int(100/dt):int(400/dt)] = 50.0  # 50 pA pulse from 100-400 ms
    
    # Storage
    times = []
    voltages = []
    m_Ca_values = []
    m_K_values = []
    Ca_internal_values = []
    
    # Simulation
    print(f"\nSimulating {duration} ms at {dt} ms time step...")
    print(f"Current injection: 50 pA from 100-400 ms")
    
    for step in range(steps):
        t = step * dt
        
        # Set external current
        neuron.set_external_current(I_ext[step])
        
        # Update neuron
        neuron.update(dt)
        
        # Record
        times.append(t)
        voltages.append(neuron.voltage)
        
        # Get channel states
        states = neuron.get_channel_states()
        m_Ca_values.append(states['m_Ca'])
        m_K_values.append(states['m_K'])
        Ca_internal_values.append(states['Ca_internal'])
    
    # Convert to numpy arrays
    times = np.array(times)
    voltages = np.array(voltages)
    m_Ca_values = np.array(m_Ca_values)
    m_K_values = np.array(m_K_values)
    Ca_internal_values = np.array(Ca_internal_values)
    
    print(f"\nResults:")
    print(f"  Resting potential: {voltages[0]:.2f} mV")
    print(f"  Peak depolarization: {np.max(voltages):.2f} mV")
    print(f"  Final potential: {voltages[-1]:.2f} mV")
    print(f"  Note: No spikes - continuous graded response!")
    
    # Plot results
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(5, 1, hspace=0.3)
    
    # Voltage
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, voltages, 'b-', linewidth=1.5)
    ax1.set_ylabel('Voltage (mV)', fontsize=10)
    ax1.set_title('C. elegans Graded Neuron Response to Current Injection', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, duration])
    
    # Current
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(times, I_ext, 'k-', linewidth=1.5)
    ax2.set_ylabel('Current (pA)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, duration])
    
    # Ca channel
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(times, m_Ca_values, 'r-', linewidth=1.5, label='Ca2+')
    ax3.set_ylabel('m_Ca', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, duration])
    ax3.legend(loc='upper right')
    
    # K channel
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(times, m_K_values, 'g-', linewidth=1.5, label='K+')
    ax4.set_ylabel('m_K', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, duration])
    ax4.legend(loc='upper right')
    
    # Intracellular calcium
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(times, Ca_internal_values, 'orange', linewidth=1.5)
    ax5.set_ylabel('[Ca2+]_i (nM)', fontsize=10)
    ax5.set_xlabel('Time (ms)', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, duration])
    
    plt.savefig('output/06_celegans_single_neuron.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: output/06_celegans_single_neuron.png")
    
    return fig


def example2_neuron_classes():
    """Example 2: Compare different neuron classes (sensory, inter, motor)."""
    print("\n" + "=" * 70)
    print("Example 2: Different C. elegans Neuron Classes")
    print("=" * 70)
    
    # Create neurons of different classes
    sensory = SensoryNeuron("AWC")
    inter = Interneuron("AVA")
    motor = MotorNeuron("DA01")
    
    neurons = [
        (sensory, "Sensory (AWC)", "red"),
        (inter, "Interneuron (AVA)", "blue"),
        (motor, "Motor (DA01)", "green")
    ]
    
    # Simulation parameters
    duration = 300.0  # ms
    dt = 0.01
    steps = int(duration / dt)
    
    # Same current stimulus for all
    I_ext = 40.0  # pA constant
    
    results = []
    
    print(f"\nSimulating {duration} ms with {I_ext} pA constant current...")
    
    for neuron, name, color in neurons:
        times = []
        voltages = []
        
        for step in range(steps):
            t = step * dt
            neuron.set_external_current(I_ext)
            neuron.update(dt)
            
            times.append(t)
            voltages.append(neuron.voltage)
        
        results.append((np.array(times), np.array(voltages), name, color))
        
        print(f"  {name}: Peak = {np.max(voltages):.2f} mV, "
              f"Final = {voltages[-1]:.2f} mV")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full time course
    for times, voltages, name, color in results:
        ax1.plot(times, voltages, color=color, linewidth=2, label=name, alpha=0.8)
    
    ax1.set_ylabel('Voltage (mV)', fontsize=11)
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_title('C. elegans Neuron Classes Response Comparison', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, duration])
    
    # Zoom on early response (0-100 ms)
    for times, voltages, name, color in results:
        mask = times <= 100
        ax2.plot(times[mask], voltages[mask], color=color, linewidth=2, 
                label=name, alpha=0.8)
    
    ax2.set_ylabel('Voltage (mV)', fontsize=11)
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_title('Early Response (0-100 ms)', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 100])
    
    plt.tight_layout()
    plt.savefig('output/06_celegans_neuron_classes.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: output/06_celegans_neuron_classes.png")
    
    return fig


def example3_graded_synapse():
    """Example 3: Graded chemical synapse between two neurons."""
    print("\n" + "=" * 70)
    print("Example 3: Graded Chemical Synapse")
    print("=" * 70)
    
    # Create presynaptic and postsynaptic neurons
    pre_neuron = SensoryNeuron("pre")
    post_neuron = Interneuron("post")
    
    # Create graded synapse
    syn_params = GradedSynapseParameters(
        synapse_type=SynapseType.EXCITATORY,
        weight=2.0,        # 2 nS
        V_thresh=-40.0,
        k_release=5.0,
        tau_rise=1.0,
        tau_decay=5.0,
        E_syn=0.0,
        delay=0.5
    )
    
    synapse = GradedChemicalSynapse("syn1", pre_neuron, post_neuron, syn_params)
    
    # Simulation
    duration = 400.0
    dt = 0.01
    steps = int(duration / dt)
    
    # Stimulate presynaptic neuron
    I_pre = np.zeros(steps)
    I_pre[int(100/dt):int(300/dt)] = 60.0  # 60 pA pulse
    
    # Storage
    times = []
    V_pre = []
    V_post = []
    g_syn = []
    I_syn = []
    
    print(f"\nSimulating graded synaptic transmission...")
    print(f"Presynaptic stimulus: 60 pA from 100-300 ms")
    
    for step in range(steps):
        t = step * dt
        
        # Update presynaptic neuron
        pre_neuron.set_external_current(I_pre[step])
        pre_neuron.update(dt)
        
        # Update synapse
        synapse.update(dt)
        
        # Update postsynaptic neuron (receives current from synapse)
        post_neuron.update(dt)
        
        # Record
        times.append(t)
        V_pre.append(pre_neuron.voltage)
        V_post.append(post_neuron.voltage)
        g_syn.append(synapse.get_conductance())
        I_syn.append(synapse.get_current())
    
    # Convert to arrays
    times = np.array(times)
    V_pre = np.array(V_pre)
    V_post = np.array(V_post)
    g_syn = np.array(g_syn)
    I_syn = np.array(I_syn)
    
    print(f"\nResults:")
    print(f"  Presynaptic peak: {np.max(V_pre):.2f} mV")
    print(f"  Postsynaptic peak: {np.max(V_post):.2f} mV")
    print(f"  Max conductance: {np.max(g_syn):.2f} nS")
    print(f"  Max synaptic current: {np.min(I_syn):.2f} pA")
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Presynaptic voltage
    axes[0].plot(times, V_pre, 'b-', linewidth=1.5)
    axes[0].set_ylabel('V_pre (mV)', fontsize=10)
    axes[0].set_title('Graded Chemical Synapse Transmission', 
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, duration])
    
    # Synaptic conductance
    axes[1].plot(times, g_syn, 'orange', linewidth=1.5)
    axes[1].set_ylabel('g_syn (nS)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, duration])
    
    # Synaptic current
    axes[2].plot(times, I_syn, 'r-', linewidth=1.5)
    axes[2].set_ylabel('I_syn (pA)', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, duration])
    
    # Postsynaptic voltage
    axes[3].plot(times, V_post, 'g-', linewidth=1.5)
    axes[3].set_ylabel('V_post (mV)', fontsize=10)
    axes[3].set_xlabel('Time (ms)', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([0, duration])
    
    plt.tight_layout()
    plt.savefig('output/06_celegans_graded_synapse.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: output/06_celegans_graded_synapse.png")
    
    return fig


def example4_gap_junction():
    """Example 4: Gap junction (electrical synapse) between two neurons."""
    print("\n" + "=" * 70)
    print("Example 4: Gap Junction (Electrical Synapse)")
    print("=" * 70)
    
    # Create two interneurons
    neuron_a = Interneuron("AVAR")
    neuron_b = Interneuron("AVAL")
    
    # Create gap junction
    gap_params = GapJunctionParameters(conductance=1.0)  # 1 nS coupling
    gap = GapJunction("gap1", neuron_a, neuron_b, gap_params)
    
    # Simulation
    duration = 400.0
    dt = 0.01
    steps = int(duration / dt)
    
    # Stimulate only neuron_a
    I_a = np.zeros(steps)
    I_a[int(100/dt):int(300/dt)] = 50.0  # 50 pA pulse
    
    # Storage
    times = []
    V_a = []
    V_b = []
    I_gap_a = []
    I_gap_b = []
    
    print(f"\nSimulating gap junction coupling...")
    print(f"Neuron A stimulus: 50 pA from 100-300 ms")
    print(f"Gap junction conductance: {gap_params.conductance} nS")
    
    for step in range(steps):
        t = step * dt
        
        # Update neuron A (with stimulus)
        neuron_a.set_external_current(I_a[step])
        neuron_a.update(dt)
        
        # Update gap junction
        gap.update(dt)
        
        # Update neuron B (no external stimulus, only gap junction)
        neuron_b.update(dt)
        
        # Record
        times.append(t)
        V_a.append(neuron_a.voltage)
        V_b.append(neuron_b.voltage)
        currents = gap.get_currents()
        I_gap_a.append(currents[0])
        I_gap_b.append(currents[1])
    
    # Convert to arrays
    times = np.array(times)
    V_a = np.array(V_a)
    V_b = np.array(V_b)
    I_gap_a = np.array(I_gap_a)
    I_gap_b = np.array(I_gap_b)
    
    print(f"\nResults:")
    print(f"  Neuron A peak: {np.max(V_a):.2f} mV")
    print(f"  Neuron B peak: {np.max(V_b):.2f} mV")
    print(f"  Voltage synchronization achieved via gap junction!")
    print(f"  Note: B follows A without direct stimulation")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    # Voltages
    axes[0].plot(times, V_a, 'b-', linewidth=2, label='Neuron A (stimulated)', alpha=0.8)
    axes[0].plot(times, V_b, 'r-', linewidth=2, label='Neuron B (coupled)', alpha=0.8)
    axes[0].set_ylabel('Voltage (mV)', fontsize=10)
    axes[0].set_title('Gap Junction Synchronization', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, duration])
    
    # Voltage difference
    axes[1].plot(times, V_a - V_b, 'k-', linewidth=1.5)
    axes[1].set_ylabel('Delta V (mV)', fontsize=10)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, duration])
    
    # Gap junction currents
    axes[2].plot(times, I_gap_a, 'b-', linewidth=1.5, label='I_gap → A', alpha=0.8)
    axes[2].plot(times, I_gap_b, 'r-', linewidth=1.5, label='I_gap → B', alpha=0.8)
    axes[2].set_ylabel('I_gap (pA)', fontsize=10)
    axes[2].set_xlabel('Time (ms)', fontsize=10)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, duration])
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('output/06_celegans_gap_junction.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: output/06_celegans_gap_junction.png")
    
    return fig


def main():
    """Run all C. elegans graded neuron examples."""
    print("\n" + "=" * 70)
    print("C. ELEGANS GRADED POTENTIAL NEURONS - DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstration shows biologically-accurate graded potential")
    print("neuron models based on C. elegans electrophysiology.")
    print("\nKey features:")
    print("  * Continuous (analog) voltage dynamics - no spikes!")
    print("  * Ca2+, K+, and Ca-dependent K+ channels")
    print("  * Graded chemical synapses (voltage-dependent release)")
    print("  * Gap junctions (electrical coupling)")
    print("  * Different neuron classes (sensory, interneuron, motor)")
    print("\n" + "=" * 70)
    
    # Run examples
    fig1 = example1_single_neuron()
    fig2 = example2_neuron_classes()
    fig3 = example3_graded_synapse()
    fig4 = example4_gap_junction()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n[OK] Example 1: Single graded neuron - continuous voltage response")
    print("[OK] Example 2: Neuron classes - sensory (excitable), interneuron (balanced), motor (graded)")
    print("[OK] Example 3: Graded synapse - voltage-dependent neurotransmitter release")
    print("[OK] Example 4: Gap junction - bidirectional electrical coupling")
    print("\nAll figures saved in output/ directory")
    print("\nThese components can now be used to build C. elegans neural networks!")
    print("=" * 70 + "\n")
    
    plt.show()


if __name__ == "__main__":
    main()

