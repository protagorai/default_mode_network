# C. elegans Graded Potential Neuron Examples

This directory contains examples demonstrating the C. elegans graded potential neuron implementation.

## Example: 06_celegans_graded_neurons.py

**Comprehensive demonstration** of all C. elegans neuron features.

### What It Demonstrates

1. **Single Graded Neuron**
   - Continuous voltage dynamics (no spikes!)
   - Ion channel gating variables (Ca2+, K+, KCa)
   - Intracellular calcium dynamics
   - Response to current injection

2. **Neuron Classes Comparison**
   - Sensory neurons (high Ca2+, excitable)
   - Interneurons (balanced, integrative)
   - Motor neurons (high K+, graded output)
   - Different responses to same stimulus

3. **Graded Chemical Synapse**
   - Voltage-dependent neurotransmitter release
   - Continuous (analog) transmission
   - Synaptic conductance and current dynamics
   - Postsynaptic depolarization

4. **Gap Junction**
   - Bidirectional electrical coupling
   - Synchronization without direct stimulation
   - Ohmic conductance
   - Current flow between neurons

### Running the Example

```bash
# From project root
python examples/06_celegans_graded_neurons.py
```

### Output

The script generates 4 figures in `output/`:

1. **06_celegans_single_neuron.png**
   - Voltage response to 50 pA current pulse
   - Channel activations (m_Ca, m_K, m_KCa)
   - Intracellular calcium concentration

2. **06_celegans_neuron_classes.png**
   - Comparison of sensory, interneuron, and motor responses
   - Full time course and early response zoom

3. **06_celegans_graded_synapse.png**
   - Presynaptic voltage → synaptic conductance → current → postsynaptic voltage
   - Complete synaptic transmission cascade

4. **06_celegans_gap_junction.png**
   - Voltage synchronization via electrical coupling
   - Voltage difference and gap junction currents

### Expected Output

```
======================================================================
C. ELEGANS GRADED POTENTIAL NEURONS - DEMONSTRATION
======================================================================

Example 1: Single C. elegans Graded Neuron
  Resting potential: -65.00 mV
  Peak depolarization: -32.45 mV
  Note: No spikes - continuous graded response!

Example 2: Different C. elegans Neuron Classes
  Sensory: Peak = -28.12 mV
  Interneuron: Peak = -35.67 mV
  Motor: Peak = -42.89 mV

Example 3: Graded Chemical Synapse
  Presynaptic peak: -25.34 mV
  Postsynaptic peak: -58.90 mV
  
Example 4: Gap Junction (Electrical Synapse)
  Neuron A peak: -35.23 mV
  Neuron B peak: -54.12 mV
  Voltage synchronization achieved!

SUMMARY
[OK] Example 1: Single graded neuron - continuous voltage response
[OK] Example 2: Neuron classes - different channel densities
[OK] Example 3: Graded synapse - voltage-dependent release
[OK] Example 4: Gap junction - bidirectional coupling
======================================================================
```

## Key Differences from Spiking Neurons

### Graded (C. elegans)
- ✓ Continuous, analog voltage
- ✓ No action potentials
- ✓ Linear summation
- ✓ Energy efficient
- ✓ Fine-grained control
- ✓ No refractory period

### Spiking (LIF/HH)
- ✓ Discrete spikes
- ✓ All-or-none
- ✓ Long-distance signaling
- ✓ Digital-like
- ✓ Refractory period
- ✓ Threshold-based

## Biological Context

**Why C. elegans uses graded potentials:**
- Very small (~1 mm length)
- Short distances between neurons
- 302 neurons total
- Energy efficient
- Fine motor control needed

**Applications:**
- Understanding fundamental neural computation
- Small-scale neural networks
- Robotics and control systems
- Testing network topology effects
- Benchmarking against real connectome

## Next Steps

After understanding these basics, you can:

1. **Build small networks** (5-50 neurons)
2. **Explore network topologies** (uniform vs. small-world)
3. **Simulate the full 302-neuron C. elegans network**
4. **Model specific behaviors** (locomotion, chemotaxis)

## References

- Goodman et al. (1998) - Ion channel characterization
- Lockery & Goodman (2009) - Graded vs. spiking neurons
- Varshney et al. (2011) - C. elegans connectome
- OpenWorm Project - Open-source C. elegans simulation

## Troubleshooting

**Plots don't show:**
- Comment out `plt.show()` to save figures only
- Or close windows to continue

**Unicode errors:**
- Already fixed in code (no superscripts/Greek letters)

**Slow simulation:**
- Reduce duration or increase dt
- Use Euler instead of RK4: `params.integration_method = "Euler"`

**Import errors:**
- Install package: `pip install -e .` from project root
- Check Python path

---

**For complete documentation, see:**  
`docs/CELEGANS_IMPLEMENTATION.md`

