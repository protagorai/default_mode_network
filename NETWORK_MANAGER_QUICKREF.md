# NetworkManager Quick Reference

Fast lookup for common operations with the C. elegans NetworkManager.

---

## Import

```python
from sdmn.networks.celegans import CElegansNetwork, SimulationState
```

---

## Create Network

```python
# Default configuration
net = CElegansNetwork()

# Custom configuration
from sdmn.networks.celegans.network_manager import SimulationConfig
config = SimulationConfig(dt=0.01, record_interval=10)
net = CElegansNetwork(config=config)
```

---

## Add Neurons

```python
# By class (recommended)
n1 = net.add_sensory_neuron("AWC")
n2 = net.add_interneuron("AVA")
n3 = net.add_motor_neuron("DA01")

# Generic (with custom params)
from sdmn.neurons.graded import CElegansNeuron, CElegansParameters
neuron = CElegansNeuron("custom", CElegansParameters(g_Ca=1.2))
net.add_neuron(neuron)
```

---

## Add Connections

```python
# Graded chemical synapse
syn_id = net.add_graded_synapse(
    "pre_id", "post_id",
    weight=2.0,
    synapse_type=SynapseType.EXCITATORY,
    tau_rise=1.0,
    tau_decay=5.0
)

# Gap junction
gap_id = net.add_gap_junction(
    "neuron_a", "neuron_b",
    conductance=1.0
)

# Auto-generated IDs
net.add_graded_synapse("n1", "n2", weight=2.0)  # ID: "syn_n1_n2_0"
net.add_gap_junction("n1", "n2")                # ID: "gap_n1_n2_0"
```

---

## Stimulate Network

```python
# Single neuron
net.set_external_current("neuron_id", 60.0)  # 60 pA

# Multiple neurons
net.set_external_currents({
    "AWC": 70.0,
    "ASE": 60.0
})
```

---

## Run Simulation

```python
# Basic
net.simulate(duration=1000)  # 1000 ms

# Custom time step
net.simulate(duration=1000, dt=0.005)  # 5 μs steps

# No progress output
net.simulate(duration=1000, progress=False)
```

---

## Simulation Control

```python
# Pause
net.simulate(duration=200)
net.pause()

# Resume
net.resume(duration=300)  # Continues from 200 ms

# Extend (just keep simulating)
net.simulate(duration=500)  # Now at 700 ms total

# Reset
net.reset()  # Back to t=0, all voltages reset
```

---

## Modify Connectivity

```python
# Change weights
net.set_synapse_weight("syn_id", 3.0)
net.set_gap_conductance("gap_id", 1.5)

# Add connections during simulation
net.add_graded_synapse("n1", "n5", weight=2.0)

# Remove connections
net.remove_synapse("syn_id")
net.remove_gap_junction("gap_id")

# Remove neuron (and all its connections)
net.remove_neuron("neuron_id")
```

---

## Access Results

```python
# Current voltages
voltages = net.get_current_voltages()  # Dict[str, float]
v = net.get_neuron("n1").voltage       # Single neuron

# Voltage history
history = net.get_voltage_history()              # All neurons
history = net.get_voltage_history("neuron_id")   # Single neuron
# Returns: List[(time, voltage)]

# As numpy arrays
times, voltages = net.get_voltages_array()
# times: (n_timepoints,)
# voltages: (n_neurons, n_timepoints)

# Network info
summary = net.get_connectivity_summary()
# Returns: {
#   'n_neurons': int,
#   'n_chemical_synapses': int,
#   'n_gap_junctions': int,
#   'neurons_by_class': dict,
#   'avg_synapses_per_neuron': float,
#   'avg_gaps_per_neuron': float
# }
```

---

## Access Components

```python
# Get neuron
neuron = net.get_neuron("neuron_id")
states = neuron.get_channel_states()  # m_Ca, m_K, m_KCa, Ca_internal
currents = neuron.get_currents()      # All ionic currents

# Count
n_neurons = net.get_neuron_count()

# Iterate
for neuron_id, neuron in net.neurons.items():
    print(f"{neuron_id}: {neuron.voltage:.2f} mV")

for syn_id, syn in net.chemical_synapses.items():
    print(f"{syn_id}: weight={syn.graded_params.weight}")

for gap_id, gap in net.gap_junctions.items():
    print(f"{gap_id}: g={gap.gap_params.conductance}")
```

---

## Network State

```python
# Check state
print(net.state)  # SimulationState.RUNNING, PAUSED, etc.

# Check time
print(f"Simulation time: {net.current_time:.1f} ms")
print(f"Steps: {net.step_count}")

# String representation
print(net)  # CElegansNetwork(neurons=10, synapses=12, ...)
```

---

## Common Patterns

### Build Simple Chain

```python
net = CElegansNetwork()

# Create chain
neurons = []
for i in range(5):
    nid = net.add_interneuron(f"N{i}")
    neurons.append(nid)

# Connect sequentially
for i in range(len(neurons) - 1):
    net.add_graded_synapse(neurons[i], neurons[i+1], weight=2.0)
```

### Stimulation Protocol

```python
# Step current
net.set_external_current("N0", 0.0)
net.simulate(duration=100)  # Baseline

net.set_external_current("N0", 60.0)
net.simulate(duration=500)  # Stimulus

net.set_external_current("N0", 0.0)
net.simulate(duration=400)  # Recovery
```

### Parameter Sweep

```python
results = []
for weight in [0.5, 1.0, 2.0, 5.0]:
    net.reset()
    net.set_synapse_weight("syn1", weight)
    net.set_external_current("N0", 60.0)
    net.simulate(duration=500, progress=False)
    
    v_final = net.get_neuron("N4").voltage
    results.append((weight, v_final))
```

### Save Results

```python
import numpy as np

# Get data
times, voltages = net.get_voltages_array()

# Save
np.savez('results.npz', times=times, voltages=voltages)

# Load later
data = np.load('results.npz')
times = data['times']
voltages = data['voltages']
```

---

## Performance Tips

```python
# Reduce recording for speed
config = SimulationConfig(
    record_interval=100,  # Record every 100 steps (default: 10)
    progress_interval=50000  # Less progress output
)

# Disable recording entirely
config = SimulationConfig(enable_recording=False)

# Larger time step (less accurate but faster)
net.simulate(duration=1000, dt=0.05)  # 50 μs steps

# Use Euler instead of RK4 (neurons only)
from sdmn.neurons.graded import CElegansParameters
params = CElegansParameters(integration_method="Euler")
```

---

## Troubleshooting

```python
# Voltages not changing?
# - Make sure you're calling set_external_current() before simulate()
# - Check that currents are reasonable (10-100 pA)

# Simulation too slow?
# - Reduce record_interval
# - Use larger dt
# - Disable progress output (progress=False)

# Memory issues?
# - Disable recording or increase record_interval
# - Simulate in smaller chunks

# Need more control?
# - Access net.neurons, net.chemical_synapses, net.gap_junctions directly
# - Each neuron has update(dt) method
# - Each synapse/gap has update(dt) method
```

---

## Complete Example

```python
from sdmn.networks.celegans import CElegansNetwork
from sdmn.synapses import SynapseType

# Create network
net = CElegansNetwork()

# Build
s = net.add_sensory_neuron("S")
i1 = net.add_interneuron("I1")
i2 = net.add_interneuron("I2")
m = net.add_motor_neuron("M")

# Connect
net.add_graded_synapse(s, i1, weight=2.5)
net.add_graded_synapse(i1, i2, weight=2.0)
net.add_graded_synapse(i2, m, weight=1.5)
net.add_gap_junction(i1, i2, conductance=0.8)

# Stimulate
net.set_external_current("S", 60.0)

# Simulate
net.simulate(duration=500)

# Analyze
voltages = net.get_current_voltages()
for nid, v in voltages.items():
    print(f"{nid}: {v:.2f} mV")

# Continue
net.simulate(duration=500)  # Now at 1000 ms

# Modify and continue
net.set_synapse_weight("syn_S_I1_0", 5.0)
net.simulate(duration=500)  # Now at 1500 ms

print(f"Total time: {net.current_time:.1f} ms")
```

---

**For full documentation, see:**
- `docs/CELEGANS_IMPLEMENTATION.md`
- `docs/plan/celegans_graded_potential_design.md`
- `examples/07_network_manager_demo.py`

