# Quickstart Guide: Synthetic Default Mode Network Framework

This guide will walk you through the essential steps to get started with the SDMN Framework, from basic setup to running your first neural network simulation with synthetic brain wave generation.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Concepts](#basic-concepts)
3. [Creating Your First Neuron](#creating-your-first-neuron)
4. [Building Simple Networks](#building-simple-networks)
5. [Adding Feedback Loops](#adding-feedback-loops)
6. [Monitoring with Probes](#monitoring-with-probes)
7. [Running Simulations](#running-simulations)
8. [Visualizing Results](#visualizing-results)
9. [Advanced Examples](#advanced-examples)
10. [Debugging and Analysis](#debugging-and-analysis)

## Installation and Setup

### Option 1: Using Containers (Recommended)

The fastest way to get started is using the containerized version:

```bash
# Clone the repository
git clone <repository-url>
cd sdmn-framework

# Build the container
./scripts/build.sh

# Start interactive shell
./scripts/run.sh shell

# Or start Jupyter Lab for interactive development
./scripts/run.sh jupyter
```

### Option 2: Local Installation

```bash
# Create virtual environment
python3.11 -m venv sdmn-env
source sdmn-env/bin/activate  # Linux/Mac
# sdmn-env\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r dev-requirements.txt

# Set Python path
export PYTHONPATH="${PWD}/src:$PYTHONPATH"
```

## Basic Concepts

The SDMN Framework is built around these core components:

- **SimulationEngine**: Coordinates time stepping and event processing
- **Neurons**: Individual spiking neural units with various models (LIF, HH, etc.)
- **Synapses**: Connections between neurons with configurable dynamics
- **Networks**: Collections of neurons and their connections
- **Probes**: Monitoring tools for recording neural activity
- **Visualizers**: Tools for plotting and analyzing results

## Creating Your First Neuron

Let's start with a simple Leaky Integrate-and-Fire (LIF) neuron:

```python
import numpy as np
from src.neurons import LIFNeuron, LIFParameters

# Create neuron parameters
lif_params = LIFParameters(
    tau_m=20.0,           # Membrane time constant (ms)
    v_rest=-70.0,         # Resting potential (mV)
    v_thresh=-50.0,       # Spike threshold (mV)
    v_reset=-80.0,        # Reset potential (mV)
    r_mem=10.0,           # Membrane resistance (MΩ)
    refractory_period=2.0 # Refractory period (ms)
)

# Create the neuron
neuron = LIFNeuron("neuron_001", lif_params)

# Stimulate with constant current
neuron.set_external_input(2.0)  # 2 nA input current

# Update for one time step
neuron.update(dt=0.1)

# Check if it spiked
if neuron.has_spiked():
    print(f"Neuron spiked at {neuron.get_last_spike_time():.1f} ms")

print(f"Membrane potential: {neuron.get_membrane_potential():.1f} mV")
```

## Building Simple Networks

Now let's create a small network with connected neurons:

```python
from src.neurons import LIFNeuron, LIFParameters, SynapseFactory
from src.core import SimulationEngine, SimulationConfig

# Create simulation configuration
config = SimulationConfig(
    dt=0.1,           # 0.1 ms time steps
    max_time=1000.0,  # 1 second simulation
    enable_logging=True
)

# Create simulation engine
engine = SimulationEngine(config)

# Create neurons
neurons = {}
for i in range(5):
    neuron_id = f"neuron_{i:03d}"
    neurons[neuron_id] = LIFNeuron(neuron_id, LIFParameters())

# Create synaptic connections
synapses = {}

# Connect neuron 0 to neurons 1-4 (excitatory)
for i in range(1, 5):
    syn_id = f"syn_0_to_{i}"
    synapse = SynapseFactory.create_excitatory_synapse(
        syn_id, "neuron_000", f"neuron_{i:03d}",
        weight=1.5, delay=2.0
    )
    synapses[syn_id] = synapse

# Register synapses with neurons
for synapse in synapses.values():
    pre_neuron = neurons[synapse.presynaptic_neuron_id]
    post_neuron = neurons[synapse.postsynaptic_neuron_id]
    
    pre_neuron.add_postsynaptic_connection(synapse)
    post_neuron.add_presynaptic_connection(synapse)

# Create a simple network class
class SimpleNetwork:
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
    
    def update(self, dt):
        # Update all synapses
        for synapse in self.synapses.values():
            synapse.update(dt)
        
        # Update all neurons
        for neuron in self.neurons.values():
            # Calculate synaptic inputs
            synaptic_currents = []
            for synapse in neuron.presynaptic_connections:
                if synapse.synapse_id in self.synapses:
                    current = synapse.calculate_current(
                        neuron.get_membrane_potential()
                    )
                    synaptic_currents.append(current)
            
            # Add synaptic inputs to neuron
            for current in synaptic_currents:
                neuron.add_synaptic_input(current)
            
            # Update neuron
            neuron.update(dt)

# Create network
network = SimpleNetwork(neurons, synapses)

# Add network to simulation engine
engine.add_network("simple_network", network)

# Add constant stimulus to first neuron
def stimulus_callback(step, time):
    neurons["neuron_000"].set_external_input(3.0)  # 3 nA stimulus

engine.register_step_callback(stimulus_callback)
```

## Adding Feedback Loops

For default mode network behavior, we need feedback loops. Here's how to create them:

```python
# Create a ring network with feedback
def create_ring_network(n_neurons=10):
    neurons = {}
    synapses = {}
    
    # Create neurons in ring
    for i in range(n_neurons):
        neuron_id = f"ring_neuron_{i:03d}"
        # Vary parameters slightly for heterogeneity
        params = LIFParameters(
            tau_m=20.0 + np.random.normal(0, 2.0),
            v_thresh=-50.0 + np.random.normal(0, 2.0)
        )
        neurons[neuron_id] = LIFNeuron(neuron_id, params)
    
    # Connect in ring (each neuron connects to next)
    for i in range(n_neurons):
        next_i = (i + 1) % n_neurons
        syn_id = f"ring_syn_{i}_to_{next_i}"
        
        # Create excitatory connection
        synapse = SynapseFactory.create_excitatory_synapse(
            syn_id, 
            f"ring_neuron_{i:03d}", 
            f"ring_neuron_{next_i:03d}",
            weight=1.2, 
            delay=5.0
        )
        synapses[syn_id] = synapse
        
        # Register connections
        neurons[f"ring_neuron_{i:03d}"].add_postsynaptic_connection(synapse)
        neurons[f"ring_neuron_{next_i:03d}"].add_presynaptic_connection(synapse)
    
    # Add long-range feedback connections
    for i in range(0, n_neurons, 3):  # Every 3rd neuron
        target_i = (i + n_neurons//2) % n_neurons  # Connect to opposite side
        syn_id = f"feedback_syn_{i}_to_{target_i}"
        
        synapse = SynapseFactory.create_excitatory_synapse(
            syn_id,
            f"ring_neuron_{i:03d}",
            f"ring_neuron_{target_i:03d}",
            weight=0.8,
            delay=10.0
        )
        synapses[syn_id] = synapse
        
        # Register connections
        neurons[f"ring_neuron_{i:03d}"].add_postsynaptic_connection(synapse)
        neurons[f"ring_neuron_{target_i:03d}"].add_presynaptic_connection(synapse)
    
    return SimpleNetwork(neurons, synapses)

# Create ring network with feedback
feedback_network = create_ring_network(20)
```

## Monitoring with Probes

Probes are essential for recording neural activity and generating synthetic "brain waves":

```python
from src.probes import VoltageProbe, SpikeProbe, PopulationActivityProbe

# Create voltage probe for membrane potential traces
voltage_probe = VoltageProbe(
    probe_id="voltage_monitor",
    target_neurons=["neuron_000", "neuron_001", "neuron_002"],
    sampling_interval=0.5,  # Sample every 0.5 ms
    enable_filtering=True,
    filter_cutoff=200.0     # 200 Hz low-pass filter
)

# Register neuron objects with the probe
for neuron_id, neuron in neurons.items():
    if neuron_id in voltage_probe.target_ids:
        voltage_probe.register_neuron_object(neuron_id, neuron)

# Create spike probe for precise spike timing
spike_probe = SpikeProbe(
    probe_id="spike_monitor",
    target_neurons=list(neurons.keys()),
    detection_threshold=-30.0,
    record_waveforms=False
)

# Register neurons with spike probe
for neuron_id, neuron in neurons.items():
    spike_probe.register_neuron_object(neuron_id, neuron)

# Create population activity probe for "brain wave" generation
population_probe = PopulationActivityProbe(
    probe_id="population_monitor",
    target_population="simple_network",
    target_neurons=list(neurons.keys()),
    bin_size=5.0,           # 5 ms bins
    sliding_window=100.0,   # 100 ms window
    record_synchrony=True
)

population_probe.register_neuron_objects(neurons)

# Add probes to simulation engine
engine.add_probe("voltage_probe", voltage_probe)
engine.add_probe("spike_probe", spike_probe)
engine.add_probe("population_probe", population_probe)
```

## Running Simulations

Now let's run the simulation and collect data:

```python
# Start recording on all probes
voltage_probe.start_recording()
spike_probe.start_recording()
population_probe.start_recording()

# Add periodic stimulus to maintain activity
def periodic_stimulus(step, time):
    # Add noise to first neuron every 50 ms
    if step % 500 == 0:  # Every 50 ms (500 steps * 0.1 ms)
        stimulus_current = np.random.normal(2.0, 0.5)
        neurons["neuron_000"].set_external_input(stimulus_current)
    else:
        neurons["neuron_000"].set_external_input(0.1)  # Small baseline current

engine.register_step_callback(periodic_stimulus)

# Run simulation
print("Starting simulation...")
results = engine.run()

if results.success:
    print(f"Simulation completed successfully!")
    print(f"Total steps: {results.total_steps}")
    print(f"Simulation time: {results.simulation_time:.1f} ms")
    print(f"Wall time: {results.wall_time:.2f} seconds")
else:
    print(f"Simulation failed: {results.error_message}")
```

## Visualizing Results

Let's visualize the results to see synthetic brain wave patterns:

```python
import matplotlib.pyplot as plt

def plot_results():
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # 1. Voltage traces (like EEG)
    voltage_traces = voltage_probe.get_voltage_traces()
    for i, (neuron_id, trace) in enumerate(voltage_traces.items()):
        if i < 3:  # Plot first 3 neurons
            axes[0].plot(trace['time'], trace['voltage'], 
                        label=f'{neuron_id}', alpha=0.7)
    axes[0].set_ylabel('Membrane Potential (mV)')
    axes[0].set_title('Neural "EEG" - Membrane Potential Traces')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Spike raster plot
    spike_times = spike_probe.get_spike_times()
    y_pos = 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(spike_times)))
    
    for neuron_id, spikes in spike_times.items():
        if spikes:
            axes[1].scatter(spikes, [y_pos] * len(spikes), 
                           s=2, c=[colors[y_pos]], alpha=0.7)
        y_pos += 1
    axes[1].set_ylabel('Neuron ID')
    axes[1].set_title('Spike Raster Plot')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Population firing rate (synthetic brain wave)
    times, rates = population_probe.get_population_rate_trace()
    axes[2].plot(times, rates, 'b-', linewidth=2)
    axes[2].set_ylabel('Population Rate (Hz)')
    axes[2].set_title('Synthetic Brain Wave - Population Activity')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Synchrony index (network coherence)
    times, synchrony = population_probe.get_synchrony_trace()
    if len(synchrony) > 0:
        axes[3].plot(times, synchrony, 'r-', linewidth=2)
        axes[3].set_ylabel('Synchrony Index')
        axes[3].set_title('Network Synchronization')
        axes[3].set_xlabel('Time (ms)')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('output/sdmn_simulation_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to output/sdmn_simulation_results.png")

# Generate plots
plot_results()
```

## Advanced Examples

### Creating Complex Default Mode Network

```python
def create_dmn_network():
    """Create a more complex default mode network with multiple regions."""
    
    # Define network regions
    regions = {
        'pcc': 15,    # Posterior Cingulate Cortex
        'mpfc': 12,   # Medial Prefrontal Cortex
        'ag': 10,     # Angular Gyrus
        'hip': 8      # Hippocampus
    }
    
    neurons = {}
    synapses = {}
    
    # Create neurons for each region
    for region, n_neurons in regions.items():
        for i in range(n_neurons):
            neuron_id = f"{region}_{i:03d}"
            # Regional parameter variations
            if region == 'pcc':
                params = LIFParameters(tau_m=25.0, v_thresh=-45.0)
            elif region == 'mpfc':
                params = LIFParameters(tau_m=22.0, v_thresh=-48.0)
            elif region == 'ag':
                params = LIFParameters(tau_m=20.0, v_thresh=-52.0)
            else:  # hippocampus
                params = LIFParameters(tau_m=18.0, v_thresh=-55.0)
            
            neurons[neuron_id] = LIFNeuron(neuron_id, params)
    
    # Create intra-regional connections
    syn_counter = 0
    for region, n_neurons in regions.items():
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
                            weight=0.8,
                            delay=2.0
                        )
                        synapses[syn_id] = synapse
    
    # Create inter-regional connections (DMN connectivity pattern)
    inter_connections = [
        ('pcc', 'mpfc', 0.4, 15.0),  # Strong PCC-MPFC connection
        ('pcc', 'ag', 0.3, 12.0),    # PCC-Angular Gyrus
        ('mpfc', 'hip', 0.25, 20.0), # MPFC-Hippocampus
        ('ag', 'hip', 0.2, 18.0)     # Angular Gyrus-Hippocampus
    ]
    
    for region1, region2, prob, delay in inter_connections:
        n1, n2 = regions[region1], regions[region2]
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
    
    return SimpleNetwork(neurons, synapses), regions

# Create DMN network
dmn_network, dmn_regions = create_dmn_network()
print(f"Created DMN network with {len(dmn_network.neurons)} neurons and {len(dmn_network.synapses)} synapses")
```

### Frequency Analysis of Synthetic Brain Waves

```python
def analyze_brain_waves(population_probe):
    """Analyze frequency content of population activity."""
    
    times, rates = population_probe.get_population_rate_trace()
    
    if len(rates) < 100:
        print("Not enough data for frequency analysis")
        return
    
    # Calculate power spectral density
    from scipy import signal
    
    # Sampling frequency
    dt = np.mean(np.diff(times)) / 1000.0  # Convert ms to seconds
    fs = 1.0 / dt
    
    # Calculate PSD
    frequencies, psd = signal.welch(rates, fs, nperseg=min(256, len(rates)//4))
    
    # Define frequency bands (like EEG)
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-100 Hz)': (30, 100)
    }
    
    # Calculate power in each band
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        if np.any(mask):
            band_power = np.trapz(psd[mask], frequencies[mask])
            band_powers[band_name] = band_power
    
    # Plot frequency analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Power spectral density
    ax1.semilogy(frequencies, psd)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title('Synthetic Brain Wave Frequency Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)
    
    # Band powers
    band_names = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = ['purple', 'blue', 'green', 'orange', 'red']
    
    bars = ax2.bar(range(len(band_names)), powers, color=colors[:len(band_names)])
    ax2.set_xlabel('Frequency Band')
    ax2.set_ylabel('Power')
    ax2.set_title('Power in Different Frequency Bands')
    ax2.set_xticks(range(len(band_names)))
    ax2.set_xticklabels(band_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add power values on bars
    for bar, power in zip(bars, powers):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + power*0.01,
                f'{power:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('output/brain_wave_analysis.png', dpi=300, bbox_inches='tight')
    
    return band_powers

# Analyze the frequency content
if 'population_probe' in locals():
    band_powers = analyze_brain_waves(population_probe)
    print("Band Powers:", band_powers)
```

## Debugging and Analysis

### Common Issues and Solutions

1. **No Spikes Generated**:
   ```python
   # Check if input current is sufficient
   rheobase = neuron.get_rheobase_current()
   print(f"Minimum current needed: {rheobase:.2f} nA")
   
   # Increase input current
   neuron.set_external_input(rheobase * 1.5)
   ```

2. **Simulation Runs Too Slowly**:
   ```python
   # Use larger time steps (with caution)
   config.dt = 0.2  # Instead of 0.1
   
   # Reduce number of neurons or connections
   # Use simpler neuron models (LIF instead of HH)
   ```

3. **Unstable Network Activity**:
   ```python
   # Add inhibitory connections
   inhibitory_synapse = SynapseFactory.create_inhibitory_synapse(
       "inh_syn", "neuron_000", "neuron_001", weight=1.0
   )
   
   # Reduce connection weights
   for synapse in synapses.values():
       current_weight = synapse.get_weight()
       synapse.set_weight(current_weight * 0.8)
   ```

### Performance Monitoring

```python
# Get simulation performance statistics
def analyze_performance(results):
    print("Performance Analysis:")
    print(f"Total simulation time: {results.simulation_time:.1f} ms")
    print(f"Wall clock time: {results.wall_time:.2f} seconds")
    print(f"Simulation speed ratio: {results.simulation_time/results.wall_time/1000:.2f}x real-time")
    
    stats = results.performance_stats
    print(f"Average step time: {stats['avg_step_time']*1000:.2f} ms")
    print(f"Steps per second: {stats['steps_per_second']:.0f}")

# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Check network connectivity
def analyze_network_stats(network):
    total_neurons = len(network.neurons)
    total_synapses = len(network.synapses)
    
    # Calculate connectivity statistics
    connections_per_neuron = {}
    for neuron_id, neuron in network.neurons.items():
        in_count, out_count = neuron.get_connection_count()
        connections_per_neuron[neuron_id] = (in_count, out_count)
    
    avg_in = np.mean([counts[0] for counts in connections_per_neuron.values()])
    avg_out = np.mean([counts[1] for counts in connections_per_neuron.values()])
    
    print(f"Network Statistics:")
    print(f"Neurons: {total_neurons}")
    print(f"Synapses: {total_synapses}")
    print(f"Average in-degree: {avg_in:.1f}")
    print(f"Average out-degree: {avg_out:.1f}")
    print(f"Connection density: {total_synapses/(total_neurons**2)*100:.2f}%")
```

## Next Steps

After completing this quickstart:

1. **Explore Advanced Neuron Models**: Try Hodgkin-Huxley neurons for more biophysical detail
2. **Implement Learning**: Add spike-timing dependent plasticity (STDP)
3. **Scale Up**: Create larger networks with thousands of neurons
4. **Add Stimuli**: Implement various input patterns and stimulation protocols
5. **Custom Analysis**: Develop your own analysis methods for synthetic brain waves
6. **Contribute**: Help extend the framework with new features

## Additional Resources

- **API Documentation**: See `docs/api/` for detailed class and method documentation
- **Examples**: Check `examples/` directory for more complex scenarios
- **Architecture Guide**: Read `docs/plan/architecture.md` for framework design details
- **Vision Document**: See `docs/plan/vision.md` for research goals and roadmap

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Join the research community discussions
- Contact the development team

---

*Happy simulating! You're now ready to explore the fascinating world of synthetic default mode networks and artificial brain waves.*
