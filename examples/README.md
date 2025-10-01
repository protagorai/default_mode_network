# SDMN Framework Examples

This directory contains comprehensive examples demonstrating all aspects of the Synthetic Default Mode Network Framework. Each example is designed to be educational, well-documented, and immediately runnable.

## 🚀 Quick Start

```bash
# Using containers (recommended)
./scripts/build.sh                                      # Build container (first time)
./scripts/run.sh custom python examples/04_default_mode_networks.py  # Run DMN example

# Local installation  
export PYTHONPATH="${PWD}/src:$PYTHONPATH"             # Set path
python examples/04_default_mode_networks.py            # Run example
```

## 📚 Example Descriptions

### 01. Basic Neuron Demonstration
**File:** `01_basic_neuron_demo.py`  
**Runtime:** ~10 seconds  
**Focus:** Individual neuron models and their properties

**Demonstrates:**
- Leaky Integrate-and-Fire (LIF) neurons
- Hodgkin-Huxley (HH) biophysical neurons  
- Membrane dynamics and action potentials
- F-I curve analysis and rheobase calculation
- Comparison between neuron models

**Key Outputs:**
- Membrane potential traces
- Action potential shapes
- Firing rate vs. current curves
- Gating variable dynamics

**Good for:** Understanding basic building blocks, neuron model comparison

---

### 02. Network Topologies  
**File:** `02_network_topologies.py`  
**Runtime:** ~30 seconds  
**Focus:** Network architectures and connectivity patterns

**Demonstrates:**
- Random networks
- Ring networks  
- Small-world networks
- 2D grid networks
- Network connectivity analysis

**Key Outputs:**
- Network structure comparisons
- Connectivity statistics
- Spike raster plots
- Population activity patterns

**Good for:** Understanding how network topology affects dynamics

---

### 03. Probe Monitoring System
**File:** `03_probe_monitoring.py`  
**Runtime:** ~20 seconds  
**Focus:** Comprehensive neural activity monitoring

**Demonstrates:**
- Voltage probes (synthetic EEG)
- Spike detection and analysis
- Population activity monitoring
- Local Field Potential (LFP) simulation
- Frequency domain analysis

**Key Outputs:**
- High-resolution voltage traces
- Spike timing statistics
- Population brain waves
- LFP signals and spectra
- Inter-spike interval analysis

**Good for:** Learning the monitoring system, data analysis techniques

---

### 04. Default Mode Networks ⭐
**File:** `04_default_mode_networks.py`  
**Runtime:** ~45 seconds  
**Focus:** Synthetic default mode networks (CORE FEATURE)

**Demonstrates:**
- Multi-regional brain architecture (PCC, mPFC, AG, PCu, TP)
- Biologically-realistic connectivity
- Spontaneous oscillatory activity
- EEG frequency band analysis
- Network synchronization

**Key Outputs:**
- Regional network visualization
- Synthetic brain waves
- EEG band power analysis (Delta, Theta, Alpha, Beta, Gamma)
- Inter-regional connectivity
- Network synchronization patterns

**Good for:** Main framework purpose, brain wave research, DMN studies

---

### 05. Self-Aware Networks 🧠
**File:** `05_self_aware_network.py`  
**Runtime:** ~30 seconds  
**Focus:** Basic self-awareness and self-preservation (ADVANCED FEATURE)

**Demonstrates:**
- Self-monitoring and health assessment
- Risk-reward evaluation for self-preservation
- Adaptive decision-making based on internal state
- Internal narrative construction
- Basic artificial consciousness indicators
- Self-preservation strategy evolution

**Key Outputs:**
- Self-health monitoring over time
- Decision patterns and threat avoidance
- Strategy evolution (exploration/conservation/recovery)
- Internal narrative samples
- Self-awareness performance metrics

**Good for:** Consciousness research, self-preservation studies, AI safety research

---

### Quickstart Simulation
**File:** `quickstart_simulation.py`  
**Runtime:** ~15 seconds  
**Focus:** Complete tutorial example

**Demonstrates:**
- End-to-end simulation workflow
- Ring network with feedback
- Basic monitoring and visualization
- Result analysis and plotting

**Good for:** Complete walkthrough, tutorial following

## 🎯 Recommended Learning Path

### For Beginners:
```bash
1. python examples/01_basic_neuron_demo.py      # Learn neurons
2. python examples/04_default_mode_networks.py  # See main capabilities  
3. python examples/03_probe_monitoring.py       # Understand monitoring
4. python examples/02_network_topologies.py     # Explore architectures
```

### For Brain Wave Research:
```bash
# Go directly to the core feature
python examples/04_default_mode_networks.py
```

### For Self-Awareness/Consciousness Research:
```bash
1. python examples/04_default_mode_networks.py  # DMN foundations
2. python examples/05_self_aware_network.py     # Self-awareness demo
```

### For Network Architecture Studies:
```bash
1. python examples/02_network_topologies.py     # Compare topologies
2. python examples/04_default_mode_networks.py  # Biological architecture
```

### For Monitoring and Analysis:
```bash
1. python examples/03_probe_monitoring.py       # Full monitoring demo
2. python examples/04_default_mode_networks.py  # Advanced analysis
3. python examples/05_self_aware_network.py     # Self-monitoring systems
```

## 🔧 Customization Tips

All examples are designed for easy modification:

### Change Simulation Parameters:
```python
# In any example file:
config = SimulationConfig(
    dt=0.05,        # Smaller time step for higher precision
    max_time=3000.0 # Longer simulation for better frequency resolution
)
```

### Modify Network Size:
```python
# Increase network complexity:
n_neurons = 100    # More neurons
connection_probability = 0.15  # More connections
```

### Adjust Monitoring:
```python
# Higher resolution monitoring:
sampling_interval=0.1  # Sample every 0.1 ms instead of 1.0 ms
enable_filtering=True  # Add low-pass filtering
filter_cutoff=500.0    # 500 Hz cutoff frequency
```

## 📊 Understanding Outputs

Each example creates files in the `output/` directory:

```
output/
├── 01_neuron_comparison.png           # Neuron model comparison
├── 02_network_comparison.png          # Network topology comparison  
├── 02_raster_comparison.png           # Spike raster plots
├── 03_comprehensive_monitoring.png    # Complete monitoring demo
└── 04_default_mode_network.png        # DMN analysis (main result)
```

### Plot Interpretation:

- **Voltage traces:** Synthetic EEG-like signals
- **Spike rasters:** When and which neurons fire
- **Population rates:** Network-level brain waves
- **Frequency spectra:** Power in different EEG bands
- **Synchrony plots:** Network coherence over time

## 🐛 Troubleshooting

### Common Issues:

1. **Module Import Errors:**
   ```bash
   export PYTHONPATH="${PWD}/src:$PYTHONPATH"
   ```

2. **Missing Dependencies:**
   ```bash
   poetry install --only=main
   # or: pip install -e .
   ```

3. **Plots Don't Display:**
   - Examples save plots to `output/` directory even without display
   - Use `./scripts/run.sh jupyter` for interactive plotting

4. **Slow Performance:**
   - Examples are optimized for speed (~10-45 seconds each)
   - Reduce `max_time` or `n_neurons` for faster execution

### Getting Help:

- Check console output for detailed progress and statistics
- All examples print summary information and next steps
- Plots include comprehensive legends and titles
- Code is extensively commented for learning

## 🔬 Research Applications

These examples support research in:

- **Computational Neuroscience:** Network dynamics, neural oscillations
- **Brain Wave Analysis:** EEG frequency bands, synchronization
- **Network Theory:** Topology effects, connectivity patterns  
- **Default Mode Networks:** Resting-state activity, DMN connectivity
- **Neural Engineering:** Brain-computer interfaces, neuromorphic systems

## 📖 Further Reading

After running examples, see:
- `docs/quickstart.md` - Complete tutorial with code explanations
- `docs/plan/vision.md` - Research vision and goals
- `docs/plan/architecture.md` - Framework design details
- `src/` directory - Complete source code with documentation

---

*Start with any example that interests you - they're all designed to be educational and immediately rewarding!*
