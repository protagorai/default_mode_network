# SDMN Framework Example Outputs

This document shows what outputs and visualizations you can expect when running the SDMN framework examples.

## 📊 Example 01: Basic Neuron Demo (`01_basic_neuron_demo.py`)

### **Console Output:**
```
SDMN Framework - Basic Neuron Demonstration
==================================================

=== LIF Neuron Demonstration ===
Simulating LIF neuron for 200.0 ms...
Neuron fired 6 spikes
Rheobase current: 2.15 nA
Inter-spike intervals: [12.3 14.7 13.1 12.9 13.8] ms
Mean firing rate: 30.0 Hz

=== Hodgkin-Huxley Neuron Demonstration ===
Simulating HH neuron for 50.0 ms...
HH neuron fired 2 spikes

=== Summary ===
✓ LIF neuron: Simple, computationally efficient
✓ HH neuron: Biophysically detailed, shows action potential shape
✓ Both models show spiking behavior with current input
✓ LIF suitable for large networks, HH for detailed biophysics

Next steps:
• Try adjusting neuron parameters
• Test different input patterns
• Run example 02 for network simulations

Plot saved to: output/01_neuron_comparison.png
```

### **Visual Output (3×2 Subplot Figure):**

**File:** `output/01_neuron_comparison.png` (300 DPI, publication quality)

| **LIF Neuron (Left)** | **Hodgkin-Huxley Neuron (Right)** |
|----------------------|-----------------------------------|
| **Row 1**: Membrane potential trace<br/>• Shows voltage vs time<br/>• Threshold line at -50mV<br/>• Resting line at -70mV<br/>• Red spike markers | **Row 1**: Action potential waveform<br/>• Detailed spike shape<br/>• Shows overshoot and afterhyperpolarization<br/>• Classic HH action potential |
| **Row 2**: Input current profile<br/>• Step current 50-150ms<br/>• Rheobase line indicator<br/>• Current amplitude: 3.0 nA | **Row 2**: Current injection<br/>• Pulse current 10-40ms<br/>• Current amplitude: 10 μA/cm² |
| **Row 3**: F-I (Frequency-Intensity) curve<br/>• Simulated vs analytical<br/>• Shows rheobase threshold<br/>• Linear response above threshold | **Row 3**: Gating variables<br/>• m (Na+ activation) - red<br/>• h (Na+ inactivation) - blue<br/>• n (K+ activation) - green |

---

## 🌐 Example 02: Network Topologies (`02_network_topologies.py`)

### **Console Output:**
```
SDMN Framework - Network Topologies Demonstration
================================================

=== Creating Sample Networks ===
Creating Random Network (50 neurons)...
✓ Random network: 50 neurons, 125 synapses, density: 0.10

Creating Ring Network (50 neurons)...
✓ Ring network: 50 neurons, 50 synapses, density: 0.04

Creating Small-World Network (50 neurons)...
✓ Small-world network: 50 neurons, 89 synapses, density: 0.07

Creating 2D Grid Network (49 neurons)...
✓ Grid network: 49 neurons, 84 synapses, density: 0.07

=== Network Analysis ===
Random Network:
  Path length: 2.3 ± 0.4
  Clustering: 0.08 ± 0.03
  Small-world index: 0.12

Ring Network:
  Path length: 12.5 ± 3.2
  Clustering: 0.33 ± 0.05
  Small-world index: 2.64

Small-World Network:
  Path length: 3.1 ± 0.6
  Clustering: 0.28 ± 0.04
  Small-world index: 2.13

Grid Network:
  Path length: 6.2 ± 1.8
  Clustering: 0.22 ± 0.06
  Small-world index: 1.42

Plot saved to: output/02_network_topologies.png
```

### **Visual Output:**
- **Network topology diagrams** (4 different layouts)
- **Connectivity matrices** (color-coded adjacency matrices)
- **Analysis plots** (path length, clustering coefficient)

---

## 🔬 Example 03: Probe Monitoring (`03_probe_monitoring.py`)

### **Console Output:**
```
Creating monitored neural network...
✓ Created network with 25 neurons
✓ Added external stimulation to 5 neurons
✓ Installed 4 monitoring probes

Running 2000ms simulation...
[========================================] 100% Complete

=== Probe Analysis Results ===

Voltage Probe (5 neurons monitored):
  Mean voltage: -64.2 ± 8.7 mV
  Spike events: 234 detected
  Voltage range: [-80.0, +15.3] mV

Spike Probe (25 neurons monitored):
  Total spikes: 1,247
  Population rate: 24.9 Hz
  Active neurons: 23/25 (92%)
  Mean ISI: 42.3 ± 18.9 ms

Population Activity Probe:
  Average rate: 24.9 ± 8.3 Hz
  Synchrony index: 0.34
  Burst episodes: 8 detected
  Network oscillations: 12.5 Hz dominant

LFP Probe (Synthetic EEG):
  Alpha band (8-13 Hz): 15.2% power
  Beta band (13-30 Hz): 32.8% power
  Gamma band (30-100 Hz): 8.9% power

Plot saved to: output/03_probe_monitoring.png
```

### **Visual Output:**
- **Voltage traces** from multiple neurons
- **Spike raster plot** showing population activity
- **Population firing rate** over time
- **Synthetic EEG signal** with frequency analysis

---

## 🧠 Example 04: Default Mode Networks (`04_default_mode_networks.py`)

### **Console Output:**
```
Creating Default Mode Network Architecture...
✓ PCC region: 20 neurons (Posterior Cingulate Cortex)
✓ mPFC region: 15 neurons (medial Prefrontal Cortex) 
✓ AG region: 12 neurons (Angular Gyrus)
✓ Inter-region connectivity established

Simulating DMN dynamics for 5000ms...
[========================================] 100% Complete

=== DMN Analysis Results ===

Regional Activity:
  PCC: 18.3 Hz (high baseline activity)
  mPFC: 12.7 Hz (cognitive control)
  AG: 15.1 Hz (semantic processing)

Default Mode Oscillations:
  Delta (1-4 Hz): 23.1% power - Slow cortical rhythms
  Theta (4-8 Hz): 18.7% power - Memory consolidation
  Alpha (8-13 Hz): 35.2% power - Resting state
  Beta (13-30 Hz): 15.4% power - Cognitive processing

Connectivity Analysis:
  Within-region correlation: 0.67 ± 0.12
  Between-region correlation: 0.34 ± 0.08
  Global efficiency: 0.42

DMN State Detection:
  Active state: 68.3% of time
  Deactivated state: 31.7% of time
  Transition frequency: 4.2 per minute

Plot saved to: output/04_dmn_analysis.png
```

### **Visual Output:**
- **DMN network architecture** diagram
- **Regional activity plots** for each brain area
- **Synthetic EEG** showing realistic brain waves
- **Connectivity heatmaps** between regions

---

## 🤖 Example 05: Self-Aware Network (`05_self_aware_network.py`)

### **Console Output:**
```
Creating Self-Aware Neural Network...
✓ Core network: 50 neurons
✓ Self-monitoring system installed
✓ Risk-reward assessment enabled
✓ Decision-making modules active

Running self-aware simulation for 3000ms...

=== Self-Monitoring Events ===
[t=245ms] MONITORING: Network health check - Status: HEALTHY
[t=456ms] RISK DETECTED: High activity in region 2 - Risk level: 0.73
[t=467ms] DECISION: Reduce excitation in region 2 (self-preservation)
[t=623ms] MONITORING: Activity normalized - Status: STABLE
[t=891ms] REWARD: Successful adaptation - Confidence: +0.15
[t=1205ms] SELF-ASSESSMENT: Learning rate adjusted based on outcomes

=== Self-Awareness Analysis ===

Decision Statistics:
  Total decisions made: 23
  Successful outcomes: 19 (82.6%)
  Failed outcomes: 4 (17.4%)
  Average decision time: 12.3ms

Risk Management:
  Risk events detected: 15
  Preventive actions: 12 (80%)
  Network stabilizations: 11 (91.7% success)
  
Self-Monitoring:
  Health checks: 47
  State assessments: 23
  Adaptive modifications: 8

Emergent Behaviors:
  ✓ Self-preservation responses
  ✓ Adaptive parameter tuning
  ✓ Risk-reward learning
  ✓ Internal state monitoring

Plot saved to: output/05_self_awareness.png
```

### **Visual Output:**
- **Network activity timeline** with decision points
- **Risk-reward learning curves**
- **Self-monitoring state evolution**
- **Decision outcome analysis**

---

## 🚀 Example 06: Quickstart Simulation (`quickstart_simulation.py`)

### **Console Output:**
```
SDMN Framework - Quickstart Simulation
====================================

Creating simple neural network...
✓ Network: 20 neurons, 38 synapses
✓ Added voltage and spike probes
✓ Applied external stimulation

Running simulation (1000ms)...
Progress: [████████████████████████] 100%

Results:
✓ Simulation completed successfully
✓ Total simulation steps: 10,000
✓ Wall time: 2.34 seconds
✓ Average firing rate: 15.7 Hz
✓ Network synchronization: 0.42

Data saved to: output/quickstart_results.json
Plot saved to: output/quickstart_simulation.png
```

### **Visual Output:**
- **Network activity overview**
- **Voltage traces** from sample neurons
- **Population spike raster**
- **Firing rate histogram**

---

## 📁 Output File Structure

After running examples, you'll have:

```
output/
├── 01_neuron_comparison.png      # LIF vs HH neuron analysis
├── 02_network_topologies.png     # Network structure comparisons  
├── 03_probe_monitoring.png       # Comprehensive monitoring data
├── 04_dmn_analysis.png           # Default mode network dynamics
├── 05_self_awareness.png         # Self-aware network behavior
├── quickstart_simulation.png     # Basic simulation results
├── quickstart_results.json       # Numerical data export
└── README.txt                    # Generated results summary
```

---

## 🎯 **To Generate These Outputs:**

### **Prerequisites:**
```bash
# Install the package first
./scripts/setup_development.sh
# or
poetry install --with dev
```

### **Run Examples:**
```bash
# Basic neuron demo
python examples/01_basic_neuron_demo.py

# Network topologies  
python examples/02_network_topologies.py

# Probe monitoring
python examples/03_probe_monitoring.py

# Default mode networks
python examples/04_default_mode_networks.py

# Self-aware networks
python examples/05_self_aware_network.py

# Quick start
python examples/quickstart_simulation.py
```

### **Interactive Features:**
- **Real-time plotting** (if display available)
- **Adjustable parameters** in script headers
- **Exportable data** in multiple formats
- **Publication-ready figures**

---

## 🎨 **Visualization Features:**

### **Professional Plots:**
- **High DPI** (300 DPI) for publications
- **Color-coded** data series
- **Grid lines** and proper labeling
- **Legends** and annotations
- **Multiple subplots** for comparison

### **Interactive Elements:**
- **Matplotlib viewers** (zoom, pan, save)
- **Real-time updates** during simulation
- **Parameter adjustment** capabilities

### **Data Export:**
- **PNG images** for presentations
- **JSON data** for further analysis
- **CSV exports** for spreadsheet analysis

**The examples are designed to provide immediate visual feedback and comprehensive analysis of neural dynamics!** 🎉
