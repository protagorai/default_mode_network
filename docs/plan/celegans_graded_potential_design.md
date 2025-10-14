# C. elegans Graded Potential Neural Network - Design Document

**Version:** 1.0  
**Date:** October 13, 2025  
**Status:** Design Phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Biological Background](#biological-background)
3. [Project Goals](#project-goals)
4. [Requirements](#requirements)
5. [Architecture Design](#architecture-design)
6. [Component Specifications](#component-specifications)
7. [Network Architectures](#network-architectures)
8. [Experimental Framework](#experimental-framework)
9. [Implementation Plan](#implementation-plan)
10. [References](#references)

---

## 1. Executive Summary

This document outlines the design for implementing biologically-accurate graded potential neurons and network architectures inspired by *C. elegans*, the nematode worm with a fully-mapped 302-neuron connectome. Unlike traditional spiking neuron models, *C. elegans* neurons primarily operate using graded (analog) voltage changes rather than discrete action potentials.

**Key Innovation:** Most computational neuroscience frameworks focus on spiking neurons. This implementation will create a parallel track for graded potential neurons with chemical and electrical (gap junction) synapses, enabling realistic *C. elegans* simulations.

---

## 2. Biological Background

### 2.1 C. elegans Nervous System

- **302 neurons** in adult hermaphrodite (385 in male)
- **~7,000 chemical synapses**
- **~600 electrical gap junctions**
- **Fully mapped connectome** (White et al., 1986)
- **Small neurons:** 2-5 μm diameter, simple morphology

### 2.2 Graded vs. Spiking Neurons

**Graded Potential Characteristics:**
- Continuous, analog voltage changes
- Amplitude proportional to stimulus strength
- Can summate linearly (no refractory period)
- Propagate with decrement (cable properties important)
- Enable smooth, continuous control signals

**Why C. elegans Uses Graded Potentials:**
- Very short distances (~1 mm worm length)
- Energy efficient for small nervous system
- No need for long-distance signal regeneration
- Enables fine-grained motor control

### 2.3 Key Biophysical Properties

**Membrane Properties:**
- Capacitance: 1-5 pF (very small!)
- Resting potential: -60 to -70 mV
- Input resistance: 100-500 MΩ (high)
- Time constant: 10-20 ms (typical), can be lower

**Ion Channels (Goodman et al., 1998; Lockery & Goodman, 2009):**
- **Leak channels:** Set resting potential
- **Calcium channels:** Main depolarizing current (Ca²⁺)
- **Potassium channels:** Repolarization and adaptation
- **Ca²⁺-dependent K⁺ channels:** Negative feedback, adaptation
- **Minimal/no sodium channels** in most neurons

**Synaptic Properties:**
- **Graded chemical transmission:** Vesicle release proportional to voltage
- **No action potential threshold** for release
- **Fast kinetics:** τ_rise ~ 1-2 ms, τ_decay ~ 5-15 ms
- **Gap junctions:** Bidirectional electrical coupling, ~100 pS - 1 nS

### 2.4 The Small-World Network Discovery

**Critical Finding (Watts & Strogatz, 1998):**
- Regular lattice (uniform distribution): Poor performance, long path lengths
- Random rewiring creates "small-world" topology
- **C. elegans connectome is a small-world network:**
  - High local clustering
  - Short average path length
  - Efficient information transmission
  - Robustness to damage

**Implications:**
- Average ~14 synapses/neuron, but highly non-uniform
- Rich club hubs (heavily connected neurons)
- Modular organization
- Critical for emergent behavior

---

## 3. Project Goals

### 3.1 Primary Goals

1. **Implement biologically-accurate graded potential neuron models**
   - Based on published electrophysiological data
   - Configurable parameters for different neuron classes
   - Efficient computation for networks of 300+ neurons

2. **Create graded synaptic transmission system**
   - Chemical synapses with graded release
   - Electrical gap junctions
   - Both excitatory and inhibitory types

3. **Build network architecture tools**
   - Uniform (lattice-like) networks for baseline
   - Small-world networks with configurable rewiring
   - Load real *C. elegans* connectome data

4. **Experimental framework**
   - Topology analysis tools
   - Behavioral emergence metrics
   - Information flow analysis

### 3.2 Secondary Goals

1. Compare uniform vs. small-world network performance
2. Investigate emergent dynamics and oscillations
3. Model specific *C. elegans* behaviors (chemotaxis, locomotion)
4. Validate against OpenWorm results

---

## 4. Requirements

### 4.1 Functional Requirements

**FR1: Graded Potential Neuron Model**
- Must implement continuous membrane voltage dynamics
- Must include Ca²⁺, K⁺, and leak currents
- Must support external current injection
- Time constants must be configurable (< 10 ms possible)
- Must integrate with existing simulation engine

**FR2: Chemical Synapses (Graded)**
- Must transmit signals proportional to presynaptic voltage
- Must support excitatory and inhibitory types
- Must have configurable rise/decay kinetics
- Must support synaptic plasticity (future)

**FR3: Gap Junctions**
- Must implement bidirectional electrical coupling
- Must support heterogeneous coupling strengths
- Must be computationally efficient

**FR4: Network Construction**
- Must support 302+ neuron networks
- Must implement uniform topology
- Must implement small-world topology with rewiring
- Must load connectome from external data files

**FR5: Analysis and Visualization**
- Must compute network topology metrics (clustering, path length, degree distribution)
- Must track voltage traces for all neurons
- Must generate connectivity matrices
- Must visualize network structure

### 4.2 Non-Functional Requirements

**NFR1: Performance**
- Must simulate 302 neurons in real-time or faster
- Must support simulation time steps down to 0.01 ms

**NFR2: Accuracy**
- Parameters must match published experimental data
- Must reproduce known *C. elegans* dynamics when possible

**NFR3: Extensibility**
- Must allow easy addition of new neuron classes
- Must support future addition of neuromodulators

**NFR4: Integration**
- Must integrate with existing SDMN framework
- Must use existing probe and visualization systems

---

## 5. Architecture Design

### 5.1 Module Structure

```
src/sdmn/
├── neurons/
│   ├── __init__.py
│   ├── base_neuron.py          # Existing
│   ├── lif_neuron.py           # Existing
│   ├── hh_neuron.py            # Existing
│   └── graded/                 # NEW MODULE
│       ├── __init__.py
│       ├── graded_neuron.py    # Base graded potential neuron
│       ├── celegans_neuron.py  # C. elegans specific implementation
│       └── neuron_classes.py   # Different C. elegans neuron types
│
├── synapses/                   # NEW MODULE (promote from neurons/)
│   ├── __init__.py
│   ├── base_synapse.py         # Abstract base
│   ├── synapse.py              # Existing spike-triggered (moved)
│   ├── graded_synapse.py       # Graded chemical synapse
│   └── gap_junction.py         # Electrical synapse
│
├── networks/
│   ├── __init__.py
│   ├── network_builder.py      # Existing
│   └── celegans/               # NEW MODULE
│       ├── __init__.py
│       ├── topology.py         # Network topology generators
│       ├── connectome_loader.py # Load real connectome data
│       └── small_world.py      # Small-world network builder
│
└── analysis/                   # NEW MODULE
    ├── __init__.py
    ├── network_metrics.py      # Topology analysis
    ├── information_flow.py     # Signal propagation analysis
    └── behavior_analysis.py    # Emergent behavior metrics
```

### 5.2 Class Hierarchy

```
BaseNeuron (abstract)
├── LIFNeuron (existing)
├── HodgkinHuxleyNeuron (existing)
└── GradedNeuron (new)
    └── CElegansNeuron (new)
        ├── MotorNeuron
        ├── Interneuron
        └── SensoryNeuron

BaseSynapse (abstract)
├── SpikeSynapse (existing, for LIF/HH)
├── GradedChemicalSynapse (new)
│   ├── ExcitatorySynapse
│   └── InhibitorySynapse
└── GapJunction (new)
```

---

## 6. Component Specifications

### 6.1 Graded Neuron Model

#### Mathematical Model

Based on single-compartment conductance model:

```
C_m * dV/dt = -I_leak - I_Ca - I_K - I_KCa + I_ext + I_syn + I_gap

Where:
I_leak = g_leak * (V - E_leak)
I_Ca = g_Ca * m_Ca^2 * (V - E_Ca)
I_K = g_K * m_K^4 * (V - E_K)
I_KCa = g_KCa * m_KCa * (V - E_K)

Gating variables follow first-order kinetics:
dm/dt = (m_inf(V) - m) / τ_m(V)

m_inf(V) = 1 / (1 + exp(-(V - V_half) / k))
```

#### Parameters (from Lockery & Goodman, 2009; Liu et al., 2018)

**Membrane Properties:**
```python
C_m = 3.0 pF                    # Membrane capacitance
g_leak = 0.3 nS                 # Leak conductance
E_leak = -65.0 mV               # Leak reversal potential
```

**Calcium Current:**
```python
g_Ca = 0.8 nS                   # Max calcium conductance
E_Ca = +50.0 mV                 # Calcium reversal potential
V_half_Ca = -20.0 mV            # Half-activation voltage
k_Ca = 5.0 mV                   # Slope factor
τ_Ca_min = 0.5 ms               # Minimum time constant
τ_Ca_max = 5.0 ms               # Maximum time constant
```

**Potassium Current:**
```python
g_K = 1.5 nS                    # Max potassium conductance
E_K = -80.0 mV                  # Potassium reversal potential
V_half_K = -25.0 mV             # Half-activation voltage
k_K = 10.0 mV                   # Slope factor
τ_K_min = 1.0 ms                # Minimum time constant
τ_K_max = 10.0 ms               # Maximum time constant
```

**Calcium-Dependent Potassium Current:**
```python
g_KCa = 0.5 nS                  # Max Ca-dependent K conductance
[Ca]_half = 100 nM              # Half-activation calcium
τ_KCa = 50 ms                   # Time constant
```

**Numerical Integration:**
```python
dt_default = 0.01 ms            # Time step (10 μs)
dt_min = 0.001 ms               # Minimum allowed (1 μs)
dt_max = 0.1 ms                 # Maximum recommended
method = "RK4"                  # Runge-Kutta 4th order (recommended)
                                # Alternative: "Euler" (faster, less accurate)
```

**Voltage-Dependent Time Constants:**
```python
# Time constant varies with voltage for realistic dynamics
τ_m(V) = τ_min + (τ_max - τ_min) / (1 + exp((V - V_half) / k_tau))

# For most channels:
V_half_tau = -30.0 mV           # Voltage of half-maximal time constant
k_tau = 10.0 mV                 # Slope factor
```

**Intracellular Calcium Dynamics:**
```python
# Required for Ca-dependent K channels
d[Ca]/dt = -f * (I_Ca / (2 * F * vol)) - ([Ca] - [Ca]_rest) / τ_Ca_removal

[Ca]_rest = 50 nM               # Resting calcium concentration
τ_Ca_removal = 100 ms           # Calcium removal time constant
f = 0.01                        # Fraction of free calcium
vol = 1 pL                      # Cell volume (approximate)
F = 96485 C/mol                 # Faraday constant
```

#### Neuron Classes

Different *C. elegans* neuron types have different channel densities:

**Sensory Neurons:** High Ca²⁺, low K⁺ (excitable)
**Interneurons:** Balanced (integrative)
**Motor Neurons:** High K⁺, moderate Ca²⁺ (graded output)

### 6.2 Graded Chemical Synapse

#### Mathematical Model

```
I_syn(t) = g_syn(t) * (V_post - E_syn)

Presynaptic release (graded):
release(V_pre) = 1 / (1 + exp(-(V_pre - V_thresh) / k_syn))

Synaptic conductance (dual exponential):
dg_syn/dt = release(V_pre) * w - g_syn/τ_decay

Or alpha function:
g_syn(t) = g_max * (t/τ) * exp(1 - t/τ)
```

#### Parameters

**General:**
```python
V_thresh = -40.0 mV             # Threshold for release
k_syn = 5.0 mV                  # Sigmoid slope
w_default = 1.0 nS              # Default synaptic weight
```

**Excitatory (Glutamate-like):**
```python
E_syn_exc = 0.0 mV              # Reversal potential
τ_rise = 1.0 ms                 # Rise time
τ_decay = 5.0 ms                # Decay time
```

**Inhibitory (GABA-like):**
```python
E_syn_inh = -75.0 mV            # Reversal potential
τ_rise = 0.5 ms                 # Rise time (faster)
τ_decay = 10.0 ms               # Decay time
```

**Synaptic Delays:**
```python
delay_default = 0.5 ms          # Typical chemical synapse delay
delay_min = 0.3 ms              # Fast synapses
delay_max = 2.0 ms              # Slow synapses
# Note: Gap junctions have negligible delay (~0 ms)
```

**Synaptic Noise (optional, for realism):**
```python
noise_std = 0.1                 # Standard deviation of release probability
# Adds stochasticity: release(V) + noise_std * randn()
```

### 6.3 Gap Junction

#### Mathematical Model

Ohmic coupling:
```
I_gap,i = Σ_j g_gap,ij * (V_j - V_i)

Where:
- i is the neuron index
- j are all neurons coupled to i
- g_gap,ij is the coupling conductance (symmetric)
```

#### Parameters

```python
g_gap_default = 0.5 nS          # Default coupling strength
g_gap_min = 0.1 nS              # Weak coupling
g_gap_max = 2.0 nS              # Strong coupling
```

**Note:** Gap junctions are bidirectional and symmetric in strength.

---

## 7. Network Architectures

### 7.1 Uniform/Regular Network

**Structure:**
- N neurons arranged in regular topology
- Each neuron connected to K nearest neighbors
- Uniform synaptic weights
- No hubs or special structure

**Purpose:** Baseline for comparison, expected poor performance

**Why Uniform Networks Fail:**
- **Long path lengths:** Information must traverse many hops to reach distant neurons
- **Poor global integration:** Local clusters cannot communicate efficiently
- **Slow signal propagation:** Average path length L ~ N/K scales poorly
- **No hubs:** Cannot broadcast signals network-wide quickly
- **Predictable dynamics:** Lack of rich, complex behavior
- **Example:** In a ring with 302 neurons and K=14, average path length ~11 hops
  vs. ~3 hops in small-world network

**Parameters:**
```python
N = 302                         # Number of neurons
K_chem = 14                     # Average chemical synapses per neuron
K_gap = 4                       # Average gap junctions per neuron
topology = "ring" | "lattice"   # Spatial arrangement
```

### 7.2 Small-World Network (Watts-Strogatz)

**Algorithm:**
1. Start with regular lattice (K connections per neuron)
2. For each edge, rewire with probability p
3. Rewiring: Keep one endpoint, randomly choose new endpoint
4. Avoid self-loops and duplicate connections

**Parameters:**
```python
N = 302
K = 14                          # Initial regular connections
p_rewire = 0.1 to 0.3          # Rewiring probability (tunable)
```

**Expected Properties:**
- High clustering coefficient (C >> C_random)
- Short average path length (L ≈ L_random)
- Creates "small-world" regime

### 7.3 Scale-Free Network (Barabási-Albert)

**Algorithm:**
- Preferential attachment: "rich get richer"
- New neurons connect to existing neurons with probability proportional to their degree

**Purpose:** Test hub-based architecture

### 7.4 Empirical C. elegans Connectome

**Data Sources:**
- WormAtlas (http://www.wormatlas.org/)
- OpenWorm connectome data
- Varshney et al. (2011) refined connectome

**Format:**
```csv
pre_neuron, post_neuron, synapse_type, weight
AVAL, VA01, chemical_exc, 3
AVAL, AVAR, gap_junction, 2
...
```

**Neuron Classes:**
- Sensory: AWC, ASE, etc. (sensors)
- Interneurons: AVA, AVB, PVC, etc. (command)
- Motor: DA, DB, VD, VB, etc. (output)

---

## 8. Experimental Framework

### 8.1 Topology Analysis

**Metrics to Compute:**

1. **Degree Distribution**
   - In-degree, out-degree histograms
   - Hub identification

2. **Clustering Coefficient**
   ```
   C = (# of triangles) / (# of possible triangles)
   ```

3. **Average Path Length**
   ```
   L = average shortest path between all pairs
   ```

4. **Small-World Coefficient**
   ```
   σ = (C / C_random) / (L / L_random)
   σ > 1 indicates small-world
   ```

5. **Modularity**
   - Community detection
   - Q-value (Newman modularity)

6. **Rich Club Coefficient**
   - Connectivity among high-degree nodes

### 8.2 Dynamical Analysis

**Metrics to Compute:**

1. **Signal Propagation Speed**
   - Stimulus at sensory neurons → response at motor neurons
   - Latency measurement

2. **Information Transfer**
   - Transfer entropy between neuron pairs
   - Mutual information

3. **Synchronization**
   - Cross-correlation of voltage traces
   - Phase locking analysis

4. **Attractor States**
   - PCA of voltage trajectories
   - Fixed point/limit cycle identification

5. **Sensitivity to Perturbations**
   - Lesion studies (remove neurons/synapses)
   - Noise robustness

### 8.3 Behavioral Emergence

**Target Behaviors (simplified models):**

1. **Forward Locomotion**
   - Coordinated motor neuron activation
   - Traveling wave in ventral/dorsal motor neurons

2. **Reversals**
   - AVA/AVB command neuron competition
   - State switching dynamics

3. **Chemotaxis**
   - Sensory input → turn modulation
   - Gradient climbing

### 8.4 Network Initialization

**Initial Conditions:**
```python
# Voltage initialization
V_init_mean = -65.0 mV          # Mean initial voltage
V_init_std = 2.0 mV             # Standard deviation (heterogeneity)
# Each neuron: V(0) ~ N(V_init_mean, V_init_std^2)

# Gating variables at steady state for V_init
m_init = m_inf(V_init)

# Calcium concentration
[Ca]_init = [Ca]_rest           # Start at resting

# Synaptic conductances
g_syn_init = 0.0                # No initial synaptic input

# Equilibration period
t_equilibrate = 100 ms          # Let network settle before experiment
```

**Noise Sources (optional):**
```python
# Channel noise (intrinsic)
noise_channel = 0.1 mV/√ms      # Voltage noise amplitude

# Synaptic noise (see section 6.2)
noise_syn = 0.1                 # Release probability noise

# External noise (background input)
I_noise_mean = 0.0 pA           # Mean background current
I_noise_std = 0.05 pA           # Std of background current
```

### 8.5 Computational Considerations

**Time Complexity:**
- **Single neuron update:** O(1) - fixed number of state variables
- **Chemical synapses:** O(S) where S = number of synapses
- **Gap junctions:** O(G) where G = number of gap junctions
- **Full network step:** O(N + S + G) ≈ O(N) for fixed connectivity
- **Per second simulation:** O(N * T/dt) 
  - For 302 neurons, dt=0.01ms, T=1s: ~3M operations

**Memory Complexity:**
- **Neuron states:** ~10 floats/neuron × 302 ≈ 12 KB
- **Synapse states:** ~5 floats/synapse × 7000 ≈ 140 KB
- **Voltage history:** (if recorded) N × (T/dt) × 8 bytes
  - For 1s @ 0.01ms: 302 × 100,000 × 8 ≈ 240 MB
- **Total:** < 500 MB for typical simulation (very manageable)

**Performance Targets:**
- Real-time or faster for 302 neurons (1 s simulation in < 1 s wall time)
- On modern CPU: ~1-10 ms per network time step (dt=0.01ms) → 10-100× real-time
- Easily parallelizable: neuron updates independent, gap junction vectorizable

**Optimization Strategies:**
- Use NumPy vectorization for neuron population updates
- Pre-compute sigmoid lookup tables for release functions
- Sparse matrix representation for connectivity
- JIT compilation (Numba/JAX) for critical loops

### 8.6 Experimental Workflow

```
1. Network Construction
   ├── Define topology (uniform/small-world/empirical)
   ├── Instantiate neurons (with class-specific parameters)
   └── Create synapses and gap junctions

2. Initialization
   ├── Set initial conditions (resting state with noise)
   ├── Equilibrate network (100 ms)
   └── Configure stimulus protocol

3. Simulation
   ├── Run for T seconds (e.g., 1-10 s)
   ├── Record voltages, currents, connectivity
   └── Apply stimuli (current injection, sensory input)

4. Analysis
   ├── Compute topology metrics
   ├── Analyze dynamics
   ├── Identify emergent patterns
   └── Compare to baseline (uniform network)

5. Visualization
   ├── Network graphs (spring layout, anatomical)
   ├── Voltage raster plots
   ├── Information flow diagrams
   └── Behavioral read-outs
```

---

## 9. Implementation Plan

### Phase 1: Core Components (Week 1)

**Tasks:**
1. Create `neurons/graded/` module
2. Implement `GradedNeuron` base class
3. Implement `CElegansNeuron` with full ion channel dynamics
4. Unit tests for single neuron dynamics

**Deliverables:**
- Functional graded neuron model
- Parameter validation against literature
- Documentation

### Phase 2: Synaptic Components (Week 1-2)

**Tasks:**
1. Refactor synapse into `synapses/` module
2. Implement `GradedChemicalSynapse`
3. Implement `GapJunction`
4. Test two-neuron coupling

**Deliverables:**
- Functional graded synapses
- Functional gap junctions
- Integration tests

### Phase 3: Network Construction (Week 2)

**Tasks:**
1. Create `networks/celegans/` module
2. Implement uniform network builder
3. Implement small-world network builder
4. Implement connectome data loader

**Deliverables:**
- Network generation tools
- Example networks (uniform, small-world)
- Topology validation

### Phase 4: Analysis Tools (Week 2-3)

**Tasks:**
1. Create `analysis/` module
2. Implement topology metrics
3. Implement information flow analysis
4. Create visualization utilities

**Deliverables:**
- Network analysis tools
- Plotting functions
- Comprehensive metrics

### Phase 5: Experiments (Week 3-4)

**Tasks:**
1. Uniform vs. small-world comparison
2. Parameter sensitivity analysis
3. Empirical connectome simulation
4. Behavior emergence experiments

**Deliverables:**
- Experimental scripts
- Results and plots
- Scientific report

### Phase 6: Documentation and Examples (Week 4)

**Tasks:**
1. Comprehensive documentation
2. Tutorial notebooks
3. Example gallery
4. Scientific validation report

**Deliverables:**
- User guide
- API documentation
- Published examples

---

## 10. References

### Key Papers

1. **White, J.G., et al. (1986)**  
   "The structure of the nervous system of the nematode Caenorhabditis elegans."  
   *Philosophical Transactions of the Royal Society B*, 314(1165), 1-340.  
   *The original connectome paper*

2. **Watts, D.J., & Strogatz, S.H. (1998)**  
   "Collective dynamics of 'small-world' networks."  
   *Nature*, 393(6684), 440-442.  
   *Small-world network discovery*

3. **Goodman, M.B., et al. (1998)**  
   "Active currents regulate sensitivity and dynamic range in C. elegans neurons."  
   *Neuron*, 20(4), 763-772.  
   *Electrophysiology and ion channels*

4. **Lockery, S.R., & Goodman, M.B. (2009)**  
   "The quest for action potentials in C. elegans neurons hits a plateau."  
   *Nature Neuroscience*, 12(4), 377-378.  
   *Graded vs. spiking neurons*

5. **Varshney, L.R., et al. (2011)**  
   "Structural properties of the Caenorhabditis elegans neuronal network."  
   *PLOS Computational Biology*, 7(2), e1001066.  
   *Refined connectome and network analysis*

6. **Liu, Q., et al. (2018)**  
   "C. elegans AWA olfactory neurons fire calcium-mediated all-or-none action potentials."  
   *Cell*, 175(1), 57-70.  
   *Some neurons DO spike! Calcium-based*

7. **Rakowski, F., et al. (2013)**  
   "Synaptic polarity of the interneuron circuit controlling C. elegans locomotion."  
   *Frontiers in Computational Neuroscience*, 7, 128.  
   *Locomotion circuit model*

8. **Gleeson, P., et al. (2018)**  
   "c302: a multiscale framework for modelling the nervous system of Caenorhabditis elegans."  
   *Philosophical Transactions of the Royal Society B*, 373(1758), 20170379.  
   *OpenWorm c302 project - state of the art*

### Online Resources

- **OpenWorm Project:** https://openworm.org/
- **WormAtlas:** http://www.wormatlas.org/
- **WormBase:** https://wormbase.org/
- **c302 GitHub:** https://github.com/openworm/c302
- **NeuroML DB:** https://neuroml-db.org/

---

## Appendix A: Comparison with Existing Models

| Aspect | LIF Neuron | HH Neuron | Graded Neuron (New) |
|--------|-----------|-----------|---------------------|
| Signal type | Spike | Spike | Continuous |
| Threshold | Yes | Emergent | No |
| Ion channels | Implicit | Na⁺, K⁺ | Ca²⁺, K⁺, leak |
| Refractory period | Yes | Yes | No |
| Summation | Integrate-and-fire | Nonlinear | Linear-ish |
| Computation | Fast | Medium | Fast |
| Biological match | Abstract | Squid axon | C. elegans |

---

## Appendix B: Code Interface Examples

### Example 1: Create a Graded Neuron

```python
from sdmn.neurons.graded import CElegansNeuron

# Create interneuron with default parameters
neuron = CElegansNeuron(
    neuron_id="AVA",
    neuron_class="interneuron",
    tau_min=5.0,  # Fast dynamics
)

# Inject current and simulate
neuron.set_external_current(0.1)  # nA
for t in range(1000):  # 10 ms at 0.01 ms steps
    neuron.step(dt=0.01)
    print(f"V = {neuron.voltage:.2f} mV")
```

### Example 2: Create Graded Synapse

```python
from sdmn.synapses import GradedChemicalSynapse

synapse = GradedChemicalSynapse(
    pre_neuron=sensory,
    post_neuron=inter,
    synapse_type="excitatory",
    weight=1.5,  # nS
    tau_rise=1.0,
    tau_decay=5.0
)
```

### Example 3: Create Gap Junction

```python
from sdmn.synapses import GapJunction

gap = GapJunction(
    neuron_a=AVA,
    neuron_b=AVAR,
    conductance=0.8  # nS, bidirectional
)
```

### Example 4: Build Small-World Network

```python
from sdmn.networks.celegans import SmallWorldBuilder

builder = SmallWorldBuilder(
    n_neurons=302,
    k_neighbors=14,
    rewire_prob=0.2,
    gap_junction_prob=0.15
)

network = builder.build()
network.simulate(duration=5000)  # 5 seconds
network.analyze_topology()
```

---

## Appendix C: Validation Criteria

To ensure biological accuracy, we will validate against:

1. **Single Neuron:**
   - Voltage response to step current matches literature
   - Time constants in expected range (5-20 ms)
   - No spiking (except rare calcium spikes)

2. **Two-Neuron:**
   - Graded synaptic transmission shows sigmoid relationship
   - Gap junction produces symmetric coupling
   - Temporal dynamics match experimental data

3. **Network:**
   - Small-world metrics: C_network/C_random > 1, L_network ≈ L_random
   - Degree distribution matches empirical connectome
   - Signal propagation speed: 10-100 mm/s (for worm scale)

4. **Behavior (if implemented):**
   - Forward locomotion frequency: ~0.3 Hz
   - Reversal duration: ~2-3 seconds
   - Chemotaxis: biased random walk toward attractant

---

## Appendix D: Implementation Best Practices

### State Management

**Neuron State Variables:**
```python
class GradedNeuron:
    # Core state (must be tracked)
    voltage: float          # Membrane voltage (mV)
    m_Ca: float            # Calcium channel activation
    m_K: float             # Potassium channel activation
    m_KCa: float           # Ca-dependent K activation
    Ca_internal: float     # Intracellular calcium (nM)
    
    # Input accumulation
    I_syn: float           # Total synaptic current
    I_gap: float           # Total gap junction current
    I_ext: float           # External injected current
```

**Synapse State Variables:**
```python
class GradedSynapse:
    g_syn: float           # Synaptic conductance (nS)
    release: float         # Current release probability
    delay_buffer: deque    # For synaptic delays
```

### Numerical Stability

**Safeguards:**
```python
# Clip extreme voltages
V = np.clip(V, -100, +100)  # mV

# Clip gating variables to [0, 1]
m = np.clip(m, 0, 1)

# Clip calcium to physiological range
Ca = np.clip(Ca, 0, 10000)  # nM (0-10 μM)

# Prevent division by zero in time constants
tau = np.maximum(tau, 0.001)  # ms
```

**Adaptive Time Step (optional):**
```python
# Reduce dt if voltage changes too rapidly
dV_max = 5.0  # mV per step
if abs(dV) > dV_max:
    dt_adaptive = dt * dV_max / abs(dV)
else:
    dt_adaptive = dt
```

### Code Organization

**Separation of Concerns:**
1. **Neuron models** (`neurons/graded/`) - Only neuron dynamics
2. **Synapses** (`synapses/`) - Only synaptic transmission
3. **Networks** (`networks/celegans/`) - Only connectivity/topology
4. **Simulation** (`core/simulation_engine.py`) - Only time evolution
5. **Analysis** (`analysis/`) - Only post-processing

**Interface Contracts:**
```python
# All neurons must implement:
class BaseNeuron(ABC):
    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance neuron by one time step"""
        
    @abstractmethod
    def add_current(self, current: float) -> None:
        """Add external/synaptic current"""
        
    @property
    @abstractmethod
    def voltage(self) -> float:
        """Get current membrane voltage"""
```

### Testing Strategy

**Unit Tests:**
- Single neuron: step current → voltage response
- Synapse: presynaptic voltage → postsynaptic current
- Gap junction: voltage difference → bidirectional current

**Integration Tests:**
- Two coupled neurons (synapse + gap junction)
- Small network (10 neurons) with known topology
- Verify energy conservation (if no external input)

**Validation Tests:**
- Compare to published data (voltage traces, time constants)
- Reproduce OpenWorm results (if available)
- Network metrics vs. analytical predictions

### Documentation Standards

**Code Documentation:**
```python
def sigmoid_activation(V: float, V_half: float, k: float) -> float:
    """
    Boltzmann sigmoid activation function for ion channels.
    
    Args:
        V: Membrane voltage (mV)
        V_half: Half-activation voltage (mV)
        k: Slope factor (mV), positive for activation
        
    Returns:
        Activation value in range [0, 1]
        
    Reference:
        Hodgkin & Huxley (1952), standard voltage-gated channel formulation
    """
    return 1.0 / (1.0 + np.exp(-(V - V_half) / k))
```

**Parameter Documentation:**
- Always include units in comments
- Always include literature reference for parameters
- Always include physiological range and typical values

---

## Appendix E: Troubleshooting Guide

### Common Issues

**1. Network Doesn't Show Emergent Behavior**
- Check: Is it small-world or uniform? (Uniform won't work)
- Check: Are synaptic weights strong enough?
- Check: Is there sufficient heterogeneity (noise)?
- Check: Are you waiting long enough (transients)?

**2. Numerical Instability**
- Reduce dt (try 0.005 ms instead of 0.01 ms)
- Use RK4 instead of Euler
- Add voltage/state clipping
- Check for unrealistic parameter values

**3. Too Slow Performance**
- Profile code to find bottleneck
- Vectorize neuron updates using NumPy
- Use sparse matrices for connectivity
- Consider JIT compilation (Numba)
- Reduce recording frequency

**4. Results Don't Match Literature**
- Double-check parameter units (ms vs s, pA vs nA)
- Verify parameter values against original papers
- Check if you're comparing same neuron types
- Ensure proper initialization and equilibration

**5. Gap Junctions Not Working**
- Verify bidirectional coupling (symmetric matrix)
- Check conductance values (too weak?)
- Ensure voltage differences exist
- Plot individual gap junction currents

---

**End of Design Document**


