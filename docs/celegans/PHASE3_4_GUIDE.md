# C. elegans Phase 3-4: Topology Builders, Analysis Tools & Full Connectome Guide

## Overview

Phases 3 and 4 complete the C. elegans simulation pipeline:

| Component | Module | Purpose |
|-----------|--------|---------|
| **ScaleFreeBuilder** | `sdmn.networks.celegans` | Barabasi-Albert preferential-attachment topology |
| **ConnectomeLoader** | `sdmn.networks.celegans` | Load real 302-neuron connectome from CSV data |
| **TopologyAnalyzer** | `sdmn.analysis` | Graph-theoretic metrics (degree, clustering, path length, modularity) |
| **DynamicsAnalyzer** | `sdmn.analysis` | Post-simulation analysis (spectral, PCA, synchronization, information theory) |
| **BehaviorDetector** | `sdmn.analysis` | C. elegans behavioral pattern detection (locomotion, reversals) |
| **NetworkVisualizer** | `sdmn.analysis` | Matplotlib plotting utilities for all of the above |

---

## Phase 3: Topology Builders

### Scale-Free Network (Barabasi-Albert)

```python
from sdmn.networks.celegans import build_scale_free_network

network = build_scale_free_network(
    n_neurons=302,
    m=7,              # edges per new node (avg degree ~2*m)
    random_seed=42,
)
```

Or using the class directly:

```python
from sdmn.networks.celegans import ScaleFreeBuilder, TopologyParameters

params = TopologyParameters(n_neurons=302, k_neighbors=14, random_seed=42)
builder = ScaleFreeBuilder(params, m=7, m0=8)
network = builder.build()
```

### Real Connectome Loader

```python
from sdmn.networks.celegans import ConnectomeLoader, build_connectome_network

# One-liner
network = build_connectome_network()

# Or with full control
loader = ConnectomeLoader()
data = loader.load()                          # uses bundled CSVs

print(f"Neurons: {data.n_neurons}")
print(f"Chemical synapses: {data.n_chemical}")
print(f"Gap junctions: {data.n_gap}")

# Build network with custom weight scaling
network = loader.build_network(weight_scale=1.5, gap_conductance_scale=0.8)

# Adjacency matrix
mat, names = loader.get_adjacency_matrix(conn_type="chemical")

# Filter to a subcircuit
subset = loader.filter(neuron_names=["AVAL", "AVAR", "AVBL", "AVBR"], min_weight=2)
```

### Bundled Data

The connectome data lives in `data/connectome/`:

- **`celegans_neurons.csv`** -- 302 neurons with class (sensory/interneuron/motor)
- **`celegans_connectome.csv`** -- edges with type (chemical/gap), weight, and neurotransmitter

Sources: Varshney et al. (2011), Cook et al. (2019), OpenWorm.

---

## Phase 4: Analysis Tools

### Topology Analyzer

```python
from sdmn.analysis import TopologyAnalyzer

analyzer = TopologyAnalyzer.from_network(network)

# Individual metrics
cc = analyzer.clustering_coefficient()
apl = analyzer.average_path_length()
sigma = analyzer.small_world_coefficient()
hubs = analyzer.hub_neurons(top_n=10)
communities = analyzer.detect_communities()
q = analyzer.modularity()

# Everything at once
report = analyzer.full_report()
```

### Dynamics Analyzer

Requires a simulated network (call `network.simulate()` first):

```python
from sdmn.analysis import DynamicsAnalyzer

network.simulate(duration=500, progress=True)
dyn = DynamicsAnalyzer.from_simulation(network)

# Synchronization
corr = dyn.pairwise_correlation_matrix()
sync = dyn.mean_synchronization()

# Spectral analysis
freqs, psd = dyn.power_spectrum("AVAL")
peak_hz = dyn.dominant_frequency("AVAL")

# Dimensionality reduction
traj, explained = dyn.pca_trajectories(n_components=3)

# Information theory
te = dyn.transfer_entropy("AWCL", "AVAL", lag=1)
mi = dyn.mutual_information("AVAL", "AVAR")

# Signal propagation
delay_ms = dyn.propagation_delay("AWCL", "AVAL")
```

### Behavior Detector

```python
from sdmn.analysis import BehaviorDetector

detector = BehaviorDetector.from_simulation(network)

fwd_bouts = detector.detect_forward_locomotion()
reversals = detector.detect_reversals()
states = detector.classify_activity_state(window_ms=50)
latency = detector.stimulus_response_latency(0.0, ["DA01", "DA02"])

report = detector.behavioral_report()
```

### Visualization

```python
from sdmn.analysis import NetworkVisualizer, TopologyAnalyzer, DynamicsAnalyzer

topo = TopologyAnalyzer.from_network(network)
dyn = DynamicsAnalyzer.from_simulation(network)

# Network structure
fig = NetworkVisualizer.plot_network_graph(topo, layout="spring", color_by="class")
fig = NetworkVisualizer.plot_degree_distribution(topo)
fig = NetworkVisualizer.plot_community_structure(topo)

# Simulation results
fig = NetworkVisualizer.plot_voltage_raster(dyn)
fig = NetworkVisualizer.plot_voltage_traces(dyn, ["AVAL", "AVBL", "DA01"])
fig = NetworkVisualizer.plot_correlation_matrix(dyn)
fig = NetworkVisualizer.plot_pca_trajectory(dyn)
fig = NetworkVisualizer.plot_power_spectrum(dyn, ["AVAL"])

# Topology comparison across architectures
reports = {"connectome": topo.full_report(), "random": random_topo.full_report()}
fig = NetworkVisualizer.plot_topology_comparison(reports)
```

All plot functions return `matplotlib.figure.Figure` -- call `fig.savefig(...)` or `plt.show()`.

---

## Full Connectome Example

The capstone demo at `examples/09_full_connectome_simulation.py` runs the complete pipeline:

```bash
python examples/09_full_connectome_simulation.py
```

Steps:
1. Load 302-neuron connectome
2. Build network with correct neuron classes
3. Topology analysis (degree, clustering, path length, modularity, hubs)
4. Simulate 200 ms with sensory stimulation (AWCL, AWCR, ASEL, ASER)
5. Dynamical analysis (synchronization, PCA, propagation delay)
6. Behavioral detection (locomotion, reversals, activity states)
7. Save plots to `output/09_*.png`

### Performance

On a modern CPU, 200 ms of simulation (20,000 steps at dt=0.01 ms) with 302 neurons
completes in roughly 30-120 seconds depending on hardware. The analysis and
visualisation steps add a few more seconds.

---

## Running Tests

```bash
# Phase 3 tests (builders, connectome loader)
pytest tests/test_celegans_phase3.py -v

# Phase 4 tests (analysis module)
pytest tests/test_analysis.py -v
```

---

## File Inventory

### New source files
- `src/sdmn/networks/celegans/connectome_loader.py`
- `src/sdmn/analysis/__init__.py`
- `src/sdmn/analysis/topology.py`
- `src/sdmn/analysis/dynamics.py`
- `src/sdmn/analysis/behavior.py`
- `src/sdmn/analysis/visualization.py`

### Data
- `data/connectome/celegans_neurons.csv`
- `data/connectome/celegans_connectome.csv`
- `scripts/generate_connectome_data.py`

### Tests
- `tests/test_celegans_phase3.py`
- `tests/test_analysis.py`

### Examples
- `examples/09_full_connectome_simulation.py`
