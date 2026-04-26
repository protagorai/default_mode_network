# SDMN Development Status

**Last updated:** April 12, 2026
**Test status:** 138 tests passing across 6 test files

---

## Architecture Overview

```
src/sdmn/
├── core/                       # SimulationEngine, TimeManager, EventQueue, StateManager
├── neurons/
│   ├── base_neuron.py          # BaseNeuron ABC, NeuronFactory
│   ├── lif_neuron.py           # Leaky integrate-and-fire
│   ├── hh_neuron.py            # Hodgkin-Huxley
│   ├── synapse.py              # Spike-triggered synapses
│   └── graded/                 # C. elegans graded-potential neurons
│       ├── graded_neuron.py    # GradedNeuron base (RK4/Euler, conductance-based)
│       ├── celegans_neuron.py  # CElegansNeuron (Ca/K/KCa channels)
│       ├── neuron_classes.py   # SensoryNeuron, Interneuron, MotorNeuron
│       ├── multichannel_neuron.py  # MultiChannelNeuron (per-type ion channel complements)
│       └── neuropeptide_layer.py   # ExtrasynapticLayer (neuropeptide/monoamine signaling)
├── synapses/
│   ├── graded_synapse.py       # GradedChemicalSynapse (voltage-dependent release)
│   └── gap_junction.py         # GapJunction (bidirectional electrical coupling)
├── networks/
│   ├── network_builder.py      # Generic NetworkBuilder (LIF/HH spiking networks)
│   └── celegans/
│       ├── network_manager.py  # CElegansNetwork, SimulationConfig, PlasticityConfig, DMNConfig
│       ├── topology_builders.py # Uniform, SmallWorld, Random, ScaleFree builders
│       ├── connectome_loader.py # Real connectome CSV loader + network builder
│       └── synapse_sign_predictor.py # Gene-expression-based synapse sign prediction
├── probes/                     # VoltageProbe, SpikeProbe, PopulationActivityProbe, etc.
├── analysis/
│   ├── topology.py             # TopologyAnalyzer (graph metrics via networkx)
│   ├── dynamics.py             # DynamicsAnalyzer (spectral, PCA, sync, info theory)
│   ├── behavior.py             # BehaviorDetector (locomotion, reversals, states)
│   └── visualization.py        # NetworkVisualizer (static matplotlib plots)
├── validation/
│   ├── single_neuron.py        # SingleNeuronValidator (vs published electrophysiology)
│   └── network_validation.py   # NetworkValidator (vs Randi atlas, Kato dynamics)
├── visualization/
│   ├── live_visualizer.py      # LiveVisualizer controller (threading, speed, frames)
│   ├── browser_backend.py      # Starlette + WebSocket server for browser rendering
│   ├── matplotlib_backend.py   # FuncAnimation local renderer
│   ├── fast_sim.py             # Vectorized NumPy simulation with STDP + habituation
│   ├── environment.py          # 2D worm body + arena + chemotaxis
│   ├── layouts.py              # grid, graph, anatomical position computation
│   └── static/index.html       # Browser Canvas UI (3-panel, dark theme, controls)
├── cli.py                      # Click CLI (sdmn info/simulate/run-example)
└── version.py
```

---

## Simulation Engines

There are two simulation engines. They can be switched at runtime via the browser UI dropdown.

### Object-Based Engine (`CElegansNetwork._step()`)

- **Speed:** ~300 steps/sec for 302 neurons (~0.01x realtime at dt=0.1ms)
- **Integration:** RK4 (4th-order Runge-Kutta) or Euler, configurable per neuron
- **Ion channels:** Full Ca/K/KCa dynamics with voltage-dependent gating
- **Synapses:** Dual-exponential kinetics, reversal potential-based current, delay buffer, release noise
- **Gap junctions:** Correct per-synapse conductance from parameters
- **Plasticity:** NONE -- no STDP or habituation in this engine
- **DMN:** Applied via `LiveVisualizer` (not in the engine itself)

### Vectorized Engine (`fast_sim._fast_step()`)

- **Speed:** ~25,000 steps/sec for 302 neurons (~2-3x realtime at dt=0.1ms)
- **Integration:** Euler only, single-compartment
- **Ion channels:** NONE -- only leak conductance (g_leak, E_leak, C_m)
- **Synapses:** Instantaneous sigmoid release, weight * sign (no kinetics, no reversal potential, no delay)
- **Gap junctions:** BUG -- always uses conductance=0.5 regardless of actual parameter
- **Plasticity:** STDP + habituation (reads from `PlasticityConfig`)
- **DMN:** Applied via `LiveVisualizer` before each batch

### Engine Comparison Matrix

| Capability | Object Engine | Fast Engine |
|-----------|---------------|-------------|
| Ca channel (depolarizing) | 8 params, gating ODE | Not implemented |
| K channel (repolarizing) | 8 params, gating ODE | Not implemented |
| K(Ca) channel (adaptation) | 3 params, Ca-dependent | Not implemented |
| Intracellular calcium | 4 params, influx/removal ODE | Not implemented |
| Leak conductance | Per-neuron configurable | Per-neuron from network |
| Membrane capacitance | Per-neuron configurable | Per-neuron from network |
| RK4 integration | Yes | No (Euler only) |
| Voltage noise | Configurable std | Not implemented |
| Current noise | Configurable std | Not implemented |
| Synaptic rise/decay kinetics | Dual-exponential | Instantaneous |
| Synaptic reversal potential | Conductance-based I = g*(V-E) | Signed current approximation |
| Synaptic delay | Configurable delay buffer | Not implemented |
| Release noise | Configurable std | Not implemented |
| Release threshold (V_thresh) | Per-synapse configurable | Hardcoded -40 mV |
| Release slope (k_release) | Per-synapse configurable | Hardcoded 5 mV |
| Gap junction conductance | Per-junction from params | BUG: always 0.5 nS |
| STDP learning | Not implemented | Hebbian + anti-Hebbian |
| Habituation (synaptic depression) | Not implemented | Depletion + recovery |
| Weight bounds | N/A | Configurable min/max |

---

## Property Inventory: What Lives Where

### Neuron Properties (on `GradedNeuronParameters` / `CElegansParameters`)

| Property | Default | Object Engine | Fast Engine | Notes |
|----------|---------|:---:|:---:|-------|
| `C_m` (membrane capacitance, pF) | 3.0 | Used | Used | |
| `g_leak` (leak conductance, nS) | 0.3 | Used | Used | |
| `E_leak` (leak reversal, mV) | -65.0 | Used | Used | Also sets initial V |
| `g_Ca` (Ca conductance, nS) | 0.8 | Used | **Ignored** | |
| `E_Ca` (Ca reversal, mV) | 50.0 | Used | **Ignored** | |
| `V_half_Ca`, `k_Ca` (activation curve) | -20, 5 | Used | **Ignored** | |
| `tau_Ca_min/max` (time constants) | 0.5, 5.0 | Used | **Ignored** | |
| `Ca_power` (activation exponent) | 2 | Used | **Ignored** | |
| `g_K` (K conductance, nS) | 1.5 | Used | **Ignored** | |
| `E_K` (K reversal, mV) | -80.0 | Used | **Ignored** | |
| `V_half_K`, `k_K` | -25, 10 | Used | **Ignored** | |
| `tau_K_min/max` | 1.0, 10.0 | Used | **Ignored** | |
| `K_power` | 4 | Used | **Ignored** | |
| `g_KCa` (Ca-dependent K, nS) | 0.5 | Used | **Ignored** | |
| `Ca_half` (half-activation [Ca], nM) | 100.0 | Used | **Ignored** | |
| `tau_KCa` (time constant, ms) | 50.0 | Used | **Ignored** | |
| `Ca_rest` (resting [Ca], nM) | 50.0 | Used | **Ignored** | |
| `tau_Ca_removal` (Ca clearance, ms) | 100.0 | Used | **Ignored** | |
| `f_Ca` (free Ca fraction) | 0.01 | Used | **Ignored** | |
| `cell_volume` (pL) | 1.0 | Used | **Ignored** | |
| `integration_method` ("RK4"/"Euler") | "RK4" | Used | **Ignored** (always Euler) |
| `V_min`, `V_max` (clipping) | -100, 100 | Used | **Ignored** (hardcoded -90, 20) |
| `voltage_noise_std` (mV/sqrt(ms)) | 0.0 | Used if >0 | **Ignored** | |
| `current_noise_std` (pA) | 0.0 | Used if >0 | **Ignored** | |
| `neuron_class` (sensory/inter/motor) | Per subclass | Used | Stored, not used in dynamics |

### Neuron Class Specializations

| Class | g_Ca | g_K | g_KCa | Behavior |
|-------|------|-----|-------|----------|
| Sensory | 1.2 (high) | 0.8 (low) | 0.6 | Highly excitable, fast response |
| Interneuron | 0.8 (balanced) | 1.5 (balanced) | 0.5 | Integrative |
| Motor | 0.7 (moderate) | 2.0 (high) | 0.8 (high) | Dampened, precise graded output |

These specializations are **only expressed in the object engine**. In the fast engine, all neurons behave identically (same leak dynamics).

### Synapse Properties (on `GradedSynapseParameters`)

| Property | Default | Object Engine | Fast Engine | Notes |
|----------|---------|:---:|:---:|-------|
| `synapse_type` (exc/inh) | EXCITATORY | Used | Used (as +1/-1) | |
| `weight` (nS) | 1.0 | Used | Used (STDP modifies) | |
| `V_thresh` (release threshold, mV) | -40.0 | Used | **Hardcoded** -40 | |
| `k_release` (sigmoid slope, mV) | 5.0 | Used | **Hardcoded** 5 | |
| `tau_rise` (ms) | 1.0 | Used | **Ignored** (instantaneous) | |
| `tau_decay` (ms) | 5.0 | Used | **Ignored** (instantaneous) | |
| `E_syn` (reversal potential, mV) | 0.0 (exc), -75.0 (inh) | Used | **Ignored** | Fast uses signed current, not conductance-based |
| `delay` (ms) | 0.5 | Used (deque buffer) | **Ignored** | |
| `release_noise_std` | 0.0 | Used if >0 | **Ignored** | |

### Gap Junction Properties (on `GapJunctionParameters`)

| Property | Default | Object Engine | Fast Engine | Notes |
|----------|---------|:---:|:---:|-------|
| `conductance` (nS) | 0.5 | Used | **BUG**: reads `_conductance` attr (doesn't exist), falls back to 0.5 | |
| `conductance_min/max` | 0.1, 2.0 | Used for bounds checking | **Ignored** | |

### Plasticity Properties (on `PlasticityConfig`)

| Property | Default | Object Engine | Fast Engine | Notes |
|----------|---------|:---:|:---:|-------|
| `enable_stdp` | False | **Not used** | Used | |
| `stdp_learning_rate` | 0.0003 | N/A | Used | |
| `stdp_tau_trace` (ms) | 20.0 | N/A | Used | STDP eligibility window |
| `stdp_w_min` (nS) | 0.05 | N/A | Used | Weight floor |
| `stdp_w_max_factor` | 3.0 | N/A | Used | max = initial * factor |
| `stdp_update_interval` (steps) | 100 | N/A | Used | Perf: batch updates |
| `enable_habituation` | False | **Not used** | Used | |
| `habituation_depression_rate` | 0.002 | N/A | Used | Per active step |
| `habituation_recovery_rate` | 0.0005 | N/A | Used | Recovery toward 1.0 |
| `habituation_floor` | 0.1 | N/A | Used | Minimum efficacy |

### DMN Properties (on `DMNConfig`)

| Property | Default | Where Applied | Notes |
|----------|---------|:---:|-------|
| `enabled` | False | `LiveVisualizer` | Falls back to constructor `enable_dmn` arg |
| `mode` ("oscillator"/"recurrent") | "oscillator" | `LiveVisualizer` | Runtime switchable |
| `amplitude` (pA) | 35.0 | `LiveVisualizer` | Current injection strength |
| `frequency` (kHz) | 0.08 | `LiveVisualizer` | Oscillator mode only |
| `noise_std` | 0.4 | `LiveVisualizer` | |
| `target_class` | "interneuron" | `LiveVisualizer` | Which neurons receive DMN |
| `recurrent_boost` | 1.5 | `LiveVisualizer` | Recurrent mode: intra-DMN weight multiplier |
| `recurrent_tau_adapt` (ms) | 200.0 | `LiveVisualizer` | Recurrent mode: adaptation time constant |

---

## What Is NOT Implemented (Gap Analysis)

### Biophysical properties missing from BOTH engines

| Property | What it would do | Priority |
|----------|-----------------|----------|
| **Sensitivity / input gain** | Per-neuron input resistance scaling | Medium |
| **Saturation / inactivation** | Ion channel inactivation preventing unlimited depolarization | High (fast engine) |
| **Recovery time** | Post-activation refractory-like period | Medium (K(Ca) does this in object engine) |
| **Inhibitory rebound** | Post-inhibition excitation from de-inactivation | Low |
| **Synaptic facilitation** | Short bursts temporarily strengthen transmission | Medium |
| **Per-synapse plasticity rates** | Different synapses learn/habituate at different rates | Medium |
| **Neuromodulation** | Global modulatory signals (dopamine, serotonin) affecting weights/excitability | High |
| **Homeostatic plasticity** | Network-wide weight normalization preventing runaway | High |

### Properties on objects but ignored by fast engine

| Property | Fix needed |
|----------|-----------|
| All ion channel parameters (Ca, K, KCa) | Add vectorized gating arrays to fast_sim |
| Synaptic kinetics (tau_rise, tau_decay) | Add exponential filters to fast_sim |
| Synaptic delay | Add delay buffer arrays |
| Reversal potential-based current | Replace signed-current with conductance model |
| Per-synapse V_thresh, k_release | Read from synapse params instead of hardcoding |
| Gap junction conductance | Fix `_prepare_arrays` to read `gap.gap_params.conductance` |
| Voltage/current noise | Add noise arrays |
| RK4 integration | Optional RK4 path (performance cost) |

### Architectural debt

1. **Two engines don't share a common interface** -- switching engines at runtime changes which physics are simulated, not just performance
2. **Plasticity exists only in fast engine** -- object-based simulations can't learn
3. **DMN is a visualization concern, not a network property** -- `_apply_dmn_to_arrays` lives in `LiveVisualizer`, not in the simulation engine
4. **Neuron class differences are invisible in fast engine** -- sensory/inter/motor have identical dynamics

### Recommended unification path

Add vectorized ion channel arrays to `fast_sim.py`:
- `m_Ca`, `m_K`, `m_KCa`, `Ca_internal` arrays (shape: n_neurons)
- Per-neuron channel parameters extracted from `CElegansParameters`
- Vectorized gating ODEs (same math as `CElegansNeuron._update_gating_variables`, but on arrays)
- This gives both speed (~10-20k steps/sec) and biophysical fidelity
- Plasticity stays in the vectorized path
- Gap junction bug gets fixed in the same pass

---

## Test Inventory

| File | Tests | Scope |
|------|-------|-------|
| `tests/test_core.py` | ~40 | SimulationEngine, events, state, time |
| `tests/test_neurons.py` | ~20 | LIF, HH, parameters, factories |
| `tests/test_networks.py` | ~15 | NetworkBuilder, topologies, Network |
| `tests/test_probes.py` | ~20 | Probe types, data, managers |
| `tests/test_integration.py` | ~15 | End-to-end spiking simulations |
| `tests/test_celegans_phase3.py` | 16 | ScaleFreeBuilder, ConnectomeLoader, refactored builders |
| `tests/test_analysis.py` | 37 | TopologyAnalyzer, DynamicsAnalyzer, BehaviorDetector, all plots |
| `tests/test_live_visualizer.py` | 37 | Layouts, callbacks, speed control, environment, DMN, backends |
| `tests/test_visualization_integration.py` | 35 | Full pipeline: sim -> frames -> layouts -> plots -> protocol |
| `tests/test_plasticity.py` | 13 | STDP weight changes, habituation decay/recovery, stats |

```bash
pytest tests/test_celegans_phase3.py tests/test_analysis.py tests/test_live_visualizer.py tests/test_visualization_integration.py tests/test_plasticity.py -v
```

---

## Examples

| File | Description | Runtime |
|------|-------------|---------|
| `01_basic_neuron_demo.py` | Single LIF neuron | <1s |
| `02_network_topologies.py` | Spiking network topologies | ~5s |
| `03_probe_monitoring.py` | Probe system demo | ~5s |
| `04_default_mode_networks.py` | DMN-themed simulation | ~10s |
| `05_self_aware_network.py` | Self-monitoring demo | ~10s |
| `06_celegans_graded_neurons.py` | Graded neuron dynamics (4 figures) | ~5s |
| `07_network_manager_demo.py` | CElegansNetwork API | ~10s |
| `08_topology_comparison.py` | Uniform vs small-world vs random | ~30s |
| `09_full_connectome_simulation.py` | Full 302-neuron pipeline + analysis + plots | ~60-120s |
| `10_live_visualization.py` | Real-time 3-panel browser visualizer | Interactive |

---

## Key Interfaces for Development

### Adding a new neuron model

Subclass `GradedNeuron` and override `_compute_ionic_currents()` and `_update_gating_variables()`:

```python
from sdmn.neurons.graded import GradedNeuron, GradedNeuronParameters

class MyNeuron(GradedNeuron):
    def _compute_ionic_currents(self, V: float) -> float:
        return ...  # pA
    def _update_gating_variables(self, dt: float) -> None:
        ...
```

### Configuring plasticity and DMN on a network

```python
from sdmn.networks.celegans import (
    SimulationConfig, PlasticityConfig, DMNConfig, build_connectome_network,
)

config = SimulationConfig(
    dt=0.1,
    record_interval=5,
    plasticity=PlasticityConfig(
        enable_stdp=True,
        stdp_learning_rate=0.0003,
        enable_habituation=True,
        habituation_depression_rate=0.002,
    ),
    dmn=DMNConfig(
        enabled=True,
        mode="recurrent",      # or "oscillator"
        amplitude=35.0,
        target_class="interneuron",
    ),
)
network = build_connectome_network()
network.config = config
```

### Registering a simulation callback

```python
from sdmn.networks.celegans import CElegansNetwork, SimulationFrame

def my_callback(frame: SimulationFrame):
    print(f"t={frame.time_ms:.1f}ms")

network.register_callback(my_callback)
network.simulate(duration=100, progress=False)
network.unregister_callback(my_callback)
```

---

## Dependencies

### Core (always required)
- numpy, scipy, matplotlib, networkx, pyyaml, click, tqdm

### Web visualization (optional, `poetry install --with web`)
- starlette, uvicorn, websockets, wsproto

### Full dev environment
```bash
poetry install --with dev,test,web
```

---

## File Inventory (cumulative)

### Phase 3-4: Topology Builders + Analysis (April 10, 2026)
```
NEW  src/sdmn/networks/celegans/connectome_loader.py
NEW  src/sdmn/analysis/__init__.py, topology.py, dynamics.py, behavior.py, visualization.py
NEW  data/connectome/celegans_neurons.csv, celegans_connectome.csv
NEW  scripts/generate_connectome_data.py
NEW  examples/09_full_connectome_simulation.py
NEW  tests/test_celegans_phase3.py, test_analysis.py
NEW  docs/celegans/PHASE3_4_GUIDE.md
```

### Phase 7: Live Visualization (April 11, 2026)
```
NEW  src/sdmn/visualization/__init__.py, live_visualizer.py, browser_backend.py
NEW  src/sdmn/visualization/matplotlib_backend.py, layouts.py, environment.py, fast_sim.py
NEW  src/sdmn/visualization/static/index.html
NEW  examples/10_live_visualization.py
NEW  tests/test_live_visualizer.py, test_visualization_integration.py, test_plasticity.py
MOD  src/sdmn/networks/celegans/network_manager.py (SimulationFrame, callbacks, PlasticityConfig, DMNConfig)
MOD  pyproject.toml (websockets dep)
```

### Phase 8: Biological Plausibility & Validation (April 12, 2026)
```
NEW  src/sdmn/neurons/graded/multichannel_neuron.py (MultiChannelNeuron, AWC/RMD factories)
NEW  src/sdmn/neurons/graded/neuropeptide_layer.py (ExtrasynapticLayer, monoamine pathways)
NEW  src/sdmn/networks/celegans/synapse_sign_predictor.py (gene-expression-based sign prediction)
NEW  src/sdmn/validation/__init__.py, single_neuron.py, network_validation.py
NEW  data/references/REFERENCE_CATALOG.md (complete bibliography with 30+ papers)
NEW  data/validation/neurotransmitters/celegans_neurotransmitter_map.csv (~180 neurons)
NEW  data/validation/neurotransmitters/celegans_receptor_reversal_potentials.csv
NEW  data/validation/neurotransmitters/synapse_sign_rules.csv
NEW  data/validation/neuron_models/nicoletti_awc_parameters.csv (from Nicoletti 2019 ODE)
NEW  data/validation/neuron_models/nicoletti_rmd_parameters.csv (from Nicoletti 2019 ODE)
NEW  data/validation/electrophysiology/published_neuron_properties.csv
NEW  data/validation/functional_atlas/README.md (Randi atlas integration guide)
NEW  docs/BIOLOGICAL_PLAUSIBILITY_ANALYSIS.md (criticism analysis + validation strategy)
NEW  docs/IMPLEMENTATION_PLAN.md (7-phase improvement plan with task tracking)
MOD  src/sdmn/neurons/graded/__init__.py (export new models)
MOD  docs/DEVELOPMENT_STATUS.md (updated architecture, file inventory)
```
