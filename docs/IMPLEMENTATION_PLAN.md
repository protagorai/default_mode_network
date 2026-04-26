# Implementation Plan: Biological Plausibility Improvements

**Date:** April 12, 2026
**Status:** Active development plan
**Prerequisite:** `docs/BIOLOGICAL_PLAUSIBILITY_ANALYSIS.md`

---

## Phase 1: Foundation Corrections (Week 1–2)

### 1.1 Fix Synapse Sign Assignment [HIGH PRIORITY]

**Problem:** Current rule maps GABA→inhibitory, everything else→excitatory. This misassigns
glutamatergic synapses (many are inhibitory via GluCl channels) and ignores that ACh can be
inhibitory via ACC-family channels.

**Solution:** `SynapseSignPredictor` (NEW — `src/sdmn/networks/celegans/synapse_sign_predictor.py`)

**Tasks:**
- [x] Create `celegans_neurotransmitter_map.csv` with per-neuron NT identity from Loer & Rand / Wang et al. 2024
- [x] Create `synapse_sign_rules.csv` with NT+receptor → sign mapping
- [x] Create `celegans_receptor_reversal_potentials.csv` with biologically accurate E_rev values
- [x] Implement `SynapseSignPredictor` class
- [ ] Integrate `SynapseSignPredictor` into `ConnectomeLoader.build_network()`
- [ ] Fix GABA reversal potential: change from -75 mV to -35 mV
- [ ] Add GluCl-based inhibitory glutamate assignment
- [ ] Add confidence levels to each synapse sign prediction
- [ ] Write unit tests comparing predictions to Fenyves et al. (2020) published results
- [ ] Generate report: % of synapses with each sign, comparison to published 4:1 E:I ratio

**Validation criterion:** Predicted E:I ratio should be approximately 4:1 (Fenyves et al. 2020).

### 1.2 Fix GABA Reversal Potential [HIGH PRIORITY]

**Problem:** `GradedSynapseParameters.__post_init__` sets `E_syn = -75.0` for inhibitory
synapses. In C. elegans, GABA-A reversal is -30 to -40 mV (Bamber et al. 1999; Richmond & Jorgensen 1999).

**Tasks:**
- [ ] Change default inhibitory E_syn from -75.0 to -35.0 in `graded_synapse.py`
- [ ] Add per-receptor-type reversal potentials:
  - GABA (UNC-49): -35 mV
  - GluCl: -50 mV
  - ACh Cl- (ACC): -35 mV
  - Excitatory cation: -5 mV
- [ ] Update `ConnectomeLoader` to pass receptor-specific E_syn

### 1.3 Fix Gap Junction Conductance Bug [HIGH PRIORITY]

**Problem:** Fast engine always uses 0.5 nS regardless of actual parameters.

**Tasks:**
- [ ] Fix `fast_sim._prepare_arrays()` to read `gap.gap_params.conductance`
- [ ] Add per-junction conductance array to vectorized engine
- [ ] Write regression test

---

## Phase 2: Per-Neuron-Type Parameterization (Week 2–3)

### 2.1 Multi-Channel Neuron Model [HIGH PRIORITY]

**Problem:** All 302 neurons share 3 parameter sets (sensory/inter/motor) with a
generic Ca/K/K(Ca) triplet. Real neurons have 6–12 distinct ion channel types with
dramatically different dynamics.

**Solution:** `MultiChannelNeuron` (NEW — `src/sdmn/neurons/graded/multichannel_neuron.py`)

**Tasks:**
- [x] Implement `MultiChannelNeuron` with configurable ion channel complement
- [x] Implement `ChannelParams` dataclass for per-channel parameterization
- [x] Create `create_awc_neuron()` factory with Nicoletti et al. (2019) AWC parameters
- [x] Create `create_rmd_neuron()` factory with Nicoletti et al. (2019) RMD parameters
- [ ] Validate AWC model against published traces (regenerative response, bistability)
- [ ] Validate RMD model against published traces (oscillations, bistable regime)
- [ ] Create factory functions for additional well-studied neurons:
  - [ ] `create_awa_neuron()` — calcium action potentials (Liu et al. 2018)
  - [ ] `create_afd_neuron()` — thermosensory oscillations (Clark et al. 2006)
  - [ ] `create_ash_neuron()` — nociceptor plateau potentials (Mellem et al. 2002)
  - [ ] `create_ase_neuron()` — salt sensory, asymmetric L/R (Suzuki et al. 2008)
  - [ ] `create_ava_neuron()` — command interneuron for backward locomotion
  - [ ] `create_avb_neuron()` — command interneuron for forward locomotion
- [ ] For remaining ~100 neuron types: use class-average parameters but with
      neurotransmitter-appropriate channel complements

### 2.2 Neuron-Type Registry

**Tasks:**
- [ ] Create `NeuronTypeRegistry` class mapping neuron name → factory function
- [ ] Integrate registry into `ConnectomeLoader.build_network()` so each neuron
      gets type-specific parameters automatically
- [ ] Fallback to class-average parameters for unmapped neuron types

### 2.3 Extract and Catalog Published Parameters

**Tasks:**
- [x] Extract AWC parameters from Nicoletti et al. ODE files → `data/validation/neuron_models/`
- [x] Extract RMD parameters from Nicoletti et al. ODE files → `data/validation/neuron_models/`
- [x] Create `published_neuron_properties.csv` with literature values
- [ ] Extract ASH calcium dynamics from Zahratka et al. (2015)
- [ ] Compile AWA parameters from Liu et al. (2018) supplementary data
- [ ] Download Mendeley dataset tngf9w3pgd for AWA calcium traces

---

## Phase 3: Extrasynaptic Signaling (Week 3–4)

### 3.1 Neuropeptide/Monoamine Signaling Layer [HIGH PRIORITY]

**Problem:** Randi et al. (2023) showed that functional connectivity differs dramatically
from the anatomical connectome. Much of this is due to neuropeptide signaling not captured
in synaptic wiring.

**Solution:** `ExtrasynapticLayer` (NEW — `src/sdmn/neurons/graded/neuropeptide_layer.py`)

**Tasks:**
- [x] Implement `NeuropeptideParams`, `MonoamineParams` dataclasses
- [x] Implement `ExtrasynapticSignal` with slow release/clearance kinetics
- [x] Implement `ExtrasynapticLayer` managing all extrasynaptic signaling
- [x] Create `create_serotonin_pathway()`, `create_dopamine_pathway()`, `create_tyramine_pathway()`
- [ ] Integrate `ExtrasynapticLayer` into `CElegansNetwork._step()`
- [ ] Integrate into `fast_sim._fast_step()`
- [ ] Populate target neurons from receptor expression data (CeNGEN / wormneuroatlas)
- [ ] Add octopamine pathway
- [ ] Create initial neuropeptide pathways from Beets et al. (2023) top-20 validated couples
- [ ] Validate: compare WT vs unc-31 simulation to Randi et al. WT vs unc-31 atlas data

### 3.2 Install and Integrate wormneuroatlas

**Tasks:**
- [ ] Add `wormneuroatlas` to `pyproject.toml` optional dependencies
- [ ] Create `src/sdmn/networks/celegans/atlas_loader.py` that wraps wormneuroatlas:
  - Load functional connectivity matrix
  - Load monoamine connectome
  - Load neuropeptide connectome
  - Load CeNGEN gene expression data
  - Extract synapse sign predictions (Fenyves et al.)
- [ ] Create comparison script: atlas predictions vs SDMN predictions

---

## Phase 4: Validation Infrastructure (Week 4–5)

### 4.1 Single-Neuron Validation [HIGH PRIORITY]

**Solution:** `SingleNeuronValidator` (NEW — `src/sdmn/validation/single_neuron.py`)

**Tasks:**
- [x] Implement `validate_resting_potential()`
- [x] Implement `validate_step_response()` with RMSE and correlation
- [x] Implement `validate_passive_properties()` (R_input, tau_m, C_m)
- [x] Implement `generate_iv_curve()`
- [ ] Load published reference traces for AWCon, RMD, AWA, ASH
- [ ] Create automated validation test suite (pytest markers for slow tests)
- [ ] Generate comparison figures: simulated vs published traces

### 4.2 Network-Level Validation

**Solution:** `NetworkValidator` (NEW — `src/sdmn/validation/network_validation.py`)

**Tasks:**
- [x] Implement `validate_signal_propagation()` (stimulate one neuron, measure all)
- [x] Implement `compute_sign_accuracy()` (compare to Randi atlas)
- [x] Implement `compute_amplitude_correlation()` (Spearman rho)
- [x] Implement `analyze_spontaneous_dynamics()` (PCA, compare to Kato)
- [ ] Load Randi atlas data via wormneuroatlas
- [ ] Run validation for all head neurons with atlas coverage
- [ ] Generate accuracy report: sign accuracy, amplitude correlation per source neuron

### 4.3 Behavioral Validation Metrics (Long-term)

**Tasks:**
- [ ] Define behavioral metrics from Yemini et al. (2013): locomotion speed, reversal frequency
- [ ] Create `BehavioralValidator` class
- [ ] Requires body model integration (Phase 7)

---

## Phase 5: Engine Unification (Week 5–6)

### 5.1 Vectorize Ion Channels in Fast Engine [HIGH PRIORITY]

**Problem:** Fast engine uses leak-only dynamics. All ion channels, synaptic kinetics,
and neuron class differences are stripped away.

**Tasks:**
- [ ] Add per-neuron arrays to `fast_sim.py`:
  - `m_Ca[n]`, `m_K[n]`, `m_KCa[n]`, `Ca_internal[n]`
  - Per-neuron channel parameters extracted from `CElegansParameters`
- [ ] Implement vectorized gating ODEs (same math as `CElegansNeuron`, but on arrays)
- [ ] Replace signed-current synapse model with conductance-based:
  - `I_syn = g_syn * (V_post - E_syn)` instead of `weight * sign`
- [ ] Add synaptic delay buffer arrays
- [ ] Read per-synapse V_thresh and k_release from parameters
- [ ] Fix gap junction conductance bug
- [ ] Add per-neuron noise arrays
- [ ] Benchmark: target 10,000–20,000 steps/sec for 302 neurons

### 5.2 Shared Engine Interface

**Tasks:**
- [ ] Create `SimulationEngine` ABC with `step(dt)` and `get_state()` methods
- [ ] Object engine and fast engine both implement the interface
- [ ] Engine selection no longer changes which physics are simulated
- [ ] Move DMN injection from `LiveVisualizer` into the engine

---

## Phase 6: Neuromodulation System (Week 6–7)

### 6.1 Global Neuromodulatory State

**Tasks:**
- [ ] Implement `NeuromodulationState` class tracking global levels of:
  - Serotonin (5-HT): dwelling vs roaming state
  - Dopamine: food-present vs food-absent
  - Tyramine/Octopamine: escape/stress response
- [ ] Each modulatory state adjusts:
  - Synaptic weights (multiplicative modulation)
  - Neuron excitability (leak conductance shift)
  - Gap junction conductance
- [ ] Allow runtime switching between behavioral states

### 6.2 Homeostatic Plasticity

**Tasks:**
- [ ] Implement synaptic scaling to prevent runaway excitation
- [ ] Implement intrinsic plasticity (adjust neuron excitability based on activity)
- [ ] Add weight normalization (per-neuron total synaptic weight constraint)

---

## Phase 7: Body Model Integration (Week 8+, Long-term)

### 7.1 Simple Body Model

**Tasks:**
- [ ] Implement dorsal/ventral muscle activation model
- [ ] Connect motor neurons to muscle segments
- [ ] Implement proprioceptive feedback (stretch receptors)
- [ ] Generate body curvature from muscle activation
- [ ] Compare locomotion wavelength and frequency to biological data

### 7.2 Environment Coupling

**Tasks:**
- [ ] Chemical gradient environment for chemotaxis
- [ ] Connect gradient to sensory neurons (AWC, AWA, ASE)
- [ ] Close the sensory→brain→motor→body→environment→sensory loop
- [ ] Compare chemotaxis index to published values

---

## Data Files Created

| File | Contents | Status |
|------|----------|--------|
| `data/references/REFERENCE_CATALOG.md` | Complete bibliography with URLs and data access | Done |
| `data/validation/neurotransmitters/celegans_neurotransmitter_map.csv` | NT identity for ~180 neurons | Done |
| `data/validation/neurotransmitters/celegans_receptor_reversal_potentials.csv` | Receptor types with E_rev | Done |
| `data/validation/neurotransmitters/synapse_sign_rules.csv` | NT+receptor → sign rules | Done |
| `data/validation/neuron_models/nicoletti_awc_parameters.csv` | AWCon ion channel parameters | Done |
| `data/validation/neuron_models/nicoletti_rmd_parameters.csv` | RMD ion channel parameters | Done |
| `data/validation/electrophysiology/published_neuron_properties.csv` | Published V_rest, R_input, etc. | Done |
| `data/validation/functional_atlas/README.md` | Guide for Randi atlas integration | Done |

## Code Files Created

| File | Contents | Status |
|------|----------|--------|
| `src/sdmn/neurons/graded/multichannel_neuron.py` | Multi-channel neuron model + AWC/RMD factories | Done |
| `src/sdmn/neurons/graded/neuropeptide_layer.py` | Extrasynaptic signaling layer | Done |
| `src/sdmn/networks/celegans/synapse_sign_predictor.py` | Improved sign prediction | Done |
| `src/sdmn/validation/__init__.py` | Validation module | Done |
| `src/sdmn/validation/single_neuron.py` | Single-neuron validation tools | Done |
| `src/sdmn/validation/network_validation.py` | Network-level validation tools | Done |

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|---------------|
| Synapse sign accuracy vs Randi atlas | Unknown (no validation) | >60% | `NetworkValidator.compute_sign_accuracy()` |
| Response amplitude correlation | Unknown | Spearman ρ > 0.3 | `NetworkValidator.compute_amplitude_correlation()` |
| Single-neuron V_rest within published range | Not tested | >80% of tested neurons | `SingleNeuronValidator.validate_resting_potential()` |
| Spontaneous dynamics dimensionality | Unknown | 3–8 effective dims | `NetworkValidator.analyze_spontaneous_dynamics()` |
| Top-3 PCA variance | Unknown | 50–70% | Same as above |
| E:I ratio of predicted synapses | Binary (GABA-only) | ~4:1 | `SynapseSignPredictor.get_sign_statistics()` |
| Fast engine biophysical fidelity | Leak-only | Full ion channels | Engine comparison test |
