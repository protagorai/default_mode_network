# Randi et al. (2023) Functional Atlas Integration

## Overview

The Neural Signal Propagation Atlas (Randi et al., Nature 2023) provides experimentally
measured signal propagation for 23,433 neuron pairs in the C. elegans head, including
response sign, amplitude, temporal dynamics, and directionality.

## Data Access

### Python Package (Recommended)
```bash
pip install wormneuroatlas
```

```python
import wormneuroatlas as wa

# Load the atlas
atlas = wa.NeuroAtlas()

# Get functional connectivity matrix
fun_conn = atlas.get_signal_propagation_map()

# Get anatomical connectome for comparison
anat_conn = atlas.get_anatomical_connectome()

# Get neurotransmitter assignments
nt_map = atlas.get_neurotransmitter_ids()

# Get neuropeptide/GPCR data
peptide_data = atlas.get_peptidergic_connectome()
```

### Raw Data
- GitHub: https://github.com/francescorandi/wormneuroatlas
- Pump-probe data: https://github.com/leiferlab/pumpprobe

### Interactive Browsers
- FunCoNN: https://funconn.princeton.edu/
- Connectome Toolbox: https://openworm.org/ConnectomeToolbox/Randi_2023/

## What the Atlas Contains

| Dataset | Description | Format |
|---------|-------------|--------|
| Signal propagation (WT) | Response of each neuron to optogenetic activation of each other neuron | Matrix (sign, amplitude, latency) |
| Signal propagation (unc-31) | Same but in neuropeptide-release mutant | Matrix |
| Anatomical connectome | Synaptic + gap junction connectivity | Adjacency matrix |
| CeNGEN transcriptome | Single-cell gene expression per neuron type | Gene x Neuron matrix |
| Neuropeptide/GPCR couples | 461 validated peptide-receptor pairs with EC50 | Table |
| Monoamine connectome | Monoamine source-target predictions | Adjacency matrix |
| Chemical synapse signs | Predicted E/I based on gene expression | Per-synapse labels |

## Validation Protocol

### Step 1: Install wormneuroatlas
```bash
pip install wormneuroatlas
```

### Step 2: Extract functional connectivity matrix
```python
import wormneuroatlas as wa
import numpy as np

atlas = wa.NeuroAtlas()
sig_prop = atlas.get_signal_propagation_map()

# sig_prop is a neuron x neuron matrix
# Positive values = excitatory response
# Negative values = inhibitory response
# Near zero = no measurable response
```

### Step 3: Compare to SDMN simulation
```python
from sdmn.validation import NetworkValidator

validator = NetworkValidator()
# Convert atlas data to expected format and load
validator.load_functional_atlas(atlas_dict)

# For each source neuron in the atlas:
for source in atlas_neurons:
    sim_results = validator.validate_signal_propagation(network, source)
    accuracy = validator.compute_sign_accuracy(sim_results, source)
    correlation = validator.compute_amplitude_correlation(sim_results, source)
```

## Key Findings from the Atlas (for Model Calibration)

1. **Functional ≠ Anatomical**: Many strong functional connections have no
   anatomical synapse. These are mediated by neuropeptide signaling.

2. **unc-31 comparison**: The unc-31 mutant (deficient in neuropeptide release)
   shows dramatically reduced functional connectivity, confirming the role of
   extrasynaptic signaling.

3. **Sign prediction**: The anatomical connectome alone poorly predicts
   response signs. Gene-expression-based predictions (Fenyves et al. 2020)
   are better but still imperfect.

4. **Timescales**: Neuropeptide-mediated responses operate on 100ms-1s
   timescales, overlapping with but generally slower than synaptic transmission.
