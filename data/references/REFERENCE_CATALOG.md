# Research Reference Catalog

Curated bibliography and data source catalog for SDMN C. elegans validation.

---

## Category 1: Connectome Structure

### Varshney et al. (2011) — Structural Connectome
- **Title:** Structural properties of the Caenorhabditis elegans neuronal network
- **Journal:** PLOS Computational Biology, 7(2):e1001066
- **DOI:** 10.1371/journal.pcbi.1001066
- **URL:** https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001066
- **Data:** Adjacency matrices, neuron classifications
- **Status:** Bundled in `data/connectome/` — used by `ConnectomeLoader`

### Cook et al. (2019) — Whole-Animal Connectome
- **Title:** Whole-animal connectomes of both Caenorhabditis elegans sexes
- **Journal:** Nature, 571:63-71
- **DOI:** 10.1038/s41586-019-1352-7
- **URL:** https://www.nature.com/articles/s41586-019-1352-7
- **Data:** Updated synapse counts, new connections, male connectome
- **Status:** Partially integrated via connectome CSVs

### White et al. (1986) — Original Connectome
- **Title:** The structure of the nervous system of the nematode Caenorhabditis elegans
- **Journal:** Phil Trans R Soc Lond B, 314:1-340
- **DOI:** 10.1098/rstb.1986.0056
- **Status:** Historical reference; superseded by Varshney/Cook

---

## Category 2: Functional Connectome & Signal Propagation

### Randi et al. (2023) — Neural Signal Propagation Atlas *** CRITICAL ***
- **Title:** Neural signal propagation atlas of Caenorhabditis elegans
- **Journal:** Nature, 623:406-414
- **DOI:** 10.1038/s41586-023-06683-4
- **URL:** https://www.nature.com/articles/s41586-023-06683-4
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10632145/
- **Code/Data:**
  - Python package: `pip install wormneuroatlas` (francescorandi/wormneuroatlas)
  - GitHub: https://github.com/francescorandi/wormneuroatlas
  - Raw data: https://github.com/leiferlab/pumpprobe
  - FunCoNN browser: https://funconn.princeton.edu/
  - Connectome Toolbox: https://openworm.org/ConnectomeToolbox/Randi_2023/
- **Contents:**
  - Signal propagation measurements for 23,433 neuron pairs
  - Response sign (excitatory/inhibitory), amplitude, temporal dynamics
  - Wild-type and unc-31 (neuropeptide-deficient) strains
  - Functional vs anatomical connectivity comparison
- **Status:** To be integrated — see `data/validation/functional_atlas/`

### Fenyves et al. (2020) — Synaptic Sign Predictions
- **Title:** Synaptic polarity and sign-balance prediction using gene expression data
- **Journal:** PLOS Computational Biology, 16(12):e1007974
- **DOI:** 10.1371/journal.pcbi.1007974
- **URL:** https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007974
- **Data:** Gene-expression-based sign predictions for ~2/3 of chemical synapses
- **Key finding:** Excitatory:inhibitory ratio ≈ 4:1
- **Status:** To be integrated into synapse sign assignment

---

## Category 3: Extrasynaptic Signaling

### Bentley et al. (2016) — Multilayer Connectome
- **Title:** The multilayer connectome of Caenorhabditis elegans
- **Journal:** PLOS Computational Biology, 12(12):e1005283
- **DOI:** 10.1371/journal.pcbi.1005283
- **URL:** https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005283
- **Data:** Monoamine and neuropeptide connectomes overlaid on synaptic connectome
- **Key findings:** Monoamine network has star-like broadcasting hubs; neuropeptide network is recurrent/clustered

### Beets et al. (2023) — Neuropeptide-GPCR Deorphanization
- **Title:** System-wide mapping of peptide-GPCR interactions in C. elegans
- **Journal:** Cell Reports, 42(9):113058
- **DOI:** 10.1016/j.celrep.2023.113058
- **Data:** 461 validated peptide-GPCR couples with EC50 values (from 55,384 tested)
- **Status:** Available via wormneuroatlas package

### Ripoll-Sánchez et al. (2023) — Neuropeptidergic Connectome
- **Title:** The neuropeptidergic connectome of C. elegans
- **Journal:** Neuron, 111(22):3570-3589
- **DOI:** 10.1016/j.neuron.2023.09.043
- **URL:** https://pubmed.ncbi.nlm.nih.gov/37935195/
- **Data:** Draft neuropeptidergic connectome integrating single-cell expression with receptor-ligand interactions

---

## Category 4: Ion Channel & Neuron Electrophysiology

### Goodman et al. (1998) — Active Currents in C. elegans Neurons
- **Title:** Active currents regulate sensitivity and dynamic range in C. elegans neurons
- **Journal:** Neuron, 20(4):763-772
- **DOI:** 10.1016/S0896-6273(00)81014-4
- **Data:** First patch-clamp recordings from identified C. elegans neurons; I-V curves

### Liu et al. (2018) — AWA Calcium Action Potentials
- **Title:** C. elegans AWA olfactory neurons fire calcium-mediated all-or-none action potentials
- **Journal:** Cell, 175(1):57-70
- **DOI:** 10.1016/j.cell.2018.08.018
- **Data:** Mendeley dataset tngf9w3pgd — calcium imaging movies and analyzed data
- **URL:** https://data.mendeley.com/datasets/tngf9w3pgd/1

### Nicoletti et al. (2019) — AWCon and RMD Biophysical Models *** CRITICAL ***
- **Title:** Biophysical modeling of C. elegans neurons: Single ion currents and whole-cell dynamics of AWCon and RMD
- **Journal:** PLOS ONE, 14(7):e0218738
- **DOI:** 10.1371/journal.pone.0218738
- **Code/Data:**
  - GitHub: https://github.com/openworm/NicolettiEtAl2019_NeuronModels
  - ModelDB: https://modeldb.science/showmodel?model=267187
  - ODE files: AWC.ode, RMD.ode (XPPAUT format)
  - NeuroML2: AWCon.cell.nml, RMD.cell.nml
- **Ion channels modeled:**
  - AWCon: SHL-1, KVS-1, SHK-1, KQT-3, EGL-2, IRK, CCA-1, UNC-2, EGL-19, SLO-1, SLO-2, KCNL
  - RMD: SHL-1, SHK-1, EGL-36, IRK, CCA-1, UNC-2, EGL-19, SLO-1, SLO-2, KCNL
- **Status:** Parameters extracted to `data/validation/neuron_models/`

### Mellem et al. (2002) — Graded Transmission
- **Title:** Decoding of polymodal sensory stimuli by postsynaptic glutamate receptors in C. elegans
- **Journal:** Neuron, 36(5):933-944
- **DOI:** 10.1016/S0896-6273(02)01063-3
- **Data:** Electrophysiology of graded synaptic transmission between identified neuron pairs

### Lockery & Goodman (2009) — Neuroscience in C. elegans
- **Title:** The quest for action potentials in C. elegans neurons hits a plateau
- **Journal:** Nature Neuroscience, 12(4):377-378
- **DOI:** 10.1038/nn0409-377

---

## Category 5: Whole-Brain Dynamics & Calcium Imaging

### Kato et al. (2015) — Global Brain Dynamics
- **Title:** Global brain dynamics embed the motor command sequence of Caenorhabditis elegans
- **Journal:** Cell, 163(3):656-669
- **DOI:** 10.1016/j.cell.2015.09.034
- **URL:** https://www.cell.com/cell/fulltext/S0092-8674(15)01196-4
- **Key findings:**
  - Brain dynamics are low-dimensional (~3 PCs capture motor command sequence)
  - Cyclic trajectories in neural state space map to forward/reverse/turn behaviors
  - Provides quantitative targets for spontaneous dynamics validation

### Nguyen et al. (2016) — Freely-Behaving Whole-Brain Imaging
- **Title:** Whole-brain calcium imaging with cellular resolution in freely behaving C. elegans
- **Journal:** PNAS, 113(8):E1074-E1081
- **DOI:** 10.1073/pnas.1507110112
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4776509/

### WormID-Bench (2025) — Multi-Lab Benchmark
- **Title:** WormID-Bench: A benchmark for whole-brain activity extraction in C. elegans
- **Journal:** bioRxiv (preprint)
- **DOI:** 10.1101/2025.01.06.631621
- **URL:** https://www.biorxiv.org/content/10.1101/2025.01.06.631621
- **Data:** 118 worms from 5 labs, standardized evaluation metrics, publicly available

---

## Category 6: Neurotransmitter Identity

### Loer & Rand (2022/2025) — Neurotransmitter Table *** CRITICAL ***
- **Title:** The evidence for classical neurotransmitters in C. elegans neurons
- **WormAtlas:** https://www.wormatlas.org/neurotransmitterstable.htm
- **Updated version:** https://home.sandiego.edu/~cloer/nt_table_update/ce_neurotransmitters_2025.htm
- **Excel download:** https://home.sandiego.edu/~cloer/nt_table_update/Ce_NTtables_Loer&Rand2022.xlsx
- **Coverage:** Neurotransmitter identity for >90% of C. elegans neurons
- **Status:** Extracted to `data/validation/neurotransmitters/`

### Pereira et al. (2015) — Cholinergic Nervous System Map
- **Title:** A cellular and regulatory map of the cholinergic nervous system of C. elegans
- **Journal:** eLife, 4:e12432
- **DOI:** 10.7554/eLife.12432
- **Data:** 52 of 118 neuron types are cholinergic; complete ACh map

### Wang et al. (2024) — Definitive Neurotransmitter Atlas
- **Title:** A neurotransmitter atlas of the nervous system of C. elegans males and hermaphrodites
- **Journal:** eLife, 13:e95402
- **DOI:** 10.7554/eLife.95402
- **Data:** Incorporates CeNGEN scRNA-seq data; near-complete assignments

### Taylor et al. (2021) — CeNGEN Single-Cell Transcriptome
- **Title:** Molecular topography of an entire nervous system
- **Journal:** Cell, 184(16):4329-4347
- **DOI:** 10.1016/j.cell.2021.06.023
- **URL:** https://www.cengen.org/
- **Data:** Single-cell RNA-seq for all 302 neurons; gene expression per neuron type

---

## Category 7: Receptor Biology

### Richmond & Bhatt (1999) — Neuromuscular Junction Receptors
- **Title:** One GABA and two acetylcholine receptors function at the C. elegans neuromuscular junction
- **Journal:** Nature Neuroscience, 2:791-797
- **DOI:** 10.1038/12160
- **Key data:** UNC-49 (GABA-A), levamisole-sensitive nAChR, nicotine-sensitive nAChR

### Cully et al. (1994) / Dent et al. (2006) — Glutamate-Gated Chloride Channels
- GluCl channels: avr-14, avr-15, glc-1, glc-2, glc-3, glc-4
- These make glutamate **inhibitory** in many C. elegans synapses
- **Review:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3504739/

### Ringstad et al. (2009) — Chloride Transporters and GABA Reversal
- **Title:** Two types of chloride transporters are required for GABA-A receptor-mediated inhibition in C. elegans
- **Journal:** EMBO J, 30:1852-1863
- **Key finding:** GABA reversal potential depends on KCC-2 and ABTS-1 chloride transporters
- **Data:** GABA reversal potential ≈ -30 to -40 mV (not -75 mV as in current SDMN model)

---

## Category 8: Behavioral Phenotyping

### Yemini et al. (2013) — Behavioral Database
- **Title:** A database of C. elegans behavioral phenotypes
- **Journal:** Nature Methods, 10:877-879
- **DOI:** 10.1038/nmeth.2560
- **Data:**
  - 305 strains, ~10,000 worms, 702 phenotypic features
  - Database: http://movement.openworm.org/
  - Code: https://github.com/Yemini-Lab/WT2
  - Dropbox archive: https://www.dropbox.com/sh/2mw4m4pfd1mkmm7

---

## Category 9: Competing/Related Simulations

### Jiang et al. (2024) — BAAIWorm
- **Title:** An integrative data-driven model simulating C. elegans brain, body and environment interactions
- **Journal:** Nature Computational Science, 4:967-981
- **DOI:** 10.1038/s43588-024-00738-w
- **Code:** https://github.com/Jessie940611/BAAIWorm (Apache-2.0)
- **Key innovation:** Closed-loop brain-body-environment with multicompartment neurons

### Sarma et al. (2018) — OpenWorm Overview
- **Title:** OpenWorm: overview and recent advances in integrative biological simulation of C. elegans
- **Journal:** Phil Trans R Soc B, 373:20170382
- **DOI:** 10.1098/rstb.2017.0382
- **PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6158220/

### OpenWorm Tools & Data
- **c302 (neural model):** https://github.com/openworm/c302
- **Sibernetic (body physics):** https://github.com/openworm/sibernetic
- **Connectome Toolbox:** https://github.com/openworm/ConnectomeToolbox
- **ChannelWorm (archived):** https://github.com/openworm/channelworm

---

## Software Tools

| Tool | Install | Purpose |
|------|---------|---------|
| wormneuroatlas | `pip install wormneuroatlas` | Aggregated atlas data (Randi + transcriptome + connectome) |
| pyNeuroML | `pip install pyNeuroML` | NeuroML model simulation |
| XPPAUT | Platform-specific | ODE model simulation (Nicoletti models) |
| NemaNode | https://nemanode.org/ | Interactive connectome browser |
| FunCoNN | https://funconn.princeton.edu/ | Functional connectivity browser |
