# Idea Graft — Cross-Instance Neural Knowledge Transfer

> **Status (candidate, Jun 2026):** A principle proposed for evaluation within the SDMN
> framework. It enters as a candidate idea (see the pointer in
> [docs/plan/IDEAS.md](IDEAS.md)) and is written up here in its own document because it
> spans measurement, simulation, and intervention rather than fitting in the scratchpad.
> It leans on the substrate/inner-simulation split of [docs/TWO_SIMULATIONS.md](../TWO_SIMULATIONS.md)
> and on the precision/priors machinery of [docs/plan/world_simulation.md](world_simulation.md)
> and [docs/plan/inner_simulation_architecture.md](inner_simulation_architecture.md).

## The principle

**Neural knowledge is encoded in a substrate-independent representation, so it can be
transcribed between two distinct network instances.** Given two synthetic organisms
running on the same substrate engine, we fit a mapping between their neuronal dynamics,
define a representation that lives *outside either instance*, and use that shared
representation not merely to *compare* the two but to **graft** — to induce learning in,
or evoke latent knowledge from, the recipient by carrying activity, plasticity, and
dynamical state across the map.

It is one claim with three separable layers. Naming them separately keeps the
conversation honest, the same way the project separates the substrate engine from the
inner simulation.

| Layer | What it is | Name |
|---|---|---|
| The **claim** | "Neural knowledge lives in an instance-independent representation, so it can be transcribed between brains." | the **Idea Graft** principle |
| The **artifact** | The shared space both organisms map into and out of | the **idea** (an *eidos* / interlingua) |
| The **procedure** | Fit the map → encode donor activity + plasticity + state → induce it in the recipient | the **graft** (the algorithm) |

## Terminology

Grafting is borrowed from horticulture deliberately, because the metaphor is exact:

- **Scion** — the donor fragment that carries the knowledge (a trained instance, or a
  slice of its dynamics).
- **Rootstock** — the recipient instance that receives and must integrate the scion.
- **The idea** — the instance-independent representation onto which the scion is
  encoded; the "graftable" form of the knowledge. (Greek *eidos* = idea/form; the
  representation that exists apart from any particular instance.)
- **Take** — borrowing the horticultural term for a successful graft: the recipient not
  only accepts the induced state but *retains and uses* it. "Take" is the thing we
  ultimately measure.

A graft can be **static** (transfer a learned configuration once) or **continuous**
(hold a live channel open so the two instances co-evolve through the idea).

## Why rate, sequence, and correlation are necessary but not sufficient

Rate, sequence, and correlation describe *what fired, in what order, and what co-fired*.
They are the right starting point, but they under-specify the thing we are trying to move:

- They are **point/pairwise** descriptions. The knowledge we want to graft lives in the
  **collective geometry** of the ensemble and in **how that geometry moves** — neither of
  which is recoverable from marginal rates and pairwise correlations alone.
- They are **state descriptions**, not **mechanism descriptions**. Transferring activity
  reproduces a snapshot; transferring *knowledge* requires also moving the **plasticity**
  and **modulatory state** that would regenerate that activity under new inputs.
- In genuinely **cyclic** C. elegans-style circuits, "activity" is a **trajectory through
  a dynamical system**, not a feedforward code. Two instances can match on rate, sequence,
  and correlation yet sit in differently-shaped flow fields, so a naive transfer fails to
  "take."

So the measurement program below adds the families needed to characterize the ensemble's
geometry, its dynamics, its information content, and its learning/modulatory state.

## The measurement program — characteristics of activity to investigate

Each family lists *what it captures*, *why it matters for grafting*, and *how to measure
it* in this framework's probe vocabulary. Families A–C extend the three measures already
named; D–J are the additions the principle requires.

### A. Rate & firing statistics (extends "rate")
- **Mean & instantaneous firing rate** — the baseline. Time-varying rate (PSTH /
  rate-coded trace) is the minimal donor signal.
- **Fano factor** (spike-count variance / mean) — trial-to-trial reliability; tells you
  whether the code is rate-precise or noisy.
- **ISI distribution, CV and CV2** — regularity vs. irregularity of spiking; distinguishes
  clock-like from Poisson-like units.
- **Burstiness / burst fraction** — bursts carry different downstream meaning than tonic
  firing and must be preserved or the graft mistranslates emphasis.
- **Population & lifetime sparseness** — how distributed the code is; sets how many units
  must participate in a faithful graft.

### B. Temporal structure & sequence (extends "sequence")
- **Spike-timing precision / jitter** — the temporal tolerance of the code; defines how
  tightly the graft must reproduce timing.
- **Response latency** — stimulus-to-spike delay; an alignment landmark across instances.
- **Sequence templates / spatiotemporal motifs** — ordered activations (replay-like
  sequences). Extract with seqNMF or tensor component analysis (TCA).
- **Conduction-delay / lag structure** — lead–lag relationships from cross-correlogram
  peaks; the temporal skeleton onto which sequences hang.
- **Oscillation-phase of firing (phase coding)** — *when* in a cycle a unit fires; see
  family F.

### C. Dependency & effective connectivity (extends "correlation")
- **Signal vs. noise correlations** — shared tuning vs. shared variability; the graft must
  respect both.
- **Mutual information** — nonlinear dependency that Pearson correlation misses.
- **Transfer entropy / Granger causality** — *directed* information flow; correlation is
  symmetric, but knowledge has direction, and the map must preserve who drives whom.
- **Partial correlation / precision matrix** — conditional (direct) dependencies, removing
  common-input confounds.
- **Higher-order (triplet+) correlations** — interactions invisible to pairwise measures.

### D. Population geometry & representational structure (the core of "the idea")
- **Dimensionality** — participation ratio / power-law spectrum of activity covariance.
  The graft happens in a low-dimensional latent space, not in raw neuron space.
- **Neural manifold / latent trajectory** — the low-D surface the population lives on; the
  natural coordinate system of the idea.
- **Representational geometry (RDM / RSA)** — the matrix of dissimilarities between states.
  This is **instance-free by construction** (it describes relations, not neurons), which is
  exactly why it is the prime candidate for the shared representation.
- **Tuning curves / selectivity & mixed selectivity** — what each axis of the manifold
  encodes; the semantic labels of the idea.
- **Decodability** — linear-readout accuracy for task variables; the operational definition
  of "how much knowledge is present" before and after a graft.
- **Manifold topology** — Betti numbers / persistent homology (e.g. detecting a ring
  attractor's loop). Topology is a coarse invariant that must match for a graft to take.

### E. Dynamical-systems structure (because cyclic circuits compute with trajectories)
- **Fixed points & attractors** — locations and stability of the flow's resting states
  (found via fixed-point optimization). Memories/decisions are attractors; grafting a
  memory means grafting an attractor.
- **Line/ring/plane attractors** — continuous attractors implementing integration
  (heading, evidence). Their shape is the knowledge.
- **Intrinsic timescales** — autocorrelation-decay time constants per unit/region; a
  temporal fingerprint that must be matched or the recipient runs the idea too fast/slow.
- **Lyapunov spectrum / chaos vs. stability** — sensitivity to perturbation; sets how
  robust a graft will be and whether continuous grafting stays bounded.
- **Recurrence quantification (RQA)** — determinism, laminarity, recurrence rate of the
  trajectory; model-free dynamical fingerprints.
- **Metastability / state transitions** — discrete brain-state sequences (HMM states) and
  dwell times; the macro-grammar above the moment-to-moment trajectory.

### F. Oscillatory & spectral structure (the project's "brain waves")
- **Band power** (delta–gamma analogs) and the **spectral fingerprint** of each region.
- **Coherence & phase-locking** between regions — oscillatory binding; a transfer channel
  in its own right.
- **Cross-frequency (phase–amplitude) coupling** — how slow rhythms gate fast ones; a
  carrier for hierarchical control during a graft.
- **Traveling waves / phase gradients** — spatial propagation; ties to the online-learning
  wave idea in [IDEAS.md](IDEAS.md).

### G. Information-theoretic structure
- **Entropy & information capacity** — how much the activity *could* carry; an upper bound
  on graftable content.
- **Complexity** (Lempel–Ziv; perturbational-complexity-index analog) — richness vs.
  triviality of dynamics; distinguishes a knowledge-bearing state from an attractor
  collapse.
- **Redundancy / synergy (partial information decomposition)** — whether information is
  duplicated across units (graft is robust to unit dropout) or synergistic (graft must move
  whole coalitions intact).

### H. Plasticity & learning state (so we transfer *knowledge*, not just a snapshot)
- **Synaptic weight distribution & structure** — the stored configuration; the static part
  of the scion.
- **Short-term plasticity state** — facilitation/depression variables (Tsodyks–Markram);
  the immediate dynamical context.
- **Eligibility traces & learning-rate / meta-plasticity state** — *how ready* a synapse is
  to change; grafting this transfers the capacity to keep learning, not just the result.
- **STDP-window parameters** — the local learning rule's shape; must match for transferred
  dynamics to be reinforced rather than erased in the rootstock.
- **Structural plasticity** — synaptogenesis/pruning state; the slow scaffold.

### I. Neuromodulatory & global state (the precision/priors the inner simulation runs on)
- **Neuromodulatory tone** (dopamine/serotonin analogs) — gain and learning-rate control;
  the "settings" under which the idea was formed.
- **Excitation/inhibition (E/I) balance** — operating regime; a mismatched E/I makes a
  faithful graft behave wrongly.
- **Precision / gain field** — the predictive-coding precision weights
  ([world_simulation.md](world_simulation.md)); these decide *which* errors the recipient
  will treat the grafted content as.
- **Global state / drive** — roaming vs. dwelling, arousal, default-mode rest; context that
  conditions whether a graft is even expressible.

### J. Structural substrate (the connectome that hosts all of the above)
- **Graph metrics** — degree, modularity, small-worldness, rich-club; constrain which maps
  between two connectomes are even feasible.
- **Structure–function coupling** — how much functional activity is predicted by wiring;
  sets whether you graft onto activity, onto weights, or onto wiring.
- **Motif statistics** — recurring wiring micro-patterns; the reusable vocabulary shared
  across instances.

## Alignment & transfer operators (how the idea is constructed and applied)

The shared representation is *built* by an alignment operator and *applied* by an induction
step. Candidate operators, roughly from "assume known neuron correspondence" to "assume
none":

- **Procrustes alignment** — orthogonal map between two latent spaces; cheap, assumes
  matched dimensionality.
- **Canonical Correlation Analysis (CCA / deep CCA)** — finds maximally-correlated shared
  axes between two instances' activity; a direct interlingua constructor.
- **Centered Kernel Alignment (CKA)** — similarity score robust to rotation/scaling; good as
  an *evaluation* metric for how alignable two instances are.
- **Representational Similarity Analysis (RSA / RDM correlation)** — compares the
  *relational geometry* (family D); instance-free, the most natural definition of "the idea."
- **Gromov–Wasserstein optimal transport** — matches two representational spaces using only
  their *internal* distance structures, i.e. **without any assumed neuron-to-neuron
  correspondence**. This is the key operator when the two organisms have different cell
  counts/identities.
- **Dynamical Similarity Analysis (DSA)** — compares the *temporal structure of computation*
  (the flow), not just static geometry; the right tool given family E.
- **Latent dynamics alignment** — align the recurrent dynamics (e.g. stable latent
  manifolds across instances/days), then map control inputs through the alignment.

Sketch of the graft, in the framework's terms:

```text
1. Probe donor (scion) and recipient (rootstock) on shared, instance-free landmarks.
2. Fit alignment A: donor-latent  <->  idea  <->  recipient-latent
     using {RSA geometry, CCA/GW-OT correspondence, DSA dynamics}.
3. Encode the scion into the idea:  enc = A_donor(activity, weights, STP, traces, state).
4. Induce in the rootstock:  clamp / nudge recipient latents toward A_recipient^{-1}(enc),
     and set its plasticity + precision state to match.
5. Release and measure "take": does the recipient retain and use the knowledge under
     novel inputs, with no further clamping?
```

## The four jobs — what each family is *for*

The framework asks four things of any candidate: can we **investigate** it, **evaluate**
it, **simulate** it, and **prescribe** (intervene with) it? Each measure family plays a
role in each job.

| Activity family | Investigate (is a shared structure there?) | Evaluate (how good is the graft?) | Simulate (what must be reproduced?) | Prescribe (what do we set/clamp?) |
|---|---|---|---|---|
| A. Rate & firing stats | Compare rate codes across instances | Rate-trace error donor→recipient | Generate matched rate statistics | Drive target rates |
| B. Temporal & sequence | Detect shared motifs/sequences | Sequence-match score, timing jitter | Reproduce sequences & latencies | Inject ordered activations |
| C. Dependency & effective conn. | Shared directed-info graph? | Preservation of TE/MI structure | Reproduce correlation+causal graph | Bias coupling / co-activation |
| D. Population geometry | Aligned manifolds / RDMs? | RSA/CKA/decodability transfer | Reproduce manifold & tuning | Clamp recipient latent state |
| E. Dynamical structure | Matching attractors/timescales? | DSA distance, attractor overlap | Reproduce flow field & attractors | Steer toward target attractor |
| F. Oscillatory/spectral | Shared rhythms & coupling? | Coherence/PAC preservation | Reproduce spectral fingerprint | Entrain via phase/coherence |
| G. Information-theoretic | Capacity & complexity match? | Info retained after graft | Match entropy/complexity budget | Set precision to gate content |
| H. Plasticity & learning state | Comparable plasticity regimes? | Does the graft keep learning ("take")? | Reproduce STP/traces/STDP | Set weights, traces, learn-rate |
| I. Neuromodulatory & state | Same operating regime? | Behavioral expression of graft | Reproduce E/I, gain, drive | Set neuromodulatory/precision state |
| J. Structural substrate | Feasible map between connectomes? | Wiring-respecting map quality | Reproduce graph & motifs | Choose/edit donor↔recipient mapping |

## Minimal evaluation protocol

1. **Train a scion.** Run instance A to competence on a task the framework already supports
   (e.g. chemotaxis), recording families A–I on a set of shared, instance-free landmark
   probes.
2. **Fit the idea.** Construct the alignment with RSA + (CCA or Gromov–Wasserstein OT) +
   DSA; report CKA/CCA strength as the "alignability" score.
3. **Graft into a naive rootstock.** Induce the encoded state + plasticity + precision in a
   fresh instance B; then **release** all clamps.
4. **Measure take.** Few-shot learning speedup and decodability in B vs. controls.
5. **Controls & ablations:**
   - *Shuffle control* — graft a permuted idea (should not help).
   - *Snapshot vs. full* — transfer activity-only vs. +plasticity vs. +precision/state, to
     isolate what actually carries knowledge.
   - *Rate/sequence/correlation-only vs. +geometry/dynamics*, to show the added families are
     load-bearing (this is the experiment that justifies the whole measurement program).

## What would falsify the principle

- Best-achievable alignment (max CKA/CCA, min DSA) still yields **no transfer above the
  shuffle control** → knowledge is not in an instance-independent representation at this
  scale.
- Transfer succeeds **only** when neuron-to-neuron correspondence is hand-supplied (no
  Gromov–Wasserstein-style map works) → there is no genuine *shared* idea, only
  instance-specific recoding.
- Activity-only grafts take just as well as +plasticity/+state grafts → the "carry
  plasticity and state" claim is unnecessary and the idea reduces to ordinary stimulation.

## Open questions

- Is the right shared representation **geometric** (RDM/manifold) or **dynamical** (flow/DSA)
  — or must the idea encode both, and how are they reconciled?
- What is the **minimum landmark set** (shared probes) needed to fit a usable idea between
  two instances?
- Does a graft need **continuous** correction to stay "taken," or can a single induction
  leave a self-sustaining attractor?
- How does graft fidelity scale with **instance dissimilarity** (different connectomes,
  sizes, neuron models) — where does the shared idea break down?
- Can the recipient **reject** a graft (immune-like response), and is rejection a feature
  (stability) rather than a bug?
- Does grafting plasticity state transfer the capacity to **generalize**, or only to
  reproduce the donor's specific solution?

## References (starting bibliography)

Shared representations & alignment
1. Haxby, J. V. et al. (2011). "A common, high-dimensional model of the representational space in human ventral temporal cortex." *Neuron*, 72(2), 404–416. DOI: [10.1016/j.neuron.2011.08.026](https://doi.org/10.1016/j.neuron.2011.08.026)
2. Kriegeskorte, N., Mur, M. & Bandettini, P. (2008). "Representational similarity analysis – connecting the branches of systems neuroscience." *Front. Syst. Neurosci.*, 2, 4. DOI: [10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008)
3. Kornblith, S., Norouzi, M., Lee, H. & Hinton, G. (2019). "Similarity of Neural Network Representations Revisited" (CKA). *ICML*. arXiv: [1905.00414](https://arxiv.org/abs/1905.00414)
4. Lenc, K. & Vedaldi, A. (2015). "Understanding image representations by measuring their equivariance and equivalence" (model stitching). *CVPR*. arXiv: [1411.5908](https://arxiv.org/abs/1411.5908)
5. Bansal, Y., Nakkiran, P. & Barak, B. (2021). "Revisiting Model Stitching to Compare Neural Representations." *NeurIPS*. arXiv: [2106.07682](https://arxiv.org/abs/2106.07682)
6. Huh, M., Cheung, B., Wang, T. & Isola, P. (2024). "The Platonic Representation Hypothesis." *ICML*. arXiv: [2405.07987](https://arxiv.org/abs/2405.07987)

Transfer precedents
7. Pais-Vieira, M. et al. (2013). "A Brain-to-Brain Interface for Real-Time Sharing of Sensorimotor Information." *Scientific Reports*, 3, 1319. DOI: [10.1038/srep01319](https://doi.org/10.1038/srep01319)
8. Bédécarrats, A. et al. (2018). "RNA from Trained Aplysia Can Induce an Epigenetic Engram for Long-Term Sensitization in Untrained Aplysia." *eNeuro*, 5(3). DOI: [10.1523/ENEURO.0038-18.2018](https://doi.org/10.1523/ENEURO.0038-18.2018)
9. Hinton, G., Vinyals, O. & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." arXiv: [1503.02531](https://arxiv.org/abs/1503.02531)

Population geometry & dynamics
10. Gallego, J. A., Perich, M. G., Miller, L. E. & Solla, S. A. (2017). "Neural Manifolds for the Control of Movement." *Neuron*, 94(5), 978–984. DOI: [10.1016/j.neuron.2017.05.025](https://doi.org/10.1016/j.neuron.2017.05.025)
11. Gallego, J. A. et al. (2020). "Long-term stability of cortical population dynamics underlying consistent behavior." *Nature Neuroscience*, 23, 260–270. DOI: [10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4)
12. Sussillo, D. & Barak, O. (2013). "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks." *Neural Computation*, 25(3), 626–649. DOI: [10.1162/NECO_a_00409](https://doi.org/10.1162/NECO_a_00409)
13. Ostrow, M., Eisen, A., Kozachkov, L. & Fiete, I. (2023). "Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis." *NeurIPS*. arXiv: [2306.10168](https://arxiv.org/abs/2306.10168)
14. Murray, J. D. et al. (2014). "A hierarchy of intrinsic timescales across primate cortex." *Nature Neuroscience*, 17, 1661–1663. DOI: [10.1038/nn.3862](https://doi.org/10.1038/nn.3862)
15. Stringer, C. et al. (2019). "High-dimensional geometry of population responses in visual cortex." *Nature*, 571, 361–365. DOI: [10.1038/s41586-019-1346-5](https://doi.org/10.1038/s41586-019-1346-5)
16. Williams, A. H. et al. (2018). "Unsupervised Discovery of Demixed, Low-Dimensional Neural Dynamics across Multiple Timescales through Tensor Component Analysis." *Neuron*, 98(6), 1099–1115. DOI: [10.1016/j.neuron.2018.05.015](https://doi.org/10.1016/j.neuron.2018.05.015)
17. Giusti, C. et al. (2015). "Clique topology reveals intrinsic geometric structure in neural correlations." *PNAS*, 112(44), 13455–13460. DOI: [10.1073/pnas.1506407112](https://doi.org/10.1073/pnas.1506407112)

Dependency, information & plasticity
18. Schreiber, T. (2000). "Measuring Information Transfer" (transfer entropy). *Phys. Rev. Lett.*, 85, 461. DOI: [10.1103/PhysRevLett.85.461](https://doi.org/10.1103/PhysRevLett.85.461)
19. Tsodyks, M. & Markram, H. (1997). "The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability" (short-term plasticity). *PNAS*, 94(2), 719–723. DOI: [10.1073/pnas.94.2.719](https://doi.org/10.1073/pnas.94.2.719)
20. Marwan, N. et al. (2007). "Recurrence plots for the analysis of complex systems." *Physics Reports*, 438(5–6), 237–329. DOI: [10.1016/j.physrep.2006.11.001](https://doi.org/10.1016/j.physrep.2006.11.001)
21. Peyré, G., Cuturi, M. & Solomon, J. (2016). "Gromov–Wasserstein Averaging of Kernel and Distance Matrices." *ICML*, PMLR 48, 2664–2672.
