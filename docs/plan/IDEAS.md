# Ideas & Brainstorm

A scratchpad for rough ideas, half-formed thoughts, and "what if" questions.
Nothing here is committed — it's a place to think out loud.

---

## Simulation & Modeling

- What if we modeled neuromodulator diffusion as a slow "weather system" overlaying the fast spiking network? Could give rise to mood-like states without explicitly programming them.
- Could we use graph wavelets on the connectome to find natural functional modules, rather than hand-defining subsystems?
- Explore treating the C. elegans body wall as a continuous sheet rather than discrete muscle cells — might simplify physics while preserving key dynamics.
- Dream/replay: during idle periods, replay recent sensory traces through the network with noise injected. See if this improves learning retention without any explicit memory module.

## Architecture & Engineering

- Plugin system for swapping neuron models at runtime — would let us A/B test Hodgkin-Huxley vs. graded potential vs. LIF on the same connectome without code changes.
- Could the probe system be extended into a "flight recorder" that continuously logs compressed snapshots, so we can rewind any simulation to an interesting moment?
- Investigate WebGPU for in-browser simulation visualization — lower barrier to entry than requiring a local Python install.

## Training & Learning

- What if the attention-based LLM and the spiking network trained adversarially — one generates behavior, the other tries to distinguish it from real worm data?
- Curriculum idea: start the worm in a perfectly uniform environment (no gradients) and slowly introduce asymmetry. Does it "discover" taxis on its own?
- Meta-learning across environments: train on many random chemical landscapes, then test few-shot adaptation to a novel one. Compare to biological adaptation speed.

## Biology & Neuroscience

- C. elegans has gap junctions (electrical synapses) that are often ignored in models. What behaviors break when we remove them? What emerges when we add them back?
- The pharyngeal nervous system is almost independent — could be modeled as a separate "gut brain" that communicates with the main nervous system through a narrow interface.
- Dauer formation (stress-induced developmental arrest) as a test case for homeostatic regulation — the worm basically decides to "hibernate." Can the model learn this?

## Wild / Long-term

- Two worms in the same simulator — can they develop simple social strategies (e.g., trail following, aggregation) without explicit social programming?
- Evolve the connectome itself, not just weights. Start from random small networks and see if evolution converges on something resembling the real C. elegans wiring.
- Use the trained C. elegans model as a "cognitive unit" and wire 302 of them together as neurons in a meta-network. Turtles all the way down.
- Port the minimal C. elegans controller to a real microrobot. Cheapest possible embodied neuroscience.

## Brain Waves as Online Learning Mechanism

- Recurrent circuits repeating and re-circulating inputs could produce brain wave-like oscillatory patterns in network activity. These oscillations might serve as a mechanism for online learning — each pass through the loop is an opportunity to compare the current representation against some internal ground truth or target, refining it incrementally.
- Different brain regions have distinct dynamics; these may stem directly from this recurrent compare-and-refine process rather than being a separate phenomenon.
- Circular flow of information through network regions should naturally produce propagating waves. There are likely biological substrate constraints (membrane time constants, synaptic delays, refractory periods) that govern the frequency and phase profile of such waves — worth investigating what these constraints are and whether they can be derived from first principles in our model.

## Intrinsic Simulation / World Model & Predictive Coding

> **Status (Apr 2026):** The intuition in this section has been formalized as a first-class architectural concept. See [docs/plan/inner_simulation_architecture.md](inner_simulation_architecture.md) for the full design, [docs/plan/world_simulation.md](world_simulation.md) for the dynamics-and-priors refinement (always-on substrate, stability-of-priors hierarchy, drive-driven priority field, default mode as priority-rest, C. elegans implementation path), and [docs/TWO_SIMULATIONS.md](../TWO_SIMULATIONS.md) for the concept split between the substrate engine and the inner simulation. The notes below are preserved as the informal antecedent.

**Note (Feb 2026 research):** These were initially captured as separate ideas, but research shows the "intrinsic simulation" concept closely mirrors the established **predictive coding / predictive processing** framework. The core insight — that the brain runs a continuous internal model, compares predictions to percepts at multiple layers, and uses the mismatch as a learning signal — is essentially what Rao & Ballard (1999) formalized and what Friston later generalized into the **free energy principle**. This is not a weakness; it means the intuition independently converged on a well-supported theory with deep literature to draw from. The value for this project is in *implementing* it in our cyclic architecture rather than reinventing the theory.

### The Idea (as originally conceived)

- The network always maintains an active internal simulation of the world — a "mental model" running continuously, not just responding to inputs. This simulation can be partial, contextual, covering only what's currently relevant.
- The core mechanism: compare the simulated/expected outcome with actual incoming percepts and propagate the difference. Crucially, this error signal should be apparent at many layers and regions of the network, not only at the output. The cyclic architecture of the circuits is key — prediction errors flow back through the same loops that generated the prediction.
- Analogy to adversarial networks but intrinsic: the network has its own "idea" of what the output should be and what input to expect given its current mental image. The mismatch between expectation and reality is the learning signal.
- **Concrete example (weather):** Two active concepts — (temperature, pressure, humidity) and (visual weather conditions). The network receives multidimensional input: gray clouds, wind, rain drops, knows it's February evening/twilight. From all prior experience, it predicts: very low temperature, high humidity, high pressure. But the actual temperature input contradicts this — it's warm. The coordinates are unfamiliar or it's a genuine anomaly. The mismatch between the expected thermal profile and the actual one creates an error signal that propagates through multiple layers, updating the internal model. The network doesn't just learn "I was wrong at the output" — it refines the intermediate representations that linked visual conditions to thermal expectations.
- This multi-level error propagation in cyclic circuits is fundamentally different from backprop's single backward pass through a feedforward graph.

### How This Maps to Predictive Coding Literature

- **Rao & Ballard (1999):** Hierarchical model where each layer has *expectation units* and *error units*. Expectations flow top-down, errors flow bottom-up. Errors propagate at every layer, not just the output — exactly the multi-level mechanism described above. Originally demonstrated on visual cortex (V1 receptive fields).
- **Friston's Free Energy Principle:** Generalizes predictive coding into a unified theory. The brain minimizes "free energy" (a bound on surprise/prediction error). Crucially includes **active inference** — the agent also *acts* to make the world match its predictions, not just passively updating the model. This adds the "simulation drives behavior" angle.
- **Connection to oscillations:** Predictive coding in cortical microcircuits naturally produces oscillatory dynamics. Feedforward prediction errors travel at **gamma frequencies**, top-down predictions at **alpha/beta frequencies**. This directly connects to the brain waves idea above — the waves are the *temporal signature* of the prediction-error message passing.

### What's Potentially Novel for This Project

- Most predictive coding implementations assume feedforward hierarchies with feedback. Our architecture has **genuinely cyclic circuits** (C. elegans-inspired), which could produce richer dynamics — prediction errors don't just flow "up" and "down" but circulate.
- The always-on, partial simulation aspect (context-dependent world model) is closer to Friston's active inference than to classic Rao-Ballard, but implementing it in a small recurrent network (rather than a deep hierarchy) is relatively unexplored.
- Using this as the **learning rule itself** (replacing backprop entirely) in our framework would be a concrete contribution — most predictive coding work is theoretical or uses simplified simulations.

### Key References

1. **Rao, Rajesh P. N. & Ballard, Dana H.** (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature Neuroscience*, 2(1), 79–87. DOI: [10.1038/4580](https://doi.org/10.1038/4580)
2. **Friston, Karl J.** (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127–138. DOI: [10.1038/nrn2787](https://doi.org/10.1038/nrn2787)
3. **Bastos, Andre M.; Usrey, W. Martin; Adams, Rick A.; Mangun, George R.; Fries, Pascal & Friston, Karl J.** (2012). "Canonical microcircuits for predictive coding." *Neuron*, 76(4), 695–711. DOI: [10.1016/j.neuron.2012.10.038](https://doi.org/10.1016/j.neuron.2012.10.038)
4. **Heeger, David J.** (2017). "Theory of cortical function." *Proceedings of the National Academy of Sciences*, 114(8), 1773–1782. DOI: [10.1073/pnas.1619788114](https://doi.org/10.1073/pnas.1619788114)
5. **Heeger, David J. & Mackey, Wayne E.** (2019). "Oscillatory recurrent gated neural integrator circuits (ORGaNICs), a unifying theoretical framework for neural dynamics." *Proceedings of the National Academy of Sciences*, 116(45), 22783–22794. DOI: [10.1073/pnas.1911633116](https://doi.org/10.1073/pnas.1911633116)

## Bidirectional Inter-Layer Communication

The predictive coding / intrinsic simulation ideas above require information to flow in two directions: forward (input → output) for inference/prediction, and backward (output → input) for learning from errors. This section collects the biological mechanisms that enable this and computational approaches we could try.

### How Biology Does It

**Separate forward and backward pathways (not the same wire):**
The cortex uses physically distinct connections for each direction. Feedforward projections originate from superficial layers (L2/3) and target L4 of the next area. Feedback projections originate from deep layers (L5/6) and target L1 and L6. They don't share the same axons — they are separate populations of neurons with different laminar origins, different target layers, and even different frequency bands (gamma for feedforward, alpha/beta for feedback). This is the architecture described in Bastos et al. (2012).

**Retrograde chemical signaling at individual synapses:**
Even at a single synapse, communication is not one-way. The postsynaptic neuron sends chemical messengers *backward* to the presynaptic terminal:
- **Endocannabinoids** — released from the postsynaptic dendrite, they cross the synaptic cleft backward and bind to CB1 receptors on the presynaptic terminal, suppressing further neurotransmitter release. This is a direct "turn down the signal" message from receiver to sender.
- **Nitric oxide (NO)** — a gas that diffuses freely through membranes in all directions. Released postsynaptically, it modulates presynaptic release probability and gates whether endocannabinoid signaling causes potentiation or depression.
- **BDNF (brain-derived neurotrophic factor)** — secreted by the postsynaptic cell, promotes survival and strengthening of the presynaptic connection.
- **Transsynaptic bridges** — physical protein complexes (e.g., neurexin-neuroligin) that span the synaptic cleft and allow bidirectional structural signaling, matching transmitter identity to receptor type.

**Spike-Timing-Dependent Plasticity (STDP):**
A local learning rule where the *relative timing* of pre and post spikes determines whether the synapse strengthens or weakens. Pre fires before post → strengthen (LTP). Post fires before pre → weaken (LTD). This is inherently bidirectional: the synapse "knows" about activity on both sides through the timing, without needing a global error signal. The temporal window is ~20-40ms.

**Gap junctions (electrical synapses):**
Direct electrical coupling between neurons through gap junction channels. Inherently bidirectional — current flows in whichever direction the voltage gradient dictates. Already present in C. elegans connectome data. These could serve as a fast, symmetric communication channel complementing the slower, asymmetric chemical synapses.

### Computational Approaches to Try

**1. Predictive coding with dual populations (most biologically grounded):**
Each "layer" in our network has two neuron populations: *prediction units* and *error units*. Prediction units send top-down to the layer below, error units send bottom-up to the layer above. No single connection needs to be bidirectional — the bidirectionality emerges from the circuit architecture. This is the Bastos et al. canonical microcircuit. Could be implemented directly in our framework.

**2. STDP as the local learning rule (no backprop needed):**
Let forward connections carry inference (spikes propagate input → output). Learning happens locally at each synapse via STDP — the timing relationship between connected neurons adjusts weights without any backward error signal. The question is whether STDP alone produces useful credit assignment across multiple layers, or whether it needs to be augmented.

**3. Dendritic Localized Learning (DLL) — recent, 2025:**
Inspired by pyramidal neuron anatomy. Each neuron has separate dendritic compartments for feedforward input (basal dendrites) and feedback/error input (apical dendrites). The error doesn't propagate backward through the same connections — it arrives on a separate dendritic tree. This solves the "weight transport problem" (backprop requires the backward pass to know the forward weights, which is biologically implausible). Published results show it works on MLPs, CNNs, and RNNs.

**4. Reciprocal feedback / pseudoinverse learning:**
Use autoencoder-like bidirectional connections where the backward weights learn to approximate the pseudoinverse of the forward weights. The network doesn't need to explicitly transport weights backward; the backward connections learn their own appropriate mapping. Each layer can compute its own local error.

**5. Generalized Latent Equilibrium (GLE):**
Each neuron has a local energy function. The network settles to equilibrium through local dynamics — forward and backward information propagation happens simultaneously through the same physical connections, but separated by phase (different temporal phases carry inference vs. error). This connects directly to the brain waves idea: oscillation phases could multiplex forward and backward signals on the same connections.

**6. Contrastive Hebbian Learning / Equilibrium Propagation:**
Run the network in two phases — a "free" phase (inference, forward) and a "clamped" phase (output nudged toward target). The difference in neural activities between phases gives a local learning signal at each synapse. No explicit backward pass. Works naturally in recurrent/cyclic networks, which makes it particularly interesting for our architecture.

### What Might Work Best for This Project

For our cyclic C. elegans-inspired architecture, the most promising approaches are probably:
- **Equilibrium propagation or contrastive Hebbian learning** — because they work natively in recurrent networks without requiring a feedforward/feedback hierarchy.
- **STDP** — because the C. elegans connectome already has temporal dynamics and we can implement it directly. Combined with neuromodulatory gating (dopamine-modulated STDP), it could handle credit assignment.
- **Phase-multiplexed signals (GLE-inspired)** — because our brain waves idea already implies oscillatory dynamics; using oscillation phase to separate inference from learning on the same connections would unify two of our ideas.

The key insight from biology: **there is no single backward channel.** The brain uses multiple overlapping mechanisms — separate anatomical pathways, retrograde chemical signals, spike timing, electrical coupling, and neuromodulatory gating — all operating at different timescales. We might want to layer several of these rather than picking just one.

### Additional References

6. **Caporale, Natalia & Dan, Yang** (2008). "Spike timing–dependent plasticity: a Hebbian learning rule." *Annual Review of Neuroscience*, 31, 25–46. DOI: [10.1146/annurev.neuro.31.060407.125639](https://doi.org/10.1146/annurev.neuro.31.060407.125639)
7. **Regehr, Wade G.; Bhatt, Dinakar H. & Bhatt, David J.** (2009). "Activity-dependent regulation of synapses by retrograde messengers." *Neuron*, 63(2), 154–170. DOI: [10.1016/j.neuron.2009.06.021](https://doi.org/10.1016/j.neuron.2009.06.021)
8. **Laborieux, Axel & Zenke, Friedemann** (2024). "Backpropagation through space, time, and the brain." arXiv: [2403.16933](https://arxiv.org/abs/2403.16933)
9. **Li, Gangyi et al.** (2025). "Dendritic Localized Learning: toward biologically plausible algorithm." arXiv: [2501.09976](https://arxiv.org/abs/2501.09976)
10. **Scellier, Benjamin & Bengio, Yoshua** (2017). "Equilibrium propagation: bridging the gap between energy-based models and backpropagation." *Frontiers in Computational Neuroscience*, 11, 24. DOI: [10.3389/fncom.2017.00024](https://doi.org/10.3389/fncom.2017.00024)

## Generative Models vs. Internal Imagination — Clarification

**Note (Feb 2026 research):** The "intrinsic simulation / autonomous internal imagination" described earlier and the concept of "generative models" in machine learning/neuroscience are **the same thing viewed from different angles**. In predictive coding, the brain's internal model is literally called a generative model — "generative" because it *generates* predictions (runs forward to produce expected sensory input), as opposed to a discriminative model that only classifies. When Friston describes the brain minimizing free energy with respect to its internal model, that model is a generative model by definition. The informal intuition ("the network has an imagination that produces expectations") and the formal term ("the brain maintains a hierarchical generative model") describe the same computational object.

The distinction that *does* matter: in ML, "generative model" often refers to models like GANs or VAEs that generate data samples. The brain's generative model is different in that it's not trying to produce realistic samples for an external observer — it's generating predictions *for itself*, to compare against incoming percepts. It's a generative model in service of inference, not in service of data synthesis.

## From Local Learning to Abstract Concepts

How do local synaptic learning rules (STDP, Hebbian, etc.) produce abstract, large-scale concepts when no single neuron has a global view?

### The Mechanism: Hierarchy + Local Rules + Time = Abstraction

- Each layer's local learning builds features (co-occurrence patterns, temporal regularities). These features become the *input vocabulary* for the next layer up. Layer 1 detects edges, layer 2 detects edge combinations (textures, corners), layer 3 detects object parts, etc. No layer "decides" to be abstract — abstraction emerges from the compositional stacking.
- Recent work (2025) on **objective-free local Hebbian learning** in hierarchical Hopfield networks shows this can happen without *any* global objective function. The system builds multi-scale compositional representations from scratch, including morphologically coherent symbolic structure, using only local correlation-based learning.
- **Information-theoretic local goals** (Wissner-Gross & Freer, 2025) derive each neuron's learning objective from local information-theoretic principles (maximize mutual information with inputs, minimize redundancy with peers). Abstract concepts emerge as the information-theoretically optimal compression at higher layers.

### The Gap for This Project

- Most demonstrations of "local learning → abstract concepts" use feedforward hierarchies. In our cyclic architecture, it's unclear how abstraction layers emerge when information circulates rather than flowing strictly upward. This is an open research question and a potential contribution.
- One hypothesis: in cyclic networks, abstraction may not be spatial (higher layers) but temporal — more abstract representations could correspond to slower dynamics (longer time constants), while concrete percepts correspond to fast dynamics. The hierarchy is in time, not in space.

## Context Switching, Zoom, and Granularity Control

### The Idea

For the internal model to produce relevant predictions, it must set the correct **context** and **level of detail** (zoom). A model reasoning about weather needs different granularity than one navigating a room. The ability to dynamically switch context and adjust the "zoom level" of its point of view is essential for context-aware prediction. This is not just an engineering convenience — it may be a fundamental cognitive capability that our architecture should support.

### How Predictive Coding Already Handles This: Precision Weighting

In Friston's framework, **attention IS precision weighting**. Each prediction error signal has an associated precision (confidence/inverse-variance). The brain modulates precision to control which errors matter:
- **High precision at fine-grained sensory layers** = zooming in on sensory details (e.g., attending to exact temperature reading).
- **High precision at abstract layers** = zooming out to the big picture (e.g., attending to the overall weather pattern, ignoring moment-to-moment fluctuations).
- **Context switching** = rapidly reallocating precision weights across the hierarchy. Shifting from "navigating a room" to "having a conversation" means suppressing spatial prediction errors and amplifying linguistic ones.

Dopamine plays a key biological role in modulating precision. Disrupted precision weighting is implicated in psychosis (everything feels equally surprising/relevant — no zoom control).

### SOTA Approaches and Considerations

**Adaptive attention spans:**
- **Mixture of Attention Spans (MoA, 2024):** Different attention heads use different window sizes, capturing heterogeneous context scales simultaneously. 3.9x effective context increase.
- **Dynamic Hierarchical Sparse Attention (DHSA):** Predicts the right sparsity/granularity at runtime per input, rather than using a fixed template. 12-20% accuracy gains over static approaches.

**Multi-resolution / multi-scale processing:**
- **Scaling on Scales (S², 2024):** Applies the same model at multiple resolutions simultaneously, merging representations across scales. Shows that multi-scale small models match single-scale large models.
- **Dragonfly (2024):** Multi-resolution zoom-in encoding for vision-language models, extracting features at multiple crop scales and merging them.

**For our architecture specifically:**
- Precision weighting could be implemented as a gain control on each neuron or layer — a scalar multiplier on prediction error signals, modulated by a slow neuromodulatory signal (dopamine analog).
- Context could be represented as a persistent activity pattern in a "context layer" that biases the precision weights across the rest of the network. Switching context = changing this pattern.
- Zoom could emerge from the temporal hierarchy idea: attending to fast dynamics = fine zoom, attending to slow dynamics = coarse zoom. The network doesn't need an explicit zoom parameter — it modulates which timescale it "listens to."

### How to Evaluate and Test

- **Context-switching benchmark:** Design scenarios that require abrupt context changes (e.g., the C. elegans model transitions from foraging to escape behavior when a noxious stimulus appears). Measure: how quickly does the model reallocate precision? How many time steps of degraded prediction before the new context is established?
- **Zoom-level benchmark:** Present stimuli that have both fine-grained and coarse structure (e.g., a chemical gradient that also has local noise). Measure: can the model attend to the gradient (coarse) while ignoring noise (fine), and can it switch to attending to local concentration when that becomes behaviorally relevant?
- **Precision ablation:** Remove or fix the precision weighting mechanism and compare performance. If zoom/context control matters, fixed-precision models should fail on tasks that require dynamic scale-switching.
- **Comparison to biological data:** C. elegans shows measurable context-dependent behavioral switching (e.g., different locomotion strategies on vs. off food). Match the model's switching dynamics to biological recordings.

### Additional References

11. **Feldman, Harriet & Friston, Karl J.** (2010). "Attention, uncertainty, and free-energy." *Frontiers in Human Neuroscience*, 4, 215. DOI: [10.3389/fnhum.2010.00215](https://doi.org/10.3389/fnhum.2010.00215)
12. **Kanai, Ryota et al.** (2015). "Cerebral hierarchies: predictive processing, precision and the pulvinar." *Philosophical Transactions of the Royal Society B*, 370(1668). DOI: [10.1098/rstb.2014.0169](https://doi.org/10.1098/rstb.2014.0169)
13. **Millidge, Beren; Seth, Anil & Buckley, Christopher L.** (2021). "Predictive coding: a theoretical and experimental review." arXiv: [2107.12979](https://arxiv.org/abs/2107.12979)
14. **Ororbia, Alexander G. & Mali, Ankur** (2022). "The neural coding framework for learning generative models." *Nature Communications*, 13, 2064. DOI: [10.1038/s41467-022-29632-7](https://doi.org/10.1038/s41467-022-29632-7)
15. **Salvatori, Tommaso et al.** (2024). "Learnable precision in hierarchical predictive coding implements Natural Gradient Descent." arXiv: [2111.06942](https://arxiv.org/abs/2111.06942)

## Idea Graft — Cross-Instance Neural Knowledge Transfer

> **Status (Jun 2026):** Formalized in its own document — see [docs/plan/idea_graft.md](idea_graft.md). The summary below is the candidate antecedent.

- Given two distinct instantiations of the network (two synthetic organisms), fit a mapping between their neuronal firing **sequences** and **rates**, then define a representation *outside either instance* (an "idea" / *eidos* / interlingua) that both organisms' dynamics map to and from.
- Use that shared representation not only to compare but to **graft**: induce learning in, or evoke latent knowledge from, the recipient's cell ensemble — carrying activity, plasticity, and dynamical state, not just static weights or a snapshot.
- Horticultural framing: the donor fragment is the **scion**, the recipient is the **rootstock**, and a successful, self-retaining transfer is a **take**.
- Beyond rate/sequence/correlation, the principle requires measuring population geometry (manifolds, RDMs), dynamical structure (attractors, intrinsic timescales, DSA), oscillatory/spectral and information-theoretic structure, and crucially the plasticity and neuromodulatory/precision state. The full measurement program, alignment operators (CCA, CKA, RSA, Gromov–Wasserstein OT, DSA), evaluation protocol, and falsification criteria are in [idea_graft.md](idea_graft.md).
- Closest prior art: hyperalignment (Haxby 2011), model stitching, the Platonic Representation Hypothesis (Huh 2024), brain-to-brain interfaces (Pais-Vieira 2013), and engram/RNA transfer (Bédécarrats 2018). The novelty is the conjunction — translation *as the substrate for* transfer, carrying plasticity and dynamical state, in cyclic circuits.

## Open Questions

- Is there a principled way to decide how much biological detail is "enough"? We need a framework for this, not just intuition.
- How do we handle the timescale mismatch between spiking (~ms), neuromodulation (~seconds), and learning (~minutes to hours) without the simulation grinding to a halt?
- What's the right level of abstraction for the physics simulator? Full Navier-Stokes is overkill, but pure kinematics might miss important sensory feedback loops.
- What biological substrate constraints govern wave propagation frequency and phase in recurrent circuits? Are there known biophysical parameters we should be building in from the start?
- How do we measure whether the network's "intrinsic simulation" is actually running a world model vs. just settling into an attractor state? What observable would distinguish the two?
- The multi-level prediction error idea maps to predictive coding (Rao & Ballard 1999). What *is* potentially new is doing it in genuinely cyclic (not hierarchical) circuits — what dynamics and learning properties emerge from that difference?
- In cyclic networks, does abstraction emerge temporally (slow dynamics = abstract, fast = concrete) rather than spatially (higher layers = abstract)? How would we detect this?
- What is the minimal precision weighting mechanism needed for effective context switching? Is a single scalar per layer enough, or does it need to be per-neuron?
- Can the network learn to set its own precision weights (meta-learning the zoom), or do they need to be externally modulated?

---

*Add ideas freely. Date and initial them if you like, or don't — this is informal.*
