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

## Open Questions

- Is there a principled way to decide how much biological detail is "enough"? We need a framework for this, not just intuition.
- How do we handle the timescale mismatch between spiking (~ms), neuromodulation (~seconds), and learning (~minutes to hours) without the simulation grinding to a halt?
- What's the right level of abstraction for the physics simulator? Full Navier-Stokes is overkill, but pure kinematics might miss important sensory feedback loops.
- What biological substrate constraints govern wave propagation frequency and phase in recurrent circuits? Are there known biophysical parameters we should be building in from the start?
- How do we measure whether the network's "intrinsic simulation" is actually running a world model vs. just settling into an attractor state? What observable would distinguish the two?
- The multi-level prediction error idea maps to predictive coding (Rao & Ballard 1999). What *is* potentially new is doing it in genuinely cyclic (not hierarchical) circuits — what dynamics and learning properties emerge from that difference?

---

*Add ideas freely. Date and initial them if you like, or don't — this is informal.*
