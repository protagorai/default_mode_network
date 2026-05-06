# Two Simulations

This project hosts two distinct concepts that both wear the word "simulation". They are related, but they are not the same thing, and they answer different questions. This document names the split, draws the boundary, and points you at the right place for details.

## The split

**Substrate simulation engine** — the machinery that integrates neuron voltages, propagates spikes, applies synaptic kinetics, and advances time. It is a numerical integrator for the equations of motion of a biological nervous system. It is what we use to *run* an entity — for example, a C. elegans connectome.

**Inner simulation** — the ongoing, emergent predictive process that the substrate gives rise to when it is operating. It is not integrated by hand; it is what the network is *doing* on top of the physics. It is a continuously-updated generative model of the body and the world, distributed across many sub-simulators operating at many levels of abstraction, with focus rapidly shifting between them and with measurable mismatch signals whenever expectation and reality part ways.

## Responsibilities

| | Substrate simulation engine | Inner simulation |
|---|---|---|
| What it is | A numerical integrator | An emergent predictive process |
| Operates on | Voltages, spikes, synaptic state, ion channels | Expectations, prediction errors, precision, policies |
| Native timescale | The integration time step dt (milliseconds) | Any; includes timescales longer than any single behavior |
| Where it lives in code | [src/sdmn/core/](../src/sdmn/core/) | Inside the substrate's neurons and connectivity — not beside them |
| Owned by | Us, the framework authors | The running entity, as emergent behavior |
| What switching it off does | The entity freezes | The entity keeps existing but stops predicting the world |
| Analogy | A physics engine | The dreaming that happens inside a running brain |

## Coupling

The inner simulation is not a parallel process sitting next to the substrate. It is realized *in* the substrate: its prediction units, error units, and precision state are patterns of activity and connectivity within the same neurons the substrate engine is integrating. The substrate engine knows nothing about prediction; it just advances voltages. The inner simulation knows nothing about integrator time steps; it operates in the natural vocabulary of activity.

This has two consequences worth stating up front:

1. You cannot debug the inner simulation by reading only the substrate engine's state variables. You need higher-order observables — error signals, precision indicators, modular activity — that are defined on top of the raw voltages.
2. You cannot fix the inner simulation by changing the integrator. Improvements to the inner simulation require changes to what the network is computing (its connectivity, its learned weights, its neuromodulatory state), not to how fast or accurately the substrate advances it.

## Why both matter

C. elegans is the first entity we simulate, and running it is the platform-utilization story. The substrate engine does that work. But the project is not ultimately about integrating a connectome well. It is about understanding what a running nervous system *does*, which is: it models the world it lives in, continuously, at multiple levels, and it notices when that model is wrong. That is the inner simulation, and it is the scientific heart of the project.

## Where to read next

- [docs/plan/inner_simulation_architecture.md](plan/inner_simulation_architecture.md) — the full design of the inner simulation: sub-simulators, hierarchy, functional modules, prediction-error-precision mechanics, focus shifts, observable signatures.
- [docs/plan/world_simulation.md](plan/world_simulation.md) — the dynamics-and-priors layer of the inner simulation: signal propagation and decay, the always-on substrate, priors organized by stability timescale, drives and the priority field, and the implementation path for a first running entity.
- [docs/plan/neuron_units.md](plan/neuron_units.md) — the functional basis and complexity spectrum of the substrate's atomic processing units, plus the research program for empirically identifying the minimum-complexity unit that can host the inner simulation.
- [docs/plan/architecture.md](plan/architecture.md) — the architecture of the substrate engine and its supporting components.
- [docs/plan/self_awareness_architecture.md](plan/self_awareness_architecture.md) — the higher-level research and ethics framing; its specific components are now understood as particular sub-simulators within the inner simulation.
- [docs/plan/IDEAS.md](plan/IDEAS.md) — especially the "Intrinsic Simulation / World Model & Predictive Coding" and "Context Switching, Zoom, and Granularity Control" sections. The intuitions captured there are what this two-concept split formalizes.
- [docs/plan/vision.md](plan/vision.md) — the broader research vision for the framework.

## Non-goals of this document

This document does not specify:

- How sub-simulators are implemented in code.
- Where inner-simulation code will live on disk.
- Which prediction, error, and precision mechanics the project adopts in detail. See the architecture doc for those.
- How the inner simulation is validated or measured. See the architecture doc's "Observable Signatures" section.

It is intentionally short. Its only job is to name the split and make sure downstream docs, code, and discussion do not confuse the two meanings of "simulation".
