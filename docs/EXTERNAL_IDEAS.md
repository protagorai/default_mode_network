I found following project:

https://en.wikipedia.org/wiki/OpenWorm

That has to do with simulation of a biological organism, namely C. Elegans in the computer. Are there similar projects that target more complex creatires? LIst them and link them please if so, otherwise point me to the research SOTA in this field

I’m checking both “whole-organism” efforts like OpenWorm and the closest larger-animal projects, because for more complex creatures the field often splits into body simulation, connectomics, and brain-only simulation rather than one complete digital organism.

I’ve got a solid set already. The main pattern is that once you move beyond C. elegans, “full organism simulation” mostly turns into one of three things: embodied body models, whole-brain/connectome projects, or partial brain-region simulations.

Thought for 2m 21s

Yes — but for organisms more complex than C. elegans, there are very few true “OpenWorm-style” full-organism projects. The field mostly splits into three tracks instead: embodied body simulations, connectome/atlas projects, and brain-only or brain-network simulations. That split is basically the present state of the art.

Here are the closest things to what you are looking for:

NeuroMechFly / FlyGym — probably the closest larger-creature analogue to OpenWorm on the body + behavior side. It is a digital twin of the adult fruit fly (Drosophila melanogaster) that can interact with the environment and is designed for neuromechanical simulation.
Fruit Fly Brain Observatory + Neurokernel — this is one of the clearest brain-emulation efforts for a more complex animal than C. elegans. It focuses on the fruit fly brain, integrating data and circuit models and providing an open platform for emulation and exploration.
FlyWire — not itself a full simulator, but a foundational project for one. It provides a whole-brain connectome of an adult female fruit fly, which is exactly the kind of structural dataset needed before realistic whole-brain simulation becomes possible.
Janelia / Google DeepMind virtual fruit fly body model — this is a very advanced whole-body physics simulation of a fruit fly. It is highly realistic and useful for studying locomotion and sensorimotor control, though its controller is not the full biological fly brain in the OpenWorm sense.
Virtual rodent (Harvard + Google DeepMind) — a biomechanically realistic digital rat controlled by neural networks trained on real rat behavior. This is a major step up in complexity, but it is still not a faithful full biological rat reconstruction; it is an embodied neuroscience model.
Blue Brain Project — one of the flagship efforts in biologically detailed mouse-brain reconstruction and simulation. Its scope is much larger than OpenWorm in neural complexity, but it is focused on brain simulation rather than a full animal body-environment digital organism.
Allen Institute mouse cortex simulations / MICrONS — these are among the strongest current efforts toward detailed computational mouse-brain models. Allen describes a proof-of-concept biophysically detailed simulation of the entire mouse cortex, and MICrONS links large-scale structure and function in mouse visual cortex, including a “functional digital twin” direction.
The Virtual Brain (TVB) / EBRAINS — for humans and large brains, this is one of the major platforms for whole-brain network simulation. It is powerful and clinically relevant, but it is not a neuron-by-neuron emulation of a complete human organism.
Zebrafish atlas efforts: mapZebrain, Z-Brain, ViBE-Z — these are important adjacent resources for a more complex vertebrate model. They provide whole-brain atlases and activity-mapping infrastructure, but not yet a full OpenWorm-style digital zebrafish.

One more thing worth noting: in March 2026, Eon Systems showed a work-in-progress embodied fruit fly brain emulation and explicitly described it as an early step rather than a finished digital organism. It is interesting, but I would treat it as promising and not yet a settled academic benchmark.

So the honest answer is:

Yes, similar projects exist.
But no, there is not yet a clean “OpenWorm, but for mouse/human” equivalent that fully recreates a more complex creature with high biological fidelity across brain + body + environment + behavior.

The current SOTA looks like this:

Fruit fly is the strongest frontier for near-term whole-brain/embodied work, because it now has a full adult connectome plus strong body simulators.
Mouse is the strongest frontier for large-scale mammalian simulation, but mostly in partial form: cortex, visual system, cell atlases, and detailed regional or whole-cortex simulations rather than a full digital mouse.
Human work is still mainly at the level of whole-brain network models and infrastructure, not cellular whole-brain emulation. A 2025 projection paper said mouse cellular whole-brain simulation might become feasible around 2034, marmoset around 2044, and human later than that.

So if you want the closest modern answer to “what comes after OpenWorm?”, I would point to this ladder:

OpenWorm → NeuroMechFly / Neurokernel / FlyWire → virtual rodent → Blue Brain / Allen mouse cortex / MICrONS → TVB / EBRAINS for human-scale brain models.

I can also turn this into a ranked table with columns like organism, full body?, full brain?, connectome-based?, open source?, official link.