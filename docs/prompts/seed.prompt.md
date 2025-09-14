I would like to do some ground-breaking curring-edge research into sythetic default mode networks. I am thinking about setting up new framework that would be based on model of artificial spiking neural networks, that are insired by biological neurons. At start it can be generic model, and over time it will evolve to match complexity, diversity and connectivity of human counterparts nad probably supersede it.

Let's start by adding directory structure that will hold documentation in docs folder, source code in src folder, scripts and tools in scripts folder, have README.md in the root, be dockerized (Dockerfile) that will be managed by podman (linux LTS latest version as base image).

I will write code in python for now with outlook of migrating to CUDA and CPU based C++ framework, but for now it will be python, while prototyping.

Create documentation with vision, architecture and implementation details. 
Project should work as simulation of the nervious system, so there will have to be a simulator engine as well.
There hsould be various neuron models I can interchange, so there should be also standardized interfaces across different neurons.
Expanding neuron models and/or changing the interface should be backward compatible where needed, so interface entrypoints should be planned in a stable, reasonable and logical way.

As helper tools, there should be network assembly graphical user interfac,e, which can get into any detail of arbitrary complex network, look into variables, state, inputs, outputs etc, in any step of the simulation once the simulation completes. In the future we will work on adding visualization in real time so that when it works as emulator, any portion of the system can be inspected as it runs. for now it's simulator, not emulator.

Please create plan fodler inside docs folder and create extensive documentation about all these aspects of the project to keep it all written down for future planing and prioritization.

To start with, implementation should focus on core simulation engine and objects that use simulation engine to advance and update their state.

Project needs to start minimal so that it is controlled how it expands and get's more sophisticated. Let's add very basic simulation engine, and create very simple spiking neurons and supporting cell models that can be componsed into network. Each model needs to ahve clearly and fully defined interfaces, so that connecting them is predictable.

Core idea of hte default mode network is to have looping (feedback loops) that, when scanned with probes (which are objects that are supported by the simulator, any point in the network, can be 'tracked' by a probe, which collects values at the point throughout the simulation timespan), plot 'synthetic neural netowrk waves" similar to brain-waves we record using EEG. THese will represent activity patterns and higher tier / order modes that network operates in at a given time, context, inputs and state.

Outputs in first pass should be:
- Full documentation that captures extensive and sophisticated design for basic elements of networks, simulation engine, probes, network / connectome tooling, 
- Network analysis and building tools, that use elementary entities like neurons, supportive cells and connection elements to build artificial neural netowrk,
- Models of neurons and supportive cells that will be used in simulations,
- Visualization tools to capture snapshits and trace probe outputs,  but also visualize structure of the network, helping understand behavior of the composed networks,

Add quickstart.md that shows how to:
1. Assemble simple network (with and without feedback loops),
2. Add monitoring or recording (like probes and other elements),
3. Start simulation, track and observe, collect outputs,
4. Visualize network behavior,
5. Debug and analyze what the simulation runs produced.