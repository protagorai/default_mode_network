"""
Example 09: Full C. elegans Connectome Simulation

Capstone demo that loads the real 302-neuron connectome, runs topology
analysis, simulates with sensory stimulation, then performs dynamical
analysis, behavioral detection, and generates visualisations.

Usage:
    python examples/09_full_connectome_simulation.py
"""

import os
import sys
from time import time as timer

import numpy as np

os.makedirs("output", exist_ok=True)

from sdmn.networks.celegans import (
    ConnectomeLoader,
    build_connectome_network,
)
from sdmn.analysis.topology import TopologyAnalyzer
from sdmn.analysis.dynamics import DynamicsAnalyzer
from sdmn.analysis.behavior import BehaviorDetector
from sdmn.analysis.visualization import NetworkVisualizer


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ======================================================================
# 1. Load connectome
# ======================================================================

def step1_load_connectome():
    section("Step 1: Loading C. elegans Connectome")

    loader = ConnectomeLoader()
    data = loader.load()

    print(f"  Neurons:           {data.n_neurons}")
    print(f"  Chemical synapses: {data.n_chemical}")
    print(f"  Gap junctions:     {data.n_gap}")

    meta = loader.get_neuron_metadata()
    from collections import Counter
    counts = Counter(meta.values())
    for cls, n in sorted(counts.items()):
        print(f"    {cls}: {n}")

    return loader


# ======================================================================
# 2. Build network
# ======================================================================

def step2_build_network(loader: ConnectomeLoader):
    section("Step 2: Building CElegansNetwork from Connectome")

    t0 = timer()
    network = loader.build_network(weight_scale=1.0, gap_conductance_scale=0.5)
    elapsed = timer() - t0

    summary = network.get_connectivity_summary()
    print(f"  Built in {elapsed:.2f} s")
    print(f"  Neurons:  {summary['n_neurons']}")
    print(f"  Synapses: {summary['n_chemical_synapses']}")
    print(f"  Gaps:     {summary['n_gap_junctions']}")
    print(f"  By class: {summary['neurons_by_class']}")

    return network


# ======================================================================
# 3. Topology analysis
# ======================================================================

def step3_topology_analysis(network):
    section("Step 3: Topology Analysis")

    t0 = timer()
    topo = TopologyAnalyzer.from_network(network)

    deg = topo.degree_distribution()
    print(f"  Mean degree:        {deg['total'].mean():.1f}")
    print(f"  Std degree:         {deg['total'].std():.1f}")
    print(f"  Max degree:         {deg['total'].max()}")

    cc = topo.clustering_coefficient()
    print(f"  Clustering coeff:   {cc:.4f}")

    apl = topo.average_path_length()
    print(f"  Avg path length:    {apl:.2f}")

    diam = topo.diameter()
    print(f"  Diameter:           {diam}")

    q = topo.modularity()
    print(f"  Modularity (Q):     {q:.4f}")

    hubs = topo.hub_neurons(10)
    print(f"  Top 10 hubs:")
    for name, d in hubs:
        print(f"    {name}: degree {d}")

    elapsed = timer() - t0
    print(f"\n  Topology analysis completed in {elapsed:.2f} s")

    return topo


# ======================================================================
# 4. Simulation with sensory stimulation
# ======================================================================

def step4_simulate(network, duration_ms: float = 200):
    section(f"Step 4: Simulating {duration_ms} ms with Sensory Stimulation")

    sensory_targets = ["AWCL", "AWCR", "ASEL", "ASER"]
    for nid in sensory_targets:
        if nid in network.neurons:
            network.set_external_current(nid, 60.0)
            print(f"  Injecting 60 pA into {nid}")

    t0 = timer()
    network.simulate(duration=duration_ms, progress=True)
    elapsed = timer() - t0

    print(f"\n  Simulation completed in {elapsed:.2f} s")
    print(f"  Real-time factor: {duration_ms / (elapsed * 1000):.1f}x")

    return network


# ======================================================================
# 5. Dynamical analysis
# ======================================================================

def step5_dynamics_analysis(network):
    section("Step 5: Dynamical Analysis")

    dyn = DynamicsAnalyzer.from_simulation(network)

    sync = dyn.mean_synchronization()
    print(f"  Mean synchronization:  {sync:.4f}")

    trace = dyn.mean_voltage_trace()
    print(f"  Mean voltage range:    [{trace.min():.1f}, {trace.max():.1f}] mV")

    active = dyn.active_neuron_count(threshold=-55.0)
    print(f"  Peak active neurons:   {active.max()}")

    traj, ev = dyn.pca_trajectories(3)
    print(f"  PCA explained var:     {ev[0]*100:.1f}%, {ev[1]*100:.1f}%, {ev[2]*100:.1f}%")

    if "AWCL" in dyn._id_index and "AVAL" in dyn._id_index:
        delay = dyn.propagation_delay("AWCL", "AVAL")
        print(f"  AWCL -> AVAL delay:    {delay:.2f} ms")

    return dyn


# ======================================================================
# 6. Behavioral detection
# ======================================================================

def step6_behavior(network):
    section("Step 6: Behavioral Detection")

    det = BehaviorDetector.from_simulation(network)
    report = det.behavioral_report()

    print(f"  Forward bouts:   {report['n_forward_bouts']}")
    print(f"  Reversal bouts:  {report['n_reversal_bouts']}")
    print(f"  Activity states: {report['activity_state_counts']}")

    return det


# ======================================================================
# 7. Visualization
# ======================================================================

def step7_visualize(topo, dyn, network):
    section("Step 7: Generating Visualizations")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = NetworkVisualizer.plot_degree_distribution(topo)
        fig.savefig("output/09_degree_distribution.png", dpi=150)
        plt.close(fig)
        print("  Saved output/09_degree_distribution.png")

        fig = NetworkVisualizer.plot_voltage_raster(dyn)
        fig.savefig("output/09_voltage_raster.png", dpi=150)
        plt.close(fig)
        print("  Saved output/09_voltage_raster.png")

        select_neurons = ["AWCL", "AIAL", "AVAL", "AVBL", "DA01", "VB01"]
        available = [n for n in select_neurons if n in dyn._id_index]
        if available:
            fig = NetworkVisualizer.plot_voltage_traces(dyn, available)
            fig.savefig("output/09_voltage_traces.png", dpi=150)
            plt.close(fig)
            print("  Saved output/09_voltage_traces.png")

        fig = NetworkVisualizer.plot_correlation_matrix(dyn)
        fig.savefig("output/09_correlation_matrix.png", dpi=150)
        plt.close(fig)
        print("  Saved output/09_correlation_matrix.png")

        fig = NetworkVisualizer.plot_pca_trajectory(dyn)
        fig.savefig("output/09_pca_trajectory.png", dpi=150)
        plt.close(fig)
        print("  Saved output/09_pca_trajectory.png")

        fig = NetworkVisualizer.plot_community_structure(topo)
        fig.savefig("output/09_community_structure.png", dpi=150)
        plt.close(fig)
        print("  Saved output/09_community_structure.png")

    except Exception as e:
        print(f"  Visualization error (non-fatal): {e}")


# ======================================================================
# Main
# ======================================================================

def main():
    print("\n" + "=" * 70)
    print("  C. ELEGANS FULL CONNECTOME SIMULATION")
    print("  302 neurons | Real connectome | Analysis pipeline")
    print("=" * 70)

    total_t0 = timer()

    loader = step1_load_connectome()
    network = step2_build_network(loader)
    topo = step3_topology_analysis(network)
    network = step4_simulate(network, duration_ms=200)
    dyn = step5_dynamics_analysis(network)
    det = step6_behavior(network)
    step7_visualize(topo, dyn, network)

    total_elapsed = timer() - total_t0

    section("Complete")
    print(f"  Total runtime: {total_elapsed:.1f} s")
    print(f"  Plots saved to output/09_*.png")
    print()


if __name__ == "__main__":
    main()
