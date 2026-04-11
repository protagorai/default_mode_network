"""
Integration tests for the visualization pipeline.

Validates the full end-to-end flow: build network -> simulate -> collect
frames -> render static outputs -> verify correctness. Uses the ``Agg``
matplotlib backend so no display is needed (CI-safe).

These tests exercise real simulation runs with real connectome data and
verify that every layer of the visualization stack produces valid output.
"""

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sdmn.networks.celegans import (
    CElegansNetwork,
    SimulationFrame,
    build_connectome_network,
    build_small_world_network,
)
from sdmn.analysis.topology import TopologyAnalyzer
from sdmn.analysis.dynamics import DynamicsAnalyzer
from sdmn.analysis.behavior import BehaviorDetector
from sdmn.analysis.visualization import NetworkVisualizer
from sdmn.visualization.live_visualizer import LiveVisualizer, brightness
from sdmn.visualization.layouts import grid_layout, graph_layout, compute_layout
from sdmn.visualization.browser_backend import BrowserVisualizer
from sdmn.visualization.matplotlib_backend import MatplotlibVisualizer


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def connectome_network():
    """Full 300-neuron connectome network (built once per module)."""
    return build_connectome_network()


@pytest.fixture(scope="module")
def simulated_connectome(connectome_network):
    """Connectome network after 10 ms simulation with sensory stimulus."""
    for nid in ["AWCL", "AWCR", "ASEL", "ASER"]:
        if nid in connectome_network.neurons:
            connectome_network.set_external_current(nid, 60.0)
    connectome_network.simulate(duration=10, progress=False)
    return connectome_network


@pytest.fixture(scope="module")
def small_stimulated():
    """Small 20-neuron network, simulated 20 ms with stimulus."""
    net = build_small_world_network(n_neurons=20, k_neighbors=6,
                                    rewire_prob=0.3, random_seed=42)
    net.set_external_current("N0", 80.0)
    net.set_external_current("N1", 50.0)
    net.simulate(duration=20, progress=False)
    return net


# ======================================================================
# 1. Callback -> Frame Pipeline (end-to-end)
# ======================================================================

class TestCallbackFramePipeline:
    """Verify frames flow correctly from simulation through callbacks."""

    @pytest.mark.integration
    def test_connectome_callbacks_produce_frames(self, connectome_network):
        """Full connectome produces valid frames during simulation."""
        frames = []
        connectome_network.register_callback(lambda f: frames.append(f))
        connectome_network.simulate(duration=2, progress=False)
        connectome_network.unregister_callback(frames.append)

        assert len(frames) > 0, "No frames emitted during connectome simulation"
        first = frames[0]
        assert isinstance(first, SimulationFrame)
        assert len(first.voltages) == connectome_network.get_neuron_count()
        assert first.time_ms >= 0
        assert first.step >= 0

    @pytest.mark.integration
    def test_frames_contain_real_voltage_dynamics(self):
        """Stimulated neuron shows voltage change across frames."""
        net = CElegansNetwork()
        net.add_sensory_neuron("S0")
        net.add_interneuron("I0")
        net.set_external_current("S0", 100.0)

        frames = []
        net.register_callback(lambda f: frames.append(f))
        net.simulate(duration=5, progress=False)
        net.unregister_callback(frames.append)

        v_first = frames[0].voltages["S0"]
        v_last = frames[-1].voltages["S0"]
        assert v_last > v_first, (
            f"Stimulated neuron should depolarise: first={v_first:.2f}, last={v_last:.2f}"
        )

    @pytest.mark.integration
    def test_frames_report_active_synapses_under_stimulus(self):
        """Stimulation should drive synaptic currents that appear in frames."""
        net = CElegansNetwork()
        net.add_sensory_neuron("S1")
        net.add_interneuron("I1")
        net.add_graded_synapse("S1", "I1", weight=5.0)
        net.set_external_current("S1", 100.0)

        frames = []
        net.register_callback(lambda f: frames.append(f))
        net.simulate(duration=5, progress=False)
        net.unregister_callback(frames.append)

        any_active = any(len(f.active_synapses) > 0 for f in frames)
        assert any_active, "Expected active synapses under strong stimulus"

    @pytest.mark.integration
    def test_frame_voltage_dict_matches_neuron_ids(self, connectome_network):
        """Every neuron in the network appears in the frame voltage dict."""
        frame = connectome_network._build_frame()
        network_ids = set(connectome_network.neurons.keys())
        frame_ids = set(frame.voltages.keys())
        assert frame_ids == network_ids

    @pytest.mark.integration
    def test_multiple_callbacks_all_receive_same_frames(self):
        """Multiple registered callbacks all see the same frame."""
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        buf_a, buf_b = [], []
        net.register_callback(lambda f: buf_a.append(f.step))
        net.register_callback(lambda f: buf_b.append(f.step))
        net.simulate(duration=2, progress=False)
        assert buf_a == buf_b


# ======================================================================
# 2. Layout Integration with Real Connectome
# ======================================================================

class TestLayoutIntegration:
    """Verify layouts produce valid, complete positions for real connectome."""

    @pytest.mark.integration
    def test_grid_layout_covers_all_connectome_neurons(self, connectome_network):
        positions = grid_layout(connectome_network)
        assert len(positions) == connectome_network.get_neuron_count()
        for nid in connectome_network.neurons:
            assert nid in positions, f"Missing position for {nid}"

    @pytest.mark.integration
    def test_grid_layout_classes_are_vertically_separated(self, connectome_network):
        """Sensory neurons should be above interneurons, which are above motors."""
        positions = grid_layout(connectome_network)
        sensory_ys, inter_ys, motor_ys = [], [], []
        for nid, (x, y) in positions.items():
            neuron = connectome_network.neurons[nid]
            cls = neuron.neuron_class.value if hasattr(neuron, "neuron_class") else "unknown"
            if cls == "sensory":
                sensory_ys.append(y)
            elif cls == "interneuron":
                inter_ys.append(y)
            elif cls == "motor":
                motor_ys.append(y)

        if sensory_ys and inter_ys:
            assert max(sensory_ys) < max(inter_ys) or min(sensory_ys) < min(inter_ys), \
                "Sensory band should be separated from interneuron band"

    @pytest.mark.integration
    def test_graph_layout_covers_all_connectome_neurons(self, connectome_network):
        positions = graph_layout(connectome_network)
        assert len(positions) == connectome_network.get_neuron_count()

    @pytest.mark.integration
    def test_positions_bounded(self, connectome_network):
        for mode in ("grid", "graph"):
            positions = compute_layout(connectome_network, mode=mode)
            for nid, (x, y) in positions.items():
                assert -0.01 <= x <= 1.01, f"[{mode}] {nid} x={x} out of bounds"
                assert -0.01 <= y <= 1.01, f"[{mode}] {nid} y={y} out of bounds"


# ======================================================================
# 3. LiveVisualizer Threaded Simulation
# ======================================================================

class TestLiveVisualizerThreaded:
    """Validate the threaded simulation driver inside LiveVisualizer."""

    @pytest.mark.integration
    def test_sim_thread_advances_simulation_time(self):
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=42)
        net.set_external_current("N0", 50.0)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        viz.set_speed(1.0)

        t = threading.Thread(target=viz._sim_thread, args=(50.0, 0.1), daemon=True)
        t.start()
        for _ in range(20):
            time.sleep(0.2)
            if viz.get_latest_frame() is not None:
                break
        viz.stop()
        t.join(timeout=3)

        frame = viz.get_latest_frame()
        assert frame is not None
        assert frame.sim.time_ms > 0

    @pytest.mark.integration
    def test_pause_halts_simulation(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        viz.set_speed(10000)
        viz.pause()

        t = threading.Thread(target=viz._sim_thread, args=(50.0, 0.01), daemon=True)
        t.start()
        time.sleep(0.3)

        frame = viz.get_latest_frame()
        assert frame is None or frame.sim.step == 0, "Paused simulation should not advance"

        viz.stop()
        t.join(timeout=3)

    @pytest.mark.integration
    def test_stop_terminates_thread(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        viz.set_speed(1.0)

        t = threading.Thread(target=viz._sim_thread, args=(100000.0, 0.1), daemon=True)
        t.start()
        time.sleep(0.5)
        viz.stop()
        t.join(timeout=5)
        assert not t.is_alive(), "Thread should terminate after stop()"

    @pytest.mark.integration
    def test_step_once_advances_one_recorded_frame(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        net.set_external_current("N0", 50.0)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        viz.set_speed(1.0)
        viz.pause()

        t = threading.Thread(target=viz._sim_thread, args=(100.0, 0.1), daemon=True)
        t.start()
        time.sleep(0.3)

        viz.step_once()
        time.sleep(1.0)

        frame = viz.get_latest_frame()
        assert frame is not None, "step_once should produce at least one frame"

        viz.stop()
        t.join(timeout=5)


# ======================================================================
# 4. Brightness Model Validation
# ======================================================================

class TestBrightnessModelIntegration:
    """Verify brightness mapping produces expected values for real simulation voltages."""

    @pytest.mark.integration
    def test_resting_neurons_are_dark(self, connectome_network):
        """Unstimulated neurons near rest should have near-zero brightness."""
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=1)
        frame = net._build_frame()
        for nid, v in frame.voltages.items():
            b = brightness(v)
            assert b < 0.15, f"Resting neuron {nid} (V={v:.1f}) should be dark, got brightness={b:.2f}"

    @pytest.mark.integration
    def test_stimulated_neuron_is_bright(self):
        """A strongly stimulated neuron should reach high brightness."""
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        net.set_external_current("N0", 200.0)
        net.simulate(duration=5, progress=False)

        frame = net._build_frame()
        b = brightness(frame.voltages["N0"])
        assert b > 0.3, f"Stimulated neuron should be bright, got {b:.2f}"

    @pytest.mark.integration
    def test_brightness_decays_after_stimulus_removed(self):
        """After removing stimulus, brightness should decrease over time."""
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        net.set_external_current("N0", 200.0)
        net.simulate(duration=5, progress=False)
        peak_v = net.neurons["N0"].voltage

        net.set_external_current("N0", 0.0)
        net.simulate(duration=10, progress=False)
        decay_v = net.neurons["N0"].voltage

        assert brightness(decay_v) < brightness(peak_v), \
            "Brightness should decay after stimulus removal"


# ======================================================================
# 5. Static Plot Output Validation
# ======================================================================

class TestStaticPlotOutputs:
    """Verify analysis visualization plots save correctly and contain expected structure."""

    @pytest.mark.integration
    def test_all_plot_types_save_to_disk(self, simulated_connectome):
        """Every NetworkVisualizer plot method produces a saveable PNG."""
        net = simulated_connectome
        topo = TopologyAnalyzer.from_network(net)
        dyn = DynamicsAnalyzer.from_simulation(net)

        with tempfile.TemporaryDirectory() as tmpdir:
            plots = {
                "degree_dist": NetworkVisualizer.plot_degree_distribution(topo),
                "network_graph": NetworkVisualizer.plot_network_graph(topo),
                "community": NetworkVisualizer.plot_community_structure(topo),
                "voltage_raster": NetworkVisualizer.plot_voltage_raster(dyn),
                "correlation": NetworkVisualizer.plot_correlation_matrix(dyn),
                "pca": NetworkVisualizer.plot_pca_trajectory(dyn),
            }

            for name, fig in plots.items():
                path = os.path.join(tmpdir, f"{name}.png")
                fig.savefig(path, dpi=72)
                plt.close(fig)
                assert os.path.exists(path), f"{name}.png was not created"
                assert os.path.getsize(path) > 1000, f"{name}.png is suspiciously small"

    @pytest.mark.integration
    def test_voltage_traces_plot_contains_correct_neurons(self, simulated_connectome):
        """Voltage trace plot for specific neurons produces a figure with correct line count."""
        dyn = DynamicsAnalyzer.from_simulation(simulated_connectome)
        target_neurons = ["AWCL", "AVAL", "DA01"]
        available = [n for n in target_neurons if n in dyn._id_index]
        fig = NetworkVisualizer.plot_voltage_traces(dyn, available)
        ax = fig.axes[0]
        assert len(ax.lines) == len(available), \
            f"Expected {len(available)} lines, got {len(ax.lines)}"
        plt.close(fig)

    @pytest.mark.integration
    def test_topology_comparison_plot(self, simulated_connectome):
        """Topology comparison bar chart renders for multiple topologies."""
        sw_net = build_small_world_network(n_neurons=30, k_neighbors=6,
                                           rewire_prob=0.3, random_seed=42)
        sw_topo = TopologyAnalyzer.from_network(sw_net)
        conn_topo = TopologyAnalyzer.from_network(simulated_connectome)
        reports = {
            "small-world": sw_topo.full_report(),
            "connectome": conn_topo.full_report(),
        }
        fig = NetworkVisualizer.plot_topology_comparison(reports)
        assert len(fig.axes) >= 4, "Expected at least 4 metric subplots"
        plt.close(fig)


# ======================================================================
# 6. Full Analysis Pipeline on Connectome
# ======================================================================

class TestFullAnalysisPipeline:
    """End-to-end: connectome -> simulate -> topology + dynamics + behavior."""

    @pytest.mark.integration
    def test_topology_analysis_on_connectome(self, simulated_connectome):
        topo = TopologyAnalyzer.from_network(simulated_connectome)
        report = topo.full_report()

        assert report["n_nodes"] == simulated_connectome.get_neuron_count()
        assert report["n_edges"] > 0
        assert report["mean_degree"] > 0
        assert 0 <= report["clustering_coefficient"] <= 1
        assert report["average_path_length"] > 0
        assert report["diameter"] >= 1

    @pytest.mark.integration
    def test_dynamics_analysis_on_connectome(self, simulated_connectome):
        dyn = DynamicsAnalyzer.from_simulation(simulated_connectome)

        assert dyn.n_neurons == simulated_connectome.get_neuron_count()
        assert dyn.n_timepoints > 0

        corr = dyn.pairwise_correlation_matrix()
        assert corr.shape[0] == dyn.n_neurons
        assert corr.shape[1] == dyn.n_neurons
        assert np.all(np.isfinite(corr))

        traj, ev = dyn.pca_trajectories(3)
        assert traj.shape[0] == 3
        assert all(e >= 0 for e in ev)

    @pytest.mark.integration
    def test_behavior_detection_on_connectome(self, simulated_connectome):
        det = BehaviorDetector.from_simulation(simulated_connectome)
        report = det.behavioral_report()

        assert "n_forward_bouts" in report
        assert "n_reversal_bouts" in report
        assert "activity_state_counts" in report
        assert isinstance(report["activity_state_counts"], dict)


# ======================================================================
# 7. LiveVisualizer + Connectome Full Integration
# ======================================================================

class TestLiveVisualizerConnectomeIntegration:
    """Full integration of LiveVisualizer with real connectome data."""

    @pytest.mark.integration
    def test_connectome_grid_layout_visualizer(self):
        """LiveVisualizer with grid layout on full connectome."""
        net = build_connectome_network()
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)

        assert len(viz.positions) == net.get_neuron_count()
        assert len(viz.get_edges()) > 0

        pos_list = viz.get_positions_list()
        classes_seen = set(item[3] for item in pos_list)
        assert "sensory" in classes_seen
        assert "interneuron" in classes_seen
        assert "motor" in classes_seen

    @pytest.mark.integration
    def test_connectome_graph_layout_visualizer(self):
        """LiveVisualizer with graph layout on full connectome."""
        net = build_connectome_network()
        viz = LiveVisualizer(net, layout="graph", enable_environment=False, enable_dmn=False)
        assert len(viz.positions) == net.get_neuron_count()

    @pytest.mark.integration
    def test_connectome_anatomical_layout_visualizer(self):
        """LiveVisualizer with anatomical layout on full connectome."""
        net = build_connectome_network()
        viz = LiveVisualizer(net, layout="anatomical", enable_environment=False, enable_dmn=False)
        assert len(viz.positions) == net.get_neuron_count()

    @pytest.mark.integration
    def test_threaded_connectome_simulation_produces_valid_frames(self):
        """Run the full connectome through the threaded sim and check frames."""
        net = build_connectome_network()
        net.set_external_current("AWCL", 60.0)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        viz.set_speed(1.0)

        t = threading.Thread(target=viz._sim_thread, args=(50.0, 0.1), daemon=True)
        t.start()
        for _ in range(20):
            time.sleep(0.2)
            if viz.get_latest_frame() is not None:
                break
        viz.stop()
        t.join(timeout=5)

        frame = viz.get_latest_frame()
        assert frame is not None
        assert len(frame.sim.voltages) == net.get_neuron_count()
        assert frame.sim.time_ms > 0

        assert len(frame.sim.voltages) == net.get_neuron_count()
        assert frame.sim.time_ms > 0


# ======================================================================
# 8. Browser Backend Protocol Validation (no actual server)
# ======================================================================

class TestBrowserBackendProtocol:
    """Validate the data structures the browser backend would send."""

    @pytest.mark.integration
    def test_init_payload_structure(self):
        """The init payload contains neurons and edges in expected format."""
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=42)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)

        payload = json.dumps({
            "type": "init",
            "neurons": viz.get_positions_list(),
            "edges": viz.get_edges(),
        })
        parsed = json.loads(payload)

        assert parsed["type"] == "init"
        assert len(parsed["neurons"]) == 10
        for item in parsed["neurons"]:
            assert len(item) == 4
            nid, x, y, cls = item
            assert isinstance(nid, str)
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            assert cls in ("sensory", "interneuron", "motor", "unknown")

        assert len(parsed["edges"]) > 0

    @pytest.mark.integration
    def test_frame_payload_structure(self):
        """A serialised frame contains the expected fields."""
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=42)
        net.set_external_current("N0", 50.0)
        net.simulate(duration=2, progress=False)

        frame = net._build_frame()
        msg = {
            "type": "frame",
            "time_ms": round(frame.time_ms, 3),
            "step": frame.step,
            "voltages": {k: round(v, 2) for k, v in frame.voltages.items()},
            "synapses": [[a, b, round(c, 2)] for a, b, c in frame.active_synapses],
            "gaps": [[a, b, round(c, 2)] for a, b, c in frame.active_gaps],
            "paused": False,
            "speed": 1.0,
        }
        serialised = json.dumps(msg)
        parsed = json.loads(serialised)

        assert parsed["type"] == "frame"
        assert isinstance(parsed["time_ms"], float)
        assert isinstance(parsed["step"], int)
        assert len(parsed["voltages"]) == 10
        assert isinstance(parsed["synapses"], list)
        assert isinstance(parsed["gaps"], list)

    @pytest.mark.integration
    def test_connectome_frame_payload_size(self):
        """Frame JSON for 300 neurons should be under 50 KB (reasonable for WS)."""
        net = build_connectome_network()
        net.set_external_current("AWCL", 60.0)
        net.simulate(duration=1, progress=False)

        frame = net._build_frame()
        msg = json.dumps({
            "type": "frame",
            "time_ms": round(frame.time_ms, 3),
            "step": frame.step,
            "voltages": {k: round(v, 2) for k, v in frame.voltages.items()},
            "synapses": [[a, b, round(c, 2)] for a, b, c in frame.active_synapses],
            "gaps": [[a, b, round(c, 2)] for a, b, c in frame.active_gaps],
        })
        size_kb = len(msg.encode("utf-8")) / 1024
        assert size_kb < 50, f"Frame payload is {size_kb:.1f} KB, should be < 50 KB"


# ======================================================================
# 9. Matplotlib Backend Rendering Validation (Agg, headless)
# ======================================================================

class TestMatplotlibBackendRendering:
    """Validate matplotlib backend creates proper figure structure."""

    @pytest.mark.integration
    def test_renderer_creates_scatter(self):
        """MatplotlibVisualizer sets up scatter plot with correct point count."""
        net = build_small_world_network(n_neurons=15, k_neighbors=4, random_seed=42)
        viz = LiveVisualizer(net, layout="grid")
        renderer = MatplotlibVisualizer(viz)

        meta = viz.get_positions_list()
        n = len(meta)
        xs = np.array([m[1] for m in meta])
        ys = np.array([m[2] for m in meta])

        fig, ax = plt.subplots(facecolor="#0a0a14")
        scatter = ax.scatter(xs, ys, s=40)
        assert scatter.get_offsets().shape == (n, 2)
        plt.close(fig)

    @pytest.mark.integration
    def test_renderer_color_update_with_frame(self):
        """Simulated frame data correctly maps to scatter colours."""
        from sdmn.visualization.matplotlib_backend import _CLASS_BASE_COLORS

        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=42)
        net.set_external_current("N0", 200.0)
        net.simulate(duration=3, progress=False)
        frame = net._build_frame()

        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        meta = viz.get_positions_list()
        ids = [m[0] for m in meta]
        classes = [m[3] for m in meta]
        base = np.array([_CLASS_BASE_COLORS.get(c, _CLASS_BASE_COLORS["unknown"]) for c in classes])

        bri = np.array([
            max(0.0, min(1.0, (frame.voltages.get(nid, -65.0) - (-65.0)) / 45.0))
            for nid in ids
        ])
        colors = base * (0.15 + bri[:, None] * 0.85)
        colors = np.clip(colors, 0, 1)

        stimulated_idx = ids.index("N0")
        assert bri[stimulated_idx] > bri[1:].min(), \
            "Stimulated neuron should be brighter than unstimulated"
        assert colors[stimulated_idx].max() > colors[-1].max(), \
            "Stimulated colour channel should be higher"


# ======================================================================
# 10. Cross-Component Data Consistency
# ======================================================================

class TestCrossComponentConsistency:
    """Verify data flows consistently across different modules."""

    @pytest.mark.integration
    def test_frame_voltages_match_analysis_voltages(self, small_stimulated):
        """Frame voltages and DynamicsAnalyzer voltages should be close to last recorded values.

        The frame is built from the *current* neuron state which may be a
        few integration steps past the last *recorded* sample, so we allow
        a tolerance of ~2 mV to account for the gap.
        """
        net = small_stimulated
        frame = net._build_frame()
        ids_sorted = sorted(net.voltage_history.keys())

        for nid in ids_sorted:
            frame_v = frame.voltages[nid]
            history_last = net.voltage_history[nid][-1][1]
            assert abs(frame_v - history_last) < 2.0, \
                f"{nid}: frame={frame_v:.2f} vs history={history_last:.2f}"

    @pytest.mark.integration
    def test_layout_ids_match_frame_ids(self, connectome_network):
        """Layout positions and frame voltages should cover identical neuron sets."""
        positions = grid_layout(connectome_network)
        frame = connectome_network._build_frame()

        assert set(positions.keys()) == set(frame.voltages.keys())

    @pytest.mark.integration
    def test_edges_in_visualizer_are_subset_of_network(self):
        """Every edge reported by LiveVisualizer exists in the network."""
        net = build_small_world_network(n_neurons=15, k_neighbors=4, random_seed=42)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        edges = viz.get_edges()
        all_neuron_ids = set(net.neurons.keys())
        for src, dst in edges:
            assert src in all_neuron_ids, f"Edge source {src} not in network"
            assert dst in all_neuron_ids, f"Edge target {dst} not in network"

    @pytest.mark.integration
    def test_analysis_and_viz_produce_consistent_degree_info(self, connectome_network):
        """TopologyAnalyzer degree and LiveVisualizer edges should be consistent."""
        topo = TopologyAnalyzer.from_network(connectome_network)
        viz = LiveVisualizer(connectome_network, layout="grid", enable_environment=False, enable_dmn=False)

        topo_edges = topo.graph.number_of_edges()
        viz_edges = len(viz.get_edges())

        assert viz_edges > 0
        assert topo_edges > 0
