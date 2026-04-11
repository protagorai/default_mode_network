"""
Tests for the live visualization module: SimulationFrame, layouts,
callback wiring, brightness helper, environment, DMN, and backends.
"""

import math
import threading
import time

import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np

from sdmn.networks.celegans import (
    CElegansNetwork,
    SimulationFrame,
    build_small_world_network,
    build_connectome_network,
)
from sdmn.visualization.layouts import (
    grid_layout, graph_layout, anatomical_layout, compute_layout,
)
from sdmn.visualization.live_visualizer import LiveVisualizer, brightness
from sdmn.visualization.environment import WormEnvironment, WormState
from sdmn.visualization.fast_sim import _prepare_arrays


# ------------------------------------------------------------------
# SimulationFrame
# ------------------------------------------------------------------

class TestSimulationFrame:

    def test_construction(self):
        frame = SimulationFrame(
            time_ms=1.5, step=150,
            voltages={"N0": -60.0, "N1": -30.0},
        )
        assert frame.time_ms == 1.5
        assert frame.step == 150
        assert frame.active_synapses == []

    def test_with_connections(self):
        frame = SimulationFrame(
            time_ms=2.0, step=200,
            voltages={"A": -50.0},
            active_synapses=[("A", "B", 1.5)],
            active_gaps=[("A", "C", 0.3)],
        )
        assert len(frame.active_synapses) == 1
        assert len(frame.active_gaps) == 1


# ------------------------------------------------------------------
# Brightness
# ------------------------------------------------------------------

class TestBrightness:

    def test_at_rest(self):
        assert brightness(-65.0) == pytest.approx(0.0)

    def test_at_peak(self):
        assert brightness(-20.0) == pytest.approx(1.0)

    def test_clamped_below(self):
        assert brightness(-80.0) == 0.0

    def test_clamped_above(self):
        assert brightness(10.0) == 1.0

    def test_midpoint(self):
        mid = brightness(-42.5)
        assert 0.4 < mid < 0.6


# ------------------------------------------------------------------
# Layouts
# ------------------------------------------------------------------

class TestGridLayout:

    def test_correct_count(self):
        net = build_small_world_network(n_neurons=20, k_neighbors=4, random_seed=1)
        assert len(grid_layout(net)) == 20

    def test_positions_in_unit_square(self):
        net = build_small_world_network(n_neurons=20, k_neighbors=4, random_seed=1)
        for _, (x, y) in grid_layout(net).items():
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_no_overlapping_positions(self):
        net = build_small_world_network(n_neurons=30, k_neighbors=4, random_seed=1)
        seen = set()
        for _, (x, y) in grid_layout(net).items():
            key = (round(x, 6), round(y, 6))
            assert key not in seen
            seen.add(key)

    def test_connectome_layout(self):
        net = build_connectome_network()
        assert len(grid_layout(net)) == net.get_neuron_count()


class TestAnatomicalLayout:

    def test_correct_count(self):
        net = build_connectome_network()
        assert len(anatomical_layout(net)) == net.get_neuron_count()

    def test_bounded(self):
        net = build_connectome_network()
        for _, (x, y) in anatomical_layout(net).items():
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_head_neurons_are_left(self):
        net = build_connectome_network()
        pos = anatomical_layout(net)
        head_xs = [pos[n][0] for n in ["AWCL", "AWCR", "ASEL", "ASER"] if n in pos]
        motor_xs = [pos[n][0] for n in ["DA05", "VA05", "VB05"] if n in pos]
        if head_xs and motor_xs:
            assert np.mean(head_xs) < np.mean(motor_xs)


class TestGraphLayout:

    def test_correct_count(self):
        net = build_small_world_network(n_neurons=20, k_neighbors=4, random_seed=1)
        assert len(graph_layout(net)) == 20


class TestComputeLayout:

    def test_dispatch_grid(self):
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=1)
        assert len(compute_layout(net, mode="grid")) == 10

    def test_dispatch_graph(self):
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=1)
        assert len(compute_layout(net, mode="graph")) == 10

    def test_dispatch_anatomical(self):
        net = build_connectome_network()
        assert len(compute_layout(net, mode="anatomical")) == net.get_neuron_count()


# ------------------------------------------------------------------
# Callback wiring
# ------------------------------------------------------------------

class TestCallbackWiring:

    def test_register_and_invoke(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        frames = []
        net.register_callback(lambda f: frames.append(f))
        net.simulate(duration=1.0, progress=False)
        assert len(frames) > 0
        assert isinstance(frames[0], SimulationFrame)

    def test_unregister(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        frames = []
        cb = lambda f: frames.append(f)
        net.register_callback(cb)
        net.simulate(duration=0.5, progress=False)
        count_before = len(frames)
        net.unregister_callback(cb)
        net.simulate(duration=0.5, progress=False)
        assert len(frames) == count_before

    def test_build_frame_voltages(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        net.set_external_current("N0", 50.0)
        net.simulate(duration=1.0, progress=False)
        frame = net._build_frame()
        assert "N0" in frame.voltages


# ------------------------------------------------------------------
# WormEnvironment
# ------------------------------------------------------------------

class TestWormEnvironment:

    def test_initial_state(self):
        env = WormEnvironment()
        st = env.get_state()
        assert len(st.segments) == 12
        assert st.speed == 0.0

    def test_step_advances_position(self):
        env = WormEnvironment()
        voltages = {"DA01": -40.0, "VA01": -40.0, "DB01": -40.0, "VB01": -40.0}
        for _ in range(100):
            env.step(voltages, 0.1)
        st = env.get_state()
        assert st.speed > 0

    def test_chemical_field(self):
        env = WormEnvironment()
        field = env.get_chemical_field(resolution=10)
        assert len(field) == 10
        assert len(field[0]) == 10

    def test_sensory_currents_near_source(self):
        env = WormEnvironment()
        env._x = env.chemical_sources[0].x
        env._y = env.chemical_sources[0].y
        env.step({}, 0.1)
        currents = env.get_sensory_currents()
        assert any(c > 0 for c in currents.values())

    def test_to_json(self):
        env = WormEnvironment()
        j = env.to_json()
        assert "segments" in j
        assert "arena" in j
        assert "sources" in j


# ------------------------------------------------------------------
# LiveVisualizer core
# ------------------------------------------------------------------

class TestLiveVisualizerCore:

    def test_speed_control(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        viz.set_speed(0.5)
        assert viz.speed == pytest.approx(0.5)

    def test_pause_resume(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        assert not viz.is_paused
        viz.pause()
        assert viz.is_paused
        viz.resume()
        assert not viz.is_paused

    def test_positions_populated(self):
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        assert len(viz.positions) == 10

    def test_get_positions_list(self):
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        pl = viz.get_positions_list()
        assert len(pl) == 10

    def test_get_edges(self):
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        assert len(viz.get_edges()) > 0

    def test_sim_thread_produces_frames(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
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

    def test_stimulus_injection(self):
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        viz.inject_stimulus("N0", 100.0)
        assert "N0" in viz._pending_stimuli

    def test_dmn_applies_current(self):
        net = build_connectome_network()
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=True)
        arrays = _prepare_arrays(net)
        arrays["I_ext"][:] = 0.0
        viz._apply_dmn_to_arrays(arrays, 100.0)
        aval_idx = arrays["id_to_idx"].get("AVAL")
        if aval_idx is not None:
            assert arrays["I_ext"][aval_idx] != 0.0

    def test_environment_integration(self):
        net = build_connectome_network()
        viz = LiveVisualizer(net, layout="grid", enable_environment=True, enable_dmn=False)
        assert viz.environment is not None
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
        assert frame.env is not None
        assert "segments" in frame.env

    def test_timeline_data(self):
        net = build_small_world_network(n_neurons=10, k_neighbors=4, random_seed=1)
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
        assert frame.timeline is not None
        assert len(frame.timeline["times"]) > 0


# ------------------------------------------------------------------
# Matplotlib backend (Agg -- no window, just no-crash)
# ------------------------------------------------------------------

class TestMatplotlibBackendNoCrash:

    def test_instantiation(self):
        from sdmn.visualization.matplotlib_backend import MatplotlibVisualizer
        net = build_small_world_network(n_neurons=5, k_neighbors=2, random_seed=1)
        viz = LiveVisualizer(net, layout="grid", enable_environment=False, enable_dmn=False)
        renderer = MatplotlibVisualizer(viz)
        assert renderer is not None
