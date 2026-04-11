"""
Core live-visualization controller with environment simulation and DMN baseline.

``LiveVisualizer`` attaches to a ``CElegansNetwork``, drives the simulation
with optional DMN background activity and a 2D worm environment, and
dispatches frames to a rendering backend.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sdmn.networks.celegans.network_manager import CElegansNetwork, SimulationFrame
from sdmn.visualization.layouts import compute_layout
from sdmn.visualization.environment import WormEnvironment
from sdmn.visualization.fast_sim import _prepare_arrays, _fast_step, build_frame_from_arrays, get_plasticity_stats

V_REST = -65.0
V_PEAK = -20.0


def brightness(voltage: float) -> float:
    """Map membrane voltage to [0, 1] brightness for visualisation."""
    return max(0.0, min(1.0, (voltage - V_REST) / (V_PEAK - V_REST)))


@dataclass
class VisualizationFrame:
    """Extended frame that includes environment state and activity timeline."""
    sim: SimulationFrame
    env: Optional[dict] = None
    timeline: Optional[dict] = None
    plasticity: Optional[dict] = None
    dmn_active: bool = True


class LiveVisualizer:
    """
    Drives a simulation with DMN baseline + environment and streams frames.

    Usage::

        viz = LiveVisualizer(network, layout="anatomical", enable_environment=True)
        viz.run(duration_ms=30000, speed=0.5, backend="browser")
    """

    def __init__(self, network: CElegansNetwork,
                 layout: str = "anatomical",
                 enable_environment: bool = True,
                 enable_dmn: bool = True,
                 dmn_amplitude: float = 35.0,
                 dmn_frequency: float = 0.08,
                 **layout_kwargs):
        self.network = network
        self.positions = compute_layout(network, mode=layout, **layout_kwargs)

        self._neuron_classes: Dict[str, str] = {}
        for nid, neuron in network.neurons.items():
            if hasattr(neuron, "neuron_class"):
                self._neuron_classes[nid] = neuron.neuron_class.value
            else:
                self._neuron_classes[nid] = "unknown"

        self.enable_environment = enable_environment
        self.environment: Optional[WormEnvironment] = None
        if enable_environment:
            self.environment = WormEnvironment()

        # DMN config: prefer network config, fall back to constructor args
        dc = network.config.dmn
        self.enable_dmn = dc.enabled if dc.enabled else enable_dmn
        self.dmn_amplitude = dc.amplitude if dc.enabled else dmn_amplitude
        self.dmn_frequency = dc.frequency if dc.enabled else dmn_frequency
        self.dmn_mode = dc.mode
        self._dmn_rng = np.random.default_rng(42)
        self._dmn_phases: Dict[str, float] = {}
        target_class = dc.target_class if dc.enabled else "interneuron"
        for nid, cls in self._neuron_classes.items():
            if cls == target_class:
                self._dmn_phases[nid] = self._dmn_rng.uniform(0, 2 * math.pi)

        # Recurrent DMN state: adaptation current per DMN neuron
        self._dmn_adaptation: Dict[str, float] = {nid: 0.0 for nid in self._dmn_phases}
        self._dmn_recurrent_tau = dc.recurrent_tau_adapt
        self._dmn_recurrent_boost = dc.recurrent_boost

        # Engine selection
        self.engine: str = "vectorized"

        self._speed: float = 1.0
        self._paused = threading.Event()
        self._paused.set()
        self._single_step = threading.Event()
        self._stop = threading.Event()
        self._latest_frame: Optional[VisualizationFrame] = None
        self._lock = threading.Lock()

        self._timeline_sensory: List[float] = []
        self._timeline_inter: List[float] = []
        self._timeline_motor: List[float] = []
        self._timeline_times: List[float] = []
        self._timeline_max = 200

        self._pending_stimuli: Dict[str, float] = {}
        self._stimuli_lock = threading.Lock()
        self._restart_flag = threading.Event()
        self._sim_arrays: Optional[dict] = None

    # ------------------------------------------------------------------
    # Speed / pause / stimulus control (thread-safe)
    # ------------------------------------------------------------------

    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, factor: float) -> None:
        self._speed = max(0.001, factor)

    def pause(self) -> None:
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    def step_once(self) -> None:
        self._single_step.set()

    def stop(self) -> None:
        self._stop.set()
        self._paused.set()

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def set_engine(self, engine: str) -> None:
        """Switch simulation engine at runtime: 'vectorized' or 'object'."""
        if engine in ("vectorized", "object"):
            self.engine = engine

    def set_dmn_mode(self, mode: str) -> None:
        """Switch DMN mode at runtime: 'oscillator' or 'recurrent'."""
        if mode in ("oscillator", "recurrent"):
            self.dmn_mode = mode
            self._dmn_adaptation = {nid: 0.0 for nid in self._dmn_phases}

    def inject_stimulus(self, neuron_id: str, current_pA: float) -> None:
        """Queue a stimulus to be applied on the next sim step."""
        with self._stimuli_lock:
            self._pending_stimuli[neuron_id] = current_pA

    def add_chemical_source(self, x: float, y: float,
                            strength: float = 1.0, radius: float = 80.0) -> None:
        if self.environment:
            self.environment.add_chemical_source(x, y, strength, radius)

    # ------------------------------------------------------------------
    # Frame callback
    # ------------------------------------------------------------------

    def _on_frame(self, frame: SimulationFrame) -> None:
        env_data = None
        if self.environment:
            env_data = self.environment.to_json()

        s_vals = [brightness(frame.voltages.get(n, V_REST))
                  for n, c in self._neuron_classes.items() if c == "sensory"]
        i_vals = [brightness(frame.voltages.get(n, V_REST))
                  for n, c in self._neuron_classes.items() if c == "interneuron"]
        m_vals = [brightness(frame.voltages.get(n, V_REST))
                  for n, c in self._neuron_classes.items() if c == "motor"]

        self._timeline_times.append(frame.time_ms)
        self._timeline_sensory.append(sum(s_vals) / max(len(s_vals), 1))
        self._timeline_inter.append(sum(i_vals) / max(len(i_vals), 1))
        self._timeline_motor.append(sum(m_vals) / max(len(m_vals), 1))

        if len(self._timeline_times) > self._timeline_max:
            self._timeline_times = self._timeline_times[-self._timeline_max:]
            self._timeline_sensory = self._timeline_sensory[-self._timeline_max:]
            self._timeline_inter = self._timeline_inter[-self._timeline_max:]
            self._timeline_motor = self._timeline_motor[-self._timeline_max:]

        timeline_data = {
            "times": [round(t, 2) for t in self._timeline_times[-self._timeline_max:]],
            "sensory": [round(v, 3) for v in self._timeline_sensory[-self._timeline_max:]],
            "inter": [round(v, 3) for v in self._timeline_inter[-self._timeline_max:]],
            "motor": [round(v, 3) for v in self._timeline_motor[-self._timeline_max:]],
        }

        plasticity_data = None
        if self._sim_arrays is not None:
            plasticity_data = get_plasticity_stats(self._sim_arrays)

        viz_frame = VisualizationFrame(
            sim=frame,
            env=env_data,
            timeline=timeline_data,
            plasticity=plasticity_data,
            dmn_active=self.enable_dmn,
        )

        with self._lock:
            self._latest_frame = viz_frame

    def get_latest_frame(self) -> Optional[VisualizationFrame]:
        with self._lock:
            return self._latest_frame

    # ------------------------------------------------------------------
    # Metadata helpers for renderers
    # ------------------------------------------------------------------

    def get_neuron_ids(self):
        return sorted(self.network.neurons.keys())

    def get_positions_list(self):
        result = []
        for nid in sorted(self.positions):
            x, y = self.positions[nid]
            result.append((nid, x, y, self._neuron_classes.get(nid, "unknown")))
        return result

    def get_edges(self):
        edges = set()
        for syn in self.network.chemical_synapses.values():
            edges.add((syn.presynaptic_neuron_id, syn.postsynaptic_neuron_id))
        for gap in self.network.gap_junctions.values():
            a, b = gap.presynaptic_neuron_id, gap.postsynaptic_neuron_id
            edges.add((min(a, b), max(a, b)))
        return list(edges)

    def get_region_labels(self) -> List[dict]:
        """Return region label data for the anatomical layout."""
        from sdmn.visualization.layouts import _HEAD_NEURONS, _NERVE_RING, _TAIL_NEURONS
        labels = []
        groups = [
            ("Head\nSensory", _HEAD_NEURONS, 0.14, 0.12),
            ("Nerve Ring\nInterneurons", _NERVE_RING, 0.38, 0.12),
            ("Ventral Cord\nMotor", set(), 0.71, 0.12),
            ("Tail", _TAIL_NEURONS, 0.90, 0.12),
        ]
        for name, _, x, y in groups:
            labels.append({"name": name, "x": x, "y": y})
        return labels

    # ------------------------------------------------------------------
    # DMN baseline activity
    # ------------------------------------------------------------------

    def _apply_dmn_to_arrays(self, arrays: dict, sim_time_ms: float) -> None:
        """Apply DMN baseline activity into the I_ext array."""
        if not self.enable_dmn:
            return
        if self.dmn_mode == "recurrent":
            self._apply_dmn_recurrent(arrays, sim_time_ms)
        else:
            self._apply_dmn_oscillator(arrays, sim_time_ms)

    def _apply_dmn_oscillator(self, arrays: dict, sim_time_ms: float) -> None:
        for nid, phase in self._dmn_phases.items():
            idx = arrays["id_to_idx"].get(nid)
            if idx is not None:
                arrays["I_ext"][idx] += self.dmn_amplitude * (
                    math.sin(sim_time_ms * self.dmn_frequency * 2 * math.pi + phase)
                    + self._dmn_rng.normal(0, 0.4)
                )

    def _apply_dmn_recurrent(self, arrays: dict, sim_time_ms: float) -> None:
        """
        Self-sustaining DMN via tonic drive + slow adaptation feedback.
        Produces emergent oscillation without an external clock.
        """
        dt = 0.1
        for nid, phase in self._dmn_phases.items():
            idx = arrays["id_to_idx"].get(nid)
            if idx is None:
                continue
            v = float(arrays["V"][idx])
            activation = max(0.0, min(1.0, (v - (-65.0)) / 25.0))
            adapt = self._dmn_adaptation.get(nid, 0.0)
            adapt += (activation * 2.0 - adapt / self._dmn_recurrent_tau) * dt
            self._dmn_adaptation[nid] = max(0.0, adapt)
            tonic = self.dmn_amplitude * 0.3
            noise = self._dmn_rng.normal(0, self.dmn_amplitude * 0.15)
            drive = tonic + noise - adapt * 8.0
            arrays["I_ext"][idx] += drive

    # ------------------------------------------------------------------
    # Simulation thread
    # ------------------------------------------------------------------

    def _sim_thread(self, duration_ms: float, dt: float) -> None:
        arrays = _prepare_arrays(self.network)
        self._sim_arrays = arrays

        target_fps = 60.0
        sim_time = 0.0
        step_i = 0
        active_stimuli: Dict[int, Tuple[float, float]] = {}

        try:
            while not self._stop.is_set():
                if self._restart_flag.is_set():
                    self._restart_flag.clear()
                    arrays["V"][:] = arrays["E_leak"]
                    arrays["I_ext"][:] = 0.0
                    arrays["syn_weight"][:] = arrays["syn_weight_initial"]
                    arrays["syn_efficacy"][:] = 1.0
                    arrays["trace_pre"][:] = 0.0
                    arrays["trace_post"][:] = 0.0
                    self._dmn_adaptation = {nid: 0.0 for nid in self._dmn_phases}
                    active_stimuli.clear()
                    sim_time = 0.0
                    step_i = 0
                    if self.environment:
                        self.environment.__init__(
                            self.environment.arena_w, self.environment.arena_h)

                if not self._paused.is_set():
                    self._single_step.wait(timeout=0.5)
                    if not self._single_step.is_set() and not self._paused.is_set():
                        continue
                    self._single_step.clear()

                wall_start = time.perf_counter()

                desired_sim_ms = (1.0 / target_fps) * self._speed * 1000.0

                if self.engine == "object":
                    batch = max(1, min(int(desired_sim_ms / dt), 50))
                    self._run_object_batch(batch, dt, sim_time, active_stimuli)
                    sim_time += batch * dt
                    step_i += batch
                    frame = self.network._build_frame()
                    frame.time_ms = sim_time
                    frame.step = step_i
                    self._on_frame(frame)
                else:
                    batch = max(5, int(desired_sim_ms / dt))
                    arrays["I_ext"][:] = 0.0
                    self._apply_dmn_to_arrays(arrays, sim_time)
                    self._apply_stimuli_to_arrays(arrays, active_stimuli, sim_time)
                    self._apply_environment_to_arrays(arrays, dt, batch)

                    for _ in range(batch):
                        _fast_step(arrays, dt)

                    sim_time += batch * dt
                    step_i += batch
                    frame = build_frame_from_arrays(arrays, sim_time, step_i)
                    self._on_frame(frame)

                wall_elapsed = time.perf_counter() - wall_start
                target_wall = (batch * dt / 1000.0) / max(self._speed, 0.001)
                remaining = target_wall - wall_elapsed
                if remaining > 0.005:
                    time.sleep(remaining)
        finally:
            pass

    def _apply_stimuli_to_arrays(self, arrays, active_stimuli, sim_time):
        with self._stimuli_lock:
            for nid, cur in self._pending_stimuli.items():
                idx = arrays["id_to_idx"].get(nid)
                if idx is not None:
                    active_stimuli[idx] = (cur, sim_time)
            self._pending_stimuli.clear()

        expired = [idx for idx, (_, t0) in active_stimuli.items()
                   if sim_time - t0 > 50.0]
        for idx in expired:
            del active_stimuli[idx]
        for idx, (cur, _) in active_stimuli.items():
            arrays["I_ext"][idx] += cur

    def _apply_environment_to_arrays(self, arrays, dt, batch):
        if self.environment:
            voltages = {nid: float(arrays["V"][i])
                        for i, nid in enumerate(arrays["ids"])}
            self.environment.step(voltages, dt * batch)
            for nid, cur in self.environment.get_sensory_currents().items():
                idx = arrays["id_to_idx"].get(nid)
                if idx is not None and cur > 0:
                    arrays["I_ext"][idx] += cur

    def _run_object_batch(self, batch, dt, sim_time, active_stimuli):
        """Run batch steps using the original object-based engine."""
        with self._stimuli_lock:
            for nid, cur in self._pending_stimuli.items():
                if nid in self.network.neurons:
                    self.network.neurons[nid].set_external_current(cur)
            self._pending_stimuli.clear()

        if self.enable_dmn:
            for nid, phase in self._dmn_phases.items():
                if nid in self.network.neurons:
                    current = self.dmn_amplitude * (
                        math.sin(sim_time * self.dmn_frequency * 2 * math.pi + phase)
                        + self._dmn_rng.normal(0, 0.4)
                    )
                    self.network.neurons[nid].set_external_current(current)

        for _ in range(batch):
            self.network._step(dt)

    def restart(self) -> None:
        """Signal the sim thread to reset voltages and timeline."""
        self._restart_flag.set()
        with self._lock:
            self._latest_frame = None
        self._timeline_times.clear()
        self._timeline_sensory.clear()
        self._timeline_inter.clear()
        self._timeline_motor.clear()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, duration_ms: float = 30000, speed: float = 0.5,
            backend: str = "browser", **backend_kwargs) -> None:
        self.set_speed(speed)
        self._stop.clear()
        self._paused.set()

        dt = self.network.config.dt

        sim = threading.Thread(target=self._sim_thread,
                               args=(duration_ms, dt), daemon=True)
        sim.start()

        if backend == "matplotlib":
            from sdmn.visualization.matplotlib_backend import MatplotlibVisualizer
            renderer = MatplotlibVisualizer(self)
            renderer.show(**backend_kwargs)
        else:
            from sdmn.visualization.browser_backend import BrowserVisualizer
            renderer = BrowserVisualizer(self, **backend_kwargs)
            renderer.serve()

        self.stop()
        sim.join(timeout=5)
