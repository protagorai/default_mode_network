"""
Simple 2D C. elegans body model and arena environment.

The worm is a chain of segments whose curvature is driven by dorsal/ventral
motor neuron output.  A chemical gradient field provides sensory input back
to the network.  The ``WormEnvironment`` advances one physics step per
simulation step and exposes its state for rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ChemicalSource:
    """A point source of chemical attractant in the arena."""
    x: float
    y: float
    strength: float = 1.0
    radius: float = 80.0


@dataclass
class WormState:
    """Observable state of the worm body for rendering."""
    segments: List[Tuple[float, float]]  # (x, y) per segment
    heading: float                        # radians
    speed: float
    dorsal_activation: List[float]        # per-segment [0,1]
    ventral_activation: List[float]       # per-segment [0,1]
    chemical_concentration: float         # at head


N_SEGMENTS = 12
SEGMENT_LENGTH = 6.0
ARENA_W = 500.0
ARENA_H = 400.0

_DORSAL_MOTORS = ["DA01", "DA02", "DA03", "DA04", "DA05", "DA06",
                  "DA07", "DA08", "DA09", "DB01", "DB02", "DB03",
                  "DB04", "DB05", "DB06", "DB07"]
_VENTRAL_MOTORS = ["VA01", "VA02", "VA03", "VA04", "VA05", "VA06",
                   "VA07", "VA08", "VA09", "VA10", "VA11", "VA12",
                   "VB01", "VB02", "VB03", "VB04", "VB05", "VB06",
                   "VB07", "VB08", "VB09", "VB10", "VB11"]
_HEAD_SENSORY = ["AWCL", "AWCR", "ASEL", "ASER", "ASHL", "ASHR",
                 "AWAL", "AWAR", "AWBL", "AWBR", "ADFL", "ADFR"]

V_REST = -65.0
V_RANGE = 45.0  # -65 to -20


def _activation(voltage: float) -> float:
    return max(0.0, min(1.0, (voltage - V_REST) / V_RANGE))


class WormEnvironment:
    """
    2D arena with a simplified worm body driven by motor neuron voltages.

    Call ``step(voltages, dt)`` each simulation step.  Read ``get_state()``
    for rendering.  Call ``get_sensory_currents()`` to feed chemical
    gradient information back into sensory neurons.
    """

    def __init__(self, arena_w: float = ARENA_W, arena_h: float = ARENA_H):
        self.arena_w = arena_w
        self.arena_h = arena_h
        self.chemical_sources: List[ChemicalSource] = [
            ChemicalSource(arena_w * 0.75, arena_h * 0.5, strength=1.0, radius=120.0),
        ]

        self._heading = 0.0
        self._x = arena_w * 0.4
        self._y = arena_h * 0.45
        self._phase = 0.0  # undulation phase
        self._segments: List[Tuple[float, float]] = []
        self._dorsal_act = [0.0] * N_SEGMENTS
        self._ventral_act = [0.0] * N_SEGMENTS
        self._speed = 0.0
        self._chem_at_head = 0.0
        self._rebuild_segments()

    def _rebuild_segments(self) -> None:
        self._segments = []
        x, y = self._x, self._y
        for i in range(N_SEGMENTS):
            self._segments.append((x, y))
            angle = self._heading + math.pi + math.sin(self._phase + i * 0.8) * 0.3
            x += math.cos(angle) * SEGMENT_LENGTH
            y += math.sin(angle) * SEGMENT_LENGTH

    def step(self, voltages: Dict[str, float], dt_ms: float) -> None:
        """Advance the worm body by one simulation step."""
        dorsal_v = [_activation(voltages.get(n, V_REST)) for n in _DORSAL_MOTORS]
        ventral_v = [_activation(voltages.get(n, V_REST)) for n in _VENTRAL_MOTORS]

        n_d = min(N_SEGMENTS, len(dorsal_v))
        n_v = min(N_SEGMENTS, len(ventral_v))

        for i in range(N_SEGMENTS):
            di = int(i * len(dorsal_v) / N_SEGMENTS) if n_d else 0
            vi = int(i * len(ventral_v) / N_SEGMENTS) if n_v else 0
            self._dorsal_act[i] = dorsal_v[di] if di < len(dorsal_v) else 0
            self._ventral_act[i] = ventral_v[vi] if vi < len(ventral_v) else 0

        mean_d = sum(self._dorsal_act) / N_SEGMENTS
        mean_v = sum(self._ventral_act) / N_SEGMENTS

        neural_drive = (mean_d + mean_v) * 0.5
        neural_turn = (mean_d - mean_v) * 3.0

        # Gradient-based chemotaxis: turn toward strongest chemical source
        chem_turn = 0.0
        hx, hy = self._x, self._y
        for src in self.chemical_sources:
            dx = src.x - hx
            dy = src.y - hy
            dist = math.hypot(dx, dy)
            if dist < src.radius and dist > 1:
                angle_to_src = math.atan2(dy, dx)
                angle_diff = angle_to_src - self._heading
                while angle_diff > math.pi: angle_diff -= 2 * math.pi
                while angle_diff < -math.pi: angle_diff += 2 * math.pi
                strength = src.strength * (1.0 - dist / src.radius)
                chem_turn += angle_diff * strength * 0.5

        dt_s = dt_ms / 1000.0
        base_speed = 15.0
        self._speed = base_speed + neural_drive * 30.0
        self._heading += (neural_turn + chem_turn) * dt_s * 2.0
        self._phase += (0.5 + neural_drive * 4.0) * dt_s * 2 * math.pi

        self._x += math.cos(self._heading) * self._speed * dt_s
        self._y += math.sin(self._heading) * self._speed * dt_s

        self._x = max(10, min(self.arena_w - 10, self._x))
        self._y = max(10, min(self.arena_h - 10, self._y))

        self._rebuild_segments()

        self._chem_at_head = 0.0
        hx, hy = self._segments[0]
        for src in self.chemical_sources:
            dist = math.hypot(hx - src.x, hy - src.y)
            self._chem_at_head += src.strength * max(0, 1.0 - dist / src.radius)

    def get_sensory_currents(self) -> Dict[str, float]:
        """Return current (pA) to inject into sensory neurons based on chemical field."""
        c = self._chem_at_head
        currents: Dict[str, float] = {}
        for nid in _HEAD_SENSORY:
            currents[nid] = c * 40.0
        return currents

    def get_state(self) -> WormState:
        return WormState(
            segments=list(self._segments),
            heading=self._heading,
            speed=self._speed,
            dorsal_activation=list(self._dorsal_act),
            ventral_activation=list(self._ventral_act),
            chemical_concentration=self._chem_at_head,
        )

    def get_chemical_field(self, resolution: int = 20) -> List[List[float]]:
        """Sample the chemical field on a grid for rendering."""
        field = []
        for iy in range(resolution):
            row = []
            y = (iy + 0.5) / resolution * self.arena_h
            for ix in range(resolution):
                x = (ix + 0.5) / resolution * self.arena_w
                val = 0.0
                for src in self.chemical_sources:
                    dist = math.hypot(x - src.x, y - src.y)
                    val += src.strength * max(0, 1.0 - dist / src.radius)
                row.append(round(min(1.0, val), 3))
            field.append(row)
        return field

    def add_chemical_source(self, x: float, y: float,
                            strength: float = 1.0, radius: float = 80.0) -> None:
        self.chemical_sources.append(ChemicalSource(x, y, strength, radius))

    def clear_chemical_sources(self) -> None:
        self.chemical_sources.clear()

    def to_json(self) -> dict:
        """Serialise environment state for WebSocket."""
        st = self.get_state()
        return {
            "segments": [[round(x, 1), round(y, 1)] for x, y in st.segments],
            "heading": round(st.heading, 3),
            "speed": round(st.speed, 2),
            "dorsal": [round(v, 3) for v in st.dorsal_activation],
            "ventral": [round(v, 3) for v in st.ventral_activation],
            "chem": round(st.chemical_concentration, 3),
            "arena": [self.arena_w, self.arena_h],
            "sources": [[round(s.x, 1), round(s.y, 1), round(s.strength, 2), round(s.radius, 1)]
                        for s in self.chemical_sources],
        }
