"""
Behavioral detection for C. elegans neural network simulations.

Higher-level analysis that maps motor neuron activity patterns to
recognisable C. elegans behaviours (forward locomotion, reversals,
activity states, stimulus responses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sdmn.networks.celegans.network_manager import CElegansNetwork


_FORWARD_MOTOR_RE = re.compile(r"^(DB|VB)\d+$")
_BACKWARD_MOTOR_RE = re.compile(r"^(DA|VA)\d+$")
_DD_RE = re.compile(r"^DD\d+$")
_VD_RE = re.compile(r"^VD\d+$")


@dataclass
class LocomotionEvent:
    """A detected locomotion bout."""
    direction: str  # "forward" | "backward"
    start_ms: float
    end_ms: float
    mean_activation: float = 0.0


@dataclass
class ActivityState:
    """Classified activity segment."""
    state: str  # "quiescent" | "active" | "bursting"
    start_ms: float
    end_ms: float


class BehaviorDetector:
    """
    Detect C. elegans behaviours from motor-neuron activity.

    Usage::

        detector = BehaviorDetector.from_simulation(network)
        events = detector.detect_forward_locomotion()
    """

    def __init__(self, times: np.ndarray, voltages: np.ndarray,
                 neuron_ids: List[str]):
        self.times = times
        self.voltages = voltages
        self.neuron_ids = neuron_ids
        self._id_index = {nid: i for i, nid in enumerate(neuron_ids)}

        self._fwd_ids = [n for n in neuron_ids if _FORWARD_MOTOR_RE.match(n)]
        self._bwd_ids = [n for n in neuron_ids if _BACKWARD_MOTOR_RE.match(n)]
        self._dd_ids = [n for n in neuron_ids if _DD_RE.match(n)]
        self._vd_ids = [n for n in neuron_ids if _VD_RE.match(n)]

    @classmethod
    def from_simulation(cls, network: CElegansNetwork) -> "BehaviorDetector":
        times, voltages = network.get_voltages_array()
        neuron_ids = sorted(network.voltage_history.keys())
        return cls(times, voltages, neuron_ids)

    @property
    def n_timepoints(self) -> int:
        return len(self.times)

    # ------------------------------------------------------------------
    # Locomotion
    # ------------------------------------------------------------------

    def _group_mean(self, ids: List[str]) -> np.ndarray:
        """Mean voltage of a neuron group over time."""
        if not ids:
            return np.zeros(len(self.times))
        indices = [self._id_index[n] for n in ids if n in self._id_index]
        if not indices:
            return np.zeros(len(self.times))
        return self.voltages[indices].mean(axis=0)

    def detect_forward_locomotion(self, threshold: float = -55.0,
                                  min_duration_ms: float = 20.0
                                  ) -> List[LocomotionEvent]:
        """
        Detect forward locomotion bouts from DB/VB motor neuron activation.

        Args:
            threshold:       Voltage above which a neuron is considered active.
            min_duration_ms: Minimum duration to count as a locomotion bout.
        """
        return self._detect_bouts(self._fwd_ids, "forward",
                                  threshold, min_duration_ms)

    def detect_reversals(self, threshold: float = -55.0,
                         min_duration_ms: float = 10.0
                         ) -> List[LocomotionEvent]:
        """Detect backward locomotion (reversal) bouts from DA/VA activation."""
        return self._detect_bouts(self._bwd_ids, "backward",
                                  threshold, min_duration_ms)

    def _detect_bouts(self, ids: List[str], direction: str,
                      threshold: float, min_dur: float
                      ) -> List[LocomotionEvent]:
        mean_v = self._group_mean(ids)
        active = mean_v > threshold

        events: List[LocomotionEvent] = []
        in_bout = False
        start_idx = 0

        for i in range(len(active)):
            if active[i] and not in_bout:
                in_bout = True
                start_idx = i
            elif not active[i] and in_bout:
                in_bout = False
                start_t = float(self.times[start_idx])
                end_t = float(self.times[i])
                if (end_t - start_t) >= min_dur:
                    events.append(LocomotionEvent(
                        direction=direction,
                        start_ms=start_t,
                        end_ms=end_t,
                        mean_activation=float(mean_v[start_idx:i].mean()),
                    ))

        if in_bout:
            end_t = float(self.times[-1])
            start_t = float(self.times[start_idx])
            if (end_t - start_t) >= min_dur:
                events.append(LocomotionEvent(
                    direction=direction,
                    start_ms=start_t,
                    end_ms=end_t,
                    mean_activation=float(mean_v[start_idx:].mean()),
                ))

        return events

    # ------------------------------------------------------------------
    # Activity-state classification
    # ------------------------------------------------------------------

    def classify_activity_state(self, window_ms: float = 50.0
                                ) -> List[ActivityState]:
        """
        Classify each time window as quiescent, active, or bursting.

        Uses population voltage variance within sliding windows.
        """
        if len(self.times) < 2:
            return []

        dt = float(self.times[1] - self.times[0])
        win_samples = max(1, int(window_ms / dt))
        n = len(self.times)

        states: List[ActivityState] = []
        for start in range(0, n, win_samples):
            end = min(start + win_samples, n)
            chunk = self.voltages[:, start:end]
            var = float(chunk.var())
            mean_v = float(chunk.mean())

            if mean_v < -60.0 and var < 5.0:
                label = "quiescent"
            elif var > 30.0:
                label = "bursting"
            else:
                label = "active"

            states.append(ActivityState(
                state=label,
                start_ms=float(self.times[start]),
                end_ms=float(self.times[min(end, n) - 1]),
            ))

        return states

    # ------------------------------------------------------------------
    # Response latency
    # ------------------------------------------------------------------

    def stimulus_response_latency(self, stim_time_ms: float,
                                  target_neurons: List[str],
                                  threshold: float = -55.0) -> float:
        """
        Time from stimulus onset to first threshold crossing in targets.

        Returns latency in ms, or ``float('inf')`` if no response detected.
        """
        mean_v = self._group_mean(target_neurons)

        stim_idx = int(np.searchsorted(self.times, stim_time_ms))
        for i in range(stim_idx, len(self.times)):
            if mean_v[i] > threshold:
                return float(self.times[i] - stim_time_ms)
        return float("inf")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def behavioral_report(self) -> Dict[str, Any]:
        """Run all detectors and return a summary dict."""
        fwd = self.detect_forward_locomotion()
        rev = self.detect_reversals()
        states = self.classify_activity_state()

        state_counts = {}
        for s in states:
            state_counts[s.state] = state_counts.get(s.state, 0) + 1

        return {
            "n_forward_bouts": len(fwd),
            "n_reversal_bouts": len(rev),
            "total_forward_ms": sum(e.end_ms - e.start_ms for e in fwd),
            "total_reversal_ms": sum(e.end_ms - e.start_ms for e in rev),
            "activity_state_counts": state_counts,
            "forward_events": fwd,
            "reversal_events": rev,
        }
