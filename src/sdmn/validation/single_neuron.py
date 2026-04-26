"""
Single-neuron validation against published electrophysiology.

Compares simulated neuron responses to published patch-clamp data from
identified C. elegans neurons, computing quantitative fit metrics.

References:
    Goodman et al. (1998) Neuron 20(4):763-772
    Liu et al. (2018) Cell 175(1):57-70
    Nicoletti et al. (2019) PLOS ONE 14(7):e0218738
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class SingleNeuronValidator:
    """Compare simulated neuron traces against published electrophysiology."""

    def __init__(self, reference_data_path: Optional[str] = None):
        self.reference_data_path = Path(reference_data_path) if reference_data_path else (
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "validation" / "electrophysiology"
        )
        self.published_properties: Dict[str, Dict] = {}
        self._load_published_properties()

    def _load_published_properties(self) -> None:
        prop_file = self.reference_data_path / "published_neuron_properties.csv"
        if not prop_file.exists():
            return
        with open(prop_file, newline='') as f:
            for row in csv.DictReader(f):
                ntype = row['neuron_type']
                if ntype not in self.published_properties:
                    self.published_properties[ntype] = {}
                self.published_properties[ntype][row['property']] = {
                    'value': row['value'],
                    'unit': row['unit'],
                    'reference': row['reference'],
                }

    def validate_resting_potential(self, neuron, neuron_type: str,
                                    settle_ms: float = 100.0) -> Dict[str, float]:
        """
        Measure resting potential and compare to published values.

        Args:
            neuron: A GradedNeuron or MultiChannelNeuron instance
            neuron_type: Key into published_properties (e.g. 'AWCon', 'RMD')
            settle_ms: Time to simulate before measuring (ms)

        Returns:
            Dictionary with simulated V_rest and comparison metrics
        """
        dt = neuron.parameters.dt if hasattr(neuron.parameters, 'dt') else 0.01
        steps = int(settle_ms / dt)
        for _ in range(steps):
            neuron.step(dt)

        V_rest_sim = neuron.state.membrane_potential
        result = {'V_rest_sim_mV': V_rest_sim}

        if neuron_type in self.published_properties:
            props = self.published_properties[neuron_type]
            if 'V_rest' in props:
                val_str = props['V_rest']['value']
                if ' to ' in val_str:
                    lo, hi = [float(x) for x in val_str.split(' to ')]
                    result['V_rest_published_lo'] = lo
                    result['V_rest_published_hi'] = hi
                    result['within_range'] = lo <= V_rest_sim <= hi
                else:
                    ref_val = float(val_str)
                    result['V_rest_published'] = ref_val
                    result['error_mV'] = V_rest_sim - ref_val

        return result

    def validate_step_response(self, neuron, I_inject_pA: float,
                                duration_ms: float,
                                reference_trace: Optional[np.ndarray] = None,
                                ) -> Dict[str, float]:
        """
        Inject step current and measure voltage response.

        Args:
            neuron: Neuron instance
            I_inject_pA: Step current amplitude (pA)
            duration_ms: Duration of current injection (ms)
            reference_trace: Optional published trace for comparison

        Returns:
            Response metrics: peak_depol, steady_state, time_to_peak, etc.
        """
        dt = neuron.parameters.dt if hasattr(neuron.parameters, 'dt') else 0.01
        neuron.reset_neuron()

        # Baseline (50 ms)
        baseline_steps = int(50.0 / dt)
        for _ in range(baseline_steps):
            neuron.step(dt)
        V_baseline = neuron.state.membrane_potential

        # Inject current
        inject_steps = int(duration_ms / dt)
        trace = np.zeros(inject_steps)
        for i in range(inject_steps):
            neuron.inject_current(I_inject_pA)
            neuron.step(dt)
            trace[i] = neuron.state.membrane_potential

        # Recovery (50 ms)
        recovery_steps = int(50.0 / dt)
        recovery_trace = np.zeros(recovery_steps)
        for i in range(recovery_steps):
            neuron.step(dt)
            recovery_trace[i] = neuron.state.membrane_potential

        peak_depol = np.max(trace) if I_inject_pA > 0 else np.min(trace)
        steady_state = np.mean(trace[-int(10.0 / dt):])
        time_to_peak_ms = np.argmax(trace) * dt if I_inject_pA > 0 else np.argmin(trace) * dt

        result = {
            'V_baseline_mV': V_baseline,
            'peak_depolarization_mV': peak_depol,
            'steady_state_mV': steady_state,
            'delta_V_peak_mV': peak_depol - V_baseline,
            'delta_V_steady_mV': steady_state - V_baseline,
            'time_to_peak_ms': time_to_peak_ms,
            'V_recovery_mV': recovery_trace[-1],
        }

        if reference_trace is not None:
            min_len = min(len(trace), len(reference_trace))
            rmse = np.sqrt(np.mean((trace[:min_len] - reference_trace[:min_len]) ** 2))
            corr = np.corrcoef(trace[:min_len], reference_trace[:min_len])[0, 1]
            result['rmse_mV'] = rmse
            result['pearson_r'] = corr

        return result

    def validate_passive_properties(self, neuron) -> Dict[str, float]:
        """
        Extract passive membrane properties via small current steps.

        Returns:
            R_input (GOhm), tau_m (ms), C_m (pF)
        """
        dt = neuron.parameters.dt if hasattr(neuron.parameters, 'dt') else 0.01
        neuron.reset_neuron()

        # Settle
        for _ in range(int(200.0 / dt)):
            neuron.step(dt)
        V_rest = neuron.state.membrane_potential

        # Small hyperpolarizing step (-1 pA)
        I_test = -1.0
        steps = int(200.0 / dt)
        trace = np.zeros(steps)
        for i in range(steps):
            neuron.inject_current(I_test)
            neuron.step(dt)
            trace[i] = neuron.state.membrane_potential

        V_ss = np.mean(trace[-int(50.0 / dt):])
        delta_V = V_ss - V_rest
        R_input = abs(delta_V / I_test)  # mV/pA = GOhm

        # Estimate tau_m from exponential fit
        delta_trace = trace - V_rest
        target_63 = 0.632 * delta_V
        crossings = np.where(np.abs(delta_trace) >= abs(target_63))[0]
        tau_m = crossings[0] * dt if len(crossings) > 0 else float('nan')

        C_m = tau_m / R_input if R_input > 0 and not np.isnan(tau_m) else float('nan')

        return {
            'R_input_GOhm': R_input,
            'tau_m_ms': tau_m,
            'C_m_estimated_pF': C_m,
            'V_rest_mV': V_rest,
        }

    def generate_iv_curve(self, neuron, current_range: np.ndarray,
                           hold_ms: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate I-V curve by injecting a range of currents.

        Returns:
            (currents_pA, steady_state_voltages_mV)
        """
        dt = neuron.parameters.dt if hasattr(neuron.parameters, 'dt') else 0.01
        voltages = np.zeros_like(current_range)

        for j, I_inj in enumerate(current_range):
            neuron.reset_neuron()
            for _ in range(int(50.0 / dt)):
                neuron.step(dt)

            steps = int(hold_ms / dt)
            for _ in range(steps):
                neuron.inject_current(I_inj)
                neuron.step(dt)

            voltages[j] = neuron.state.membrane_potential

        return current_range, voltages
