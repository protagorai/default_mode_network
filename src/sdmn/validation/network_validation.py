"""
Network-level validation against functional connectome data.

Validates network dynamics by comparing simulated signal propagation to
the Randi et al. (2023) functional atlas measurements.

References:
    Randi F, et al. (2023) Nature 623:406-414
    Kato S, et al. (2015) Cell 163(3):656-669
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class NetworkValidator:
    """
    Validate network dynamics against published functional data.

    Supports:
    - Signal propagation: stimulate one neuron, measure responses elsewhere
    - Sign accuracy: does the model correctly predict excitation vs inhibition
    - Dimensionality analysis: compare spontaneous dynamics PCA structure to Kato et al.
    """

    def __init__(self):
        self.functional_atlas: Optional[Dict] = None

    def load_functional_atlas(self, atlas_data: Dict[str, Dict]) -> None:
        """
        Load Randi et al. functional atlas data.

        Expected format:
            {
                'source_neuron': {
                    'target_neuron': {
                        'sign': 'excitatory' | 'inhibitory' | 'none',
                        'amplitude': float,
                        'latency_ms': float,
                    }
                }
            }
        """
        self.functional_atlas = atlas_data

    def validate_signal_propagation(self, network, source_neuron: str,
                                     stim_amplitude_pA: float = 20.0,
                                     stim_duration_ms: float = 500.0,
                                     dt: float = 0.1,
                                     ) -> Dict[str, Dict]:
        """
        Stimulate a single neuron and record responses across the network.

        Args:
            network: CElegansNetwork instance
            source_neuron: Name of neuron to stimulate
            stim_amplitude_pA: Current injection amplitude
            stim_duration_ms: Duration of stimulation
            dt: Time step

        Returns:
            {neuron_name: {'delta_V': float, 'sign': str, 'peak_time_ms': float}}
        """
        network.reset()

        # Baseline: 200 ms
        baseline_steps = int(200.0 / dt)
        baseline_voltages: Dict[str, float] = {}
        for _ in range(baseline_steps):
            network.step(dt)
        for name, neuron in network.neurons.items():
            baseline_voltages[name] = neuron.state.membrane_potential

        # Stimulate source neuron
        stim_steps = int(stim_duration_ms / dt)
        peak_responses: Dict[str, float] = {n: 0.0 for n in network.neurons}
        peak_times: Dict[str, float] = {n: 0.0 for n in network.neurons}

        for step_i in range(stim_steps):
            if source_neuron in network.neurons:
                network.neurons[source_neuron].inject_current(stim_amplitude_pA)
            network.step(dt)

            t = step_i * dt
            for name, neuron in network.neurons.items():
                delta = neuron.state.membrane_potential - baseline_voltages[name]
                if abs(delta) > abs(peak_responses[name]):
                    peak_responses[name] = delta
                    peak_times[name] = t

        results = {}
        for name in network.neurons:
            if name == source_neuron:
                continue
            delta = peak_responses[name]
            results[name] = {
                'delta_V_mV': delta,
                'sign': 'excitatory' if delta > 0.5 else ('inhibitory' if delta < -0.5 else 'none'),
                'peak_time_ms': peak_times[name],
                'baseline_mV': baseline_voltages[name],
            }

        return results

    def compute_sign_accuracy(self, simulated: Dict[str, Dict],
                               source_neuron: str) -> Dict[str, float]:
        """
        Compare simulated response signs to functional atlas.

        Returns:
            {'accuracy': float, 'n_compared': int, 'true_positive': int, ...}
        """
        if self.functional_atlas is None or source_neuron not in self.functional_atlas:
            return {'error': 'No atlas data for this source'}

        atlas_for_source = self.functional_atlas[source_neuron]
        correct = 0
        total = 0

        for target, atlas_data in atlas_for_source.items():
            if target not in simulated:
                continue
            if atlas_data['sign'] == 'none':
                continue

            total += 1
            if simulated[target]['sign'] == atlas_data['sign']:
                correct += 1

        accuracy = correct / total if total > 0 else float('nan')
        return {
            'sign_accuracy': accuracy,
            'n_compared': total,
            'n_correct': correct,
        }

    def compute_amplitude_correlation(self, simulated: Dict[str, Dict],
                                       source_neuron: str) -> Dict[str, float]:
        """Spearman correlation of response amplitudes vs atlas."""
        if self.functional_atlas is None or source_neuron not in self.functional_atlas:
            return {'error': 'No atlas data'}

        atlas = self.functional_atlas[source_neuron]
        sim_amps = []
        atlas_amps = []

        for target in atlas:
            if target not in simulated:
                continue
            sim_amps.append(abs(simulated[target]['delta_V_mV']))
            atlas_amps.append(abs(atlas[target].get('amplitude', 0)))

        if len(sim_amps) < 3:
            return {'error': 'Too few data points'}

        from scipy.stats import spearmanr
        rho, pval = spearmanr(sim_amps, atlas_amps)
        return {
            'spearman_rho': rho,
            'p_value': pval,
            'n_pairs': len(sim_amps),
        }

    def analyze_spontaneous_dynamics(self, network, duration_ms: float = 10000.0,
                                      dt: float = 0.1,
                                      sample_interval_ms: float = 10.0,
                                      ) -> Dict[str, any]:
        """
        Analyze spontaneous network dynamics and compare to Kato et al. (2015).

        Kato found that C. elegans brain dynamics are low-dimensional (~3 PCs)
        with cyclic trajectories mapping to motor command sequences.

        Returns:
            PCA variance explained, effective dimensionality, dominant frequencies
        """
        network.reset()
        sample_steps = int(sample_interval_ms / dt)
        total_steps = int(duration_ms / dt)
        n_samples = total_steps // sample_steps
        n_neurons = len(network.neurons)
        neuron_names = sorted(network.neurons.keys())

        activity_matrix = np.zeros((n_samples, n_neurons))

        step_count = 0
        sample_idx = 0
        for _ in range(total_steps):
            network.step(dt)
            step_count += 1
            if step_count >= sample_steps:
                for j, name in enumerate(neuron_names):
                    activity_matrix[sample_idx, j] = network.neurons[name].state.membrane_potential
                sample_idx += 1
                step_count = 0
                if sample_idx >= n_samples:
                    break

        activity_matrix = activity_matrix[:sample_idx]

        # Z-score normalize
        mean_act = np.mean(activity_matrix, axis=0)
        std_act = np.std(activity_matrix, axis=0)
        std_act[std_act < 1e-10] = 1.0
        z_activity = (activity_matrix - mean_act) / std_act

        # PCA
        cov = np.cov(z_activity.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        total_var = np.sum(eigenvalues)
        var_explained = eigenvalues / total_var if total_var > 0 else eigenvalues

        # Effective dimensionality (participation ratio)
        if total_var > 0:
            eff_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        else:
            eff_dim = 0.0

        # Top-3 PCs variance (Kato target: ~50-70%)
        top3_var = np.sum(var_explained[:3]) if len(var_explained) >= 3 else 0.0

        return {
            'effective_dimensionality': eff_dim,
            'top3_variance_explained': top3_var,
            'variance_explained_curve': var_explained[:20].tolist(),
            'n_samples': sample_idx,
            'n_neurons': n_neurons,
            'kato_comparison': {
                'target_top3_variance': '0.50-0.70 (Kato et al. 2015)',
                'target_effective_dim': '3-8 (Kato et al. 2015)',
                'matches': 0.5 <= top3_var <= 0.7 and 3 <= eff_dim <= 8,
            }
        }
