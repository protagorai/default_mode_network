"""
Dynamical analysis tools for C. elegans neural network simulations.

Operates on voltage trace arrays recorded by ``CElegansNetwork`` and
provides signal propagation, synchronization, spectral, PCA, and
information-theoretic analyses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from sdmn.networks.celegans.network_manager import CElegansNetwork


class DynamicsAnalyzer:
    """
    Analyze simulation dynamics from recorded voltage traces.

    Usage::

        analyzer = DynamicsAnalyzer.from_simulation(network)
        corr = analyzer.pairwise_correlation_matrix()
    """

    def __init__(self, times: np.ndarray, voltages: np.ndarray,
                 neuron_ids: List[str]):
        """
        Args:
            times:      1-D array of time points (ms).
            voltages:   2-D array of shape ``(n_neurons, n_timepoints)``.
            neuron_ids: Ordered neuron identifiers matching rows of *voltages*.
        """
        self.times = times
        self.voltages = voltages
        self.neuron_ids = neuron_ids
        self._id_index = {nid: i for i, nid in enumerate(neuron_ids)}

    @classmethod
    def from_simulation(cls, network: CElegansNetwork) -> "DynamicsAnalyzer":
        """Extract recorded traces from a simulated ``CElegansNetwork``."""
        times, voltages = network.get_voltages_array()
        neuron_ids = sorted(network.voltage_history.keys())
        return cls(times, voltages, neuron_ids)

    @property
    def dt(self) -> float:
        if len(self.times) < 2:
            return 1.0
        return float(self.times[1] - self.times[0])

    @property
    def n_neurons(self) -> int:
        return len(self.neuron_ids)

    @property
    def n_timepoints(self) -> int:
        return len(self.times)

    # ------------------------------------------------------------------
    # Signal propagation
    # ------------------------------------------------------------------

    def propagation_delay(self, source_id: str, target_id: str) -> float:
        """
        Estimate propagation delay via cross-correlation peak lag.

        Returns delay in ms (positive means target lags source).
        """
        si = self._id_index[source_id]
        ti = self._id_index[target_id]
        s = self.voltages[si] - self.voltages[si].mean()
        t = self.voltages[ti] - self.voltages[ti].mean()

        corr = np.correlate(t, s, mode="full")
        lags = np.arange(-len(s) + 1, len(s)) * self.dt
        peak_idx = int(np.argmax(np.abs(corr)))
        return float(lags[peak_idx])

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    def pairwise_correlation_matrix(self) -> np.ndarray:
        """Pearson correlation matrix across all neuron pairs."""
        return np.corrcoef(self.voltages)

    def mean_synchronization(self) -> float:
        """Mean absolute pairwise Pearson correlation (excluding diagonal)."""
        corr = self.pairwise_correlation_matrix()
        n = corr.shape[0]
        if n < 2:
            return 0.0
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(np.abs(corr[mask])))

    def phase_locking_value(self, id_a: str, id_b: str) -> float:
        """
        Phase-locking value between two neurons using Hilbert transform.

        Returns a value in [0, 1] where 1 = perfect phase locking.
        """
        ai = self._id_index[id_a]
        bi = self._id_index[id_b]
        analytic_a = sp_signal.hilbert(self.voltages[ai])
        analytic_b = sp_signal.hilbert(self.voltages[bi])
        phase_diff = np.angle(analytic_a) - np.angle(analytic_b)
        return float(np.abs(np.mean(np.exp(1j * phase_diff))))

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def power_spectrum(self, neuron_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum (Welch) for a single neuron.

        Returns ``(frequencies_hz, power)``.
        """
        idx = self._id_index[neuron_id]
        fs = 1000.0 / self.dt  # sampling rate in Hz (dt is in ms)
        nperseg = min(256, self.n_timepoints)
        freqs, psd = sp_signal.welch(self.voltages[idx], fs=fs, nperseg=nperseg)
        return freqs, psd

    def dominant_frequency(self, neuron_id: str) -> float:
        """Peak frequency (Hz) for a neuron."""
        freqs, psd = self.power_spectrum(neuron_id)
        return float(freqs[np.argmax(psd)])

    def population_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean power spectrum across all neurons."""
        fs = 1000.0 / self.dt
        nperseg = min(256, self.n_timepoints)
        all_psd = []
        for i in range(self.n_neurons):
            freqs, psd = sp_signal.welch(self.voltages[i], fs=fs, nperseg=nperseg)
            all_psd.append(psd)
        return freqs, np.mean(all_psd, axis=0)

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------

    def pca_trajectories(self, n_components: int = 3
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PCA on the voltage matrix (neurons x time).

        Returns ``(trajectories, explained_variance_ratio)`` where
        *trajectories* has shape ``(n_components, n_timepoints)``.
        """
        centered = self.voltages - self.voltages.mean(axis=1, keepdims=True)
        cov = np.cov(centered)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        total = eigvals.sum()
        explained = eigvals[:n_components] / total if total > 0 else eigvals[:n_components]

        components = eigvecs[:, :n_components].T
        trajectories = components @ centered
        return trajectories, explained

    def explained_variance(self, n_components: int = 10) -> np.ndarray:
        """Explained variance ratio for leading PCA components."""
        _, ev = self.pca_trajectories(n_components)
        return ev

    # ------------------------------------------------------------------
    # Information theory (simplified estimators)
    # ------------------------------------------------------------------

    def transfer_entropy(self, source_id: str, target_id: str,
                         lag: int = 1, n_bins: int = 8) -> float:
        """
        Binned transfer-entropy estimate from source to target.

        This is a simplified histogram-based estimator suitable for quick
        comparisons, not a publication-grade measure.
        """
        si = self._id_index[source_id]
        ti = self._id_index[target_id]
        s = self.voltages[si]
        t = self.voltages[ti]

        t_past = t[:-lag]
        t_future = t[lag:]
        s_past = s[:-lag]

        def _bin(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.zeros(len(arr), dtype=int)
            return np.clip(((arr - mn) / (mx - mn) * (n_bins - 1)).astype(int),
                           0, n_bins - 1)

        tb = _bin(t_past)
        tf = _bin(t_future)
        sb = _bin(s_past)

        n = len(tf)
        eps = 1e-12

        joint_tfs = np.zeros((n_bins, n_bins, n_bins))
        joint_tf = np.zeros((n_bins, n_bins))
        for i in range(n):
            joint_tfs[tf[i], tb[i], sb[i]] += 1
            joint_tf[tf[i], tb[i]] += 1

        joint_tfs /= n + eps
        joint_tf /= n + eps
        p_tb = joint_tf.sum(axis=0)
        p_tbs = joint_tfs.sum(axis=0)

        te = 0.0
        for fi in range(n_bins):
            for ti_ in range(n_bins):
                for si_ in range(n_bins):
                    p_f_ts = joint_tfs[fi, ti_, si_]
                    if p_f_ts < eps:
                        continue
                    p_f_t = joint_tf[fi, ti_]
                    p_ts = p_tbs[ti_, si_]
                    p_t = p_tb[ti_]
                    if p_f_t < eps or p_ts < eps or p_t < eps:
                        continue
                    te += p_f_ts * np.log2(
                        (p_f_ts * p_t) / (p_f_t * p_ts + eps) + eps
                    )
        return max(0.0, te)

    def mutual_information(self, id_a: str, id_b: str,
                           n_bins: int = 8) -> float:
        """Binned mutual-information estimate between two neurons."""
        ai = self._id_index[id_a]
        bi = self._id_index[id_b]
        a = self.voltages[ai]
        b = self.voltages[bi]

        def _bin(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.zeros(len(arr), dtype=int)
            return np.clip(((arr - mn) / (mx - mn) * (n_bins - 1)).astype(int),
                           0, n_bins - 1)

        ab = _bin(a)
        bb = _bin(b)
        n = len(ab)
        eps = 1e-12

        joint = np.zeros((n_bins, n_bins))
        for i in range(n):
            joint[ab[i], bb[i]] += 1
        joint /= n + eps

        pa = joint.sum(axis=1)
        pb = joint.sum(axis=0)

        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] < eps or pa[i] < eps or pb[j] < eps:
                    continue
                mi += joint[i, j] * np.log2(joint[i, j] / (pa[i] * pb[j] + eps) + eps)
        return max(0.0, mi)

    # ------------------------------------------------------------------
    # Activity statistics
    # ------------------------------------------------------------------

    def mean_voltage_trace(self) -> np.ndarray:
        """Population-average voltage at each time point."""
        return self.voltages.mean(axis=0)

    def voltage_variance_over_time(self) -> np.ndarray:
        """Variance across neurons at each time point."""
        return self.voltages.var(axis=0)

    def active_neuron_count(self, threshold: float = -50.0) -> np.ndarray:
        """Number of neurons above *threshold* (mV) at each time step."""
        return (self.voltages > threshold).sum(axis=0)
