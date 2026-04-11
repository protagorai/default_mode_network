"""
Tests for STDP learning and short-term synaptic depression (habituation)
in the fast vectorized simulation engine.
"""

import numpy as np
import pytest

from sdmn.networks.celegans import (
    CElegansNetwork, SimulationConfig, PlasticityConfig, build_connectome_network,
)
from sdmn.visualization.fast_sim import (
    _prepare_arrays, _fast_step, build_frame_from_arrays, get_plasticity_stats,
)


def _make_config(stdp=True, habituation=True):
    return SimulationConfig(
        dt=0.1,
        record_interval=5,
        plasticity=PlasticityConfig(
            enable_stdp=stdp,
            enable_habituation=habituation,
        ),
    )


@pytest.fixture
def small_net():
    """3-neuron chain with plasticity enabled."""
    net = CElegansNetwork(config=_make_config())
    net.add_sensory_neuron("S1")
    net.add_interneuron("I1")
    net.add_motor_neuron("M1")
    net.add_graded_synapse("S1", "I1", weight=2.0)
    net.add_graded_synapse("I1", "M1", weight=2.0)
    return net


@pytest.fixture
def connectome_arrays():
    net = build_connectome_network()
    net.config = _make_config()
    return _prepare_arrays(net)


class TestHabituation:
    """Short-term synaptic depression tests."""

    def test_efficacy_starts_at_one(self, small_net):
        arrays = _prepare_arrays(small_net)
        assert np.allclose(arrays["syn_efficacy"], 1.0)

    def test_repeated_activation_reduces_efficacy(self, small_net):
        arrays = _prepare_arrays(small_net)
        arrays["I_ext"][0] = 50.0  # stimulate S1

        for _ in range(5000):
            _fast_step(arrays, 0.1)

        assert arrays["syn_efficacy"].min() < 0.95, \
            f"Efficacy should drop under sustained activation, got min={arrays['syn_efficacy'].min():.3f}"

    def test_efficacy_recovers_after_stimulus_removed(self, small_net):
        arrays = _prepare_arrays(small_net)
        arrays["I_ext"][0] = 50.0
        for _ in range(3000):
            _fast_step(arrays, 0.1)
        depressed_eff = arrays["syn_efficacy"].copy()

        arrays["I_ext"][0] = 0.0
        for _ in range(10000):
            _fast_step(arrays, 0.1)

        assert arrays["syn_efficacy"].mean() > depressed_eff.mean(), \
            "Efficacy should recover after stimulus removal"

    def test_efficacy_bounded(self, connectome_arrays):
        arrays = connectome_arrays
        arrays["I_ext"][:] = 30.0
        for _ in range(2000):
            _fast_step(arrays, 0.1)
        assert arrays["syn_efficacy"].min() >= 0.1, "Efficacy should not drop below 0.1"
        assert arrays["syn_efficacy"].max() <= 1.0


class TestSTDPLearning:
    """Spike-timing-dependent plasticity tests."""

    def test_initial_weights_preserved(self, small_net):
        arrays = _prepare_arrays(small_net)
        assert np.allclose(arrays["syn_weight"], arrays["syn_weight_initial"])

    def test_correlated_activity_strengthens(self, small_net):
        """When pre and post are both active, synaptic weight should increase."""
        arrays = _prepare_arrays(small_net)
        initial_w = arrays["syn_weight"].copy()

        arrays["I_ext"][:] = 40.0
        for _ in range(20000):
            _fast_step(arrays, 0.1)

        assert arrays["syn_weight"].max() > initial_w.max(), \
            "Correlated activity should strengthen at least some synapses"

    def test_weights_bounded(self, small_net):
        arrays = _prepare_arrays(small_net)
        arrays["I_ext"][:] = 80.0
        for _ in range(50000):
            _fast_step(arrays, 0.1)

        w_max = float((arrays["syn_weight_initial"] * 3.0).max())
        assert float(arrays["syn_weight"].max()) <= w_max + 0.01, \
            "Weights should not exceed 3x initial"
        assert float(arrays["syn_weight"].min()) >= 0.05 - 0.01, \
            "Weights should not drop below minimum"

    def test_stdp_accumulator_resets(self, small_net):
        arrays = _prepare_arrays(small_net)
        for _ in range(150):
            _fast_step(arrays, 0.1)
        assert arrays["stdp_accumulator"] < 100, \
            "STDP accumulator should reset after 100 steps"

    def test_restart_resets_weights(self, small_net):
        arrays = _prepare_arrays(small_net)
        arrays["I_ext"][:] = 40.0
        for _ in range(20000):
            _fast_step(arrays, 0.1)

        arrays["syn_weight"][:] = arrays["syn_weight_initial"]
        arrays["syn_efficacy"][:] = 1.0
        assert np.allclose(arrays["syn_weight"], arrays["syn_weight_initial"])
        assert np.allclose(arrays["syn_efficacy"], 1.0)


class TestPlasticityStats:
    """Tests for the plasticity statistics helper."""

    def test_initial_stats(self, small_net):
        arrays = _prepare_arrays(small_net)
        stats = get_plasticity_stats(arrays)
        assert stats["n_strengthened"] == 0
        assert stats["n_weakened"] == 0
        assert stats["n_habituated"] == 0
        assert stats["efficacy_mean"] == pytest.approx(1.0)

    def test_stats_after_activity(self, small_net):
        arrays = _prepare_arrays(small_net)
        arrays["I_ext"][:] = 40.0
        for _ in range(20000):
            _fast_step(arrays, 0.1)
        stats = get_plasticity_stats(arrays)
        assert stats["n_total"] == 2
        assert stats["efficacy_mean"] < 1.0

    def test_connectome_stats(self, connectome_arrays):
        stats = get_plasticity_stats(connectome_arrays)
        assert stats["n_total"] > 500
        assert stats["efficacy_mean"] == pytest.approx(1.0)


class TestBuildFrameWithPlasticity:
    """Verify frame building works with the new plasticity arrays."""

    def test_frame_after_learning(self, small_net):
        arrays = _prepare_arrays(small_net)
        arrays["I_ext"][0] = 50.0
        for _ in range(1000):
            _fast_step(arrays, 0.1)
        frame = build_frame_from_arrays(arrays, 100.0, 1000)
        assert len(frame.voltages) == 3
        assert frame.time_ms == 100.0
