"""
Tests for the unified vectorized engine.

Verifies that ion channel dynamics, synaptic kinetics, and gap junctions
in fast_sim match the object-based engine within tolerance.
"""

import numpy as np
import pytest

from sdmn.networks.celegans import (
    CElegansNetwork, SimulationConfig, PlasticityConfig,
    build_connectome_network,
)
from sdmn.neurons.graded import SensoryNeuron, Interneuron, MotorNeuron
from sdmn.visualization.fast_sim import (
    _prepare_arrays, _fast_step, build_frame_from_arrays, get_plasticity_stats,
)


def _make_chain():
    """S -> I -> M chain for comparison tests."""
    net = CElegansNetwork()
    net.add_sensory_neuron("S1")
    net.add_interneuron("I1")
    net.add_motor_neuron("M1")
    net.add_graded_synapse("S1", "I1", weight=2.0)
    net.add_graded_synapse("I1", "M1", weight=2.0)
    net.add_gap_junction("S1", "I1", conductance=0.5)
    return net


class TestIonChannelArrays:
    """Verify per-neuron ion channel parameters are extracted correctly."""

    def test_class_specific_parameters(self):
        net = _make_chain()
        arrays = _prepare_arrays(net)
        idx = arrays["id_to_idx"]

        assert arrays["g_Ca"][idx["S1"]] > arrays["g_Ca"][idx["M1"]], \
            "Sensory g_Ca should be > Motor g_Ca"
        assert arrays["g_K"][idx["M1"]] > arrays["g_K"][idx["S1"]], \
            "Motor g_K should be > Sensory g_K"
        assert arrays["g_KCa"][idx["M1"]] > arrays["g_KCa"][idx["I1"]], \
            "Motor g_KCa should be > Interneuron g_KCa"

    def test_gating_variables_initialized(self):
        net = _make_chain()
        arrays = _prepare_arrays(net)
        assert arrays["m_Ca"].shape == (3,)
        assert np.all(arrays["m_Ca"] >= 0)
        assert np.all(arrays["m_Ca"] <= 1)
        assert np.all(arrays["m_K"] >= 0)
        assert np.all(arrays["Ca_internal"] >= 0)

    def test_calcium_dynamics_respond_to_depolarization(self):
        net = _make_chain()
        arrays = _prepare_arrays(net)
        idx = arrays["id_to_idx"]["S1"]

        ca_before = float(arrays["Ca_internal"][idx])
        arrays["I_ext"][idx] = 80.0
        for _ in range(2000):
            _fast_step(arrays, 0.1)

        ca_after = float(arrays["Ca_internal"][idx])
        assert ca_after > ca_before, \
            f"Ca should increase with depolarization: {ca_before:.1f} -> {ca_after:.1f}"


class TestVoltageTraceComparison:
    """Compare object engine and vectorized engine voltage traces."""

    def test_single_neuron_step_response(self):
        """A single stimulated sensory neuron should produce similar voltage
        trajectories in both engines."""
        net_obj = CElegansNetwork()
        net_obj.add_sensory_neuron("S1")

        net_vec = CElegansNetwork()
        net_vec.add_sensory_neuron("S1")

        net_obj.set_external_current("S1", 50.0)
        net_obj.simulate(duration=10, progress=False)
        obj_voltages = [v for _, v in net_obj.voltage_history["S1"]]

        arrays = _prepare_arrays(net_vec)
        arrays["I_ext"][0] = 50.0
        vec_voltages = []
        for step in range(1000):
            _fast_step(arrays, 0.01)
            if step % 10 == 0:
                vec_voltages.append(float(arrays["V"][0]))

        n = min(len(obj_voltages), len(vec_voltages))
        if n > 5:
            obj_v = np.array(obj_voltages[:n])
            vec_v = np.array(vec_voltages[:n])
            rmse = np.sqrt(np.mean((obj_v - vec_v) ** 2))
            assert rmse < 15.0, \
                f"RMSE between engines = {rmse:.2f} mV (should be < 15)"

    def test_sensory_more_excitable_than_motor(self):
        """Sensory neuron should depolarize more than motor under same stimulus."""
        net = CElegansNetwork()
        net.add_sensory_neuron("S1")
        net.add_motor_neuron("M1")
        arrays = _prepare_arrays(net)

        arrays["I_ext"][:] = 30.0
        for _ in range(5000):
            _fast_step(arrays, 0.1)

        v_s = float(arrays["V"][arrays["id_to_idx"]["S1"]])
        v_m = float(arrays["V"][arrays["id_to_idx"]["M1"]])
        assert v_s > v_m, \
            f"Sensory ({v_s:.1f} mV) should be more depolarized than Motor ({v_m:.1f} mV)"

    def test_interneuron_between_sensory_and_motor(self):
        net = CElegansNetwork()
        net.add_sensory_neuron("S1")
        net.add_interneuron("I1")
        net.add_motor_neuron("M1")
        arrays = _prepare_arrays(net)

        arrays["I_ext"][:] = 30.0
        for _ in range(5000):
            _fast_step(arrays, 0.1)

        v_s = float(arrays["V"][arrays["id_to_idx"]["S1"]])
        v_i = float(arrays["V"][arrays["id_to_idx"]["I1"]])
        v_m = float(arrays["V"][arrays["id_to_idx"]["M1"]])
        assert v_s > v_i > v_m, \
            f"Expected S({v_s:.1f}) > I({v_i:.1f}) > M({v_m:.1f})"


class TestSynapseKinetics:
    """Verify dual-exponential kinetics and reversal-potential current."""

    def test_g_rise_g_decay_update(self):
        net = _make_chain()
        arrays = _prepare_arrays(net)
        arrays["I_ext"][arrays["id_to_idx"]["S1"]] = 50.0

        for _ in range(500):
            _fast_step(arrays, 0.1)

        assert arrays["syn_g_decay"].max() > 0, "g_decay should be nonzero after stimulation"

    def test_per_synapse_params_used(self):
        """V_thresh and k_release should be read from the synapse, not hardcoded."""
        net = CElegansNetwork()
        net.add_sensory_neuron("S1")
        net.add_interneuron("I1")
        net.add_graded_synapse("S1", "I1", weight=2.0)
        arrays = _prepare_arrays(net)

        assert arrays["syn_V_thresh"][0] == pytest.approx(-40.0)
        assert arrays["syn_k_release"][0] == pytest.approx(5.0)
        assert arrays["syn_tau_rise"][0] == pytest.approx(1.0)
        assert arrays["syn_tau_decay"][0] == pytest.approx(5.0)

    def test_inhibitory_reversal_potential(self):
        net = CElegansNetwork()
        net.add_sensory_neuron("S1")
        net.add_interneuron("I1")
        from sdmn.synapses import SynapseType
        net.add_graded_synapse("S1", "I1", weight=2.0, synapse_type=SynapseType.INHIBITORY)
        arrays = _prepare_arrays(net)

        assert arrays["syn_E_syn"][0] == pytest.approx(-75.0)
        assert arrays["syn_exc"][0] == pytest.approx(-1.0)


class TestGapJunctionFix:
    """Verify gap junction conductance is read correctly."""

    def test_conductance_from_params(self):
        net = CElegansNetwork()
        net.add_sensory_neuron("S1")
        net.add_interneuron("I1")
        net.add_gap_junction("S1", "I1", conductance=1.2)
        arrays = _prepare_arrays(net)

        assert arrays["gap_g"][0] == pytest.approx(1.2), \
            f"Expected 1.2, got {arrays['gap_g'][0]}"

    def test_gap_current_flows(self):
        net = CElegansNetwork()
        net.add_sensory_neuron("S1")
        net.add_interneuron("I1")
        net.add_gap_junction("S1", "I1", conductance=1.0)
        arrays = _prepare_arrays(net)

        arrays["I_ext"][arrays["id_to_idx"]["S1"]] = 50.0
        for _ in range(1000):
            _fast_step(arrays, 0.1)

        v_s = float(arrays["V"][arrays["id_to_idx"]["S1"]])
        v_i = float(arrays["V"][arrays["id_to_idx"]["I1"]])
        assert v_i > -64.0, \
            f"Gap junction should pull I1 toward S1: I1 V={v_i:.1f}"


class TestConnectomeScale:
    """Verify the unified engine works on the full 300-neuron connectome."""

    def test_prepare_and_step(self):
        net = build_connectome_network()
        arrays = _prepare_arrays(net)

        assert arrays["n"] == net.get_neuron_count()
        assert len(arrays["syn_pre"]) > 500
        assert len(arrays["gap_a"]) > 50
        assert arrays["g_Ca"].shape == (arrays["n"],)

        arrays["I_ext"][arrays["id_to_idx"]["AWCL"]] = 60.0
        for _ in range(100):
            _fast_step(arrays, 0.1)

        frame = build_frame_from_arrays(arrays, 10.0, 100)
        assert len(frame.voltages) == arrays["n"]
        assert len(frame.active_gaps) >= 0

    def test_class_diversity_on_connectome(self):
        net = build_connectome_network()
        arrays = _prepare_arrays(net)

        g_ca_vals = set()
        for nid in ["AWCL", "AVAL", "DA01"]:
            idx = arrays["id_to_idx"].get(nid)
            if idx is not None:
                g_ca_vals.add(round(float(arrays["g_Ca"][idx]), 2))

        assert len(g_ca_vals) > 1, \
            "Different neuron classes should have different g_Ca values"


class TestPlasticityWithUnifiedEngine:
    """STDP and habituation still work with ion channels present."""

    def test_habituation_reduces_efficacy(self):
        net = _make_chain()
        net.config = SimulationConfig(
            dt=0.1,
            plasticity=PlasticityConfig(enable_habituation=True),
        )
        arrays = _prepare_arrays(net)
        arrays["I_ext"][arrays["id_to_idx"]["S1"]] = 50.0

        for _ in range(5000):
            _fast_step(arrays, 0.1)

        assert arrays["syn_efficacy"].min() < 0.95

    def test_stdp_changes_weights(self):
        net = _make_chain()
        net.config = SimulationConfig(
            dt=0.1,
            plasticity=PlasticityConfig(enable_stdp=True),
        )
        arrays = _prepare_arrays(net)
        w0 = arrays["syn_weight"].copy()
        arrays["I_ext"][:] = 40.0

        for _ in range(20000):
            _fast_step(arrays, 0.1)

        assert not np.allclose(arrays["syn_weight"], w0), \
            "STDP should change weights after sustained activity"
