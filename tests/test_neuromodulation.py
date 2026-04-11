"""
Tests for the neuromodulation system: dopamine, serotonin, octopamine,
reward-modulated STDP, food/pain environment, and sensory adaptation.
"""

import numpy as np
import pytest

from sdmn.networks.celegans import CElegansNetwork, build_connectome_network
from sdmn.visualization.neuromodulation import (
    NeuromodulationSystem, NeuromodConfig,
    DA_NEURONS, SEROTONIN_NEURONS, OCTOPAMINE_NEURONS, NOCICEPTIVE_NEURONS,
)
from sdmn.visualization.fast_sim import _prepare_arrays, _fast_step
from sdmn.visualization.environment import WormEnvironment


@pytest.fixture
def connectome_arrays():
    net = build_connectome_network()
    return _prepare_arrays(net)


@pytest.fixture
def neuromod(connectome_arrays):
    a = connectome_arrays
    nm = NeuromodulationSystem(NeuromodConfig(), a["ids"], a["id_to_idx"])
    nm.init_eligibility(len(a["syn_pre"]))
    return nm


class TestNeuromodulatorIdentities:

    def test_da_neurons_found(self, connectome_arrays, neuromod):
        assert len(neuromod.da_indices) > 0, "Should find dopamine neurons in connectome"

    def test_serotonin_neurons_found(self, connectome_arrays, neuromod):
        assert len(neuromod.serotonin_indices) > 0

    def test_octopamine_neurons_found(self, connectome_arrays, neuromod):
        assert len(neuromod.octopamine_indices) > 0

    def test_nociceptive_neurons_found(self, connectome_arrays, neuromod):
        assert len(neuromod.nociceptive_indices) > 0


class TestDopamineRelease:

    def test_dopamine_rises_when_da_neurons_active(self, connectome_arrays, neuromod):
        a = connectome_arrays
        a["V"][:] = -65.0
        for idx in neuromod.da_indices:
            a["V"][idx] = -30.0  # depolarize DA neurons
        neuromod.step(a["V"], a["I_ext"], dt_batch=10.0)
        assert neuromod.dopamine > 0.005, f"DA should rise: {neuromod.dopamine}"

    def test_dopamine_decays_without_input(self, connectome_arrays, neuromod):
        neuromod.dopamine = 0.5
        a = connectome_arrays
        a["V"][:] = -65.0  # all at rest
        for _ in range(50):
            neuromod.step(a["V"], a["I_ext"], dt_batch=50.0)
        assert neuromod.dopamine < 0.3, "DA should decay"


class TestSerotoninRelease:

    def test_serotonin_rises_when_nsm_active(self, connectome_arrays, neuromod):
        a = connectome_arrays
        a["V"][:] = -65.0
        for idx in neuromod.serotonin_indices:
            a["V"][idx] = -30.0
        neuromod.step(a["V"], a["I_ext"], dt_batch=10.0)
        assert neuromod.serotonin > 0.0


class TestOctopamineRelease:

    def test_octopamine_rises_under_stress(self, connectome_arrays, neuromod):
        a = connectome_arrays
        a["V"][:] = -65.0
        for idx in neuromod.octopamine_indices:
            a["V"][idx] = -30.0
        neuromod.step(a["V"], a["I_ext"], dt_batch=10.0)
        assert neuromod.octopamine > 0.0


class TestModulatoryCurrents:

    def test_dopamine_boosts_food_sensory(self, connectome_arrays, neuromod):
        neuromod.dopamine = 0.8
        currents = neuromod.get_modulatory_currents()
        for idx in neuromod.food_sensory_indices:
            assert currents[idx] > 0, "DA should boost food sensory neurons"

    def test_serotonin_calms(self, connectome_arrays, neuromod):
        neuromod.serotonin = 0.8
        currents = neuromod.get_modulatory_currents()
        assert currents.mean() < 0, "Serotonin should suppress excitability"

    def test_octopamine_arouses(self, connectome_arrays, neuromod):
        neuromod.octopamine = 0.8
        currents = neuromod.get_modulatory_currents()
        assert currents.mean() > 0, "Octopamine should increase excitability"


class TestRewardSignal:

    def test_positive_reward_with_dopamine(self, neuromod):
        neuromod.dopamine = 0.8
        neuromod.octopamine = 0.0
        neuromod.serotonin = 0.0
        assert neuromod.get_reward_signal() > 0

    def test_negative_reward_with_octopamine(self, neuromod):
        neuromod.dopamine = 0.0
        neuromod.octopamine = 0.8
        neuromod.serotonin = 0.0
        assert neuromod.get_reward_signal() < 0

    def test_satiety_reduces_learning(self, neuromod):
        neuromod.dopamine = 0.5
        neuromod.serotonin = 0.0
        r1 = neuromod.get_reward_signal()
        neuromod.serotonin = 0.8
        r2 = neuromod.get_reward_signal()
        assert abs(r2) < abs(r1), "Serotonin should gate down reward"


class TestSpeedModulation:

    def test_serotonin_slows(self, neuromod):
        neuromod.serotonin = 0.8
        neuromod.octopamine = 0.0
        assert neuromod.get_speed_modulation() < 1.0

    def test_octopamine_speeds(self, neuromod):
        neuromod.serotonin = 0.0
        neuromod.octopamine = 0.8
        assert neuromod.get_speed_modulation() > 1.0


class TestRewardModulatedSTDP:

    def test_eligibility_traces_update(self, connectome_arrays, neuromod):
        a = connectome_arrays
        a["trace_pre"][:] = 0.5
        a["trace_post"][:] = 0.5
        neuromod.update_eligibility(
            a["trace_pre"], a["trace_post"],
            a["syn_pre"], a["syn_post"], dt_batch=10.0)
        assert neuromod.eligibility_traces.max() > 0

    def test_reward_learning_changes_weights(self, connectome_arrays, neuromod):
        a = connectome_arrays
        w_before = a["syn_weight"].copy()
        neuromod.eligibility_traces[:] = 1.0
        neuromod.dopamine = 0.9
        neuromod.apply_reward_learning(
            a["syn_weight"], a["syn_weight_initial"],
            w_min=0.05, w_max_factor=3.0)
        assert not np.allclose(a["syn_weight"], w_before)


class TestFoodAndPainEnvironment:

    def test_food_depletes(self):
        env = WormEnvironment()
        food = env.food_sources[0]
        initial = food.amount
        env._x = food.x
        env._y = food.y
        env.step({}, 100.0)
        assert food.amount < initial

    def test_pain_detected(self):
        env = WormEnvironment()
        pz = env.pain_zones[0]
        env._x = pz.x
        env._y = pz.y
        env.step({}, 1.0)
        assert env._pain_at_head > 0

    def test_da_neurons_get_current_near_food(self):
        env = WormEnvironment()
        food = env.food_sources[0]
        env._x = food.x
        env._y = food.y
        env.step({}, 1.0)
        currents = env.get_sensory_currents()
        assert currents.get("CEPDL", 0) > 0, "DA neurons should fire near food"

    def test_ash_neurons_get_current_in_pain(self):
        env = WormEnvironment()
        pz = env.pain_zones[0]
        env._x = pz.x
        env._y = pz.y
        env.step({}, 1.0)
        currents = env.get_sensory_currents()
        assert currents.get("ASHL", 0) > 0, "ASH should fire in pain zone"

    def test_nsm_fires_during_feeding(self):
        env = WormEnvironment()
        food = env.food_sources[0]
        env._x = food.x
        env._y = food.y
        env.step({}, 1.0)
        currents = env.get_sensory_currents()
        assert currents.get("NSML", 0) > 0, "NSM should fire during feeding"


class TestNeuromodReset:

    def test_reset_clears_state(self, neuromod):
        neuromod.dopamine = 0.9
        neuromod.serotonin = 0.5
        neuromod.octopamine = 0.7
        neuromod.reset()
        assert neuromod.dopamine == 0.0
        assert neuromod.serotonin == 0.0
        assert neuromod.octopamine == 0.0


class TestToJson:

    def test_serialization(self, neuromod):
        neuromod.dopamine = 0.5
        j = neuromod.to_json()
        assert "dopamine" in j
        assert "serotonin" in j
        assert "reward" in j
        assert "speed_mod" in j
        assert j["dopamine"] == pytest.approx(0.5, abs=0.01)
