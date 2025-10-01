"""Tests for neuron models and synaptic connections."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from sdmn.neurons import (
    BaseNeuron,
    NeuronParameters,
    NeuronState,
    NeuronType,
    NeuronFactory,
    LIFNeuron,
    LIFParameters,
    HHNeuron,
    HHParameters,
    Synapse,
    BaseSynapse,
    SynapseType,
    SynapticParameters,
    SynapseFactory,
    NeurotransmitterType,
)


class TestNeuronParameters:
    """Test NeuronParameters class."""

    @pytest.mark.unit
    def test_default_parameters(self):
        """Test default parameter values."""
        params = NeuronParameters(NeuronType.LEAKY_INTEGRATE_FIRE)
        
        assert params.neuron_type == NeuronType.LEAKY_INTEGRATE_FIRE
        assert params.dt == 0.1
        assert params.v_rest == -70.0
        assert params.v_thresh == -50.0
        assert params.v_reset == -80.0
        assert params.refractory_period == 2.0

    @pytest.mark.unit
    def test_custom_parameters(self):
        """Test setting custom parameters."""
        params = NeuronParameters(
            neuron_type=NeuronType.HODGKIN_HUXLEY,
            dt=0.01,
            v_rest=-65.0,
            v_thresh=-40.0
        )
        
        assert params.neuron_type == NeuronType.HODGKIN_HUXLEY
        assert params.dt == 0.01
        assert params.v_rest == -65.0
        assert params.v_thresh == -40.0

    @pytest.mark.unit
    def test_custom_params_methods(self):
        """Test custom parameter get/set methods."""
        params = NeuronParameters(NeuronType.LEAKY_INTEGRATE_FIRE)
        
        params.set_parameter('test_param', 42)
        assert params.get_parameter('test_param') == 42
        assert params.get_parameter('nonexistent', 'default') == 'default'


class TestNeuronState:
    """Test NeuronState class."""

    @pytest.mark.unit
    def test_default_state(self):
        """Test default state values."""
        state = NeuronState(membrane_potential=-70.0)
        
        assert state.membrane_potential == -70.0
        assert state.spike_time is None
        assert state.refractory_until == 0.0
        assert len(state.state_vars) == 0

    @pytest.mark.unit
    def test_state_variables(self):
        """Test state variable get/set methods."""
        state = NeuronState(membrane_potential=-70.0)
        
        state.set_state_var('test_var', 1.5)
        assert state.get_state_var('test_var') == 1.5
        assert state.get_state_var('nonexistent', 0.0) == 0.0

    @pytest.mark.unit
    def test_refractory_check(self):
        """Test refractory period checking."""
        state = NeuronState(membrane_potential=-70.0)
        
        # Not in refractory period
        assert not state.is_refractory(10.0)
        
        # Set refractory period
        state.refractory_until = 15.0
        assert state.is_refractory(12.0)  # Still refractory
        assert not state.is_refractory(16.0)  # Past refractory


class TestLIFNeuron:
    """Test LIFNeuron class."""

    @pytest.mark.unit
    def test_lif_initialization(self, lif_neuron):
        """Test LIF neuron initialization."""
        assert lif_neuron.neuron_id == "test_lif"
        assert lif_neuron.parameters.neuron_type == NeuronType.LEAKY_INTEGRATE_FIRE
        assert lif_neuron.get_membrane_potential() == -70.0
        assert len(lif_neuron.spike_times) == 0

    @pytest.mark.unit
    def test_lif_parameters(self):
        """Test LIF specific parameters."""
        params = LIFParameters(tau_m=10.0, r_mem=5.0)
        neuron = LIFNeuron("test", params)
        
        assert neuron.tau_m == 10.0
        assert neuron.r_mem == 5.0
        assert neuron.get_membrane_time_constant() == 10.0
        assert neuron.get_membrane_resistance() == 5.0

    @pytest.mark.unit
    def test_lif_membrane_integration(self, lif_neuron):
        """Test membrane potential integration."""
        # Test with no input (should decay to rest)
        initial_v = -60.0
        lif_neuron.set_membrane_potential(initial_v)
        
        new_v = lif_neuron.integrate_membrane_equation(0.1, 0.0)
        
        # Should move toward resting potential
        assert new_v < initial_v
        assert new_v > -70.0

    @pytest.mark.unit
    def test_lif_spike_generation(self, lif_neuron):
        """Test spike generation in LIF neuron."""
        # Set voltage above threshold
        lif_neuron.set_membrane_potential(-45.0)
        
        initial_spike_count = len(lif_neuron.spike_times)
        lif_neuron.update(0.1)
        
        # Should generate spike and reset
        assert len(lif_neuron.spike_times) == initial_spike_count + 1
        assert lif_neuron.get_membrane_potential() == -80.0  # Reset value
        assert lif_neuron.has_spiked()

    @pytest.mark.unit
    def test_lif_refractory_period(self, lif_neuron):
        """Test refractory period behavior."""
        # Generate a spike
        lif_neuron.set_membrane_potential(-45.0)
        lif_neuron.update(0.1)
        
        # Should be in refractory period
        assert lif_neuron.state.is_refractory(lif_neuron.current_time + 1.0)
        
        # Try to spike again during refractory (should not spike)
        lif_neuron.set_membrane_potential(-45.0)
        spike_count_before = len(lif_neuron.spike_times)
        lif_neuron.update(0.1)
        
        assert len(lif_neuron.spike_times) == spike_count_before

    @pytest.mark.unit
    def test_lif_external_input(self, lif_neuron):
        """Test external input handling."""
        lif_neuron.set_external_input(1.0)  # 1 nA input
        
        total_input = lif_neuron.get_total_input()
        assert total_input == 1.0
        
        # Clear inputs
        lif_neuron.clear_inputs()
        assert lif_neuron.get_total_input() == 0.0

    @pytest.mark.unit
    def test_lif_synaptic_input(self, lif_neuron):
        """Test synaptic input accumulation."""
        lif_neuron.add_synaptic_input(0.5)
        lif_neuron.add_synaptic_input(0.3)
        
        total_input = lif_neuron.get_total_input()
        assert total_input == 0.8

    @pytest.mark.unit
    def test_lif_firing_rate_calculation(self, lif_neuron):
        """Test firing rate calculation."""
        # Manually add spike times
        lif_neuron.spike_times = [10.0, 20.0, 30.0, 40.0]
        lif_neuron.current_time = 50.0
        
        # Test 20ms window (should contain 2 spikes)
        rate = lif_neuron.get_firing_rate(20.0)
        expected_rate = 2 * 1000.0 / 20.0  # Convert to Hz
        assert rate == expected_rate

    @pytest.mark.unit
    def test_lif_analytical_firing_rate(self, lif_neuron):
        """Test analytical firing rate calculation."""
        # Test with suprathreshold constant current
        input_current = 1.0  # nA
        
        rate = lif_neuron.get_analytical_firing_rate(input_current)
        assert rate > 0  # Should fire with positive input
        
        # Test with subthreshold current
        rate_subthresh = lif_neuron.get_analytical_firing_rate(0.1)
        assert rate_subthresh == 0  # Should not fire

    @pytest.mark.unit
    def test_lif_reset(self, lif_neuron):
        """Test neuron reset functionality."""
        # Modify neuron state
        lif_neuron.set_membrane_potential(-50.0)
        lif_neuron.spike_times = [10.0, 20.0]
        lif_neuron.current_time = 25.0
        
        lif_neuron.reset_neuron()
        
        assert lif_neuron.get_membrane_potential() == -70.0
        assert len(lif_neuron.spike_times) == 0
        assert lif_neuron.current_time == 0.0


class TestHHNeuron:
    """Test HHNeuron class."""

    @pytest.mark.unit
    def test_hh_initialization(self, hh_neuron):
        """Test HH neuron initialization."""
        assert hh_neuron.neuron_id == "test_hh"
        assert hh_neuron.parameters.neuron_type == NeuronType.HODGKIN_HUXLEY
        assert hasattr(hh_neuron, 'g_na_max')
        assert hasattr(hh_neuron, 'g_k_max')

    @pytest.mark.unit
    def test_hh_gating_variables(self, hh_neuron):
        """Test HH gating variable initialization."""
        m, h, n = hh_neuron.get_gating_variables()
        
        # Should be initialized to steady-state values at rest
        assert 0 <= m <= 1
        assert 0 <= h <= 1  
        assert 0 <= n <= 1

    @pytest.mark.unit
    def test_hh_ionic_currents(self, hh_neuron):
        """Test ionic current calculations."""
        currents = hh_neuron.get_ionic_currents()
        
        assert 'i_na' in currents
        assert 'i_k' in currents
        assert 'i_leak' in currents
        
        # At rest, currents should be small
        assert abs(currents['i_na']) < 50
        assert abs(currents['i_k']) < 50
        assert abs(currents['i_leak']) < 10

    @pytest.mark.unit
    def test_hh_integration_methods(self, hh_neuron):
        """Test different integration methods."""
        initial_v = hh_neuron.get_membrane_potential()
        
        # Test RK4 method
        hh_neuron.set_integration_method('runge_kutta_4')
        hh_neuron.update(0.01)
        v_rk4 = hh_neuron.get_membrane_potential()
        
        # Reset and test Euler method
        hh_neuron.reset_neuron()
        hh_neuron.set_integration_method('euler')
        hh_neuron.update(0.01)
        v_euler = hh_neuron.get_membrane_potential()
        
        # Both should be close for small dt
        assert abs(v_rk4 - v_euler) < 1.0

    @pytest.mark.unit
    def test_hh_spike_detection(self, hh_neuron):
        """Test spike detection in HH neuron."""
        # Apply strong depolarizing current
        hh_neuron.set_external_input(50.0)  # Strong current
        
        spike_count_initial = len(hh_neuron.spike_times)
        
        # Run for several time steps to allow spike generation
        for _ in range(100):  # 1ms at 0.01ms steps
            hh_neuron.update(0.01)
            if hh_neuron.has_spiked():
                break
        
        # Should have generated at least one spike
        assert len(hh_neuron.spike_times) > spike_count_initial

    @pytest.mark.unit
    def test_hh_temperature_effects(self):
        """Test temperature effects on HH dynamics."""
        # Create HH neuron with different temperature
        params_warm = HHParameters(temperature=20.0)  # Warmer
        params_cold = HHParameters(temperature=0.0)   # Colder
        
        neuron_warm = HHNeuron("warm", params_warm)
        neuron_cold = HHNeuron("cold", params_cold)
        
        # Temperature factor should be different
        assert neuron_warm.temp_factor != neuron_cold.temp_factor
        assert neuron_warm.temp_factor > neuron_cold.temp_factor


class TestSynapticParameters:
    """Test SynapticParameters class."""

    @pytest.mark.unit
    def test_default_parameters(self):
        """Test default synaptic parameters."""
        params = SynapticParameters(SynapseType.EXCITATORY)
        
        assert params.synapse_type == SynapseType.EXCITATORY
        assert params.neurotransmitter == NeurotransmitterType.GLUTAMATE
        assert params.weight == 1.0
        assert params.delay == 1.0
        assert params.reversal_potential == 0.0  # Excitatory

    @pytest.mark.unit
    def test_inhibitory_parameters(self):
        """Test inhibitory synapse parameters."""
        params = SynapticParameters(SynapseType.INHIBITORY)
        
        assert params.synapse_type == SynapseType.INHIBITORY
        assert params.reversal_potential == -80.0  # Inhibitory


class TestSynapse:
    """Test Synapse class."""

    @pytest.mark.unit
    def test_synapse_initialization(self, excitatory_synapse):
        """Test synapse initialization."""
        assert excitatory_synapse.synapse_id == "syn_exc"
        assert excitatory_synapse.presynaptic_neuron_id == "pre_neuron"
        assert excitatory_synapse.postsynaptic_neuron_id == "post_neuron"
        assert excitatory_synapse.is_excitatory()
        assert not excitatory_synapse.is_inhibitory()

    @pytest.mark.unit
    def test_synapse_spike_reception(self, excitatory_synapse):
        """Test spike reception and conductance changes."""
        initial_conductance = excitatory_synapse.get_conductance()
        
        # Send spike to synapse
        excitatory_synapse.receive_spike(10.0)
        
        # Update synapse (should increase conductance)
        excitatory_synapse.update(0.1)
        
        # Conductance should increase after spike
        assert excitatory_synapse.get_conductance() > initial_conductance

    @pytest.mark.unit
    def test_synapse_conductance_decay(self, excitatory_synapse):
        """Test conductance decay over time."""
        # Generate spike and let conductance build up
        excitatory_synapse.receive_spike(10.0)
        excitatory_synapse.update(0.1)
        
        peak_conductance = excitatory_synapse.get_conductance()
        
        # Continue updating without spikes (should decay)
        for _ in range(10):
            excitatory_synapse.update(1.0)  # Large time steps
        
        final_conductance = excitatory_synapse.get_conductance()
        assert final_conductance < peak_conductance

    @pytest.mark.unit
    def test_synapse_current_calculation(self, excitatory_synapse):
        """Test synaptic current calculation."""
        # Send spike and update
        excitatory_synapse.receive_spike(10.0)
        excitatory_synapse.update(0.1)
        
        # Calculate current at different postsynaptic voltages
        current_at_rest = excitatory_synapse.calculate_current(-70.0)
        current_at_reversal = excitatory_synapse.calculate_current(0.0)
        
        # Current should be different at different voltages
        assert abs(current_at_rest) > abs(current_at_reversal)

    @pytest.mark.unit
    def test_synapse_weight_modification(self, excitatory_synapse):
        """Test synaptic weight modification."""
        initial_weight = excitatory_synapse.get_weight()
        
        excitatory_synapse.set_weight(2.0)
        
        assert excitatory_synapse.get_weight() == 2.0
        assert len(excitatory_synapse.weight_history) > 0

    @pytest.mark.unit
    def test_inhibitory_synapse(self, inhibitory_synapse):
        """Test inhibitory synapse properties."""
        assert inhibitory_synapse.is_inhibitory()
        assert not inhibitory_synapse.is_excitatory()
        assert inhibitory_synapse.parameters.reversal_potential == -80.0

    @pytest.mark.unit
    def test_synapse_plasticity(self):
        """Test synaptic plasticity mechanisms."""
        # Create plastic synapse
        plastic_synapse = SynapseFactory.create_plastic_synapse(
            "plastic", "pre", "post", learning_rate=0.1
        )
        
        assert plastic_synapse.parameters.enable_plasticity is True
        assert plastic_synapse.parameters.learning_rate == 0.1
        
        initial_weight = plastic_synapse.get_weight()
        
        # Simulate pre-then-post spike timing (should potentiate)
        plastic_synapse.receive_spike(10.0)  # Pre spike
        plastic_synapse.receive_postsynaptic_spike(12.0)  # Post spike 2ms later
        
        # Weight should have changed
        assert plastic_synapse.get_weight() != initial_weight

    @pytest.mark.unit
    def test_synapse_reset(self, excitatory_synapse):
        """Test synapse reset functionality."""
        # Modify synapse state
        excitatory_synapse.receive_spike(10.0)
        excitatory_synapse.update(0.1)
        excitatory_synapse.spike_times = [10.0, 20.0]
        
        excitatory_synapse.reset_synapse()
        
        assert excitatory_synapse.get_conductance() == 0.0
        assert excitatory_synapse.get_current() == 0.0
        assert len(excitatory_synapse.spike_times) == 0


class TestSynapseFactory:
    """Test SynapseFactory class."""

    @pytest.mark.unit
    def test_create_excitatory_synapse(self):
        """Test creating excitatory synapse."""
        synapse = SynapseFactory.create_excitatory_synapse(
            "exc_syn", "pre", "post", weight=1.5, delay=2.0
        )
        
        assert synapse.is_excitatory()
        assert synapse.parameters.weight == 1.5
        assert synapse.parameters.delay == 2.0
        assert synapse.parameters.neurotransmitter == NeurotransmitterType.GLUTAMATE

    @pytest.mark.unit  
    def test_create_inhibitory_synapse(self):
        """Test creating inhibitory synapse."""
        synapse = SynapseFactory.create_inhibitory_synapse(
            "inh_syn", "pre", "post", weight=2.0, delay=1.5
        )
        
        assert synapse.is_inhibitory()
        assert synapse.parameters.weight == 2.0
        assert synapse.parameters.delay == 1.5
        assert synapse.parameters.neurotransmitter == NeurotransmitterType.GABA

    @pytest.mark.unit
    def test_create_plastic_synapse(self):
        """Test creating plastic synapse."""
        synapse = SynapseFactory.create_plastic_synapse(
            "plastic_syn", "pre", "post", 
            synapse_type=SynapseType.EXCITATORY,
            learning_rate=0.05
        )
        
        assert synapse.parameters.enable_plasticity is True
        assert synapse.parameters.learning_rate == 0.05


class TestNeuronFactory:
    """Test NeuronFactory class."""

    @pytest.mark.unit
    def test_create_lif_neuron(self):
        """Test creating LIF neuron through factory."""
        params = LIFParameters(tau_m=15.0)
        neuron = NeuronFactory.create_neuron(
            NeuronType.LEAKY_INTEGRATE_FIRE, "factory_lif", params
        )
        
        assert isinstance(neuron, LIFNeuron)
        assert neuron.neuron_id == "factory_lif"
        assert neuron.tau_m == 15.0

    @pytest.mark.unit
    def test_create_hh_neuron(self):
        """Test creating HH neuron through factory."""
        params = HHParameters(temperature=15.0)
        neuron = NeuronFactory.create_neuron(
            NeuronType.HODGKIN_HUXLEY, "factory_hh", params
        )
        
        assert isinstance(neuron, HHNeuron)
        assert neuron.neuron_id == "factory_hh"

    @pytest.mark.unit
    def test_get_available_types(self):
        """Test getting available neuron types."""
        available_types = NeuronFactory.get_available_types()
        
        assert NeuronType.LEAKY_INTEGRATE_FIRE in available_types
        assert NeuronType.HODGKIN_HUXLEY in available_types

    @pytest.mark.unit
    def test_unknown_neuron_type(self):
        """Test error handling for unknown neuron type."""
        with pytest.raises(ValueError):
            NeuronFactory.create_neuron(
                NeuronType.CUSTOM, "test", NeuronParameters(NeuronType.CUSTOM)
            )
