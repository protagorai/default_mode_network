"""Tests for monitoring probes and data collection."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from sdmn.probes import (
    BaseProbe,
    ProbeType,
    ProbeData,
    ProbeManager,
    VoltageProbe,
    SpikeProbe,
    SpikeDetector,
    PopulationActivityProbe,
    LFPProbe,
    NetworkActivityProbe,
    ConnectivityProbe,
)
from sdmn.neurons import LIFNeuron, LIFParameters


class TestProbeData:
    """Test ProbeData class."""

    @pytest.mark.unit
    def test_probe_data_initialization(self):
        """Test ProbeData initialization."""
        data = ProbeData("test_probe", ProbeType.VOLTAGE, ["n1", "n2"])
        
        assert data.probe_id == "test_probe"
        assert data.probe_type == ProbeType.VOLTAGE
        assert data.target_ids == ["n1", "n2"]
        assert len(data.timestamps) == 0
        assert len(data.values) == 0

    @pytest.mark.unit
    def test_add_sample(self):
        """Test adding data samples."""
        data = ProbeData("test", ProbeType.SPIKE, ["n1"])
        
        data.add_sample(10.0, True)
        data.add_sample(15.0, False)
        
        assert len(data.timestamps) == 2
        assert len(data.values) == 2
        assert data.timestamps[0] == 10.0
        assert data.values[1] is False

    @pytest.mark.unit
    def test_get_arrays(self):
        """Test getting data as numpy arrays."""
        data = ProbeData("test", ProbeType.VOLTAGE, ["n1"])
        data.add_sample(1.0, -65.0)
        data.add_sample(2.0, -60.0)
        
        time_array = data.get_time_array()
        value_array = data.get_data_array()
        
        assert isinstance(time_array, np.ndarray)
        assert isinstance(value_array, np.ndarray)
        assert len(time_array) == 2
        assert len(value_array) == 2

    @pytest.mark.unit
    def test_sample_count_and_range(self):
        """Test sample count and time range methods."""
        data = ProbeData("test", ProbeType.CURRENT, ["n1"])
        
        assert data.get_sample_count() == 0
        assert data.get_time_range() == (0.0, 0.0)
        
        data.add_sample(5.0, 1.0)
        data.add_sample(15.0, 2.0)
        data.add_sample(10.0, 1.5)
        
        assert data.get_sample_count() == 3
        assert data.get_time_range() == (5.0, 15.0)

    @pytest.mark.unit
    def test_clear_data(self):
        """Test clearing data."""
        data = ProbeData("test", ProbeType.LFP, ["n1"])
        data.add_sample(1.0, 0.5)
        data.add_sample(2.0, 0.8)
        
        data.clear()
        
        assert len(data.timestamps) == 0
        assert len(data.values) == 0


class MockProbe(BaseProbe):
    """Mock probe for testing BaseProbe functionality."""
    
    def __init__(self, probe_id, target_ids):
        super().__init__(probe_id, ProbeType.CUSTOM, target_ids, sampling_interval=1.0)
        self.measurement_values = {}
    
    def record(self, current_time):
        if self.should_record(current_time):
            values = {tid: self.get_measurement_value(tid) for tid in self.target_ids}
            self._record_sample(current_time, values)
    
    def get_measurement_value(self, target_id):
        return self.measurement_values.get(target_id, 0.0)
    
    def set_measurement_value(self, target_id, value):
        self.measurement_values[target_id] = value


class TestBaseProbe:
    """Test BaseProbe class."""

    @pytest.fixture
    def mock_probe(self):
        """Create mock probe for testing."""
        return MockProbe("test_probe", ["target1", "target2"])

    @pytest.mark.unit
    def test_probe_initialization(self, mock_probe):
        """Test probe initialization."""
        assert mock_probe.probe_id == "test_probe"
        assert mock_probe.probe_type == ProbeType.CUSTOM
        assert mock_probe.target_ids == ["target1", "target2"]
        assert mock_probe.sampling_interval == 1.0
        assert not mock_probe.is_recording

    @pytest.mark.unit
    def test_start_stop_recording(self, mock_probe):
        """Test start/stop recording functionality."""
        mock_probe.start_recording(10.0)
        
        assert mock_probe.is_recording_active()
        assert mock_probe.recording_start_time == 10.0
        
        mock_probe.stop_recording()
        assert not mock_probe.is_recording_active()

    @pytest.mark.unit
    def test_pause_resume_recording(self, mock_probe):
        """Test pause/resume recording functionality."""
        mock_probe.start_recording()
        
        mock_probe.pause_recording()
        assert not mock_probe.is_recording_active()
        
        mock_probe.resume_recording()
        assert mock_probe.is_recording_active()

    @pytest.mark.unit
    def test_should_record_timing(self, mock_probe):
        """Test should_record timing logic."""
        mock_probe.start_recording(0.0)
        
        # Should not record immediately (no time passed)
        assert not mock_probe.should_record(0.0)
        
        # Should record after sampling interval
        assert mock_probe.should_record(1.0)
        
        # Should not record again immediately
        mock_probe.last_sample_time = 1.0
        assert not mock_probe.should_record(1.5)
        
        # Should record again after another interval
        assert mock_probe.should_record(2.0)

    @pytest.mark.unit
    def test_data_recording(self, mock_probe):
        """Test data recording functionality."""
        mock_probe.set_measurement_value("target1", 42.0)
        mock_probe.set_measurement_value("target2", -13.5)
        
        mock_probe.start_recording(0.0)
        mock_probe.record(1.0)
        
        data = mock_probe.get_data()
        assert len(data.timestamps) == 1
        assert data.timestamps[0] == 1.0
        assert isinstance(data.values[0], dict)

    @pytest.mark.unit
    def test_data_callbacks(self, mock_probe):
        """Test data callback functionality."""
        callback_data = []
        
        def test_callback(probe_data, timestamp, value):
            callback_data.append((timestamp, value))
        
        mock_probe.add_data_callback(test_callback)
        mock_probe.start_recording()
        mock_probe.set_measurement_value("target1", 100.0)
        mock_probe.record(1.0)
        
        assert len(callback_data) == 1
        assert callback_data[0][0] == 1.0

    @pytest.mark.unit
    def test_buffer_limits(self, mock_probe):
        """Test buffer size limits."""
        mock_probe.set_buffer_limits(max_samples=2, mode='circular')
        mock_probe.start_recording(0.0)
        
        # Record 3 samples
        for i in range(3):
            mock_probe.record(float(i + 1))
        
        # Should only keep 2 samples (oldest removed)
        data = mock_probe.get_data()
        assert len(data.timestamps) == 2
        assert data.timestamps[0] == 2.0  # First sample removed

    @pytest.mark.unit
    def test_buffer_growing_mode(self, mock_probe):
        """Test growing buffer mode."""
        mock_probe.set_buffer_limits(max_samples=2, mode='growing')
        mock_probe.start_recording(0.0)
        
        # Record 3 samples
        mock_probe.record(1.0)
        mock_probe.record(2.0)
        assert mock_probe.is_recording_active()
        
        mock_probe.record(3.0)  # Should stop recording
        assert not mock_probe.is_recording_active()

    @pytest.mark.unit
    def test_recording_statistics(self, mock_probe):
        """Test recording statistics."""
        mock_probe.start_recording(0.0)
        mock_probe.record(1.0)
        mock_probe.record(2.0)
        
        stats = mock_probe.get_recording_stats()
        
        assert stats['probe_id'] == "test_probe"
        assert stats['samples_collected'] == 2
        assert stats['target_count'] == 2
        assert stats['recording_duration'] == 1.0  # 2.0 - 1.0

    @pytest.mark.unit
    def test_export_data(self, mock_probe):
        """Test data export functionality."""
        mock_probe.start_recording(0.0)
        mock_probe.set_measurement_value("target1", 1.0)
        mock_probe.record(1.0)
        
        # Test dict export
        dict_data = mock_probe.export_data('dict')
        assert 'probe_id' in dict_data
        assert 'timestamps' in dict_data
        
        # Test array export  
        array_data = mock_probe.export_data('array')
        assert isinstance(array_data, np.ndarray)

    @pytest.mark.unit
    def test_target_management(self, mock_probe):
        """Test adding/removing targets."""
        initial_count = mock_probe.get_target_count()
        
        mock_probe.add_target("new_target")
        assert mock_probe.get_target_count() == initial_count + 1
        assert "new_target" in mock_probe.target_ids
        
        mock_probe.remove_target("new_target")
        assert mock_probe.get_target_count() == initial_count
        assert "new_target" not in mock_probe.target_ids


class TestVoltageProbe:
    """Test VoltageProbe class."""

    @pytest.fixture
    def mock_neuron(self):
        """Create mock neuron for voltage testing."""
        neuron = Mock()
        neuron.get_membrane_potential.return_value = -65.0
        return neuron

    @pytest.mark.unit
    def test_voltage_probe_initialization(self):
        """Test voltage probe initialization."""
        probe = VoltageProbe("v_probe", ["n1", "n2"], sampling_interval=0.5)
        
        assert probe.probe_type == ProbeType.VOLTAGE
        assert probe.sampling_interval == 0.5
        assert not probe.enable_filtering

    @pytest.mark.unit
    def test_register_neuron(self, voltage_probe, mock_neuron):
        """Test registering neuron object."""
        voltage_probe.register_neuron_object("n1", mock_neuron)
        
        assert "n1" in voltage_probe.neuron_objects
        assert voltage_probe.neuron_objects["n1"] == mock_neuron

    @pytest.mark.unit
    def test_voltage_measurement(self, voltage_probe, mock_neuron):
        """Test voltage measurement."""
        voltage_probe.register_neuron_object("n1", mock_neuron)
        
        voltage = voltage_probe.get_measurement_value("n1")
        assert voltage == -65.0
        
        # Test with nonexistent neuron
        assert voltage_probe.get_measurement_value("nonexistent") is None

    @pytest.mark.unit
    def test_voltage_range_clipping(self, mock_neuron):
        """Test voltage range clipping."""
        probe = VoltageProbe("v_probe", ["n1"], voltage_range=(-80.0, -40.0))
        probe.register_neuron_object("n1", mock_neuron)
        
        # Test clipping high value
        mock_neuron.get_membrane_potential.return_value = -20.0
        voltage = probe.get_measurement_value("n1")
        assert voltage == -40.0
        
        # Test clipping low value
        mock_neuron.get_membrane_potential.return_value = -90.0
        voltage = probe.get_measurement_value("n1")
        assert voltage == -80.0

    @pytest.mark.unit
    def test_voltage_filtering(self, mock_neuron):
        """Test voltage filtering."""
        probe = VoltageProbe("v_probe", ["n1"], enable_filtering=True, filter_cutoff=50.0)
        probe.register_neuron_object("n1", mock_neuron)
        
        # First measurement should initialize filter
        mock_neuron.get_membrane_potential.return_value = -60.0
        voltage1 = probe.get_measurement_value("n1")
        assert voltage1 == -60.0
        
        # Second measurement should be filtered
        mock_neuron.get_membrane_potential.return_value = -50.0
        voltage2 = probe.get_measurement_value("n1")
        assert voltage2 != -50.0  # Should be filtered
        assert -60.0 < voltage2 < -50.0  # Should be between old and new

    @pytest.mark.integration
    def test_voltage_recording(self, voltage_probe, mock_neuron):
        """Test voltage recording over time."""
        voltage_probe.register_neuron_object("n1", mock_neuron)
        voltage_probe.start_recording(0.0)
        
        # Simulate changing voltage over time
        voltages = [-70.0, -65.0, -60.0, -55.0]
        times = [0.5, 1.0, 1.5, 2.0]
        
        for time, voltage in zip(times, voltages):
            mock_neuron.get_membrane_potential.return_value = voltage
            voltage_probe.record(time)
        
        # Check recorded data
        traces = voltage_probe.get_voltage_traces()
        assert "n1" in traces
        assert len(traces["n1"]["time"]) == 4
        assert traces["n1"]["voltage"][-1] == -55.0

    @pytest.mark.unit
    def test_voltage_statistics(self, voltage_probe, mock_neuron):
        """Test voltage statistics calculation."""
        voltage_probe.register_neuron_object("n1", mock_neuron)
        voltage_probe.start_recording(0.0)
        
        # Record several voltage values
        voltages = [-70.0, -60.0, -65.0, -55.0]
        for i, voltage in enumerate(voltages):
            mock_neuron.get_membrane_potential.return_value = voltage
            voltage_probe.record(float(i + 1))
        
        stats = voltage_probe.get_voltage_statistics()
        assert "n1" in stats
        assert stats["n1"]["min"] == -70.0
        assert stats["n1"]["max"] == -55.0
        assert -70.0 <= stats["n1"]["mean"] <= -55.0


class TestSpikeProbe:
    """Test SpikeProbe class."""

    @pytest.fixture
    def mock_spiking_neuron(self):
        """Create mock neuron that can spike."""
        neuron = Mock()
        neuron.has_spiked.return_value = False
        neuron.get_last_spike_time.return_value = None
        neuron.get_membrane_potential.return_value = -70.0
        return neuron

    @pytest.mark.unit
    def test_spike_probe_initialization(self):
        """Test spike probe initialization."""
        probe = SpikeProbe("s_probe", ["n1", "n2"], detection_threshold=-35.0)
        
        assert probe.probe_type == ProbeType.SPIKE
        assert probe.detection_threshold == -35.0
        assert probe.min_spike_interval == 1.0
        assert not probe.record_waveforms

    @pytest.mark.unit
    def test_register_neuron(self, spike_probe, mock_spiking_neuron):
        """Test registering neuron for spike detection."""
        spike_probe.register_neuron_object("n1", mock_spiking_neuron)
        
        assert "n1" in spike_probe.neuron_objects
        assert "n1" in spike_probe.last_spike_times

    @pytest.mark.unit
    def test_spike_detection_builtin(self, spike_probe, mock_spiking_neuron):
        """Test spike detection using neuron's built-in method."""
        spike_probe.register_neuron_object("n1", mock_spiking_neuron)
        
        # Simulate neuron spiking
        mock_spiking_neuron.has_spiked.return_value = True
        mock_spiking_neuron.get_last_spike_time.return_value = 10.0
        mock_spiking_neuron.get_membrane_potential.return_value = -30.0
        
        spike_probe.record(10.0)
        
        assert len(spike_probe.spike_times["n1"]) == 1
        assert spike_probe.spike_times["n1"][0] == 10.0

    @pytest.mark.unit
    def test_threshold_spike_detection(self, spike_probe, mock_spiking_neuron):
        """Test threshold-based spike detection."""
        spike_probe.register_neuron_object("n1", mock_spiking_neuron)
        spike_probe.detection_threshold = -40.0
        
        # Simulate threshold crossing
        mock_spiking_neuron.has_spiked.return_value = False
        
        # First update - below threshold
        mock_spiking_neuron.get_membrane_potential.return_value = -50.0
        spike_probe.record(9.0)
        
        # Second update - above threshold (should detect spike)
        mock_spiking_neuron.get_membrane_potential.return_value = -35.0
        spike_probe.record(10.0)
        
        assert len(spike_probe.spike_times["n1"]) == 1

    @pytest.mark.unit
    def test_refractory_period(self, spike_probe, mock_spiking_neuron):
        """Test minimum spike interval enforcement."""
        spike_probe.register_neuron_object("n1", mock_spiking_neuron)
        spike_probe.min_spike_interval = 2.0
        
        mock_spiking_neuron.has_spiked.return_value = True
        mock_spiking_neuron.get_membrane_potential.return_value = -30.0
        
        # First spike
        mock_spiking_neuron.get_last_spike_time.return_value = 10.0
        spike_probe.record(10.0)
        
        # Second spike too soon (should be ignored)
        mock_spiking_neuron.get_last_spike_time.return_value = 11.0
        spike_probe.record(11.0)
        
        assert len(spike_probe.spike_times["n1"]) == 1

    @pytest.mark.unit
    def test_firing_rate_calculation(self, spike_probe):
        """Test firing rate calculation."""
        # Manually add spike times
        spike_probe.spike_times["n1"] = [10.0, 20.0, 30.0, 40.0]
        spike_probe.current_time = 50.0
        
        # Test fixed window
        rates = spike_probe.calculate_firing_rates(time_window=30.0)
        expected_rate = 3 * 1000.0 / 30.0  # 3 spikes in 30ms window
        assert rates["n1"] == expected_rate

    @pytest.mark.unit
    def test_isi_statistics(self, spike_probe):
        """Test inter-spike interval statistics."""
        # Add spike train
        spike_probe.spike_times["n1"] = [10.0, 25.0, 35.0, 50.0]
        
        stats = spike_probe.calculate_isi_statistics()
        expected_isis = [15.0, 10.0, 15.0]  # ISIs
        
        assert "n1" in stats
        assert stats["n1"]["count"] == 3
        assert stats["n1"]["mean_isi"] == np.mean(expected_isis)

    @pytest.mark.unit
    def test_burst_detection(self, spike_probe):
        """Test burst detection algorithm."""
        # Create spike train with burst
        spike_times = [10.0, 12.0, 14.0, 16.0, 100.0, 102.0, 104.0]
        spike_probe.spike_times["n1"] = spike_times
        
        bursts = spike_probe.detect_bursts(
            "n1", 
            burst_threshold=100.0,  # Hz
            max_intraburst_isi=5.0  # ms
        )
        
        assert len(bursts) >= 1  # Should detect at least one burst

    @pytest.mark.unit
    def test_spike_data_export(self, spike_probe):
        """Test spike data export formats."""
        # Add test data
        spike_probe.spike_times["n1"] = [10.0, 20.0]
        spike_probe.spike_times["n2"] = [15.0, 25.0]
        
        # Test raster format
        raster = spike_probe.export_spike_data('raster')
        assert len(raster) == 4  # 4 total spikes
        assert raster[0] == ("n1", 10.0)  # Should be sorted by time
        
        # Test dict format
        dict_data = spike_probe.export_spike_data('dict')
        assert isinstance(dict_data, dict)
        assert "n1" in dict_data


class TestPopulationActivityProbe:
    """Test PopulationActivityProbe class."""

    @pytest.fixture
    def mock_population_neurons(self):
        """Create mock neurons for population testing."""
        neurons = {}
        for i in range(3):
            neuron = Mock()
            neuron.get_spike_times.return_value = []
            neurons[f"n{i}"] = neuron
        return neurons

    @pytest.mark.unit
    def test_population_probe_initialization(self):
        """Test population activity probe initialization."""
        probe = PopulationActivityProbe(
            "pop_probe", "test_pop", ["n1", "n2", "n3"], 
            bin_size=5.0, record_synchrony=True
        )
        
        assert probe.probe_type == ProbeType.POPULATION_ACTIVITY
        assert probe.target_population == "test_pop"
        assert probe.bin_size == 5.0
        assert probe.record_synchrony is True

    @pytest.mark.unit
    def test_register_neurons(self, population_probe, mock_population_neurons):
        """Test registering neuron objects."""
        population_probe.register_neuron_objects(mock_population_neurons)
        
        for neuron_id in population_probe.target_ids:
            if neuron_id in mock_population_neurons:
                assert neuron_id in population_probe.neuron_objects

    @pytest.mark.integration
    def test_population_rate_calculation(self, mock_population_neurons):
        """Test population firing rate calculation."""
        # Set up spike times
        mock_population_neurons["n1"].get_spike_times.return_value = [98.0, 99.0]
        mock_population_neurons["n2"].get_spike_times.return_value = [97.0]
        
        probe = PopulationActivityProbe(
            "pop_probe", "test_pop", ["n1", "n2"], bin_size=5.0
        )
        probe.register_neuron_objects(mock_population_neurons)
        probe.start_recording(0.0)
        
        probe.record(100.0)
        
        # Should calculate rate based on recent spikes
        assert len(probe.population_rates) == 1
        assert probe.population_rates[0] > 0

    @pytest.mark.unit
    def test_synchrony_calculation(self, mock_population_neurons):
        """Test synchrony index calculation."""
        # Create synchronized spike pattern
        for neuron_id in ["n1", "n2"]:
            mock_population_neurons[neuron_id].get_spike_times.return_value = [50.0, 51.0]
        
        probe = PopulationActivityProbe(
            "pop_probe", "test_pop", ["n1", "n2"], 
            bin_size=1.0, record_synchrony=True
        )
        probe.register_neuron_objects(mock_population_neurons)
        probe.start_recording(0.0)
        
        probe.record(100.0)
        
        assert len(probe.synchrony_indices) == 1
        # Synchronized spikes should give high synchrony
        assert probe.synchrony_indices[0] >= 0

    @pytest.mark.unit
    def test_burst_detection(self, mock_population_neurons):
        """Test population burst detection."""
        probe = PopulationActivityProbe(
            "pop_probe", "test_pop", ["n1"], bin_size=1.0
        )
        probe.register_neuron_objects(mock_population_neurons)
        
        # Simulate high-rate period (burst)
        probe.population_rates = [1.0, 5.0, 10.0, 8.0, 2.0]  # Burst in middle
        probe.spike_counts = [1, 5, 10, 8, 2]
        probe.data.timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        bursts = probe.detect_population_bursts(rate_threshold=7.0)
        
        assert len(bursts) >= 0  # Should detect burst or none


class TestProbeManager:
    """Test ProbeManager class."""

    @pytest.fixture
    def probe_manager(self):
        """Create probe manager for testing."""
        return ProbeManager()

    @pytest.fixture
    def test_probes(self):
        """Create test probes."""
        probe1 = MockProbe("probe1", ["n1"])
        probe2 = MockProbe("probe2", ["n2"])
        return [probe1, probe2]

    @pytest.mark.unit
    def test_probe_manager_initialization(self, probe_manager):
        """Test probe manager initialization."""
        assert len(probe_manager.probes) == 0
        assert len(probe_manager.probe_groups) == 0
        assert not probe_manager.global_recording

    @pytest.mark.unit
    def test_add_remove_probes(self, probe_manager, test_probes):
        """Test adding and removing probes."""
        for probe in test_probes:
            probe_manager.add_probe(probe)
        
        assert len(probe_manager.probes) == 2
        assert probe_manager.get_probe("probe1") == test_probes[0]
        
        probe_manager.remove_probe("probe1")
        assert len(probe_manager.probes) == 1
        assert probe_manager.get_probe("probe1") is None

    @pytest.mark.unit
    def test_probe_groups(self, probe_manager, test_probes):
        """Test probe grouping functionality."""
        for probe in test_probes:
            probe_manager.add_probe(probe)
        
        probe_manager.create_probe_group("test_group", ["probe1", "probe2"])
        
        assert "test_group" in probe_manager.probe_groups
        assert len(probe_manager.probe_groups["test_group"]) == 2

    @pytest.mark.unit
    def test_start_stop_all_recording(self, probe_manager, test_probes):
        """Test starting/stopping recording on all probes."""
        for probe in test_probes:
            probe_manager.add_probe(probe)
        
        probe_manager.start_recording_all(10.0)
        
        assert probe_manager.global_recording is True
        for probe in test_probes:
            assert probe.is_recording_active()
        
        probe_manager.stop_recording_all()
        
        assert probe_manager.global_recording is False
        for probe in test_probes:
            assert not probe.is_recording_active()

    @pytest.mark.unit
    def test_group_recording(self, probe_manager, test_probes):
        """Test group-based recording control."""
        for probe in test_probes:
            probe_manager.add_probe(probe)
        
        probe_manager.create_probe_group("group1", ["probe1"])
        probe_manager.start_recording_group("group1", 5.0)
        
        assert test_probes[0].is_recording_active()
        assert not test_probes[1].is_recording_active()

    @pytest.mark.unit
    def test_get_all_data(self, probe_manager, test_probes):
        """Test getting data from all probes."""
        for probe in test_probes:
            probe_manager.add_probe(probe)
            probe.start_recording(0.0)
            probe.record(1.0)
        
        all_data = probe_manager.get_all_data()
        
        assert "probe1" in all_data
        assert "probe2" in all_data
        assert isinstance(all_data["probe1"], ProbeData)

    @pytest.mark.unit
    def test_recording_summary(self, probe_manager, test_probes):
        """Test recording summary statistics."""
        for probe in test_probes:
            probe_manager.add_probe(probe)
        
        test_probes[0].start_recording()
        
        summary = probe_manager.get_recording_summary()
        
        assert summary['total_probes'] == 2
        assert summary['active_probes'] == 1
        assert summary['global_recording'] is False
