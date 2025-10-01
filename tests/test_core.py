"""Tests for the core simulation engine components."""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch

from sdmn.core import (
    SimulationEngine,
    SimulationConfig,
    SimulationResults,
    Event,
    EventQueue,
    EventType,
    EventHandler,
    EventProcessor,
    StateManager,
    StateCheckpoint,
    TimeManager,
    TimeStep,
)


class TestSimulationConfig:
    """Test SimulationConfig class."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()
        
        assert config.dt == 0.1
        assert config.max_time == 1000.0
        assert config.checkpoint_interval == 1000
        assert config.enable_logging is True
        assert config.log_level == "INFO"
        assert config.random_seed is None

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SimulationConfig(
            dt=0.01,
            max_time=500.0,
            enable_logging=False,
            random_seed=123
        )
        
        assert config.dt == 0.01
        assert config.max_time == 500.0
        assert config.enable_logging is False
        assert config.random_seed == 123


class TestEvent:
    """Test Event class."""

    @pytest.mark.unit
    def test_event_creation(self):
        """Test event creation and properties."""
        event = Event(
            timestamp=10.0,
            event_type=EventType.SPIKE,
            source_id="neuron_1",
            target_id="neuron_2",
            data={'amplitude': 1.0}
        )
        
        assert event.timestamp == 10.0
        assert event.event_type == EventType.SPIKE
        assert event.source_id == "neuron_1" 
        assert event.target_id == "neuron_2"
        assert event.data['amplitude'] == 1.0

    @pytest.mark.unit
    def test_event_ordering(self):
        """Test event ordering for priority queue."""
        event1 = Event(5.0, EventType.SPIKE, "n1")
        event2 = Event(3.0, EventType.SPIKE, "n2")
        event3 = Event(5.0, EventType.STIMULUS, "n3")
        
        assert event2 < event1  # Earlier time
        assert not event1 < event3  # Same time
        assert event1 == event3  # Same time and other properties

    @pytest.mark.unit
    def test_event_equality(self):
        """Test event equality comparison."""
        event1 = Event(5.0, EventType.SPIKE, "n1", "n2")
        event2 = Event(5.0, EventType.SPIKE, "n1", "n2")
        event3 = Event(5.0, EventType.SPIKE, "n1", "n3")
        
        assert event1 == event2
        assert event1 != event3


class TestEventQueue:
    """Test EventQueue class."""

    @pytest.fixture
    def event_queue(self):
        return EventQueue()

    @pytest.mark.unit
    def test_empty_queue(self, event_queue):
        """Test empty queue properties."""
        assert event_queue.is_empty()
        assert event_queue.size() == 0
        assert event_queue.get_next_event_time() is None

    @pytest.mark.unit
    def test_schedule_single_event(self, event_queue):
        """Test scheduling single event."""
        event = Event(10.0, EventType.SPIKE, "n1")
        event_queue.schedule_event(event)
        
        assert not event_queue.is_empty()
        assert event_queue.size() == 1
        assert event_queue.get_next_event_time() == 10.0

    @pytest.mark.unit
    def test_schedule_multiple_events(self, event_queue):
        """Test scheduling multiple events."""
        events = [
            Event(15.0, EventType.SPIKE, "n1"),
            Event(5.0, EventType.STIMULUS, "n2"),
            Event(10.0, EventType.SPIKE, "n3")
        ]
        event_queue.schedule_events(events)
        
        assert event_queue.size() == 3
        assert event_queue.get_next_event_time() == 5.0

    @pytest.mark.unit
    def test_get_events_at_time(self, event_queue):
        """Test retrieving events at specific time."""
        events = [
            Event(10.0, EventType.SPIKE, "n1"),
            Event(10.0, EventType.SPIKE, "n2"),
            Event(15.0, EventType.STIMULUS, "n3")
        ]
        event_queue.schedule_events(events)
        
        events_at_10 = event_queue.get_events_at_time(10.0)
        assert len(events_at_10) == 2
        assert event_queue.size() == 1  # One event should remain

    @pytest.mark.unit
    def test_peek_events_at_time(self, event_queue):
        """Test peeking at events without removing them."""
        events = [
            Event(10.0, EventType.SPIKE, "n1"),
            Event(10.0, EventType.SPIKE, "n2"),
        ]
        event_queue.schedule_events(events)
        
        peeked_events = event_queue.peek_events_at_time(10.0)
        assert len(peeked_events) == 2
        assert event_queue.size() == 2  # Events should still be there

    @pytest.mark.unit
    def test_clear_queue(self, event_queue):
        """Test clearing the queue."""
        event = Event(10.0, EventType.SPIKE, "n1")
        event_queue.schedule_event(event)
        
        event_queue.clear()
        assert event_queue.is_empty()
        assert event_queue.size() == 0


class MockEventHandler(EventHandler):
    """Mock event handler for testing."""
    
    def __init__(self):
        self.handled_events = []
    
    def handle_event(self, event):
        self.handled_events.append(event)
        return []  # Return no new events


class TestEventProcessor:
    """Test EventProcessor class."""

    @pytest.fixture
    def event_processor(self):
        return EventProcessor()

    @pytest.fixture
    def mock_handler(self):
        return MockEventHandler()

    @pytest.mark.unit
    def test_register_handler(self, event_processor, mock_handler):
        """Test registering event handler."""
        event_processor.register_handler(EventType.SPIKE, mock_handler)
        
        # Process an event to verify handler is registered
        event = Event(10.0, EventType.SPIKE, "n1")
        event_processor.process_event(event)
        
        assert len(mock_handler.handled_events) == 1
        assert mock_handler.handled_events[0] == event

    @pytest.mark.unit
    def test_unregister_handler(self, event_processor, mock_handler):
        """Test unregistering event handler."""
        event_processor.register_handler(EventType.SPIKE, mock_handler)
        event_processor.unregister_handler(EventType.SPIKE, mock_handler)
        
        event = Event(10.0, EventType.SPIKE, "n1")
        event_processor.process_event(event)
        
        assert len(mock_handler.handled_events) == 0

    @pytest.mark.unit
    def test_process_multiple_events(self, event_processor, mock_handler):
        """Test processing multiple events."""
        event_processor.register_handler(EventType.SPIKE, mock_handler)
        
        events = [
            Event(10.0, EventType.SPIKE, "n1"),
            Event(15.0, EventType.SPIKE, "n2")
        ]
        
        new_events = event_processor.process_events(events)
        
        assert len(mock_handler.handled_events) == 2
        assert len(new_events) == 0


class TestTimeManager:
    """Test TimeManager class."""

    @pytest.fixture
    def time_manager(self):
        return TimeManager(dt=0.1, max_time=100.0)

    @pytest.mark.unit
    def test_initial_state(self, time_manager):
        """Test initial state of time manager."""
        assert time_manager.dt == 0.1
        assert time_manager.max_time == 100.0
        assert time_manager.current_time == 0.0
        assert time_manager.current_step == 0

    @pytest.mark.unit
    def test_start_simulation(self, time_manager):
        """Test starting simulation."""
        time_manager.start_simulation()
        assert time_manager.current_time == 0.0
        assert time_manager.current_step == 0
        assert time_manager.start_wall_time > 0

    @pytest.mark.unit
    def test_advance_time(self, time_manager):
        """Test advancing time."""
        time_manager.start_simulation()
        
        time_step = time_manager.advance_time()
        
        assert isinstance(time_step, TimeStep)
        assert time_step.step_number == 0
        assert time_step.simulation_time == 0.0
        assert time_step.dt == 0.1
        assert time_manager.current_time == 0.1
        assert time_manager.current_step == 1

    @pytest.mark.unit
    def test_simulation_progress(self, time_manager):
        """Test simulation progress tracking."""
        time_manager.start_simulation()
        
        assert time_manager.get_simulation_progress() == 0.0
        
        # Advance to 50% completion
        while time_manager.current_time < 50.0:
            time_manager.advance_time()
        
        progress = time_manager.get_simulation_progress()
        assert 0.49 <= progress <= 0.51  # Allow small numerical error

    @pytest.mark.unit
    def test_is_simulation_complete(self, time_manager):
        """Test simulation completion detection."""
        time_manager.start_simulation()
        
        assert not time_manager.is_simulation_complete()
        
        # Advance past max time
        time_manager.current_time = 100.1
        assert time_manager.is_simulation_complete()

    @pytest.mark.unit
    def test_performance_stats(self, time_manager):
        """Test performance statistics."""
        time_manager.start_simulation()
        
        # Advance a few steps
        for _ in range(10):
            time_manager.advance_time()
            time.sleep(0.001)  # Small delay to measure wall time
        
        stats = time_manager.get_performance_stats()
        
        assert 'total_steps' in stats
        assert 'simulation_time' in stats
        assert 'wall_time' in stats
        assert stats['total_steps'] == 10
        assert stats['simulation_time'] == 1.0

    @pytest.mark.unit
    def test_reset(self, time_manager):
        """Test resetting time manager."""
        time_manager.start_simulation()
        time_manager.advance_time()
        
        time_manager.reset()
        
        assert time_manager.current_time == 0.0
        assert time_manager.current_step == 0
        assert len(time_manager.time_history) == 0


class MockStateSerializable:
    """Mock state serializable object for testing."""
    
    def __init__(self, initial_state=None):
        self.state_data = initial_state or {'value': 42}
    
    def get_state(self):
        return self.state_data.copy()
    
    def set_state(self, state):
        self.state_data = state.copy()
    
    def get_state_version(self):
        return "1.0"


class TestStateManager:
    """Test StateManager class."""

    @pytest.fixture
    def state_manager(self, tmp_path):
        return StateManager(checkpoint_dir=tmp_path)

    @pytest.fixture
    def mock_object(self):
        return MockStateSerializable({'test': 'data', 'number': 123})

    @pytest.mark.unit
    def test_register_object(self, state_manager, mock_object):
        """Test registering object for state management."""
        state_manager.register_object("test_obj", mock_object)
        
        state = state_manager.get_object_state("test_obj")
        assert state['test'] == 'data'
        assert state['number'] == 123

    @pytest.mark.unit
    def test_set_object_state(self, state_manager, mock_object):
        """Test setting object state."""
        state_manager.register_object("test_obj", mock_object)
        
        new_state = {'test': 'updated', 'number': 456}
        success = state_manager.set_object_state("test_obj", new_state)
        
        assert success is True
        retrieved_state = state_manager.get_object_state("test_obj")
        assert retrieved_state['test'] == 'updated'
        assert retrieved_state['number'] == 456

    @pytest.mark.unit
    def test_get_full_state(self, state_manager):
        """Test getting full state of all objects."""
        obj1 = MockStateSerializable({'id': 1})
        obj2 = MockStateSerializable({'id': 2})
        
        state_manager.register_object("obj1", obj1)
        state_manager.register_object("obj2", obj2)
        
        full_state = state_manager.get_full_state()
        
        assert 'obj1' in full_state
        assert 'obj2' in full_state
        assert full_state['obj1']['id'] == 1
        assert full_state['obj2']['id'] == 2

    @pytest.mark.unit
    def test_create_checkpoint(self, state_manager, mock_object):
        """Test creating state checkpoint."""
        state_manager.register_object("test_obj", mock_object)
        
        checkpoint = state_manager.create_checkpoint(
            timestamp=10.0,
            step_number=100,
            metadata={'test': True}
        )
        
        assert isinstance(checkpoint, StateCheckpoint)
        assert checkpoint.timestamp == 10.0
        assert checkpoint.step_number == 100
        assert checkpoint.metadata['test'] is True
        assert len(checkpoint.state_hash) == 64  # SHA256 hash

    @pytest.mark.unit
    def test_restore_checkpoint(self, state_manager, mock_object, tmp_path):
        """Test restoring from checkpoint."""
        state_manager.register_object("test_obj", mock_object)
        
        # Modify object state
        mock_object.state_data['modified'] = True
        
        # Create checkpoint
        checkpoint = state_manager.create_checkpoint(10.0, 100)
        
        # Modify state again
        mock_object.state_data['modified'] = False
        mock_object.state_data['extra'] = 'value'
        
        # Restore checkpoint
        success = state_manager.restore_checkpoint(checkpoint)
        
        assert success is True
        current_state = state_manager.get_object_state("test_obj")
        assert 'extra' not in current_state  # Should be reverted

    @pytest.mark.unit
    def test_find_checkpoint_by_step(self, state_manager, mock_object):
        """Test finding checkpoint by step number."""
        state_manager.register_object("test_obj", mock_object)
        
        checkpoint1 = state_manager.create_checkpoint(10.0, 100)
        checkpoint2 = state_manager.create_checkpoint(20.0, 200)
        
        found = state_manager.find_checkpoint_by_step(100)
        assert found == checkpoint1
        
        found = state_manager.find_checkpoint_by_step(200)
        assert found == checkpoint2
        
        found = state_manager.find_checkpoint_by_step(300)
        assert found is None


class TestSimulationEngine:
    """Test SimulationEngine class."""

    @pytest.mark.unit
    def test_engine_initialization(self, basic_sim_config):
        """Test simulation engine initialization."""
        engine = SimulationEngine(basic_sim_config)
        
        assert engine.config == basic_sim_config
        assert hasattr(engine, 'time_manager')
        assert hasattr(engine, 'event_queue')
        assert hasattr(engine, 'state_manager')
        assert engine.is_running is False
        assert engine.is_paused is False

    @pytest.mark.unit
    def test_add_network(self, simulation_engine):
        """Test adding network to engine."""
        mock_network = Mock()
        mock_network.get_state = Mock(return_value={'net': 'data'})
        mock_network.set_state = Mock()
        
        simulation_engine.add_network("test_net", mock_network)
        
        assert "test_net" in simulation_engine.networks
        assert simulation_engine.networks["test_net"] == mock_network

    @pytest.mark.unit
    def test_pause_resume_stop(self, simulation_engine):
        """Test pause/resume/stop functionality."""
        simulation_engine.pause()
        assert simulation_engine.is_paused is True
        
        simulation_engine.resume()
        assert simulation_engine.is_paused is False
        
        simulation_engine.stop()
        assert simulation_engine.is_running is False

    @pytest.mark.integration
    def test_simple_simulation_run(self, simulation_engine):
        """Test running a simple simulation."""
        # Create a mock network
        mock_network = Mock()
        mock_network.update = Mock()
        
        simulation_engine.add_network("test_net", mock_network)
        
        # Run short simulation
        results = simulation_engine.run()
        
        assert isinstance(results, SimulationResults)
        assert results.success is True
        assert results.total_steps > 0
        assert results.simulation_time >= 0
        assert results.wall_time >= 0

    @pytest.mark.unit
    def test_get_status(self, simulation_engine):
        """Test getting simulation status."""
        status = simulation_engine.get_status()
        
        assert 'is_running' in status
        assert 'is_paused' in status
        assert 'current_step' in status
        assert 'current_time' in status
        assert 'progress' in status
