"""
Core simulation engine for the Synthetic Default Mode Network Framework.

This module implements the main simulation engine that coordinates
time management, event processing, state management, and component
integration for neural network simulations.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path

from sdmn.core.time_manager import TimeManager, TimeStep
from sdmn.core.event_system import EventQueue, EventProcessor, Event, EventType
from sdmn.core.state_manager import StateManager, StateCheckpoint


@dataclass
class SimulationConfig:
    """Configuration parameters for simulation."""
    dt: float = 0.1                    # Time step in milliseconds
    max_time: float = 1000.0           # Maximum simulation time in milliseconds  
    checkpoint_interval: int = 1000    # Steps between automatic checkpoints
    enable_logging: bool = True        # Enable simulation logging
    log_level: str = "INFO"           # Logging level
    checkpoint_dir: Optional[Path] = None  # Directory for checkpoints
    random_seed: Optional[int] = None  # Random seed for reproducibility
    max_events_per_step: int = 10000  # Maximum events to process per step


@dataclass
class SimulationResults:
    """Contains results and metadata from a completed simulation."""
    success: bool
    total_steps: int
    simulation_time: float
    wall_time: float
    final_state: Dict[str, Any]
    performance_stats: Dict[str, Any]
    error_message: Optional[str] = None
    checkpoints: List[StateCheckpoint] = field(default_factory=list)


class SimulationEngine:
    """
    Main simulation engine for neural network simulations.
    
    Coordinates all aspects of simulation execution including:
    - Time management and stepping
    - Event processing and scheduling
    - Component state management
    - Checkpointing and recovery
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation engine.
        
        Args:
            config: Simulation configuration (uses defaults if None)
        """
        self.config = config or SimulationConfig()
        
        # Initialize core components
        self.time_manager = TimeManager(
            dt=self.config.dt,
            max_time=self.config.max_time
        )
        
        self.event_queue = EventQueue()
        self.event_processor = EventProcessor()
        self.state_manager = StateManager(self.config.checkpoint_dir)
        
        # Component registries
        self.networks: Dict[str, Any] = {}
        self.probes: Dict[str, Any] = {}
        self.stimuli: Dict[str, Any] = {}
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.last_checkpoint_step = 0
        
        # Callbacks
        self.step_callbacks: List[Callable[[int, float], None]] = []
        self.start_callbacks: List[Callable[[], None]] = []
        self.end_callbacks: List[Callable[[SimulationResults], None]] = []
        
        # Setup logging
        self._setup_logging()
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            import numpy as np
            np.random.seed(self.config.random_seed)
        
        self.logger.info("Simulation engine initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger("SDMN.SimulationEngine")
        
        if self.config.enable_logging:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, self.config.log_level))
    
    def add_network(self, network_id: str, network: Any) -> None:
        """
        Add a neural network to the simulation.
        
        Args:
            network_id: Unique identifier for the network
            network: The network object to add
        """
        self.networks[network_id] = network
        
        # Register with state manager if it supports state management
        if hasattr(network, 'get_state') and hasattr(network, 'set_state'):
            self.state_manager.register_object(f"network_{network_id}", network)
        
        self.logger.info(f"Added network: {network_id}")
    
    def add_probe(self, probe_id: str, probe: Any) -> None:
        """
        Add a monitoring probe to the simulation.
        
        Args:
            probe_id: Unique identifier for the probe
            probe: The probe object to add
        """
        self.probes[probe_id] = probe
        
        # Register with state manager if it supports state management
        if hasattr(probe, 'get_state') and hasattr(probe, 'set_state'):
            self.state_manager.register_object(f"probe_{probe_id}", probe)
        
        self.logger.info(f"Added probe: {probe_id}")
    
    def add_stimulus(self, stimulus_id: str, stimulus: Any) -> None:
        """
        Add a stimulus source to the simulation.
        
        Args:
            stimulus_id: Unique identifier for the stimulus
            stimulus: The stimulus object to add
        """
        self.stimuli[stimulus_id] = stimulus
        
        # Register with state manager if it supports state management
        if hasattr(stimulus, 'get_state') and hasattr(stimulus, 'set_state'):
            self.state_manager.register_object(f"stimulus_{stimulus_id}", stimulus)
        
        self.logger.info(f"Added stimulus: {stimulus_id}")
    
    def schedule_event(self, event: Event) -> None:
        """
        Schedule an event for future processing.
        
        Args:
            event: The event to schedule
        """
        self.event_queue.schedule_event(event)
    
    def register_step_callback(self, callback: Callable[[int, float], None]) -> None:
        """Register a callback to be called after each simulation step."""
        self.step_callbacks.append(callback)
    
    def register_start_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when simulation starts."""
        self.start_callbacks.append(callback)
    
    def register_end_callback(self, callback: Callable[[SimulationResults], None]) -> None:
        """Register a callback to be called when simulation ends."""
        self.end_callbacks.append(callback)
    
    def run(self) -> SimulationResults:
        """
        Execute the complete simulation.
        
        Returns:
            SimulationResults object containing execution results
        """
        self.logger.info("Starting simulation")
        start_wall_time = time.time()
        
        try:
            # Initialize simulation
            self.is_running = True
            self.time_manager.start_simulation()
            
            # Call start callbacks
            for callback in self.start_callbacks:
                callback()
            
            # Main simulation loop
            while not self.time_manager.is_simulation_complete() and self.is_running:
                if not self.is_paused:
                    self._execute_step()
                    
                    # Create checkpoint if needed
                    if (self.time_manager.current_step > 0 and 
                        self.time_manager.current_step % self.config.checkpoint_interval == 0):
                        self._create_automatic_checkpoint()
                else:
                    time.sleep(0.001)  # Small delay when paused
            
            # Simulation completed successfully
            end_wall_time = time.time()
            results = SimulationResults(
                success=True,
                total_steps=self.time_manager.current_step,
                simulation_time=self.time_manager.current_time,
                wall_time=end_wall_time - start_wall_time,
                final_state=self.state_manager.get_full_state(),
                performance_stats=self.time_manager.get_performance_stats(),
                checkpoints=self.state_manager.get_checkpoint_history()
            )
            
            self.logger.info(f"Simulation completed successfully after {results.total_steps} steps")
            
        except Exception as e:
            # Simulation failed
            end_wall_time = time.time()
            self.logger.error(f"Simulation failed: {str(e)}")
            
            results = SimulationResults(
                success=False,
                total_steps=self.time_manager.current_step,
                simulation_time=self.time_manager.current_time,
                wall_time=end_wall_time - start_wall_time,
                final_state={},
                performance_stats=self.time_manager.get_performance_stats(),
                error_message=str(e)
            )
        
        finally:
            self.is_running = False
            
            # Call end callbacks
            for callback in self.end_callbacks:
                callback(results)
        
        return results
    
    def _execute_step(self) -> None:
        """Execute a single simulation step."""
        # Advance time
        time_step = self.time_manager.advance_time()
        
        # Process events scheduled for this time
        current_events = self.event_queue.get_events_at_time(time_step.simulation_time)
        
        # Limit number of events processed per step
        if len(current_events) > self.config.max_events_per_step:
            self.logger.warning(
                f"Too many events ({len(current_events)}) at step {time_step.step_number}, "
                f"processing only first {self.config.max_events_per_step}"
            )
            current_events = current_events[:self.config.max_events_per_step]
        
        # Process events and generate new ones
        if current_events:
            new_events = self.event_processor.process_events(current_events)
            self.event_queue.schedule_events(new_events)
        
        # Update all networks
        for network_id, network in self.networks.items():
            if hasattr(network, 'update'):
                network.update(time_step.dt)
        
        # Update all probes
        for probe_id, probe in self.probes.items():
            if hasattr(probe, 'record'):
                probe.record(time_step.simulation_time)
        
        # Update all stimuli
        for stimulus_id, stimulus in self.stimuli.items():
            if hasattr(stimulus, 'update'):
                stimulus.update(time_step.dt, time_step.simulation_time)
        
        # Call step callbacks
        for callback in self.step_callbacks:
            callback(time_step.step_number, time_step.simulation_time)
        
        # Log progress periodically
        if time_step.step_number % 1000 == 0:
            progress = self.time_manager.get_simulation_progress()
            self.logger.info(f"Step {time_step.step_number}, "
                           f"Time: {time_step.simulation_time:.1f}ms, "
                           f"Progress: {progress*100:.1f}%")
    
    def _create_automatic_checkpoint(self) -> None:
        """Create an automatic checkpoint."""
        try:
            checkpoint = self.state_manager.create_checkpoint(
                timestamp=self.time_manager.current_time,
                step_number=self.time_manager.current_step,
                metadata={
                    'checkpoint_type': 'automatic',
                    'wall_time': time.time()
                }
            )
            self.last_checkpoint_step = self.time_manager.current_step
            self.logger.debug(f"Created checkpoint at step {self.time_manager.current_step}")
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {str(e)}")
    
    def pause(self) -> None:
        """Pause the simulation."""
        self.is_paused = True
        self.logger.info("Simulation paused")
    
    def resume(self) -> None:
        """Resume a paused simulation."""
        self.is_paused = False
        self.logger.info("Simulation resumed")
    
    def stop(self) -> None:
        """Stop the simulation."""
        self.is_running = False
        self.logger.info("Simulation stop requested")
    
    def create_manual_checkpoint(self, metadata: Optional[Dict[str, Any]] = None) -> StateCheckpoint:
        """
        Create a manual checkpoint.
        
        Args:
            metadata: Optional metadata to include with checkpoint
            
        Returns:
            Created checkpoint
        """
        checkpoint_metadata = {'checkpoint_type': 'manual'}
        if metadata:
            checkpoint_metadata.update(metadata)
        
        return self.state_manager.create_checkpoint(
            timestamp=self.time_manager.current_time,
            step_number=self.time_manager.current_step,
            metadata=checkpoint_metadata
        )
    
    def restore_from_checkpoint(self, checkpoint: StateCheckpoint) -> bool:
        """
        Restore simulation state from checkpoint.
        
        Args:
            checkpoint: The checkpoint to restore from
            
        Returns:
            True if restoration successful, False otherwise
        """
        success = self.state_manager.restore_checkpoint(checkpoint)
        if success:
            # Update time manager state
            self.time_manager.current_time = checkpoint.timestamp
            self.time_manager.current_step = checkpoint.step_number
            self.logger.info(f"Restored from checkpoint at step {checkpoint.step_number}")
        else:
            self.logger.error(f"Failed to restore from checkpoint at step {checkpoint.step_number}")
        
        return success
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current simulation status.
        
        Returns:
            Dictionary containing status information
        """
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_step': self.time_manager.current_step,
            'current_time': self.time_manager.current_time,
            'progress': self.time_manager.get_simulation_progress(),
            'networks_count': len(self.networks),
            'probes_count': len(self.probes),
            'stimuli_count': len(self.stimuli),
            'events_queued': self.event_queue.size(),
            'performance_stats': self.time_manager.get_performance_stats()
        }
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.is_running = False
        self.is_paused = False
        self.time_manager.reset()
        self.event_queue.clear()
        self.last_checkpoint_step = 0
        self.logger.info("Simulation reset to initial state")
