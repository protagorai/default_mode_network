"""
Network manager for C. elegans graded potential neural networks.

Provides high-level interface for building, simulating, and analyzing
C. elegans neural networks with easy connectivity management.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import time

from sdmn.neurons.graded import (
    CElegansNeuron,
    SensoryNeuron,
    Interneuron,
    MotorNeuron,
    CElegansParameters,
    CElegansNeuronClass
)
from sdmn.synapses import (
    GradedChemicalSynapse,
    GradedSynapseParameters,
    GapJunction,
    GapJunctionParameters,
    SynapseType
)


class SimulationState(Enum):
    """Simulation state."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """Configuration for network simulation."""
    dt: float = 0.01                    # Time step (ms)
    record_interval: int = 10           # Record every N steps
    progress_interval: int = 10000      # Print progress every N steps
    enable_recording: bool = True       # Record voltage traces
    
    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if self.record_interval < 1:
            raise ValueError("Record interval must be >= 1")


class CElegansNetwork:
    """
    High-level network manager for C. elegans simulations.
    
    Provides easy interface for:
    - Adding/removing neurons and connections
    - Running simulations with control (pause/resume/extend)
    - Modifying connectivity dynamically
    - Accessing results and analysis
    
    Example:
        network = CElegansNetwork()
        n1 = network.add_sensory_neuron("AWC")
        n2 = network.add_interneuron("AVA")
        network.add_graded_synapse(n1, n2, weight=2.0)
        network.simulate(duration=1000)
        voltages = network.get_voltage_history()
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize network manager.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        
        # Network components
        self.neurons: Dict[str, CElegansNeuron] = {}
        self.chemical_synapses: Dict[str, GradedChemicalSynapse] = {}
        self.gap_junctions: Dict[str, GapJunction] = {}
        
        # Simulation state
        self.state = SimulationState.CREATED
        self.current_time = 0.0
        self.step_count = 0
        
        # Recording
        self.voltage_history: Dict[str, List[Tuple[float, float]]] = {}
        self.recorded_times: List[float] = []
        
        # Performance tracking
        self.simulation_start_time = None
        self.simulation_end_time = None
        
    # ========================================================================
    # Neuron Management
    # ========================================================================
    
    def add_neuron(self, neuron: CElegansNeuron) -> str:
        """
        Add a neuron to the network.
        
        Args:
            neuron: C. elegans neuron instance
            
        Returns:
            Neuron ID
        """
        if neuron.neuron_id in self.neurons:
            raise ValueError(f"Neuron {neuron.neuron_id} already exists")
        
        self.neurons[neuron.neuron_id] = neuron
        if self.config.enable_recording:
            self.voltage_history[neuron.neuron_id] = []
        
        return neuron.neuron_id
    
    def add_sensory_neuron(self, neuron_id: str, **params) -> str:
        """
        Add a sensory neuron.
        
        Args:
            neuron_id: Unique identifier
            **params: Override default parameters
            
        Returns:
            Neuron ID
        """
        neuron = SensoryNeuron(neuron_id, **params)
        return self.add_neuron(neuron)
    
    def add_interneuron(self, neuron_id: str, **params) -> str:
        """
        Add an interneuron.
        
        Args:
            neuron_id: Unique identifier
            **params: Override default parameters
            
        Returns:
            Neuron ID
        """
        neuron = Interneuron(neuron_id, **params)
        return self.add_neuron(neuron)
    
    def add_motor_neuron(self, neuron_id: str, **params) -> str:
        """
        Add a motor neuron.
        
        Args:
            neuron_id: Unique identifier
            **params: Override default parameters
            
        Returns:
            Neuron ID
        """
        neuron = MotorNeuron(neuron_id, **params)
        return self.add_neuron(neuron)
    
    def remove_neuron(self, neuron_id: str) -> None:
        """
        Remove a neuron and all its connections.
        
        Args:
            neuron_id: ID of neuron to remove
        """
        if neuron_id not in self.neurons:
            raise ValueError(f"Neuron {neuron_id} not found")
        
        # Remove associated synapses
        synapses_to_remove = []
        for syn_id, syn in self.chemical_synapses.items():
            if syn.presynaptic_neuron_id == neuron_id or syn.postsynaptic_neuron_id == neuron_id:
                synapses_to_remove.append(syn_id)
        
        for syn_id in synapses_to_remove:
            del self.chemical_synapses[syn_id]
        
        # Remove associated gap junctions
        gaps_to_remove = []
        for gap_id, gap in self.gap_junctions.items():
            if gap.presynaptic_neuron_id == neuron_id or gap.postsynaptic_neuron_id == neuron_id:
                gaps_to_remove.append(gap_id)
        
        for gap_id in gaps_to_remove:
            del self.gap_junctions[gap_id]
        
        # Remove neuron
        del self.neurons[neuron_id]
        if neuron_id in self.voltage_history:
            del self.voltage_history[neuron_id]
    
    def get_neuron(self, neuron_id: str) -> CElegansNeuron:
        """Get neuron by ID."""
        if neuron_id not in self.neurons:
            raise ValueError(f"Neuron {neuron_id} not found")
        return self.neurons[neuron_id]
    
    def get_neuron_count(self) -> int:
        """Get total number of neurons."""
        return len(self.neurons)
    
    # ========================================================================
    # Connectivity Management
    # ========================================================================
    
    def add_graded_synapse(self, pre_id: Union[str, CElegansNeuron], 
                          post_id: Union[str, CElegansNeuron],
                          synapse_id: Optional[str] = None,
                          synapse_type: SynapseType = SynapseType.EXCITATORY,
                          weight: float = 1.0,
                          **params) -> str:
        """
        Add a graded chemical synapse.
        
        Args:
            pre_id: Presynaptic neuron ID or instance
            post_id: Postsynaptic neuron ID or instance
            synapse_id: Optional unique ID (auto-generated if None)
            synapse_type: Excitatory or inhibitory
            weight: Synaptic weight (nS)
            **params: Additional synapse parameters
            
        Returns:
            Synapse ID
        """
        # Get neuron instances
        if isinstance(pre_id, str):
            pre_neuron = self.get_neuron(pre_id)
        else:
            pre_neuron = pre_id
            pre_id = pre_neuron.neuron_id
        
        if isinstance(post_id, str):
            post_neuron = self.get_neuron(post_id)
        else:
            post_neuron = post_id
            post_id = post_neuron.neuron_id
        
        # Generate synapse ID if not provided
        if synapse_id is None:
            synapse_id = f"syn_{pre_id}_{post_id}_{len(self.chemical_synapses)}"
        
        if synapse_id in self.chemical_synapses:
            raise ValueError(f"Synapse {synapse_id} already exists")
        
        # Create synapse
        syn_params = GradedSynapseParameters(
            synapse_type=synapse_type,
            weight=weight,
            **params
        )
        
        synapse = GradedChemicalSynapse(
            synapse_id, pre_neuron, post_neuron, syn_params
        )
        
        self.chemical_synapses[synapse_id] = synapse
        return synapse_id
    
    def add_gap_junction(self, neuron_a_id: Union[str, CElegansNeuron],
                        neuron_b_id: Union[str, CElegansNeuron],
                        gap_id: Optional[str] = None,
                        conductance: float = 0.5,
                        **params) -> str:
        """
        Add a gap junction (electrical synapse).
        
        Args:
            neuron_a_id: First neuron ID or instance
            neuron_b_id: Second neuron ID or instance
            gap_id: Optional unique ID (auto-generated if None)
            conductance: Coupling conductance (nS)
            **params: Additional gap junction parameters
            
        Returns:
            Gap junction ID
        """
        # Get neuron instances
        if isinstance(neuron_a_id, str):
            neuron_a = self.get_neuron(neuron_a_id)
        else:
            neuron_a = neuron_a_id
            neuron_a_id = neuron_a.neuron_id
        
        if isinstance(neuron_b_id, str):
            neuron_b = self.get_neuron(neuron_b_id)
        else:
            neuron_b = neuron_b_id
            neuron_b_id = neuron_b.neuron_id
        
        # Generate gap ID if not provided
        if gap_id is None:
            gap_id = f"gap_{neuron_a_id}_{neuron_b_id}_{len(self.gap_junctions)}"
        
        if gap_id in self.gap_junctions:
            raise ValueError(f"Gap junction {gap_id} already exists")
        
        # Create gap junction
        gap_params = GapJunctionParameters(
            conductance=conductance,
            **params
        )
        
        gap = GapJunction(gap_id, neuron_a, neuron_b, gap_params)
        self.gap_junctions[gap_id] = gap
        return gap_id
    
    def remove_synapse(self, synapse_id: str) -> None:
        """Remove a chemical synapse."""
        if synapse_id not in self.chemical_synapses:
            raise ValueError(f"Synapse {synapse_id} not found")
        del self.chemical_synapses[synapse_id]
    
    def remove_gap_junction(self, gap_id: str) -> None:
        """Remove a gap junction."""
        if gap_id not in self.gap_junctions:
            raise ValueError(f"Gap junction {gap_id} not found")
        del self.gap_junctions[gap_id]
    
    def set_synapse_weight(self, synapse_id: str, weight: float) -> None:
        """Modify synaptic weight."""
        if synapse_id not in self.chemical_synapses:
            raise ValueError(f"Synapse {synapse_id} not found")
        self.chemical_synapses[synapse_id].graded_params.weight = weight
    
    def set_gap_conductance(self, gap_id: str, conductance: float) -> None:
        """Modify gap junction conductance."""
        if gap_id not in self.gap_junctions:
            raise ValueError(f"Gap junction {gap_id} not found")
        self.gap_junctions[gap_id].set_conductance(conductance)
    
    def get_connectivity_summary(self) -> Dict[str, Any]:
        """Get summary of network connectivity."""
        return {
            'n_neurons': len(self.neurons),
            'n_chemical_synapses': len(self.chemical_synapses),
            'n_gap_junctions': len(self.gap_junctions),
            'neurons_by_class': self._count_neuron_classes(),
            'avg_synapses_per_neuron': len(self.chemical_synapses) / len(self.neurons) if self.neurons else 0,
            'avg_gaps_per_neuron': len(self.gap_junctions) / len(self.neurons) if self.neurons else 0,
        }
    
    def _count_neuron_classes(self) -> Dict[str, int]:
        """Count neurons by class."""
        counts = {'sensory': 0, 'interneuron': 0, 'motor': 0, 'other': 0}
        for neuron in self.neurons.values():
            if hasattr(neuron, 'neuron_class'):
                class_name = neuron.neuron_class.value
                if class_name in counts:
                    counts[class_name] += 1
                else:
                    counts['other'] += 1
            else:
                counts['other'] += 1
        return counts
    
    # ========================================================================
    # Simulation Control
    # ========================================================================
    
    def simulate(self, duration: float, dt: Optional[float] = None,
                progress: bool = True) -> None:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration (ms)
            dt: Time step (uses config if None)
            progress: Print progress updates
        """
        if not self.neurons:
            raise ValueError("Network has no neurons")
        
        if dt is None:
            dt = self.config.dt
        
        steps = int(duration / dt)
        
        if self.state == SimulationState.CREATED:
            self.simulation_start_time = time.time()
            self.state = SimulationState.RUNNING
        elif self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
        
        if progress:
            print(f"\nSimulating {duration} ms ({steps} steps, dt={dt} ms)...")
            print(f"Network: {len(self.neurons)} neurons, "
                  f"{len(self.chemical_synapses)} synapses, "
                  f"{len(self.gap_junctions)} gap junctions")
        
        start_time = time.time()
        
        for step in range(steps):
            self._step(dt)
            
            # Progress updates
            if progress and (step + 1) % self.config.progress_interval == 0:
                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed
                sim_time = self.current_time
                print(f"  Step {step + 1}/{steps} ({sim_time:.1f} ms) - "
                      f"{rate:.0f} steps/s")
        
        elapsed = time.time() - start_time
        if progress:
            print(f"Completed in {elapsed:.2f} s "
                  f"({steps/elapsed:.0f} steps/s, "
                  f"{duration/elapsed:.1f}× real-time)")
    
    def _step(self, dt: float) -> None:
        """Execute one simulation step."""
        # Update gating variables and integrate voltages
        for neuron in self.neurons.values():
            neuron.update(dt)
        
        # Update synapses
        for synapse in self.chemical_synapses.values():
            synapse.update(dt)
        
        # Update gap junctions
        for gap in self.gap_junctions.values():
            gap.update(dt)
        
        # Record
        if self.config.enable_recording and self.step_count % self.config.record_interval == 0:
            self.recorded_times.append(self.current_time)
            for neuron_id, neuron in self.neurons.items():
                self.voltage_history[neuron_id].append(
                    (self.current_time, neuron.voltage)
                )
        
        # Update state
        self.current_time += dt
        self.step_count += 1
    
    def pause(self) -> None:
        """Pause simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
    
    def resume(self, duration: float, dt: Optional[float] = None) -> None:
        """
        Resume paused simulation.
        
        Args:
            duration: Additional duration to simulate
            dt: Time step (uses config if None)
        """
        if self.state != SimulationState.PAUSED:
            raise RuntimeError("Can only resume paused simulation")
        self.simulate(duration, dt)
    
    def reset(self) -> None:
        """Reset simulation state."""
        for neuron in self.neurons.values():
            neuron.reset_neuron()
        
        for synapse in self.chemical_synapses.values():
            synapse.reset()
        
        for gap in self.gap_junctions.values():
            gap.reset()
        
        self.current_time = 0.0
        self.step_count = 0
        self.state = SimulationState.CREATED
        
        if self.config.enable_recording:
            for neuron_id in self.voltage_history:
                self.voltage_history[neuron_id] = []
            self.recorded_times = []
    
    # ========================================================================
    # Data Access
    # ========================================================================
    
    def get_voltage_history(self, neuron_id: Optional[str] = None) -> Union[List, Dict]:
        """
        Get recorded voltage traces.
        
        Args:
            neuron_id: Specific neuron ID (returns all if None)
            
        Returns:
            List of (time, voltage) tuples or dict of all traces
        """
        if neuron_id is not None:
            if neuron_id not in self.voltage_history:
                raise ValueError(f"No voltage history for neuron {neuron_id}")
            return self.voltage_history[neuron_id]
        return self.voltage_history
    
    def get_voltages_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get voltage traces as numpy arrays.
        
        Returns:
            (times, voltages) where voltages is shape (n_neurons, n_timepoints)
        """
        if not self.recorded_times:
            return np.array([]), np.array([[]])
        
        times = np.array(self.recorded_times)
        neuron_ids = sorted(self.voltage_history.keys())
        
        voltages = []
        for neuron_id in neuron_ids:
            trace = self.voltage_history[neuron_id]
            voltages.append([v for t, v in trace])
        
        return times, np.array(voltages)
    
    def get_current_voltages(self) -> Dict[str, float]:
        """Get current voltage of all neurons."""
        return {nid: n.voltage for nid, n in self.neurons.items()}
    
    def set_external_current(self, neuron_id: str, current: float) -> None:
        """Set external current for a neuron."""
        self.get_neuron(neuron_id).set_external_current(current)
    
    def set_external_currents(self, currents: Dict[str, float]) -> None:
        """Set external currents for multiple neurons."""
        for neuron_id, current in currents.items():
            self.set_external_current(neuron_id, current)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"CElegansNetwork(neurons={len(self.neurons)}, "
                f"synapses={len(self.chemical_synapses)}, "
                f"gaps={len(self.gap_junctions)}, "
                f"state={self.state.value}, "
                f"time={self.current_time:.1f}ms)")

