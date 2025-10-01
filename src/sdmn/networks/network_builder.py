"""
Network builder for assembling neural networks.

This module provides high-level tools for constructing neural networks
with various topologies and connectivity patterns.
"""

from typing import Dict, List, Optional, Any, Tuple, Type
from dataclasses import dataclass
from enum import Enum
import numpy as np

from sdmn.neurons.base_neuron import BaseNeuron, NeuronFactory, NeuronType, NeuronParameters
from sdmn.neurons.synapse import SynapseFactory, SynapticParameters, SynapseType


class NetworkTopology(Enum):
    """Standard network topologies."""
    RANDOM = "random"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    RING = "ring"
    GRID_2D = "grid_2d"
    HIERARCHICAL = "hierarchical"
    MODULAR = "modular"


@dataclass
class NetworkConfiguration:
    """Configuration for network construction."""
    name: str
    n_neurons: int
    topology: NetworkTopology
    neuron_type: NeuronType = NeuronType.LEAKY_INTEGRATE_FIRE
    connection_probability: float = 0.1
    weight_range: Tuple[float, float] = (0.5, 2.0)
    delay_range: Tuple[float, float] = (1.0, 10.0)
    excitatory_ratio: float = 0.8
    enable_plasticity: bool = False
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


class NetworkBuilder:
    """
    Builder class for constructing neural networks with various topologies.
    
    Provides methods to create networks with different connectivity patterns,
    neuron types, and structural properties.
    """
    
    def __init__(self):
        self.networks: Dict[str, 'Network'] = {}
        self.configurations: Dict[str, NetworkConfiguration] = {}
    
    def create_network(self, config: NetworkConfiguration) -> 'Network':
        """
        Create a network based on the provided configuration.
        
        Args:
            config: Network configuration parameters
            
        Returns:
            Constructed Network instance
        """
        print(f"Building network '{config.name}' with {config.n_neurons} neurons...")
        
        # Create neurons
        neurons = self._create_neurons(config)
        
        # Create connections based on topology
        synapses = self._create_connections(neurons, config)
        
        # Create network instance
        network = Network(config.name, neurons, synapses, config)
        
        # Store network
        self.networks[config.name] = network
        self.configurations[config.name] = config
        
        print(f"Created network with {len(neurons)} neurons and {len(synapses)} synapses")
        return network
    
    def _create_neurons(self, config: NetworkConfiguration) -> Dict[str, BaseNeuron]:
        """Create neurons for the network."""
        neurons = {}
        
        # Create neuron parameters
        if config.neuron_type == NeuronType.LEAKY_INTEGRATE_FIRE:
            from sdmn.neurons.lif_neuron import LIFParameters
            base_params = LIFParameters()
        else:
            base_params = NeuronParameters(neuron_type=config.neuron_type)
        
        # Add parameter variability for heterogeneity
        for i in range(config.n_neurons):
            neuron_id = f"{config.name}_neuron_{i:04d}"
            
            # Add parameter variations
            params = self._add_parameter_variability(base_params, config)
            
            # Create neuron
            neuron = NeuronFactory.create_neuron(config.neuron_type, neuron_id, params)
            neurons[neuron_id] = neuron
        
        return neurons
    
    def _add_parameter_variability(self, base_params: NeuronParameters, 
                                 config: NetworkConfiguration) -> NeuronParameters:
        """Add variability to neuron parameters."""
        # Create copy of parameters
        params = NeuronParameters(
            neuron_type=base_params.neuron_type,
            dt=base_params.dt,
            v_rest=base_params.v_rest + np.random.normal(0, 2.0),
            v_thresh=base_params.v_thresh + np.random.normal(0, 1.0),
            v_reset=base_params.v_reset + np.random.normal(0, 1.0),
            refractory_period=base_params.refractory_period + np.random.normal(0, 0.2)
        )
        
        # Copy custom parameters with variations
        for key, value in base_params.custom_params.items():
            if isinstance(value, (int, float)):
                # Add 10% variability to numeric parameters
                variation = value * np.random.normal(0, 0.1)
                params.set_parameter(key, value + variation)
            else:
                params.set_parameter(key, value)
        
        return params
    
    def _create_connections(self, neurons: Dict[str, BaseNeuron], 
                          config: NetworkConfiguration) -> Dict[str, Any]:
        """Create synaptic connections based on network topology."""
        if config.topology == NetworkTopology.RANDOM:
            return self._create_random_connections(neurons, config)
        elif config.topology == NetworkTopology.RING:
            return self._create_ring_connections(neurons, config)
        elif config.topology == NetworkTopology.SMALL_WORLD:
            return self._create_small_world_connections(neurons, config)
        elif config.topology == NetworkTopology.GRID_2D:
            return self._create_grid_connections(neurons, config)
        else:
            raise NotImplementedError(f"Topology {config.topology} not implemented")
    
    def _create_random_connections(self, neurons: Dict[str, BaseNeuron], 
                                 config: NetworkConfiguration) -> Dict[str, Any]:
        """Create random connections between neurons."""
        synapses = {}
        neuron_ids = list(neurons.keys())
        
        for i, pre_id in enumerate(neuron_ids):
            for j, post_id in enumerate(neuron_ids):
                if i != j and np.random.random() < config.connection_probability:
                    # Determine if excitatory or inhibitory
                    is_excitatory = np.random.random() < config.excitatory_ratio
                    
                    # Create synapse
                    syn_id = f"syn_{pre_id}_to_{post_id}"
                    weight = np.random.uniform(*config.weight_range)
                    delay = np.random.uniform(*config.delay_range)
                    
                    if is_excitatory:
                        synapse = SynapseFactory.create_excitatory_synapse(
                            syn_id, pre_id, post_id, weight, delay
                        )
                    else:
                        synapse = SynapseFactory.create_inhibitory_synapse(
                            syn_id, pre_id, post_id, weight, delay
                        )
                    
                    # Enable plasticity if requested
                    if config.enable_plasticity:
                        synapse.parameters.enable_plasticity = True
                        synapse.parameters.learning_rate = 0.01
                    
                    synapses[syn_id] = synapse
                    
                    # Register with neurons
                    neurons[pre_id].add_postsynaptic_connection(synapse)
                    neurons[post_id].add_presynaptic_connection(synapse)
        
        return synapses
    
    def _create_ring_connections(self, neurons: Dict[str, BaseNeuron], 
                               config: NetworkConfiguration) -> Dict[str, Any]:
        """Create ring topology connections."""
        synapses = {}
        neuron_ids = list(neurons.keys())
        n_neurons = len(neuron_ids)
        
        for i in range(n_neurons):
            # Connect to next neuron in ring
            next_i = (i + 1) % n_neurons
            pre_id = neuron_ids[i]
            post_id = neuron_ids[next_i]
            
            syn_id = f"syn_{pre_id}_to_{post_id}"
            weight = np.random.uniform(*config.weight_range)
            delay = np.random.uniform(*config.delay_range)
            
            synapse = SynapseFactory.create_excitatory_synapse(
                syn_id, pre_id, post_id, weight, delay
            )
            
            if config.enable_plasticity:
                synapse.parameters.enable_plasticity = True
                synapse.parameters.learning_rate = 0.01
            
            synapses[syn_id] = synapse
            
            # Register with neurons
            neurons[pre_id].add_postsynaptic_connection(synapse)
            neurons[post_id].add_presynaptic_connection(synapse)
        
        return synapses
    
    def _create_small_world_connections(self, neurons: Dict[str, BaseNeuron], 
                                      config: NetworkConfiguration) -> Dict[str, Any]:
        """Create small-world topology connections."""
        # Start with ring connections
        synapses = self._create_ring_connections(neurons, config)
        
        # Add additional local connections
        neuron_ids = list(neurons.keys())
        n_neurons = len(neuron_ids)
        
        # Connect each neuron to k nearest neighbors
        k = max(2, int(config.connection_probability * n_neurons))
        
        for i in range(n_neurons):
            for offset in range(2, k//2 + 2):  # Skip immediate neighbors
                # Forward connections
                j = (i + offset) % n_neurons
                if np.random.random() < config.connection_probability:
                    self._add_connection(neurons, synapses, neuron_ids[i], 
                                       neuron_ids[j], config, f"sw_forward_{i}_{j}")
                
                # Backward connections
                j = (i - offset) % n_neurons
                if np.random.random() < config.connection_probability:
                    self._add_connection(neurons, synapses, neuron_ids[i], 
                                       neuron_ids[j], config, f"sw_backward_{i}_{j}")
        
        # Rewire some connections for small-world property
        rewire_prob = 0.1
        original_synapses = list(synapses.items())
        
        for syn_id, synapse in original_synapses:
            if np.random.random() < rewire_prob:
                # Remove old connection
                pre_id = synapse.presynaptic_neuron_id
                post_id = synapse.postsynaptic_neuron_id
                
                neurons[pre_id].remove_postsynaptic_connection(synapse)
                neurons[post_id].remove_presynaptic_connection(synapse)
                del synapses[syn_id]
                
                # Add new random connection
                new_post_id = np.random.choice(neuron_ids)
                if new_post_id != pre_id:
                    self._add_connection(neurons, synapses, pre_id, new_post_id, 
                                       config, f"rewired_{syn_id}")
        
        return synapses
    
    def _create_grid_connections(self, neurons: Dict[str, BaseNeuron], 
                               config: NetworkConfiguration) -> Dict[str, Any]:
        """Create 2D grid topology connections."""
        synapses = {}
        neuron_ids = list(neurons.keys())
        
        # Determine grid dimensions
        n_neurons = len(neuron_ids)
        grid_size = int(np.sqrt(n_neurons))
        
        # Connect nearest neighbors in grid
        for i in range(grid_size):
            for j in range(grid_size):
                neuron_idx = i * grid_size + j
                if neuron_idx >= n_neurons:
                    break
                
                pre_id = neuron_ids[neuron_idx]
                
                # Connect to right neighbor
                if j < grid_size - 1:
                    right_idx = i * grid_size + (j + 1)
                    if right_idx < n_neurons:
                        post_id = neuron_ids[right_idx]
                        self._add_connection(neurons, synapses, pre_id, post_id, 
                                           config, f"grid_right_{i}_{j}")
                
                # Connect to bottom neighbor
                if i < grid_size - 1:
                    bottom_idx = (i + 1) * grid_size + j
                    if bottom_idx < n_neurons:
                        post_id = neuron_ids[bottom_idx]
                        self._add_connection(neurons, synapses, pre_id, post_id, 
                                           config, f"grid_bottom_{i}_{j}")
        
        return synapses
    
    def _add_connection(self, neurons: Dict[str, BaseNeuron], 
                      synapses: Dict[str, Any], 
                      pre_id: str, post_id: str, 
                      config: NetworkConfiguration, syn_id: str) -> None:
        """Helper method to add a synaptic connection."""
        if syn_id in synapses or pre_id == post_id:
            return
        
        is_excitatory = np.random.random() < config.excitatory_ratio
        weight = np.random.uniform(*config.weight_range)
        delay = np.random.uniform(*config.delay_range)
        
        if is_excitatory:
            synapse = SynapseFactory.create_excitatory_synapse(
                syn_id, pre_id, post_id, weight, delay
            )
        else:
            synapse = SynapseFactory.create_inhibitory_synapse(
                syn_id, pre_id, post_id, weight, delay
            )
        
        if config.enable_plasticity:
            synapse.parameters.enable_plasticity = True
            synapse.parameters.learning_rate = 0.01
        
        synapses[syn_id] = synapse
        
        # Register with neurons
        neurons[pre_id].add_postsynaptic_connection(synapse)
        neurons[post_id].add_presynaptic_connection(synapse)
    
    def get_network(self, name: str) -> Optional['Network']:
        """Get network by name."""
        return self.networks.get(name)
    
    def list_networks(self) -> List[str]:
        """Get list of created network names."""
        return list(self.networks.keys())


class Network:
    """
    Network container class that holds neurons and synapses.
    
    Provides methods for network updates and analysis.
    """
    
    def __init__(self, name: str, neurons: Dict[str, BaseNeuron], 
                 synapses: Dict[str, Any], config: NetworkConfiguration):
        self.name = name
        self.neurons = neurons
        self.synapses = synapses
        self.config = config
        self.current_time = 0.0
    
    def update(self, dt: float) -> None:
        """Update all network components for one time step."""
        self.current_time += dt
        
        # Update all synapses first
        for synapse in self.synapses.values():
            synapse.update(dt)
        
        # Update all neurons
        for neuron in self.neurons.values():
            # Calculate synaptic inputs
            for synapse in neuron.presynaptic_connections:
                if synapse.synapse_id in self.synapses:
                    current = synapse.calculate_current(
                        neuron.get_membrane_potential()
                    )
                    neuron.add_synaptic_input(current)
            
            # Update neuron state
            neuron.update(dt)
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        total_neurons = len(self.neurons)
        total_synapses = len(self.synapses)
        
        # Count excitatory/inhibitory synapses
        excitatory_count = 0
        inhibitory_count = 0
        
        for synapse in self.synapses.values():
            if synapse.is_excitatory():
                excitatory_count += 1
            elif synapse.is_inhibitory():
                inhibitory_count += 1
        
        # Calculate connectivity statistics
        in_degrees = []
        out_degrees = []
        
        for neuron in self.neurons.values():
            in_count, out_count = neuron.get_connection_count()
            in_degrees.append(in_count)
            out_degrees.append(out_count)
        
        return {
            'name': self.name,
            'total_neurons': total_neurons,
            'total_synapses': total_synapses,
            'excitatory_synapses': excitatory_count,
            'inhibitory_synapses': inhibitory_count,
            'excitatory_ratio': excitatory_count / max(total_synapses, 1),
            'connection_density': total_synapses / max(total_neurons**2, 1),
            'mean_in_degree': np.mean(in_degrees),
            'mean_out_degree': np.mean(out_degrees),
            'std_in_degree': np.std(in_degrees),
            'std_out_degree': np.std(out_degrees)
        }
    
    def reset(self) -> None:
        """Reset all network components to initial state."""
        self.current_time = 0.0
        
        for neuron in self.neurons.values():
            neuron.reset_neuron()
        
        for synapse in self.synapses.values():
            synapse.reset_synapse()
    
    def __str__(self) -> str:
        """String representation of network."""
        return (f"Network(name='{self.name}', "
                f"neurons={len(self.neurons)}, "
                f"synapses={len(self.synapses)}, "
                f"topology={self.config.topology.value})")
    
    def __repr__(self) -> str:
        """Detailed representation of network."""
        stats = self.get_network_statistics()
        return (f"Network(name='{self.name}', "
                f"neurons={stats['total_neurons']}, "
                f"synapses={stats['total_synapses']}, "
                f"E/I_ratio={stats['excitatory_ratio']:.2f}, "
                f"density={stats['connection_density']:.4f})")
