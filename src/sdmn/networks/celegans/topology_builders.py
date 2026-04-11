"""
Topology builders for C. elegans neural networks.

Provides automated generation of different network architectures:
- Uniform/Regular topology (nearest neighbors)
- Small-world topology (Watts-Strogatz with rewiring)
- Random topology (Erdos-Renyi)

These builders enable testing the hypothesis that uniform networks perform
poorly while small-world networks enable emergent behavior.

References:
    Watts & Strogatz (1998), "Collective dynamics of 'small-world' networks."
    Nature, 393(6684), 440-442.
    
    Varshney et al. (2011), "Structural properties of the C. elegans neuronal
    network." PLOS Computational Biology, 7(2), e1001066.
"""

from typing import Optional, Literal, Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from sdmn.networks.celegans.network_manager import CElegansNetwork
from sdmn.synapses import SynapseType


@dataclass
class TopologyParameters:
    """Base parameters for network topology builders."""
    
    # Network size
    n_neurons: int = 302
    
    # Neuron types
    neuron_class: Literal['sensory', 'interneuron', 'motor', 'mixed'] = 'interneuron'
    sensory_fraction: float = 0.1      # If mixed, fraction of sensory neurons
    motor_fraction: float = 0.1        # If mixed, fraction of motor neurons
    
    # Connectivity
    k_neighbors: int = 14              # Target average connections per neuron
    connection_variance: float = 2.0   # Std dev of connection count (Poisson-like)
    
    # Synaptic properties
    weight_mean: float = 2.0           # Mean synaptic weight (nS)
    weight_std: float = 0.3            # Std dev of weights (nS)
    exc_inh_ratio: float = 0.85        # Fraction excitatory (0.85 = 85% exc, 15% inh)
    
    # Gap junctions
    gap_junction_prob: float = 0.15    # Probability a connection is a gap junction
    gap_conductance_mean: float = 0.5  # Mean gap conductance (nS)
    gap_conductance_std: float = 0.15  # Std dev of gap conductance
    
    # Random seed (for reproducibility)
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate parameters."""
        if self.n_neurons < 2:
            raise ValueError("Need at least 2 neurons")
        if self.k_neighbors < 1:
            raise ValueError("Need at least 1 connection per neuron")
        if not 0 <= self.exc_inh_ratio <= 1:
            raise ValueError("exc_inh_ratio must be in [0, 1]")
        if not 0 <= self.gap_junction_prob <= 1:
            raise ValueError("gap_junction_prob must be in [0, 1]")
        if self.neuron_class == 'mixed':
            if self.sensory_fraction + self.motor_fraction > 1.0:
                raise ValueError("sensory + motor fractions must be <= 1.0")


class BaseTopologyBuilder(ABC):
    """
    Abstract base class for topology builders.
    
    Subclasses implement specific topology generation algorithms.
    """
    
    def __init__(self, params: TopologyParameters):
        """
        Initialize builder.
        
        Args:
            params: Topology parameters
        """
        self.params = params
        self.network: Optional[CElegansNetwork] = None
        
        # Set random seed for reproducibility
        if params.random_seed is not None:
            np.random.seed(params.random_seed)
        
        self.rng = np.random.default_rng(params.random_seed)
    
    @abstractmethod
    def build(self) -> CElegansNetwork:
        """
        Build network with specific topology.
        
        Returns:
            Constructed CElegansNetwork
        """
        pass
    
    def _create_neurons(self, network: CElegansNetwork) -> None:
        """
        Add neurons to network based on neuron_class parameter.
        
        Args:
            network: Network to add neurons to
        """
        if self.params.neuron_class == 'sensory':
            for i in range(self.params.n_neurons):
                network.add_sensory_neuron(f"N{i}")
        
        elif self.params.neuron_class == 'motor':
            for i in range(self.params.n_neurons):
                network.add_motor_neuron(f"N{i}")
        
        elif self.params.neuron_class == 'interneuron':
            for i in range(self.params.n_neurons):
                network.add_interneuron(f"N{i}")
        
        elif self.params.neuron_class == 'mixed':
            n_sensory = int(self.params.n_neurons * self.params.sensory_fraction)
            n_motor = int(self.params.n_neurons * self.params.motor_fraction)
            n_inter = self.params.n_neurons - n_sensory - n_motor
            
            # Sensory neurons first
            for i in range(n_sensory):
                network.add_sensory_neuron(f"S{i}")
            
            # Interneurons
            for i in range(n_inter):
                network.add_interneuron(f"I{i}")
            
            # Motor neurons
            for i in range(n_motor):
                network.add_motor_neuron(f"M{i}")
        else:
            raise ValueError(f"Unknown neuron_class: {self.params.neuron_class}")
    
    def _add_connection(self, network: CElegansNetwork, pre_id: str, post_id: str) -> None:
        """
        Add a connection (chemical synapse or gap junction).
        
        Args:
            network: Network to add connection to
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
        """
        # Decide if this is a gap junction or chemical synapse
        is_gap = self.rng.random() < self.params.gap_junction_prob
        
        if is_gap:
            # Create gap junction
            conductance = self.rng.normal(
                self.params.gap_conductance_mean,
                self.params.gap_conductance_std
            )
            conductance = max(0.1, min(2.0, conductance))  # Clip to valid range
            
            network.add_gap_junction(pre_id, post_id, conductance=conductance)
        
        else:
            # Create chemical synapse
            weight = self.rng.normal(self.params.weight_mean, self.params.weight_std)
            weight = max(0.1, weight)  # Ensure positive
            
            # Decide excitatory vs inhibitory
            is_excitatory = self.rng.random() < self.params.exc_inh_ratio
            synapse_type = SynapseType.EXCITATORY if is_excitatory else SynapseType.INHIBITORY
            
            network.add_graded_synapse(
                pre_id, post_id,
                weight=weight,
                synapse_type=synapse_type
            )
    
    def _sample_connection_count(self) -> int:
        """
        Sample number of connections for a neuron.
        
        Uses Poisson distribution to allow variability around mean.
        
        Returns:
            Number of connections
        """
        if self.params.connection_variance == 0:
            return self.params.k_neighbors
        
        # Use Poisson distribution (naturally gives variance)
        k = self.rng.poisson(self.params.k_neighbors)
        
        # Ensure at least 1 connection
        return max(1, k)
    
    def _get_neuron_id(self, index: int) -> str:
        """
        Get neuron ID for a given index based on neuron_class configuration.

        Args:
            index: Neuron index (0-based)

        Returns:
            Neuron ID string
        """
        if self.params.neuron_class == 'mixed':
            n_sensory = int(self.params.n_neurons * self.params.sensory_fraction)
            n_motor = int(self.params.n_neurons * self.params.motor_fraction)
            n_inter = self.params.n_neurons - n_sensory - n_motor

            if index < n_sensory:
                return f"S{index}"
            elif index < n_sensory + n_inter:
                return f"I{index - n_sensory}"
            else:
                return f"M{index - n_sensory - n_inter}"
        else:
            return f"N{index}"

    def get_topology_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the built topology.
        
        Returns:
            Dictionary with topology statistics
        """
        if self.network is None:
            raise RuntimeError("Must call build() first")
        
        return self.network.get_connectivity_summary()


class UniformTopologyBuilder(BaseTopologyBuilder):
    """
    Build uniform/regular topology network.
    
    Creates a ring or lattice where each neuron connects to K nearest neighbors.
    This serves as a baseline for comparison - expected to show poor performance
    due to long path lengths and lack of shortcuts.
    
    Example:
        builder = UniformTopologyBuilder(
            TopologyParameters(n_neurons=302, k_neighbors=14)
        )
        network = builder.build()
    """
    
    def __init__(self, params: Optional[TopologyParameters] = None):
        """
        Initialize uniform topology builder.
        
        Args:
            params: Topology parameters (uses defaults if None)
        """
        if params is None:
            params = TopologyParameters()
        super().__init__(params)
    
    def build(self) -> CElegansNetwork:
        """
        Build uniform topology network.
        
        Each neuron connects to K nearest neighbors (K/2 forward, K/2 backward).
        
        Returns:
            Network with uniform connectivity
        """
        print(f"\nBuilding uniform topology network...")
        print(f"  Neurons: {self.params.n_neurons}")
        print(f"  Target avg connections: {self.params.k_neighbors}")
        print(f"  Connection variance: {self.params.connection_variance}")
        
        # Create network
        network = CElegansNetwork()
        
        # Add neurons
        self._create_neurons(network)
        
        # Add uniform connections
        total_connections = 0
        
        for i in range(self.params.n_neurons):
            neuron_id = self._get_neuron_id(i)
            
            # Sample how many connections this neuron should have
            k_this = self._sample_connection_count()
            
            # Connect to k_this nearest neighbors
            # Half forward, half backward
            k_forward = k_this // 2
            k_backward = k_this - k_forward
            
            # Forward connections
            for offset in range(1, k_forward + 1):
                target_idx = (i + offset) % self.params.n_neurons
                target_id = self._get_neuron_id(target_idx)
                self._add_connection(network, neuron_id, target_id)
                total_connections += 1
            
            # Backward connections
            for offset in range(1, k_backward + 1):
                target_idx = (i - offset) % self.params.n_neurons
                target_id = self._get_neuron_id(target_idx)
                self._add_connection(network, neuron_id, target_id)
                total_connections += 1
        
        self.network = network
        
        summary = network.get_connectivity_summary()
        actual_avg = summary['avg_synapses_per_neuron'] + summary['avg_gaps_per_neuron']
        
        print(f"  Created: {summary['n_chemical_synapses']} chemical synapses, "
              f"{summary['n_gap_junctions']} gap junctions")
        print(f"  Actual avg connections/neuron: {actual_avg:.1f}")
        print(f"  Topology: Uniform (regular ring)")
        
        return network


class SmallWorldBuilder(BaseTopologyBuilder):
    """
    Build small-world topology network using Watts-Strogatz algorithm.
    
    Starts with regular lattice, then randomly rewires edges to create
    shortcuts. This creates networks with:
    - High local clustering (like regular network)
    - Short average path length (like random network)
    - Enables efficient information propagation and emergent behavior
    
    Parameters:
        rewire_prob: Probability of rewiring each edge (0.1-0.3 typical)
        distance_decay: Controls preference for distant connections when rewiring
                       (lower = prefer distant, higher = any distance equally)
    
    Example:
        builder = SmallWorldBuilder(
            TopologyParameters(n_neurons=302, k_neighbors=14),
            rewire_prob=0.2
        )
        network = builder.build()
    """
    
    def __init__(self, params: Optional[TopologyParameters] = None,
                 rewire_prob: float = 0.2,
                 distance_decay: float = 50.0,
                 allow_self_loops: bool = False):
        """
        Initialize small-world builder.
        
        Args:
            params: Topology parameters
            rewire_prob: Probability of rewiring each edge (0.0-1.0)
            distance_decay: Distance preference for rewiring (lower = prefer distant)
            allow_self_loops: Allow neurons to connect to themselves
        """
        if params is None:
            params = TopologyParameters()
        super().__init__(params)
        
        self.rewire_prob = rewire_prob
        self.distance_decay = distance_decay
        self.allow_self_loops = allow_self_loops
        
        # Validate
        if not 0 <= rewire_prob <= 1:
            raise ValueError("rewire_prob must be in [0, 1]")
        if distance_decay <= 0:
            raise ValueError("distance_decay must be positive")
    
    def build(self) -> CElegansNetwork:
        """
        Build small-world network using Watts-Strogatz algorithm.
        
        Algorithm:
        1. Start with regular ring lattice (K nearest neighbors)
        2. For each edge, rewire with probability p:
           - Keep source neuron
           - Choose new random target (with distance bias)
           - Avoid self-loops and duplicates
        
        Returns:
            Network with small-world topology
        """
        print(f"\nBuilding small-world topology network...")
        print(f"  Neurons: {self.params.n_neurons}")
        print(f"  Initial K neighbors: {self.params.k_neighbors}")
        print(f"  Rewiring probability: {self.rewire_prob}")
        print(f"  Distance decay: {self.distance_decay}")
        
        # Create network
        network = CElegansNetwork()
        
        # Add neurons
        self._create_neurons(network)
        
        # Step 1: Create regular ring lattice (undirected edges stored once)
        undirected_edges = []  # (node_a, node_b, offset)
        
        for i in range(self.params.n_neurons):
            k_this = self._sample_connection_count()
            k_half = k_this // 2
            
            # Forward neighbors only; reverse direction added when materialising
            for offset in range(1, k_half + 1):
                target_idx = (i + offset) % self.params.n_neurons
                undirected_edges.append((i, target_idx, offset))
        
        # Step 2: Rewire edges with probability p
        rewired_count = 0
        existing_connections = set()
        
        for i, (source, target, offset) in enumerate(undirected_edges):
            if self.rng.random() < self.rewire_prob:
                new_target = self._choose_rewire_target(source, existing_connections)
                
                if new_target is not None:
                    undirected_edges[i] = (source, new_target, None)
                    rewired_count += 1
            
            existing_connections.add((source, undirected_edges[i][1]))
        
        # Step 3: Create connections (both directions for each undirected edge)
        for node_a, node_b, _offset in undirected_edges:
            id_a = self._get_neuron_id(node_a)
            id_b = self._get_neuron_id(node_b)
            self._add_connection(network, id_a, id_b)
            self._add_connection(network, id_b, id_a)
        
        self.network = network
        
        summary = network.get_connectivity_summary()
        actual_avg = summary['avg_synapses_per_neuron'] + summary['avg_gaps_per_neuron']
        
        print(f"  Created: {summary['n_chemical_synapses']} chemical synapses, "
              f"{summary['n_gap_junctions']} gap junctions")
        n_edges = len(undirected_edges)
        print(f"  Rewired: {rewired_count}/{n_edges} edges ({rewired_count/n_edges*100:.1f}%)")
        print(f"  Actual avg connections/neuron: {actual_avg:.1f}")
        print(f"  Topology: Small-world (Watts-Strogatz)")
        
        return network
    
    def _choose_rewire_target(self, source: int, existing: set) -> Optional[int]:
        """
        Choose a new target for rewiring with distance bias.
        
        Args:
            source: Source neuron index
            existing: Set of existing (source, target) pairs
            
        Returns:
            New target index or None if can't find valid target
        """
        for _ in range(100):
            if self.distance_decay < np.inf:
                distance = int(self.rng.exponential(self.distance_decay))
                distance = min(distance, self.params.n_neurons - 1)
                
                if self.rng.random() < 0.5:
                    distance = -distance
                
                target = (source + distance) % self.params.n_neurons
            else:
                target = self.rng.integers(0, self.params.n_neurons)
            
            if target == source and not self.allow_self_loops:
                continue
            if (source, target) in existing:
                continue
            
            return target
        
        return None


class RandomTopologyBuilder(BaseTopologyBuilder):
    """
    Build random topology network (Erdos-Renyi).
    
    Each possible connection exists with probability p.
    Can be modified with distance bias to prefer nearby or distant connections.
    
    Parameters:
        connection_prob: Probability of each possible connection existing
        distance_bias: 'uniform', 'gaussian', or 'exponential'
        distance_sigma: Controls distance preference (for gaussian/exponential)
    
    Example:
        builder = RandomTopologyBuilder(
            TopologyParameters(n_neurons=302),
            connection_prob=0.046  # 14/302 for avg 14 connections
        )
        network = builder.build()
    """
    
    def __init__(self, params: Optional[TopologyParameters] = None,
                 connection_prob: Optional[float] = None,
                 distance_bias: Literal['uniform', 'gaussian', 'exponential'] = 'uniform',
                 distance_sigma: float = 50.0,
                 allow_self_loops: bool = False):
        """
        Initialize random topology builder.
        
        Args:
            params: Topology parameters
            connection_prob: Probability of each connection (auto if None)
            distance_bias: Type of distance preference
            distance_sigma: Scale parameter for distance distribution
            allow_self_loops: Allow neurons to connect to themselves
        """
        if params is None:
            params = TopologyParameters()
        super().__init__(params)
        
        # Auto-calculate connection probability if not provided
        if connection_prob is None:
            # For average of k connections: p = k / (n-1)
            self.connection_prob = self.params.k_neighbors / (self.params.n_neurons - 1)
        else:
            self.connection_prob = connection_prob
        
        self.distance_bias = distance_bias
        self.distance_sigma = distance_sigma
        self.allow_self_loops = allow_self_loops
        
        # Validate
        if not 0 <= self.connection_prob <= 1:
            raise ValueError("connection_prob must be in [0, 1]")
    
    def build(self) -> CElegansNetwork:
        """
        Build random network.
        
        For each pair of neurons, create connection with probability p,
        optionally weighted by distance preference.
        
        Returns:
            Network with random connectivity
        """
        print(f"\nBuilding random topology network...")
        print(f"  Neurons: {self.params.n_neurons}")
        print(f"  Connection probability: {self.connection_prob:.4f}")
        print(f"  Distance bias: {self.distance_bias}")
        
        # Create network
        network = CElegansNetwork()
        
        # Add neurons
        self._create_neurons(network)
        
        # Add random connections
        total_connections = 0
        
        for i in range(self.params.n_neurons):
            source_id = self._get_neuron_id(i)
            
            for j in range(self.params.n_neurons):
                if i == j and not self.allow_self_loops:
                    continue
                
                # Compute distance on ring
                distance = min(abs(j - i), self.params.n_neurons - abs(j - i))
                
                # Connection probability with distance bias
                p_connect = self._compute_connection_probability(distance)
                
                if self.rng.random() < p_connect:
                    target_id = self._get_neuron_id(j)
                    self._add_connection(network, source_id, target_id)
                    total_connections += 1
        
        self.network = network
        
        summary = network.get_connectivity_summary()
        actual_avg = summary['avg_synapses_per_neuron'] + summary['avg_gaps_per_neuron']
        
        print(f"  Created: {summary['n_chemical_synapses']} chemical synapses, "
              f"{summary['n_gap_junctions']} gap junctions")
        print(f"  Actual avg connections/neuron: {actual_avg:.1f}")
        print(f"  Topology: Random (Erdos-Renyi with {self.distance_bias} distance bias)")
        
        return network
    
    def _compute_connection_probability(self, distance: int) -> float:
        """
        Compute connection probability based on distance.
        
        Args:
            distance: Distance between neurons (on ring)
            
        Returns:
            Connection probability
        """
        if self.distance_bias == 'uniform':
            return self.connection_prob
        elif self.distance_bias == 'gaussian':
            factor = np.exp(-distance**2 / (2 * self.distance_sigma**2))
            return self.connection_prob * factor
        elif self.distance_bias == 'exponential':
            factor = np.exp(-distance / self.distance_sigma)
            return self.connection_prob * factor
        else:
            return self.connection_prob


class ScaleFreeBuilder(BaseTopologyBuilder):
    """
    Build scale-free topology network using Barabasi-Albert preferential attachment.

    New nodes preferentially attach to high-degree nodes, producing a
    power-law degree distribution with prominent hub neurons.  This contrasts
    with small-world networks (high clustering) and uniform networks (no hubs).

    Parameters:
        m: Number of edges each new node creates (controls avg degree ~2*m).
        m0: Number of seed nodes in the initial fully-connected clique.

    Example:
        builder = ScaleFreeBuilder(
            TopologyParameters(n_neurons=302, k_neighbors=14),
            m=7
        )
        network = builder.build()
    """

    def __init__(self, params: Optional[TopologyParameters] = None,
                 m: Optional[int] = None,
                 m0: Optional[int] = None):
        """
        Initialize scale-free topology builder.

        Args:
            params: Topology parameters
            m: Edges per new node (default: k_neighbors // 2 to approximate target avg degree)
            m0: Seed clique size (default: m + 1)
        """
        if params is None:
            params = TopologyParameters()
        super().__init__(params)

        self.m = m if m is not None else max(1, self.params.k_neighbors // 2)
        self.m0 = m0 if m0 is not None else self.m + 1

        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.m0 < self.m:
            raise ValueError("m0 must be >= m")
        if self.m0 > self.params.n_neurons:
            raise ValueError("m0 must be <= n_neurons")

    def build(self) -> CElegansNetwork:
        """
        Build scale-free network via Barabasi-Albert algorithm.

        1. Create a fully-connected seed clique of m0 nodes.
        2. For each remaining node, connect to m existing nodes chosen with
           probability proportional to their current degree.

        Returns:
            Network with scale-free (power-law) degree distribution
        """
        print(f"\nBuilding scale-free topology network...")
        print(f"  Neurons: {self.params.n_neurons}")
        print(f"  Edges per new node (m): {self.m}")
        print(f"  Seed clique size (m0): {self.m0}")

        network = CElegansNetwork()
        self._create_neurons(network)

        degrees = np.zeros(self.params.n_neurons, dtype=float)

        # Seed clique: fully connect m0 nodes
        for i in range(self.m0):
            for j in range(i + 1, self.m0):
                src_id = self._get_neuron_id(i)
                tgt_id = self._get_neuron_id(j)
                self._add_connection(network, src_id, tgt_id)
                degrees[i] += 1
                degrees[j] += 1

        # Preferential attachment for remaining nodes
        for new_idx in range(self.m0, self.params.n_neurons):
            total_degree = degrees[:new_idx].sum()
            if total_degree == 0:
                probs = np.ones(new_idx) / new_idx
            else:
                probs = degrees[:new_idx] / total_degree

            targets = set()
            attempts = 0
            while len(targets) < min(self.m, new_idx) and attempts < self.m * 20:
                chosen = self.rng.choice(new_idx, p=probs)
                targets.add(chosen)
                attempts += 1

            new_id = self._get_neuron_id(new_idx)
            for t in targets:
                tgt_id = self._get_neuron_id(t)
                self._add_connection(network, new_id, tgt_id)
                degrees[new_idx] += 1
                degrees[t] += 1

        self.network = network

        summary = network.get_connectivity_summary()
        actual_avg = summary['avg_synapses_per_neuron'] + summary['avg_gaps_per_neuron']

        print(f"  Created: {summary['n_chemical_synapses']} chemical synapses, "
              f"{summary['n_gap_junctions']} gap junctions")
        print(f"  Actual avg connections/neuron: {actual_avg:.1f}")
        print(f"  Topology: Scale-free (Barabási-Albert)")

        return network


# Convenience functions for easy usage

def build_uniform_network(n_neurons: int = 302,
                         k_neighbors: int = 14,
                         neuron_class: str = 'interneuron',
                         weight_mean: float = 2.0,
                         weight_std: float = 0.3,
                         gap_prob: float = 0.15,
                         connection_variance: float = 2.0,
                         random_seed: Optional[int] = None) -> CElegansNetwork:
    """
    Build uniform topology C. elegans network (one-liner convenience function).
    
    Args:
        n_neurons: Number of neurons (default: 302)
        k_neighbors: Average connections per neuron (default: 14)
        neuron_class: 'sensory', 'interneuron', 'motor', or 'mixed'
        weight_mean: Mean synaptic weight in nS
        weight_std: Std dev of synaptic weights
        gap_prob: Probability a connection is a gap junction
        connection_variance: Variability in connection count per neuron
        random_seed: Random seed for reproducibility
        
    Returns:
        Network with uniform topology
        
    Example:
        network = build_uniform_network(n_neurons=302, k_neighbors=14)
    """
    params = TopologyParameters(
        n_neurons=n_neurons,
        neuron_class=neuron_class,
        k_neighbors=k_neighbors,
        connection_variance=connection_variance,
        weight_mean=weight_mean,
        weight_std=weight_std,
        gap_junction_prob=gap_prob,
        random_seed=random_seed
    )
    
    builder = UniformTopologyBuilder(params)
    return builder.build()


def build_small_world_network(n_neurons: int = 302,
                              k_neighbors: int = 14,
                              rewire_prob: float = 0.2,
                              neuron_class: str = 'interneuron',
                              weight_mean: float = 2.0,
                              weight_std: float = 0.3,
                              gap_prob: float = 0.15,
                              distance_decay: float = 50.0,
                              random_seed: Optional[int] = None) -> CElegansNetwork:
    """
    Build small-world topology C. elegans network (one-liner convenience function).
    
    Args:
        n_neurons: Number of neurons (default: 302)
        k_neighbors: Initial regular connections per neuron (default: 14)
        rewire_prob: Probability of rewiring each edge (default: 0.2)
        neuron_class: 'sensory', 'interneuron', 'motor', or 'mixed'
        weight_mean: Mean synaptic weight in nS
        weight_std: Std dev of synaptic weights
        gap_prob: Probability a connection is a gap junction
        distance_decay: Preference for distant rewiring (lower = more distant)
        random_seed: Random seed for reproducibility
        
    Returns:
        Network with small-world topology
        
    Example:
        network = build_small_world_network(n_neurons=302, rewire_prob=0.2)
    """
    params = TopologyParameters(
        n_neurons=n_neurons,
        neuron_class=neuron_class,
        k_neighbors=k_neighbors,
        connection_variance=2.0,
        weight_mean=weight_mean,
        weight_std=weight_std,
        gap_junction_prob=gap_prob,
        random_seed=random_seed
    )
    
    builder = SmallWorldBuilder(params, rewire_prob=rewire_prob, 
                               distance_decay=distance_decay)
    return builder.build()


def build_random_network(n_neurons: int = 302,
                        k_neighbors: int = 14,
                        neuron_class: str = 'interneuron',
                        distance_bias: str = 'uniform',
                        distance_sigma: float = 50.0,
                        random_seed: Optional[int] = None) -> CElegansNetwork:
    """
    Build random topology C. elegans network (one-liner convenience function).
    
    Args:
        n_neurons: Number of neurons (default: 302)
        k_neighbors: Average connections per neuron (default: 14)
        neuron_class: 'sensory', 'interneuron', 'motor', or 'mixed'
        distance_bias: 'uniform', 'gaussian', or 'exponential'
        distance_sigma: Distance preference scale
        random_seed: Random seed for reproducibility
        
    Returns:
        Network with random topology
        
    Example:
        network = build_random_network(n_neurons=302, distance_bias='gaussian')
    """
    params = TopologyParameters(
        n_neurons=n_neurons,
        neuron_class=neuron_class,
        k_neighbors=k_neighbors,
        random_seed=random_seed
    )
    
    # Calculate connection probability for target average
    connection_prob = k_neighbors / (n_neurons - 1)
    
    builder = RandomTopologyBuilder(params, connection_prob=connection_prob,
                                   distance_bias=distance_bias,
                                   distance_sigma=distance_sigma)
    return builder.build()


def build_scale_free_network(n_neurons: int = 302,
                             m: Optional[int] = None,
                             neuron_class: str = 'interneuron',
                             weight_mean: float = 2.0,
                             weight_std: float = 0.3,
                             gap_prob: float = 0.15,
                             random_seed: Optional[int] = None) -> CElegansNetwork:
    """
    Build scale-free topology C. elegans network (one-liner convenience function).

    Args:
        n_neurons: Number of neurons (default: 302)
        m: Edges per new node (default: ~k/2 for avg degree ~k)
        neuron_class: 'sensory', 'interneuron', 'motor', or 'mixed'
        weight_mean: Mean synaptic weight in nS
        weight_std: Std dev of synaptic weights
        gap_prob: Probability a connection is a gap junction
        random_seed: Random seed for reproducibility

    Returns:
        Network with scale-free (power-law) degree distribution

    Example:
        network = build_scale_free_network(n_neurons=302, m=7)
    """
    params = TopologyParameters(
        n_neurons=n_neurons,
        neuron_class=neuron_class,
        k_neighbors=14,
        weight_mean=weight_mean,
        weight_std=weight_std,
        gap_junction_prob=gap_prob,
        random_seed=random_seed
    )

    builder = ScaleFreeBuilder(params, m=m)
    return builder.build()

