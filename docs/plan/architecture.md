# Architecture Document: Synthetic Default Mode Network Framework

## System Overview

The SDMN Framework is designed as a modular, scalable system for simulating complex spiking neural networks. The architecture emphasizes separation of concerns, extensibility, and performance optimization while maintaining scientific rigor and reproducibility.

## Core Components

### 1. Simulation Engine (`src/core/`)

The heart of the framework, responsible for:
- Time-stepped simulation management
- Event scheduling and processing
- State management across all network components
- Performance optimization and parallel processing coordination

#### Key Classes:
- `SimulationEngine`: Main simulation controller
- `TimeManager`: Handles simulation time progression and scheduling
- `EventQueue`: Manages spike events and other time-based events
- `StateManager`: Manages global and local state across all components

### 2. Neuron Models (`src/neurons/`)

Biologically-inspired neuron implementations with standardized interfaces:
- Base abstract classes defining neuron behavior contracts
- Specific implementations (Integrate-and-Fire, Hodgkin-Huxley, etc.)
- Synaptic models and plasticity mechanisms
- Supporting cellular models (astrocytes, oligodendrocytes)

#### Interface Design:
```python
class BaseNeuron(ABC):
    @abstractmethod
    def update(self, dt: float, inputs: List[float]) -> None:
        """Update neuron state for one time step"""
        pass
    
    @abstractmethod
    def get_membrane_potential(self) -> float:
        """Get current membrane potential"""
        pass
    
    @abstractmethod
    def has_spiked(self) -> bool:
        """Check if neuron has spiked in current time step"""
        pass
```

### 3. Network Architecture (`src/networks/`)

Tools for building and managing complex neural networks:
- Network topology management
- Connection matrices and synaptic weight handling
- Population-level organization
- Connectivity patterns (random, small-world, scale-free)

#### Key Components:
- `Network`: Main network container
- `Population`: Groups of similar neurons
- `Connection`: Synaptic connections between neurons/populations
- `Topology`: Network structure definitions

### 4. Probe System (`src/probes/`)

Comprehensive monitoring and data collection:
- Voltage probes for membrane potential tracking
- Spike detectors for action potential recording
- Population activity monitors
- Custom probe implementations for specific research needs

#### Probe Architecture:
```python
class BaseProbe(ABC):
    @abstractmethod
    def record(self, target: Any, timestamp: float) -> None:
        """Record data from target at given timestamp"""
        pass
    
    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """Retrieve recorded data"""
        pass
```

### 5. Visualization System (`src/visualization/`)

Multi-modal visualization capabilities:
- Real-time network activity visualization
- Post-simulation analysis and plotting
- 3D network structure rendering
- EEG-like signal analysis and frequency domain analysis

## Data Flow Architecture

```
Input Stimuli → Simulation Engine → Network Updates → Probe Recording → Visualization/Analysis
     ↑              ↓                      ↑                ↓
User Config → Time Manager ← State Manager ← Event Queue ← Output Processing
```

### Simulation Loop

1. **Initialization Phase**:
   - Load network configuration
   - Initialize neuron states
   - Set up probe systems
   - Configure simulation parameters

2. **Simulation Phase** (repeated for each time step):
   - Process input stimuli
   - Update neuron states
   - Handle spike propagation
   - Record probe data
   - Update visualization (if real-time)

3. **Analysis Phase**:
   - Process recorded data
   - Generate visualizations
   - Export results
   - Cleanup resources

## Interface Standardization

### Neuron Interface Contract

All neuron models must implement:
- **State Management**: Consistent state representation and serialization
- **Update Protocol**: Standardized time-step update mechanism
- **Spike Generation**: Uniform spike detection and propagation
- **Parameter Access**: Consistent parameter getting/setting interface
- **Introspection**: Ability to query internal state for debugging/analysis

### Network Interface Standards

- **Connection Protocol**: Standardized way to connect neurons/populations
- **Topology Definition**: Consistent network structure representation
- **Scaling Interface**: Methods for network growth and modification
- **Serialization**: Network save/load functionality

### Probe Interface Requirements

- **Registration System**: Standardized probe attachment to network elements
- **Data Format**: Consistent data structure for all probe types
- **Temporal Alignment**: Synchronized timing across all probes
- **Export Interface**: Uniform data export and analysis integration

## Backward Compatibility Strategy

### Version Management
- Semantic versioning (MAJOR.MINOR.PATCH)
- Deprecation warnings for interface changes
- Migration tools for major version upgrades

### Interface Stability
- Core interfaces marked as stable vs. experimental
- Abstract base classes provide stable contracts
- Extension points for new functionality without breaking changes

### Data Compatibility
- Forward-compatible data formats
- Schema versioning for network definitions
- Migration utilities for older data formats

## Performance Optimization

### Current Implementation (Python)
- NumPy vectorization for mathematical operations
- Efficient event-driven simulation
- Memory-mapped files for large datasets
- Multiprocessing support for parallel neuron updates

### Future Migration Path (C++/CUDA)
- Identified performance-critical components
- Clear interfaces for language-agnostic implementation
- GPU acceleration for massively parallel operations
- Hybrid Python/C++ architecture maintaining ease of use

## Scalability Architecture

### Horizontal Scaling
- Distributed simulation across multiple nodes
- Network partitioning strategies
- Inter-node communication protocols
- Load balancing for heterogeneous networks

### Vertical Scaling
- Multi-threading for single-node performance
- GPU acceleration for parallel operations
- Memory optimization for large networks
- Efficient data structures for sparse connections

## Security and Reproducibility

### Reproducible Research
- Deterministic random number generation
- Complete simulation state serialization
- Version-controlled configuration management
- Automated testing and validation

### Data Security
- Secure handling of sensitive neural data
- Encryption for stored simulations
- Access control for shared resources
- Audit trails for research integrity

## Extension Points

### Plugin Architecture
- Dynamic loading of neuron models
- Custom probe implementations
- Visualization plugin system
- Analysis tool integration

### API Design
- RESTful API for remote simulation control
- WebSocket connections for real-time monitoring
- GraphQL interface for complex queries
- Command-line interface for batch processing

## Quality Assurance

### Testing Strategy
- Unit tests for all core components
- Integration tests for system interactions
- Performance benchmarks and regression testing
- Biological validation against known neural data

### Documentation Standards
- Comprehensive API documentation
- Architecture decision records (ADRs)
- Code examples and tutorials
- Scientific methodology documentation

---

This architecture provides a solid foundation for building sophisticated neural network simulations while maintaining flexibility for future research directions and technological advancement.
