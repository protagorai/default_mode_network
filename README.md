<div align="center">
  <img src="assets/logo/logo-wave.svg" width="256" height="256" alt="SDMN Framework - Triangulated Brain Network Logo">
  <br>
  
# Synthetic Default Mode Network (SDMN) Framework

[![CI](https://github.com/username/synthetic-default-mode-network/workflows/CI/badge.svg)](https://github.com/username/synthetic-default-mode-network/actions)
[![codecov](https://codecov.io/gh/username/synthetic-default-mode-network/branch/main/graph/badge.svg)](https://codecov.io/gh/username/synthetic-default-mode-network)
[![PyPI version](https://badge.fury.io/py/synthetic-default-mode-network.svg)](https://badge.fury.io/py/synthetic-default-mode-network)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*A comprehensive computational neuroscience platform for building, studying, and analyzing artificial brain networks with surprisingly lifelike neural dynamics.*

</div>

## Overview

The **Synthetic Default Mode Network (SDMN) Framework** is a cutting-edge research platform designed to simulate biologically-inspired artificial spiking neural networks that emulate the complexity and connectivity of human brain networks. The framework specifically focuses on modeling default mode networks (DMNs) - the brain regions active during rest and introspection - using sophisticated spiking neuron models like Leaky Integrate-and-Fire (LIF) and Hodgkin-Huxley neurons.

## Key Features

✨ **Biologically-Inspired Neural Models**
- Advanced spiking neuron models (LIF, Hodgkin-Huxley)
- Realistic synaptic dynamics with plasticity
- Biophysically accurate membrane dynamics

🔬 **Comprehensive Simulation Engine** 
- High-performance event-driven simulation
- Flexible time management with adaptive stepping
- State checkpointing and recovery

🌐 **Network Topology Generation**
- Multiple network topologies (random, small-world, scale-free, grid)
- Modular network assembly with connectivity patterns
- Population-based network organization

📊 **Advanced Monitoring System**
- Real-time voltage, spike, and population activity probes  
- Local field potential (LFP) simulation
- Network-wide connectivity and synchronization analysis

🎯 **Self-Awareness Research Platform**
- Self-monitoring and risk-reward assessment capabilities
- Internal narrative construction mechanisms
- Self-referential processing for consciousness studies

## Quick Start

> **📖 For detailed installation instructions, see [INSTALL.md](docs/INSTALL.md)**

### Installation

#### Using pip (recommended)
```bash
pip install synthetic-default-mode-network
```

#### Using Poetry (for development)
```bash
git clone https://github.com/username/synthetic-default-mode-network.git
cd synthetic-default-mode-network

# Smart platform-aware setup (recommended)
./scripts/setup_platform.sh

# Or choose platform-specific scripts:
./scripts/setup_development.sh        # Linux/Unix
./scripts/setup_development_macos.sh  # macOS with Homebrew
# scripts\setup_development.bat       # Windows (PowerShell/CMD)
```

#### Using Docker
```bash
docker pull ghcr.io/username/synthetic-default-mode-network:latest
docker run -it ghcr.io/username/synthetic-default-mode-network:latest
```

### Basic Usage

```python
import sdmn

# Create simulation configuration
config = sdmn.SimulationConfig(dt=0.1, max_time=1000.0)
engine = sdmn.SimulationEngine(config)

# Build a small-world network
network_config = sdmn.NetworkConfiguration(
    name="example_network",
    n_neurons=100,
    topology=sdmn.NetworkTopology.SMALL_WORLD,
    neuron_type=sdmn.NeuronType.LEAKY_INTEGRATE_FIRE
)

builder = sdmn.NetworkBuilder()
network = builder.create_network(network_config)
engine.add_network("main", network)

# Add monitoring probes
voltage_probe = sdmn.VoltageProbe("voltage", list(network.neurons.keys())[:5])
spike_probe = sdmn.SpikeProbe("spikes", list(network.neurons.keys()))

engine.add_probe("voltage", voltage_probe)
engine.add_probe("spikes", spike_probe)

# Run simulation
results = engine.run()
print(f"Simulation completed: {results.total_steps} steps in {results.wall_time:.2f}s")
```

### Command Line Interface

```bash
# Get package information
sdmn info

# Run a simulation
sdmn simulate --neurons 200 --topology small_world --duration 2000

# List available examples
sdmn examples

# Run a specific example
sdmn run-example quickstart_simulation.py
```

## Project Structure

```
synthetic-default-mode-network/
├── src/sdmn/                     # Main package
│   ├── core/                     # Simulation engine
│   ├── neurons/                  # Neuron models
│   ├── networks/                 # Network builders
│   ├── probes/                   # Monitoring system
│   └── cli.py                    # Command line interface
├── tests/                        # Comprehensive test suite
│   ├── test_core.py             # Core engine tests
│   ├── test_neurons.py          # Neuron model tests  
│   ├── test_networks.py         # Network tests
│   ├── test_probes.py           # Probe tests
│   └── test_integration.py      # Integration tests
├── examples/                     # Usage examples
├── docs/                         # Documentation
├── .github/workflows/            # GitHub Actions CI/CD
├── .gitlab-ci.yml               # GitLab CI/CD
├── Dockerfile                    # Docker configuration
├── pyproject.toml               # Poetry configuration
└── README.md                    # This file
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/username/synthetic-default-mode-network.git
cd synthetic-default-mode-network

# Smart platform-aware setup (recommended)
./scripts/setup_platform.sh

# Or platform-specific setup
./scripts/setup_development.sh        # Linux/Unix
./scripts/setup_development_macos.sh  # macOS with Homebrew
# scripts\setup_development.bat       # Windows native

# Verify installation
python scripts/verify_installation.py

# Start development
poetry shell                          # Activate environment
poetry run pytest                     # Run tests
python examples/01_basic_neuron_demo.py  # Test with example
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run only unit tests
poetry run pytest -m unit

# Run only integration tests  
poetry run pytest -m integration

# Run with coverage
poetry run pytest --cov=src --cov-report=html
```

### Docker Development

```bash
# Interactive setup assistant
./scripts/setup.sh

# Or build and run directly
./scripts/build.sh
./scripts/run.sh dev          # Development mode
./scripts/run.sh jupyter      # Jupyter Lab
./scripts/run.sh simulation   # Run simulation
./scripts/run.sh test         # Run tests
```

### Deployment Options

The SDMN Framework supports multiple deployment strategies:

| Strategy | Best For | Setup Command |
|----------|----------|---------------|
| **Local Development** | Active development, debugging | `./scripts/setup_development.sh` |
| **Local Production** | Production servers | `./scripts/setup_production.sh` |
| **Containerized** | Cloud, K8s, teams | `./scripts/build.sh && ./scripts/run.sh` |

See [`scripts/README.md`](scripts/README.md) for detailed comparison.

## Architecture

### Core Components

- **SimulationEngine**: Main simulation coordinator with time management and event processing
- **Neuron Models**: Biologically-inspired spiking neurons (LIF, Hodgkin-Huxley)
- **Network Builder**: Tools for creating complex network topologies  
- **Probe System**: Comprehensive monitoring and data collection framework
- **State Management**: Checkpointing and recovery for long simulations

### Key Design Principles

- **Modularity**: Interchangeable components with standardized interfaces
- **Performance**: Event-driven simulation with optimized data structures
- **Extensibility**: Plugin architecture for custom neuron models and probes
- **Reproducibility**: Deterministic simulations with configurable random seeds

## Research Applications

This framework supports research into:

- **Neural Oscillations**: Synthetic EEG-like brain wave generation and analysis
- **Default Mode Networks**: Simulation of rest-state brain network dynamics  
- **Network Neuroscience**: Large-scale connectivity patterns and graph analysis
- **Consciousness Studies**: Self-awareness mechanisms in artificial neural systems
- **Neuromorphic Computing**: Brain-inspired computing architectures

## Performance

The framework is optimized for:

- **Scalability**: Networks up to 100K+ neurons on standard hardware
- **Speed**: Real-time simulation of medium networks (1K neurons)
- **Memory Efficiency**: Minimal memory footprint with optional data compression
- **Parallel Processing**: Multi-core CPU utilization and GPU acceleration (planned)

## Documentation

- [**API Reference**](https://sdmn-docs.github.io/api/) - Complete API documentation
- [**User Guide**](https://sdmn-docs.github.io/guide/) - Comprehensive usage guide  
- [**Examples**](examples/) - Working code examples
- [**Architecture**](docs/plan/architecture.md) - System design documentation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run the full test suite (`poetry run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Continuous Integration

The project uses comprehensive CI/CD pipelines:

- **GitHub Actions**: Multi-platform testing, code quality checks, automated releases
- **GitLab CI**: Additional testing, security scanning, Docker image builds
- **Pre-commit Hooks**: Code formatting, linting, and basic testing
- **Automated Testing**: Unit, integration, and performance tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{sdmn_framework,
  author = {SDMN Development Team},
  title = {Synthetic Default Mode Network Framework},
  url = {https://github.com/username/synthetic-default-mode-network},
  version = {0.1.0},
  year = {2024}
}
```

## Support

- 📖 [Documentation](https://sdmn-docs.github.io)
- 💬 [Discussions](https://github.com/username/synthetic-default-mode-network/discussions)  
- 🐛 [Issue Tracker](https://github.com/username/synthetic-default-mode-network/issues)
- 📧 [Email Support](mailto:support@sdmn-framework.org)

## Acknowledgments

- Inspired by computational neuroscience research and the brain's default mode network
- Built with modern Python development practices and tools
- Thanks to the open-source scientific computing community

---

*"Advancing our understanding of neural networks through synthetic biology-inspired simulation, one artificial synapse at a time."*
