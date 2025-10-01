"""Command line interface for the SDMN framework."""

import click
import sys
from pathlib import Path
from typing import Optional

from sdmn.version import __version__
from sdmn.core import SimulationEngine, SimulationConfig
from sdmn.networks import NetworkBuilder, NetworkConfiguration, NetworkTopology
from sdmn.neurons import NeuronType


@click.group()
@click.version_option(__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, verbose):
    """Synthetic Default Mode Network Framework CLI."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@main.command()
def info():
    """Display package information."""
    from sdmn import get_info
    info_dict = get_info()
    click.echo(f"Package: {info_dict['name']}")
    click.echo(f"Version: {info_dict['version']}")
    click.echo(f"Description: {info_dict['description']}")
    click.echo(f"License: {info_dict['license']}")


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path')
@click.option('--output', '-o', type=click.Path(), 
              help='Output directory for results')
@click.option('--duration', '-d', default=1000.0, type=float,
              help='Simulation duration (ms)')
@click.option('--dt', default=0.1, type=float,
              help='Time step size (ms)')
@click.option('--neurons', '-n', default=100, type=int,
              help='Number of neurons in network')
@click.option('--topology', '-t', 
              type=click.Choice(['random', 'small_world', 'ring', 'grid_2d']),
              default='small_world', help='Network topology')
@click.option('--neuron-type', 
              type=click.Choice(['lif', 'hh']),
              default='lif', help='Neuron model type')
def simulate(config: Optional[str], output: Optional[str], duration: float,
             dt: float, neurons: int, topology: str, neuron_type: str):
    """Run a neural network simulation."""
    click.echo(f"Starting simulation with {neurons} {neuron_type} neurons...")
    
    # Create simulation configuration
    sim_config = SimulationConfig(
        dt=dt,
        max_time=duration,
        enable_logging=True,
        log_level="INFO"
    )
    
    # Create simulation engine
    engine = SimulationEngine(sim_config)
    
    # Create network configuration
    neuron_type_map = {
        'lif': NeuronType.LEAKY_INTEGRATE_FIRE,
        'hh': NeuronType.HODGKIN_HUXLEY
    }
    
    topology_map = {
        'random': NetworkTopology.RANDOM,
        'small_world': NetworkTopology.SMALL_WORLD,
        'ring': NetworkTopology.RING,
        'grid_2d': NetworkTopology.GRID_2D
    }
    
    network_config = NetworkConfiguration(
        name="cli_network",
        n_neurons=neurons,
        topology=topology_map[topology],
        neuron_type=neuron_type_map[neuron_type],
        connection_probability=0.1,
        weight_range=(0.5, 2.0),
        delay_range=(1.0, 10.0)
    )
    
    # Build network
    builder = NetworkBuilder()
    network = builder.create_network(network_config)
    
    # Add network to engine
    engine.add_network("main_network", network)
    
    # Run simulation
    try:
        results = engine.run()
        
        if results.success:
            click.echo(f"Simulation completed successfully!")
            click.echo(f"Total steps: {results.total_steps}")
            click.echo(f"Simulation time: {results.simulation_time:.1f} ms")
            click.echo(f"Wall time: {results.wall_time:.2f} s")
            
            if output:
                output_path = Path(output)
                output_path.mkdir(parents=True, exist_ok=True)
                click.echo(f"Results saved to: {output_path}")
                
        else:
            click.echo(f"Simulation failed: {results.error_message}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        click.echo("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Simulation error: {str(e)}")
        sys.exit(1)


@main.command()
def examples():
    """List available examples."""
    click.echo("Available examples:")
    click.echo("  01_basic_neuron_demo.py - Basic neuron model demonstration")
    click.echo("  02_network_topologies.py - Different network topologies")
    click.echo("  03_probe_monitoring.py - Monitoring with probes")
    click.echo("  04_default_mode_networks.py - Default mode network simulation")
    click.echo("  05_self_aware_network.py - Self-aware network dynamics")
    click.echo("  quickstart_simulation.py - Quick start simulation")


@main.command()
@click.argument('example_name')
@click.option('--output', '-o', type=click.Path(), 
              help='Output directory for results')
def run_example(example_name: str, output: Optional[str]):
    """Run a specific example."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    example_file = examples_dir / f"{example_name}"
    
    if not example_file.exists():
        click.echo(f"Example '{example_name}' not found.")
        click.echo("Use 'sdmn examples' to list available examples.")
        sys.exit(1)
    
    click.echo(f"Running example: {example_name}")
    
    # Execute the example file
    import subprocess
    try:
        result = subprocess.run([sys.executable, str(example_file)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            click.echo("Example completed successfully!")
            if result.stdout:
                click.echo(result.stdout)
        else:
            click.echo(f"Example failed with error:")
            click.echo(result.stderr)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error running example: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
