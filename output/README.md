# SDMN Framework - Output Directory

This directory contains generated outputs from SDMN simulations and examples.

## Generated Files

When you run examples, output files will be automatically created here:

### Example Outputs

- `01_neuron_comparison.png` - LIF vs Hodgkin-Huxley neuron comparison
- `02_network_comparison.png` - Network topology analysis  
- `02_raster_comparison.png` - Spike raster plots for different topologies
- `sdmn_quickstart_results.png` - Quickstart simulation results
- `working_neuron_demo.png` - Comprehensive neural network activity demo

### Plot Characteristics

All plots are:
- **High resolution**: 300 DPI for publication quality
- **Professional formatting**: Proper labels, legends, grids
- **Color-coded**: Clear visual distinction between data series
- **Multiple subplots**: Comprehensive analysis in single figure

## Usage

Examples automatically create this directory and save plots here:

```python
python examples/01_basic_neuron_demo.py      # Creates 01_neuron_comparison.png
python examples/02_network_topologies.py     # Creates 02_*.png files
python examples/quickstart_simulation.py     # Creates sdmn_quickstart_results.png
```

## File Management

- **Generated files are ignored by git** (see `.gitignore`)
- **Directory structure is preserved** (via `.gitkeep`)
- **Safe to delete** - files will be regenerated when examples run
- **Not distributed** - excluded from package distribution

---

*This directory is for local development and research outputs only.*

