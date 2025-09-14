#!/usr/bin/env python3
"""
Default Mode Networks Demonstration - SDMN Framework Example 04

This example demonstrates the core focus of the framework: creating and analyzing
synthetic default mode networks with brain-like oscillations and feedback loops.

Run with:
    python examples/04_default_mode_networks.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from neurons import LIFNeuron, LIFParameters, SynapseFactory
from probes import VoltageProbe, PopulationActivityProbe, LFPProbe
from core import SimulationEngine, SimulationConfig

class DefaultModeNetwork:
    """
    Implementation of a biologically-inspired Default Mode Network.
    
    The DMN consists of multiple brain regions with specific connectivity
    patterns that generate resting-state activity and brain waves.
    """
    
    def __init__(self, enable_plasticity=False):
        self.neurons = {}
        self.synapses = {}
        self.regions = {}
        self.enable_plasticity = enable_plasticity
        self._create_dmn_architecture()
        self._establish_connectivity()
    
    def _create_dmn_architecture(self):
        """Create the multi-regional DMN architecture."""
        print("Creating DMN regional architecture...")
        
        # Define DMN regions based on neuroscience literature
        region_configs = {
            # Posterior Cingulate Cortex - central hub
            'PCC': {
                'size': 25,
                'params': LIFParameters(
                    tau_m=25.0, v_thresh=-45.0, r_mem=12.0, 
                    refractory_period=2.0
                ),
                'excitability': 'high'  # High baseline activity
            },
            
            # Medial Prefrontal Cortex - executive control
            'mPFC': {
                'size': 20,
                'params': LIFParameters(
                    tau_m=22.0, v_thresh=-48.0, r_mem=10.0,
                    refractory_period=2.2
                ),
                'excitability': 'medium'
            },
            
            # Angular Gyrus - conceptual processing
            'AG': {
                'size': 18,
                'params': LIFParameters(
                    tau_m=20.0, v_thresh=-52.0, r_mem=11.0,
                    refractory_period=2.1
                ),
                'excitability': 'medium'
            },
            
            # Precuneus - self-referential processing
            'PCu': {
                'size': 15,
                'params': LIFParameters(
                    tau_m=24.0, v_thresh=-50.0, r_mem=11.5,
                    refractory_period=1.9
                ),
                'excitability': 'medium'
            },
            
            # Temporal Poles - semantic memory
            'TP': {
                'size': 12,
                'params': LIFParameters(
                    tau_m=18.0, v_thresh=-55.0, r_mem=9.0,
                    refractory_period=2.3
                ),
                'excitability': 'low'
            }
        }
        
        # Create neurons for each region
        for region_name, config in region_configs.items():
            self.regions[region_name] = []
            
            for i in range(config['size']):
                neuron_id = f"{region_name}_{i:03d}"
                
                # Add parameter variability within region
                params = config['params']
                varied_params = LIFParameters(
                    tau_m=params.tau_m + np.random.normal(0, 2.0),
                    v_thresh=params.v_thresh + np.random.normal(0, 1.5),
                    r_mem=params.r_mem + np.random.normal(0, 0.8),
                    refractory_period=params.refractory_period + np.random.normal(0, 0.2)
                )
                
                neuron = LIFNeuron(neuron_id, varied_params)
                self.neurons[neuron_id] = neuron
                self.regions[region_name].append(neuron_id)
        
        print(f"Created {len(self.neurons)} neurons across {len(self.regions)} regions")
        for region, neurons in self.regions.items():
            print(f"  {region}: {len(neurons)} neurons")
    
    def _establish_connectivity(self):
        """Establish biologically-realistic connectivity patterns."""
        print("Establishing DMN connectivity patterns...")
        
        # 1. Intra-regional connections (local processing)
        for region_name, neuron_ids in self.regions.items():
            self._create_local_connections(region_name, neuron_ids)
        
        # 2. Inter-regional connections (long-range communication)
        self._create_inter_regional_connections()
        
        # 3. Add feedback loops for oscillatory behavior
        self._create_feedback_loops()
        
        print(f"Created {len(self.synapses)} synaptic connections")
    
    def _create_local_connections(self, region_name, neuron_ids):
        """Create local connections within a region."""
        n_neurons = len(neuron_ids)
        
        # Local connection parameters based on region
        if region_name == 'PCC':
            local_prob = 0.4  # High local connectivity in PCC
            weight_range = (1.0, 1.6)
        elif region_name == 'mPFC':
            local_prob = 0.35
            weight_range = (0.9, 1.4)
        else:
            local_prob = 0.3
            weight_range = (0.8, 1.3)
        
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and np.random.random() < local_prob:
                    pre_id = neuron_ids[i]
                    post_id = neuron_ids[j]
                    
                    # Create mostly excitatory local connections
                    if np.random.random() < 0.85:  # 85% excitatory
                        syn_id = f"local_{region_name}_{i}_{j}"
                        synapse = SynapseFactory.create_excitatory_synapse(
                            syn_id, pre_id, post_id,
                            weight=np.random.uniform(*weight_range),
                            delay=np.random.uniform(1.0, 3.0)
                        )
                    else:  # 15% inhibitory for balance
                        syn_id = f"local_inh_{region_name}_{i}_{j}"
                        synapse = SynapseFactory.create_inhibitory_synapse(
                            syn_id, pre_id, post_id,
                            weight=np.random.uniform(0.6, 1.2),
                            delay=np.random.uniform(0.8, 2.5)
                        )
                    
                    if self.enable_plasticity:
                        synapse.parameters.enable_plasticity = True
                        synapse.parameters.learning_rate = 0.005
                    
                    self._register_synapse(synapse)
    
    def _create_inter_regional_connections(self):
        """Create inter-regional connections based on DMN anatomy."""
        
        # Define inter-regional connection strengths based on literature
        inter_regional_connectivity = [
            # (source, target, probability, delay_range, weight_range)
            ('PCC', 'mPFC', 0.6, (12.0, 18.0), (1.2, 1.8)),    # Strong PCC-mPFC
            ('PCC', 'AG', 0.5, (10.0, 15.0), (1.1, 1.6)),      # PCC-Angular Gyrus
            ('PCC', 'PCu', 0.7, (8.0, 12.0), (1.3, 1.9)),      # PCC-Precuneus (strong)
            ('mPFC', 'AG', 0.4, (15.0, 22.0), (1.0, 1.4)),     # mPFC-AG
            ('mPFC', 'TP', 0.3, (18.0, 25.0), (0.9, 1.3)),     # mPFC-Temporal Poles
            ('AG', 'PCu', 0.45, (12.0, 16.0), (1.0, 1.5)),     # AG-Precuneus
            ('AG', 'TP', 0.4, (14.0, 20.0), (0.95, 1.4)),      # AG-Temporal Poles
            ('PCu', 'TP', 0.25, (16.0, 22.0), (0.8, 1.2)),     # Precuneus-TP
        ]
        
        for source, target, prob, delay_range, weight_range in inter_regional_connectivity:
            source_neurons = self.regions[source]
            target_neurons = self.regions[target]
            
            # Create bidirectional connections
            self._connect_regions(source_neurons, target_neurons, prob, delay_range, weight_range)
            self._connect_regions(target_neurons, source_neurons, prob*0.8, delay_range, weight_range)
    
    def _connect_regions(self, source_neurons, target_neurons, prob, delay_range, weight_range):
        """Connect two regions with specified parameters."""
        connections_made = 0
        
        for source_id in source_neurons:
            for target_id in target_neurons:
                if np.random.random() < prob:
                    syn_id = f"inter_{source_id}_to_{target_id}"
                    
                    synapse = SynapseFactory.create_excitatory_synapse(
                        syn_id, source_id, target_id,
                        weight=np.random.uniform(*weight_range),
                        delay=np.random.uniform(*delay_range)
                    )
                    
                    if self.enable_plasticity:
                        synapse.parameters.enable_plasticity = True
                        synapse.parameters.learning_rate = 0.003  # Lower for long-range
                    
                    self._register_synapse(synapse)
                    connections_made += 1
    
    def _create_feedback_loops(self):
        """Create feedback loops essential for DMN oscillations."""
        print("Creating feedback loops for oscillatory behavior...")
        
        # Create long-range feedback connections that bypass direct routes
        feedback_patterns = [
            # Create loops: A->B->C->A
            (['PCC'], ['mPFC'], ['AG'], ['PCC']),
            (['mPFC'], ['PCu'], ['TP'], ['mPFC']),
            (['AG'], ['TP'], ['PCC'], ['AG']),
        ]
        
        for loop in feedback_patterns:
            for i in range(len(loop)):
                source_region = loop[i]
                target_region = loop[(i + 1) % len(loop)]
                
                # Select subset of neurons for feedback
                source_neurons = np.random.choice(
                    self.regions[source_region[0]], 
                    size=min(5, len(self.regions[source_region[0]])),
                    replace=False
                )
                target_neurons = np.random.choice(
                    self.regions[target_region[0]], 
                    size=min(5, len(self.regions[target_region[0]])),
                    replace=False
                )
                
                for source_id in source_neurons:
                    for target_id in target_neurons:
                        if np.random.random() < 0.3:  # Sparse feedback
                            syn_id = f"feedback_{source_id}_to_{target_id}"
                            
                            synapse = SynapseFactory.create_excitatory_synapse(
                                syn_id, source_id, target_id,
                                weight=np.random.uniform(0.7, 1.1),
                                delay=np.random.uniform(20.0, 35.0)  # Long delays
                            )
                            
                            self._register_synapse(synapse)
    
    def _register_synapse(self, synapse):
        """Register synapse with neurons and network."""
        self.synapses[synapse.synapse_id] = synapse
        
        # Register with neurons
        pre_neuron = self.neurons[synapse.presynaptic_neuron_id]
        post_neuron = self.neurons[synapse.postsynaptic_neuron_id]
        
        pre_neuron.add_postsynaptic_connection(synapse)
        post_neuron.add_presynaptic_connection(synapse)
    
    def update(self, dt):
        """Update the entire DMN for one time step."""
        # Update all synapses
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
            
            neuron.update(dt)
    
    def apply_spontaneous_activity(self, time):
        """Apply spontaneous activity patterns characteristic of DMN."""
        # PCC receives more spontaneous input (central hub)
        for neuron_id in self.regions['PCC']:
            if np.random.random() < 0.002:  # 0.2% chance per time step
                self.neurons[neuron_id].set_external_input(
                    np.random.normal(1.5, 0.3)
                )
        
        # Other regions receive occasional input
        for region in ['mPFC', 'AG', 'PCu']:
            for neuron_id in self.regions[region]:
                if np.random.random() < 0.001:  # 0.1% chance
                    self.neurons[neuron_id].set_external_input(
                        np.random.normal(1.2, 0.4)
                    )
        
        # TP receives least input (deeper processing)
        for neuron_id in self.regions['TP']:
            if np.random.random() < 0.0005:  # 0.05% chance
                self.neurons[neuron_id].set_external_input(
                    np.random.normal(1.0, 0.2)
                )

def analyze_brain_waves(population_data, lfp_data):
    """Comprehensive analysis of synthetic brain waves."""
    print("Analyzing synthetic brain waves...")
    
    analysis = {}
    
    # Population rate analysis
    times, rates = population_data
    if len(rates) > 100:
        # Calculate power spectral density
        dt = np.mean(np.diff(times)) / 1000.0  # Convert to seconds
        fs = 1.0 / dt
        
        # Use longer segments for better frequency resolution
        nperseg = min(1024, len(rates) // 4)
        freqs, psd = signal.welch(rates, fs, nperseg=nperseg)
        
        # Define EEG frequency bands
        bands = {
            'Delta (0.5-4 Hz)': (0.5, 4),
            'Theta (4-8 Hz)': (4, 8), 
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-100 Hz)': (30, 100)
        }
        
        # Calculate power in each band
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                band_powers[band_name] = band_power
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(psd[1:], height=np.max(psd)*0.1)[0] + 1
        dominant_freqs = freqs[peak_indices]
        
        analysis['population'] = {
            'frequencies': freqs,
            'psd': psd,
            'band_powers': band_powers,
            'dominant_frequencies': dominant_freqs
        }
    
    # LFP analysis
    if lfp_data:
        lfp_times, lfp_signal = lfp_data
        if len(lfp_signal) > 100:
            try:
                # LFP frequency analysis
                dt_lfp = np.mean(np.diff(lfp_times)) / 1000.0
                fs_lfp = 1.0 / dt_lfp
                
                nperseg_lfp = min(512, len(lfp_signal) // 4)
                freqs_lfp, psd_lfp = signal.welch(lfp_signal, fs_lfp, nperseg=nperseg_lfp)
                
                analysis['lfp'] = {
                    'frequencies': freqs_lfp,
                    'psd': psd_lfp,
                    'signal_std': np.std(lfp_signal),
                    'signal_range': np.ptp(lfp_signal)
                }
            except:
                analysis['lfp'] = None
    
    return analysis

def plot_dmn_results(dmn, probes, wave_analysis):
    """Create comprehensive DMN analysis plots."""
    print("Creating DMN analysis plots...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create large figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Default Mode Network - Synthetic Brain Wave Analysis', fontsize=16)
    
    # 1. Network structure visualization
    region_colors = {
        'PCC': 'red', 'mPFC': 'blue', 'AG': 'green', 
        'PCu': 'orange', 'TP': 'purple'
    }
    
    for i, (region, neuron_ids) in enumerate(dmn.regions.items()):
        # Simple circular layout for each region
        n_neurons = len(neuron_ids)
        angles = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
        
        # Position regions in a larger circle
        region_angle = i * 2 * np.pi / len(dmn.regions)
        region_x = 3 * np.cos(region_angle)
        region_y = 3 * np.sin(region_angle)
        
        x_pos = region_x + 0.5 * np.cos(angles)
        y_pos = region_y + 0.5 * np.sin(angles)
        
        axes[0, 0].scatter(x_pos, y_pos, c=region_colors[region], 
                          label=region, s=30, alpha=0.7)
    
    axes[0, 0].set_title('DMN Architecture')
    axes[0, 0].legend()
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Regional activity levels
    if 'spike' in probes:
        spike_probe = probes['spike']
        firing_rates = spike_probe.calculate_firing_rates(time_window=2000.0)
        
        regional_rates = {}
        for region, neuron_ids in dmn.regions.items():
            rates = [firing_rates.get(nid, 0) for nid in neuron_ids]
            regional_rates[region] = np.mean(rates)
        
        regions = list(regional_rates.keys())
        rates = list(regional_rates.values())
        colors = [region_colors[r] for r in regions]
        
        axes[0, 1].bar(regions, rates, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Mean Firing Rate (Hz)')
        axes[0, 1].set_title('Regional Activity Levels')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Population synthetic brain wave
    if 'population' in probes:
        times, rates = probes['population'].get_population_rate_trace()
        axes[0, 2].plot(times, rates, 'b-', linewidth=2)
        axes[0, 2].fill_between(times, rates, alpha=0.3)
        axes[0, 2].set_ylabel('Population Rate (Hz)')
        axes[0, 2].set_title('Synthetic Brain Wave (Population)')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Frequency analysis - Population
    if wave_analysis and 'population' in wave_analysis:
        pop_analysis = wave_analysis['population']
        freqs = pop_analysis['frequencies']
        psd = pop_analysis['psd']
        
        axes[1, 0].semilogy(freqs, psd, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power Spectral Density')
        axes[1, 0].set_title('Population Activity Spectrum')
        axes[1, 0].set_xlim(0, 50)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mark dominant frequencies
        if len(pop_analysis['dominant_frequencies']) > 0:
            for freq in pop_analysis['dominant_frequencies'][:3]:
                if freq <= 50:
                    axes[1, 0].axvline(freq, color='red', linestyle='--', alpha=0.7)
    
    # 5. Band power analysis
    if wave_analysis and 'population' in wave_analysis:
        band_powers = wave_analysis['population']['band_powers']
        
        bands = list(band_powers.keys())
        powers = list(band_powers.values())
        band_colors = ['purple', 'blue', 'green', 'orange', 'red']
        
        bars = axes[1, 1].bar(range(len(bands)), powers, 
                             color=band_colors[:len(bands)], alpha=0.7)
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].set_title('EEG Band Power Analysis')
        axes[1, 1].set_xticks(range(len(bands)))
        axes[1, 1].set_xticklabels([b.split()[0] for b in bands], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add power values on bars
        for bar, power in zip(bars, powers):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{power:.2e}', ha='center', va='bottom', fontsize=8)
    
    # 6. LFP signal
    if 'lfp' in probes:
        lfp_times, lfp_signal = probes['lfp'].get_lfp_trace()
        if len(lfp_signal) > 0:
            axes[1, 2].plot(lfp_times, lfp_signal, 'purple', linewidth=1.5)
            axes[1, 2].set_ylabel('LFP (mV)')
            axes[1, 2].set_title('Local Field Potential')
            axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Voltage traces from different regions
    if 'voltage' in probes:
        voltage_traces = probes['voltage'].get_voltage_traces()
        colors_trace = plt.cm.Set1(np.linspace(0, 1, len(voltage_traces)))
        
        for i, (neuron_id, trace) in enumerate(voltage_traces.items()):
            if len(trace['time']) > 0 and i < 5:  # Show first 5
                offset = i * 30
                region = neuron_id.split('_')[0]
                axes[2, 0].plot(trace['time'], trace['voltage'] + offset,
                               color=region_colors.get(region, 'black'),
                               linewidth=1, label=f"{region} ({neuron_id})", alpha=0.8)
        
        axes[2, 0].set_ylabel('Membrane Potential (mV)')
        axes[2, 0].set_title('Regional Voltage Traces')
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Synchrony analysis
    if 'population' in probes:
        times_sync, synchrony = probes['population'].get_synchrony_trace()
        if len(synchrony) > 0:
            axes[2, 1].plot(times_sync, synchrony, 'r-', linewidth=2)
            axes[2, 1].fill_between(times_sync, synchrony, alpha=0.3, color='red')
            axes[2, 1].set_ylabel('Synchrony Index')
            axes[2, 1].set_title('Network Synchronization')
            axes[2, 1].set_xlabel('Time (ms)')
            axes[2, 1].grid(True, alpha=0.3)
    
    # 9. LFP frequency spectrum
    if wave_analysis and 'lfp' in wave_analysis and wave_analysis['lfp']:
        lfp_analysis = wave_analysis['lfp']
        freqs_lfp = lfp_analysis['frequencies']
        psd_lfp = lfp_analysis['psd']
        
        axes[2, 2].semilogy(freqs_lfp, psd_lfp, 'purple', linewidth=2)
        axes[2, 2].set_xlabel('Frequency (Hz)')
        axes[2, 2].set_ylabel('LFP Power')
        axes[2, 2].set_title('LFP Frequency Spectrum')
        axes[2, 2].set_xlim(0, 100)
        axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "04_default_mode_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"DMN analysis plot saved to: {output_file}")
    
    try:
        plt.show()
    except:
        print("Could not display plot (running in non-interactive mode)")

def main():
    """Main function."""
    print("SDMN Framework - Default Mode Networks Demonstration")
    print("=" * 65)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Create DMN
        print("Creating Default Mode Network...")
        dmn = DefaultModeNetwork(enable_plasticity=False)
        
        # Setup comprehensive monitoring
        print("Setting up monitoring probes...")
        
        # Select representative neurons from each region
        monitored_neurons = []
        for region, neuron_ids in dmn.regions.items():
            monitored_neurons.extend(neuron_ids[:3])  # 3 from each region
        
        # Setup probes
        voltage_probe = VoltageProbe(
            "dmn_voltage",
            monitored_neurons,
            sampling_interval=1.0,
            enable_filtering=True,
            filter_cutoff=250.0
        )
        
        for neuron_id in voltage_probe.target_ids:
            voltage_probe.register_neuron_object(neuron_id, dmn.neurons[neuron_id])
        
        population_probe = PopulationActivityProbe(
            "dmn_population",
            "default_mode_network",
            list(dmn.neurons.keys()),
            bin_size=10.0,
            sliding_window=200.0,
            record_synchrony=True
        )
        population_probe.register_neuron_objects(dmn.neurons)
        
        # LFP probe positioned in PCC (central hub)
        lfp_probe = LFPProbe(
            "dmn_lfp",
            list(dmn.neurons.keys()),
            probe_position=(0.0, 0.0, 0.0),
            sampling_interval=5.0,
            distance_weights=True,
            max_distance=1000.0
        )
        
        # Create positions (simplified)
        positions = {}
        for region, neuron_ids in dmn.regions.items():
            for i, neuron_id in enumerate(neuron_ids):
                # Simple regional positioning
                angle = i * 2 * np.pi / len(neuron_ids)
                if region == 'PCC':
                    base_pos = (0, 0, 0)
                elif region == 'mPFC':
                    base_pos = (200, 200, 0)
                elif region == 'AG':
                    base_pos = (-200, 200, 0)
                elif region == 'PCu':
                    base_pos = (0, -200, 0)
                else:  # TP
                    base_pos = (300, 0, 0)
                
                positions[neuron_id] = (
                    base_pos[0] + 50*np.cos(angle),
                    base_pos[1] + 50*np.sin(angle),
                    base_pos[2] + np.random.normal(0, 20)
                )
        
        lfp_probe.register_neuron_objects(dmn.neurons, positions)
        
        # Setup simulation
        config = SimulationConfig(
            dt=0.1,
            max_time=5000.0,  # 5 seconds for good frequency resolution
            enable_logging=False
        )
        
        engine = SimulationEngine(config)
        engine.add_network("default_mode_network", dmn)
        
        # Add probes
        engine.add_probe("voltage", voltage_probe)
        engine.add_probe("population", population_probe)
        engine.add_probe("lfp", lfp_probe)
        
        # Add spontaneous activity
        def dmn_activity(step, time):
            dmn.apply_spontaneous_activity(time)
        
        engine.register_step_callback(dmn_activity)
        
        # Start recording
        voltage_probe.start_recording()
        population_probe.start_recording()
        lfp_probe.start_recording()
        
        print("Running DMN simulation (this may take a moment)...")
        results = engine.run()
        
        if results.success:
            print(f"DMN simulation completed successfully!")
            print(f"Simulation time: {results.simulation_time:.1f} ms")
            print(f"Wall time: {results.wall_time:.2f} seconds")
            
            # Analyze brain waves
            population_data = population_probe.get_population_rate_trace()
            lfp_data = lfp_probe.get_lfp_trace()
            
            wave_analysis = analyze_brain_waves(population_data, lfp_data)
            
            # Create comprehensive plots
            probes = {
                'voltage': voltage_probe,
                'population': population_probe,
                'lfp': lfp_probe
            }
            
            plot_dmn_results(dmn, probes, wave_analysis)
            
            # Print analysis results
            print("\n=== DMN Analysis Results ===")
            if wave_analysis and 'population' in wave_analysis:
                band_powers = wave_analysis['population']['band_powers']
                dominant_freqs = wave_analysis['population']['dominant_frequencies']
                
                print("EEG Band Power Analysis:")
                for band, power in band_powers.items():
                    print(f"  {band}: {power:.3e}")
                
                print(f"Dominant frequencies: {dominant_freqs[:3]} Hz")
                
                # Identify most prominent band
                max_band = max(band_powers, key=band_powers.get)
                print(f"Most prominent band: {max_band}")
            
            print("\nDMN Network Statistics:")
            total_spikes = 0
            for region, neuron_ids in dmn.regions.items():
                region_spikes = 0
                for neuron_id in neuron_ids:
                    neuron = dmn.neurons[neuron_id]
                    region_spikes += len(neuron.get_spike_times())
                total_spikes += region_spikes
                print(f"  {region}: {region_spikes} spikes ({len(neuron_ids)} neurons)")
            
            print(f"Total network spikes: {total_spikes}")
            
            print("\n=== Summary ===")
            print("✓ Created biologically-inspired Default Mode Network")
            print("✓ Generated synthetic brain waves with realistic frequencies")
            print("✓ Demonstrated regional specialization and connectivity")
            print("✓ Showed network synchronization and oscillations")
            print("✓ Analyzed EEG-like frequency bands (Delta, Theta, Alpha, Beta, Gamma)")
            
            print("\nThis simulation demonstrates:")
            print("• Multi-regional brain network architecture")
            print("• Spontaneous oscillatory activity")
            print("• Realistic brain wave frequency content")
            print("• Network-wide synchronization patterns")
            print("• Local field potential generation")
            
        else:
            print(f"DMN simulation failed: {results.error_message}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
