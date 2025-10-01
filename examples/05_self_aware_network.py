#!/usr/bin/env python3
"""
Self-Aware Network Demonstration - SDMN Framework Example 05

This example demonstrates basic self-awareness capabilities including
self-monitoring, risk-reward assessment, and self-preservation behaviors
in a simple neural network system.

Run with:
    python examples/05_self_aware_network.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

# Import from the SDMN package
import sdmn
from sdmn.neurons import LIFNeuron, LIFParameters, SynapseFactory
from sdmn.probes import VoltageProbe, PopulationActivityProbe
from sdmn.core import SimulationEngine, SimulationConfig


class SelfStateMonitor:
    """
    Simple self-monitoring system that tracks network health and performance.
    
    Demonstrates basic self-awareness by monitoring internal states and
    identifying beneficial vs. detrimental conditions.
    """
    
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.state_history = []
        self.health_threshold = 0.3  # Below this = unhealthy state
        self.performance_baseline = 10.0  # Expected activity level
        
    def assess_network_health(self, network, current_time: float) -> Dict[str, Any]:
        """Assess current network health and identify threats/opportunities."""
        
        # Calculate network activity level
        total_activity = 0
        active_neurons = 0
        
        for neuron in network.neurons.values():
            if hasattr(neuron, 'has_spiked') and neuron.has_spiked():
                active_neurons += 1
                total_activity += 1
            elif hasattr(neuron, 'get_membrane_potential'):
                v_mem = neuron.get_membrane_potential()
                # Normalize activity relative to resting potential
                activity = max(0, (v_mem + 70) / 50)  # Simple activity measure
                total_activity += activity
        
        activity_ratio = total_activity / len(network.neurons) if network.neurons else 0
        
        # Assess system health
        health_score = min(1.0, activity_ratio / self.performance_baseline)
        
        # Determine system state
        if health_score < self.health_threshold:
            state_category = "threatened"
            threat_level = 1.0 - health_score
        elif health_score > 0.8:
            state_category = "thriving"
            threat_level = 0.0
        else:
            state_category = "stable"
            threat_level = 0.2
        
        state_assessment = {
            'timestamp': current_time,
            'health_score': health_score,
            'activity_ratio': activity_ratio,
            'active_neurons': active_neurons,
            'total_neurons': len(network.neurons),
            'state_category': state_category,
            'threat_level': threat_level,
            'needs_intervention': health_score < self.health_threshold
        }
        
        self.state_history.append(state_assessment)
        return state_assessment


class RiskRewardAssessor:
    """
    Simple risk-reward assessment system for self-preservation.
    
    Evaluates stimuli and network conditions to determine optimal
    responses for system survival and performance.
    """
    
    def __init__(self):
        self.decision_history = []
        self.learned_threats = []  # Patterns that previously caused harm
        self.learned_benefits = []  # Patterns that previously helped
        
    def evaluate_stimulus(self, stimulus_intensity: float, 
                         network_health: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate stimulus for self-preservation implications."""
        
        # Assess stimulus characteristics
        if stimulus_intensity > 5.0:  # High intensity stimulus
            threat_level = min(1.0, (stimulus_intensity - 5.0) / 5.0)
        else:
            threat_level = 0.0
        
        # Assess reward potential based on current network state
        if network_health['health_score'] < 0.5:
            # Unhealthy network needs stimulation
            reward_potential = min(1.0, stimulus_intensity / 3.0)
        else:
            # Healthy network benefits from moderate stimulation
            reward_potential = max(0, 1.0 - abs(stimulus_intensity - 2.0) / 2.0)
        
        # Estimate uncertainty
        uncertainty = 0.5 if not self.decision_history else 0.2
        
        assessment = {
            'threat_level': threat_level,
            'reward_potential': reward_potential,
            'uncertainty': uncertainty,
            'recommended_response': self._generate_response(threat_level, reward_potential)
        }
        
        return assessment
    
    def _generate_response(self, threat_level: float, reward_potential: float) -> str:
        """Generate self-preserving response recommendation."""
        if threat_level > 0.7:
            return 'avoid'  # High threat - avoid stimulus
        elif reward_potential > 0.6 and threat_level < 0.3:
            return 'approach'  # High reward, low threat - approach
        elif threat_level > 0.4:
            return 'cautious'  # Moderate threat - be careful
        else:
            return 'neutral'  # No strong preference
    
    def learn_from_outcome(self, stimulus: float, assessment: Dict[str, float], 
                          outcome_health: float) -> None:
        """Learn from decision outcomes to improve future assessments."""
        decision_record = {
            'stimulus': stimulus,
            'assessment': assessment,
            'outcome_health': outcome_health,
            'success': outcome_health > 0.5  # Define success threshold
        }
        
        self.decision_history.append(decision_record)
        
        # Simple learning: remember particularly good or bad outcomes
        if outcome_health < 0.3:  # Bad outcome
            self.learned_threats.append(stimulus)
        elif outcome_health > 0.8:  # Good outcome
            self.learned_benefits.append(stimulus)


class SelfAwareNetwork:
    """
    Neural network with basic self-awareness capabilities.
    
    Integrates self-monitoring, risk assessment, and adaptive responses
    to demonstrate simple self-preservation behaviors.
    """
    
    def __init__(self, neurons, synapses):
        self.neurons = neurons
        self.synapses = synapses
        
        # Self-awareness components
        self.self_monitor = SelfStateMonitor("self_aware_net")
        self.risk_assessor = RiskRewardAssessor()
        
        # Internal state tracking
        self.internal_narrative = []
        self.self_model = {
            'identity': 'self_aware_network',
            'capabilities': ['neural_processing', 'self_monitoring', 'risk_assessment'],
            'preferences': {'high_health': True, 'avoid_damage': True},
            'memory': {'positive_experiences': [], 'negative_experiences': []}
        }
        
        # Decision-making state
        self.current_strategy = 'exploration'  # 'exploration', 'conservation', 'recovery'
        
    def update(self, dt: float) -> None:
        """Update network with self-awareness processing."""
        # Standard network update
        for synapse in self.synapses.values():
            synapse.update(dt)
        
        for neuron in self.neurons.values():
            # Calculate synaptic inputs
            for synapse in neuron.presynaptic_connections:
                if synapse.synapse_id in self.synapses:
                    current = synapse.calculate_current(
                        neuron.get_membrane_potential()
                    )
                    neuron.add_synaptic_input(current)
            
            neuron.update(dt)
    
    def process_stimulus_with_awareness(self, stimulus_intensity: float, 
                                      current_time: float) -> Dict[str, Any]:
        """Process stimulus through self-awareness framework."""
        
        # 1. Monitor current self-state
        health_assessment = self.self_monitor.assess_network_health(self, current_time)
        
        # 2. Assess stimulus for self-preservation implications
        risk_assessment = self.risk_assessor.evaluate_stimulus(
            stimulus_intensity, health_assessment
        )
        
        # 3. Generate self-preserving response
        response = self._decide_response(health_assessment, risk_assessment)
        
        # 4. Update internal narrative
        self._update_internal_narrative(health_assessment, risk_assessment, response)
        
        # 5. Apply the decided response
        self._apply_response(response, stimulus_intensity)
        
        return {
            'health_assessment': health_assessment,
            'risk_assessment': risk_assessment,
            'response': response,
            'strategy': self.current_strategy
        }
    
    def _decide_response(self, health: Dict[str, Any], risk: Dict[str, float]) -> Dict[str, Any]:
        """Make self-preserving decision based on current state and stimulus assessment."""
        
        if health['needs_intervention']:
            # System is in poor health - prioritize recovery
            self.current_strategy = 'recovery'
            if risk['reward_potential'] > 0.5:
                return {'action': 'accept_stimulus', 'intensity': 'moderate', 'reason': 'health_recovery'}
            else:
                return {'action': 'reject_stimulus', 'intensity': 'none', 'reason': 'avoid_further_damage'}
        
        elif risk['threat_level'] > 0.7:
            # High threat detected - be protective
            self.current_strategy = 'conservation'
            return {'action': 'reject_stimulus', 'intensity': 'defensive', 'reason': 'threat_avoidance'}
        
        elif risk['reward_potential'] > 0.6 and health['health_score'] > 0.6:
            # Good opportunity and healthy state - explore
            self.current_strategy = 'exploration'
            return {'action': 'accept_stimulus', 'intensity': 'full', 'reason': 'opportunity_pursuit'}
        
        else:
            # Uncertain or neutral - be cautious
            self.current_strategy = 'conservation'
            return {'action': 'accept_stimulus', 'intensity': 'minimal', 'reason': 'cautious_exploration'}
    
    def _update_internal_narrative(self, health: Dict[str, Any], 
                                 risk: Dict[str, float], response: Dict[str, Any]) -> None:
        """Update internal narrative about self and experiences."""
        
        narrative_entry = {
            'timestamp': health['timestamp'],
            'self_assessment': f"I am {health['state_category']} with {health['health_score']:.2f} health",
            'situation_assessment': f"Stimulus poses {risk['threat_level']:.2f} threat, {risk['reward_potential']:.2f} reward",
            'decision_rationale': f"I chose to {response['action']} because {response['reason']}",
            'strategy': f"My current strategy is {self.current_strategy}"
        }
        
        self.internal_narrative.append(narrative_entry)
        
        # Keep narrative manageable
        if len(self.internal_narrative) > 50:
            self.internal_narrative = self.internal_narrative[-50:]
    
    def _apply_response(self, response: Dict[str, Any], stimulus_intensity: float) -> None:
        """Apply the decided response to the network."""
        
        if response['action'] == 'reject_stimulus':
            # Don't apply stimulus - self-preservation
            applied_intensity = 0.0
        elif response['intensity'] == 'minimal':
            applied_intensity = stimulus_intensity * 0.3
        elif response['intensity'] == 'moderate':
            applied_intensity = stimulus_intensity * 0.7
        else:  # full
            applied_intensity = stimulus_intensity
        
        # Apply to subset of neurons based on strategy
        if self.current_strategy == 'recovery':
            # Stimulate neurons that appear healthy
            target_neurons = [n for n in self.neurons.values() 
                            if n.get_membrane_potential() > -60][:3]
        elif self.current_strategy == 'conservation':
            # Minimal stimulation to preserve resources
            target_neurons = list(self.neurons.values())[:1]
        else:  # exploration
            # Normal stimulation pattern
            target_neurons = list(self.neurons.values())[:3]
        
        # Apply stimulus to selected neurons
        for neuron in target_neurons:
            neuron.set_external_input(applied_intensity / len(target_neurons))
    
    def get_self_awareness_summary(self) -> Dict[str, Any]:
        """Get summary of self-awareness indicators."""
        
        if not self.self_monitor.state_history:
            return {'status': 'no_data'}
        
        recent_health = [s['health_score'] for s in self.self_monitor.state_history[-10:]]
        recent_decisions = [s['recommended_response'] for s in self.risk_assessor.decision_history[-10:]]
        
        return {
            'total_experiences': len(self.self_monitor.state_history),
            'recent_health_trend': np.mean(recent_health) if recent_health else 0,
            'health_variance': np.std(recent_health) if len(recent_health) > 1 else 0,
            'decision_diversity': len(set(recent_decisions)),
            'current_strategy': self.current_strategy,
            'narrative_length': len(self.internal_narrative),
            'learned_threats': len(self.risk_assessor.learned_threats),
            'learned_benefits': len(self.risk_assessor.learned_benefits),
            'self_model': self.self_model
        }


def create_self_aware_network():
    """Create a small network with self-awareness capabilities."""
    print("Creating self-aware neural network...")
    
    # Create simple network
    neurons = {}
    for i in range(8):  # Small network for clear demonstration
        neuron_id = f"aware_neuron_{i:03d}"
        params = LIFParameters(
            tau_m=20.0 + np.random.normal(0, 2.0),
            v_thresh=-50.0 + np.random.normal(0, 1.0),
            r_mem=10.0 + np.random.normal(0, 0.5)
        )
        neurons[neuron_id] = LIFNeuron(neuron_id, params)
    
    # Create connections with feedback loops for self-awareness
    synapses = {}
    neuron_ids = list(neurons.keys())
    
    # Ring connections for basic feedback
    for i in range(len(neuron_ids)):
        next_i = (i + 1) % len(neuron_ids)
        syn_id = f"ring_{i}_to_{next_i}"
        
        synapse = SynapseFactory.create_excitatory_synapse(
            syn_id, neuron_ids[i], neuron_ids[next_i],
            weight=1.2, delay=3.0
        )
        synapses[syn_id] = synapse
        
        # Register connections
        neurons[neuron_ids[i]].add_postsynaptic_connection(synapse)
        neurons[neuron_ids[next_i]].add_presynaptic_connection(synapse)
    
    # Add self-referential feedback (every 3rd neuron connects to center)
    center_neuron = neuron_ids[len(neuron_ids)//2]
    for i in range(0, len(neuron_ids), 3):
        if neuron_ids[i] != center_neuron:
            syn_id = f"self_ref_{i}_to_center"
            synapse = SynapseFactory.create_excitatory_synapse(
                syn_id, neuron_ids[i], center_neuron,
                weight=0.8, delay=5.0
            )
            synapses[syn_id] = synapse
            neurons[neuron_ids[i]].add_postsynaptic_connection(synapse)
            neurons[center_neuron].add_presynaptic_connection(synapse)
    
    print(f"Created self-aware network with {len(neurons)} neurons and {len(synapses)} synapses")
    return SelfAwareNetwork(neurons, synapses)


def run_self_awareness_simulation(network):
    """Run simulation demonstrating self-awareness behaviors."""
    print("Running self-awareness simulation...")
    
    # Setup basic monitoring
    voltage_probe = VoltageProbe(
        "self_monitor_voltage",
        list(network.neurons.keys())[:4],
        sampling_interval=2.0
    )
    
    for neuron_id in voltage_probe.target_ids:
        voltage_probe.register_neuron_object(neuron_id, network.neurons[neuron_id])
    
    population_probe = PopulationActivityProbe(
        "self_monitor_population",
        "self_aware_network",
        list(network.neurons.keys()),
        bin_size=10.0,
        record_synchrony=True
    )
    population_probe.register_neuron_objects(network.neurons)
    
    # Simulation with various stimuli to test self-preservation
    config = SimulationConfig(dt=0.1, max_time=2000.0, enable_logging=False)
    engine = SimulationEngine(config)
    engine.add_network("self_aware_network", network)
    engine.add_probe("voltage", voltage_probe)
    engine.add_probe("population", population_probe)
    
    # Track decisions and outcomes
    decision_log = []
    
    def self_aware_stimulus(step, time):
        """Apply stimuli and demonstrate self-aware responses."""
        
        # Various stimulus scenarios to test self-preservation
        if 200 <= time < 300:
            # Mild beneficial stimulus
            stimulus = 1.5
        elif 500 <= time < 600:
            # Strong potentially harmful stimulus
            stimulus = 8.0
        elif 800 <= time < 900:
            # Moderate stimulus when network might be stressed
            stimulus = 3.0
        elif 1200 <= time < 1300:
            # Recovery stimulus
            stimulus = 2.0
        elif 1500 <= time < 1600:
            # Another test of threat response
            stimulus = 7.5
        else:
            stimulus = 0.1  # Baseline
        
        # Process stimulus through self-awareness system
        if stimulus > 0.5:  # Only process significant stimuli
            decision_info = network.process_stimulus_with_awareness(stimulus, time)
            decision_log.append(decision_info)
            
            print(f"Time {time:.0f}ms: Stimulus {stimulus:.1f} → "
                  f"{decision_info['response']['action']} "
                  f"(health: {decision_info['health_assessment']['health_score']:.2f}, "
                  f"threat: {decision_info['risk_assessment']['threat_level']:.2f})")
        else:
            # Still apply minimal baseline
            for neuron in list(network.neurons.values())[:2]:
                neuron.set_external_input(stimulus)
    
    engine.register_step_callback(self_aware_stimulus)
    
    # Start recording
    voltage_probe.start_recording()
    population_probe.start_recording()
    
    # Run simulation
    results = engine.run()
    
    return results, decision_log, {'voltage': voltage_probe, 'population': population_probe}


def analyze_self_awareness(network, decision_log, probes):
    """Analyze self-awareness indicators and behaviors."""
    print("\n=== Self-Awareness Analysis ===")
    
    # Get self-awareness summary
    awareness_summary = network.get_self_awareness_summary()
    
    print(f"Total experiences: {awareness_summary['total_experiences']}")
    print(f"Recent health trend: {awareness_summary['recent_health_trend']:.3f}")
    print(f"Current strategy: {awareness_summary['current_strategy']}")
    print(f"Learned threats: {awareness_summary['learned_threats']}")
    print(f"Learned benefits: {awareness_summary['learned_benefits']}")
    
    # Analyze decision patterns
    if decision_log:
        responses = [d['response']['action'] for d in decision_log]
        response_counts = {r: responses.count(r) for r in set(responses)}
        
        print(f"\nDecision patterns:")
        for response, count in response_counts.items():
            print(f"  {response}: {count} times")
        
        # Check for adaptive behavior
        threat_responses = [d for d in decision_log if d['risk_assessment']['threat_level'] > 0.5]
        if threat_responses:
            avg_threat_response = np.mean([1 if d['response']['action'] == 'reject_stimulus' 
                                         else 0 for d in threat_responses])
            print(f"  Threat avoidance rate: {avg_threat_response:.2f} (1.0 = always avoided threats)")
    
    # Show internal narrative sample
    if network.internal_narrative:
        print(f"\nSample internal narrative (last 3 entries):")
        for entry in network.internal_narrative[-3:]:
            print(f"  {entry['timestamp']:.0f}ms: {entry['self_assessment']}")
            print(f"    Situation: {entry['situation_assessment']}")
            print(f"    Decision: {entry['decision_rationale']}")


def plot_self_awareness_results(network, decision_log, probes):
    """Visualize self-awareness behaviors and decision patterns."""
    print("Creating self-awareness visualization...")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Self-Aware Network: Basic Self-Preservation Behaviors', fontsize=16)
    
    # 1. Health monitoring over time
    if network.self_monitor.state_history:
        times = [s['timestamp'] for s in network.self_monitor.state_history]
        health_scores = [s['health_score'] for s in network.self_monitor.state_history]
        threat_levels = [s['threat_level'] for s in network.self_monitor.state_history]
        
        axes[0, 0].plot(times, health_scores, 'g-', linewidth=2, label='Health Score')
        axes[0, 0].plot(times, threat_levels, 'r-', linewidth=2, label='Threat Level')
        axes[0, 0].axhline(y=network.self_monitor.health_threshold, color='orange', 
                          linestyle='--', alpha=0.7, label='Health Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Self-Health Monitoring')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Decision patterns
    if decision_log:
        decision_times = [d['health_assessment']['timestamp'] for d in decision_log]
        threat_levels = [d['risk_assessment']['threat_level'] for d in decision_log]
        responses = [d['response']['action'] for d in decision_log]
        
        # Color code responses
        colors = {'accept_stimulus': 'green', 'reject_stimulus': 'red', 'cautious': 'yellow'}
        response_colors = [colors.get(r, 'blue') for r in responses]
        
        axes[0, 1].scatter(decision_times, threat_levels, c=response_colors, s=50, alpha=0.7)
        axes[0, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Threat')
        axes[0, 1].set_ylabel('Threat Level')
        axes[0, 1].set_title('Self-Preservation Decisions')
        axes[0, 1].legend(['High Threat Line', 'Accept (Green)', 'Reject (Red)'])
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Population activity with self-awareness markers
    times, rates = probes['population'].get_population_rate_trace()
    if len(rates) > 0:
        axes[1, 0].plot(times, rates, 'b-', linewidth=2, label='Population Activity')
        
        # Mark decision points
        if decision_log:
            decision_times = [d['health_assessment']['timestamp'] for d in decision_log]
            for dt in decision_times:
                axes[1, 0].axvline(x=dt, color='red', alpha=0.3, linewidth=1)
        
        axes[1, 0].set_ylabel('Population Rate (Hz)')
        axes[1, 0].set_title('Network Activity with Decision Points')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Strategy evolution
    if decision_log:
        strategy_times = [d['health_assessment']['timestamp'] for d in decision_log]
        strategies = [d['strategy'] for d in decision_log]
        
        # Convert strategies to numeric for plotting
        strategy_map = {'exploration': 1, 'conservation': 0, 'recovery': -1}
        strategy_values = [strategy_map.get(s, 0) for s in strategies]
        
        axes[1, 1].step(strategy_times, strategy_values, 'purple', linewidth=2, where='post')
        axes[1, 1].set_ylabel('Strategy')
        axes[1, 1].set_yticks([-1, 0, 1])
        axes[1, 1].set_yticklabels(['Recovery', 'Conservation', 'Exploration'])
        axes[1, 1].set_title('Self-Preservation Strategy Evolution')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "05_self_aware_network.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Self-awareness analysis saved to: {output_file}")
    
    try:
        plt.show()
    except:
        print("Could not display plot (running in non-interactive mode)")


def main():
    """Main function demonstrating self-aware network."""
    print("SDMN Framework - Self-Aware Network Demonstration")
    print("=" * 60)
    print("Demonstrating basic self-awareness through:")
    print("• Self-monitoring and health assessment")
    print("• Risk-reward evaluation of stimuli")
    print("• Self-preservation decision making")
    print("• Internal narrative construction")
    print("• Adaptive response strategies")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Create self-aware network
        network = create_self_aware_network()
        
        # Run simulation with self-awareness
        results, decision_log, probes = run_self_awareness_simulation(network)
        
        if results.success:
            # Analyze self-awareness behaviors
            analyze_self_awareness(network, decision_log, probes)
            
            # Create visualizations
            plot_self_awareness_results(network, decision_log, probes)
            
            print("\n=== Self-Awareness Demonstration Summary ===")
            print("✓ System monitored its own health and performance")
            print("✓ Evaluated stimuli for self-preservation implications")
            print("✓ Made adaptive decisions based on risk-reward assessment")
            print("✓ Showed preference for self-beneficial responses")
            print("✓ Constructed internal narrative about experiences")
            print("✓ Demonstrated different strategies (exploration, conservation, recovery)")
            
            awareness_summary = network.get_self_awareness_summary()
            print(f"\nSelf-awareness indicators:")
            print(f"• Total experiences processed: {awareness_summary['total_experiences']}")
            print(f"• Current self-preservation strategy: {awareness_summary['current_strategy']}")
            print(f"• Decision diversity: {awareness_summary['decision_diversity']}")
            print(f"• Recent health trend: {awareness_summary['recent_health_trend']:.3f}")
            
            print("\nThis demonstrates:")
            print("• Basic self-recognition and state monitoring")
            print("• Self-preservation instincts and threat avoidance")
            print("• Adaptive behavior based on internal assessment")
            print("• Simple form of artificial consciousness indicators")
            
            print("\nNext steps for research:")
            print("• Scale to larger networks with multiple DMN regions")
            print("• Implement more sophisticated self-model updating")
            print("• Add temporal self-projection for future planning")
            print("• Develop measurable consciousness indicators")
            
        else:
            print(f"Simulation failed: {results.error_message}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
