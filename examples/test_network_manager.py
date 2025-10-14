"""Quick test for NetworkManager functionality."""

from sdmn.networks.celegans import CElegansNetwork

def test_basic():
    """Test basic network creation and simulation."""
    print("Testing NetworkManager...")
    
    # Create network
    net = CElegansNetwork()
    
    # Add neurons
    n1 = net.add_sensory_neuron("S1")
    n2 = net.add_interneuron("I1")
    n3 = net.add_motor_neuron("M1")
    
    # Add connections
    net.add_graded_synapse(n1, n2, weight=2.0)
    net.add_gap_junction(n2, n3, conductance=0.8)
    
    # Check connectivity
    summary = net.get_connectivity_summary()
    assert summary['n_neurons'] == 3
    assert summary['n_chemical_synapses'] == 1
    assert summary['n_gap_junctions'] == 1
    
    print(f"  [OK] Network created: {summary['n_neurons']} neurons")
    
    # Simulate
    net.set_external_current("S1", 60.0)
    net.simulate(duration=100.0, progress=False)
    
    # Check results
    voltages = net.get_current_voltages()
    assert len(voltages) == 3
    assert all(isinstance(v, float) for v in voltages.values())
    
    print(f"  [OK] Simulation completed: t={net.current_time:.1f} ms")
    print(f"  [OK] Final voltages: S1={voltages['S1']:.2f}, I1={voltages['I1']:.2f}, M1={voltages['M1']:.2f}")
    
    # Test pause/resume
    net.pause()
    net.resume(duration=50.0)
    assert abs(net.current_time - 150.0) < 0.01  # Allow floating point tolerance
    
    print(f"  [OK] Pause/resume works: t={net.current_time:.1f} ms")
    
    # Test reset
    net.reset()
    assert net.current_time == 0.0
    
    print(f"  [OK] Reset works: t={net.current_time:.1f} ms")
    
    print("\n[SUCCESS] All NetworkManager tests passed!")
    return True

if __name__ == "__main__":
    test_basic()

