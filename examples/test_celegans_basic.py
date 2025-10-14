"""
Quick test script for C. elegans graded neurons.

Verifies basic functionality without plotting.
Use this to quickly check if the implementation works.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from sdmn.neurons.graded import (
            GradedNeuron,
            CElegansNeuron,
            CElegansParameters,
            SensoryNeuron,
            Interneuron,
            MotorNeuron
        )
        from sdmn.synapses import (
            GradedChemicalSynapse,
            GradedSynapseParameters,
            GapJunction,
            GapJunctionParameters,
            SynapseType
        )
        print("  [OK] All imports successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_single_neuron():
    """Test single neuron creation and simulation."""
    print("\nTesting single neuron...")
    try:
        from sdmn.neurons.graded import CElegansNeuron, CElegansParameters
        
        params = CElegansParameters(dt=0.01)
        neuron = CElegansNeuron("test", params)
        
        # Simulate 100 steps
        for _ in range(100):
            neuron.set_external_current(50.0)
            neuron.update(dt=0.01)
        
        V_final = neuron.voltage
        states = neuron.get_channel_states()
        
        # Basic checks
        assert -100 < V_final < 100, f"Voltage out of range: {V_final}"
        assert 0 <= states['m_Ca'] <= 1, f"m_Ca out of range: {states['m_Ca']}"
        assert 0 <= states['m_K'] <= 1, f"m_K out of range: {states['m_K']}"
        assert states['Ca_internal'] >= 0, f"Negative calcium: {states['Ca_internal']}"
        
        print(f"  [OK] Neuron simulation: V={V_final:.2f} mV")
        return True
    except Exception as e:
        print(f"  [FAIL] Neuron test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neuron_classes():
    """Test different neuron classes."""
    print("\nTesting neuron classes...")
    try:
        from sdmn.neurons.graded import SensoryNeuron, Interneuron, MotorNeuron
        
        sensory = SensoryNeuron("S1")
        inter = Interneuron("I1")
        motor = MotorNeuron("M1")
        
        # Quick simulation
        for neuron in [sensory, inter, motor]:
            neuron.set_external_current(40.0)
            neuron.update(dt=0.01)
        
        print(f"  [OK] All neuron classes created successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Neuron classes error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graded_synapse():
    """Test graded chemical synapse."""
    print("\nTesting graded synapse...")
    try:
        from sdmn.neurons.graded import SensoryNeuron, Interneuron
        from sdmn.synapses import GradedChemicalSynapse, GradedSynapseParameters, SynapseType
        
        pre = SensoryNeuron("pre")
        post = Interneuron("post")
        
        syn_params = GradedSynapseParameters(
            synapse_type=SynapseType.EXCITATORY,
            weight=2.0
        )
        synapse = GradedChemicalSynapse("syn", pre, post, syn_params)
        
        # Simulate
        for _ in range(100):
            pre.set_external_current(60.0)
            pre.update(dt=0.01)
            synapse.update(dt=0.01)
            post.update(dt=0.01)
        
        g_syn = synapse.get_conductance()
        I_syn = synapse.get_current()
        
        print(f"  [OK] Synapse simulation: g={g_syn:.3f} nS, I={I_syn:.2f} pA")
        return True
    except Exception as e:
        print(f"  [FAIL] Graded synapse error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gap_junction():
    """Test gap junction."""
    print("\nTesting gap junction...")
    try:
        from sdmn.neurons.graded import Interneuron
        from sdmn.synapses import GapJunction, GapJunctionParameters
        
        neuron_a = Interneuron("A")
        neuron_b = Interneuron("B")
        
        gap_params = GapJunctionParameters(conductance=1.0)
        gap = GapJunction("gap", neuron_a, neuron_b, gap_params)
        
        # Simulate longer to see effect
        for _ in range(1000):  # 10 ms
            neuron_a.set_external_current(50.0)
            neuron_a.update(dt=0.01)
            gap.update(dt=0.01)
            neuron_b.update(dt=0.01)
        
        V_a = neuron_a.voltage
        V_b = neuron_b.voltage
        currents = gap.get_currents()
        
        # B should be influenced by A (may be subtle)
        assert V_b > -66.0, f"Neuron B should show some depolarization: {V_b:.2f} mV"
        
        print(f"  [OK] Gap junction: V_a={V_a:.2f} mV, V_b={V_b:.2f} mV")
        return True
    except Exception as e:
        print(f"  [FAIL] Gap junction error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("C. ELEGANS GRADED NEURONS - BASIC FUNCTIONALITY TEST")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Single Neuron", test_single_neuron),
        ("Neuron Classes", test_neuron_classes),
        ("Graded Synapse", test_graded_synapse),
        ("Gap Junction", test_gap_junction),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Implementation is working correctly.")
        print("=" * 70 + "\n")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Check error messages above.")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

