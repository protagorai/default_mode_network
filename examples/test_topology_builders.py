"""Quick test for topology builders."""

import sys

def test_topology_builders():
    """Test all topology builders."""
    print("=" * 70)
    print("TOPOLOGY BUILDERS - FUNCTIONALITY TEST")
    print("=" * 70)
    
    from sdmn.networks.celegans import (
        build_uniform_network,
        build_small_world_network,
        build_random_network
    )
    
    results = []
    
    # Test 1: Uniform network
    print("\n[1/3] Testing uniform network builder...")
    try:
        net = build_uniform_network(n_neurons=50, k_neighbors=8, random_seed=42)
        summary = net.get_connectivity_summary()
        avg_conn = summary['avg_synapses_per_neuron'] + summary['avg_gaps_per_neuron']
        
        assert summary['n_neurons'] == 50, "Wrong neuron count"
        assert 7 <= avg_conn <= 9, f"Wrong avg connections: {avg_conn}"
        
        print(f"  [PASS] Uniform: {summary['n_neurons']} neurons, "
              f"{avg_conn:.1f} avg connections/neuron")
        results.append(("Uniform Builder", True))
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append(("Uniform Builder", False))
    
    # Test 2: Small-world network
    print("\n[2/3] Testing small-world network builder...")
    try:
        net = build_small_world_network(
            n_neurons=50, k_neighbors=8, rewire_prob=0.2, random_seed=42
        )
        summary = net.get_connectivity_summary()
        avg_conn = summary['avg_synapses_per_neuron'] + summary['avg_gaps_per_neuron']
        
        assert summary['n_neurons'] == 50, "Wrong neuron count"
        assert avg_conn > 0, "No connections created"
        
        print(f"  [PASS] Small-world: {summary['n_neurons']} neurons, "
              f"{avg_conn:.1f} avg connections/neuron")
        results.append(("Small-World Builder", True))
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append(("Small-World Builder", False))
    
    # Test 3: Random network
    print("\n[3/3] Testing random network builder...")
    try:
        net = build_random_network(n_neurons=50, k_neighbors=8, random_seed=42)
        summary = net.get_connectivity_summary()
        avg_conn = summary['avg_synapses_per_neuron'] + summary['avg_gaps_per_neuron']
        
        assert summary['n_neurons'] == 50, "Wrong neuron count"
        assert avg_conn > 0, "No connections created"
        
        print(f"  [PASS] Random: {summary['n_neurons']} neurons, "
              f"{avg_conn:.1f} avg connections/neuron")
        results.append(("Random Builder", True))
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append(("Random Builder", False))
    
    # Test 4: Parameter variations
    print("\n[4/4] Testing parameter variations...")
    try:
        # Variable connection counts
        net = build_uniform_network(n_neurons=30, k_neighbors=6, 
                                    connection_variance=2.0, random_seed=42)
        
        # Mixed neuron types
        from sdmn.networks.celegans import TopologyParameters, UniformTopologyBuilder
        params = TopologyParameters(
            n_neurons=30, k_neighbors=6, neuron_class='mixed',
            sensory_fraction=0.2, motor_fraction=0.2
        )
        builder = UniformTopologyBuilder(params)
        net_mixed = builder.build()
        summary = net_mixed.get_connectivity_summary()
        
        assert summary['neurons_by_class']['sensory'] == 6, "Wrong sensory count"
        assert summary['neurons_by_class']['motor'] == 6, "Wrong motor count"
        assert summary['neurons_by_class']['interneuron'] == 18, "Wrong inter count"
        
        print(f"  [PASS] Parameters: variance, mixed neuron types working")
        results.append(("Parameter Variations", True))
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        results.append(("Parameter Variations", False))
    
    # Test 5: 302 neuron network (full scale)
    print("\n[5/5] Testing 302-neuron network (full C. elegans scale)...")
    try:
        net = build_uniform_network(n_neurons=302, k_neighbors=14, random_seed=42)
        summary = net.get_connectivity_summary()
        avg_conn = summary['avg_synapses_per_neuron'] + summary['avg_gaps_per_neuron']
        
        assert summary['n_neurons'] == 302, "Wrong neuron count"
        assert 13 <= avg_conn <= 15, f"Wrong avg connections: {avg_conn}"
        
        print(f"  [PASS] Full-scale: 302 neurons, {avg_conn:.1f} avg connections")
        results.append(("302-Neuron Network", True))
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append(("302-Neuron Network", False))
    
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
        print("\n[SUCCESS] All topology builders working correctly!")
        print("=" * 70 + "\n")
        return 0
    else:
        print("\n[WARNING] Some tests failed.")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(test_topology_builders())



