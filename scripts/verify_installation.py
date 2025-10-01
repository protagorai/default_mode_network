#!/usr/bin/env python3
"""
SDMN Installation Verification Script

This script verifies that the SDMN package has been properly installed
and can be imported correctly in editable mode.
"""

import sys
import importlib.util


def test_basic_import():
    """Test basic package import."""
    print("[TEST] Testing basic SDMN package import...")
    
    try:
        import sdmn
        print(f"[SUCCESS] SDMN package imported - version: {sdmn.__version__}")
        return True
    except ImportError as e:
        print(f"[ERROR] Failed to import SDMN package: {e}")
        print("[INFO] Make sure you have installed the package with:")
        print("  poetry install --with dev")
        print("  # or")
        print("  pip install -e .")
        return False


def test_component_imports():
    """Test importing individual components."""
    print("[TEST] Testing component imports...")
    
    try:
        from sdmn.core import SimulationEngine, SimulationConfig
        from sdmn.neurons import LIFNeuron, NeuronType
        from sdmn.networks import NetworkBuilder, NetworkTopology
        from sdmn.probes import VoltageProbe, SpikeProbe
        
        print("[SUCCESS] All core components imported successfully")
        return True
    except ImportError as e:
        print(f"[ERROR] Failed to import components: {e}")
        return False


def test_cli_availability():
    """Test that CLI is available."""
    print("[TEST] Testing CLI availability...")
    
    try:
        import sdmn.cli
        print("[SUCCESS] CLI module imported")
        return True
    except ImportError as e:
        print(f"[ERROR] CLI import failed: {e}")
        return False


def test_package_functionality():
    """Test basic package functionality."""
    print("[TEST] Testing basic functionality...")
    
    try:
        from sdmn.core import SimulationConfig, TimeManager
        from sdmn.neurons import NeuronType, LIFParameters
        
        # Test creating configuration
        config = SimulationConfig(dt=0.1, max_time=10.0, enable_logging=False)
        print("[SUCCESS] SimulationConfig created")
        
        # Test creating time manager
        time_mgr = TimeManager(dt=0.1, max_time=10.0)
        print("[SUCCESS] TimeManager created")
        
        # Test creating neuron parameters
        params = LIFParameters(tau_m=20.0)
        print("[SUCCESS] LIFParameters created")
        
        print("[SUCCESS] Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Functionality test failed: {e}")
        return False


def main():
    """Main verification function."""
    print("SDMN Installation Verification")
    print("=" * 50)
    
    tests = [
        test_basic_import,
        test_component_imports,
        test_cli_availability,
        test_package_functionality
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("[SUCCESS] SDMN package installation verified!")
        print("")
        print("You can now use the package:")
        print("  python -m sdmn info")
        print("  python -m sdmn simulate --help")
        print("  python examples/quickstart_simulation.py")
    else:
        print("[ERROR] Installation verification failed!")
        print("Please check the error messages above.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
