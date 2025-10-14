"""
Different classes of C. elegans neurons with specialized parameter sets.

Includes sensory, interneuron, and motor neuron types with channel
densities tuned for their respective functions.

References:
    Goodman et al. (1998), Neuron, 20(4), 763-772.
    Varshney et al. (2011), PLOS Computational Biology, 7(2), e1001066.
"""

from enum import Enum
from typing import Dict, Any
from dataclasses import replace

from .celegans_neuron import CElegansNeuron, CElegansParameters


class CElegansNeuronClass(Enum):
    """Types of C. elegans neurons."""
    SENSORY = "sensory"
    INTERNEURON = "interneuron"
    MOTOR = "motor"


def create_sensory_parameters(**kwargs) -> CElegansParameters:
    """
    Create parameters for sensory neurons.
    
    Sensory neurons have:
    - High Ca2+ conductance (excitable)
    - Low K+ conductance (sustain depolarization)
    - Moderate K(Ca) for adaptation
    
    Args:
        **kwargs: Override specific parameters
        
    Returns:
        CElegansParameters for sensory neuron
    """
    params = CElegansParameters(
        # High calcium for excitability
        g_Ca=1.2,              # Increased from 0.8
        E_Ca=50.0,
        V_half_Ca=-22.0,       # Slightly more negative
        k_Ca=5.0,
        tau_Ca_min=0.5,
        tau_Ca_max=4.0,        # Faster than default
        
        # Low potassium (less repolarization)
        g_K=0.8,               # Decreased from 1.5
        E_K=-80.0,
        V_half_K=-25.0,
        k_K=10.0,
        tau_K_min=2.0,         # Slower activation
        tau_K_max=15.0,
        
        # Moderate K(Ca) for sensory adaptation
        g_KCa=0.6,
        Ca_half=100.0,
        tau_KCa=50.0,
        
        # Standard membrane properties
        C_m=3.0,
        g_leak=0.3,
        E_leak=-65.0,
        
        # Calcium dynamics
        Ca_rest=50.0,
        tau_Ca_removal=100.0,
        f_Ca=0.01,
        cell_volume=1.0,
        
        # Integration
        dt=0.01,
        integration_method="RK4"
    )
    
    # Apply any overrides
    if kwargs:
        params = replace(params, **kwargs)
    
    return params


def create_interneuron_parameters(**kwargs) -> CElegansParameters:
    """
    Create parameters for interneurons.
    
    Interneurons have:
    - Balanced Ca2+ and K+ (integrative)
    - Moderate all conductances
    - Good for summing inputs
    
    Args:
        **kwargs: Override specific parameters
        
    Returns:
        CElegansParameters for interneuron
    """
    # Use default parameters (already balanced for interneurons)
    params = CElegansParameters(
        # Balanced calcium
        g_Ca=0.8,
        E_Ca=50.0,
        V_half_Ca=-20.0,
        k_Ca=5.0,
        tau_Ca_min=0.5,
        tau_Ca_max=5.0,
        
        # Balanced potassium
        g_K=1.5,
        E_K=-80.0,
        V_half_K=-25.0,
        k_K=10.0,
        tau_K_min=1.0,
        tau_K_max=10.0,
        
        # Moderate K(Ca)
        g_KCa=0.5,
        Ca_half=100.0,
        tau_KCa=50.0,
        
        # Standard membrane
        C_m=3.0,
        g_leak=0.3,
        E_leak=-65.0,
        
        # Calcium dynamics
        Ca_rest=50.0,
        tau_Ca_removal=100.0,
        f_Ca=0.01,
        cell_volume=1.0,
        
        # Integration
        dt=0.01,
        integration_method="RK4"
    )
    
    # Apply any overrides
    if kwargs:
        params = replace(params, **kwargs)
    
    return params


def create_motor_parameters(**kwargs) -> CElegansParameters:
    """
    Create parameters for motor neurons.
    
    Motor neurons have:
    - Moderate Ca2+ (graded output)
    - High K+ (strong repolarization, graded control)
    - High K(Ca) for precise control
    
    Args:
        **kwargs: Override specific parameters
        
    Returns:
        CElegansParameters for motor neuron
    """
    params = CElegansParameters(
        # Moderate calcium
        g_Ca=0.7,              # Slightly decreased from 0.8
        E_Ca=50.0,
        V_half_Ca=-18.0,       # Slightly depolarized
        k_Ca=5.0,
        tau_Ca_min=0.5,
        tau_Ca_max=5.0,
        
        # High potassium for graded output
        g_K=2.0,               # Increased from 1.5
        E_K=-80.0,
        V_half_K=-25.0,
        k_K=10.0,
        tau_K_min=0.8,         # Faster
        tau_K_max=8.0,
        
        # High K(Ca) for fine control
        g_KCa=0.8,             # Increased from 0.5
        Ca_half=100.0,
        tau_KCa=40.0,          # Faster adaptation
        
        # Standard membrane
        C_m=3.0,
        g_leak=0.3,
        E_leak=-65.0,
        
        # Calcium dynamics
        Ca_rest=50.0,
        tau_Ca_removal=100.0,
        f_Ca=0.01,
        cell_volume=1.0,
        
        # Integration
        dt=0.01,
        integration_method="RK4"
    )
    
    # Apply any overrides
    if kwargs:
        params = replace(params, **kwargs)
    
    return params


class SensoryNeuron(CElegansNeuron):
    """
    Sensory neuron with high excitability.
    
    Specialized for detecting and responding to sensory stimuli.
    High Ca2+ conductance makes them highly responsive.
    """
    
    def __init__(self, neuron_id: str, **param_overrides):
        """
        Initialize sensory neuron.
        
        Args:
            neuron_id: Unique identifier
            **param_overrides: Override specific parameters
        """
        params = create_sensory_parameters(**param_overrides)
        super().__init__(neuron_id, params)
        self.neuron_class = CElegansNeuronClass.SENSORY


class Interneuron(CElegansNeuron):
    """
    Interneuron with balanced properties.
    
    Specialized for integrating inputs and coordinating responses.
    Balanced Ca2+ and K+ allow for flexible computation.
    """
    
    def __init__(self, neuron_id: str, **param_overrides):
        """
        Initialize interneuron.
        
        Args:
            neuron_id: Unique identifier
            **param_overrides: Override specific parameters
        """
        params = create_interneuron_parameters(**param_overrides)
        super().__init__(neuron_id, params)
        self.neuron_class = CElegansNeuronClass.INTERNEURON


class MotorNeuron(CElegansNeuron):
    """
    Motor neuron with graded output control.
    
    Specialized for controlling muscle activity with graded signals.
    High K+ conductance provides precise, graded voltage control.
    """
    
    def __init__(self, neuron_id: str, **param_overrides):
        """
        Initialize motor neuron.
        
        Args:
            neuron_id: Unique identifier
            **param_overrides: Override specific parameters
        """
        params = create_motor_parameters(**param_overrides)
        super().__init__(neuron_id, params)
        self.neuron_class = CElegansNeuronClass.MOTOR


def create_neuron_by_class(neuron_id: str, neuron_class: CElegansNeuronClass,
                          **param_overrides) -> CElegansNeuron:
    """
    Factory function to create neurons by class.
    
    Args:
        neuron_id: Unique identifier
        neuron_class: Type of neuron to create
        **param_overrides: Override specific parameters
        
    Returns:
        Instantiated neuron of specified class
    """
    if neuron_class == CElegansNeuronClass.SENSORY:
        return SensoryNeuron(neuron_id, **param_overrides)
    elif neuron_class == CElegansNeuronClass.INTERNEURON:
        return Interneuron(neuron_id, **param_overrides)
    elif neuron_class == CElegansNeuronClass.MOTOR:
        return MotorNeuron(neuron_id, **param_overrides)
    else:
        raise ValueError(f"Unknown neuron class: {neuron_class}")


def get_parameter_summary(neuron_class: CElegansNeuronClass) -> Dict[str, Any]:
    """
    Get summary of parameters for a neuron class.
    
    Args:
        neuron_class: Type of neuron
        
    Returns:
        Dictionary summarizing key parameters
    """
    if neuron_class == CElegansNeuronClass.SENSORY:
        params = create_sensory_parameters()
    elif neuron_class == CElegansNeuronClass.INTERNEURON:
        params = create_interneuron_parameters()
    elif neuron_class == CElegansNeuronClass.MOTOR:
        params = create_motor_parameters()
    else:
        raise ValueError(f"Unknown neuron class: {neuron_class}")
    
    return {
        'neuron_class': neuron_class.value,
        'g_Ca': params.g_Ca,
        'g_K': params.g_K,
        'g_KCa': params.g_KCa,
        'g_leak': params.g_leak,
        'C_m': params.C_m,
        'E_leak': params.E_leak,
        'V_half_Ca': params.V_half_Ca,
        'V_half_K': params.V_half_K,
        'description': _get_class_description(neuron_class)
    }


def _get_class_description(neuron_class: CElegansNeuronClass) -> str:
    """Get description of neuron class."""
    descriptions = {
        CElegansNeuronClass.SENSORY: "High Ca2+, low K+ - excitable, responsive to stimuli",
        CElegansNeuronClass.INTERNEURON: "Balanced Ca2+ and K+ - integrative, flexible",
        CElegansNeuronClass.MOTOR: "Moderate Ca2+, high K+ - graded output control"
    }
    return descriptions.get(neuron_class, "Unknown class")

