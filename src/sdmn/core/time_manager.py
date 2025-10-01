"""
Time management for neural network simulations.

This module provides precise time management capabilities for simulation,
including variable time steps, time synchronization, and temporal
scheduling.
"""

from typing import Optional, Callable, List
from dataclasses import dataclass
import time


@dataclass
class TimeStep:
    """Represents a single time step in the simulation."""
    step_number: int
    simulation_time: float
    dt: float
    wall_time: float


class TimeManager:
    """
    Manages simulation time and temporal scheduling.
    
    Provides capabilities for:
    - Fixed and adaptive time stepping
    - Time synchronization across components
    - Performance monitoring
    - Temporal event scheduling
    """
    
    def __init__(self, dt: float = 0.1, max_time: float = 1000.0):
        """
        Initialize the time manager.
        
        Args:
            dt: Default time step in milliseconds
            max_time: Maximum simulation time in milliseconds
        """
        self.dt = dt
        self.max_time = max_time
        self.current_time = 0.0
        self.current_step = 0
        self.start_wall_time = 0.0
        self.time_history: List[TimeStep] = []
        
        # Adaptive time stepping parameters
        self.adaptive_dt = False
        self.min_dt = dt / 10.0
        self.max_dt = dt * 10.0
        self.error_tolerance = 1e-6
        
        # Callbacks
        self.step_callbacks: List[Callable[[TimeStep], None]] = []
    
    def start_simulation(self) -> None:
        """Start the simulation timer."""
        self.start_wall_time = time.time()
        self.current_time = 0.0
        self.current_step = 0
        self.time_history.clear()
    
    def advance_time(self, custom_dt: Optional[float] = None) -> TimeStep:
        """
        Advance simulation time by one step.
        
        Args:
            custom_dt: Optional custom time step for this advance
            
        Returns:
            TimeStep object representing this step
        """
        step_dt = custom_dt if custom_dt is not None else self.dt
        
        # Create time step record
        time_step = TimeStep(
            step_number=self.current_step,
            simulation_time=self.current_time,
            dt=step_dt,
            wall_time=time.time() - self.start_wall_time
        )
        
        # Update state
        self.current_time += step_dt
        self.current_step += 1
        self.time_history.append(time_step)
        
        # Call registered callbacks
        for callback in self.step_callbacks:
            callback(time_step)
        
        return time_step
    
    def is_simulation_complete(self) -> bool:
        """Check if simulation has reached maximum time."""
        return self.current_time >= self.max_time
    
    def get_simulation_progress(self) -> float:
        """Get simulation progress as percentage (0.0 to 1.0)."""
        return min(self.current_time / self.max_time, 1.0)
    
    def get_remaining_time(self) -> float:
        """Get remaining simulation time in milliseconds."""
        return max(0.0, self.max_time - self.current_time)
    
    def get_performance_stats(self) -> dict:
        """
        Get simulation performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.time_history:
            return {}
        
        total_wall_time = time.time() - self.start_wall_time
        simulation_time_ratio = self.current_time / total_wall_time if total_wall_time > 0 else 0
        
        # Calculate average step time
        if len(self.time_history) > 1:
            step_times = []
            for i in range(1, len(self.time_history)):
                step_time = self.time_history[i].wall_time - self.time_history[i-1].wall_time
                step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
            max_step_time = max(step_times)
            min_step_time = min(step_times)
        else:
            avg_step_time = max_step_time = min_step_time = 0.0
        
        return {
            'total_steps': self.current_step,
            'simulation_time': self.current_time,
            'wall_time': total_wall_time,
            'simulation_speed_ratio': simulation_time_ratio,
            'avg_step_time': avg_step_time,
            'max_step_time': max_step_time,
            'min_step_time': min_step_time,
            'steps_per_second': self.current_step / total_wall_time if total_wall_time > 0 else 0
        }
    
    def register_step_callback(self, callback: Callable[[TimeStep], None]) -> None:
        """
        Register a callback to be called after each time step.
        
        Args:
            callback: Function to call with TimeStep object
        """
        self.step_callbacks.append(callback)
    
    def unregister_step_callback(self, callback: Callable[[TimeStep], None]) -> None:
        """
        Unregister a step callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        try:
            self.step_callbacks.remove(callback)
        except ValueError:
            pass  # Callback not found
    
    def enable_adaptive_stepping(self, error_tolerance: float = 1e-6) -> None:
        """
        Enable adaptive time stepping based on error tolerance.
        
        Args:
            error_tolerance: Maximum allowed error for adaptive stepping
        """
        self.adaptive_dt = True
        self.error_tolerance = error_tolerance
    
    def disable_adaptive_stepping(self) -> None:
        """Disable adaptive time stepping and use fixed dt."""
        self.adaptive_dt = False
    
    def calculate_adaptive_dt(self, error_estimate: float) -> float:
        """
        Calculate adaptive time step based on error estimate.
        
        Args:
            error_estimate: Estimated numerical error from last step
            
        Returns:
            Recommended time step for next iteration
        """
        if not self.adaptive_dt or error_estimate <= 0:
            return self.dt
        
        # Scale time step based on error
        safety_factor = 0.9
        dt_scale = safety_factor * (self.error_tolerance / error_estimate) ** 0.2
        
        new_dt = self.dt * dt_scale
        
        # Clamp to reasonable bounds
        new_dt = max(self.min_dt, min(self.max_dt, new_dt))
        
        return new_dt
    
    def reset(self) -> None:
        """Reset time manager to initial state."""
        self.current_time = 0.0
        self.current_step = 0
        self.start_wall_time = 0.0
        self.time_history.clear()
    
    def set_time_bounds(self, max_time: float) -> None:
        """
        Set maximum simulation time.
        
        Args:
            max_time: Maximum simulation time in milliseconds
        """
        self.max_time = max_time
    
    def get_time_step_history(self) -> List[TimeStep]:
        """Get complete history of time steps."""
        return self.time_history.copy()
    
    def estimate_completion_time(self) -> Optional[float]:
        """
        Estimate wall-clock time to simulation completion.
        
        Returns:
            Estimated seconds to completion, or None if cannot estimate
        """
        if self.current_step < 2:
            return None
            
        recent_steps = self.time_history[-10:]  # Use last 10 steps for estimate
        if len(recent_steps) < 2:
            return None
            
        # Calculate average time per simulation unit
        wall_time_span = recent_steps[-1].wall_time - recent_steps[0].wall_time
        sim_time_span = recent_steps[-1].simulation_time - recent_steps[0].simulation_time
        
        if sim_time_span <= 0:
            return None
            
        time_per_sim_unit = wall_time_span / sim_time_span
        remaining_sim_time = self.get_remaining_time()
        
        return remaining_sim_time * time_per_sim_unit
