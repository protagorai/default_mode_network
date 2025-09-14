"""
State management for neural network simulations.

This module provides centralized state management capabilities,
including serialization, checkpointing, and state synchronization
across distributed components.
"""

from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, asdict
import pickle
import json
import hashlib
from pathlib import Path
from abc import ABC, abstractmethod


class StateSerializable(ABC):
    """Abstract base class for objects that can be serialized to state."""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state from a dictionary."""
        pass
    
    @abstractmethod
    def get_state_version(self) -> str:
        """Get the version of the state format."""
        pass


@dataclass
class StateCheckpoint:
    """Represents a simulation state checkpoint."""
    timestamp: float
    step_number: int
    state_hash: str
    metadata: Dict[str, Any]
    file_path: Optional[Path] = None


class StateManager:
    """
    Manages state for all simulation components.
    
    Provides capabilities for:
    - Centralized state storage and retrieval
    - State serialization and deserialization
    - Checkpointing and recovery
    - State validation and consistency checking
    """
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize state manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints (optional)
        """
        self._state_registry: Dict[str, StateSerializable] = {}
        self._state_cache: Dict[str, Dict[str, Any]] = {}
        self._checkpoints: List[StateCheckpoint] = []
        self.checkpoint_dir = checkpoint_dir
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def register_object(self, object_id: str, obj: StateSerializable) -> None:
        """
        Register an object for state management.
        
        Args:
            object_id: Unique identifier for the object
            obj: The object to register
        """
        if not isinstance(obj, StateSerializable):
            raise TypeError(f"Object {object_id} must implement StateSerializable")
        
        self._state_registry[object_id] = obj
        # Cache initial state
        self._state_cache[object_id] = obj.get_state()
    
    def unregister_object(self, object_id: str) -> None:
        """
        Unregister an object from state management.
        
        Args:
            object_id: Identifier of object to unregister
        """
        self._state_registry.pop(object_id, None)
        self._state_cache.pop(object_id, None)
    
    def get_object_state(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state of a registered object.
        
        Args:
            object_id: Identifier of the object
            
        Returns:
            State dictionary or None if object not found
        """
        if object_id in self._state_registry:
            state = self._state_registry[object_id].get_state()
            self._state_cache[object_id] = state  # Update cache
            return state
        return None
    
    def set_object_state(self, object_id: str, state: Dict[str, Any]) -> bool:
        """
        Set state of a registered object.
        
        Args:
            object_id: Identifier of the object
            state: State dictionary to set
            
        Returns:
            True if successful, False otherwise
        """
        if object_id in self._state_registry:
            try:
                self._state_registry[object_id].set_state(state)
                self._state_cache[object_id] = state
                return True
            except Exception:
                return False
        return False
    
    def get_full_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Get complete state of all registered objects.
        
        Returns:
            Dictionary mapping object IDs to their states
        """
        full_state = {}
        for object_id in self._state_registry:
            full_state[object_id] = self.get_object_state(object_id)
        return full_state
    
    def set_full_state(self, full_state: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """
        Set state for multiple objects.
        
        Args:
            full_state: Dictionary mapping object IDs to states
            
        Returns:
            Dictionary mapping object IDs to success status
        """
        results = {}
        for object_id, state in full_state.items():
            results[object_id] = self.set_object_state(object_id, state)
        return results
    
    def create_checkpoint(self, timestamp: float, step_number: int, 
                         metadata: Optional[Dict[str, Any]] = None) -> StateCheckpoint:
        """
        Create a state checkpoint.
        
        Args:
            timestamp: Simulation timestamp
            step_number: Simulation step number
            metadata: Additional metadata to store
            
        Returns:
            Created checkpoint object
        """
        full_state = self.get_full_state()
        
        # Calculate state hash for verification
        state_str = json.dumps(full_state, sort_keys=True)
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        
        checkpoint = StateCheckpoint(
            timestamp=timestamp,
            step_number=step_number,
            state_hash=state_hash,
            metadata=metadata or {}
        )
        
        # Save to file if checkpoint directory is configured
        if self.checkpoint_dir:
            filename = f"checkpoint_step_{step_number}_{int(timestamp)}.pkl"
            file_path = self.checkpoint_dir / filename
            
            checkpoint_data = {
                'checkpoint': asdict(checkpoint),
                'state': full_state
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            checkpoint.file_path = file_path
        
        self._checkpoints.append(checkpoint)
        return checkpoint
    
    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> bool:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint: The checkpoint to restore
            
        Returns:
            True if restoration successful, False otherwise
        """
        try:
            if checkpoint.file_path and checkpoint.file_path.exists():
                # Load from file
                with open(checkpoint.file_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    full_state = checkpoint_data['state']
            else:
                # This checkpoint might not have been saved to file
                return False
            
            # Verify state integrity
            state_str = json.dumps(full_state, sort_keys=True)
            computed_hash = hashlib.sha256(state_str.encode()).hexdigest()
            
            if computed_hash != checkpoint.state_hash:
                raise ValueError("State hash mismatch - checkpoint may be corrupted")
            
            # Restore state
            results = self.set_full_state(full_state)
            return all(results.values())
            
        except Exception:
            return False
    
    def get_checkpoint_history(self) -> List[StateCheckpoint]:
        """Get list of all checkpoints."""
        return self._checkpoints.copy()
    
    def find_checkpoint_by_step(self, step_number: int) -> Optional[StateCheckpoint]:
        """
        Find checkpoint by step number.
        
        Args:
            step_number: The step number to search for
            
        Returns:
            Checkpoint if found, None otherwise
        """
        for checkpoint in self._checkpoints:
            if checkpoint.step_number == step_number:
                return checkpoint
        return None
    
    def find_checkpoint_by_time(self, timestamp: float, 
                               tolerance: float = 1e-6) -> Optional[StateCheckpoint]:
        """
        Find checkpoint by timestamp.
        
        Args:
            timestamp: The timestamp to search for
            tolerance: Time tolerance for matching
            
        Returns:
            Checkpoint if found, None otherwise
        """
        for checkpoint in self._checkpoints:
            if abs(checkpoint.timestamp - timestamp) <= tolerance:
                return checkpoint
        return None
    
    def cleanup_checkpoints(self, keep_count: int = 10) -> int:
        """
        Clean up old checkpoint files, keeping only the most recent.
        
        Args:
            keep_count: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints removed
        """
        if not self.checkpoint_dir:
            return 0
        
        # Sort checkpoints by step number (most recent first)
        sorted_checkpoints = sorted(self._checkpoints, 
                                  key=lambda cp: cp.step_number, reverse=True)
        
        removed_count = 0
        for checkpoint in sorted_checkpoints[keep_count:]:
            if checkpoint.file_path and checkpoint.file_path.exists():
                try:
                    checkpoint.file_path.unlink()
                    removed_count += 1
                except OSError:
                    pass  # File might be in use or protected
        
        # Keep only recent checkpoints in memory
        self._checkpoints = sorted_checkpoints[:keep_count]
        
        return removed_count
    
    def validate_state_consistency(self) -> Dict[str, List[str]]:
        """
        Validate consistency of all registered object states.
        
        Returns:
            Dictionary mapping object IDs to lists of validation errors
        """
        validation_results = {}
        
        for object_id, obj in self._state_registry.items():
            errors = []
            
            try:
                # Get current state
                current_state = obj.get_state()
                
                # Try to set the state back (round-trip test)
                obj.set_state(current_state)
                
                # Check if state is serializable
                json.dumps(current_state)
                
            except Exception as e:
                errors.append(f"State consistency error: {str(e)}")
            
            validation_results[object_id] = errors
        
        return validation_results
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary information about managed state.
        
        Returns:
            Summary statistics and information
        """
        total_objects = len(self._state_registry)
        total_checkpoints = len(self._checkpoints)
        
        # Calculate total state size (approximate)
        total_state_size = 0
        for state in self._state_cache.values():
            try:
                total_state_size += len(json.dumps(state))
            except (TypeError, ValueError):
                pass  # Skip non-serializable states
        
        return {
            'registered_objects': total_objects,
            'total_checkpoints': total_checkpoints,
            'approximate_state_size_bytes': total_state_size,
            'checkpoint_directory': str(self.checkpoint_dir) if self.checkpoint_dir else None,
            'object_ids': list(self._state_registry.keys())
        }
