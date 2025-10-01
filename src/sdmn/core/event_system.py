"""
Event system for managing simulation events and spike propagation.

This module implements a priority queue-based event system for handling
neural spikes, stimulus events, and other time-based occurrences in the
simulation.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
from abc import ABC, abstractmethod


class EventType(Enum):
    """Types of events that can occur in the simulation."""
    SPIKE = "spike"
    STIMULUS = "stimulus"
    PROBE_RECORD = "probe_record"
    NETWORK_UPDATE = "network_update"
    CUSTOM = "custom"


@dataclass
class Event:
    """
    Represents a single event in the simulation.
    
    Events are ordered by timestamp for processing in chronological order.
    """
    timestamp: float
    event_type: EventType
    source_id: str
    target_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    def __lt__(self, other: 'Event') -> bool:
        """Comparison for priority queue ordering."""
        return self.timestamp < other.timestamp
    
    def __eq__(self, other: 'Event') -> bool:
        """Equality comparison."""
        return (self.timestamp == other.timestamp and 
                self.event_type == other.event_type and
                self.source_id == other.source_id and
                self.target_id == other.target_id)


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    def handle_event(self, event: Event) -> List[Event]:
        """
        Handle an event and return any new events to schedule.
        
        Args:
            event: The event to handle
            
        Returns:
            List of new events to schedule (can be empty)
        """
        pass


class EventQueue:
    """
    Priority queue for managing simulation events.
    
    Events are processed in chronological order (earliest first).
    Supports scheduling future events and batch processing of
    simultaneous events.
    """
    
    def __init__(self):
        self._events: List[Event] = []
        self._event_counter = 0  # For maintaining insertion order when timestamps are equal
        
    def schedule_event(self, event: Event) -> None:
        """
        Schedule an event for future processing.
        
        Args:
            event: The event to schedule
        """
        # Use counter to maintain insertion order for simultaneous events
        heapq.heappush(self._events, (event.timestamp, self._event_counter, event))
        self._event_counter += 1
    
    def schedule_events(self, events: List[Event]) -> None:
        """
        Schedule multiple events.
        
        Args:
            events: List of events to schedule
        """
        for event in events:
            self.schedule_event(event)
    
    def get_next_event_time(self) -> Optional[float]:
        """
        Get the timestamp of the next event without removing it.
        
        Returns:
            Timestamp of next event, or None if queue is empty
        """
        if not self._events:
            return None
        return self._events[0][0]
    
    def get_events_at_time(self, timestamp: float, tolerance: float = 1e-10) -> List[Event]:
        """
        Retrieve all events scheduled for a specific time.
        
        Args:
            timestamp: The time to get events for
            tolerance: Numerical tolerance for time comparison
            
        Returns:
            List of events at the specified time
        """
        events = []
        while (self._events and 
               abs(self._events[0][0] - timestamp) <= tolerance):
            _, _, event = heapq.heappop(self._events)
            events.append(event)
        return events
    
    def peek_events_at_time(self, timestamp: float, tolerance: float = 1e-10) -> List[Event]:
        """
        View events at a specific time without removing them.
        
        Args:
            timestamp: The time to get events for
            tolerance: Numerical tolerance for time comparison
            
        Returns:
            List of events at the specified time
        """
        events = []
        for ts, _, event in self._events:
            if abs(ts - timestamp) <= tolerance:
                events.append(event)
            elif ts > timestamp + tolerance:
                break
        return events
    
    def is_empty(self) -> bool:
        """Check if the event queue is empty."""
        return len(self._events) == 0
    
    def size(self) -> int:
        """Get the number of events in the queue."""
        return len(self._events)
    
    def clear(self) -> None:
        """Clear all events from the queue."""
        self._events.clear()
        self._event_counter = 0


class EventProcessor:
    """
    Processes events using registered handlers.
    
    Maintains a registry of event handlers and routes events
    to appropriate handlers based on event type.
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
    
    def register_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Register an event handler for a specific event type.
        
        Args:
            event_type: The type of event to handle
            handler: The handler instance
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def unregister_handler(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Unregister an event handler.
        
        Args:
            event_type: The type of event
            handler: The handler instance to remove
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass  # Handler not found
    
    def process_event(self, event: Event) -> List[Event]:
        """
        Process an event using registered handlers.
        
        Args:
            event: The event to process
            
        Returns:
            List of new events generated by handlers
        """
        new_events = []
        
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                handler_events = handler.handle_event(event)
                new_events.extend(handler_events)
        
        return new_events
    
    def process_events(self, events: List[Event]) -> List[Event]:
        """
        Process multiple events.
        
        Args:
            events: List of events to process
            
        Returns:
            List of new events generated by all handlers
        """
        all_new_events = []
        for event in events:
            new_events = self.process_event(event)
            all_new_events.extend(new_events)
        return all_new_events
