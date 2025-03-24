import logging
import time
import json
import os
from typing import Dict, Any, Optional, Callable, List, TypeVar, Tuple
from datetime import datetime
import threading
import queue
import functools
import sys

# Type variables for generic function return types
R = TypeVar('R')

def timed_execution(func):
    """Decorator to time function execution and include elapsed time in result"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # If result is a dict, add execution time to it
        if isinstance(result, dict):
            result["execution_time"] = execution_time
        
        return result, execution_time
    return wrapper

class ChatbotLogger:
    """Structured logging for chatbot operations"""
    
    def __init__(self, log_file: Optional[str] = None, log_level: int = logging.INFO):
        self.logger = logging.getLogger("chatbot_logger")
        self.logger.setLevel(log_level)
        
        # Configure console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console.setFormatter(formatter)
        
        # Add console handler to logger
        self.logger.addHandler(console)
        
        # Configure file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_request(self, message_id: str, text: str, context: Dict[str, Any]) -> None:
        """Log incoming request"""
        self.logger.info(
            f"Request {message_id}: '{text[:50]}{'...' if len(text) > 50 else ''}' " 
            f"Context: {json.dumps(context, default=str)[:100]}..."
        )
    
    def log_response(self, message_id: str, response: str, latency: float, state: Dict[str, Any]) -> None:
        """Log outgoing response with latency and state info"""
        # Create a simplified state summary for logging
        state_summary = {
            "flow": state.get("flow", "unknown"),
            "intent": state.get("intent", "unknown"),
            "negation": state.get("detected_negation", False),
            "context_switch": state.get("detected_context_switch", False),
            "turn_count": state.get("turn_count", 0)
        }
        
        self.logger.info(
            f"Response {message_id} ({latency:.4f}s): '{response[:50]}{'...' if len(response) > 50 else ''}' " 
            f"State: {json.dumps(state_summary, default=str)}"
        )
    
    def log_error(self, message_id: str, error: Exception, context: Dict[str, Any]) -> None:
        """Log errors with context"""
        self.logger.error(
            f"Error {message_id}: {error.__class__.__name__}: {str(error)} " 
            f"Context: {json.dumps(context, default=str)[:100]}..."
        )
    
    def log_feature_flag_change(self, flag_name: str, enabled: bool) -> None:
        """Log changes to feature flags"""
        self.logger.info(f"Feature flag '{flag_name}' set to: {enabled}")
    
    def log_workflow_node(self, node_name: str, state_before: Dict[str, Any], state_after: Dict[str, Any]) -> None:
        """Log state changes within LangGraph workflow nodes"""
        # Only log important changes for brevity
        important_keys = ["flow", "intent", "detected_negation", "detected_context_switch"]
        changes = {}
        
        for key in important_keys:
            if key in state_before or key in state_after:
                before_val = state_before.get(key, "N/A")
                after_val = state_after.get(key, "N/A")
                if before_val != after_val:
                    changes[key] = {"before": before_val, "after": after_val}
        
        if changes:
            self.logger.debug(
                f"Node '{node_name}' processed with changes: {json.dumps(changes, default=str)}"
            )

class MetricsCollector:
    """Collect and aggregate performance metrics"""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics = {
            "requests": 0,
            "successful_responses": 0,
            "errors": 0,
            "average_latency": 0.0,
            "negation_detections": 0,
            "context_switches": 0,
            "fallbacks": 0,
            "rule_based_decisions": 0,
            "ml_based_decisions": 0
        }
        self.history = []
        self.lock = threading.Lock()

    def update_request_count(self) -> int:
        """Increment and return request count"""
        with self.lock:
            self.metrics["requests"] += 1
            return self.metrics["requests"]

    def record_response(self, latency: float, state: Dict[str, Any]) -> None:
        """Record response metrics"""
        with self.lock:
            # Update success count
            self.metrics["successful_responses"] += 1

            # Update latency metrics
            total_latency = self.metrics["average_latency"] * (self.metrics["successful_responses"] - 1)
            self.metrics["average_latency"] = (total_latency + latency) / self.metrics["successful_responses"]

            # Update feature-specific metrics
            if state.get("detected_negation", False):
                self.metrics["negation_detections"] += 1

            if state.get("detected_context_switch", False):
                self.metrics["context_switches"] += 1

            if state.get("should_fallback", False):
                self.metrics["fallbacks"] += 1

            # Record which system made the decision
            confidence_scores = state.get("confidence_scores", {})
            if confidence_scores.get("rule_based_decision", False):
                self.metrics["rule_based_decisions"] += 1
            elif confidence_scores.get("ml_based_decision", False):
                self.metrics["ml_based_decisions"] += 1

            # Add to history
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "latency": latency,
                "intent": state.get("intent", "unknown"),
                "flow": state.get("flow", "unknown"),
                "negation": state.get("detected_negation", False),
                "context_switch": state.get("detected_context_switch", False)
            })

            # Trim history if needed
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

    def record_error(self) -> None:
        """Record an error"""
        with self.lock:
            self.metrics["errors"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            metrics_copy = self.metrics.copy()
            metrics_copy["timestamp"] = datetime.now().isoformat()
            return metrics_copy

    def get_history(self) -> List[Dict[str, Any]]:
        """Get request history"""
        with self.lock:
            return self.history.copy()

class MonitoringSystem:
    """Combined monitoring system with logging and metrics"""
    
    def __init__(self, 
                 log_file: Optional[str] = None, 
                 log_level: int = logging.INFO,
                 max_history: int = 100,
                 polling_interval: float = 0.1):
        self.logger = ChatbotLogger(log_file, log_level)
        self.metrics = MetricsCollector(max_history)
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.task_queue = queue.Queue()
        self.polling_interval = polling_interval
    
    def start_worker(self) -> None:
        """Start background worker thread for processing metrics"""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return  # Already running
        
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def stop_worker(self) -> None:
        """Stop background worker thread"""
        if self.worker_thread is not None:
            self.stop_event.set()
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
    
    def _worker_loop(self) -> None:
        """Background worker loop for processing tasks"""
        while not self.stop_event.is_set():
            try:
                # Process all pending tasks
                while not self.task_queue.empty():
                    task, args, kwargs = self.task_queue.get_nowait()
                    try:
                        task(*args, **kwargs)
                    except Exception as e:
                        print(f"Error in monitoring task: {str(e)}")
                    finally:
                        self.task_queue.task_done()
                
                # Sleep before checking again
                time.sleep(self.polling_interval)
            except Exception as e:
                print(f"Error in monitoring worker: {str(e)}")
    
    def track_request(self, message_id: str, text: str, context: Dict[str, Any]) -> int:
        """Track incoming request"""
        request_count = self.metrics.update_request_count()
        self.task_queue.put((self.logger.log_request, (message_id, text, context), {}))
        return request_count
    
    def track_response(self, message_id: str, response: str, latency: float, state: Dict[str, Any]) -> None:
        """Track outgoing response"""
        # Process metrics directly for the test to work correctly
        self.metrics.record_response(latency, state)
        
        # Queue logging task
        self.task_queue.put((self.logger.log_response, (message_id, response, latency, state), {}))
    
    def track_error(self, message_id: str, error: Exception, context: Dict[str, Any]) -> None:
        """Track error"""
        # Process error directly for the test to work correctly
        self.metrics.record_error()
        
        # Queue logging task
        self.task_queue.put((self.logger.log_error, (message_id, error, context), {}))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.get_metrics()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get request history"""
        return self.metrics.get_history() 