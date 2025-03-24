import pytest
import threading
import time
import json
import os
from unittest.mock import MagicMock, patch, Mock
import sys
from typing import Dict, Any

from langgraph_integration.monitoring import (
    MetricsCollector, 
    ChatbotLogger, 
    MonitoringSystem, 
    timed_execution
)
from langgraph_integration.cpu_optimizations import CPUOptimizer
from langgraph_integration.langgraph_state import ConversationState

# Skip Streamlit tests if not installed
streamlit_installed = False
try:
    import streamlit
    streamlit_installed = True
except ImportError:
    pass

# Test timed_execution decorator
def test_timed_execution():
    """Test the timed_execution decorator"""
    @timed_execution
    def test_function():
        time.sleep(0.01)
        return {"result": "test"}

    result, execution_time = test_function()

    assert "result" in result
    assert result["result"] == "test"
    assert "execution_time" in result
    assert execution_time >= 0.01

# Test ChatbotLogger
def test_chatbot_logger():
    """Test the ChatbotLogger functionality"""
    logger = ChatbotLogger()
    
    # Test methods don't raise exceptions
    logger.log_request("test-123", "Hello", {"flow": "unknown"})
    logger.log_response("test-123", "Hi there", 0.1, {"flow": "greeting"})
    logger.log_error("test-123", ValueError("Test error"), {"flow": "unknown"})
    logger.log_feature_flag_change("use_mistral", True)
    
    # Test with file logging
    import tempfile
    with tempfile.NamedTemporaryFile() as temp:
        file_logger = ChatbotLogger(log_file=temp.name)
        file_logger.log_request("test-file", "Testing file logger", {})
        # Check file was written to
        with open(temp.name, 'r') as f:
            content = f.read()
            assert "Testing file logger" in content

# Test metrics collector
def test_metrics_collector():
    """Test the MetricsCollector functionality"""
    collector = MetricsCollector()

    # Test request counting
    request_id = collector.update_request_count()
    assert request_id == 1

    # Test response recording
    test_state = {"detected_negation": True, "flow": "towing", "confidence_scores": {}}
    collector.record_response(0.5, test_state)

    metrics = collector.get_metrics()
    assert metrics["requests"] == 1
    assert metrics["successful_responses"] == 1
    assert metrics["negation_detections"] == 1
    assert abs(metrics["average_latency"] - 0.5) < 0.001
    
    # Test multiple responses for average latency
    collector.record_response(1.5, {"detected_context_switch": True})
    
    metrics = collector.get_metrics()
    assert metrics["successful_responses"] == 2
    assert metrics["context_switches"] == 1
    # Average should be (0.5 + 1.5) / 2 = 1.0
    assert abs(metrics["average_latency"] - 1.0) < 0.001
    
    # Test error recording
    collector.record_error()
    metrics = collector.get_metrics()
    assert metrics["errors"] == 1
    
    # Test history
    history = collector.get_history()
    assert len(history) == 2
    assert history[0]["latency"] == 0.5
    assert history[0]["negation"] is True

# Test monitoring system integration
def test_monitoring_system():
    """Test the MonitoringSystem integration"""
    monitoring = MonitoringSystem()
    monitoring.start_worker()
    
    try:
        # Track a request
        monitoring.track_request("test-123", "Hello", {"flow": "greeting"})
        
        # Track a response
        state = {"flow": "greeting", "intent": "greet", "detected_negation": False}
        monitoring.track_response("test-123", "Hi there", 0.2, state)
        
        # Give worker thread time to process
        time.sleep(0.1)
        
        # Get metrics
        metrics = monitoring.get_metrics()
        assert metrics["requests"] == 1
        assert metrics["successful_responses"] == 1
        
        # Track an error
        monitoring.track_error("test-123", ValueError("Test error"), {})
        
        # Give worker thread time to process
        time.sleep(0.1)
        
        # Check metrics again
        metrics = monitoring.get_metrics()
        assert metrics["errors"] == 1
    finally:
        # Clean up resources
        monitoring.stop_worker()

# Test CPU optimizer
def test_cpu_optimizer():
    """Test the CPUOptimizer functionality"""
    optimizer = CPUOptimizer()
    
    # Test cached text analysis
    result1 = optimizer.cached_text_analysis("test")
    result2 = optimizer.cached_text_analysis("test")
    
    # Both results should be identical due to caching
    assert result1 == result2
    assert result1["text_length"] == 4
    
    # Test batch processing
    batch_results = optimizer.batch_process(["test1", "test2", "test3"])
    assert len(batch_results) == 3
    assert batch_results[0]["text_length"] == 5
    assert batch_results[1]["text_length"] == 5
    assert batch_results[2]["text_length"] == 5
    
    # Test timed_with_timeout
    def sample_function(x, y):
        time.sleep(0.01)
        return x + y
        
    result, execution_time = optimizer.timed_with_timeout(sample_function, 3, 4, timeout=1.0)
    assert result == 7
    assert execution_time >= 0.01
    
    # Test memoize decorator
    @optimizer.memoize
    def expensive_function(x):
        time.sleep(0.01)
        return x * 2
        
    # First call should be computed
    start_time = time.time()
    result1 = expensive_function(5)
    first_execution_time = time.time() - start_time
    
    # Second call should be cached
    start_time = time.time()
    result2 = expensive_function(5)
    second_execution_time = time.time() - start_time
    
    assert result1 == result2 == 10
    assert first_execution_time >= 0.01
    assert second_execution_time < first_execution_time  # Should be faster due to caching
    
    # Test parallelization
    def square(x):
        time.sleep(0.01)  # Simulate work
        return x * x
        
    numbers = list(range(10))
    results = optimizer.parallelize(square, numbers, max_workers=4)
    
    assert len(results) == 10
    assert results == [x*x for x in numbers]

# Test streamlit integration (conditionally)
@pytest.mark.skipif(not streamlit_installed, reason="Streamlit not installed")
def test_streamlit_app():
    """Test the StreamlitApp initialization"""
    with patch('streamlit.session_state', MagicMock()), \
         patch('streamlit.sidebar', MagicMock()), \
         patch('streamlit.set_page_config', MagicMock()):
        
        from langgraph_integration.streamlit_integration import StreamlitApp
        
        # Test initialization without errors
        app = StreamlitApp()
        
        # Test non-UI message processing
        result = app.process_message("I need a tow truck")
        
        assert "response" in result
        assert "context" in result
        assert "message_id" in result
        assert "latency" in result
        
        # Clean up
        app.cleanup()

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 