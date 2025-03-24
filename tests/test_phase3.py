import unittest
from unittest.mock import MagicMock
from typing import Dict, Any, Optional

from langgraph_integration.langgraph_state import ConversationState
from langgraph_integration.feature_flags import FeatureFlags
from langgraph_integration.langgraph_workflow import LangGraphWorkflow
from langgraph_integration.state_converter import StateConverter
from langgraph_integration.adapters import ExistingDetectionAdapter
from langgraph_integration.hybrid_detection import HybridDetectionSystem

# Mock classes for testing
class MockFeatureFlags(FeatureFlags):
    def __init__(self, flags=None):
        self.flags = flags or {}

    def is_enabled(self, flag_name):
        return self.flags.get(flag_name, False)

class MockHybridSystem:
    def detect_negation(self, text, context=None):
        return {"is_negation": "don't" in text.lower() or "not" in text.lower(), 
                "confidence": 0.9 if ("don't" in text.lower() or "not" in text.lower()) else 0.1}

    def detect_context_switch(self, text, context=None):
        has_switch = "instead" in text.lower() or "actually" in text.lower()
        new_context = "towing" if "tow" in text.lower() else "roadside" if "roadside" in text.lower() else None
        return {
            "has_context_switch": has_switch, 
            "confidence": 0.9 if has_switch else 0.1,
            "new_context": new_context if has_switch else None
        }
    
    def analyze_intent(self, text, flow, context=None):
        if "tow" in text.lower():
            return {"intent": "request_tow", "confidence": 0.9}
        elif "road" in text.lower() or "assist" in text.lower():
            return {"intent": "request_roadside", "confidence": 0.9}
        else:
            return {"intent": "unknown", "confidence": 0.4}

class MockExistingDetector:
    def process_message(self, text, context=None):
        # Create a simple response based on the text
        if "tow" in text.lower():
            return {
                "intent": "request_tow",
                "flow": "towing",
                "response": "I'll help you with a tow truck.",
                "confidence": 0.9
            }
        elif "road" in text.lower() or "assist" in text.lower():
            return {
                "intent": "request_roadside",
                "flow": "roadside",
                "response": "I'll send roadside assistance to help you.",
                "confidence": 0.9
            }
        else:
            return {
                "intent": "unknown",
                "flow": "unknown",
                "response": "How can I help you today?",
                "confidence": 0.5
            }
    
    def detect_negation(self, text):
        return {"is_negation": "don't" in text.lower() or "not" in text.lower(), "confidence": 0.9}
    
    def detect_context_switch(self, text):
        return {"has_context_switch": "instead" in text.lower() or "actually" in text.lower(), "confidence": 0.9}

class TestStateConverter(unittest.TestCase):
    """Test the StateConverter class"""
    
    def test_from_context(self):
        """Test conversion from context to LangGraph state"""
        converter = StateConverter()
        
        # Create existing context
        context = {
            "conversation_id": "test-123",
            "turn_count": 2,
            "flow": "towing",
            "last_intent": "request_tow",
            "vehicle_type": "sedan",
            "location": "downtown"
        }
        
        # Convert to LangGraph state
        state = converter.from_context(context, "I need a tow truck")
        
        # Verify conversion
        self.assertEqual(state["conversation_id"], "test-123")
        self.assertEqual(state["turn_count"], 2)
        self.assertEqual(state["flow"], "towing")
        self.assertEqual(state["intent"], "request_tow")
        self.assertEqual(state["entities"]["vehicle_type"], "sedan")
        self.assertEqual(state["current_message"], "I need a tow truck")
    
    def test_to_context(self):
        """Test conversion from LangGraph state back to context"""
        converter = StateConverter()
        
        # Create LangGraph state
        state = ConversationState(
            conversation_id="test-123",
            turn_count=3,
            current_message="I need roadside assistance",
            flow="roadside",
            intent="request_roadside",
            entities={"vehicle_type": "SUV", "location": "highway"},
            context={"previous_context": "some_value"}
        )
        
        # Convert to context
        context = converter.to_context(state)
        
        # Verify conversion
        self.assertEqual(context["conversation_id"], "test-123")
        self.assertEqual(context["turn_count"], 3)
        self.assertEqual(context["flow"], "roadside")
        self.assertEqual(context["last_intent"], "request_roadside")
        self.assertEqual(context["vehicle_type"], "SUV")
        self.assertEqual(context["location"], "highway")
        self.assertEqual(context["previous_context"], "some_value")

class TestLangGraphWorkflow(unittest.TestCase):
    """Test the LangGraph workflow"""
    
    def test_workflow_with_langgraph_disabled(self):
        """Test workflow with LangGraph disabled"""
        # Setup
        flags = MockFeatureFlags({"use_langgraph": False})
        hybrid_system = MockHybridSystem()
        existing_detector = MockExistingDetector()
        
        workflow = LangGraphWorkflow(flags, hybrid_system, existing_detector)
        
        # Create initial state
        state = ConversationState(
            conversation_id="test-123",
            turn_count=0,
            current_message="I need a tow truck",
            messages=[],
            context={},
            flow="unknown"
        )
        
        # Invoke workflow
        result = workflow.invoke(state)
        
        # Verify result still works with fallback to existing system
        self.assertIn("response", result)
        self.assertIn("tow", result["response"].lower())
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "assistant")
    
    def test_workflow_with_langgraph_enabled(self):
        """Test workflow with LangGraph enabled"""
        # Setup
        flags = MockFeatureFlags({"use_langgraph": True})
        hybrid_system = MockHybridSystem()
        existing_detector = MockExistingDetector()
        
        workflow = LangGraphWorkflow(flags, hybrid_system, existing_detector)
        
        # Create initial state
        state = ConversationState(
            conversation_id="test-123",
            turn_count=0,
            current_message="I need a tow truck",
            messages=[],
            context={},
            flow="unknown"
        )
        
        # Invoke workflow
        result = workflow.invoke(state)
        
        # Verify result includes expected fields
        self.assertIn("response", result)
        self.assertEqual(result["turn_count"], 1)
        self.assertIn("tow", result["response"].lower())
        self.assertEqual(len(result["messages"]), 2)  # User message + assistant response
    
    def test_workflow_with_negation(self):
        """Test workflow with negation detection"""
        # Setup
        flags = MockFeatureFlags({"use_langgraph": True})
        hybrid_system = MockHybridSystem()
        existing_detector = MockExistingDetector()
        
        workflow = LangGraphWorkflow(flags, hybrid_system, existing_detector)
        
        # Create initial state with negation
        state = ConversationState(
            conversation_id="test-123",
            turn_count=0,
            current_message="I don't need a tow truck",
            messages=[],
            context={"last_intent": "request_tow"},
            flow="towing"
        )
        
        # Invoke workflow
        result = workflow.invoke(state)
        
        # Verify negation was detected and handled
        self.assertTrue(result["detected_negation"])
        self.assertIn("response", result)
        self.assertEqual(result["context"]["negated_intent"], "request_tow")
        self.assertTrue(result["needs_clarification"])
    
    def test_workflow_with_context_switch(self):
        """Test workflow with context switch detection"""
        # Setup
        flags = MockFeatureFlags({"use_langgraph": True})
        hybrid_system = MockHybridSystem()
        existing_detector = MockExistingDetector()
        
        workflow = LangGraphWorkflow(flags, hybrid_system, existing_detector)
        
        # Create initial state with context switch
        state = ConversationState(
            conversation_id="test-123",
            turn_count=1,
            current_message="Actually, I need roadside assistance instead",
            messages=[{"role": "user", "content": "I need a tow truck"}],
            context={"flow": "towing"},
            flow="towing"
        )
        
        # Invoke workflow
        result = workflow.invoke(state)
        
        # Verify context switch was detected and handled
        self.assertTrue(result["detected_context_switch"])
        self.assertEqual(result["flow"], "roadside")
        self.assertEqual(result["context"]["previous_flow"], "towing")
        self.assertEqual(result["context"]["context_switch_count"], 1)
        self.assertIn("response", result)

if __name__ == '__main__':
    unittest.main() 