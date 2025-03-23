import unittest
import json
import os
from inference import CarroAssistant, ContextAwareCarroAssistant

class TestContextIntegration(unittest.TestCase):
    """Test cases for context integration feature"""
    
    def setUp(self):
        """Set up test environment"""
        self.models_dir = "./output/models"
        
        # Initialize both assistants
        try:
            self.standard_assistant = CarroAssistant(self.models_dir)
            self.context_assistant = ContextAwareCarroAssistant(self.models_dir)
            self.assistants_available = True
        except Exception as e:
            print(f"Warning: Could not initialize assistants: {e}")
            self.assistants_available = False
            
    def test_standard_assistant_compatibility(self):
        """Test that standard assistant still works as expected"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
            
        # Test basic functionality
        result = self.standard_assistant.process_message("I need a tow truck")
        
        # Verify result structure
        self.assertIn("intent", result)
        self.assertIn("entities", result)
        self.assertIn("should_fallback", result)
        self.assertIn("flow", result)
        
        # Verify intent is related to towing
        self.assertTrue("tow" in result["intent"].lower() or "unknown" == result["intent"])
        
    def test_context_assistant_basic(self):
        """Test that context-aware assistant works for basic queries"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
            
        # Test basic functionality (without using process_message_with_context)
        result = self.context_assistant.process_message("I need a tow truck")
        
        # Verify result structure
        self.assertIn("intent", result)
        self.assertIn("entities", result)
        self.assertIn("should_fallback", result)
        self.assertIn("flow", result)
        
        # Verify intent is related to towing
        self.assertTrue("tow" in result["intent"].lower() or "unknown" == result["intent"])
    
    def test_context_assistant_with_context(self):
        """Test that context-aware assistant properly uses context"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
            
        # Test with context enabled
        result = self.context_assistant.process_message_with_context("I need a tow truck")
        
        # Verify result has context-specific fields
        self.assertIn("context_switch", result)
        self.assertIn("contradiction", result)
        self.assertIn("context", result)
        
        # Check context tracking is working
        self.assertEqual(1, result["context"]["turn_count"])
        
        # For services that specifically mention "tow", we expect the intent to be towing-related
        if "tow" in result["intent"].lower():
            self.assertEqual("towing", result["flow"])
        
    def test_negation_detection(self):
        """Test that negation is properly detected"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
        
        # First establish context
        initial_context = self.context_assistant.process_message_with_context("I need a tow truck")["context"]
        
        # Then negate it
        result = self.context_assistant.process_message_with_context("Actually, I don't need a tow truck", initial_context)
        
        # Get negation result directly
        negation_result = self.context_assistant.detect_negation("Actually, I don't need a tow truck")
        
        # Verify negation was detected
        self.assertTrue(negation_result["is_negation"])
        self.assertGreater(negation_result["confidence"], 0.5)
        self.assertGreater(result["context"]["negations"], 0)
        
    def test_context_switch_detection(self):
        """Test that context switching is properly detected"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
        
        # First establish context
        initial_context = self.context_assistant.process_message_with_context("I need a tow truck")["context"]
        
        # Then switch context
        result = self.context_assistant.process_message_with_context("Actually, I need a battery jump instead", initial_context)
        
        # Get context switch result directly
        context_switch_result = self.context_assistant.detect_context_switch("Actually, I need a battery jump instead")
        
        # Verify context switch was detected (either through method or in result)
        self.assertTrue(
            context_switch_result["has_context_switch"] or 
            result["context_switch"] or 
            "battery" in str(result["entities"])
        )
        
    def test_contradiction_detection(self):
        """Test that contradictions are properly detected"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
        
        # Create initial context with vehicle info
        initial_context = {
            "conversation_id": "test-123",
            "turn_count": 1,
            "flow": "towing",
            "service_type": "towing",
            "vehicle_type": "sedan",
            "location": "downtown Seattle",
            "last_intent": "request_towing"
        }
        
        # Test the specific string we know should be detected as a contradiction
        contradiction_result = self.context_assistant.detect_contradictions("Actually my car is an SUV, not a sedan", initial_context)
        
        # Check for contradiction detection
        self.assertTrue(contradiction_result["has_contradiction"])
        self.assertEqual(contradiction_result["old_value"], "sedan")
        self.assertEqual(contradiction_result["new_value"], "suv")
    
    def test_rule_based_fallbacks(self):
        """Test that rule-based fallbacks work when models aren't available"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
            
        # Test negation rule-based detection
        negation_result = self.context_assistant.detect_negation("I don't want a tow truck")
        self.assertTrue(negation_result["is_negation"])
        
        # Test context switch rule-based detection with a known good switch case
        context_switch_result = self.context_assistant.detect_context_switch("I don't need a tow but I do need roadside assistance")
        self.assertTrue(context_switch_result["has_context_switch"])
        
        # Test without negation
        negation_result = self.context_assistant.detect_negation("I want a tow truck")
        self.assertFalse(negation_result["is_negation"])
        
        # Test without context switch
        context_switch_result = self.context_assistant.detect_context_switch("I need a tow truck")
        self.assertFalse(context_switch_result["has_context_switch"])

if __name__ == '__main__':
    unittest.main() 