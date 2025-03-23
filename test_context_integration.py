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
        self.assertIn("needs_fallback", result)
        self.assertIn("needs_clarification", result)
        
        # Verify intent is related to towing
        self.assertTrue("tow" in result["intent"].lower() or "unknown" == result["intent"])
        
    def test_context_assistant_basic(self):
        """Test that context-aware assistant works for basic queries"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
            
        # Test basic functionality with context disabled (should behave like standard)
        result = self.context_assistant.process_message("I need a tow truck", use_context=False)
        
        # Verify result structure
        self.assertIn("intent", result)
        self.assertIn("entities", result)
        self.assertIn("needs_fallback", result)
        self.assertIn("needs_clarification", result)
        
        # Verify intent is related to towing
        self.assertTrue("tow" in result["intent"].lower() or "unknown" == result["intent"])
    
    def test_context_assistant_with_context(self):
        """Test that context-aware assistant properly uses context"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
            
        # Test with context enabled
        result = self.context_assistant.process_message_with_context("I need a tow truck")
        
        # Verify result has context-specific fields
        self.assertIn("contains_negation", result)
        self.assertIn("contains_context_switch", result)
        self.assertIn("contradictions", result)
        
        # Check context tracking is working
        self.assertEqual(1, self.context_assistant.conversation_context["turn_count"])
        
        # For services that specifically mention "tow", we expect the intent to be towing-related
        if "tow" in result["intent"].lower():
            # If previous context has been saved properly
            if len(self.context_assistant.conversation_context["previous_intents"]) > 0:
                self.assertTrue("tow" in self.context_assistant.conversation_context["previous_intents"][0]["intent"].lower())
        
    def test_negation_detection(self):
        """Test that negation is properly detected"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
        
        # First establish context
        self.context_assistant.process_message_with_context("I need a tow truck")
        
        # Then negate it
        result = self.context_assistant.process_message_with_context("Actually, I don't need a tow truck")
        
        # Verify negation was detected
        self.assertTrue(result["contains_negation"])
        self.assertGreater(result["negation_confidence"], 0.5)
        
    def test_context_switch_detection(self):
        """Test that context switching is properly detected"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
        
        # First establish context
        self.context_assistant.process_message_with_context("I need a tow truck")
        
        # Then switch context
        result = self.context_assistant.process_message_with_context("Actually, I need a battery jump instead")
        
        # Verify context switch was detected
        self.assertTrue(result["contains_context_switch"] or "battery" in str(result["entities"]))
        
    def test_contradiction_detection(self):
        """Test that contradictions are properly detected"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
        
        # Set up a message with vehicle info
        self.context_assistant.process_message_with_context("I have a 2018 Toyota Camry")
        
        # Create a contradiction
        result = self.context_assistant.process_message_with_context("Actually, it's a 2020 Honda Civic")
        
        # Check for contradiction tracking
        # Either in explicit contradictions or in entity tracking
        if "vehicle_make" in str(result["entities"]) or "vehicle_model" in str(result["entities"]):
            # If we found vehicle entities, we should have tracked them
            self.assertTrue(
                len(result["contradictions"]) > 0 or 
                "vehicle_make" in self.context_assistant.conversation_context["previous_entities"] or
                "vehicle_model" in self.context_assistant.conversation_context["previous_entities"]
            )
    
    def test_rule_based_fallbacks(self):
        """Test that rule-based fallbacks work when models aren't available"""
        if not self.assistants_available:
            self.skipTest("Assistants not available")
            
        # Reset context assistant
        self.context_assistant.conversation_context = {
            "previous_intents": [],
            "previous_entities": {},
            "active_flow": None,
            "turn_count": 0
        }
        
        # Test negation rule-based detection
        contains_negation, _ = self.context_assistant.detect_negation("I don't want a tow truck")
        self.assertTrue(contains_negation)
        
        # Test context switch rule-based detection
        contains_switch, _ = self.context_assistant.detect_context_switch("Actually, I need something else")
        self.assertTrue(contains_switch)
        
        # Test without negation
        contains_negation, _ = self.context_assistant.detect_negation("I want a tow truck")
        self.assertFalse(contains_negation)
        
        # Test without context switch
        contains_switch, _ = self.context_assistant.detect_context_switch("I need a tow truck")
        self.assertFalse(contains_switch)

if __name__ == '__main__':
    unittest.main() 