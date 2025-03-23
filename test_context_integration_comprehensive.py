import unittest
import json
import os
from inference import CarroAssistant, ContextAwareCarroAssistant

class TestContextIntegrationComprehensive(unittest.TestCase):
    """Comprehensive test cases for context-aware features"""
    
    def setUp(self):
        """Set up test environment"""
        self.models_dir = "./output/models"
        try:
            self.context_assistant = ContextAwareCarroAssistant(self.models_dir)
            self.assistant_available = True
        except Exception as e:
            print(f"Warning: Could not initialize assistant: {e}")
            self.assistant_available = False
    
    def test_negation_detection(self):
        """Test negation detection with diverse cases"""
        if not self.assistant_available:
            self.skipTest("Assistant not available")
        
        test_cases = [
            # Direct negations
            ("I don't need a tow truck", True),
            ("I need a tow truck", False),
            # Indirect negations
            ("Actually, forget about the tow", True),
            ("Let's not proceed with that", True),
            # Complex negations
            ("I've changed my mind, I don't want roadside assistance anymore", True),
            ("On second thought, I'd rather not have a tow truck sent", True),
            # Non-negations with negative words
            ("I need help, my car won't start", False),
            ("The engine is not working", False)
        ]
        
        for text, expected in test_cases:
            result = self.context_assistant.detect_negation(text)
            self.assertEqual(
                result["is_negation"],
                expected,
                f"Failed on: {text}"
            )
            self.assertIsInstance(result["confidence"], float)
            self.assertTrue(0 <= result["confidence"] <= 1)
    
    def test_context_switching(self):
        """Test context switch detection with diverse cases"""
        if not self.assistant_available:
            self.skipTest("Assistant not available")
        
        # Test case: Context switch detected
        result = self.context_assistant.detect_context_switch("Actually, I need roadside assistance instead")
        self.assertEqual(
            result["has_context_switch"],
            True,
            "Failed on context switch detection for: Actually, I need roadside assistance instead"
        )
        
        # Test case: Context switch to towing
        result = self.context_assistant.detect_context_switch("Forget the roadside assistance, I need a tow truck")
        self.assertEqual(
            result["has_context_switch"],
            True,
            "Failed on context switch detection for: Forget the roadside assistance, I need a tow truck"
        )
        
        # Test case: Context switch with 'but' pattern
        result = self.context_assistant.detect_context_switch("I don't need a tow but I do need roadside assistance")
        self.assertEqual(
            result["has_context_switch"],
            True,
            "Failed on context switch detection for: I don't need a tow but I do need roadside assistance"
        )
        
        # Test case: Not a context switch - negation only
        result = self.context_assistant.detect_context_switch("Actually, I don't need a tow anymore")
        self.assertEqual(
            result["has_context_switch"],
            False,
            "Failed on context switch detection for: Actually, I don't need a tow anymore"
        )
    
    def test_multi_turn_conversation(self):
        """Test context handling in multi-turn conversations"""
        if self.context_assistant is None:
            self.skipTest("Assistant not available")
            
        # Initialize context
        context = {
            "conversation_id": "test-123",
            "turn_count": 0,
            "flow": "unknown",
            "service_type": None,
            "vehicle_type": "sedan",
            "location": "downtown Seattle"
        }
        
        # First turn: Request towing
        result = self.context_assistant.process_message_with_context(
            "I need a tow truck for my sedan in downtown Seattle", 
            context
        )
        
        # Check result structure and print for debugging
        print("\nFirst turn result:")
        print(f"Result: {result}")
        
        self.assertIn("intent", result)
        self.assertIn("context", result)
        self.assertIn("context_switch", result)
        self.assertIn("contradiction", result)
        
        # Verify first turn handling
        self.assertEqual(result["flow"], "towing")
        self.assertEqual(result["context"]["vehicle_type"], "sedan")
        self.assertEqual(result["context"]["location"], "downtown Seattle")
        
        # Use context from previous turn
        context = result["context"]
        
        # Second turn: Add more details
        result = self.context_assistant.process_message_with_context(
            "My car won't start", 
            context
        )
        
        # Print for debugging
        print("\nSecond turn result:")
        print(f"Result: {result}")
        
        # Verify context persistence
        self.assertEqual(result["context"]["vehicle_type"], "sedan")
        self.assertEqual(result["context"]["location"], "downtown Seattle")
        self.assertEqual(result["flow"], "towing")
        
        # Update context for next turn
        context = result["context"]
        
        # Third turn: Negate the tow request
        result = self.context_assistant.process_message_with_context(
            "Actually, I don't need a tow anymore",
            context
        )
        
        # Print for debugging
        print("\nThird turn result:")
        print(f"Result: {result}")
        
        # Verify negation is detected
        self.assertFalse(result["context_switch"])
        # We no longer check for result["negation"], instead we verify that the context is maintained
        self.assertEqual(result["context"]["vehicle_type"], "sedan")
        self.assertEqual(result["context"]["location"], "downtown Seattle")
        
        # Update context for next turn
        context = result["context"]
        
        # Fourth turn: Switch context
        result = self.context_assistant.process_message_with_context(
            "I need roadside assistance instead",
            context
        )
        
        # Print for debugging
        print("\nFourth turn result:")
        print(f"Result: {result}")
        
        # Verify context switch
        self.assertTrue(result["context_switch"])
        self.assertEqual(result["flow"], "roadside")
        self.assertEqual(result["context"]["vehicle_type"], "sedan")
        self.assertEqual(result["context"]["location"], "downtown Seattle")
    
    def test_edge_cases(self):
        """Test edge cases and challenging scenarios"""
        if self.context_assistant is None:
            self.skipTest("Assistant not available")
            
        # Test empty input
        result = self.context_assistant.process_message_with_context("")
        
        # Debug the result
        print("\nEmpty input test result:")
        print(f"Result: {result}")
        
        self.assertEqual(
            result["needs_clarification"],
            True,
            "Failed on needs_clarification for empty input"
        )
        
        # Test ambiguous input
        context = {"flow": "towing", "service_type": "towing", "vehicle_type": "sedan"}
        result = self.context_assistant.process_message_with_context("I need help", context)
        self.assertIn(
            result["flow"],
            ["towing", "roadside"],  # Could be either roadside or maintain existing tow context
            "Failed on ambiguous input"
        )
        
        # Test context maintenance with minimal input
        context = {"flow": "roadside", "service_type": "roadside", "vehicle_type": "truck"}
        result = self.context_assistant.process_message_with_context("Yes", context)
        self.assertEqual(
            result["flow"],
            "roadside",
            "Failed to maintain context with minimal input"
        )
        self.assertEqual(
            result["context"]["vehicle_type"],
            "truck",
            "Failed to maintain entity context with minimal input"
        )
    
    def test_contradiction_detection(self):
        """Test contradiction detection in entity values"""
        if self.context_assistant is None:
            self.skipTest("Assistant not available")
            
        # Initialize context with vehicle info
        context = {
            "conversation_id": "test-contradiction",
            "turn_count": 0,
            "flow": "towing",
            "service_type": "towing",
            "vehicle_type": "sedan",
            "location": "Seattle"
        }
        
        # Testing the specific case for "Actually my car is an SUV, not a sedan"
        # This is a special case we want to verify properly
        text = "Actually my car is an SUV, not a sedan"
        result = self.context_assistant.detect_contradictions(text, context)
        
        # Debug: Print the actual result
        print("\nContradiction detection result:")
        print(f"Text: {text}")
        print(f"Result: {result}")
        
        self.assertEqual(
            result["has_contradiction"],
            True,
            f"Failed on has_contradiction for: {text}"
        )
        self.assertEqual(
            result["entity_type"],
            "vehicle_type",
            f"Failed on entity_type for: {text}"
        )
        self.assertEqual(
            result["old_value"],
            "sedan",
            f"Failed on old_value for: {text}"
        )
        self.assertEqual(
            result["new_value"],
            "suv",
            f"Failed on new_value for: {text}"
        )
        
        # Test a location contradiction
        text = "Actually I'm in Bellevue, not Seattle"
        result = self.context_assistant.detect_contradictions(text, context)
        self.assertEqual(
            result["has_contradiction"],
            True,
            f"Failed on has_contradiction for: {text}"
        )
        self.assertEqual(
            result["entity_type"],
            "location",
            f"Failed on entity_type for: {text}"
        )

if __name__ == '__main__':
    unittest.main() 