import unittest
from unittest.mock import patch, MagicMock
import json
import os
from typing import Dict, Any, Optional

from langgraph_integration import MistralEnhancer, HybridDetectionSystem, FeatureFlags

class MockFeatureFlags(FeatureFlags):
    def __init__(self, enable_mistral: bool = False):
        self.enable_mistral = enable_mistral
        self.flags = {"use_mistral": enable_mistral, "hybrid_detection": enable_mistral}
    
    def is_enabled(self, flag_name: str) -> bool:
        return self.flags.get(flag_name, False)
    
    # Keep the original method for compatibility
    def is_feature_enabled(self, feature_name: str) -> bool:
        if feature_name == "mistral_integration":
            return self.enable_mistral
        return False

class MockExistingDetector:
    def detect_negation(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if "not" in text.lower() or "don't" in text.lower():
            return {"is_negation": True, "confidence": 0.9}
        return {"is_negation": False, "confidence": 0.8}
    
    def detect_context_switch(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if "instead" in text.lower() or "actually" in text.lower():
            return {"has_context_switch": True, "confidence": 0.9, "new_context": "towing"}
        return {"has_context_switch": False, "confidence": 0.8, "new_context": None}
    
    def analyze_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if "tow" in text.lower():
            return {"intent": "request_tow", "confidence": 0.9}
        elif "road" in text.lower():
            return {"intent": "request_roadside", "confidence": 0.9}
        else:
            return {"intent": "unknown", "confidence": 0.5}
    
    def process_message(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process message to extract intent and other information"""
        intent_result = self.analyze_intent(text, context)
        
        if intent_result["intent"] == "request_tow":
            flow = "towing"
        elif intent_result["intent"] == "request_roadside":
            flow = "roadside"
        else:
            flow = "unknown"
            
        result = {
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "flow": flow,
            "needs_clarification": intent_result["intent"] == "unknown",
            "entities": []
        }
        
        return result

class TestMistralEnhancer(unittest.TestCase):
    def test_mistral_not_available(self):
        """Test MistralEnhancer when API key is not provided"""
        # Make sure no key is in the environment
        with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}, clear=True):
            enhancer = MistralEnhancer(api_key=None)
            
            # Check availability
            self.assertFalse(enhancer.is_available())
            
            # Test intent analysis without API key
            result = enhancer.analyze_intent("I need a tow truck")
            self.assertEqual(result["intent"], "unknown")
            self.assertEqual(result["confidence"], 0.0)
            
            # Test negation detection without API key
            result = enhancer.detect_negation("No, I don't want that")
            self.assertFalse(result["is_negation"])
            self.assertEqual(result["confidence"], 0.0)

    @patch('requests.post')
    def test_mistral_mocked(self, mock_post):
        """Test MistralEnhancer with mocked API responses"""
        # Configure mock responses
        mock_intent_response = MagicMock()
        mock_intent_response.json.return_value = {
            "choices": [{"message": {"content": "request_tow"}}]
        }
        mock_intent_response.status_code = 200
        
        mock_negation_response = MagicMock()
        mock_negation_response.json.return_value = {
            "choices": [{"message": {"content": "Yes"}}]
        }
        mock_negation_response.status_code = 200
        
        mock_context_switch_response = MagicMock()
        mock_context_switch_response.json.return_value = {
            "choices": [{"message": {"content": "Yes, towing service"}}]
        }
        mock_context_switch_response.status_code = 200
        
        # Set up the side effect for the post method
        mock_post.side_effect = [
            mock_intent_response,
            mock_negation_response,
            mock_context_switch_response
        ]
        
        # Create enhancer with a fake API key
        enhancer = MistralEnhancer(api_key="fake_key")
        
        # Test availability
        self.assertTrue(enhancer.is_available())
        
        # Test intent analysis with mocked response
        result = enhancer.analyze_intent("I need a tow truck")
        self.assertEqual(result["intent"], "request_tow")
        self.assertGreater(result["confidence"], 0.0)
        
        # Test negation detection with mocked response
        result = enhancer.detect_negation("No, I don't want that")
        self.assertTrue(result["is_negation"])
        self.assertGreater(result["confidence"], 0.0)
        
        # Test context switch detection with mocked response
        result = enhancer.detect_context_switch("Actually, I need a tow truck instead")
        self.assertTrue(result["has_context_switch"])
        self.assertEqual(result["new_context"], "towing")
        self.assertGreater(result["confidence"], 0.0)

class TestHybridDetectionSystem(unittest.TestCase):
    def test_hybrid_negation_detection_rule_only(self):
        """Test hybrid negation detection using only rule-based methods"""
        # Set up mock objects
        flags = MockFeatureFlags(enable_mistral=False)
        detector = MockExistingDetector()
        enhancer = MistralEnhancer(api_key=None)  # No API key
        
        # Create hybrid system
        hybrid = HybridDetectionSystem(flags, detector, enhancer)
        
        # Test negation detection
        result = hybrid.detect_negation("I don't want that")
        self.assertTrue(result["is_negation"])
        self.assertEqual(result["confidence"], 0.9)  # Should match mock detector
    
    @patch('requests.post')
    def test_hybrid_negation_detection_with_mistral(self, mock_post):
        """Test hybrid negation detection using both rule-based and ML methods"""
        # Configure mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Yes"}}]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Set up mock objects
        flags = MockFeatureFlags(enable_mistral=True)
        detector = MockExistingDetector()
        enhancer = MistralEnhancer(api_key="fake_key")
        
        # Create hybrid system
        hybrid = HybridDetectionSystem(flags, detector, enhancer)
        
        # Test negation detection - agreement scenario
        result = hybrid.detect_negation("I don't want that")
        self.assertTrue(result["is_negation"])
        self.assertGreater(result["confidence"], 0.8)  # Confidence should be high
        
        # Test negation detection - disagreement scenario
        # Mock detector says no negation, Mistral says yes
        mock_post.return_value = mock_response  # Still says "Yes"
        result = hybrid.detect_negation("Maybe later")  # No obvious negation words
        
        # Default weighting would favor rule-based (70%), so result should match mock detector
        self.assertFalse(result["is_negation"])
    
    @patch('requests.post')
    def test_hybrid_context_switch_detection(self, mock_post):
        """Test hybrid context switch detection"""
        # Configure mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "roadside assistance"}}]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Set up mock objects
        flags = MockFeatureFlags(enable_mistral=True)
        detector = MockExistingDetector()
        enhancer = MistralEnhancer(api_key="fake_key")
        
        # Create hybrid system
        hybrid = HybridDetectionSystem(flags, detector, enhancer)
        
        # Test context switch detection - rule-based says true, ML says true but different context
        result = hybrid.detect_context_switch("Actually, I need help")
        self.assertTrue(result["has_context_switch"])
        
        # With default weights, rule-based should take precedence for the context
        self.assertEqual(result["new_context"], "towing")
    
    @patch('requests.post')
    def test_hybrid_intent_analysis(self, mock_post):
        """Test hybrid intent analysis"""
        # Configure mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "request_roadside"}}]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Set up mock objects
        flags = MockFeatureFlags(enable_mistral=True)
        detector = MockExistingDetector()
        enhancer = MistralEnhancer(api_key="fake_key")
        
        # Create hybrid system
        hybrid = HybridDetectionSystem(flags, detector, enhancer)
        
        # Test intent analysis - disagreement scenario (rule says tow, ML says roadside)
        result = hybrid.analyze_intent("I need a tow", flow="unknown")
        
        # With default weights, rule-based (70%) should win over ML (30%)
        self.assertEqual(result["intent"], "request_tow")
        
        # Now test when rule-based has low confidence
        # Create a special detector with low confidence
        class LowConfidenceDetector(MockExistingDetector):
            def analyze_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"intent": "request_tow", "confidence": 0.4}
                
            def process_message(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                """Override to return low confidence result"""
                return {
                    "intent": "request_tow",
                    "confidence": 0.4,
                    "flow": "towing",
                    "needs_clarification": False,
                    "entities": []
                }
        
        detector = LowConfidenceDetector()
        hybrid = HybridDetectionSystem(flags, detector, enhancer)
        
        # Set weights to favor ML over rule-based since rule has low confidence
        hybrid.set_weights(0.3)  # 30% rule-based, 70% ML-based
        
        # Now ML should have more influence
        result = hybrid.analyze_intent("I need a tow", flow="unknown")
        self.assertEqual(result["intent"], "request_roadside")

if __name__ == '__main__':
    unittest.main() 