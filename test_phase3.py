import os
import json
import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

# Mock classes for testing
class MockModule:
    def __init__(self):
        self.logits = torch.tensor([[0.8, 0.1, 0.1]])
        
    def cpu(self):
        return self
        
    def to(self, device):
        return self

class MockModelForSequenceClassification:
    def __init__(self):
        self.outputs = MockModule()
    
    def __call__(self, **kwargs):
        return self.outputs
    
    def to(self, device):
        return self
    
    def eval(self):
        pass

class MockModelForTokenClassification:
    def __init__(self):
        self.outputs = MockModule()
        # Add a second dimension for token-level predictions
        self.outputs.logits = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]])
    
    def __call__(self, **kwargs):
        return self.outputs
    
    def to(self, device):
        return self
    
    def eval(self):
        pass

class MockTokenizer:
    def __init__(self):
        pass
    
    def __call__(self, text, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False):
        result = MockModule()
        result.input_ids = torch.tensor([[101, 1000, 1001, 102]])  # CLS, two tokens, SEP
        result.attention_mask = torch.tensor([[1, 1, 1, 1]])
        return result
    
    def word_ids(self, batch_index=0):
        # Return a simple word_ids mapping
        return [None, 0, 1, None]  # CLS, two word ids, SEP

# Apply the mocks
patch('transformers.DistilBertForSequenceClassification.from_pretrained', 
      return_value=MockModelForSequenceClassification()).start()
patch('transformers.DistilBertForTokenClassification.from_pretrained', 
      return_value=MockModelForTokenClassification()).start()
patch('transformers.DistilBertTokenizer.from_pretrained', 
      return_value=MockTokenizer()).start()

# Now import the NLUInferencer after mocking
from inference import NLUInferencer

class TestPhase3(unittest.TestCase):
    def test_init(self):
        """Test if NLUInferencer initializes properly."""
        inferencer = NLUInferencer()
        self.assertIsNotNone(inferencer.intent_model)
        self.assertIsNotNone(inferencer.entity_model)
        self.assertIsNotNone(inferencer.intent_tokenizer)
        self.assertIsNotNone(inferencer.entity_tokenizer)
        self.assertIsNotNone(inferencer.intent2id)
        self.assertIsNotNone(inferencer.id2intent)
        self.assertIsNotNone(inferencer.tag2id)
        self.assertIsNotNone(inferencer.id2tag)
    
    def test_predict_intent(self):
        """Test intent prediction."""
        inferencer = NLUInferencer()
        result = inferencer._predict_intent("I need a tow truck")
        
        self.assertIn("name", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["name"], str)
        self.assertIsInstance(result["confidence"], float)
        self.assertTrue(0 <= result["confidence"] <= 1)
    
    def test_predict_entities(self):
        """Test entity prediction."""
        inferencer = NLUInferencer()
        entities = inferencer._predict_entities("I need a tow truck at Main Street")
        
        self.assertIsInstance(entities, list)
        # Our mock will return entities
        if entities:
            for entity in entities:
                self.assertIn("entity", entity)
                self.assertIn("value", entity)
                self.assertIsInstance(entity["entity"], str)
                self.assertIsInstance(entity["value"], str)
    
    def test_predict_full(self):
        """Test the complete prediction pipeline."""
        inferencer = NLUInferencer()
        result = inferencer.predict("I need a tow truck at Main Street")
        
        self.assertIn("text", result)
        self.assertIn("intent", result)
        self.assertIn("entities", result)
        
        self.assertIn("name", result["intent"])
        self.assertIn("confidence", result["intent"])
        
        self.assertEqual(result["text"], "I need a tow truck at Main Street")
    
    def test_error_handling(self):
        """Test error handling in prediction."""
        inferencer = NLUInferencer()
        
        # Force an error in _predict_intent
        with patch.object(inferencer, '_predict_intent', side_effect=Exception("Test error")):
            result = inferencer.predict("This should cause an error")
            
            # Should return a fallback response
            self.assertEqual(result["intent"]["name"], "fallback_runtime_error")
            self.assertEqual(result["intent"]["confidence"], 1.0)
            self.assertEqual(result["entities"], [])

if __name__ == '__main__':
    unittest.main() 