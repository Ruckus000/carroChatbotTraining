import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Create directory mocking for tests that might not have trained models
os.makedirs("./trained_nlu_model/intent_model", exist_ok=True)
os.makedirs("./trained_nlu_model/entity_model", exist_ok=True)

# Mock data for intent model
intent2id = {
    "towing_request_tow": 0,
    "roadside_request_battery": 1,
    "appointment_book_service": 2,
    "fallback_out_of_scope": 3,
    "clarification_ambiguous_request": 4,
}

# Mock data for entity model
tag2id = {
    "O": 0,
    "B-pickup_location": 1,
    "I-pickup_location": 2,
    "B-vehicle_make": 3,
    "I-vehicle_make": 4,
    "B-vehicle_model": 5,
    "I-vehicle_model": 6,
}

# Create mock files if they don't exist
if not os.path.exists("./trained_nlu_model/intent_model/intent2id.json"):
    with open("./trained_nlu_model/intent_model/intent2id.json", "w") as f:
        json.dump(intent2id, f)

if not os.path.exists("./trained_nlu_model/entity_model/tag2id.json"):
    with open("./trained_nlu_model/entity_model/tag2id.json", "w") as f:
        json.dump(tag2id, f)


# Mock the transformers module
class MockModule:
    pass


class MockModelForSequenceClassification:
    def __init__(self, *args, **kwargs):
        self.outputs = MockModule()
        self.outputs.logits = torch.tensor([[2.0, 1.0, 0.5, 0.2, 0.1]])

    def __call__(self, **kwargs):
        return self.outputs

    def to(self, device):
        return self

    def eval(self):
        pass


class MockModelForTokenClassification:
    def __init__(self, *args, **kwargs):
        self.outputs = MockModule()
        # Create sample logits for entity detection
        # We'll make a simple pattern where every third token is a B-pickup_location
        # and every third+1 token is I-pickup_location
        sample_logits = []
        for i in range(20):  # Reasonable sequence length
            if i % 3 == 1:
                # Make B-pickup_location the highest logit
                token_logits = [0.0] * len(tag2id)
                token_logits[1] = 5.0  # B-pickup_location
                sample_logits.append(token_logits)
            elif i % 3 == 2:
                # Make I-pickup_location the highest logit
                token_logits = [0.0] * len(tag2id)
                token_logits[2] = 5.0  # I-pickup_location
                sample_logits.append(token_logits)
            else:
                # Make O the highest logit
                token_logits = [0.0] * len(tag2id)
                token_logits[0] = 5.0  # O
                sample_logits.append(token_logits)

        self.outputs.logits = torch.tensor([sample_logits])

    def __call__(self, **kwargs):
        return self.outputs

    def to(self, device):
        return self

    def eval(self):
        pass


class MockTokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(
        self,
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        is_split_into_words=False,
    ):
        result = MockModule()
        result.input_ids = torch.tensor([[101, 2054, 2003, 1037, 2943, 4178, 102]])
        result.attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]])

        # For the word_ids method
        result._word_ids = {}

        # Create word mapping for tokens
        if isinstance(text, str):
            words = text.split()
            # Create a simple mapping where each word corresponds to a token
            # (this is a simplification, in reality, tokenizers can split words)
            result._word_ids[0] = [None] + [i for i in range(len(words))] + [None]

        return result

    def word_ids(self, batch_index=0):
        return self._word_ids.get(batch_index, [None, 0, 0, 1, 1, 2, None])


# Apply the mocks
patch(
    "transformers.DistilBertForSequenceClassification.from_pretrained",
    return_value=MockModelForSequenceClassification(),
).start()
patch(
    "transformers.DistilBertForTokenClassification.from_pretrained",
    return_value=MockModelForTokenClassification(),
).start()
patch(
    "transformers.DistilBertTokenizer.from_pretrained", return_value=MockTokenizer()
).start()

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
        with patch.object(
            inferencer, "_predict_intent", side_effect=Exception("Test error")
        ):
            result = inferencer.predict("This should cause an error")

            # Should return a fallback response
            self.assertEqual(result["intent"]["name"], "fallback_runtime_error")
            self.assertEqual(result["intent"]["confidence"], 1.0)
            self.assertEqual(result["entities"], [])


if __name__ == "__main__":
    unittest.main()
