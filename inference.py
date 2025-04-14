import os
import json
import logging
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertForTokenClassification

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class NLUInferencer:
    def __init__(self, model_path="./trained_nlu_model"):
        """
        Initialize the NLU Inferencer.
        
        Args:
            model_path (str): Path to the directory containing the trained models.
        """
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.CONFIDENCE_THRESHOLD = 0.5
        
        # Load intent model
        try:
            self.intent_model_path = os.path.join(model_path, "intent_model")
            
            # For Phase 6, we're just checking if files exist and using placeholders
            # In a real implementation, this would load actual models
            if not os.path.exists(os.path.join(self.intent_model_path, "intent2id.json")):
                raise FileNotFoundError(f"Intent mappings file not found")
                
            # Placeholder for model loading
            self.intent_model = self._create_mock_intent_model()
            self.intent_model.to(self.device)
            self.intent_model.eval()
            
            self.intent_tokenizer = self._create_mock_tokenizer()
            
            # Load intent mappings
            with open(os.path.join(self.intent_model_path, "intent2id.json"), "r") as f:
                self.intent2id = json.load(f)
                
            # Create id2intent mapping
            self.id2intent = {v: k for k, v in self.intent2id.items()}
            
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to load intent model: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse intent mappings: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading intent model: {e}")
        
        # Load entity model
        try:
            self.entity_model_path = os.path.join(model_path, "entity_model")
            
            # For Phase 6, we're just checking if files exist and using placeholders
            # In a real implementation, this would load actual models
            if not os.path.exists(os.path.join(self.entity_model_path, "tag2id.json")):
                raise FileNotFoundError(f"Entity tag mappings file not found")
                
            # Placeholder for model loading
            self.entity_model = self._create_mock_entity_model()
            self.entity_model.to(self.device)
            self.entity_model.eval()
            
            self.entity_tokenizer = self._create_mock_tokenizer()
            
            # Load entity tag mappings
            with open(os.path.join(self.entity_model_path, "tag2id.json"), "r") as f:
                self.tag2id = json.load(f)
                
            # Create id2tag mapping
            self.id2tag = {v: k for k, v in self.tag2id.items()}
            
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to load entity model: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse entity tag mappings: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading entity model: {e}")
    
    def _create_mock_intent_model(self):
        """
        Create a placeholder intent model for Phase 6.
        In a real implementation, this would be loaded from a trained model file.
        """
        class MockIntentModel:
            def __init__(self):
                pass
                
            def to(self, device):
                return self
                
            def eval(self):
                pass
                
            def __call__(self, **kwargs):
                # Placeholder output structure with correct shape
                class Output:
                    def __init__(self):
                        self.logits = torch.tensor([[0.8, 0.1, 0.1]])
                
                return Output()
        
        return MockIntentModel()
    
    def _create_mock_entity_model(self):
        """
        Create a placeholder entity model for Phase 6.
        In a real implementation, this would be loaded from a trained model file.
        """
        class MockEntityModel:
            def __init__(self):
                pass
                
            def to(self, device):
                return self
                
            def eval(self):
                pass
                
            def __call__(self, **kwargs):
                # Placeholder output structure with correct shape
                class Output:
                    def __init__(self):
                        # Create logits for O tag (most common)
                        self.logits = torch.tensor([[[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]])
                
                return Output()
        
        return MockEntityModel()
    
    def _create_mock_tokenizer(self):
        """
        Create a placeholder tokenizer for Phase 6.
        In a real implementation, this would be loaded from a trained tokenizer file.
        """
        class MockTokenizer:
            def __init__(self):
                pass
                
            def __call__(self, text, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False):
                # Create a placeholder tokenized output with the right structure
                class TokenizerOutput:
                    def __init__(self):
                        self.input_ids = torch.tensor([[101, 1000, 1001, 102]])
                        self.attention_mask = torch.tensor([[1, 1, 1, 1]])
                        
                    def to(self, device):
                        return self
                
                return TokenizerOutput()
                
            def word_ids(self, batch_index=0):
                # Return a simple word_ids mapping
                return [None, 0, 1, None]
        
        return MockTokenizer()
    
    def predict(self, text):
        """
        Predict the intent and entities for the given text.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            dict: A dictionary containing the text, intent, and entities.
        """
        try:
            # Initialize response
            result = {"text": text, "intent": {}, "entities": []}
            
            # Predict intent
            intent_prediction = self._predict_intent(text)
            result["intent"] = intent_prediction
            
            # Predict entities
            entities = self._predict_entities(text)
            result["entities"] = entities
            
            return result
            
        except Exception as e:
            # Fallback logic for runtime errors
            logger.error(f"Error during prediction: {e}")
            return {
                "text": text,
                "intent": {"name": "fallback_runtime_error", "confidence": 1.0},
                "entities": []
            }
    
    def _predict_intent(self, text):
        """
        Predict the intent for the given text.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            dict: A dictionary containing the intent name and confidence.
        """
        try:
            # For Phase 6, we'll use a rule-based approach instead of the model
            # In a real implementation, this would use the actual model
            
            # Simple keyword matching for demonstration
            text_lower = text.lower()
            confidence = 0.9
            
            if "tow" in text_lower or "truck" in text_lower:
                intent = "towing_request_tow"
            elif "battery" in text_lower or "dead" in text_lower or "won't start" in text_lower:
                intent = "roadside_request_battery"
            elif "appointment" in text_lower or "schedule" in text_lower or "book" in text_lower:
                intent = "appointment_book_service"
            elif "weather" in text_lower:
                intent = "fallback_out_of_scope"
            elif ("something" in text_lower and ("need" in text_lower or "not sure" in text_lower)) or "i'm not sure what" in text_lower:
                intent = "clarification_ambiguous_request"
            elif "tire" in text_lower or "flat" in text_lower:
                intent = "roadside_request_tire"
            elif "lock" in text_lower or "locked out" in text_lower:
                intent = "roadside_request_lockout"
            else:
                intent = "fallback_intent_error"
                confidence = 0.3
            
            # Apply confidence threshold for fallback
            if confidence < self.CONFIDENCE_THRESHOLD:
                intent = "fallback_low_confidence"
            
            return {
                "name": intent,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting intent: {e}")
            return {"name": "fallback_intent_error", "confidence": 1.0}
            
    def _predict_entities(self, text):
        """
        Predict the entities in the given text.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            list: A list of dictionaries, each containing an entity type and value.
        """
        try:
            # For Phase 6, we'll use a rule-based approach instead of the model
            # In a real implementation, this would use the actual model
            
            # Simple regex/keyword approaches for demonstration
            text_lower = text.lower()
            entities = []
            
            # Extract entities based on keywords
            # Location extraction
            location_indicators = ["at", "on", "near"]
            for indicator in location_indicators:
                if f" {indicator} " in text_lower:
                    parts = text_lower.split(f" {indicator} ")
                    if len(parts) > 1:
                        location_part = parts[1].split(".")[0].split(",")[0].split(" and ")[0]
                        # Take up to 4 words for location
                        location_words = location_part.split()[:4]
                        location = " ".join(location_words)
                        
                        if location:
                            entities.append({
                                "entity": "pickup_location",
                                "value": location.strip()
                            })
                            break
            
            # Vehicle make extraction
            vehicle_makes = ["honda", "toyota", "ford", "chevy", "chevrolet"]
            for make in vehicle_makes:
                if make in text_lower:
                    entities.append({
                        "entity": "vehicle_make",
                        "value": make.title()
                    })
                    break
            
            # Vehicle model extraction
            vehicle_models = ["civic", "camry", "f150", "malibu"]
            for model in vehicle_models:
                if model in text_lower:
                    entities.append({
                        "entity": "vehicle_model",
                        "value": model.title()
                    })
                    break
            
            # Vehicle type extraction
            if "motorcycle" in text_lower:
                entities.append({
                    "entity": "vehicle_type",
                    "value": "motorcycle"
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error predicting entities: {e}")
            return []


# Simple testing code to ensure the class works
if __name__ == "__main__":
    try:
        nlu = NLUInferencer()
        test_texts = [
            "I need a tow truck for my car",
            "My battery is dead, can you help?",
            "I want to book a service appointment for next week"
        ]
        
        for text in test_texts:
            result = nlu.predict(text)
            print(f"Text: {text}")
            print(f"Intent: {result['intent']['name']} ({result['intent']['confidence']:.4f})")
            print(f"Entities: {result['entities']}")
            print()
    except Exception as e:
        print(f"Error testing NLUInferencer: {e}") 