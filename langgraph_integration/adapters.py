from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Import your existing assistant
from inference import ContextAwareCarroAssistant

class ModelAdapter(ABC):
    """Base adapter for model integration"""
    
    @abstractmethod
    def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a prediction using the model"""
        pass

class ExistingModelAdapter(ModelAdapter):
    """Adapter for existing DistilBERT models"""
    
    def __init__(self, model_path: str, model_type: str = "intent"):
        self.model_path = model_path
        self.model_type = model_type
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the existing model"""
        try:
            if os.path.exists(self.model_path):
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
                self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
                self.model.eval()
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
    
    def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a prediction using the existing model"""
        if self.model is None or self.tokenizer is None:
            # Fallback to rule-based defaults if model isn't loaded
            return {"intent": "unknown", "confidence": 0.0}
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get predicted class
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Calculate confidence
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence = probabilities[0][predicted_class].item()
        
        # Map to label
        predicted_label = self.model.config.id2label.get(str(predicted_class), "unknown")
        
        return {
            "intent": predicted_label,
            "confidence": confidence
        }

class ExistingDetectionAdapter:
    """Adapter for existing detection methods"""
    
    def __init__(self):
        self.assistant = ContextAwareCarroAssistant()
    
    def detect_negation(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use existing negation detection"""
        return self.assistant.detect_negation(text)
    
    def detect_context_switch(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use existing context switch detection"""
        return self.assistant.detect_context_switch(text)
    
    def detect_contradictions(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use existing contradiction detection"""
        # If the method exists in the assistant, use it; otherwise return a default response
        if hasattr(self.assistant, 'detect_contradictions'):
            return self.assistant.detect_contradictions(text, context)
        return {"has_contradiction": False, "confidence": 0.0}
    
    def process_message(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process message using existing assistant"""
        if context:
            return self.assistant.process_message_with_context(text, context)
        else:
            return self.assistant.process_message(text) 