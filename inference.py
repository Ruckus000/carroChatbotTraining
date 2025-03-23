import os
import json
import logging
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertForTokenClassification

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Force CPU
device = torch.device("cpu")

class CarroAssistant:
    def __init__(self, models_dir):
        """
        Initialize the assistant with the trained models.
        
        Args:
            models_dir: Directory containing the trained models
        """
        self.models_dir = models_dir
        
        # Load intent classifier
        intent_model_path = os.path.join(models_dir, "intent_classifier")
        self.intent_tokenizer = DistilBertTokenizer.from_pretrained(intent_model_path)
        self.intent_model = DistilBertForSequenceClassification.from_pretrained(intent_model_path)
        self.intent_model.to(device)
        self.intent_model.eval()
        
        # Load intent labels
        with open(os.path.join(intent_model_path, "intent_labels.json"), 'r') as f:
            self.intent_labels = json.load(f)
        
        # Load fallback detector
        fallback_model_path = os.path.join(models_dir, "fallback_detector")
        self.fallback_tokenizer = DistilBertTokenizer.from_pretrained(fallback_model_path)
        self.fallback_model = DistilBertForSequenceClassification.from_pretrained(fallback_model_path)
        self.fallback_model.to(device)
        self.fallback_model.eval()
        
        # Load clarification detector
        clarification_model_path = os.path.join(models_dir, "clarification_detector")
        self.clarification_tokenizer = DistilBertTokenizer.from_pretrained(clarification_model_path)
        self.clarification_model = DistilBertForSequenceClassification.from_pretrained(clarification_model_path)
        self.clarification_model.to(device)
        self.clarification_model.eval()
        
        # Load entity extractor
        entity_model_path = os.path.join(models_dir, "entity_extractor")
        self.entity_tokenizer = DistilBertTokenizer.from_pretrained(entity_model_path)
        self.entity_model = DistilBertForTokenClassification.from_pretrained(entity_model_path)
        self.entity_model.to(device)
        self.entity_model.eval()
        
        # Load entity labels
        with open(os.path.join(entity_model_path, "entity_labels.json"), 'r') as f:
            self.entity_labels = json.load(f)

    def predict_intent(self, text):
        """Predict the intent of the input text."""
        inputs = self.intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
        
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        return self.intent_labels[predicted_class]
    
    def needs_fallback(self, text):
        """Determine if the input requires a fallback response."""
        inputs = self.fallback_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.fallback_model(**inputs)
        
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        return predicted_class == 1  # Assuming 1 is the "needs fallback" class
    
    def needs_clarification(self, text):
        """Determine if the input requires clarification."""
        inputs = self.clarification_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clarification_model(**inputs)
        
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        
        return predicted_class == 1  # Assuming 1 is the "needs clarification" class
    
    def extract_entities(self, text):
        """Extract entities from the input text."""
        # Tokenize input
        tokens = text.split()
        tokenized_inputs = self.entity_tokenizer(
            tokens, 
            is_split_into_words=True, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        )
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
        
        # Get subword mappings
        word_ids = tokenized_inputs.word_ids()
        
        # Make prediction
        with torch.no_grad():
            outputs = self.entity_model(**tokenized_inputs)
        
        # Get predictions for each token
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Map predictions back to original tokens
        entity_predictions = []
        prev_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != prev_word_idx:
                label_id = predictions[idx]
                entity_label = self.entity_labels[str(label_id)]
                entity_predictions.append(entity_label)
                prev_word_idx = word_idx
        
        # Extract named entities
        entities = {}
        current_entity = None
        current_tokens = []
        
        for token, tag in zip(tokens, entity_predictions):
            if tag.startswith("B-"):
                if current_entity:
                    entity_text = " ".join(current_tokens)
                    if current_entity not in entities:
                        entities[current_entity] = []
                    entities[current_entity].append(entity_text)
                
                current_entity = tag[2:]  # Remove B- prefix
                current_tokens = [token]
            
            elif tag.startswith("I-") and current_entity == tag[2:]:
                current_tokens.append(token)
            
            elif tag == "O":
                if current_entity:
                    entity_text = " ".join(current_tokens)
                    if current_entity not in entities:
                        entities[current_entity] = []
                    entities[current_entity].append(entity_text)
                    current_entity = None
                    current_tokens = []
        
        # Handle the last entity if there is one
        if current_entity:
            entity_text = " ".join(current_tokens)
            if current_entity not in entities:
                entities[current_entity] = []
            entities[current_entity].append(entity_text)
        
        return entities
    
    def process_message(self, text):
        """
        Process an incoming message and return the appropriate response information.
        
        Args:
            text: User input text
            
        Returns:
            Dict containing:
                - intent: Predicted intent
                - entities: Extracted entities
                - needs_fallback: Whether a fallback is needed
                - needs_clarification: Whether clarification is needed
        """
        # Check if we need a fallback or clarification first
        fallback = self.needs_fallback(text)
        clarification = self.needs_clarification(text)
        
        # Get intent and entities (if no fallback needed)
        if not fallback:
            intent = self.predict_intent(text)
            entities = self.extract_entities(text)
        else:
            intent = "unknown"
            entities = {}
        
        return {
            "intent": intent,
            "entities": entities,
            "needs_fallback": fallback,
            "needs_clarification": clarification
        } 