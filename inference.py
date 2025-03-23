import os
import json
import logging
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertForTokenClassification
from typing import Dict, List, Any, Tuple, Optional

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

class ContextAwareCarroAssistant(CarroAssistant):
    """
    Extended version of CarroAssistant with context awareness capabilities.
    Maintains backward compatibility with the original assistant while adding
    context handling features.
    """
    
    def __init__(self, models_dir):
        """
        Initialize with both standard and context-aware models.
        
        Args:
            models_dir: Directory containing the trained models
        """
        # Initialize the base class first
        super().__init__(models_dir)
        
        # Version tracking to identify context-aware vs standard models
        self.version = "context_aware_v1"
        
        # Initialize conversation context tracking
        self.conversation_context = {
            "previous_intents": [],        # Track last 3 intents
            "previous_entities": {},       # Track entities from previous turns
            "active_flow": None,           # Current conversation flow
            "turn_count": 0                # Track conversation length
        }
        
        # Try to load context-aware models if available
        self._load_context_models(models_dir)
    
    def _load_context_models(self, models_dir):
        """
        Load context-aware models if available, falling back to standard models if not.
        
        Args:
            models_dir: Directory containing the trained models
        """
        # Initialize flag to track if context-aware models are available
        self.has_context_models = False
        
        try:
            # Check for context-specific models - if not available, will use base models
            context_model_path = os.path.join(models_dir, "context_aware")
            
            if os.path.exists(context_model_path):
                logger.info("Loading context-aware models...")
                
                # Load negation detector if available
                negation_path = os.path.join(context_model_path, "negation_detector")
                if os.path.exists(negation_path):
                    self.negation_tokenizer = DistilBertTokenizer.from_pretrained(negation_path)
                    self.negation_model = DistilBertForSequenceClassification.from_pretrained(negation_path)
                    self.negation_model.to(device)
                    self.negation_model.eval()
                    logger.info("Negation detector loaded successfully")
                    self.has_context_models = True
                else:
                    self.negation_model = None
                    logger.info("Negation detector not found, will use rule-based detection")
                
                # Load context switch detector if available
                context_switch_path = os.path.join(context_model_path, "context_switch_detector")
                if os.path.exists(context_switch_path):
                    self.context_switch_tokenizer = DistilBertTokenizer.from_pretrained(context_switch_path)
                    self.context_switch_model = DistilBertForSequenceClassification.from_pretrained(context_switch_path)
                    self.context_switch_model.to(device)
                    self.context_switch_model.eval()
                    logger.info("Context switch detector loaded successfully")
                    self.has_context_models = True
                else:
                    self.context_switch_model = None
                    logger.info("Context switch detector not found, will use rule-based detection")
            else:
                logger.info("Context-aware models not found, using base models only")
                self.negation_model = None
                self.context_switch_model = None
                
        except Exception as e:
            logger.error(f"Error loading context-aware models: {str(e)}")
            logger.info("Falling back to standard models")
            self.negation_model = None
            self.context_switch_model = None
    
    def detect_negation(self, text) -> Tuple[bool, float]:
        """
        Detect if the text contains negation.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (contains_negation, confidence)
        """
        # Use model-based detection if available
        if self.negation_model is not None:
            inputs = self.negation_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.negation_model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Assuming label 1 is "contains negation"
            contains_negation = torch.argmax(logits, dim=1).item() == 1
            confidence = probabilities[0, 1].item()  # Probability of negation
            
            return contains_negation, confidence
        
        # Fall back to rule-based detection
        else:
            # Simple rule-based detection
            negation_phrases = ["don't", "do not", "doesn't", "does not", "not", "no longer", 
                              "forget", "cancel", "stop", "instead", "rather than"]
            
            text_lower = text.lower()
            for phrase in negation_phrases:
                if phrase in text_lower:
                    # Found a negation phrase
                    return True, 0.8  # Confidence hardcoded for rule-based
            
            return False, 0.1
    
    def detect_context_switch(self, text) -> Tuple[bool, float]:
        """
        Detect if the text indicates a context switch.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (contains_context_switch, confidence)
        """
        # Use model-based detection if available
        if self.context_switch_model is not None:
            inputs = self.context_switch_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.context_switch_model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
            # Assuming label 1 is "contains context switch"
            contains_switch = torch.argmax(logits, dim=1).item() == 1
            confidence = probabilities[0, 1].item()  # Probability of context switch
            
            return contains_switch, confidence
        
        # Fall back to rule-based detection
        else:
            # Simple rule-based detection
            switch_phrases = ["actually", "instead", "rather", "changed my mind", 
                            "but now", "forget about", "never mind", "switch to", "different"]
            
            text_lower = text.lower()
            for phrase in switch_phrases:
                if phrase in text_lower:
                    # Found a context switch phrase
                    return True, 0.8  # Confidence hardcoded for rule-based
            
            return False, 0.1
    
    def update_conversation_context(self, result: Dict[str, Any]) -> None:
        """
        Update the conversation context with the latest processing result.
        
        Args:
            result: Processing result dictionary
        """
        # Increment turn count
        self.conversation_context["turn_count"] += 1
        
        # Store intent in context history (if not unknown)
        if result["intent"] != "unknown":
            self.conversation_context["previous_intents"].append({
                "intent": result["intent"],
                "turn": self.conversation_context["turn_count"],
                "confidence": result.get("intent_confidence", 1.0)
            })
            
            # Keep only the last 3 intents
            if len(self.conversation_context["previous_intents"]) > 3:
                self.conversation_context["previous_intents"].pop(0)
        
        # Store entities in context history
        for entity_type, values in result.get("entities", {}).items():
            if entity_type not in self.conversation_context["previous_entities"]:
                self.conversation_context["previous_entities"][entity_type] = []
            
            # Add each entity value with metadata
            for value in values:
                self.conversation_context["previous_entities"][entity_type].append({
                    "value": value,
                    "turn": self.conversation_context["turn_count"]
                })
        
        # Update active flow if present
        if "flow" in result and result["flow"] not in ["clarification", "fallback"]:
            self.conversation_context["active_flow"] = result["flow"]
    
    def detect_contradictions(self, current_entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Detect contradictions between current entities and previous ones.
        
        Args:
            current_entities: Current entities extracted from the message
            
        Returns:
            List of contradiction dictionaries with entity_type, current_value, and previous_value
        """
        contradictions = []
        
        for entity_type, current_values in current_entities.items():
            # Check if we have this entity type in history
            if entity_type in self.conversation_context["previous_entities"]:
                previous_values = self.conversation_context["previous_entities"][entity_type]
                
                # Only check the most recent value
                if previous_values:
                    most_recent = max(previous_values, key=lambda x: x["turn"])
                    previous_value = most_recent["value"]
                    
                    # Check for contradiction (simple string comparison)
                    for current_value in current_values:
                        if current_value.lower() != previous_value.lower():
                            contradictions.append({
                                "entity_type": entity_type,
                                "current_value": current_value,
                                "previous_value": previous_value,
                                "turn_difference": self.conversation_context["turn_count"] - most_recent["turn"]
                            })
        
        return contradictions
    
    def process_message_with_context(self, text: str) -> Dict[str, Any]:
        """
        Process an incoming message with context awareness.
        This enhanced version considers conversation history and context.
        
        Args:
            text: User input text
            
        Returns:
            Enhanced result dictionary with context information
        """
        # Initialize the result with base processing
        base_result = super().process_message(text)
        
        # Enhanced result with context information
        result = {
            **base_result,
            "contains_negation": False,
            "contains_context_switch": False,
            "contradictions": [],
            "context_used": False,
        }
        
        # Check for negation and context switching
        contains_negation, negation_conf = self.detect_negation(text)
        contains_switch, switch_conf = self.detect_context_switch(text)
        
        result["contains_negation"] = contains_negation
        result["negation_confidence"] = negation_conf
        result["contains_context_switch"] = contains_switch
        result["context_switch_confidence"] = switch_conf
        
        # Check for contradictions with previous entities
        contradictions = self.detect_contradictions(result["entities"])
        result["contradictions"] = contradictions
        
        # Determine what flow we're in (if not already determined)
        if "flow" not in result and self.conversation_context["active_flow"]:
            result["flow"] = self.conversation_context["active_flow"]
            result["context_used"] = True
        
        # Handle negation
        if contains_negation:
            # If we have previous intents, mark the most recent one as negated
            if self.conversation_context["previous_intents"]:
                last_intent = self.conversation_context["previous_intents"][-1]
                result["negated_intent"] = last_intent["intent"]
                
                # If we have an entity of type "service_type", this might be an alternative
                if "service_type" in result["entities"]:
                    result["alternative_requested"] = True
        
        # Update conversation context with this result
        self.update_conversation_context(result)
        
        return result
    
    def process_message(self, text: str, use_context: bool = True) -> Dict[str, Any]:
        """
        Process an incoming message with optional context awareness.
        Provides backward compatibility with the original method.
        
        Args:
            text: User input text
            use_context: Whether to use context awareness
            
        Returns:
            Processing result dictionary
        """
        if use_context:
            return self.process_message_with_context(text)
        else:
            # Use the non-context-aware processing from the parent class
            return super().process_message(text) 