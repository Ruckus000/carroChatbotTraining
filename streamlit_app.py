import streamlit as st
import logging
import os
import torch
from typing import Tuple, List, Dict, Any
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path settings - use the actual trained models
MODEL_DIR = "./output/models"
CONFIDENCE_THRESHOLD = 0.6

# Cache model loading to improve performance
@st.cache_resource(show_spinner=True)
def load_actual_models(model_dir: str):
    """
    Load all trained models from the model directory.
    """
    models = {}
    
    try:
        # Load flow classifier
        flow_path = os.path.join(model_dir, "flow_classifier")
        if os.path.exists(flow_path):
            st.sidebar.text("Loading flow classifier...")
            models["flow_tokenizer"] = DistilBertTokenizer.from_pretrained(flow_path)
            models["flow_model"] = DistilBertForSequenceClassification.from_pretrained(flow_path)
            flow_labels = models["flow_model"].config.id2label
            models["flow_labels"] = [flow_labels[i] for i in range(len(flow_labels))]
            st.sidebar.success(f"Flow classifier loaded with {len(models['flow_labels'])} labels")
        else:
            st.sidebar.warning(f"Flow classifier not found at {flow_path}")
            models["flow_model"] = None
            models["flow_tokenizer"] = None
            models["flow_labels"] = ["towing", "roadside", "appointment", "clarification", "fallback"]
            
        # Load fallback and clarification special classifiers
        for special in ["fallback", "clarification"]:
            special_path = os.path.join(model_dir, f"{special}_classifier")
            if os.path.exists(special_path):
                st.sidebar.text(f"Loading {special} classifier...")
                models[f"{special}_tokenizer"] = DistilBertTokenizer.from_pretrained(special_path)
                models[f"{special}_model"] = DistilBertForSequenceClassification.from_pretrained(special_path)
                st.sidebar.success(f"{special.capitalize()} classifier loaded")
            else:
                st.sidebar.warning(f"{special.capitalize()} classifier not found at {special_path}")
                models[f"{special}_model"] = None
                models[f"{special}_tokenizer"] = None
        
        # Load intent classifiers for each flow
        models["intent_models"] = {}
        models["intent_tokenizers"] = {}
        models["intent_labels"] = {}
        
        for flow in models["flow_labels"]:
            flow_path = os.path.join(model_dir, f"{flow}_intent_classifier")
            if os.path.exists(flow_path):
                st.sidebar.text(f"Loading {flow} intent classifier...")
                models["intent_tokenizers"][flow] = DistilBertTokenizer.from_pretrained(flow_path)
                models["intent_models"][flow] = DistilBertForSequenceClassification.from_pretrained(flow_path)
                intent_labels = models["intent_models"][flow].config.id2label
                models["intent_labels"][flow] = [intent_labels[i] for i in range(len(intent_labels))]
                st.sidebar.success(f"{flow.capitalize()} intent classifier loaded with {len(models['intent_labels'][flow])} intents")
            else:
                st.sidebar.warning(f"{flow} intent classifier not found at {flow_path}")
                models["intent_models"][flow] = None
                models["intent_tokenizers"][flow] = None
                
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
    
    return models

class ChatbotPipeline:
    """A pipeline for running inference with the trained models."""
    
    def __init__(self, models: dict):
        self.models = models
        
    def predict_flow(self, text: str) -> Tuple[str, float]:
        """
        Predict the conversation flow for input text using the trained model.
        Falls back to rule-based logic if model not available.
        """
        if self.models["flow_model"] is None or self.models["flow_tokenizer"] is None:
            # Fallback to rule-based logic
            text_lower = text.lower()
            if 'tow' in text_lower or 'broke down' in text_lower:
                return 'towing', 0.92
            elif 'flat tire' in text_lower or 'battery' in text_lower or 'keys' in text_lower:
                return 'roadside', 0.89
            elif 'appointment' in text_lower or 'schedule' in text_lower:
                return 'appointment', 0.85
            elif len(text_lower.split()) < 4:
                return 'clarification', 0.75
            elif 'weather' in text_lower or 'pizza' in text_lower:
                return 'fallback', 0.95
            else:
                return 'clarification', 0.65
        
        # Use the trained model
        try:
            inputs = self.models["flow_tokenizer"](
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.models["flow_model"](**inputs)
                
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            
            # Get highest probability class
            max_prob, max_idx = torch.max(probabilities, dim=-1)
            predicted_flow = self.models["flow_labels"][max_idx.item()]
            confidence = max_prob.item()
            
            return predicted_flow, confidence
            
        except Exception as e:
            logger.error(f"Error in flow prediction: {str(e)}", exc_info=True)
            return "clarification", 0.5
    
    def predict_intent(self, text: str, flow: str) -> Tuple[str, float]:
        """
        Predict the intent for input text within a given flow.
        Uses the trained model if available, otherwise falls back to rules.
        """
        if (flow not in self.models["intent_models"] or 
            self.models["intent_models"][flow] is None or 
            self.models["intent_tokenizers"][flow] is None):
            
            # Fallback to rule-based logic
            text_lower = text.lower()
            if flow == 'towing':
                if 'location' in text_lower or ('from' in text_lower and 'to' in text_lower):
                    return 'request_tow_location', 0.88
                elif any(make in text_lower for make in ['honda', 'toyota', 'ford']):
                    return 'request_tow_vehicle', 0.85
                elif 'asap' in text_lower or 'urgent' in text_lower:
                    return 'request_tow_urgent', 0.92
                else:
                    return 'request_tow_basic', 0.78
            elif flow == 'roadside':
                if 'battery' in text_lower or 'jump' in text_lower:
                    return 'request_roadside_battery', 0.91
                elif 'tire' in text_lower or 'flat' in text_lower:
                    return 'request_roadside_tire', 0.93
                elif 'key' in text_lower or 'lock' in text_lower:
                    return 'request_roadside_keys', 0.89
                else:
                    return 'request_roadside_basic', 0.75
            elif flow == 'appointment':
                if 'oil' in text_lower:
                    return 'book_service_type', 0.90
                elif any(day in text_lower for day in ['monday', 'tuesday', 'wednesday']):
                    return 'book_service_date', 0.86
                else:
                    return 'book_service_basic', 0.80
            else:
                return 'unknown_intent', 0.50
        
        # Use the trained model
        try:
            inputs = self.models["intent_tokenizers"][flow](
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.models["intent_models"][flow](**inputs)
                
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            
            # Get highest probability class
            max_prob, max_idx = torch.max(probabilities, dim=-1)
            predicted_intent = self.models["intent_labels"][flow][max_idx.item()]
            confidence = max_prob.item()
            
            return predicted_intent, confidence
            
        except Exception as e:
            logger.error(f"Error in intent prediction: {str(e)}", exc_info=True)
            return "unknown_intent", 0.5
    
    def is_fallback(self, text: str) -> Tuple[bool, float]:
        """
        Check if message requires fallback handling.
        """
        if self.models["fallback_model"] is None:
            text_lower = text.lower()
            if any(word in text_lower for word in ['weather', 'pizza', 'movie']):
                return True, 0.95
            return False, 0.8
            
        try:
            inputs = self.models["fallback_tokenizer"](
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.models["fallback_model"](**inputs)
                
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            
            # Fallback is typically label 1, not_fallback is 0
            fallback_prob = probabilities[0, 1].item()
            
            return fallback_prob > CONFIDENCE_THRESHOLD, fallback_prob
            
        except Exception as e:
            logger.error(f"Error in fallback detection: {str(e)}", exc_info=True)
            return False, 0.5

# Session management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm your car maintenance assistant. How can I help you today?"}
    ]

if "flow" not in st.session_state:
    st.session_state.flow = None

# Load the trained models
models = load_actual_models(MODEL_DIR)
pipeline = ChatbotPipeline(models)

# Streamlit UI
st.title("Car Maintenance Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Debug panel in the sidebar
with st.sidebar:
    st.subheader("Debug Information")
    if st.session_state.flow:
        st.write(f"Current flow: {st.session_state.flow}")
    
    debug_section = st.checkbox("Show debug information")

# Input area
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Process the message
    with st.spinner("Thinking..."):
        # Check for fallback first
        is_fallback, fallback_conf = pipeline.is_fallback(user_input)
        
        if debug_section:
            with st.sidebar:
                st.write(f"Fallback check: {is_fallback} ({fallback_conf:.2f})")
        
        if is_fallback and fallback_conf > CONFIDENCE_THRESHOLD:
            response = "I'm sorry, I can't help with that. I'm designed to assist with car maintenance, towing, and roadside assistance."
            st.session_state.flow = "fallback"
        else:
            # Determine the flow
            predicted_flow, flow_conf = pipeline.predict_flow(user_input)
            
            if debug_section:
                with st.sidebar:
                    st.write(f"Flow prediction: {predicted_flow} ({flow_conf:.2f})")
            
            # Get the intent
            predicted_intent, intent_conf = pipeline.predict_intent(user_input, predicted_flow)
            
            if debug_section:
                with st.sidebar:
                    st.write(f"Intent prediction: {predicted_intent} ({intent_conf:.2f})")
            
            st.session_state.flow = predicted_flow
            
            # Generate response based on flow and intent
            if predicted_flow == "towing":
                if "location" in predicted_intent:
                    response = "I'll arrange for a tow truck to your location. Can you confirm your current address and destination?"
                elif "vehicle" in predicted_intent:
                    response = "To arrange the right tow truck, could you tell me the make, model, and year of your vehicle?"
                elif "urgent" in predicted_intent:
                    response = "I understand this is urgent. I'm prioritizing your request and will dispatch a tow truck as soon as possible."
                else:
                    response = "I can help arrange a tow truck. Could you please provide your location and vehicle details?"
            
            elif predicted_flow == "roadside":
                if "battery" in predicted_intent:
                    response = "I'll send someone to jump-start your battery. What's your current location?"
                elif "tire" in predicted_intent:
                    response = "I'll send a technician to help with your tire. Are you in a safe location?"
                elif "keys" in predicted_intent:
                    response = "I can send a locksmith to help you get back into your vehicle. Where are you located?"
                elif "fuel" in predicted_intent:
                    response = "I'll arrange for fuel delivery. How much fuel do you need and what's your location?"
                else:
                    response = "I'm here to help with your roadside assistance. Could you provide more details about what you need?"
            
            elif predicted_flow == "appointment":
                if "type" in predicted_intent:
                    response = "I can schedule that service for you. What day works best for you?"
                elif "date" in predicted_intent:
                    response = "We have several time slots available on that day. Would you prefer morning or afternoon?"
                elif "time" in predicted_intent:
                    response = "Great! I've scheduled your appointment. Is there anything else you'd like to know about your service?"
                else:
                    response = "I'd be happy to schedule a service appointment for you. What type of service do you need?"
            
            elif predicted_flow == "clarification":
                response = "I'd like to help you better. Could you provide more details about what you need assistance with?"
            
            else:
                response = "I'm here to help with towing, roadside assistance, or scheduling service appointments. What can I do for you today?"
                
    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Display note about models at the bottom
st.caption("This chatbot is using trained DistilBERT models for intent classification and flow detection.")