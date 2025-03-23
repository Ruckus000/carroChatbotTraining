import streamlit as st
import logging
import os
import torch
from typing import Tuple, List, Dict, Any
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import numpy as np

# Import the regular and context-aware assistants
from inference import CarroAssistant, ContextAwareCarroAssistant

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

# Cache assistant loading to improve performance
@st.cache_resource(show_spinner=True)
def load_context_aware_assistant(model_dir: str):
    """
    Load the context-aware assistant with caching for efficiency.
    """
    try:
        st.sidebar.text("Loading context-aware assistant...")
        assistant = ContextAwareCarroAssistant(model_dir)
        st.sidebar.success("Context-aware assistant loaded")
        return assistant
    except Exception as e:
        st.sidebar.error(f"Error loading context-aware assistant: {str(e)}")
        logger.error(f"Error loading context-aware assistant: {str(e)}", exc_info=True)
        return None

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

def display_enhanced_context_indicators(context: Dict[str, Any]) -> None:
    """
    Enhanced display of context information with better visualization.
    
    Args:
        context: Current conversation context
    """
    st.sidebar.subheader("Conversation Context", help="Current state of the conversation")
    
    # Active Flow with Visual Enhancement
    if context.get("active_flow"):
        flow_icon = {
            "towing": "🚛", "roadside": "🔧",
            "appointment": "📅", "information": "ℹ️"
        }.get(context["active_flow"], "🔄")
        
        st.sidebar.success(
            f"{flow_icon} Active Flow: {context['active_flow'].capitalize()}"
        )
        
        # Show flow confidence if available
        if "confidence_scores" in context and "flow" in context["confidence_scores"]:
            flow_conf = context["confidence_scores"]["flow"]
            st.sidebar.progress(flow_conf, text=f"Flow Confidence: {flow_conf:.2%}")
    
    # Context Switches with Timeline
    if context.get("context_switches"):
        with st.sidebar.expander("🔄 Context Switches", expanded=False):
            for switch in reversed(context["context_switches"]):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"Turn {switch['turn']}")
                with col2:
                    st.info(f"{switch['text']}\n*Confidence: {switch['confidence']:.2%}*")
    
    # Active Entities with Contradiction Highlighting
    if context.get("previous_entities"):
        with st.sidebar.expander("📋 Active Entities", expanded=True):
            for entity_type, values in context["previous_entities"].items():
                if values:
                    most_recent = max(values, key=lambda x: x["turn"])
                    
                    # Check for contradictions
                    is_contradicted = any(
                        c["entity_type"] == entity_type
                        for c in context.get("contradictions", [])
                    )
                    
                    if is_contradicted:
                        # Find the contradiction for this entity
                        contradiction = next(
                            c for c in context["contradictions"]
                            if c["entity_type"] == entity_type
                        )
                        
                        st.warning(
                            f"**{entity_type.replace('_', ' ').title()}**:\n"
                            f"~~{contradiction['previous_value']}~~ → "
                            f"**{contradiction['current_value']}** ⚠️\n"
                            f"*Confidence: {contradiction.get('confidence', 0):.2%}*"
                        )
                    else:
                        st.info(
                            f"**{entity_type.replace('_', ' ').title()}**:\n"
                            f"{most_recent['value']}"
                        )
    
    # Intent History with Confidence
    if context.get("previous_intents"):
        with st.sidebar.expander("🎯 Intent History", expanded=False):
            for i, intent_info in enumerate(reversed(context["previous_intents"])):
                confidence = intent_info.get("confidence", 0)
                st.write(
                    f"{i+1}. {intent_info['intent']} "
                    f"(Turn {intent_info['turn']}, "
                    f"Confidence: {confidence:.2%})"
                )
    
    # Overall Context Health
    if "confidence_scores" in context:
        with st.sidebar.expander("📊 Context Health", expanded=False):
            scores = context["confidence_scores"]
            
            # Overall confidence
            if "overall" in scores:
                st.write("**Overall Confidence**")
                st.progress(scores["overall"], text=f"{scores['overall']:.2%}")
            
            # Individual metrics
            for metric in ["intent", "negation", "context_switch"]:
                if metric in scores:
                    st.write(f"**{metric.replace('_', ' ').title()}**")
                    st.progress(scores[metric], text=f"{scores[metric]:.2%}")

def generate_response_with_context(result: Dict[str, Any]) -> str:
    """
    Generate appropriate response based on enhanced context-aware processing result.
    
    Args:
        result: Context-aware processing result
        
    Returns:
        Response text
    """
    # Handle different response types based on the enhanced processing
    response_type = result.get("response_type", "standard")
    
    if response_type == "negation":
        if result.get("alternative_requested", False):
            # User negated previous request and specified an alternative
            service_type = result["entities"]["service_type"][0]
            return f"I understand you would prefer {service_type} instead. I'll help you with that. Can you provide your location?"
        elif "negated_intent" in result:
            # Negation of previous intent with no alternative
            return "I understand you don't want that anymore. What would you like me to help you with instead?"
        else:
            # Generic negation
            return "I understand. What would you like me to help you with?"
            
    elif response_type == "context_switch":
        switch_type = result.get("switch_type", "general_change")
        
        if switch_type == "service_change":
            service = result["entities"]["service_type"][0]
            return f"I understand you'd like to switch to {service}. I can help you with that. What specific details do you need assistance with?"
        elif switch_type == "location_change":
            location = result["entities"]["location"][0]
            return f"I've updated your location to {location}. Would you like me to proceed with the service at this new location?"
        else:
            return "I notice you've changed your request. Could you provide more details about what you need now?"
            
    elif response_type == "contradiction":
        # Handle contradictions with more nuanced responses
        if result["contradictions"]:
            contradiction = result["contradictions"][0]  # Get highest confidence contradiction
            entity_type = contradiction["entity_type"].replace("_", " ")
            current = contradiction["current_value"]
            previous = contradiction["previous_value"]
            
            return f"I notice you've changed the {entity_type} from {previous} to {current}. Would you like me to update this information?"
            
    elif response_type == "alternative_service":
        service = result["entities"]["service_type"][0]
        return f"I'll help you with {service} instead. What specific assistance do you need?"
        
    # Handle standard responses with flow awareness
    elif "flow" in result:
        flow = result["flow"]
        if flow == "towing":
            return "I can help you with towing. Could you please provide your current location and where you'd like your vehicle towed to?"
        elif flow == "roadside":
            return "I can help you with roadside assistance. What specific issue are you experiencing with your vehicle?"
        elif flow == "appointment":
            return "I can help you schedule a service appointment. What type of service do you need?"
        elif flow == "information":
            return "I can provide information about our services. What would you like to know?"
            
    # Handle clarification needs
    if result.get("needs_clarification", False):
        reason = next((action for action in result.get("suggested_actions", []) 
                      if action["type"] == "request_clarification"), {}).get("reason")
                      
        if reason == "low_confidence":
            return "I'm not quite sure I understood completely. Could you please provide more details about what you need?"
        else:
            return "Could you please clarify what specific assistance you need?"
    
    # Default response if no specific handling is needed
    return "How can I assist you today with towing, roadside assistance, or scheduling a service appointment?"

# Enhanced session state management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm your car maintenance assistant. How can I help you today?"}
    ]

if "flow" not in st.session_state:
    st.session_state.flow = None

if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = "standard"

# New session state for enhanced context tracking
if "context_history" not in st.session_state:
    st.session_state.context_history = {
        "switches": [],  # Track context switches
        "contradictions": [],  # Track contradictions
        "negations": [],  # Track negations
        "confidence_history": [],  # Track confidence scores
        "last_active_flow": None,  # Last active flow before switch
        "turn_count": 0  # Track conversation turns
    }

# Streamlit UI
st.title("Car Maintenance Chatbot")

# Configure the chatbot sidebar
with st.sidebar:
    st.subheader("Chatbot Configuration")
    
    # Toggle between standard and context-aware mode
    processing_mode = st.radio(
        "Processing Mode",
        ["Standard (Original)", "Context-Aware (Enhanced)"],
        index=0,
        help="Select the processing mode for the chatbot. 'Standard' uses the original model without context tracking, while 'Context-Aware' uses the enhanced model that maintains conversation context."
    )
    
    # Update the processing mode if changed
    if processing_mode == "Standard (Original)" and st.session_state.processing_mode != "standard":
        st.session_state.processing_mode = "standard"
    elif processing_mode == "Context-Aware (Enhanced)" and st.session_state.processing_mode != "context_aware":
        st.session_state.processing_mode = "context_aware"
    
    # Debug toggle
    debug_section = st.checkbox("Show debug information", value=False)
    
    # Update the clear conversation functionality
    if st.sidebar.button("Clear Conversation", help="Reset the conversation and all context"):
        # Reset messages
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi there! I'm your car maintenance assistant. How can I help you today?"}
        ]
        
        # Reset flow
        st.session_state.flow = None
        
        # Reset context history
        st.session_state.context_history = {
            "switches": [],
            "contradictions": [],
            "negations": [],
            "confidence_history": [],
            "last_active_flow": None,
            "turn_count": 0
        }
        
        # Reset context in context-aware mode
        if st.session_state.processing_mode == "context_aware":
            context_assistant = load_context_aware_assistant(MODEL_DIR)
            if context_assistant:
                context_assistant.conversation_context = {
                    "previous_intents": [],
                    "previous_entities": {},
                    "active_flow": None,
                    "turn_count": 0,
                    "context_switches": [],
                    "contradictions": [],
                    "negations": []
                }
        
        st.success("Conversation and context cleared!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show debug info if available and debug mode is enabled
        if debug_section and "debug_info" in message and message["role"] == "assistant":
            with st.expander("Debug Information"):
                st.json(message["debug_info"])

# Initialize models based on selected mode
models = load_actual_models(MODEL_DIR)
pipeline = ChatbotPipeline(models)

if st.session_state.processing_mode == "context_aware":
    context_assistant = load_context_aware_assistant(MODEL_DIR)
    
    # Display context information if available and debug mode is enabled
    if debug_section and context_assistant:
        display_enhanced_context_indicators(context_assistant.conversation_context)

# Input area
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Process the message based on selected mode
    with st.spinner("Thinking..."):
        if st.session_state.processing_mode == "standard":
            # Standard processing with original pipeline
            is_fallback, fallback_conf = pipeline.is_fallback(user_input)
            
            if debug_section:
                with st.sidebar:
                    st.write(f"Fallback check: {is_fallback} ({fallback_conf:.2f})")
            
            if is_fallback and fallback_conf > CONFIDENCE_THRESHOLD:
                response = "I'm sorry, I can't help with that. I'm designed to assist with car maintenance, towing, and roadside assistance."
                st.session_state.flow = "fallback"
                
                # Prepare debug info
                debug_info = {
                    "mode": "standard",
                    "is_fallback": is_fallback,
                    "fallback_confidence": fallback_conf,
                    "flow": "fallback"
                }
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
                
                # Prepare debug info
                debug_info = {
                    "mode": "standard",
                    "flow": predicted_flow,
                    "flow_confidence": flow_conf,
                    "intent": predicted_intent,
                    "intent_confidence": intent_conf
                }
                
        else:
            # Context-aware processing with enhanced assistant
            context_assistant = load_context_aware_assistant(MODEL_DIR)
            
            if context_assistant:
                # Process with context awareness
                result = context_assistant.process_message_with_context(user_input)
                
                # Generate response based on context-aware result
                response = generate_response_with_context(result)
                
                # Update flow in session state if available
                if "flow" in result:
                    st.session_state.flow = result["flow"]
                
                # Prepare debug info
                debug_info = {
                    "mode": "context_aware",
                    "result": result
                }
            else:
                # Fallback to standard processing if context assistant fails to load
                response = "I'm having trouble accessing my enhanced capabilities. Let me help you with basic assistance instead."
                debug_info = {
                    "mode": "fallback_to_standard",
                    "error": "Context-aware assistant unavailable"
                }
                
    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response, "debug_info": debug_info})
    with st.chat_message("assistant"):
        st.write(response)
        
        # Show debug info if debug mode is enabled
        if debug_section:
            with st.expander("Debug Information"):
                st.json(debug_info)

# Update context history after processing
if st.session_state.processing_mode == "context_aware" and context_assistant:
    # Update context history
    st.session_state.context_history["turn_count"] += 1
    
    if result.get("contains_context_switch"):
        st.session_state.context_history["switches"].append({
            "turn": st.session_state.context_history["turn_count"],
            "from_flow": st.session_state.context_history["last_active_flow"],
            "to_flow": result.get("flow"),
            "confidence": result.get("context_switch_confidence", 0)
        })
    
    if result.get("contradictions"):
        st.session_state.context_history["contradictions"].extend(
            result["contradictions"]
        )
    
    if result.get("contains_negation"):
        st.session_state.context_history["negations"].append({
            "turn": st.session_state.context_history["turn_count"],
            "text": user_input,
            "confidence": result.get("negation_confidence", 0)
        })
    
    # Update last active flow
    if result.get("flow"):
        st.session_state.context_history["last_active_flow"] = result["flow"]
    
    # Track confidence history
    st.session_state.context_history["confidence_history"].append({
        "turn": st.session_state.context_history["turn_count"],
        "scores": result.get("confidence_scores", {})
    })

# Display note about models at the bottom
if st.session_state.processing_mode == "standard":
    st.caption("This chatbot is using the standard model for intent classification and flow detection.")
else:
    st.caption("This chatbot is using the enhanced context-aware model that maintains conversation history.")