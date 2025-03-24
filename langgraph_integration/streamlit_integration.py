import uuid
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
import json
import logging
import os

# Note: streamlit is imported conditionally to allow the module to be used
# in non-streamlit environments without errors

from langgraph_integration.feature_flags import FeatureFlags
from langgraph_integration.state_converter import StateConverter
from langgraph_integration.langgraph_workflow import LangGraphWorkflow
from langgraph_integration.adapters import ExistingDetectionAdapter
from langgraph_integration.hybrid_detection import HybridDetectionSystem
from langgraph_integration.mistral_integration import MistralEnhancer
from langgraph_integration.monitoring import MonitoringSystem, timed_execution
from langgraph_integration.cpu_optimizations import CPUOptimizer

class StreamlitApp:
    """Streamlit app with LangGraph integration"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize all components"""
        try:
            import streamlit as st
            self.st = st
        except ImportError:
            logging.warning("Streamlit not installed. UI functionality will be limited.")
            self.st = None
            
        # Create session state if using streamlit
        if self.st:
            if "initialized" not in self.st.session_state:
                self.st.session_state.initialized = True
                self._setup_session_state()
        
        # Initialize components without streamlit dependency
        self._feature_flags = FeatureFlags(self.config_path)
        self._state_converter = StateConverter()
        self._monitoring = MonitoringSystem()
        self._monitoring.start_worker()
        self._existing_detector = ExistingDetectionAdapter()
        
        # CPU optimizer
        self._cpu_optimizer = CPUOptimizer()
        
        # Mistral enhancer (if API key is provided)
        api_key = os.environ.get("MISTRAL_API_KEY", None)
        self._mistral_enhancer = MistralEnhancer(api_key)
        
        # Hybrid detection system
        self._hybrid_system = HybridDetectionSystem(
            self._feature_flags,
            self._existing_detector,
            self._mistral_enhancer
        )
        
        # LangGraph workflow
        self._workflow = LangGraphWorkflow(
            self._feature_flags,
            self._hybrid_system,
            self._existing_detector
        )
        
        # Track conversation state
        self._conversation_id = str(uuid.uuid4())
        self._context = {
            "conversation_id": self._conversation_id,
            "turn_count": 0,
            "flow": "unknown"
        }
        self._messages = []
            
    def _setup_session_state(self) -> None:
        """Initialize Streamlit session state variables"""
        self.st.session_state.conversation_id = str(uuid.uuid4())
        self.st.session_state.messages = []
        self.st.session_state.context = {
            "conversation_id": self.st.session_state.conversation_id,
            "turn_count": 0,
            "flow": "unknown"
        }
        self.st.session_state.debug_mode = False
        self.st.session_state.show_metrics = False
        
    def render_chat_ui(self) -> None:
        """Render the chat UI in Streamlit"""
        if not self.st:
            logging.error("Streamlit not available")
            return
            
        # Set page config
        self.st.set_page_config(
            page_title="Context-Aware Chatbot",
            page_icon="🚗",
            layout="wide"
        )
        
        # Create sidebar for controls
        with self.st.sidebar:
            self.st.title("Chatbot Settings")
            
            # Feature flags
            self.st.header("Feature Flags")
            
            # Create toggles for each feature flag
            for flag_name, enabled in self._feature_flags.flags.items():
                if self.st.checkbox(f"Enable {flag_name.replace('_', ' ').title()}", 
                                   value=enabled, 
                                   key=f"flag_{flag_name}"):
                    self._feature_flags.enable(flag_name)
                else:
                    self._feature_flags.disable(flag_name)
            
            # Debugging options
            self.st.header("Debugging")
            self.st.session_state.debug_mode = self.st.checkbox(
                "Show debugging info", 
                value=self.st.session_state.get("debug_mode", False)
            )
            
            self.st.session_state.show_metrics = self.st.checkbox(
                "Show metrics", 
                value=self.st.session_state.get("show_metrics", False)
            )
            
            # Clear conversation button
            if self.st.button("Clear Conversation"):
                self.st.session_state.messages = []
                self.st.session_state.context = {
                    "conversation_id": self.st.session_state.conversation_id,
                    "turn_count": 0,
                    "flow": "unknown"
                }
                self.st.rerun()
        
        # Main chat interface
        self.st.title("Automotive Assistant")
        
        # Display chat messages
        for message in self.st.session_state.messages:
            with self.st.chat_message(message["role"]):
                self.st.write(message["content"])
                
                # Show debug info if enabled
                if self.st.session_state.debug_mode and "debug_info" in message:
                    with self.st.expander("Debug Info"):
                        self.st.json(message["debug_info"])
        
        # Chat input
        if prompt := self.st.chat_input("How can I help you today?"):
            # Add user message to chat history
            self.st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with self.st.chat_message("user"):
                self.st.write(prompt)
            
            # Process message with LangGraph workflow
            with self.st.spinner("Thinking..."):
                start_time = time.time()
                
                # Convert context to LangGraph state
                langgraph_state = self._state_converter.from_context(
                    self.st.session_state.context, 
                    prompt
                )
                
                # Process with workflow
                try:
                    result_state = self._workflow.invoke(langgraph_state)
                    
                    # Convert back to context
                    self.st.session_state.context = self._state_converter.to_context(result_state)
                    
                    # Get response
                    response = result_state.get("response", "I'm not sure how to respond to that.")
                    
                    # Prepare debug info
                    debug_info = {
                        "flow": result_state.get("flow", "unknown"),
                        "intent": result_state.get("intent", "unknown"),
                        "detected_negation": result_state.get("detected_negation", False),
                        "detected_context_switch": result_state.get("detected_context_switch", False),
                        "confidence_scores": result_state.get("confidence_scores", {})
                    }
                    
                    # Record metrics
                    latency = time.time() - start_time
                    self._monitoring.track_response(
                        self.st.session_state.conversation_id,
                        response,
                        latency,
                        result_state
                    )
                    
                    # Add assistant message to chat history
                    self.st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "debug_info": debug_info
                    })
                    
                    # Display assistant response
                    with self.st.chat_message("assistant"):
                        self.st.write(response)
                        
                        # Show debug info if enabled
                        if self.st.session_state.debug_mode:
                            with self.st.expander("Debug Info"):
                                self.st.json(debug_info)
                                
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    self.st.error(error_msg)
                    
                    # Record error
                    self._monitoring.track_error(
                        self.st.session_state.conversation_id,
                        e,
                        self.st.session_state.context
                    )
                    
                    # Add error message to chat history
                    self.st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "I'm sorry, but I encountered an error processing your request."
                    })
        
        # Display metrics if enabled
        if self.st.session_state.show_metrics:
            with self.st.expander("Performance Metrics"):
                metrics = self._monitoring.get_metrics()
                
                # Display in columns
                col1, col2 = self.st.columns(2)
                
                with col1:
                    self.st.metric("Total Requests", metrics["requests"])
                    self.st.metric("Successful Responses", metrics["successful_responses"])
                    self.st.metric("Errors", metrics["errors"])
                    self.st.metric("Avg Latency (s)", round(metrics["average_latency"], 4))
                
                with col2:
                    self.st.metric("Negation Detections", metrics["negation_detections"])
                    self.st.metric("Context Switches", metrics["context_switches"])
                    self.st.metric("Rule-Based Decisions", metrics["rule_based_decisions"])
                    self.st.metric("ML-Based Decisions", metrics["ml_based_decisions"])
    
    def process_message(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a message without Streamlit UI
        Useful for API integrations
        """
        # Use provided context or default
        if context is None:
            context = self._context
            
        # Start timing
        start_time = time.time()
        
        # Generate message ID
        message_id = str(uuid.uuid4())
        
        # Track request
        self._monitoring.track_request(message_id, text, context)
        
        try:
            # Convert context to LangGraph state
            langgraph_state = self._state_converter.from_context(context, text)
            
            # Process with workflow
            result_state = self._workflow.invoke(langgraph_state)
            
            # Convert back to context
            updated_context = self._state_converter.to_context(result_state)
            
            # Get response
            response = result_state.get("response", "I'm not sure how to respond to that.")
            
            # Record metrics
            latency = time.time() - start_time
            self._monitoring.track_response(message_id, response, latency, result_state)
            
            # Build response object
            result = {
                "response": response,
                "context": updated_context,
                "message_id": message_id,
                "latency": latency,
                "intent": result_state.get("intent", "unknown"),
                "flow": result_state.get("flow", "unknown"),
                "detected_negation": result_state.get("detected_negation", False),
                "detected_context_switch": result_state.get("detected_context_switch", False)
            }
            
            # Update internal context
            self._context = updated_context
            
            return result
            
        except Exception as e:
            # Record error
            self._monitoring.track_error(message_id, e, context)
            
            # Return error response
            return {
                "error": str(e),
                "message_id": message_id,
                "context": context
            }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self._monitoring:
            self._monitoring.stop_worker()
            
def run_streamlit_app():
    """Entry point for running the Streamlit app"""
    app = StreamlitApp()
    app.render_chat_ui() 