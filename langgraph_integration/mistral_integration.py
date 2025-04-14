import os
import json
import requests
from typing import Dict, Any, Optional

class MistralEnhancer:
    """Enhanced language understanding using Mistral API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model_name = "mistral-small-latest"
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = None
        self._initialize_headers()
    
    def _initialize_headers(self) -> None:
        """Initialize API headers"""
        if self.api_key:
            self.headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
    
    def is_available(self) -> bool:
        """Check if Mistral is available"""
        return self.headers is not None
    
    def _chat_completion(self, prompt: str) -> str:
        """Make a chat completion call to Mistral API"""
        if not self.is_available():
            return ""
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"Error in Mistral API call: {e}")
            return ""
    
    def analyze_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze intent using Mistral"""
        if not self.is_available():
            return {"intent": "unknown", "confidence": 0.0}
        
        prompt = self._create_intent_prompt(text, context)
        response_text = self._chat_completion(prompt)
        
        # Parse response - in a real implementation, use more robust parsing
        # Simplified version for demonstration
        if "roadside" in response_text.lower():
            return {"intent": "request_roadside", "confidence": 0.8}
        elif "tow" in text.lower() or "tow" in response_text.lower():
            return {"intent": "request_tow", "confidence": 0.8}
        elif "road" in text.lower() or "road" in response_text.lower():
            return {"intent": "request_roadside", "confidence": 0.8}
        else:
            return {"intent": "unknown", "confidence": 0.5}
    
    def detect_negation(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect negation using Mistral"""
        if not self.is_available():
            return {"is_negation": False, "confidence": 0.0}
        
        prompt = self._create_negation_prompt(text, context)
        response_text = self._chat_completion(prompt)
        
        # Simple parsing for demonstration
        is_negation = "yes" in response_text.lower() or "true" in response_text.lower()
        confidence = 0.9 if is_negation else 0.1
        
        return {"is_negation": is_negation, "confidence": confidence}
    
    def detect_context_switch(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect context switch using Mistral"""
        if not self.is_available():
            return {"has_context_switch": False, "confidence": 0.0, "new_context": None}
        
        prompt = self._create_context_switch_prompt(text, context)
        response_text = self._chat_completion(prompt)
        
        # Simple parsing for demonstration
        has_context_switch = "yes" in response_text.lower() or "true" in response_text.lower()
        
        # Try to extract new context type with a simple approach
        new_context = None
        if has_context_switch:
            lower_response = response_text.lower()
            if "tow" in lower_response:
                new_context = "towing"
            elif "road" in lower_response or "assist" in lower_response:
                new_context = "roadside"
            elif "appoint" in lower_response or "schedul" in lower_response:
                new_context = "appointment"
        
        confidence = 0.8 if has_context_switch else 0.2
        
        return {
            "has_context_switch": has_context_switch,
            "confidence": confidence,
            "new_context": new_context
        }
    
    def _create_intent_prompt(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """Create prompt for intent analysis"""
        context_str = json.dumps(context) if context else "No context available"
        
        return f"""
        Analyze the following user request for an automotive assistance chatbot:
        
        User message: "{text}"
        
        Previous context: {context_str}
        
        What is the primary intent of this message? Is the user asking for:
        - Towing service
        - Roadside assistance (battery, tire, etc.)
        - Appointment scheduling
        - Vehicle service information
        - Something else
        
        Respond with just the primary intent category.
        """
    
    def _create_negation_prompt(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """Create prompt for negation detection"""
        context_str = json.dumps(context) if context else "No context available"
        
        return f"""
        Analyze the following user message for an automotive assistance chatbot:
        
        User message: "{text}"
        
        Previous context: {context_str}
        
        Is the user negating or declining a previous request or suggestion?
        Respond with Yes or No.
        """
    
    def _create_context_switch_prompt(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """Create prompt for context switch detection"""
        context_str = json.dumps(context) if context else "No context available"
        
        return f"""
        Analyze the following user message for an automotive assistance chatbot:
        
        User message: "{text}"
        
        Previous context: {context_str}
        
        Is the user switching to a different service or topic than what was previously discussed?
        If yes, what is the new service or topic they want to discuss?
        Respond with Yes or No, followed by the new service (towing, roadside assistance, appointment) if applicable.
        """ 