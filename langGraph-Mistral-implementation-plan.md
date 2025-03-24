# LangGraph & Mistral Implementation Plan for Context-Aware Chatbot

This plan outlines the integration of LangGraph with Mistral 7B to enhance the existing context-aware chatbot, while preserving and prioritizing current rule-based methods.

## Core Design Principles

- ✅ **Existing Code First**: Rule-based methods remain primary, with Mistral as enhancement
- ✅ **Adapter Pattern**: Use adapters to integrate existing components with LangGraph
- ✅ **Feature Flags**: Implement flags for controlled, incremental deployment
- ✅ **CPU Optimization**: All performance optimizations designed for CPU-only environments
- ✅ **Preserve UX**: Maintain existing Streamlit UI while enhancing backend
- ✅ **Robust Monitoring**: Implement structured logging and performance metrics

## Phase 1: Foundation and Adapter Layer

### Objectives

- [x] Set up development environment
- [x] Create adapters for existing models and components
- [x] Implement feature flag system
- [x] Define core interfaces

### Implementation Steps

#### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install langgraph langchain langchain-community
pip install mistralai
pip install pytest pytest-cov
```

#### 2. Feature Flag System

```python
# feature_flags.py
import os
import json
from typing import Dict, Any, Optional

class FeatureFlags:
    """
    Feature flag management for controlling LangGraph integration
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get(
            "FEATURE_FLAG_CONFIG",
            "./config/feature_flags.json"
        )
        self.flags = self._load_flags()

    def _load_flags(self) -> Dict[str, bool]:
        """Load feature flags from file or use defaults"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading feature flags: {e}")

        # Default flags - all disabled initially
        return {
            "use_langgraph": False,        # Use LangGraph for orchestration
            "use_mistral": False,          # Use Mistral for enhanced NLU
            "hybrid_detection": False,     # Use hybrid rule/ML detection
            "enhanced_logging": True,      # Use enhanced logging and metrics
            "retain_entities": True,       # Keep entity values across context switches
            "async_processing": False      # Use async processing (CPU optimization)
        }

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled"""
        return self.flags.get(flag_name, False)

    def enable(self, flag_name: str) -> None:
        """Enable a feature flag"""
        self.flags[flag_name] = True
        self._save_flags()

    def disable(self, flag_name: str) -> None:
        """Disable a feature flag"""
        self.flags[flag_name] = False
        self._save_flags()

    def _save_flags(self) -> None:
        """Save current flags to config file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.flags, f, indent=2)
        except Exception as e:
            print(f"Error saving feature flags: {e}")
```

#### 3. Adapter Interfaces

```python
# adapters.py
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
        return self.assistant.detect_contradictions(text, context)

    def process_message(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process message using existing assistant"""
        if context:
            return self.assistant.process_message_with_context(text, context)
        else:
            return self.assistant.process_message(text)
```

#### 4. LangGraph State Interface

```python
# langgraph_state.py
from typing import TypedDict, Dict, List, Any, Optional

class ConversationState(TypedDict, total=False):
    """State definition for LangGraph nodes"""
    conversation_id: str
    turn_count: int
    current_message: str
    messages: List[Dict[str, str]]
    context: Dict[str, Any]
    flow: str
    intent: str
    entities: Dict[str, Any]
    needs_clarification: bool
    detected_negation: bool
    detected_context_switch: bool
    confidence_scores: Dict[str, float]
    should_fallback: bool
    response: Optional[str]
```

### Phase 1 Testing

```python
# test_phase1.py
import pytest
import os
import json
import tempfile
from feature_flags import FeatureFlags
from adapters import ExistingModelAdapter, ExistingDetectionAdapter
from langgraph_state import ConversationState

def test_feature_flags():
    """Test feature flag functionality"""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        json.dump({"use_langgraph": True, "use_mistral": False}, f)
        config_path = f.name

    try:
        # Initialize feature flags
        flags = FeatureFlags(config_path)

        # Test flag access
        assert flags.is_enabled("use_langgraph") is True
        assert flags.is_enabled("use_mistral") is False
        assert flags.is_enabled("nonexistent_flag") is False

        # Test flag modification
        flags.enable("use_mistral")
        assert flags.is_enabled("use_mistral") is True

        flags.disable("use_langgraph")
        assert flags.is_enabled("use_langgraph") is False
    finally:
        os.unlink(config_path)

def test_existing_detection_adapter():
    """Test adapter for existing detection methods"""
    adapter = ExistingDetectionAdapter()

    # Test negation detection
    negation_result = adapter.detect_negation("I don't need a tow truck")
    assert "is_negation" in negation_result
    assert negation_result["is_negation"] is True

    # Test context switch detection
    switch_result = adapter.detect_context_switch("Actually, I need roadside assistance instead")
    assert "has_context_switch" in switch_result

    # Test message processing
    process_result = adapter.process_message("I need a tow truck")
    assert "intent" in process_result
    assert "flow" in process_result
```

## Phase 2: Mistral Integration as Enhancement

### Objectives

- [x] Set up Mistral 7B integration
- [x] Implement hybrid detection system (rule-based + ML)
- [x] Create weighting/confidence mechanism to combine results

### Implementation Steps

#### 1. Mistral Integration

```python
# mistral_integration.py
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
```

#### 2. Hybrid Detection System

```python
# hybrid_detection.py
from typing import Dict, Any, Optional
from langgraph_integration.feature_flags import FeatureFlags
from langgraph_integration.adapters import ExistingDetectionAdapter
from langgraph_integration.mistral_integration import MistralEnhancer

class HybridDetectionSystem:
    """
    Combines rule-based and ML-based detection with configurable weighting
    """

    def __init__(
        self,
        flags: FeatureFlags,
        existing_detector: ExistingDetectionAdapter,
        mistral_enhancer: Optional[MistralEnhancer] = None
    ):
        self.flags = flags
        self.existing_detector = existing_detector
        self.mistral_enhancer = mistral_enhancer
        self.rule_weight = 0.7  # Default to prioritizing rule-based detection
        self.ml_weight = 0.3    # Lower weight for ML initially

    def set_weights(self, rule_weight: float) -> None:
        """Set the weighting between rule-based and ML detection"""
        self.rule_weight = max(0.0, min(1.0, rule_weight))
        self.ml_weight = 1.0 - self.rule_weight

    def detect_negation(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect negation using hybrid approach
        Prioritizes rule-based methods but enhances with ML when available
        """
        # Get rule-based result
        rule_result = self.existing_detector.detect_negation(text)

        # If Mistral not enabled or available, just return rule-based result
        if not self.flags.is_enabled("use_mistral") or self.mistral_enhancer is None:
            return rule_result

        # If hybrid detection disabled, return rule-based result
        if not self.flags.is_enabled("hybrid_detection"):
            return rule_result

        # Get ML result
        ml_result = self.mistral_enhancer.detect_negation(text, context)

        # Apply weighted decision logic
        rule_confidence = rule_result.get("confidence", 0.0) * self.rule_weight
        ml_confidence = ml_result.get("confidence", 0.0) * self.ml_weight

        # If they agree, combine confidence
        if rule_result.get("is_negation") == ml_result.get("is_negation"):
            return {
                "is_negation": rule_result.get("is_negation"),
                "confidence": rule_confidence + ml_confidence,
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": False,
                "ml_based_decision": False
            }

        # If they disagree, go with the higher weighted confidence
        if rule_confidence >= ml_confidence:
            return {
                "is_negation": rule_result.get("is_negation"),
                "confidence": rule_confidence,
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": True,
                "ml_based_decision": False
            }
        else:
            return {
                "is_negation": ml_result.get("is_negation"),
                "confidence": ml_confidence,
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": False,
                "ml_based_decision": True
            }

    def detect_context_switch(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect context switching using hybrid approach
        Implementation follows the same pattern as detect_negation
        """
        # Get rule-based result
        rule_result = self.existing_detector.detect_context_switch(text)

        # If Mistral not enabled or available, just return rule-based result
        if not self.flags.is_enabled("use_mistral") or self.mistral_enhancer is None:
            return rule_result

        # If hybrid detection disabled, return rule-based result
        if not self.flags.is_enabled("hybrid_detection"):
            return rule_result

        # Get ML result from Mistral
        ml_result = self.mistral_enhancer.detect_context_switch(text, context)

        # Apply weighted decision logic
        rule_confidence = rule_result.get("confidence", 0.0) * self.rule_weight
        ml_confidence = ml_result.get("confidence", 0.0) * self.ml_weight

        # If they agree, combine confidence
        if rule_result.get("has_context_switch") == ml_result.get("has_context_switch"):
            # For new_context, prefer rule-based if available, otherwise use ML
            new_context = rule_result.get("new_context") or ml_result.get("new_context")

            return {
                "has_context_switch": rule_result.get("has_context_switch"),
                "confidence": rule_confidence + ml_confidence,
                "new_context": new_context,
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": False,
                "ml_based_decision": False
            }

        # If they disagree, go with the higher weighted confidence
        if rule_confidence >= ml_confidence:
            return {
                "has_context_switch": rule_result.get("has_context_switch"),
                "confidence": rule_confidence,
                "new_context": rule_result.get("new_context"),
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": True,
                "ml_based_decision": False
            }
        else:
            return {
                "has_context_switch": ml_result.get("has_context_switch"),
                "confidence": ml_confidence,
                "new_context": ml_result.get("new_context"),
                "rule_based": rule_result,
                "ml_based": ml_result,
                "rule_based_decision": False,
                "ml_based_decision": True
            }

    def analyze_intent(self, text: str, flow: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze intent using hybrid approach
        Implementation follows the same pattern as other hybrid methods
        """
        # Currently, we don't have a direct intent analysis in ExistingDetectionAdapter
        # So we'll process the message and extract intent
        rule_result = self.existing_detector.process_message(text, context)
        rule_intent = {
            "intent": rule_result.get("intent", "unknown"),
            "confidence": rule_result.get("confidence", 0.5) if "confidence" in rule_result else 0.5
        }

        # If Mistral not enabled or available, just return rule-based result
        if not self.flags.is_enabled("use_mistral") or self.mistral_enhancer is None:
            return rule_intent

        # If hybrid detection disabled, return rule-based result
        if not self.flags.is_enabled("hybrid_detection"):
            return rule_intent

        # Get ML result
        ml_intent = self.mistral_enhancer.analyze_intent(text, context)

        # Apply weighted decision logic
        rule_confidence = rule_intent.get("confidence", 0.0) * self.rule_weight
        ml_confidence = ml_intent.get("confidence", 0.0) * self.ml_weight

        # If they agree, combine confidence
        if rule_intent.get("intent") == ml_intent.get("intent"):
            return {
                "intent": rule_intent.get("intent"),
                "confidence": rule_confidence + ml_confidence,
                "rule_based": rule_intent,
                "ml_based": ml_intent,
                "rule_based_decision": False,
                "ml_based_decision": False
            }

        # If they disagree, go with the higher weighted confidence
        if rule_confidence >= ml_confidence:
            return {
                "intent": rule_intent.get("intent"),
                "confidence": rule_confidence,
                "rule_based": rule_intent,
                "ml_based": ml_intent,
                "rule_based_decision": True,
                "ml_based_decision": False
            }
        else:
            return {
                "intent": ml_intent.get("intent"),
                "confidence": ml_confidence,
                "rule_based": rule_intent,
                "ml_based": ml_intent,
                "rule_based_decision": False,
                "ml_based_decision": True
            }
    }
```

### Phase 2 Testing

```python
# tests/test_phase2.py
import unittest
from unittest.mock import patch, MagicMock
import json
import os
from typing import Dict, Any, Optional

from langgraph_integration import MistralEnhancer, HybridDetectionSystem, FeatureFlags

class MockFeatureFlags(FeatureFlags):
    def __init__(self, enable_mistral: bool = False):
        self.enable_mistral = enable_mistral
        self.flags = {"use_mistral": enable_mistral, "hybrid_detection": enable_mistral}

    def is_enabled(self, flag_name: str) -> bool:
        return self.flags.get(flag_name, False)

    # Keep the original method for compatibility
    def is_feature_enabled(self, feature_name: str) -> bool:
        if feature_name == "mistral_integration":
            return self.enable_mistral
        return False

class MockExistingDetector:
    def detect_negation(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if "not" in text.lower() or "don't" in text.lower():
            return {"is_negation": True, "confidence": 0.9}
        return {"is_negation": False, "confidence": 0.8}

    def detect_context_switch(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if "instead" in text.lower() or "actually" in text.lower():
            return {"has_context_switch": True, "confidence": 0.9, "new_context": "towing"}
        return {"has_context_switch": False, "confidence": 0.8, "new_context": None}

    def analyze_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if "tow" in text.lower():
            return {"intent": "request_tow", "confidence": 0.9}
        elif "road" in text.lower():
            return {"intent": "request_roadside", "confidence": 0.9}
        else:
            return {"intent": "unknown", "confidence": 0.5}

    def process_message(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process message to extract intent and other information"""
        intent_result = self.analyze_intent(text, context)

        if intent_result["intent"] == "request_tow":
            flow = "towing"
        elif intent_result["intent"] == "request_roadside":
            flow = "roadside"
        else:
            flow = "unknown"

        result = {
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "flow": flow,
            "needs_clarification": intent_result["intent"] == "unknown",
            "entities": []
        }

        return result

# Test classes for MistralEnhancer and HybridDetectionSystem
# ...
```

---

#### Implementation Notes:

1. We modified the implementation to use direct API calls to Mistral via the requests library instead of using LangChain, providing:

   - Reduced dependencies
   - More direct control over API interactions
   - Better stability across updates

2. We added a proper context switch detection method to the Mistral integration, ensuring that all three key detection methods are consistently implemented:

   - Intent analysis
   - Negation detection
   - Context switch detection

3. The hybrid detection system adds diagnostic information to the results:
   - `rule_based_decision`: Indicates when rule-based logic overrode ML
   - `ml_based_decision`: Indicates when ML overrode rule-based logic
   - Includes both original results for traceability

## Phase 3: LangGraph Integration for Flow Control

### Objectives

- [ ] Implement core LangGraph nodes
- [ ] Create LangGraph state converter for existing context
- [ ] Build the graph with conditional routing
- [ ] Maintain compatibility with existing system

### Implementation Steps

#### 1. LangGraph Node Implementation

```python
# langgraph_nodes.py
from typing import Dict, Any, Optional
from langgraph_state import ConversationState
from adapters import ExistingDetectionAdapter
from hybrid_detection import HybridDetectionSystem

def context_tracker_node(state: ConversationState) -> ConversationState:
    """Track conversation context"""
    new_state = dict(state)

    # Update turn count
    if "turn_count" not in new_state:
        new_state["turn_count"] = 0
    new_state["turn_count"] += 1

    # Add current message to history
    if "messages" not in new_state:
        new_state["messages"] = []

    if "current_message" in new_state:
        new_state["messages"].append({
            "role": "user",
            "content": new_state["current_message"]
        })

    return new_state

def detection_node(
    state: ConversationState,
    hybrid_system: HybridDetectionSystem
) -> ConversationState:
    """Detect intent, negation, and context switches"""
    new_state = dict(state)
    text = new_state.get("current_message", "")
    context = new_state.get("context", {})

    # Detect negation
    negation_result = hybrid_system.detect_negation(text, context)
    new_state["detected_negation"] = negation_result.get("is_negation", False)

    # Detect context switch
    switch_result = hybrid_system.detect_context_switch(text, context)
    new_state["detected_context_switch"] = switch_result.get("has_context_switch", False)

    # Store confidence scores
    if "confidence_scores" not in new_state:
        new_state["confidence_scores"] = {}

    new_state["confidence_scores"]["negation"] = negation_result.get("confidence", 0.0)
    new_state["confidence_scores"]["context_switch"] = switch_result.get("confidence", 0.0)

    # Update flow if context switch detected
    if new_state["detected_context_switch"] and switch_result.get("new_context"):
        new_state["flow"] = switch_result.get("new_context")

    return new_state

def negation_handler_node(state: ConversationState) -> ConversationState:
    """Handle negation cases"""
    new_state = dict(state)

    # Extract negated intent/flow from context
    context = new_state.get("context", {})
    last_intent = context.get("last_intent", "unknown")

    # Update context
    if "context" not in new_state:
        new_state["context"] = {}

    new_state["context"]["negated_intent"] = last_intent
    new_state["context"]["requires_clarification"] = True

    return new_state

def context_switch_handler_node(state: ConversationState) -> ConversationState:
    """Handle context switch cases"""
    new_state = dict(state)
    context = new_state.get("context", {})

    # Track the switch
    if "context" not in new_state:
        new_state["context"] = {}

    new_state["context"]["previous_flow"] = context.get("flow", "unknown")
    new_state["context"]["context_switch_count"] = context.get("context_switch_count", 0) + 1

    return new_state

def regular_handler_node(state: ConversationState) -> ConversationState:
    """Handle regular (non-negation, non-context-switch) requests"""
    new_state = dict(state)

    # Process using existing logic (simplified here)
    if "context" not in new_state:
        new_state["context"] = {}

    return new_state

def response_node(
    state: ConversationState,
    existing_detector: ExistingDetectionAdapter
) -> ConversationState:
    """Generate response using existing system"""
    new_state = dict(state)

    # Get input text and context
    text = new_state.get("current_message", "")
    context = new_state.get("context", {})

    # Process with existing system
    result = existing_detector.process_message(text, context)

    # Set response
    new_state["response"] = result.get("response", "I'm not sure how to respond to that.")

    # Add to message history
    if "messages" not in new_state:
        new_state["messages"] = []

    new_state["messages"].append({
        "role": "assistant",
        "content": new_state["response"]
    })

    return new_state
```

#### 2. State Converter

```python
# state_converter.py
from typing import Dict, Any, Optional
from langgraph_state import ConversationState

class StateConverter:
    """Convert between LangGraph state and existing context format"""

    def from_context(self, context: Dict[str, Any], text: str) -> ConversationState:
        """Convert existing context to LangGraph state"""
        state = ConversationState(
            conversation_id=context.get("conversation_id", ""),
            turn_count=context.get("turn_count", 0),
            current_message=text,
            messages=[],  # Will be populated by context_tracker_node
            context=context,
            flow=context.get("flow", "unknown"),
            intent=context.get("last_intent", "unknown"),
            entities=self._extract_entities(context),
            needs_clarification=context.get("needs_clarification", False),
            detected_negation=False,  # Will be determined in graph
            detected_context_switch=False,  # Will be determined in graph
            confidence_scores={},
            should_fallback=False
        )
        return state

    def to_context(self, state: ConversationState) -> Dict[str, Any]:
        """Convert LangGraph state back to existing context format"""
        context = state.get("context", {}).copy()

        # Update with latest values from state
        context["conversation_id"] = state.get("conversation_id", context.get("conversation_id", ""))
        context["turn_count"] = state.get("turn_count", context.get("turn_count", 0))
        context["flow"] = state.get("flow", context.get("flow", "unknown"))
        context["last_intent"] = state.get("intent", context.get("last_intent", "unknown"))
        context["needs_clarification"] = state.get("needs_clarification", context.get("needs_clarification", False))

        # Add entity values to flattened context for compatibility
        for entity_type, value in state.get("entities", {}).items():
            context[entity_type] = value

        return context

    def _extract_entities(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from existing context format"""
        entities = {}

        # Common entity types in automotive domain
        entity_types = [
            "vehicle_type", "vehicle_make", "vehicle_model", "vehicle_year",
            "location", "service_type", "appointment_time", "issue_type"
        ]

        for entity_type in entity_types:
            if entity_type in context:
                entities[entity_type] = context[entity_type]

        return entities
```

#### 3. LangGraph Workflow Construction

```python
# langgraph_workflow.py
from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph_state import ConversationState
from feature_flags import FeatureFlags
from hybrid_detection import HybridDetectionSystem
from adapters import ExistingDetectionAdapter
from langgraph_nodes import (
    context_tracker_node,
    detection_node,
    negation_handler_node,
    context_switch_handler_node,
    regular_handler_node,
    response_node
)

class LangGraphWorkflow:
    """LangGraph workflow for conversation management"""

    def __init__(
        self,
        flags: FeatureFlags,
        hybrid_system: HybridDetectionSystem,
        existing_detector: ExistingDetectionAdapter
    ):
        self.flags = flags
        self.hybrid_system = hybrid_system
        self.existing_detector = existing_detector
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create the workflow
        workflow = StateGraph(ConversationState)

        # Define node processors with dependencies
        def detection_with_deps(state):
            return detection_node(state, self.hybrid_system)

        def response_with_deps(state):
            return response_node(state, self.existing_detector)

        # Add nodes
        workflow.add_node("context_tracker", context_tracker_node)
        workflow.add_node("detection", detection_with_deps)
        workflow.add_node("negation_handler", negation_handler_node)
        workflow.add_node("context_switch_handler", context_switch_handler_node)
        workflow.add_node("regular_handler", regular_handler_node)
        workflow.add_node("response", response_with_deps)

        # Add edges
        workflow.add_edge("context_tracker", "detection")

        # Add conditional routing
        workflow.add_conditional_edges(
            "detection",
            lambda state: (
                "negation_handler" if state["detected_negation"]
                else "context_switch_handler" if state["detected_context_switch"]
                else "regular_handler"
            )
        )

        workflow.add_edge("negation_handler", "response")
        workflow.add_edge("context_switch_handler", "response")
        workflow.add_edge("regular_handler", "response")
        workflow.add_edge("response", END)

        # Set entry point
        workflow.set_entry_point("context_tracker")

        return workflow

    def invoke(self, state: ConversationState) -> ConversationState:
        """Process conversation using LangGraph workflow"""
        if not self.flags.is_enabled("use_langgraph"):
            # Fallback to existing system if LangGraph is disabled
            text = state.get("current_message", "")
            context = state.get("context", {})

            result = self.existing_detector.process_message(text, context)

            # Convert result to expected format
            response = result.get("response", "I'm not sure how to respond to that.")

            # Update state with response
            state["response"] = response

            if "messages" not in state:
                state["messages"] = []

            state["messages"].append({
                "role": "assistant",
                "content": response
            })

            return state

        # Use LangGraph workflow
        return self.graph.invoke(state)
```

### Phase 3 Testing

```python
# test_phase3.py
import pytest
from unittest.mock import MagicMock
from langgraph_state import ConversationState
from feature_flags import FeatureFlags
from langgraph_workflow import LangGraphWorkflow
from state_converter import StateConverter

# Mock classes
class MockFeatureFlags:
    def __init__(self, flags=None):
        self.flags = flags or {}

    def is_enabled(self, flag_name):
        return self.flags.get(flag_name, False)

class MockHybridSystem:
    def detect_negation(self, text, context=None):
        return {"is_negation": "don't" in text.lower(), "confidence": 0.9}

    def detect_context_switch(self, text, context=None):
        return {"has_context_switch": "instead" in text.lower(), "confidence": 0.9}

class MockExistingDetector:
    def process_message(self, text, context=None):
        return {
            "intent": "request_tow" if "tow" in text.lower() else "unknown",
            "flow": "towing" if "tow" in text.lower() else "unknown",
            "response": "I'll help you with a tow truck." if "tow" in text.lower() else "How can I help you?"
        }

# Tests
def test_state_converter():
    """Test conversion between contexts"""
    converter = StateConverter()

    # Create existing context
    context = {
        "conversation_id": "test-123",
        "turn_count": 2,
        "flow": "towing",
        "last_intent": "request_tow",
        "vehicle_type": "sedan",
        "location": "downtown"
    }

    # Convert to LangGraph state
    state = converter.from_context(context, "I need a tow truck")

    # Verify conversion
    assert state["conversation_id"] == "test-123"
    assert state["turn_count"] == 2
    assert state["flow"] == "towing"
    assert state["intent"] == "request_tow"
    assert state["entities"]["vehicle_type"] == "sedan"

    # Modify state
    state["flow"] = "roadside"
    state["entities"]["vehicle_make"] = "Honda"

    # Convert back
    new_context = converter.to_context(state)

    # Verify conversion back
    assert new_context["conversation_id"] == "test-123"
    assert new_context["flow"] == "roadside"  # Updated value
    assert new_context["vehicle_type"] == "sedan"  # Preserved
    assert new_context["vehicle_make"] == "Honda"  # Added

def test_langgraph_workflow_with_flags_disabled():
    """Test workflow with LangGraph disabled"""
    flags = MockFeatureFlags({"use_langgraph": False})
    hybrid_system = MockHybridSystem()
    existing_detector = MockExistingDetector()

    workflow = LangGraphWorkflow(flags, hybrid_system, existing_detector)

    # Create initial state
    state = ConversationState(
        conversation_id="test-123",
        turn_count=0,
        current_message="I need a tow truck",
        messages=[],
        context={},
        flow="unknown"
    )

    # Invoke workflow
    result = workflow.invoke(state)

    # Verify result still works with fallback to existing system
    assert "response" in result
    assert "tow" in result["response"].lower()
    assert len(result["messages"]) == 1

def test_langgraph_workflow_with_flags_enabled():
    """Test workflow with LangGraph enabled"""
    flags = MockFeatureFlags({"use_langgraph": True})
    hybrid_system = MockHybridSystem()
    existing_detector = MockExistingDetector()

    workflow = LangGraphWorkflow(flags, hybrid_system, existing_detector)

    # Create initial state
    state = ConversationState(
        conversation_id="test-123",
        turn_count=0,
        current_message="I need a tow truck",
        messages=[],
        context={},
        flow="unknown"
    )

    # Invoke workflow
    result = workflow.invoke(state)

    # Verify result includes expected fields
    assert "response" in result
    assert result["turn_count"] == 1
    assert len(result["messages"]) == 2  # User message + assistant response
```

## Phase 4: Logging, Metrics, and Streamlit Integration

### Objectives

- [ ] Create structured logging and metrics system
- [ ] Ensure CPU optimization
- [ ] Integrate with existing Streamlit UI
- [ ] Implement monitoring dashboard

### Implementation Steps

#### 1. Structured Logging and Metrics

```python
# monitoring.py
import logging
import time
import json
import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import threading
import queue

class MetricsCollector:
    """Collect and aggregate performance metrics"""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics = {
            "requests": 0,
            "successful_responses": 0,
            "errors": 0,
            "average_latency": 0.0,
            "negation_detections": 0,
            "context_switches": 0,
            "fallbacks": 0,
            "rule_based_decisions": 0,
            "ml_based_decisions": 0
        }
        self.history = []
        self.lock = threading.Lock()

    def update_request_count(self) -> int:
        """Increment and return request count"""
        with self.lock:
            self.metrics["requests"] += 1
            return self.metrics["requests"]

    def record_response(self, latency: float, state: Dict[str, Any]) -> None:
        """Record response metrics"""
        with self.lock:
            # Update success count
            self.metrics["successful_responses"] += 1

            # Update latency metrics
            total_latency = self.metrics["average_latency"] * (self.metrics["successful_responses"] - 1)
            self.metrics["average_latency"] = (total_latency + latency) / self.metrics["successful_responses"]

            # Update feature-specific metrics
            if state.get("detected_negation", False):
                self.metrics["negation_detections"] += 1

            if state.get("detected_context_switch", False):
                self.metrics["context_switches"] += 1

            if state.get("should_fallback", False):
                self.metrics["fallbacks"] += 1

            # Record which system made the decision
            if "rule_based_decision" in state.get("confidence_scores", {}):
                self.metrics["rule_based_decisions"] += 1
            elif "ml_based_decision" in state.get("confidence_scores", {}):
                self.metrics["ml_based_decisions"] += 1

            # Add to history
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "latency": latency,
                "intent": state.get("intent", "unknown"),
                "flow": state.get("flow", "unknown"),
                "negation": state.get("detected_negation", False),
                "context_switch": state.get("detected_context_switch", False)
            })

            # Trim history if needed
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

    def record_error(self) -> None:
        """Record an error"""
        with self.lock:
            self.metrics["errors"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self.lock:
            metrics_copy = self.metrics.copy()
            metrics_copy["timestamp"] = datetime.now().isoformat()
            return metrics_copy

    def get_history(self) -> list:
        """Get request history"""
        with self.lock:
            return self.history.copy()
```

#### 2. CPU Optimization

```python
# cpu_optimizations.py
import time
from typing import Dict, Any, Optional, List, Tuple
import threading
from functools import lru_cache

class CPUOptimizer:
    """Optimize performance for CPU-only environments"""

    def __init__(self, max_cache_size: int = 1000, cache_ttl: int = 3600):
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl  # Time-to-live in seconds
        self.cache = {}
        self.cache_lock = threading.Lock()

    @lru_cache(maxsize=1024)
    def cached_text_analysis(self, text: str) -> Dict[str, Any]:
        """
        Cached text analysis for frequently repeated queries
        Using lru_cache for simplicity - adequate for CPU usage
        """
        # This would be replaced with actual analysis logic
        time.sleep(0.01)  # Simulate processing time
        return {"result": "cached_analysis", "text_length": len(text)}

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts in a single batch
        More efficient than individual processing
        """
        results = []
        for text in texts:
            # Use cache when possible
            result = self.cached_text_analysis(text)
            results.append(result)
        return results

    def timed_with_timeout(self, func, *args, timeout: float = 1.0, **kwargs) -> Tuple[Any, float]:
        """
        Run a function with a timeout
        Returns result and execution time
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # For CPU optimization, we don't actually stop execution
        # but we can mark it as taking too long
        if execution_time > timeout:
            print(f"Warning: Function took {execution_time:.4f}s, exceeding timeout of {timeout}s")

        return result, execution_time

    def clean_cache(self) -> None:
        """Clean expired cache entries"""
        with self.cache_lock:
            current_time = time.time()
            expired_keys = [
                key for key, (value, timestamp) in self.cache.items()
                if current_time - timestamp > self.cache_ttl
            ]

            for key in expired_keys:
                del self.cache[key]
```

#### 3. Streamlit Integration

```python
# streamlit_integration.py
import streamlit as st
import time
import uuid
from typing import Dict, Any, Optional
from feature_flags import FeatureFlags
from state_converter import StateConverter
from langgraph_workflow import LangGraphWorkflow
from adapters import ExistingDetectionAdapter
from hybrid_detection import HybridDetectionSystem
from mistral_integration import MistralEnhancer
from monitoring import MonitoringSystem, timed_execution

class StreamlitApp:
    """Streamlit app with LangGraph integration"""

    def __init__(self):
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all components"""
        # Feature flags
        if "feature_flags" not in st.session_state:
            st.session_state.feature_flags = FeatureFlags()

        # State converter
        if "state_converter" not in st.session_state:
            st.session_state.state_converter = StateConverter()

        # Monitoring system
        if "monitoring" not in st.session_state:
            st.session_state.monitoring = MonitoringSystem()
            st.session_state.monitoring.start_worker()

        # Existing detector
        if "existing_detector" not in st.session_state:
            st.session_state.existing_detector = ExistingDetectionAdapter()

        # Mistral enhancer (if API key is provided)
        if "mistral_enhancer" not in st.session_state:
            api_key = st.secrets.get("MISTRAL_API_KEY", None)
            st.session_state.mistral_enhancer = MistralEnhancer(api_key)

        # Hybrid detection system
        if "hybrid_system" not in st.session_state:
            st.session_state.hybrid_system = HybridDetectionSystem(
                st.session_state.feature_flags,
                st.session_state.existing_detector,
                st.session_state.mistral_enhancer
            )

        # LangGraph workflow
        if "workflow" not in st.session_state:
            st.session_state.workflow = LangGraphWorkflow(
                st.session_state.feature_flags,
                st.session_state.hybrid_system,
                st.session_state.existing_detector
            )

        # Conversation state
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "context" not in st.session_state:
            st.session_state.context = {
                "conversation_id": st.session_state.conversation_id,
                "turn_count": 0,
                "flow": "unknown"
            }
```

### Phase 4 Testing

```python
# test_phase4.py
import pytest
import threading
import time
from unittest.mock import MagicMock, patch
from monitoring import MetricsCollector, ChatbotLogger, MonitoringSystem, timed_execution
from cpu_optimizations import CPUOptimizer

# Test timed_execution decorator
def test_timed_execution():
    @timed_execution
    def test_function():
        time.sleep(0.01)
        return {"result": "test"}

    result, execution_time = test_function()

    assert "result" in result
    assert result["result"] == "test"
    assert "execution_time" in result
    assert execution_time >= 0.01

# Test metrics collector
def test_metrics_collector():
    collector = MetricsCollector()

    # Test request counting
    request_id = collector.update_request_count()
    assert request_id == 1

    # Test response recording
    collector.record_response(0.5, {"detected_negation": True, "flow": "towing"})

    metrics = collector.get_metrics()
    assert metrics["requests"] == 1
    assert metrics["successful_responses"] == 1
    assert metrics["negation_detections"] == 1
    assert abs(metrics["average_latency"] - 0.5) < 0.001
```

## Phase 5: Final Integration and Deployment

### Objectives

- [ ] Create full system integration tests
- [ ] Implement CI/CD configuration
- [ ] Create deployment scripts
- [ ] Prepare thorough documentation

### Implementation Steps

#### 1. System Integration Tests

```python
# test_integration.py
import pytest
from feature_flags import FeatureFlags
from adapters import ExistingDetectionAdapter
from hybrid_detection import HybridDetectionSystem
from langgraph_workflow import LangGraphWorkflow
from state_converter import StateConverter
from monitoring import MonitoringSystem
from mistral_integration import MistralEnhancer

# Test the complete pipeline with mock components
class TestCompletePipeline:
    @pytest.fixture
    def setup_components(self):
        """Set up all components needed for the integration test"""
        # Create mock API key environment
        import os
        os.environ["MISTRAL_API_KEY"] = "test-key"

        # Set up components
        flags = FeatureFlags()
        flags.enable("use_langgraph")  # Enable LangGraph for testing

        existing_detector = ExistingDetectionAdapter()
        mistral_enhancer = MistralEnhancer()  # Will use mock API key

        hybrid_system = HybridDetectionSystem(
            flags=flags,
            existing_detector=existing_detector,
            mistral_enhancer=mistral_enhancer
        )

        workflow = LangGraphWorkflow(
            flags=flags,
            hybrid_system=hybrid_system,
            existing_detector=existing_detector
        )

        state_converter = StateConverter()
        monitoring = MonitoringSystem()

        return {
            "flags": flags,
            "existing_detector": existing_detector,
            "mistral_enhancer": mistral_enhancer,
            "hybrid_system": hybrid_system,
            "workflow": workflow,
            "state_converter": state_converter,
            "monitoring": monitoring
        }
```

#### 2. CI/CD Configuration

```yaml
# .github/workflows/ci.yml
name: Chatbot CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest --cov=. --cov-report=xml
```

#### 3. Deployment Script

```bash
#!/bin/bash
# deploy.sh

# Set environment
ENV=${1:-"production"}
CONFIG_FILE="config/deployment.${ENV}.yaml"

echo "Deploying chatbot to ${ENV} environment..."

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found"
    exit 1
fi

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run tests
echo "Running tests..."
pytest
```

## Implementation Tracking

### Phase 1: Foundation and Adapter Layer

- [x] Set up development environment
- [x] Implement feature flag system (`feature_flags.py`)
- [x] Create adapter for existing detection methods (`adapters.py`)
- [x] Define LangGraph state interface (`langgraph_state.py`)
- [x] Run Phase 1 tests (`test_phase1.py`)

### Phase 2: Mistral Integration as Enhancement

- [x] Implement Mistral integration (`mistral_integration.py`)
- [x] Create hybrid detection system (`hybrid_detection.py`)
- [x] Configure confidence weighting mechanism
- [x] Run Phase 2 tests (`test_phase2.py`)

### Phase 3: LangGraph Integration for Flow Control

- [ ] Implement LangGraph nodes (`langgraph_nodes.py`)
- [ ] Create state converter (`state_converter.py`)
- [ ] Build LangGraph workflow (`langgraph_workflow.py`)
- [ ] Run Phase 3 tests (`test_phase3.py`)

### Phase 4: Logging, Metrics, and Streamlit Integration

- [ ] Create monitoring system (`monitoring.py`)
- [ ] Implement CPU optimizations (`cpu_optimizations.py`)
- [ ] Integrate with Streamlit UI (`streamlit_integration.py`)
- [ ] Run Phase 4 tests (`test_phase4.py`)

### Phase 5: Final Integration and Deployment

- [ ] Create system integration tests (`test_integration.py`)
- [ ] Implement CI/CD configuration (`.github/workflows/ci.yml`)
- [ ] Create deployment script (`deploy.sh`)
- [ ] Prepare documentation
- [ ] Run Phase 5 tests (`test_phase5.py`)

## Implementation Schedule

1. **Week 1**: Implement Phase 1 (Foundation and Adapter Layer)
2. **Week 2**: Implement Phase 2 (Mistral Integration as Enhancement)
3. **Week 3**: Implement Phase 3 (LangGraph Integration for Flow Control)
4. **Week 4**: Implement Phase 4 (Logging, Metrics, and Streamlit Integration)
5. **Week 5**: Implement Phase 5 (Final Integration and Deployment)
