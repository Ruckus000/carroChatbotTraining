from typing import Dict, Any, Optional
from langgraph_integration.langgraph_state import ConversationState
from langgraph_integration.adapters import ExistingDetectionAdapter
from langgraph_integration.hybrid_detection import HybridDetectionSystem

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
    flow = new_state.get("flow", "unknown")

    # Detect negation
    negation_result = hybrid_system.detect_negation(text, context)
    new_state["detected_negation"] = negation_result.get("is_negation", False)

    # Detect context switch
    switch_result = hybrid_system.detect_context_switch(text, context)
    new_state["detected_context_switch"] = switch_result.get("has_context_switch", False)

    # Analyze intent if not handling a negation or context switch
    if not new_state["detected_negation"] and not new_state["detected_context_switch"]:
        intent_result = hybrid_system.analyze_intent(text, flow, context)
        new_state["intent"] = intent_result.get("intent", "unknown")
    
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
    new_state["needs_clarification"] = True

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

    # Preserve entities from previous context if flag is enabled
    new_state["context"]["previous_entities"] = context.get("entities", {})

    return new_state

def regular_handler_node(state: ConversationState) -> ConversationState:
    """Handle regular (non-negation, non-context-switch) requests"""
    new_state = dict(state)

    # Process using existing logic (simplified here)
    if "context" not in new_state:
        new_state["context"] = {}
    
    # Update context with current intent
    if "intent" in new_state:
        new_state["context"]["last_intent"] = new_state["intent"]
    
    # Check if clarification is needed based on confidence
    if "confidence_scores" in new_state and "intent" in new_state["confidence_scores"]:
        if new_state["confidence_scores"]["intent"] < 0.5:
            new_state["needs_clarification"] = True
            new_state["context"]["requires_clarification"] = True

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
    if "response" in result:
        new_state["response"] = result["response"]
    else:
        new_state["response"] = "I'm not sure how to respond to that."

    # Add to message history
    if "messages" not in new_state:
        new_state["messages"] = []

    new_state["messages"].append({
        "role": "assistant",
        "content": new_state["response"]
    })

    return new_state 