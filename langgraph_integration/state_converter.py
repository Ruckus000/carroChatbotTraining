from typing import Dict, Any, Optional
from langgraph_integration.langgraph_state import ConversationState

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