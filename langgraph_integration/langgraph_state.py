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