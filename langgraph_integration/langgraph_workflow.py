from typing import Dict, Any, Optional, Callable, Literal
from langgraph.graph import StateGraph
from langgraph_integration.langgraph_state import ConversationState
from langgraph_integration.feature_flags import FeatureFlags
from langgraph_integration.hybrid_detection import HybridDetectionSystem
from langgraph_integration.adapters import ExistingDetectionAdapter
from langgraph_integration.langgraph_nodes import (
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
        workflow.add_node("generate_response", response_with_deps)

        # Add edges
        workflow.add_edge("context_tracker", "detection")

        # Define conditional routing based on detection results
        def route_based_on_detection(state: ConversationState) -> Literal["negation_handler", "context_switch_handler", "regular_handler"]:
            if state.get("detected_negation", False):
                return "negation_handler"
            elif state.get("detected_context_switch", False):
                return "context_switch_handler"
            else:
                return "regular_handler"
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "detection",
            route_based_on_detection
        )

        # Connect each handler to the response node
        workflow.add_edge("negation_handler", "generate_response")
        workflow.add_edge("context_switch_handler", "generate_response")
        workflow.add_edge("regular_handler", "generate_response")
        
        # Set entry point
        workflow.set_entry_point("context_tracker")

        return workflow.compile()

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