"""
Utility functions for managing Streamlit session state.
"""
import streamlit as st
from typing import Dict, List, Any, Optional

def initialize_session_state():
    """
    Initialize all session state variables if they don't exist.
    
    This ensures that all required state variables are available 
    throughout the application.
    """
    # Support for both real Streamlit and testing with dict
    state = st.session_state if hasattr(st, 'session_state') else st.session_state

    # Page navigation state
    if "current_page" not in state:
        state["current_page"] = "home"
    
    # Model selection
    if "selected_model_id" not in state:
        state["selected_model_id"] = None
    
    # Filter states
    if "intent_filter" not in state:
        state["intent_filter"] = "all"
    
    if "entity_filter" not in state:
        state["entity_filter"] = "all"
    
    # Tour state
    if "tour_step" not in state:
        state["tour_step"] = 0
    
    # Expanded sections state
    if "expanded_sections" not in state:
        state["expanded_sections"] = set()


def set_page(page_name: str):
    """
    Set the current page in the session state.
    
    Args:
        page_name: Name of the page to navigate to
    """
    state = st.session_state if hasattr(st, 'session_state') else st.session_state
    state["current_page"] = page_name


def toggle_section_expanded(section_id: str):
    """
    Toggle whether a section is expanded or collapsed.
    
    Args:
        section_id: Unique identifier for the section
    """
    state = st.session_state if hasattr(st, 'session_state') else st.session_state
    
    if "expanded_sections" not in state:
        state["expanded_sections"] = set()
        
    if section_id in state["expanded_sections"]:
        state["expanded_sections"].remove(section_id)
    else:
        state["expanded_sections"].add(section_id)


def is_section_expanded(section_id: str) -> bool:
    """
    Check if a section is expanded.
    
    Args:
        section_id: Unique identifier for the section
        
    Returns:
        True if the section is expanded, False otherwise
    """
    state = st.session_state if hasattr(st, 'session_state') else st.session_state
    
    if "expanded_sections" not in state:
        return False
        
    return section_id in state["expanded_sections"]


def set_selected_model(model_id: str):
    """
    Set the currently selected model ID.
    
    Args:
        model_id: ID of the selected model
    """
    state = st.session_state if hasattr(st, 'session_state') else st.session_state
    state["selected_model_id"] = model_id


def get_selected_model() -> Optional[str]:
    """
    Get the currently selected model ID.
    
    Returns:
        The ID of the selected model or None if no model is selected
    """
    state = st.session_state if hasattr(st, 'session_state') else st.session_state
    return state.get("selected_model_id")


def set_filter(filter_type: str, value: str):
    """
    Set a filter value in the session state.
    
    Args:
        filter_type: Type of filter (e.g., 'intent', 'entity')
        value: Value to set for the filter
    """
    state = st.session_state if hasattr(st, 'session_state') else st.session_state
    
    filter_key = f"{filter_type}_filter"
    state[filter_key] = value


def get_filter(filter_type: str) -> str:
    """
    Get a filter value from the session state.
    
    Args:
        filter_type: Type of filter (e.g., 'intent', 'entity')
        
    Returns:
        Current value of the filter
    """
    state = st.session_state if hasattr(st, 'session_state') else st.session_state
    
    filter_key = f"{filter_type}_filter"
    return state.get(filter_key, "all")  # Default value 