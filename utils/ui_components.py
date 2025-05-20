"""
UI Components - Reusable UI elements for the NLU Benchmarking Dashboard
"""

import streamlit as st
import os
from utils.help_content import get_metric_help, get_section_help

def render_metric_card(title, value, icon=None, is_percentage=False, delta=None, help_text=None):
    """
    Render a metric card with title, value, and optional icon/delta
    
    Args:
        title: Title of the metric
        value: Value to display
        icon: Optional emoji or icon to display
        is_percentage: Whether to format value as percentage
        delta: Optional delta/change to display
        help_text: Optional help text for the metric
    """
    # Format the value
    if is_percentage and isinstance(value, (int, float)):
        formatted_value = f"{value:.1f}%"
    elif isinstance(value, float):
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = str(value)
    
    # Create the container
    container = st.container()
    with container:
        # Create columns for icon and value
        if icon:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"<div style='font-size:2.5rem; text-align:center;'>{icon}</div>", unsafe_allow_html=True)
            content_col = col2
        else:
            content_col = st
        
        # Display the content
        with content_col:
            st.markdown(f"<div style='font-size:1.1rem; font-weight:400; color:#555;'>{title}</div>", unsafe_allow_html=True)
            
            # Display value
            delta_html = ""
            if delta is not None:
                delta_color = "green" if delta >= 0 else "red"
                delta_symbol = "▲" if delta >= 0 else "▼"
                delta_html = f'<span style="font-size:1rem; color:{"#4bff9d" if delta_color=="green" else "#ff4b4b"}">{delta_symbol} {abs(delta):.2f}</span>'
            
            st.markdown(f"<div style='font-size:2rem; font-weight:600;'>{formatted_value} {delta_html}</div>", unsafe_allow_html=True)
            
            # Display help text if provided
            if help_text:
                st.markdown(f"<div style='font-size:0.8rem; color:#777; margin-top:0.5rem;'>{help_text}</div>", unsafe_allow_html=True)
    
    return container

def set_page(page_name):
    """
    Set the current page in session state
    
    Args:
        page_name: Name of the page to set
    """
    st.session_state.current_page = page_name

def create_navigation(pages_dict):
    """
    Create navigation sidebar with visual indicators for active page
    
    Args:
        pages_dict: Dictionary of pages {name: (icon, label)}
    """
    # Store current page in session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    
    # Create sidebar navigation
    with st.sidebar:
        # Use text instead of image for the logo
        st.markdown("<h1 style='text-align: center; font-size: 1.5rem;'>🤖 NLU</h1>", unsafe_allow_html=True)
        st.title("Navigation")
        
        # Navigation buttons with visual indicators for active page
        for page, (icon, label) in pages_dict.items():
            if st.session_state.current_page == page:
                st.sidebar.button(
                    f"{icon} {label} ←",
                    key=f"nav_{page}",
                    on_click=set_page,
                    args=(page,),
                    use_container_width=True,
                )
            else:
                st.sidebar.button(
                    f"{icon} {label}",
                    key=f"nav_{page}",
                    on_click=set_page,
                    args=(page,),
                    use_container_width=True,
                )
                
def render_home_page():
    """
    Render a welcoming home page with key metrics and quick actions
    """
    # Use text instead of image for the logo
    st.markdown("<h1 style='font-size: 2.5rem;'>🤖 NLU Model Performance Dashboard</h1>", unsafe_allow_html=True)
    
    # Placeholder data - would normally be replaced with actual metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Current Model", "model_v2.3", "🤖")
    with col2:
        render_metric_card("Intent Accuracy", "94.2%", "🎯", is_percentage=True)
    with col3:
        render_metric_card("Entity F1", "87.5%", "🏷️", is_percentage=True)
    
    # Quick actions section
    st.subheader("Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        st.button("📊 View Latest Results", on_click=set_page, args=("results",))
    with action_col2:
        st.button("📈 Performance History", on_click=set_page, args=("history",))
    with action_col3:
        st.button("❌ Error Analysis", on_click=set_page, args=("errors",)) 

# New Phase 4 components

def render_metric_with_help(title, value, help_text=None, metric_key=None):
    """
    Render a metric with a help tooltip
    
    Args:
        title: Title of the metric
        value: Value to display
        help_text: Custom help text (overrides metric_key lookup)
        metric_key: Key to look up help text in METRIC_HELP
    """
    if not help_text and metric_key:
        help_text = get_metric_help(metric_key)
    
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.metric(label=title, value=value)
    with col2:
        with st.expander("?"):
            st.markdown(help_text)
    
    return col1

def get_performance_color(value):
    """
    Returns a color based on performance value
    
    Args:
        value: Performance value (0-1)
    
    Returns:
        CSS color string
    """
    if value >= 0.9:
        return "#4bff9d"  # bright green
    elif value >= 0.8:
        return "#a5f2a5"  # light green
    elif value >= 0.7:
        return "#ffcb3c"  # yellow/orange
    else:
        return "#ff4b4b"  # red

def render_performance_indicator(label, value):
    """
    Render a color-coded performance indicator
    
    Args:
        label: Name of the metric
        value: Value of the metric (0-1)
    """
    color = get_performance_color(value)
    st.markdown(f"""
    <div style="display:flex; align-items:center; margin-bottom:10px;">
        <div style="width:120px;">{label}:</div>
        <div style="width:50px; text-align:right;">{value:.2f}</div>
        <div style="flex-grow:1; margin-left:10px;">
            <div style="width:{value*100}%; height:8px; background-color:{color}; border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def add_section_help(section_key):
    """
    Add an interpretation guide for a dashboard section
    
    Args:
        section_key: Key to look up in SECTION_HELP
    """
    help_text = get_section_help(section_key)
    with st.expander("💡 Interpretation Guide"):
        st.markdown(help_text)

def create_dashboard_tour():
    """
    Create an interactive tour of the dashboard features
    """
    if "tour_step" not in st.session_state:
        st.session_state.tour_step = 0

    # Tour button in sidebar
    with st.sidebar:
        if st.button("Dashboard Tour"):
            st.session_state.tour_step = 1

    # Tour steps
    if st.session_state.tour_step > 0:
        # Create overlay with tour information
        tour_steps = [
            {"title": "Welcome to the Tour", "text": "This tour will walk you through the dashboard features.", "element": "body"},
            {"title": "Performance Summary", "text": "This section shows the key performance metrics for your model.", "element": ".performance-summary"},
            {"title": "Navigation", "text": "Use the sidebar to navigate between different views.", "element": ".sidebar"},
            {"title": "Intent Metrics", "text": "This section shows detailed intent classification performance.", "element": ".intent-metrics"},
            {"title": "Entity Metrics", "text": "This section shows detailed entity recognition performance.", "element": ".entity-metrics"},
            {"title": "Error Analysis", "text": "Explore model errors to understand where improvements are needed.", "element": ".error-analysis"},
        ]

        current_step = tour_steps[st.session_state.tour_step - 1]

        # Show tour dialog
        with st.sidebar:
            st.markdown(f"## {current_step['title']}")
            st.markdown(current_step['text'])

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous") and st.session_state.tour_step > 1:
                    st.session_state.tour_step -= 1
            with col2:
                if st.session_state.tour_step < len(tour_steps):
                    if st.button("Next"):
                        st.session_state.tour_step += 1
                else:
                    if st.button("Finish"):
                        st.session_state.tour_step = 0 