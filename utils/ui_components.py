"""
UI Components - Reusable UI elements for the NLU Benchmarking Dashboard
"""

import streamlit as st
import os

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