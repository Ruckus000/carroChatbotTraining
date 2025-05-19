#!/usr/bin/env python3
"""
NLU Model Benchmarking Dashboard - Efficient, interactive Streamlit dashboard
for visualizing and analyzing NLU model performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from datetime import datetime

# Import UI and state components
from utils.ui_components import (
    render_metric_card, 
    create_navigation, 
    render_home_page,
    set_page,
    render_metric_with_help,
    render_performance_indicator,
    add_section_help,
    create_dashboard_tour
)
from utils.state_management import (
    initialize_session_state,
    get_selected_model,
    set_selected_model
)

# Import visualization components
from utils.visualization import (
    create_intent_radar_chart,
    create_performance_timeline,
    create_confusion_matrix_heatmap,
    create_error_pattern_sankey,
    create_confidence_distribution
)

# Import interactive components
from utils.interactive import create_error_explorer

# Import page components
from pages.model_comparison import render_model_comparison_page
from pages.error_explorer import render_error_explorer_page
from pages.guided_analysis import render_guided_analysis_page

# Import data processing functions
from utils.data_processing import (
    extract_intent_distributions,
    process_confusion_matrix,
    analyze_errors,
    process_history_data,
    extract_entity_metrics,
    load_available_models,
    load_model_metrics
)

# Import export utilities
from utils.export import create_export_section, download_all_plots

# Set page config for better UI
st.set_page_config(
    page_title="NLU Model Benchmarking",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS file
def load_custom_css():
    """Load custom CSS file for enhanced styling"""
    css_file = os.path.join("assets", "css", "custom.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback to inline CSS if file not found
        st.markdown("""
        <style>
            .main {
                padding: 1rem 1.5rem;
            }
            .stApp {
                margin: 0 auto;
            }
            .metric-card {
                border: 1px solid #f0f0f0;
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                background-color: white;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            .metric-card h3 {
                margin-top: 0;
            }
            .performance-summary {
                padding: 1rem;
                background-color: #f8f8f8;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
            }
            .stTabs {
                background-color: white;
                border-radius: 0.5rem;
                padding: 1rem;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            .icon-text {
                font-size: 1.2em;
                vertical-align: middle;
            }
            .st-bb {
                padding-top: 0.5rem;
            }
            .help-btn {
                float: right;
                margin-right: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)

# Constants
BENCHMARK_DIR = "benchmark_results"
HISTORY_FILE = os.path.join(BENCHMARK_DIR, "metrics_history.csv")

# Define pages for navigation
PAGES = {
    "home": ("🏠", "Home"),
    "results": ("📊", "Latest Results"),
    "history": ("📈", "Performance History"),
    "errors": ("❌", "Error Analysis"),
    "comparison": ("🔍", "Model Comparison"),
    "guided": ("🧭", "Guided Analysis")
}

# Caching for expensive operations
@st.cache_data(ttl=60)  # Cache for 1 minute
def load_available_runs():
    """Load available benchmark runs with efficient caching"""
    metric_files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "metrics_*.json")), reverse=True)
    runs = []

    for file in metric_files:
        # Extract timestamp from filename
        timestamp = os.path.basename(file).replace("metrics_", "").replace(".json", "")
        try:
            # Format timestamp for display
            formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            file_size = os.path.getsize(file) / 1024  # KB

            # Get model_id if available (quick peek without loading full file)
            model_id = "Unknown"
            try:
                with open(file, 'r') as f:
                    for i, line in enumerate(f):
                        if '"model_id":' in line:
                            model_id = line.split('"model_id":')[1].strip().strip('",')
                            break
                        if i > 10:  # Only check first few lines
                            break
            except:
                pass

            runs.append({
                "file": file,
                "timestamp": formatted_time,
                "raw_timestamp": timestamp,
                "model_id": model_id,
                "size": f"{file_size:.1f} KB"
            })
        except Exception:
            # Skip files with invalid naming
            continue

    return runs

@st.cache_data
def load_metrics(file_path):
    """Load metrics file with caching"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics file: {str(e)}")
        return None

@st.cache_data
def load_history():
    """Load metrics history with caching"""
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception as e:
            st.error(f"Error loading history file: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def load_latest_metrics():
    """Load the most recent metrics file"""
    runs = load_available_runs()
    if not runs:
        return {}
    return load_metrics(runs[0]["file"])

def render_header():
    """Render dashboard header with information and controls"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
            <div class="header-container">
                <div>
                    <h1 class="header-title">NLU Model Benchmarking Dashboard</h1>
                    <p class="header-subtitle">Track and analyze model performance metrics</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Check when dashboard was last refreshed
        last_refresh = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"<div style='text-align: right; color: #777;'>Last refreshed: {last_refresh}</div>", unsafe_allow_html=True)
        # Add a refresh button
        if st.button("🔄 Refresh Data"):
            # Clear all cached data
            st.cache_data.clear()
            st.rerun()

def format_class_name(name, max_length=30):
    """Format class names for better display"""
    if len(name) <= max_length:
        return name
    half = max_length // 2 - 2
    return f"{name[:half]}...{name[-half:]}"

def render_confusion_matrix(cm, labels, width=None, height=None):
    """Render a clean, readable confusion matrix"""
    # Scale figure size based on number of classes
    n_classes = len(labels)

    if width is None:
        width = min(12, max(8, n_classes * 0.5))
    if height is None:
        height = min(12, max(8, n_classes * 0.5))

    # Format labels for better display
    display_labels = [format_class_name(label) for label in labels]

    fig, ax = plt.subplots(figsize=(width, height))

    # Normalize if many samples
    if np.sum(cm) > 100:
        im = sns.heatmap(
            cm.astype(float) / cm.sum(axis=1, keepdims=True),
            annot=True,
            fmt='.1%',
            cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax
        )
    else:
        im = sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax
        )

    # Adjust font size based on number of classes
    fontsize = max(8, min(12, 16 - n_classes * 0.5))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize)

    ax.set_ylabel('True Intent')
    ax.set_xlabel('Predicted Intent')
    plt.tight_layout()

    return fig

def render_intent_metrics(metrics):
    """Render intent metrics with enhanced UI components"""
    st.markdown('<div class="dashboard-section performance intent-metrics">', unsafe_allow_html=True)
    
    st.subheader("Intent Classification Performance")
    add_section_help("intent_performance")
    
    if 'intent_metrics' not in metrics:
        st.warning("No intent metrics found in the evaluation results")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    intent_metrics = metrics['intent_metrics']
    
    # Display summary metrics with colored indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_with_help(
            "Accuracy", 
            f"{intent_metrics.get('accuracy', 0):.2f}", 
            metric_key="intent_accuracy"
        )
        render_performance_indicator("Accuracy", intent_metrics.get('accuracy', 0))
    
    with col2:
        render_metric_with_help(
            "Precision", 
            f"{intent_metrics.get('precision', 0):.2f}", 
            metric_key="intent_precision"
        )
        render_performance_indicator("Precision", intent_metrics.get('precision', 0))
    
    with col3:
        render_metric_with_help(
            "Recall", 
            f"{intent_metrics.get('recall', 0):.2f}", 
            metric_key="intent_recall"
        )
        render_performance_indicator("Recall", intent_metrics.get('recall', 0))
    
    with col4:
        render_metric_with_help(
            "F1 Score", 
            f"{intent_metrics.get('f1', 0):.2f}", 
            metric_key="intent_f1"
        )
        render_performance_indicator("F1 Score", intent_metrics.get('f1', 0))
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for best and worst intents
        radar_chart = create_intent_radar_chart(metrics)
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True, key="intent_radar_chart")
    
    with col2:
        # Extract confusion matrix data
        confusion_matrix = metrics.get('intent_metrics', {}).get('confusion_matrix', [])
        intent_labels = metrics.get('intent_metrics', {}).get('labels', [])
        
        # Create enhanced confusion matrix heatmap
        if confusion_matrix and intent_labels:
            cm_data = process_confusion_matrix(confusion_matrix, intent_labels)
            cm_heatmap = create_confusion_matrix_heatmap(confusion_matrix, intent_labels)
            if cm_heatmap:
                st.plotly_chart(cm_heatmap, use_container_width=True, key="confusion_matrix_heatmap")
                
                # Display top confused pairs
                if 'confused_pairs' in cm_data and cm_data['confused_pairs']:
                    with st.expander("Top Confusion Pairs"):
                        for pair in cm_data['confused_pairs'][:5]:
                            st.markdown(f"**{pair['true']}** → **{pair['predicted']}**: {pair['count']} examples ({pair['percentage']:.1%} of '{pair['true']}' examples)")
        else:
            st.warning("No confusion matrix data available")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_entity_metrics(metrics):
    """Render entity metrics with enhanced UI components"""
    st.markdown('<div class="dashboard-section entities entity-metrics">', unsafe_allow_html=True)
    
    st.subheader("Entity Recognition Performance")
    add_section_help("entity_performance")
    
    if 'entity_metrics' not in metrics or not metrics['entity_metrics']:
        st.warning("No entity metrics found in the evaluation results")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    entity_metrics = metrics['entity_metrics']
    
    # Extract metrics
    micro_metrics = entity_metrics.get('micro avg', {})
    macro_metrics = entity_metrics.get('macro avg', {})
    
    # Display summary metrics with colored indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        precision = micro_metrics.get('precision', 0)
        render_metric_with_help(
            "Precision (micro)", 
            f"{precision:.2f}", 
            metric_key="entity_precision"
        )
        render_performance_indicator("Precision", precision)
    
    with col2:
        recall = micro_metrics.get('recall', 0)
        render_metric_with_help(
            "Recall (micro)", 
            f"{recall:.2f}", 
            metric_key="entity_recall"
        )
        render_performance_indicator("Recall", recall)
    
    with col3:
        f1 = micro_metrics.get('f1-score', 0)
        render_metric_with_help(
            "F1 Score (micro)", 
            f"{f1:.2f}", 
            metric_key="entity_f1"
        )
        render_performance_indicator("F1 Score", f1)
    
    with col4:
        support = micro_metrics.get('support', 0)
        render_metric_with_help(
            "Support", 
            f"{int(support)}", 
            metric_key="support"
        )
    
    # Per-entity performance table
    st.subheader("Per-Entity Performance")
    
    # Filter entity metrics to exclude averages
    entity_only_metrics = {k: v for k, v in entity_metrics.items() if k not in ['micro avg', 'macro avg', 'weighted avg']}
    
    if entity_only_metrics:
        # Convert to DataFrame for better display
        entity_df = pd.DataFrame(entity_only_metrics).T
        entity_df.index.name = 'Entity'
        entity_df.reset_index(inplace=True)
        
        # Display as interactive table
        st.dataframe(
            entity_df,
            column_config={
                'Entity': st.column_config.TextColumn("Entity Type"),
                'precision': st.column_config.NumberColumn("Precision", format="%.3f"),
                'recall': st.column_config.NumberColumn("Recall", format="%.3f"),
                'f1-score': st.column_config.NumberColumn("F1 Score", format="%.3f"),
                'support': st.column_config.NumberColumn("Support", format="%d")
            },
            use_container_width=True
        )
    else:
        st.info("No entity-specific metrics available")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_error_analysis(metrics):
    """Render error analysis section with interactive elements"""
    st.markdown('<div class="dashboard-section error-analysis">', unsafe_allow_html=True)
    
    st.subheader("Error Analysis")
    add_section_help("error_analysis")
    
    # Check if we have the necessary data
    if 'detailed_results' not in metrics or not metrics['detailed_results']:
        st.warning("No detailed results found for error analysis")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Extract key data for analysis
    results = metrics['detailed_results']
    error_analysis = analyze_errors(results)
    
    if not error_analysis:
        st.warning("Failed to analyze errors from the results")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Display error summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_metric_with_help(
            "Error Rate", 
            f"{error_analysis['error_rate']:.1%}", 
            metric_key="error_rate"
        )
    
    with col2:
        render_metric_with_help(
            "Misclassifications", 
            f"{error_analysis['error_count']} / {error_analysis['total_examples']}", 
            metric_key="misclassifications"
        )
    
    with col3:
        render_metric_with_help(
            "Avg Error Confidence", 
            f"{error_analysis['avg_error_confidence']:.3f}", 
            metric_key="avg_error_confidence"
        )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sankey diagram for error flows
        sankey_chart = create_error_pattern_sankey(error_analysis)
        if sankey_chart:
            st.plotly_chart(sankey_chart, use_container_width=True, key="error_sankey_chart")
    
    with col2:
        # Confidence distribution
        confidence_chart = create_confidence_distribution(error_analysis)
        if confidence_chart:
            st.plotly_chart(confidence_chart, use_container_width=True, key="confidence_distribution_chart")
    
    # Interactive error explorer
    st.subheader("Interactive Error Explorer")
    create_error_explorer(metrics)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_performance_history():
    """Render performance history page"""
    st.title("Performance History")
    
    # Load metrics history
    history_df = load_history()
    
    if history_df.empty:
        st.warning("No performance history data found")
        return
    
    # Process history data
    processed_history = process_history_data(history_df)
    
    # Create visualization - pass the DataFrame directly
    timeline_fig = create_performance_timeline(history_df)
    
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True, key="performance_timeline_chart")
    
    # Performance table
    st.subheader("Performance History Table")
    
    # Prepare table with formatted metrics
    display_df = history_df.copy()
    
    # Format date for better display
    if 'date' in display_df.columns:
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Format numeric columns as percentages
    for col in ['intent_accuracy', 'intent_f1', 'entity_f1']:
        if col in display_df.columns:
            display_df[col] = display_df[col].map('{:.1%}'.format)
    
    # Rename columns for display
    column_mapping = {
        'date': 'Date',
        'model_id': 'Model',
        'intent_accuracy': 'Intent Accuracy',
        'intent_f1': 'Intent F1',
        'entity_f1': 'Entity F1',
        'num_intents': 'Intents',
        'num_examples': 'Examples'
    }
    
    # Keep only columns we want to display
    display_columns = [col for col in column_mapping.keys() if col in display_df.columns]
    
    # Apply renaming and display
    st.dataframe(
        display_df[display_columns].rename(columns=column_mapping),
        use_container_width=True
    )

def render_latest_results():
    """Render latest results page with export functionality"""
    st.title("Latest Benchmark Results")
    
    # Get all available runs
    runs = load_available_runs()
    
    if not runs:
        st.warning("No benchmark runs found. Please run a benchmark first.")
        return
    
    # Display model selection
    selected_run = st.selectbox(
        "Select benchmark run:",
        options=runs,
        format_func=lambda x: f"{x['model_id']} ({x['timestamp']})"
    )
    
    # Display model info
    st.markdown(f"**Model ID:** {selected_run['model_id']}")
    st.markdown(f"**Run Time:** {selected_run['timestamp']}")
    
    # Load metrics for the selected run
    metrics = load_metrics(selected_run['file'])
    
    if not metrics:
        st.error(f"Failed to load metrics for {selected_run['model_id']}")
        return
    
    # Set the selected model in session state
    set_selected_model(selected_run['model_id'])
    
    # Store visualizations for export
    visualizations = {}
    
    # Create tabs for different sections
    tabs = st.tabs(["Overview", "Intent Classification", "Entity Recognition", "Error Analysis"])
    
    with tabs[0]:
        st.subheader("Performance Summary")
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            intent_accuracy = metrics.get('intent_metrics', {}).get('accuracy', 0)
            render_metric_card("Intent Accuracy", f"{intent_accuracy:.1f}%", "🎯", is_percentage=True)
            render_performance_indicator("Intent Accuracy", intent_accuracy)
        
        with col2:
            intent_f1 = metrics.get('intent_metrics', {}).get('f1', 0)
            render_metric_card("Intent F1", f"{intent_f1:.2f}", "📈")
            render_performance_indicator("Intent F1", intent_f1)
        
        with col3:
            entity_f1 = metrics.get('entity_metrics', {}).get('micro avg', {}).get('f1-score', 0)
            render_metric_card("Entity F1", f"{entity_f1:.2f}", "🏷️")
            render_performance_indicator("Entity F1", entity_f1)
        
        # Add intent performance radar chart
        st.subheader("Intent Performance Overview")
        radar_fig = create_intent_radar_chart(metrics)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
            visualizations['intent_radar'] = radar_fig
    
    with tabs[1]:
        render_intent_metrics(metrics)
    
    with tabs[2]:
        render_entity_metrics(metrics)
    
    with tabs[3]:
        render_error_analysis(metrics)
    
    # Add export section using components
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    create_export_section(metrics, selected_run['model_id'])
    
    if visualizations:
        download_all_plots(visualizations, selected_run['model_id'])
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_home_page_with_data(latest_runs, latest_metrics):
    """Render home page with actual data from the latest run"""
    
    if not latest_runs or not latest_metrics:
        render_home_page()  # Use placeholder data if no real data available
        return
    
    latest_run = latest_runs[0]
    
    # Extract metrics
    intent_metrics = latest_metrics.get("intent_metrics", {})
    entity_metrics = latest_metrics.get("entity_metrics", {})
    
    # Get entity metrics if available
    entity_f1 = entity_metrics.get("micro avg", {}).get("f1-score", 0) if entity_metrics else 0
    
    # Format model ID
    model_id = latest_metrics.get("model_id", latest_run.get("model_id", "Unknown"))
    
    # Render home page with actual data
    st.title("NLU Model Performance Dashboard")
    st.markdown("### Real-time metrics and analysis for NLU model performance")
    
    # Key metrics overview cards in a row
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Current Model", model_id, "🤖")
    with col2:
        render_metric_card("Intent Accuracy", intent_metrics.get("accuracy", 0) * 100, "🎯", is_percentage=True)
    with col3:
        render_metric_card("Entity F1", entity_f1 * 100, "🏷️", is_percentage=True)
    
    # Quick actions section
    st.subheader("Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        st.button("📊 View Latest Results", on_click=set_page, args=("results",))
    with action_col2:
        st.button("📈 Performance History", on_click=set_page, args=("history",))
    with action_col3:
        st.button("❌ Error Analysis", on_click=set_page, args=("errors",))
    
    # Recent performance visualization
    st.subheader("Recent Performance")
    
    # Load and process history data
    history_df = load_history()
    if not history_df.empty:
        # Take last 5 runs
        recent_df = history_df.sort_values('timestamp', ascending=True).tail(5)
        
        # Create recent performance chart
        timeline_chart = create_performance_timeline(recent_df)
        if timeline_chart:
            st.plotly_chart(timeline_chart, use_container_width=True)
    else:
        st.info("No performance history available")

def main():
    """Main function to run the dashboard"""
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS
    load_custom_css()
    
    # Create navigation sidebar
    create_navigation(PAGES)
    
    # Create dashboard tour
    create_dashboard_tour()
    
    # Get all available runs
    latest_runs = load_available_runs()
    
    # Get latest metrics
    latest_metrics = None
    if latest_runs:
        latest_metrics = load_metrics(latest_runs[0]['file'])
    
    # Display the selected page based on session state
    current_page = st.session_state.current_page
    
    if current_page == "home":
        render_home_page_with_data(latest_runs, latest_metrics)
    elif current_page == "results":
        render_latest_results()
    elif current_page == "history":
        render_performance_history()
    elif current_page == "errors":
        render_error_explorer_page()
    elif current_page == "comparison":
        render_model_comparison_page()
    elif current_page == "guided":
        render_guided_analysis_page()

if __name__ == "__main__":
    main() 