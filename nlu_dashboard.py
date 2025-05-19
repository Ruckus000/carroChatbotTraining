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
import time
from datetime import datetime
from functools import lru_cache

# Import UI components
from utils.ui_components import (
    render_metric_card, 
    create_navigation, 
    render_home_page,
    set_page
)

# Set page config for better UI
st.set_page_config(
    page_title="NLU Model Benchmarking",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply clean styling
st.markdown("""
<style>
    .main {
        padding: 1rem 1.5rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 16px;
        line-height: 1.5;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 600;
    }
    .metric-label {
        font-size: 14px;
        font-weight: 400;
        color: #555;
        text-transform: uppercase;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4b9dff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .card.warning {
        border-left: 4px solid #ff9d4b;
    }
    .card.success {
        border-left: 4px solid #4bff9d;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .header-title {
        font-size: 24px;
        font-weight: 600;
        margin: 0;
    }
    .header-subtitle {
        font-size: 16px;
        color: #555;
        margin: 0;
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
    "comparison": ("🔍", "Model Comparison")
}

# Caching for expensive operations
@st.cache_data(ttl=60)  # Cache for 1 minute
def load_available_runs():
    """Load available benchmark runs with efficient caching"""
    # Find all metrics files
    metric_files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "metrics_*.json")), reverse=True)
    runs = []

    for file in metric_files:
        # Extract timestamp from filename
        timestamp = os.path.basename(file).replace("metrics_", "").replace(".json", "")
        try:
            # Format timestamp for display
            formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

            # Get file size for information
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
        except Exception as e:
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

def custom_metric_card(label, value, delta=None, help_text=None, color="blue"):
    """Create a custom, visually appealing metric card"""
    # Determine color based on value for visual feedback
    if color == "auto":
        if value > 0.9:
            color = "green"
        elif value > 0.7:
            color = "blue"
        elif value > 0.5:
            color = "orange"
        else:
            color = "red"

    # Set color code
    color_code = {
        "blue": "#4b9dff",
        "green": "#4bff9d",
        "orange": "#ff9d4b",
        "red": "#ff4b4b"
    }.get(color, "#4b9dff")

    # Format value
    if isinstance(value, float):
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = str(value)

    # Create HTML for metric card
    html = f"""
    <div style="border-left: 4px solid {color_code}; padding: 10px 15px; border-radius: 5px; background: #f8f9fa; margin-bottom: 10px;">
        <div style="font-size: 14px; color: #555; font-weight: 500;">{label}</div>
        <div style="font-size: 28px; font-weight: 600; margin: 5px 0;">{formatted_value}</div>
    """

    if delta is not None:
        delta_color = "green" if delta >= 0 else "red"
        delta_symbol = "▲" if delta >= 0 else "▼"
        html += f'<div style="font-size: 14px; color: {"#4bff9d" if delta_color == "green" else "#ff4b4b"};">{delta_symbol} {abs(delta):.4f}</div>'

    if help_text:
        html += f'<div style="font-size: 12px; color: #777; margin-top: 5px;">{help_text}</div>'

    html += "</div>"

    return html

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

    # Determine where to truncate
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
    """Render intent classification metrics"""
    st.subheader("Intent Classification Performance")

    intent_metrics = metrics.get("intent_metrics", {})
    # Extract key metrics
    accuracy = intent_metrics.get("accuracy", 0)
    f1 = intent_metrics.get("f1", 0)
    precision = intent_metrics.get("precision", 0)
    recall = intent_metrics.get("recall", 0)
    
    # Create metric cards in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Accuracy", accuracy, "🎯", is_percentage=True)
    with col2:
        render_metric_card("F1 Score", f1, "⚖️")
    with col3:
        render_metric_card("Precision", precision, "📌")
    with col4:
        render_metric_card("Recall", recall, "🔍")

def render_entity_metrics(metrics):
    """Render entity recognition metrics"""
    entity_metrics = metrics.get("entity_metrics", {})
    if not entity_metrics:
        st.warning("No entity metrics available in this benchmark")
        return

    st.subheader("Entity Recognition Performance")

    # Get aggregate metrics if available
    agg_metrics = entity_metrics.get("micro avg", {})
    f1 = agg_metrics.get("f1-score", 0)
    precision = agg_metrics.get("precision", 0)
    recall = agg_metrics.get("recall", 0)
    
    # Create metric cards in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Entity F1", f1, "🏷️")
    with col2:
        render_metric_card("Entity Precision", precision, "📌")
    with col3:
        render_metric_card("Entity Recall", recall, "🔍")

def render_class_performance(class_report, class_type="intent"):
    """Render per-class performance metrics"""
    # Create a dataframe from the class report
    df = pd.DataFrame(class_report).T
    
    # Filter out aggregate metrics
    df = df[~df.index.isin(['micro avg', 'macro avg', 'weighted avg'])]
    
    # Calculate support percentage
    total_support = df['support'].sum()
    df['support_pct'] = df['support'] / total_support * 100
    
    # Sort by F1 score descending
    df = df.sort_values(by='f1-score', ascending=False)
    
    # Format for display
    df_display = df.copy()
    for col in ['precision', 'recall', 'f1-score']:
        df_display[col] = df_display[col].map(lambda x: f"{x:.4f}")
    df_display['support_pct'] = df_display['support_pct'].map(lambda x: f"{x:.1f}%")
    
    # Show the table
    st.dataframe(
        df_display,
        column_config={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1 Score",
            "support": "Examples",
            "support_pct": "% of Total"
        },
        height=min(35 * len(df) + 38, 400)
    )
    
    # Create a bar chart of F1 scores
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
    
    # Sort for the chart
    df_plot = df.sort_values(by='f1-score')
    
    # Plot horizontal bars
    ax.barh(df_plot.index, df_plot['f1-score'], color='#4b9dff')
    
    # Set limits and labels
    ax.set_xlim(0, 1)
    ax.set_xlabel('F1 Score')
    ax.set_ylabel(f'{class_type.capitalize()} Class')
    ax.set_title(f'{class_type.capitalize()} Classification Performance (F1)')
    
    # Improve readability
    plt.tight_layout()
    
    # Show the chart
    st.pyplot(fig)

def format_entities(entities):
    """Format entities for display"""
    if not entities:
        return "-"
    
    entity_strs = []
    for entity in entities:
        entity_type = entity.get("entity", "")
        value = entity.get("value", "")
        entity_strs.append(f"{entity_type}: {value}")
    
    return ", ".join(entity_strs)

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
    st.image("assets/nlu_logo.txt", width=100)
    st.title("NLU Model Performance Dashboard")
    
    # Key metrics overview cards in a row
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Current Model", model_id, "🤖")
    with col2:
        render_metric_card("Intent Accuracy", intent_metrics.get("accuracy", 0), "🎯", is_percentage=True)
    with col3:
        render_metric_card("Entity F1", entity_f1, "🏷️")
    
    # Quick actions section
    st.subheader("Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        st.button("📊 View Latest Results", on_click=set_page, args=("results",))
    with action_col2:
        st.button("📈 Performance History", on_click=set_page, args=("history",))
    with action_col3:
        st.button("❌ Error Analysis", on_click=set_page, args=("errors",))

def main():
    """Main function to run the dashboard"""
    
    # Create navigation sidebar
    create_navigation(PAGES)
    
    # Render the appropriate page based on session state
    current_page = st.session_state.get("current_page", "home")
    
    # Load data that might be needed across multiple pages
    available_runs = load_available_runs()
    history_df = load_history()
    
    # Load the latest metrics file if available
    latest_metrics = None
    if available_runs:
        latest_metrics = load_metrics(available_runs[0]["file"])
    
    # Render the correct page
    if current_page == "home":
        render_home_page_with_data(available_runs, latest_metrics)
        
    elif current_page == "results":
        render_header()
        
        st.subheader("Latest Benchmark Results")
        
        if not available_runs:
            st.warning("No benchmark results found. Please run evaluations first.")
            return
            
        run_select = st.selectbox(
            "Select benchmark run:",
            options=[f"{run['model_id']} ({run['timestamp']})" for run in available_runs],
            index=0
        )
        
        # Find the selected run
        selected_index = [f"{run['model_id']} ({run['timestamp']})" for run in available_runs].index(run_select)
        selected_run = available_runs[selected_index]
        
        # Load metrics for the selected run
        metrics = load_metrics(selected_run["file"])
        
        if not metrics:
            st.error(f"Could not load metrics for the selected run: {selected_run['file']}")
            return
            
        # Show metadata
        st.markdown(f"**Model ID:** {metrics.get('model_id', 'Unknown')}")
        st.markdown(f"**Timestamp:** {selected_run['timestamp']}")
        
        # Show tabs for different metric types
        tabs = st.tabs(["Intent Metrics", "Entity Metrics", "Confusion Matrix", "Error Analysis"])
        
        # Intent Metrics Tab
        with tabs[0]:
            render_intent_metrics(metrics)
            
            # Per-class metrics
            st.markdown("### Per-Intent Performance")
            if "intent_metrics" in metrics and "per_class_report" in metrics["intent_metrics"]:
                render_class_performance(metrics["intent_metrics"]["per_class_report"], "intent")
            else:
                st.warning("No per-intent metrics available.")
                
        # Entity Metrics Tab
        with tabs[1]:
            render_entity_metrics(metrics)
            
            # Per-entity metrics
            st.markdown("### Per-Entity Performance")
            if "entity_metrics" in metrics and "report" in metrics["entity_metrics"]:
                # Filter out the averages
                entity_report = {k: v for k, v in metrics["entity_metrics"]["report"].items() 
                               if k not in ["micro avg", "macro avg", "weighted avg"]}
                if entity_report:
                    render_class_performance(entity_report, "entity")
                else:
                    st.warning("No per-entity metrics available.")
            else:
                st.warning("No per-entity metrics available.")
                
        # Confusion Matrix Tab
        with tabs[2]:
            st.markdown("### Intent Confusion Matrix")
            if "intent_metrics" in metrics and "confusion_matrix" in metrics["intent_metrics"]:
                cm = np.array(metrics["intent_metrics"]["confusion_matrix"])
                labels = metrics["intent_metrics"].get("labels", [])
                
                if len(cm) > 0 and len(labels) > 0:
                    fig = render_confusion_matrix(cm, labels)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough data to generate confusion matrix.")
            else:
                st.warning("No confusion matrix data available.")
                
        # Error Analysis Tab
        with tabs[3]:
            st.markdown("### Error Analysis")
            if "detailed_results" in metrics:
                # Filter to errors only
                errors = [r for r in metrics["detailed_results"] if not r.get("intent_correct", True)]
                
                if errors:
                    st.markdown(f"Found **{len(errors)}** errors out of **{len(metrics['detailed_results'])}** examples.")
                    
                    # Show errors in an expander
                    for i, error in enumerate(errors):
                        with st.expander(f"Error {i+1}: {error.get('text', 'No text')}"):
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown(f"**True Intent:** {error.get('true_intent', 'Unknown')}")
                                st.markdown(f"**True Entities:** {format_entities(error.get('true_entities', []))}")
                            with cols[1]:
                                st.markdown(f"**Predicted Intent:** {error.get('pred_intent', 'Unknown')}")
                                st.markdown(f"**Confidence:** {error.get('confidence', 0):.4f}")
                else:
                    st.success("No errors found in this benchmark!")
            else:
                st.warning("No detailed results available for error analysis.")
    
    elif current_page == "history":
        render_header()
        
        st.subheader("Performance History")
        
        if history_df.empty:
            st.warning("No history data available. Please run more evaluations to build history.")
            return
            
        st.markdown("### Performance Trends")
        
        # Create timeline plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ensure timestamp is in datetime format
        if 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Plot intent metrics
            if 'intent_accuracy' in history_df.columns:
                ax.plot(history_df['timestamp'], history_df['intent_accuracy'], 
                      marker='o', label='Intent Accuracy', color='#4b9dff')
                
            if 'intent_f1' in history_df.columns:
                ax.plot(history_df['timestamp'], history_df['intent_f1'], 
                      marker='s', label='Intent F1', color='#4bff9d')
                
            # Plot entity metrics if available
            if 'entity_f1' in history_df.columns:
                ax.plot(history_df['timestamp'], history_df['entity_f1'], 
                      marker='^', label='Entity F1', color='#ff9d4b')
                
            # Set labels and limits
            ax.set_xlabel('Date')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis
            fig.autofmt_xdate()
            
            st.pyplot(fig)
        else:
            st.warning("Invalid history data format. Missing timestamp column.")
            
        # Show the history data table
        st.markdown("### History Data")
        st.dataframe(history_df)
    
    elif current_page == "errors":
        render_header()
        
        st.subheader("Error Analysis")
        
        if not available_runs:
            st.warning("No benchmark results found. Please run evaluations first.")
            return
            
        run_select = st.selectbox(
            "Select benchmark run:",
            options=[f"{run['model_id']} ({run['timestamp']})" for run in available_runs],
            index=0
        )
        
        # Find the selected run
        selected_index = [f"{run['model_id']} ({run['timestamp']})" for run in available_runs].index(run_select)
        selected_run = available_runs[selected_index]
        
        # Load metrics for the selected run
        metrics = load_metrics(selected_run["file"])
        
        if not metrics or "detailed_results" not in metrics:
            st.error("No detailed results available for error analysis.")
            return
            
        # Filter to errors only
        errors = [r for r in metrics["detailed_results"] if not r.get("intent_correct", True)]
        
        if not errors:
            st.success("No errors found in this benchmark!")
            return
            
        st.markdown(f"Found **{len(errors)}** errors out of **{len(metrics['detailed_results'])}** examples.")
        
        # Group errors by predicted/true intent pair
        error_groups = {}
        for error in errors:
            key = f"{error.get('true_intent', 'Unknown')} → {error.get('pred_intent', 'Unknown')}"
            if key not in error_groups:
                error_groups[key] = []
            error_groups[key].append(error)
            
        # Sort groups by count
        sorted_groups = sorted(error_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Show error pattern breakdown
        st.markdown("### Error Patterns")
        
        # Create a bar chart of error patterns
        fig, ax = plt.subplots(figsize=(10, min(max(6, len(sorted_groups) * 0.3), 12)))
        
        # Get data for plot (up to top 15 patterns)
        labels = [group[0] for group in sorted_groups[:15]]
        counts = [len(group[1]) for group in sorted_groups[:15]]
        
        # Sort for the chart (ascending for horizontal bars)
        sort_idx = np.argsort(counts)
        labels = [labels[i] for i in sort_idx]
        counts = [counts[i] for i in sort_idx]
        
        # Plot horizontal bars
        ax.barh(labels, counts, color='#ff9d4b')
        
        # Set labels
        ax.set_xlabel('Number of Errors')
        ax.set_ylabel('Error Pattern')
        ax.set_title('Most Common Error Patterns')
        
        # Improve readability
        plt.tight_layout()
        
        # Show the chart
        st.pyplot(fig)
        
        # Show errors grouped by pattern
        st.markdown("### Errors by Pattern")
        
        for pattern, errors in sorted_groups:
            with st.expander(f"{pattern} ({len(errors)} errors)"):
                for i, error in enumerate(errors):
                    st.markdown(f"**{i+1}. {error.get('text', 'No text')}**")
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**True Intent:** {error.get('true_intent', 'Unknown')}")
                        st.markdown(f"**True Entities:** {format_entities(error.get('true_entities', []))}")
                    with cols[1]:
                        st.markdown(f"**Predicted Intent:** {error.get('pred_intent', 'Unknown')}")
                        st.markdown(f"**Confidence:** {error.get('confidence', 0):.4f}")
                    st.markdown("---")
    
    elif current_page == "comparison":
        render_header()
        
        st.subheader("Model Comparison")
        
        if len(available_runs) < 2:
            st.warning("At least two benchmark runs are needed for comparison. Please run more evaluations.")
            return
            
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Select Base Model")
            base_select = st.selectbox(
                "Base model:",
                options=[f"{run['model_id']} ({run['timestamp']})" for run in available_runs],
                index=1,
                key="base_model"
            )
            
        with col2:
            st.markdown("### Select Comparison Model")
            comp_select = st.selectbox(
                "Comparison model:",
                options=[f"{run['model_id']} ({run['timestamp']})" for run in available_runs],
                index=0,
                key="comp_model"
            )
            
        # Get indices
        base_idx = [f"{run['model_id']} ({run['timestamp']})" for run in available_runs].index(base_select)
        comp_idx = [f"{run['model_id']} ({run['timestamp']})" for run in available_runs].index(comp_select)
        
        if base_idx == comp_idx:
            st.warning("Please select different models to compare.")
            return
            
        # Load metrics
        base_metrics = load_metrics(available_runs[base_idx]["file"])
        comp_metrics = load_metrics(available_runs[comp_idx]["file"])
        
        if not base_metrics or not comp_metrics:
            st.error("Could not load metrics for one or both selected models.")
            return
            
        # Display comparison
        st.markdown("### Overall Metrics Comparison")
        
        # Intent metrics
        base_intent = base_metrics.get("intent_metrics", {})
        comp_intent = comp_metrics.get("intent_metrics", {})
        
        # Intent metrics comparison
        cols = st.columns(4)
        metrics_to_compare = [
            ("Accuracy", "accuracy", "🎯"),
            ("F1 Score", "f1", "⚖️"),
            ("Precision", "precision", "📌"),
            ("Recall", "recall", "🔍")
        ]
        
        for i, (label, key, icon) in enumerate(metrics_to_compare):
            with cols[i]:
                base_val = base_intent.get(key, 0)
                comp_val = comp_intent.get(key, 0)
                delta = comp_val - base_val
                
                # Use updated metric card with delta
                render_metric_card(label, comp_val, icon, delta=delta)
        
        # Entity metrics if available
        if "entity_metrics" in base_metrics and "entity_metrics" in comp_metrics:
            st.markdown("### Entity Metrics Comparison")
            
            base_entity = base_metrics.get("entity_metrics", {}).get("micro avg", {})
            comp_entity = comp_metrics.get("entity_metrics", {}).get("micro avg", {})
            
            cols = st.columns(3)
            entity_metrics_to_compare = [
                ("F1 Score", "f1-score", "🏷️"),
                ("Precision", "precision", "📌"),
                ("Recall", "recall", "🔍")
            ]
            
            for i, (label, key, icon) in enumerate(entity_metrics_to_compare):
                with cols[i]:
                    base_val = base_entity.get(key, 0)
                    comp_val = comp_entity.get(key, 0)
                    delta = comp_val - base_val
                    
                    # Use updated metric card with delta
                    render_metric_card(f"Entity {label}", comp_val, icon, delta=delta)

if __name__ == "__main__":
    main() 