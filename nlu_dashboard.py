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

# Import visualization components
from utils.visualization import (
    create_intent_radar_chart,
    create_performance_timeline,
    create_confusion_matrix_heatmap,
    create_error_pattern_sankey,
    create_confidence_distribution
)

# Import data processing functions
from utils.data_processing import (
    extract_intent_distributions,
    process_confusion_matrix,
    analyze_errors,
    process_history_data,
    extract_entity_metrics
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

@st.cache_data
def load_latest_metrics():
    """Load the most recent metrics file"""
    runs = load_available_runs()
    if not runs:
        return {}
        
    # Get the most recent run (first in the list, as they're sorted by timestamp)
    latest_run = runs[0]
    
    # Load metrics for the latest run
    return load_metrics(latest_run["file"])

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
    """Render intent classification metrics with enhanced visualizations"""
    st.header("Intent Classification Performance")
    
    # Extract intent metrics
    intent_metrics = metrics.get('intent_metrics', {})
    if not intent_metrics:
        st.warning("No intent metrics available")
        return
    
    # Summary metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            "Accuracy", 
            intent_metrics.get('accuracy', 0), 
            "🎯", 
            is_percentage=True,
            help_text="Percentage of examples where predicted intent matches true intent"
        )
    with col2:
        render_metric_card(
            "Precision", 
            intent_metrics.get('precision', 0), 
            "📌", 
            is_percentage=True,
            help_text="Ratio of correctly predicted intents to all predicted intents"
        )
    with col3:
        render_metric_card(
            "Recall", 
            intent_metrics.get('recall', 0), 
            "🔍", 
            is_percentage=True,
            help_text="Ratio of correctly predicted intents to all actual intents"
        )
    with col4:
        render_metric_card(
            "F1 Score", 
            intent_metrics.get('f1', 0), 
            "⚖️", 
            is_percentage=True,
            help_text="Harmonic mean of precision and recall"
        )
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for best and worst intents
        radar_chart = create_intent_radar_chart(metrics)
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)
    
    with col2:
        # Extract confusion matrix data
        confusion_matrix = metrics.get('intent_metrics', {}).get('confusion_matrix', [])
        intent_labels = metrics.get('intent_metrics', {}).get('labels', [])
        
        # Create enhanced confusion matrix heatmap
        if confusion_matrix and intent_labels:
            cm_data = process_confusion_matrix(confusion_matrix, intent_labels)
            cm_heatmap = create_confusion_matrix_heatmap(confusion_matrix, intent_labels)
            if cm_heatmap:
                st.plotly_chart(cm_heatmap, use_container_width=True)
                
                # Display top confused pairs
                if 'confused_pairs' in cm_data and cm_data['confused_pairs']:
                    with st.expander("Top Confusion Pairs"):
                        for pair in cm_data['confused_pairs'][:5]:
                            st.markdown(f"**{pair['true']}** → **{pair['predicted']}**: {pair['count']} examples ({pair['percentage']:.1%} of '{pair['true']}' examples)")
        else:
            st.warning("No confusion matrix data available")

def render_entity_metrics(metrics):
    """Render entity recognition metrics with enhanced visualizations"""
    st.header("Entity Recognition Performance")
    
    # Extract entity metrics
    entity_metrics = extract_entity_metrics(metrics.get('entity_metrics', {}))
    if not entity_metrics:
        st.warning("No entity metrics available")
        return
    
    # Get aggregate metrics
    micro_avg = entity_metrics.get('micro avg', {})
    
    # Summary metrics in cards
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            "Entity Precision", 
            micro_avg.get('precision', 0), 
            "📌", 
            is_percentage=True,
            help_text="How many predicted entities are correct"
        )
    with col2:
        render_metric_card(
            "Entity Recall", 
            micro_avg.get('recall', 0), 
            "🔍", 
            is_percentage=True,
            help_text="How many actual entities were found"
        )
    with col3:
        render_metric_card(
            "Entity F1", 
            micro_avg.get('f1-score', 0), 
            "⚖️", 
            is_percentage=True,
            help_text="Harmonic mean of precision and recall"
        )
    
    # Show entity breakdown if available
    if 'entity_data' in entity_metrics and entity_metrics['entity_data']:
        entity_df = pd.DataFrame(entity_metrics['entity_data'])
        
        # Plot entity performance
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        
        # Sort by F1 score
        entity_df = entity_df.sort_values('f1_score')
        
        # Create barplot
        sns.barplot(x='f1_score', y='entity', data=entity_df, palette='Blues_d', ax=ax)
        ax.set_title('Entity Type Performance (F1 Score)')
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('Entity Type')
        
        # Display plot
        st.pyplot(fig)
        
        # Display detailed metrics table
        with st.expander("Detailed Entity Metrics"):
            # Format for display
            display_df = entity_df.copy()
            for col in ['f1_score', 'precision', 'recall']:
                display_df[col] = display_df[col].map('{:.1%}'.format)
            
            st.dataframe(
                display_df.rename(columns={
                    'entity': 'Entity Type',
                    'f1_score': 'F1 Score',
                    'precision': 'Precision',
                    'recall': 'Recall',
                    'support': 'Examples'
                }),
                use_container_width=True
            )

def render_error_analysis(metrics):
    """Render error analysis with enhanced visualizations"""
    st.header("Error Analysis")
    
    # Get detailed results
    detailed_results = metrics.get('detailed_results', [])
    if not detailed_results:
        st.warning("No detailed results available for error analysis")
        return
    
    # Process errors
    error_data = analyze_errors(detailed_results)
    if not error_data or not error_data.get('errors'):
        st.warning("No errors found in the evaluation dataset")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            "Error Rate", 
            error_data['error_rate'] * 100, 
            "❌", 
            is_percentage=True,
            help_text="Percentage of examples with incorrect intent predictions"
        )
    with col2:
        render_metric_card(
            "High Confidence Errors", 
            error_data['high_confidence_errors'], 
            "⚠️",
            help_text="Number of errors with confidence > 0.8"
        )
    with col3:
        render_metric_card(
            "Avg. Error Confidence", 
            error_data['avg_error_confidence'] * 100, 
            "📉", 
            is_percentage=True,
            help_text="Average confidence score for incorrect predictions"
        )
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Create Sankey diagram for error patterns
        sankey_diagram = create_error_pattern_sankey(error_data['errors'])
        if sankey_diagram:
            st.plotly_chart(sankey_diagram, use_container_width=True)
    
    with col2:
        # Create confidence distribution histogram
        conf_histogram = create_confidence_distribution(
            error_data['errors'], 
            error_data['correct']
        )
        if conf_histogram:
            st.plotly_chart(conf_histogram, use_container_width=True)
    
    # Error pattern table
    st.subheader("Common Error Patterns")
    if error_data.get('error_patterns'):
        # Create dataframe for display
        patterns_df = pd.DataFrame(error_data['error_patterns'])
        patterns_df['percentage'] = patterns_df['percentage'].map('{:.1f}%'.format)
        
        st.dataframe(
            patterns_df.rename(columns={
                'true_intent': 'True Intent',
                'predicted_intent': 'Predicted As',
                'count': 'Count',
                'percentage': '% of Errors'
            }),
            use_container_width=True
        )
    else:
        st.info("No clear error patterns detected")
    
    # Interactive error explorer
    st.subheader("Error Examples")
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        # Get unique true intents from errors
        true_intents = list(set(e.get('true_intent', '') for e in error_data['errors']))
        true_intents.sort()
        
        selected_true_intent = st.selectbox(
            "Filter by True Intent",
            ["All"] + true_intents
        )
    
    with col2:
        # Get unique predicted intents from errors
        pred_intents = list(set(e.get('pred_intent', '') for e in error_data['errors']))
        pred_intents.sort()
        
        selected_pred_intent = st.selectbox(
            "Filter by Predicted Intent",
            ["All"] + pred_intents
        )
    
    # Apply filters
    filtered_errors = error_data['errors']
    
    if selected_true_intent != "All":
        filtered_errors = [e for e in filtered_errors if e.get('true_intent') == selected_true_intent]
    
    if selected_pred_intent != "All":
        filtered_errors = [e for e in filtered_errors if e.get('pred_intent') == selected_pred_intent]
    
    # Show filtered errors
    if filtered_errors:
        st.write(f"Showing {len(filtered_errors)} of {len(error_data['errors'])} errors")
        
        for i, error in enumerate(filtered_errors[:20]):  # Limit to 20 examples for performance
            with st.expander(f"'{error.get('text', 'No text')}' (Confidence: {error.get('confidence', 0):.2f})"):
                col1, col2 = st.columns(2)
                col1.markdown(f"**True Intent:** {error.get('true_intent', 'Unknown')}")
                col2.markdown(f"**Predicted Intent:** {error.get('pred_intent', 'Unknown')}")
                
                # Show entities if available
                if 'entities' in error:
                    st.markdown("**Entities:**")
                    for entity in error['entities']:
                        st.markdown(f"- {entity.get('entity', '')}: {entity.get('value', '')}")
    else:
        st.info("No errors match the selected filters")

def render_performance_history():
    """Render performance history with enhanced visualizations"""
    st.header("Performance History")
    
    # Load history data
    history_df = load_history()
    if history_df.empty:
        st.warning("No historical data available")
        return
    
    # Process history data
    processed_history = process_history_data(history_df)
    
    # Summary metrics with trend indicators
    col1, col2, col3 = st.columns(3)
    
    if 'trends' in processed_history:
        trends = processed_history['trends']
        
        with col1:
            if 'intent_accuracy' in trends:
                trend = trends['intent_accuracy']
                render_metric_card(
                    "Intent Accuracy", 
                    trend['current'] * 100, 
                    "🎯", 
                    is_percentage=True,
                    delta=trend['change'] * 100 if 'change' in trend else None,
                    help_text="Current accuracy with change from previous run"
                )
        
        with col2:
            if 'intent_f1' in trends:
                trend = trends['intent_f1']
                render_metric_card(
                    "Intent F1", 
                    trend['current'] * 100, 
                    "⚖️", 
                    is_percentage=True,
                    delta=trend['change'] * 100 if 'change' in trend else None,
                    help_text="Current F1 score with change from previous run"
                )
        
        with col3:
            if 'entity_f1' in trends:
                trend = trends['entity_f1']
                render_metric_card(
                    "Entity F1", 
                    trend['current'] * 100, 
                    "🏷️", 
                    is_percentage=True,
                    delta=trend['change'] * 100 if 'change' in trend else None,
                    help_text="Current entity F1 score with change from previous run"
                )
    
    # Enhanced timeline visualization
    timeline_chart = create_performance_timeline(history_df)
    if timeline_chart:
        st.plotly_chart(timeline_chart, use_container_width=True)
    
    # Model version history
    st.subheader("Model Version History")
    
    # Prepare model history table
    if not history_df.empty and 'model_id' in history_df.columns:
        display_df = history_df.copy()
        
        # Format columns for display
        for col in ['intent_accuracy', 'intent_f1', 'entity_f1']:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")
        
        # Extract timestamp and format it
        if 'timestamp' in display_df.columns:
            display_df['date'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Select and rename columns for display
        columns_to_show = ['date', 'model_id', 'intent_accuracy', 'intent_f1', 'entity_f1']
        rename_map = {
            'date': 'Date',
            'model_id': 'Model ID',
            'intent_accuracy': 'Intent Accuracy',
            'intent_f1': 'Intent F1',
            'entity_f1': 'Entity F1'
        }
        
        # Filter columns that exist in the dataframe
        available_cols = [col for col in columns_to_show if col in display_df.columns]
        
        if available_cols:
            st.dataframe(
                display_df[available_cols].rename(columns={
                    col: rename_map.get(col, col) for col in available_cols
                }).sort_values('date', ascending=False),
                use_container_width=True
            )
        else:
            st.warning("No model history data available to display")
    else:
        st.warning("No model history data available")

def render_model_comparison():
    """Render model comparison view"""
    st.header("Model Comparison")
    
    # Load available models
    models = load_available_runs()
    if not models:
        st.warning("No models available for comparison")
        return
    
    # Select models to compare
    col1, col2 = st.columns(2)
    
    with col1:
        base_model_index = 0
        base_model = st.selectbox(
            "Base Model",
            [f"{m['model_id']} ({m['timestamp']})" for m in models],
            index=base_model_index,
            key="base_model"
        )
    
    with col2:
        comparison_model_index = min(1, len(models) - 1)
        comparison_model = st.selectbox(
            "Comparison Model",
            [f"{m['model_id']} ({m['timestamp']})" for m in models],
            index=comparison_model_index,
            key="comparison_model"
        )
    
    # Skip if same model selected
    if base_model == comparison_model:
        st.warning("Please select different models to compare")
        return
    
    # Load metrics for selected models
    base_model_info = models[base_model_index]
    comparison_model_info = models[comparison_model_index]
    
    base_metrics = load_metrics(base_model_info["file"])
    comparison_metrics = load_metrics(comparison_model_info["file"])
    
    if not base_metrics or not comparison_metrics:
        st.warning("Could not load metrics for the selected models")
        return
    
    # Show comparison summary
    st.subheader("Comparison Summary")
    
    # Compare key metrics
    comparison_rows = []
    
    # Intent metrics
    base_intent = base_metrics.get('intent_metrics', {})
    comp_intent = comparison_metrics.get('intent_metrics', {})
    
    for metric_name, display_name in [
        ('accuracy', 'Intent Accuracy'),
        ('precision', 'Intent Precision'),
        ('recall', 'Intent Recall'),
        ('f1', 'Intent F1')
    ]:
        base_value = base_intent.get(metric_name, 0)
        comp_value = comp_intent.get(metric_name, 0)
        diff = comp_value - base_value
        
        comparison_rows.append({
            'Metric': display_name,
            'Base Model': f"{base_value:.2%}",
            'Comparison Model': f"{comp_value:.2%}",
            'Difference': f"{diff:.2%}",
            'raw_diff': diff
        })
    
    # Entity metrics
    base_entity = base_metrics.get('entity_metrics', {}).get('report', {}).get('micro avg', {})
    comp_entity = comparison_metrics.get('entity_metrics', {}).get('report', {}).get('micro avg', {})
    
    for metric_name, display_name in [
        ('precision', 'Entity Precision'),
        ('recall', 'Entity Recall'),
        ('f1-score', 'Entity F1')
    ]:
        base_value = base_entity.get(metric_name, 0)
        comp_value = comp_entity.get(metric_name, 0)
        diff = comp_value - base_value
        
        comparison_rows.append({
            'Metric': display_name,
            'Base Model': f"{base_value:.2%}",
            'Comparison Model': f"{comp_value:.2%}",
            'Difference': f"{diff:.2%}",
            'raw_diff': diff
        })
    
    # Convert to dataframe
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Display summary table with highlighting
    def highlight_diff(val):
        """Highlight positive/negative differences"""
        if 'raw_diff' in val:
            diff = val['raw_diff']
            if diff > 0:
                color = '#c6ecc6'  # light green
            elif diff < 0:
                color = '#ffcccc'  # light red
            else:
                color = '#ffffff'  # white
            return [f"background-color: {color}" if col == 'Difference' else "" for col in comparison_df.columns]
        return ["" for _ in comparison_df.columns]
    
    # Display comparison dataframe without the raw_diff column
    st.dataframe(
        comparison_df.drop(columns=['raw_diff']).style.apply(highlight_diff, axis=1),
        use_container_width=True
    )
    
    # Most improved and degraded intents
    st.subheader("Intent Performance Changes")
    
    # Get per-class metrics for both models
    base_per_class = base_metrics.get('intent_metrics', {}).get('per_class_report', {})
    comp_per_class = comparison_metrics.get('intent_metrics', {}).get('per_class_report', {})
    
    # Skip aggregate metrics
    skip_intents = ['micro avg', 'macro avg', 'weighted avg']
    
    # Find common intents
    common_intents = [
        intent for intent in base_per_class.keys() 
        if intent in comp_per_class and intent not in skip_intents
    ]
    
    if common_intents:
        # Calculate differences
        intent_changes = []
        
        for intent in common_intents:
            base_f1 = base_per_class[intent].get('f1-score', 0)
            comp_f1 = comp_per_class[intent].get('f1-score', 0)
            diff = comp_f1 - base_f1
            
            intent_changes.append({
                'intent': intent,
                'base_f1': base_f1,
                'comp_f1': comp_f1,
                'diff': diff
            })
        
        # Sort by difference
        intent_changes.sort(key=lambda x: x['diff'])
        
        # Display most improved and degraded intents
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Most Degraded Intents")
            degraded = intent_changes[:5]  # 5 most degraded
            
            if degraded:
                degraded_df = pd.DataFrame(degraded)
                degraded_df['base_f1'] = degraded_df['base_f1'].map('{:.2%}'.format)
                degraded_df['comp_f1'] = degraded_df['comp_f1'].map('{:.2%}'.format)
                degraded_df['diff'] = degraded_df['diff'].map('{:.2%}'.format)
                
                st.dataframe(
                    degraded_df.rename(columns={
                        'intent': 'Intent',
                        'base_f1': 'Base F1',
                        'comp_f1': 'Comparison F1',
                        'diff': 'Change'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No degraded intents found")
        
        with col2:
            st.markdown("#### Most Improved Intents")
            improved = intent_changes[-5:]  # 5 most improved, reversed
            improved.reverse()
            
            if improved:
                improved_df = pd.DataFrame(improved)
                improved_df['base_f1'] = improved_df['base_f1'].map('{:.2%}'.format)
                improved_df['comp_f1'] = improved_df['comp_f1'].map('{:.2%}'.format)
                improved_df['diff'] = improved_df['diff'].map('{:.2%}'.format)
                
                st.dataframe(
                    improved_df.rename(columns={
                        'intent': 'Intent',
                        'base_f1': 'Base F1',
                        'comp_f1': 'Comparison F1',
                        'diff': 'Change'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No improved intents found")
    else:
        st.warning("No common intents found between the models")

def render_latest_results():
    """Render latest benchmark results with tabs"""
    st.header("Latest Benchmark Results")
    
    # Load latest metrics
    metrics = load_latest_metrics()
    if not metrics:
        st.warning("No benchmark results available. Please run the benchmark first.")
        return
    
    # Format timestamp
    timestamp = metrics.get('timestamp', '')
    if timestamp:
        formatted_time = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    else:
        formatted_time = "Unknown"
    
    # Show model info
    model_id = metrics.get('model_id', 'Unknown')
    st.markdown(f"**Model ID:** {model_id}")
    st.markdown(f"**Benchmark Time:** {formatted_time}")
    
    # Create tabs for different types of metrics
    tabs = st.tabs(["Intent Metrics", "Entity Metrics", "Error Analysis"])
    
    with tabs[0]:
        render_intent_metrics(metrics)
    
    with tabs[1]:
        render_entity_metrics(metrics)
    
    with tabs[2]:
        render_error_analysis(metrics)

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
    # Use text instead of image logo
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
    """Main function to render the dashboard"""
    # Create navigation
    create_navigation(PAGES)
    
    # Determine which page to show
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    
    current_page = st.session_state.current_page
    
    # Load data that might be needed
    available_models = load_available_runs()
    latest_metrics = load_latest_metrics()
    
    # Render the appropriate page
    if current_page == "home":
        render_home_page_with_data(available_models, latest_metrics)
    elif current_page == "results":
        render_latest_results()
    elif current_page == "history":
        render_performance_history()
    elif current_page == "errors":
        # Use error analysis from latest results
        if latest_metrics:
            render_error_analysis(latest_metrics)
        else:
            st.warning("No benchmark results available for error analysis")
    elif current_page == "comparison":
        render_model_comparison()
    else:
        st.error(f"Unknown page: {current_page}")

if __name__ == "__main__":
    main() 