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
        ax.set_title('Normalized Confusion Matrix')
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
        ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')

    # Rotate labels if there are many classes
    if n_classes > 10:
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
    else:
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

    plt.tight_layout()
    return fig

def render_class_performance(class_report, class_type="intent"):
    """Render class performance visualization"""
    if not class_report:
        return None

    # Extract data from report
    classes = list(class_report.keys())
    f1_scores = [class_report[cls]['f1-score'] for cls in classes]
    support = [class_report[cls]['support'] for cls in classes]

    # Skip classes with specific names if needed
    if class_type == "entity":
        classes = [c for c in classes if not c.startswith('micro') and not c.startswith('macro')]
        if not classes:
            return None

    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)

    # If too many classes, focus on worst performers
    if len(classes) > 25:
        # Show bottom 10 and top 5 classes
        indices_to_show = list(sorted_indices[:10])
        # Add a few top performers
        indices_to_show.extend(sorted_indices[-5:])
    else:
        indices_to_show = sorted_indices

    selected_classes = [classes[i] for i in indices_to_show]
    selected_f1 = [f1_scores[i] for i in indices_to_show]
    selected_support = [support[i] for i in indices_to_show]

    # Format class names for better display
    display_classes = [format_class_name(cls) for cls in selected_classes]

    # Create figure with adjusted size based on number of classes
    height = max(5, min(20, len(selected_classes) * 0.4))
    fig, ax = plt.subplots(figsize=(10, height))

    # Create color gradient based on F1 scores
    colors = plt.cm.RdYlGn(np.array(selected_f1))

    bars = ax.barh(display_classes, selected_f1, color=colors)

    # Add support count annotations
    for i, (bar, sup) in enumerate(zip(bars, selected_support)):
        ax.text(
            bar.get_width() + 0.02,
            i,
            f'n={sup}',
            va='center',
            alpha=0.7
        )

    ax.set_xlabel('F1 Score')
    ax.set_title(f'{class_type.capitalize()} Class F1 Scores')
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')

    plt.tight_layout()
    return fig

def format_entities(entities):
    """Format entity list for display"""
    if not entities:
        return "<em>No entities</em>"

    result = ""
    for entity in entities:
        entity_type = entity.get('entity', 'unknown')
        entity_value = entity.get('value', '')
        result += f"<span style='background:#e6f0ff; padding:1px 5px; border-radius:3px; margin-right:5px;'>{entity_type}: <strong>{entity_value}</strong></span> "

    return result

def main():
    """Main dashboard function"""
    render_header()

    # Load available runs
    runs = load_available_runs()

    if not runs:
        st.warning("No benchmark results found. Run the evaluation script first.")
        st.markdown("""
        ### Getting Started
        To generate benchmark results:
        ```bash
        python evaluate_nlu.py --benchmark data/nlu_benchmark_data.json
        ```

        Once you've run the benchmark, refresh this page to view the results.
        """)
        return

    # Sidebar for run selection and comparison
    with st.sidebar:
        st.subheader("Benchmark Runs")

        # Run selection
        selected_run_index = st.selectbox(
            "Select a benchmark run:",
            range(len(runs)),
            format_func=lambda i: f"{runs[i]['timestamp']} ({runs[i]['model_id']})"
        )

        selected_file = runs[selected_run_index]["file"]
        metrics = load_metrics(selected_file)

        # Comparison selection (only if multiple runs available)
        compare_enabled = st.checkbox("Compare with another run", value=False, disabled=len(runs) < 2)
        
        if compare_enabled and len(runs) > 1:
            compare_options = [i for i in range(len(runs)) if i != selected_run_index]
            compare_run_index = st.selectbox(
                "Select run to compare:",
                compare_options,
                format_func=lambda i: f"{runs[i]['timestamp']} ({runs[i]['model_id']})"
            )
            compare_file = runs[compare_run_index]["file"]
            compare_metrics = load_metrics(compare_file)
        else:
            compare_metrics = None

        # Run details section
        st.subheader("Run Details")
        st.markdown(f"**Date:** {runs[selected_run_index]['timestamp']}")
        st.markdown(f"**Model ID:** {runs[selected_run_index]['model_id']}")

    # Main content area
    if metrics is None:
        st.error(f"Failed to load metrics from file: {selected_file}")
        return

    # Display key metrics in a clean card layout
    st.markdown("## Performance Summary")

    # Extract metrics for display
    intent_metrics = metrics.get('intent_metrics', {})
    entity_metrics = metrics.get('entity_metrics', {})
    error_analysis = metrics.get('error_analysis', {})

    intent_accuracy = intent_metrics.get('accuracy', 0)
    intent_f1 = intent_metrics.get('f1', 0)
    entity_f1 = entity_metrics.get('micro avg', {}).get('f1-score', 0) if isinstance(entity_metrics, dict) else 0
    intent_error_rate = error_analysis.get('intent_error_rate', 0)
    entity_error_rate = error_analysis.get('entity_error_rate', 0)

    # Calculate deltas if comparing
    if compare_metrics:
        compare_intent_metrics = compare_metrics.get('intent_metrics', {})
        compare_entity_metrics = compare_metrics.get('entity_metrics', {})

        intent_accuracy_delta = intent_accuracy - compare_intent_metrics.get('accuracy', 0)
        intent_f1_delta = intent_f1 - compare_intent_metrics.get('f1', 0)
        
        if isinstance(compare_entity_metrics, dict) and 'micro avg' in compare_entity_metrics:
            entity_f1_delta = entity_f1 - compare_entity_metrics.get('micro avg', {}).get('f1-score', 0)
        else:
            entity_f1_delta = None
    else:
        intent_accuracy_delta = None
        intent_f1_delta = None
        entity_f1_delta = None

    # Display metrics in clean, responsive grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            custom_metric_card(
                "Intent Accuracy",
                intent_accuracy,
                delta=intent_accuracy_delta,
                help_text="Overall accuracy of intent classification",
                color="auto"
            ),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            custom_metric_card(
                "Intent F1 Score",
                intent_f1,
                delta=intent_f1_delta,
                help_text="Weighted F1 score across all intents",
                color="auto"
            ),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            custom_metric_card(
                "Entity F1 Score",
                entity_f1,
                delta=entity_f1_delta,
                help_text="Micro-average F1 score for entity recognition",
                color="auto"
            ),
            unsafe_allow_html=True
        )

    # Create tabs for different sections
    tabs = st.tabs([
        "📈 Performance Trends",
        "🔍 Intent Analysis",
        "🏷️ Entity Analysis",
        "❌ Error Analysis"
    ])

    # Tab 1: Performance Trends
    with tabs[0]:
        st.markdown("### Performance History")

        # Load history data
        history_df = load_history()

        if history_df is not None and len(history_df) > 1:
            # Prepare data for plotting
            history_df['date_formatted'] = pd.to_datetime(history_df['date'])
            history_df = history_df.sort_values('date_formatted')

            # Create two columns
            trend_col1, trend_col2 = st.columns([3, 1])

            with trend_col1:
                # Create trend charts
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

                # Intent metrics
                ax1.plot(history_df['date_formatted'], history_df['intent_accuracy'], marker='o', label='Accuracy')
                ax1.plot(history_df['date_formatted'], history_df['intent_f1'], marker='s', label='F1')
                ax1.set_title('Intent Classification Metrics')
                ax1.set_ylabel('Score')
                ax1.set_ylim(0, 1)
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend()

                # Format x-axis
                ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

                # Entity metrics
                ax2.plot(history_df['date_formatted'], history_df['entity_precision'], marker='o', label='Precision')
                ax2.plot(history_df['date_formatted'], history_df['entity_recall'], marker='s', label='Recall')
                ax2.plot(history_df['date_formatted'], history_df['entity_f1'], marker='^', label='F1')
                ax2.set_title('Entity Recognition Metrics')
                ax2.set_ylabel('Score')
                ax2.set_ylim(0, 1)
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend()

                # Format x-axis
                ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

                plt.tight_layout()
                st.pyplot(fig)

            with trend_col2:
                # Show data table with expandable details
                with st.expander("View History Data"):
                    st.dataframe(
                        history_df[['date', 'intent_f1', 'entity_f1', 'model_id']].sort_values('date', ascending=False),
                        hide_index=True
                    )
        else:
            st.info("Not enough history data to show trends. Run the benchmark at least twice to see performance over time.")

    # Tab 2: Intent Analysis
    with tabs[1]:
        st.markdown("### Intent Classification Analysis")

        # Get confusion matrix data
        cm = np.array(intent_metrics.get('confusion_matrix', []))
        labels = intent_metrics.get('labels', [])

        # Show confusion matrix if available
        if len(cm) > 0 and len(labels) > 0:
            st.pyplot(render_confusion_matrix(cm, labels))

        # Intent class performance
        st.markdown("### Intent Class Performance")

        # Show class performance chart
        per_class_report = intent_metrics.get('per_class_report', {})
        if per_class_report:
            # Get class metrics
            fig = render_class_performance(per_class_report, "intent")
            if fig:
                st.pyplot(fig)

                # Extract low performers
                low_performers = {cls: metrics for cls, metrics in per_class_report.items()
                                if metrics['f1-score'] < 0.7 and metrics['support'] >= 3}

                # Show low performers table if any
                if low_performers:
                    st.markdown("#### Low Performing Intents")
                    st.markdown("These intents have low F1 scores and may need attention:")

                    # Convert to DataFrame for better display
                    low_df = pd.DataFrame({
                        'Intent': list(low_performers.keys()),
                        'F1 Score': [m['f1-score'] for m in low_performers.values()],
                        'Precision': [m['precision'] for m in low_performers.values()],
                        'Recall': [m['recall'] for m in low_performers.values()],
                        'Support': [m['support'] for m in low_performers.values()]
                    }).sort_values('F1 Score')

                    st.dataframe(low_df, hide_index=True)
            else:
                st.info("No per-class performance data available.")

    # Tab 3: Entity Analysis
    with tabs[2]:
        st.markdown("### Entity Recognition Analysis")

        # Check if entity metrics are available
        if 'entity_metrics' in metrics and isinstance(entity_metrics, dict) and entity_metrics:
            # Filter out aggregate metrics
            entity_types = [key for key in entity_metrics.keys()
                          if key not in ['micro avg', 'macro avg', 'weighted avg']
                          and not (key.startswith('B-') and f'I-{key[2:]}' in entity_metrics)]

            if entity_types:
                # Display entity metrics table
                st.markdown("#### Entity Metrics")
                
                # Create DataFrame for display
                entity_data = {
                    'Entity': [],
                    'F1 Score': [],
                    'Precision': [],
                    'Recall': [],
                    'Support': []
                }
                
                for entity_type in entity_types:
                    entity_data['Entity'].append(entity_type)
                    entity_data['F1 Score'].append(entity_metrics[entity_type].get('f1-score', 0))
                    entity_data['Precision'].append(entity_metrics[entity_type].get('precision', 0))
                    entity_data['Recall'].append(entity_metrics[entity_type].get('recall', 0))
                    entity_data['Support'].append(entity_metrics[entity_type].get('support', 0))
                
                entity_df = pd.DataFrame(entity_data)
                st.dataframe(entity_df.sort_values('F1 Score', ascending=False), hide_index=True)
                
                # Add overall metrics
                st.markdown("#### Overall Entity Metrics")
                if 'micro avg' in entity_metrics:
                    micro_avg = entity_metrics['micro avg']
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>Micro Avg F1:</strong> {micro_avg.get('f1-score', 0):.4f}</p>
                        <p><strong>Micro Avg Precision:</strong> {micro_avg.get('precision', 0):.4f}</p>
                        <p><strong>Micro Avg Recall:</strong> {micro_avg.get('recall', 0):.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No entity types found in the metrics.")
        else:
            st.info("No entity recognition metrics available in this benchmark run.")

    # Tab 4: Error Analysis
    with tabs[3]:
        st.markdown("### Error Analysis")

        # Get error data
        detailed_results = metrics.get('detailed_results', [])
        intent_errors = [r for r in detailed_results if not r.get('intent_correct', True)]
        entity_errors = [r for r in detailed_results if not r.get('entities_correct', True)]

        error_patterns = {}
        for err in intent_errors:
            key = (err.get('true_intent', ''), err.get('pred_intent', ''))
            if key not in error_patterns:
                error_patterns[key] = []
            error_patterns[key].append(err)

        # Display error statistics
        error_stat_col1, error_stat_col2 = st.columns(2)

        with error_stat_col1:
            total_examples = len(detailed_results)
            intent_error_count = len(intent_errors)
            entity_error_count = len(entity_errors)

            # Calculate error rates
            intent_error_rate = intent_error_count / total_examples if total_examples > 0 else 0
            entity_error_rate = entity_error_count / total_examples if total_examples > 0 else 0

            # Display error rate metrics
            st.markdown(
                custom_metric_card(
                    "Intent Error Rate",
                    intent_error_rate,
                    help_text=f"{intent_error_count} errors out of {total_examples} examples",
                    color="auto"
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                custom_metric_card(
                    "Entity Error Rate",
                    entity_error_rate,
                    help_text=f"{entity_error_count} errors out of {total_examples} examples",
                    color="auto"
                ),
                unsafe_allow_html=True
            )

        with error_stat_col2:
            # Calculate confidence statistics
            if intent_errors:
                error_confidences = [err.get('confidence', 0) for err in intent_errors]
                avg_error_confidence = sum(error_confidences) / len(error_confidences)

                # Display confidence metrics
                st.markdown(
                    custom_metric_card(
                        "Avg Error Confidence",
                        avg_error_confidence,
                        help_text="Average confidence score for incorrect predictions",
                        color="auto"
                    ),
                    unsafe_allow_html=True
                )

        # Most common error patterns
        if error_patterns:
            st.markdown("#### Most Common Error Patterns")

            # Sort patterns by frequency
            sorted_patterns = sorted(
                [(k, len(v)) for k, v in error_patterns.items()],
                key=lambda x: x[1],
                reverse=True
            )

            # Create error pattern visualization
            fig, ax = plt.subplots(figsize=(10, min(12, max(5, len(sorted_patterns[:10]) * 0.6))))

            pattern_labels = [f"{true} → {pred}" for (true, pred), _ in sorted_patterns[:10]]
            pattern_counts = [count for _, count in sorted_patterns[:10]]

            bars = ax.barh(pattern_labels, pattern_counts, color='salmon')

            # Add count as text
            for i, count in enumerate(pattern_counts):
                ax.text(count + 0.1, i, str(count), va='center')

            ax.set_xlabel('Number of Occurrences')
            ax.set_title('Most Common Error Patterns')
            ax.grid(True, linestyle='--', alpha=0.7, axis='x')

            plt.tight_layout()
            st.pyplot(fig)

            # Display error examples for top patterns
            st.markdown("#### Error Examples by Pattern")

            for i, ((true_intent, pred_intent), count) in enumerate(sorted_patterns[:5]):
                with st.expander(f"{true_intent} → {pred_intent} ({count} examples)"):
                    examples = error_patterns[(true_intent, pred_intent)]

                    for j, example in enumerate(examples[:5]):
                        st.markdown(f"""
                        <div style="margin-bottom:10px; padding:10px; border-left:3px solid #ff9d4b; background:#f8f9fa;">
                            <p><strong>Text:</strong> "{example['text']}"</p>
                            <p><strong>Confidence:</strong> {example.get('confidence', 0):.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)

# Initialize session state
if 'show_export' not in st.session_state:
    st.session_state.show_export = False
if 'show_raw' not in st.session_state:
    st.session_state.show_raw = False

# Run the app
if __name__ == "__main__":
    main() 