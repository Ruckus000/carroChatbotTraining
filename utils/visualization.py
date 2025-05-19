"""
Visualization components for the NLU Benchmarking Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def create_intent_radar_chart(metrics):
    """
    Create a radar chart showing performance of intents
    
    Args:
        metrics: Dictionary containing intent metrics
        
    Returns:
        Plotly figure object or None if data is not available
    """
    # Extract per-class metrics
    if 'intent_metrics' not in metrics or 'per_class_report' not in metrics['intent_metrics']:
        return None
        
    per_class = metrics['intent_metrics']['per_class_report']
    
    # Extract intents and their F1 scores, filtering out aggregate metrics
    intents = [(intent, data['f1-score']) for intent, data in per_class.items() 
              if intent not in ['micro avg', 'macro avg', 'weighted avg']]
    
    if not intents:
        return None
    
    # Sort by F1 score
    intents.sort(key=lambda x: x[1])
    
    # Extract top 5 and bottom 5 intents
    bottom_5 = intents[:min(5, len(intents)//2)]
    top_5 = intents[-min(5, len(intents)//2):]
    
    # Create radar chart data
    categories = [i[0] for i in bottom_5 + top_5]
    values = [i[1] for i in bottom_5 + top_5]
    
    # Radar chart using Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='F1 Score',
        line=dict(color='#4b9dff', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Best and Worst Performing Intents"
    )
    
    return fig

def create_performance_timeline(history_data):
    """
    Create a timeline visualization of performance metrics with annotations
    
    Args:
        history_data: DataFrame containing historical metrics or dictionary with processed history data
        
    Returns:
        Plotly figure object or None if data is not available
    """
    # Handle both DataFrame and dictionary inputs
    history_df = None
    if isinstance(history_data, pd.DataFrame):
        history_df = history_data
    elif isinstance(history_data, dict) and 'history_df' in history_data:
        history_df = history_data['history_df']
    else:
        return None
        
    if history_df is None or history_df.empty:
        return None
        
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Ensure timestamp is in datetime format
    time_col = None
    for col in ['timestamp', 'date']:
        if col in history_df.columns:
            history_df[col] = pd.to_datetime(history_df[col])
            time_col = col
            break
            
    if not time_col:
        return None
        
    # Add traces for each metric
    for metric_name, color in [('intent_f1', "#1f77b4"), ('entity_f1', "#ff7f0e")]:
        if metric_name in history_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=history_df[time_col],
                    y=history_df[metric_name],
                    name=metric_name.replace('_', ' ').title(),
                    line=dict(color=color, width=3)
                ),
                secondary_y=False,
            )
    
    # Find significant model changes and add annotations
    if 'intent_f1' in history_df.columns and len(history_df) > 1:
        for i, row in history_df.iterrows():
            if i > 0 and abs(row['intent_f1'] - history_df.iloc[i-1]['intent_f1']) > 0.05:
                fig.add_annotation(
                    x=row[time_col],
                    y=row['intent_f1'],
                    text="Major Change",
                    showarrow=True,
                    arrowhead=2,
                )
    
    # Add figure layout
    fig.update_layout(
        title_text="Performance History",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="F1 Score", range=[0, 1], secondary_y=False)
    
    return fig

def create_confusion_matrix_heatmap(confusion_matrix, labels):
    """
    Create an enhanced heatmap visualization for confusion matrix
    
    Args:
        confusion_matrix: 2D array of confusion matrix
        labels: List of class labels
        
    Returns:
        Plotly figure object or None if data is not available
    """
    if not confusion_matrix or not labels or len(confusion_matrix) == 0 or len(labels) == 0:
        return None
        
    # Convert to numpy array if needed
    cm = np.array(confusion_matrix)
    
    # Normalize the confusion matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaNs with zeros
    
    # Create the heatmap using Plotly
    fig = px.imshow(
        cm_normalized,
        labels=dict(x="Predicted Intent", y="True Intent", color="Probability"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto='.2%',
        aspect="auto"
    )
    
    # Update layout for better visibility
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Intent",
        yaxis_title="True Intent",
        xaxis=dict(tickangle=-45),
        height=600,
    )
    
    # Add custom hover template
    fig.update_traces(
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z:.2%}<extra></extra>"
    )
    
    return fig

def create_error_pattern_sankey(error_analysis):
    """
    Create a Sankey diagram showing error patterns
    
    Args:
        error_analysis: Dictionary with error analysis data
        
    Returns:
        Plotly figure object or None if data is not available
    """
    if not error_analysis or 'errors' not in error_analysis or not error_analysis['errors']:
        return None
    
    errors = error_analysis['errors']
        
    # Count error patterns
    patterns = {}
    for error in errors:
        true_intent = error.get('true_intent', 'Unknown')
        pred_intent = error.get('pred_intent', 'Unknown')
        key = (true_intent, pred_intent)
        
        patterns[key] = patterns.get(key, 0) + 1
    
    # Sort by count and take top 10
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if not sorted_patterns:
        return None
    
    # Create lists for Sankey diagram
    sources = []
    targets = []
    values = []
    
    # Build node lists
    true_intents = set(true for (true, _), _ in sorted_patterns)
    pred_intents = set(pred for (_, pred), _ in sorted_patterns)
    
    # Map intents to indices
    true_map = {intent: i for i, intent in enumerate(true_intents)}
    pred_map = {intent: i + len(true_intents) for i, intent in enumerate(pred_intents)}
    
    # Build link lists
    for (true, pred), count in sorted_patterns:
        sources.append(true_map[true])
        targets.append(pred_map[pred])
        values.append(count)
    
    # Build label list
    labels = list(true_intents) + list(pred_intents)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    fig.update_layout(
        title_text="Error Patterns Flow",
        font_size=10
    )
    
    return fig

def create_confidence_distribution(error_analysis):
    """
    Create a confidence distribution histogram comparing errors vs correct predictions
    
    Args:
        error_analysis: Dictionary with error analysis data
        
    Returns:
        Plotly figure object or None if data is not available
    """
    if not error_analysis or 'errors' not in error_analysis or not error_analysis['errors']:
        return None
    
    errors = error_analysis['errors']
    correct = error_analysis.get('correct', [])
    
    # Extract confidence scores
    error_confidences = [e.get('confidence', 0) for e in errors if 'confidence' in e]
    
    if not error_confidences:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add error confidence histogram
    fig.add_trace(go.Histogram(
        x=error_confidences,
        name='Incorrect Predictions',
        opacity=0.7,
        marker=dict(color='#ff6b6b'),
        histnorm='probability density',
        bingroup='group1'
    ))
    
    # Add correct predictions if available
    if correct:
        correct_confidences = [p.get('confidence', 0) for p in correct if 'confidence' in p]
        if correct_confidences:
            fig.add_trace(go.Histogram(
                x=correct_confidences,
                name='Correct Predictions',
                opacity=0.7,
                marker=dict(color='#4b9dff'),
                histnorm='probability density',
                bingroup='group1'
            ))
    
    # Layout
    fig.update_layout(
        title='Confidence Score Distribution',
        xaxis_title='Confidence Score',
        yaxis_title='Probability Density',
        barmode='overlay',
        bargap=0.1,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add vertical lines for confidence thresholds
    fig.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.8, line_width=1, line_dash="dash", line_color="gray")
    
    # Add annotations for the threshold lines
    fig.add_annotation(x=0.5, y=1.0, yref="paper", text="Low Confidence", showarrow=False, yshift=10)
    fig.add_annotation(x=0.8, y=1.0, yref="paper", text="High Confidence", showarrow=False, yshift=10)
    
    return fig 