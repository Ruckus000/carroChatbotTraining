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
    """
    # Extract per-class metrics
    if 'intent_metrics' not in metrics or 'per_class_report' not in metrics['intent_metrics']:
        st.warning("No per-class intent metrics available for radar chart")
        return None
        
    per_class = metrics['intent_metrics']['per_class_report']
    
    # Extract intents and their F1 scores
    intents = [(intent, data['f1-score']) for intent, data in per_class.items() 
              if intent not in ['micro avg', 'macro avg', 'weighted avg']]
    
    # Sort by F1 score
    intents.sort(key=lambda x: x[1])
    
    # Extract top 5 and bottom 5 intents
    bottom_5 = intents[:5] if len(intents) >= 5 else intents[:len(intents)//2]
    top_5 = intents[-5:] if len(intents) >= 5 else intents[len(intents)//2:]
    
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

def create_performance_timeline(history_df):
    """
    Create a timeline visualization of performance metrics with annotations
    
    Args:
        history_df: DataFrame containing historical metrics
    """
    if history_df.empty:
        st.warning("No history data available for timeline visualization")
        return None
        
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Ensure timestamp is in datetime format
    if 'timestamp' in history_df.columns:
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    elif 'date' in history_df.columns:
        history_df['timestamp'] = pd.to_datetime(history_df['date'])
    else:
        st.warning("No timestamp or date column found in history data")
        return None
        
    # Add traces
    if 'intent_f1' in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=history_df['timestamp'],
                y=history_df['intent_f1'],
                name="Intent F1",
                line=dict(color="#1f77b4", width=3)
            ),
            secondary_y=False,
        )
    
    if 'entity_f1' in history_df.columns:
        fig.add_trace(
            go.Scatter(
                x=history_df['timestamp'],
                y=history_df['entity_f1'],
                name="Entity F1",
                line=dict(color="#ff7f0e", width=3)
            ),
            secondary_y=False,
        )
    
    # Find significant model changes and add annotations
    if 'intent_f1' in history_df.columns and len(history_df) > 1:
        for i, row in history_df.iterrows():
            if i > 0 and abs(row['intent_f1'] - history_df.iloc[i-1]['intent_f1']) > 0.05:
                fig.add_annotation(
                    x=row['timestamp'],
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
    """
    if len(confusion_matrix) == 0 or len(labels) == 0:
        st.warning("Insufficient data for confusion matrix visualization")
        return None
        
    # Convert to numpy array if needed
    cm = np.array(confusion_matrix)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
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
        width=800,
        height=800,
    )
    
    # Add custom hover template
    fig.update_traces(
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z:.2%}<extra></extra>"
    )
    
    return fig

def create_error_pattern_sankey(errors):
    """
    Create a Sankey diagram showing error patterns
    
    Args:
        errors: List of error examples
    """
    if not errors:
        st.warning("No errors available for Sankey diagram")
        return None
        
    # Count error patterns
    patterns = {}
    for error in errors:
        true_intent = error.get('true_intent', 'Unknown')
        pred_intent = error.get('pred_intent', 'Unknown')
        key = (true_intent, pred_intent)
        
        if key not in patterns:
            patterns[key] = 0
        patterns[key] += 1
    
    # Sort by count and take top 10
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Create lists for Sankey diagram
    sources = []
    targets = []
    values = []
    labels = []
    
    # Build node lists
    true_intents = set()
    pred_intents = set()
    
    for (true, pred), count in sorted_patterns:
        true_intents.add(true)
        pred_intents.add(pred)
    
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

def create_confidence_distribution(errors, correct_predictions=None):
    """
    Create a confidence distribution histogram comparing errors vs correct predictions
    
    Args:
        errors: List of error examples
        correct_predictions: List of correct prediction examples
    """
    # Extract confidence scores for errors
    error_confidences = [e.get('confidence', 0) for e in errors if 'confidence' in e]
    
    if not error_confidences:
        st.warning("No confidence scores available for errors")
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
    if correct_predictions:
        correct_confidences = [p.get('confidence', 0) for p in correct_predictions if 'confidence' in p]
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
    if error_confidences:
        fig.add_vline(x=0.5, line_width=1, line_dash="dash", line_color="gray")
        fig.add_vline(x=0.8, line_width=1, line_dash="dash", line_color="gray")
        
        # Add annotations for the threshold lines
        fig.add_annotation(x=0.5, y=1.0, yref="paper", text="Low Confidence", showarrow=False, yshift=10)
        fig.add_annotation(x=0.8, y=1.0, yref="paper", text="High Confidence", showarrow=False, yshift=10)
    
    return fig 