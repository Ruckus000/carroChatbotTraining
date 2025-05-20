"""
Model comparison page for comparing performance of different NLU models.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Callable

from utils.interactive import create_model_comparison_view
from utils.data_processing import load_available_models, load_model_metrics

def render_model_comparison_page():
    """
    Render the model comparison page with interactive model selection and comparison.
    """
    st.title("Model Comparison")
    
    st.markdown("""
    This page allows you to compare the performance of different NLU models side-by-side.
    Select two models from the dropdowns below to see a detailed comparison.
    """)
    
    # Load available models
    models = load_available_models()
    
    if not models:
        st.warning("No models found. Please run some benchmarks first.")
        return
    
    # Create the model comparison view
    create_model_comparison_view(models, load_model_metrics)
    
    # Performance difference visualization
    st.subheader("Performance Evolution")
    st.markdown("""
    The chart below shows how different metrics have evolved across model versions.
    This helps identify trends and patterns in model improvements over time.
    """)
    
    # Create evolution chart based on available models
    create_performance_evolution_chart(models)


def create_performance_evolution_chart(models: List[Dict]):
    """
    Create a chart showing the evolution of performance metrics across model versions.
    
    Args:
        models: List of model metadata dictionaries
    """
    # Check if we have enough models for a chart
    if len(models) < 2:
        st.info("Need at least 2 models to show performance evolution.")
        return
    
    # Extract model data
    model_data = []
    
    for model in models:
        # Load metrics for the model
        metrics = load_model_metrics(model["id"])
        if not metrics:
            continue
            
        # Extract key metrics
        model_entry = {
            "Model": model["id"],
            "Intent Accuracy": metrics.get("intent_metrics", {}).get("accuracy", 0),
            "Intent F1": metrics.get("intent_metrics", {}).get("f1", 0),
            "Entity F1": metrics.get("entity_metrics", {}).get("micro avg", {}).get("f1-score", 0),
        }
        
        # Add additional metadata if available
        model_entry["Date"] = model.get("date", "Unknown")
        model_entry["Version"] = model.get("version", "Unknown")
        
        model_data.append(model_entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(model_data)
    
    # Sort by date/model ID if possible
    if "Date" in df.columns and not df["Date"].equals(pd.Series(["Unknown"] * len(df))):
        df = df.sort_values("Date")
    else:
        df = df.sort_values("Model")
    
    # Create line chart for metrics
    metrics_to_plot = ["Intent Accuracy", "Intent F1", "Entity F1"]
    
    fig = go.Figure()
    
    for metric in metrics_to_plot:
        fig.add_trace(go.Scatter(
            x=df["Model"],
            y=df[metric],
            mode="lines+markers",
            name=metric
        ))
    
    # Update layout
    fig.update_layout(
        title="Performance Metrics Across Model Versions",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True) 