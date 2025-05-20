"""
Error explorer page for analyzing NLU model errors in detail.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

from utils.interactive import create_error_explorer
from utils.data_processing import load_model_metrics
from utils.state_management import get_selected_model

def render_error_explorer_page():
    """
    Render the error explorer page with interactive filtering and exploration.
    """
    st.title("Error Analysis")
    
    st.markdown("""
    This page allows you to explore and analyze errors made by the NLU model.
    You can filter errors by patterns, confidence scores, and more to identify
    systematic issues that might need attention.
    """)
    
    # Get the currently selected model
    model_id = get_selected_model()
    
    if not model_id:
        st.warning("No model selected. Please select a model from the dashboard.")
        return
    
    # Load model metrics
    metrics = load_model_metrics(model_id)
    
    if not metrics:
        st.error(f"Could not load metrics for model: {model_id}")
        return
    
    # Display model info
    st.subheader(f"Analyzing Errors for Model: {model_id}")
    
    # Create error pattern visualization
    create_error_pattern_visualization(metrics)
    
    # Create interactive error explorer
    create_error_explorer(metrics)


def create_error_pattern_visualization(metrics: Dict):
    """
    Create visualizations of error patterns to help identify systematic issues.
    
    Args:
        metrics: Dictionary containing detailed evaluation results
    """
    # Check if detailed results are available
    if "detailed_results" not in metrics or not metrics["detailed_results"]:
        st.warning("No detailed results available for error pattern analysis.")
        return
    
    # Get errors from metrics
    errors = [r for r in metrics["detailed_results"] if not r.get("intent_correct", True)]
    
    if not errors:
        st.success("No intent classification errors found in this evaluation!")
        return
    
    st.subheader("Error Patterns")
    
    # Create error pattern grouping
    error_patterns = {}
    for e in errors:
        pair = (e.get("true_intent", "unknown"), e.get("pred_intent", "unknown"))
        if pair not in error_patterns:
            error_patterns[pair] = []
        error_patterns[pair].append(e)
    
    # Sort patterns by frequency
    sorted_patterns = sorted(
        error_patterns.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    # Create visualization data
    pattern_data = []
    
    for (true_intent, pred_intent), err_list in sorted_patterns:
        if len(err_list) >= 2:  # Only show patterns with at least 2 errors
            pattern_data.append({
                "True Intent": true_intent,
                "Predicted Intent": pred_intent,
                "Count": len(err_list),
                "Average Confidence": sum(e.get("confidence", 0) for e in err_list) / len(err_list)
            })
    
    if not pattern_data:
        st.info("No significant error patterns found (need at least 2 errors of the same type).")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(pattern_data)
    
    # Display top patterns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### Most Common Error Patterns")
        
        # Create bar chart
        fig = px.bar(
            df.head(10),  # Top 10 patterns
            y="True Intent",
            x="Count",
            color="Average Confidence",
            hover_data=["Predicted Intent"],
            labels={"Count": "Error Count", "True Intent": "True Intent", "Predicted Intent": "Predicted As"},
            color_continuous_scale="RdYlGn_r"  # Reverse scale (red = high confidence errors)
        )
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### High Confidence Errors")
        
        # Sort by confidence
        high_conf_df = df.sort_values("Average Confidence", ascending=False).head(10)
        
        fig = px.bar(
            high_conf_df,
            y="True Intent",
            x="Average Confidence",
            color="Count",
            hover_data=["Predicted Intent"],
            labels={"Average Confidence": "Avg. Confidence", "True Intent": "True Intent", 
                   "Predicted Intent": "Predicted As", "Count": "Error Count"},
        )
        
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Provide insights based on patterns
    st.subheader("Insights")
    
    high_conf_errors = [p for p in pattern_data if p["Average Confidence"] > 0.7]
    high_count_errors = [p for p in pattern_data if p["Count"] > max(2, len(errors) * 0.1)]
    
    insights = []
    
    if high_conf_errors:
        insights.append(f"⚠️ Found {len(high_conf_errors)} error patterns with high confidence (>0.7), which indicates the model is confidently wrong. "
                        f"Consider refining the training data for: " + 
                        ", ".join([f"`{p['True Intent']}`→`{p['Predicted Intent']}`" for p in high_conf_errors[:3]]))
    
    if high_count_errors:
        insights.append(f"🔍 Found {len(high_count_errors)} common error patterns. The most frequent ones are: " +
                        ", ".join([f"`{p['True Intent']}`→`{p['Predicted Intent']}` ({p['Count']} errors)" for p in high_count_errors[:3]]))
    
    if len(set([p["True Intent"] for p in pattern_data])) < len(pattern_data) * 0.7:
        confused_intents = [p["True Intent"] for p in sorted(pattern_data, key=lambda x: x["Count"], reverse=True)[:5]]
        insights.append(f"📊 Some intents are frequently misclassified as different classes: {', '.join(confused_intents)}. "
                        f"These may need clearer definitions or more training examples.")
    
    if insights:
        for i, insight in enumerate(insights):
            st.markdown(f"{insight}")
    else:
        st.markdown("No significant error patterns detected. The errors appear to be relatively random.") 