"""
Utility functions for interactive dashboard components.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional

def create_model_comparison_view(models: List[Dict], load_model_metrics_func):
    """
    Create an interactive model comparison view.
    
    Args:
        models: List of model metadata dictionaries
        load_model_metrics_func: Function to load model metrics
    """
    st.header("Model Comparison")

    if not models:
        st.warning("No models available for comparison.")
        return

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select Base Model")
        base_model = st.selectbox(
            "Base model",
            options=[m["id"] for m in models],
            index=0,
            key="base_model"
        )

    with col2:
        st.subheader("Select Comparison Model")
        comparison_model = st.selectbox(
            "Comparison model",
            options=[m["id"] for m in models],
            index=min(1, len(models)-1),
            key="comparison_model"
        )

    if base_model == comparison_model:
        st.warning("Please select different models to compare")
        return

    # Load metrics for both models
    base_metrics = load_model_metrics_func(base_model)
    comparison_metrics = load_model_metrics_func(comparison_model)

    if not base_metrics or not comparison_metrics:
        st.error("Could not load metrics for one or both models")
        return

    # Create comparison visualizations
    create_comparison_summary(base_metrics, comparison_metrics)
    create_detailed_comparison_tables(base_metrics, comparison_metrics)


def create_comparison_summary(base_metrics: Dict, comparison_metrics: Dict):
    """Create a summary of model comparison metrics."""
    st.subheader("Comparison Summary")
    
    # Extract key metrics
    metrics_to_compare = {
        "Intent Accuracy": (
            base_metrics.get("intent_metrics", {}).get("accuracy", 0),
            comparison_metrics.get("intent_metrics", {}).get("accuracy", 0)
        ),
        "Intent F1": (
            base_metrics.get("intent_metrics", {}).get("f1", 0),
            comparison_metrics.get("intent_metrics", {}).get("f1", 0)
        ),
        "Entity F1": (
            base_metrics.get("entity_metrics", {}).get("micro avg", {}).get("f1-score", 0),
            comparison_metrics.get("entity_metrics", {}).get("micro avg", {}).get("f1-score", 0)
        ),
    }
    
    # Create comparison cards
    cols = st.columns(len(metrics_to_compare))
    
    for i, (metric_name, (base_value, comparison_value)) in enumerate(metrics_to_compare.items()):
        with cols[i]:
            delta = comparison_value - base_value
            delta_percentage = (delta / base_value) * 100 if base_value > 0 else 0
            
            st.metric(
                label=metric_name,
                value=f"{comparison_value:.2f}",
                delta=f"{delta_percentage:+.1f}%",
                delta_color="normal"
            )


def create_detailed_comparison_tables(base_metrics: Dict, comparison_metrics: Dict):
    """Create detailed tables comparing performance across models."""
    st.subheader("Detailed Comparison")
    
    # Intent metrics comparison
    if "intent_metrics" in base_metrics and "intent_metrics" in comparison_metrics:
        st.markdown("#### Intent Performance")
        
        # Create per-intent comparison
        base_per_class = base_metrics["intent_metrics"].get("per_class_report", {})
        comparison_per_class = comparison_metrics["intent_metrics"].get("per_class_report", {})
        
        # Collect all intents from both models
        all_intents = sorted(list(set(list(base_per_class.keys()) + list(comparison_per_class.keys()))))
        
        # Create comparison data
        intent_comparison = []
        
        for intent in all_intents:
            base_f1 = base_per_class.get(intent, {}).get("f1-score", 0)
            comparison_f1 = comparison_per_class.get(intent, {}).get("f1-score", 0)
            change = comparison_f1 - base_f1
            
            intent_comparison.append({
                "Intent": intent,
                "Base F1": f"{base_f1:.3f}",
                "New F1": f"{comparison_f1:.3f}",
                "Change": f"{change:+.3f}",
                "Change %": f"{(change / base_f1 * 100) if base_f1 > 0 else 0:+.1f}%",
                "_change_value": change  # For sorting
            })
            
        # Convert to DataFrame for displaying
        df = pd.DataFrame(intent_comparison)
        
        # Show most improved intents
        st.markdown("##### Most Improved Intents")
        improved_df = df.sort_values("_change_value", ascending=False).head(5)
        st.dataframe(improved_df.drop("_change_value", axis=1))
        
        # Show most degraded intents
        st.markdown("##### Most Degraded Intents")
        degraded_df = df.sort_values("_change_value", ascending=True).head(5)
        st.dataframe(degraded_df.drop("_change_value", axis=1))
        
        # Option to show all intents
        with st.expander("Show All Intents Comparison"):
            st.dataframe(df.drop("_change_value", axis=1))


def create_error_explorer(metrics: Dict):
    """
    Create an interactive error explorer interface.
    
    Args:
        metrics: Dictionary containing detailed evaluation results
    """
    # Check if detailed results are available
    if "detailed_results" not in metrics or not metrics["detailed_results"]:
        st.warning("No detailed results available for error analysis.")
        return

    # Get errors from metrics
    errors = [r for r in metrics["detailed_results"] if not r.get("intent_correct", True)]
    
    if not errors:
        st.success("No intent classification errors found in this evaluation!")
        return

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

    # Display error statistics
    st.subheader("Error Statistics")
    total_examples = len(metrics.get("detailed_results", []))
    error_rate = (len(errors) / total_examples) * 100 if total_examples > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Examples", str(total_examples))
    with col2:
        st.metric("Error Count", str(len(errors)))
    with col3:
        st.metric("Error Rate", f"{error_rate:.1f}%")

    # Create filter sidebar
    st.sidebar.subheader("Error Filters")

    # Filter by error pattern
    pattern_options = ["All Errors"] + [f"{true} → {pred} ({len(errs)})" 
                                      for (true, pred), errs in sorted_patterns]
    selected_pattern = st.sidebar.selectbox(
        "Error Pattern",
        options=pattern_options
    )

    # Filter by confidence
    confidence_values = [e.get("confidence", 0) for e in errors]
    min_conf = min(confidence_values) if confidence_values else 0
    max_conf = max(confidence_values) if confidence_values else 1
    
    min_confidence, max_confidence = st.sidebar.slider(
        "Confidence Range",
        min_value=0.0,
        max_value=1.0,
        value=(min_conf, max_conf),
        step=0.05
    )

    # Apply filters
    filtered_errors = errors
    if selected_pattern != "All Errors":
        pattern = selected_pattern.split(" (")[0]
        true, pred = pattern.split(" → ")
        filtered_errors = error_patterns.get((true, pred), [])

    filtered_errors = [
        e for e in filtered_errors
        if min_confidence <= e.get("confidence", 0) <= max_confidence
    ]

    # Show filtered errors
    st.subheader("Error Examples")
    st.write(f"Showing {len(filtered_errors)} of {len(errors)} errors")

    # Enable pagination for large error sets
    errors_per_page = 10
    num_pages = (len(filtered_errors) + errors_per_page - 1) // errors_per_page
    
    if num_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=num_pages, value=1) - 1
        start_idx = page * errors_per_page
        end_idx = min(start_idx + errors_per_page, len(filtered_errors))
        displayed_errors = filtered_errors[start_idx:end_idx]
        st.write(f"Page {page+1} of {num_pages}")
    else:
        displayed_errors = filtered_errors

    for i, error in enumerate(displayed_errors):
        with st.expander(f"Error {i+1}: {error.get('text', 'No text available')}"):
            col1, col2 = st.columns(2)
            col1.markdown(f"**True Intent:** {error.get('true_intent', 'unknown')}")
            col2.markdown(f"**Predicted Intent:** {error.get('pred_intent', 'unknown')}")
            st.markdown(f"**Confidence:** {error.get('confidence', 0):.4f}")

            # Show entities if available
            if "true_entities" in error and error["true_entities"]:
                st.markdown("**True Entities:**")
                for entity in error["true_entities"]:
                    st.markdown(f"- {entity.get('entity', 'unknown')}: {entity.get('value', 'unknown')}")
                    
            if "pred_entities" in error and error["pred_entities"]:
                st.markdown("**Predicted Entities:**")
                for entity in error["pred_entities"]:
                    st.markdown(f"- {entity.get('entity', 'unknown')}: {entity.get('value', 'unknown')}")
                    
            # Show entity errors if available
            if "entity_errors" in error and error["entity_errors"]:
                st.markdown("**Entity Errors:**")
                for err in error["entity_errors"]:
                    err_type = err.get("type", "unknown")
                    if err_type == "false_positive":
                        st.markdown(f"- False Positive: {err.get('entity', 'unknown')} = {err.get('value', 'unknown')}")
                    elif err_type == "false_negative":
                        st.markdown(f"- False Negative: {err.get('entity', 'unknown')} = {err.get('value', 'unknown')}")
                    elif err_type == "value_error":
                        st.markdown(f"- Value Error: {err.get('entity', 'unknown')} = {err.get('pred', 'unknown')} (should be {err.get('true', 'unknown')})") 