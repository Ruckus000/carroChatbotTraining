"""
Guided analysis page for walking users through a structured NLU model analysis.
"""
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

from utils.data_processing import load_model_metrics
from utils.state_management import get_selected_model, initialize_session_state

def render_guided_analysis_page():
    """
    Render the guided analysis page with step-by-step workflow.
    """
    st.title("Guided Analysis")
    
    st.markdown("""
    This page provides a step-by-step guide to analyzing your NLU model's performance.
    Follow the steps below to understand your model's strengths and weaknesses.
    """)
    
    # Initialize workflow state if not present
    if "analysis_step" not in st.session_state:
        st.session_state.analysis_step = 1
    
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
    
    # Display step navigator
    create_step_navigator()
    
    # Render current step
    if st.session_state.analysis_step == 1:
        render_step_overview(model_id, metrics)
    elif st.session_state.analysis_step == 2:
        render_step_intent_analysis(metrics)
    elif st.session_state.analysis_step == 3:
        render_step_entity_analysis(metrics)
    elif st.session_state.analysis_step == 4:
        render_step_error_analysis(metrics)
    elif st.session_state.analysis_step == 5:
        render_step_recommendations(metrics)


def create_step_navigator():
    """Create a horizontal step navigator."""
    steps = [
        ("1. Overview", "Overview of key metrics"),
        ("2. Intent Analysis", "Analyze intent classification"),
        ("3. Entity Analysis", "Analyze entity recognition"),
        ("4. Error Analysis", "Investigate errors"),
        ("5. Recommendations", "Get improvement suggestions")
    ]
    
    # Create columns for each step
    cols = st.columns(len(steps))
    
    for i, (label, tooltip) in enumerate(steps, 1):
        with cols[i-1]:
            if i == st.session_state.analysis_step:
                st.button(
                    f"**{label}**", 
                    key=f"step_{i}", 
                    on_click=set_analysis_step, 
                    args=(i,),
                    help=tooltip,
                    use_container_width=True
                )
            else:
                st.button(
                    label, 
                    key=f"step_{i}", 
                    on_click=set_analysis_step, 
                    args=(i,),
                    help=tooltip,
                    use_container_width=True
                )
    
    st.markdown("---")


def set_analysis_step(step: int):
    """Set the current analysis step."""
    st.session_state.analysis_step = step


def render_step_overview(model_id: str, metrics: Dict):
    """Render the overview step."""
    st.header("1. Performance Overview")
    
    st.markdown(f"""
    Let's start by looking at the overall performance of model **{model_id}**.
    This gives us a high-level view of how well the model is performing.
    """)
    
    # Extract key metrics
    intent_accuracy = metrics.get("intent_metrics", {}).get("accuracy", 0)
    intent_f1 = metrics.get("intent_metrics", {}).get("f1", 0)
    intent_precision = metrics.get("intent_metrics", {}).get("precision", 0)
    intent_recall = metrics.get("intent_metrics", {}).get("recall", 0)
    
    entity_metrics = metrics.get("entity_metrics", {}).get("micro avg", {})
    entity_f1 = entity_metrics.get("f1-score", 0)
    entity_precision = entity_metrics.get("precision", 0)
    entity_recall = entity_metrics.get("recall", 0)
    
    # Create gauge charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Intent Classification")
        fig = create_gauge_chart(intent_accuracy, "Accuracy")
        st.plotly_chart(fig, use_container_width=True)
        
        subcol1, subcol2, subcol3 = st.columns(3)
        subcol1.metric("Precision", f"{intent_precision:.2f}")
        subcol2.metric("Recall", f"{intent_recall:.2f}")
        subcol3.metric("F1 Score", f"{intent_f1:.2f}")
    
    with col2:
        st.subheader("Entity Recognition")
        fig = create_gauge_chart(entity_f1, "F1 Score")
        st.plotly_chart(fig, use_container_width=True)
        
        subcol1, subcol2, subcol3 = st.columns(3)
        subcol1.metric("Precision", f"{entity_precision:.2f}")
        subcol2.metric("Recall", f"{entity_recall:.2f}")
        subcol3.metric("F1 Score", f"{entity_f1:.2f}")
    
    # Performance assessment
    st.subheader("Performance Assessment")
    
    # Intent performance assessment
    intent_rating = get_performance_rating(intent_f1)
    entity_rating = get_performance_rating(entity_f1)
    
    st.markdown(f"""
    **Intent Classification**: {intent_rating['emoji']} {intent_rating['label']}
    
    {intent_rating['description']}
    
    **Entity Recognition**: {entity_rating['emoji']} {entity_rating['label']}
    
    {entity_rating['description']}
    """)
    
    # Next step button
    st.button("Next: Intent Analysis →", on_click=set_analysis_step, args=(2,))


def render_step_intent_analysis(metrics: Dict):
    """Render the intent analysis step."""
    st.header("2. Intent Analysis")
    
    st.markdown("""
    Let's analyze how well the model performs for each intent.
    This helps identify which intents need improvement.
    """)
    
    intent_metrics = metrics.get("intent_metrics", {})
    if not intent_metrics or "per_class_report" not in intent_metrics:
        st.warning("Detailed intent metrics not available.")
        return
    
    per_class = intent_metrics["per_class_report"]
    
    # Create a sorted list of intents by F1 score
    intents = [(intent, data.get("f1-score", 0)) for intent, data in per_class.items()]
    intents.sort(key=lambda x: x[1])
    
    # Display best and worst performing intents
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Performing Intents")
        for intent, f1 in reversed(intents[-5:]):
            st.markdown(f"- **{intent}**: {f1:.3f}")
    
    with col2:
        st.subheader("Bottom Performing Intents")
        for intent, f1 in intents[:5]:
            st.markdown(f"- **{intent}**: {f1:.3f}")
    
    # Show intent distribution
    st.subheader("Intent Performance Distribution")
    
    # Count intents in different performance buckets
    excellent = sum(1 for _, f1 in intents if f1 >= 0.9)
    good = sum(1 for _, f1 in intents if 0.8 <= f1 < 0.9)
    fair = sum(1 for _, f1 in intents if 0.7 <= f1 < 0.8)
    poor = sum(1 for _, f1 in intents if f1 < 0.7)
    
    total = len(intents)
    
    # Create horizontal stacked bar
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=["Intents"],
        x=[excellent / total * 100],
        name="Excellent (≥0.9)",
        orientation="h",
        marker=dict(color="green"),
        text=[f"{excellent} intents"],
        textposition="inside"
    ))
    
    fig.add_trace(go.Bar(
        y=["Intents"],
        x=[good / total * 100],
        name="Good (0.8-0.9)",
        orientation="h",
        marker=dict(color="lightgreen"),
        text=[f"{good} intents"],
        textposition="inside"
    ))
    
    fig.add_trace(go.Bar(
        y=["Intents"],
        x=[fair / total * 100],
        name="Fair (0.7-0.8)",
        orientation="h",
        marker=dict(color="orange"),
        text=[f"{fair} intents"],
        textposition="inside"
    ))
    
    fig.add_trace(go.Bar(
        y=["Intents"],
        x=[poor / total * 100],
        name="Poor (<0.7)",
        orientation="h",
        marker=dict(color="red"),
        text=[f"{poor} intents"],
        textposition="inside"
    ))
    
    fig.update_layout(
        barmode="stack",
        xaxis=dict(
            title="Percentage of Intents",
            ticksuffix="%"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("← Previous: Overview", on_click=set_analysis_step, args=(1,))
    with col2:
        st.button("Next: Entity Analysis →", on_click=set_analysis_step, args=(3,))


def render_step_entity_analysis(metrics: Dict):
    """Render the entity analysis step."""
    st.header("3. Entity Analysis")
    
    st.markdown("""
    Now let's examine how well the model recognizes different entity types.
    Entity recognition is crucial for extracting structured information from user inputs.
    """)
    
    entity_metrics = metrics.get("entity_metrics", {})
    if not entity_metrics:
        st.warning("Entity metrics not available.")
        return
    
    # Filter out summary metrics
    per_entity_metrics = {
        entity: data 
        for entity, data in entity_metrics.items() 
        if entity not in ["micro avg", "macro avg", "weighted avg"]
    }
    
    if not per_entity_metrics:
        st.info("No per-entity metrics available.")
        return
    
    # Create entity performance data
    entity_performance = []
    
    for entity, data in per_entity_metrics.items():
        entity_performance.append({
            "Entity": entity,
            "F1 Score": data.get("f1-score", 0),
            "Precision": data.get("precision", 0),
            "Recall": data.get("recall", 0),
            "Support": data.get("support", 0)
        })
    
    # Sort by F1 score
    entity_performance.sort(key=lambda x: x["F1 Score"])
    
    # Display horizontal bar chart
    st.subheader("Entity Performance Overview")
    
    fig = go.Figure()
    
    entities = [e["Entity"] for e in entity_performance]
    f1_scores = [e["F1 Score"] for e in entity_performance]
    precision = [e["Precision"] for e in entity_performance]
    recall = [e["Recall"] for e in entity_performance]
    
    fig.add_trace(go.Bar(
        y=entities,
        x=f1_scores,
        name="F1 Score",
        orientation="h",
        marker=dict(color="blue"),
        text=[f"{f1:.3f}" for f1 in f1_scores],
        textposition="auto"
    ))
    
    fig.add_trace(go.Bar(
        y=entities,
        x=precision,
        name="Precision",
        orientation="h",
        marker=dict(color="green"),
        visible="legendonly"
    ))
    
    fig.add_trace(go.Bar(
        y=entities,
        x=recall,
        name="Recall",
        orientation="h",
        marker=dict(color="red"),
        visible="legendonly"
    ))
    
    fig.update_layout(
        xaxis=dict(
            title="Score",
            range=[0, 1]
        ),
        yaxis=dict(
            title="Entity Type",
            categoryorder="total ascending"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=max(300, len(entities) * 30)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Entity insights
    st.subheader("Entity Insights")
    
    # Identify performance categories
    low_f1_entities = [e for e in entity_performance if e["F1 Score"] < 0.7]
    precision_recall_gap = [
        e for e in entity_performance 
        if abs(e["Precision"] - e["Recall"]) > 0.2
    ]
    low_support_entities = [
        e for e in entity_performance 
        if e["Support"] < 10 and e["F1 Score"] < 0.85
    ]
    
    # Generate insights
    insights = []
    
    if low_f1_entities:
        insights.append(
            f"⚠️ **Low Performing Entities**: {', '.join([e['Entity'] for e in low_f1_entities])} "
            f"have F1 scores below 0.7 and need improvement."
        )
    
    if precision_recall_gap:
        for entity in precision_recall_gap:
            gap_type = "precision" if entity["Precision"] < entity["Recall"] else "recall"
            insights.append(
                f"🔍 **{entity['Entity']}** has a large gap between precision and recall. "
                f"Focus on improving {gap_type}."
            )
    
    if low_support_entities:
        insights.append(
            f"📊 **Low Support Entities**: {', '.join([e['Entity'] for e in low_support_entities])} "
            f"have few examples in the test set, which may affect the reliability of their metrics. "
            f"Consider adding more test examples."
        )
    
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.success("All entities are performing reasonably well!")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("← Previous: Intent Analysis", on_click=set_analysis_step, args=(2,))
    with col2:
        st.button("Next: Error Analysis →", on_click=set_analysis_step, args=(4,))


def render_step_error_analysis(metrics: Dict):
    """Render the error analysis step."""
    st.header("4. Error Analysis")
    
    st.markdown("""
    Let's examine specific errors made by the model to understand patterns
    and potential areas for improvement.
    """)
    
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
    
    # Display top error patterns
    st.subheader("Top Error Patterns")
    
    for i, ((true_intent, pred_intent), err_list) in enumerate(sorted_patterns[:5]):
        with st.expander(f"{true_intent} → {pred_intent} ({len(err_list)} errors)"):
            st.markdown(f"The model confused **{true_intent}** with **{pred_intent}** in {len(err_list)} cases.")
            
            # Display example errors
            st.markdown("##### Example Errors")
            for j, err in enumerate(err_list[:3]):
                st.markdown(f"""
                **Example {j+1}**: "{err.get('text', 'No text available')}"
                
                Confidence: {err.get('confidence', 0):.3f}
                """)
            
            # Provide potential reasons for confusion
            st.markdown("##### Potential Reasons")
            st.markdown("""
            - Similar phrasing or vocabulary between these intents
            - Overlapping concepts or functionality
            - Insufficient training examples to differentiate them
            - Potentially ambiguous utterances that could fit both intents
            """)
    
    # Error distribution visualization
    st.subheader("Error Confidence Distribution")
    
    # Create histogram of error confidences
    confidences = [e.get("confidence", 0) for e in errors]
    
    fig = go.Figure(data=[go.Histogram(
        x=confidences,
        nbinsx=10,
        marker_color="indianred"
    )])
    
    fig.update_layout(
        title="Distribution of Model Confidence in Errors",
        xaxis_title="Confidence Score",
        yaxis_title="Number of Errors",
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights based on confidence
    high_conf_errors = [e for e in errors if e.get("confidence", 0) > 0.8]
    low_conf_errors = [e for e in errors if e.get("confidence", 0) < 0.3]
    
    st.subheader("Confidence Insights")
    
    if high_conf_errors:
        st.markdown(f"""
        ⚠️ Found {len(high_conf_errors)} high-confidence errors (>0.8). 
        These are concerning because the model is very confident in its wrong predictions.
        
        This typically indicates systematic confusions in your training data or very similar intents.
        """)
    
    if low_conf_errors:
        st.markdown(f"""
        📊 Found {len(low_conf_errors)} low-confidence errors (<0.3).
        
        These may be less concerning as the model exhibits uncertainty, 
        and a confidence threshold could potentially filter them out.
        """)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("← Previous: Entity Analysis", on_click=set_analysis_step, args=(3,))
    with col2:
        st.button("Next: Recommendations →", on_click=set_analysis_step, args=(5,))


def render_step_recommendations(metrics: Dict):
    """Render the recommendations step."""
    st.header("5. Recommendations")
    
    st.markdown("""
    Based on the analysis, here are recommendations to improve your NLU model's performance.
    These suggestions are tailored to the specific patterns and issues identified.
    """)
    
    # Overall model metrics
    intent_f1 = metrics.get("intent_metrics", {}).get("f1", 0)
    entity_f1 = metrics.get("entity_metrics", {}).get("micro avg", {}).get("f1-score", 0)
    
    # Intent metrics
    per_class = metrics.get("intent_metrics", {}).get("per_class_report", {})
    if not per_class:
        per_class = {}
    
    # Sort intents by F1 score
    intents = [(intent, data.get("f1-score", 0)) for intent, data in per_class.items()]
    intents.sort(key=lambda x: x[1])
    
    # Entity metrics
    entity_metrics = metrics.get("entity_metrics", {})
    per_entity_metrics = {
        entity: data 
        for entity, data in entity_metrics.items() 
        if entity not in ["micro avg", "macro avg", "weighted avg"]
    }
    
    # Get errors
    errors = []
    if "detailed_results" in metrics and metrics["detailed_results"]:
        errors = [r for r in metrics["detailed_results"] if not r.get("intent_correct", True)]
    
    # Create recommendations
    recommendations = []
    
    # Intent recommendations
    low_intents = [intent for intent, f1 in intents if f1 < 0.7]
    if low_intents:
        recommendations.append({
            "category": "Intent Classification",
            "title": "Improve Low-Performing Intents",
            "description": f"Add more training examples for: {', '.join(low_intents[:5])}",
            "priority": "High" if len(low_intents) > 3 else "Medium"
        })
    
    # Error pattern recommendations
    if errors:
        error_patterns = {}
        for e in errors:
            pair = (e.get("true_intent", "unknown"), e.get("pred_intent", "unknown"))
            if pair not in error_patterns:
                error_patterns[pair] = []
            error_patterns[pair].append(e)
        
        sorted_patterns = sorted(
            error_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        if sorted_patterns:
            top_confusion = sorted_patterns[0]
            true_intent, pred_intent = top_confusion[0]
            recommendations.append({
                "category": "Error Patterns",
                "title": f"Resolve {true_intent}→{pred_intent} Confusion",
                "description": f"These intents are frequently confused. Consider refining their definitions or adding examples that clearly differentiate them.",
                "priority": "High"
            })
    
    # Entity recommendations
    if per_entity_metrics:
        low_entities = [
            entity for entity, data in per_entity_metrics.items()
            if data.get("f1-score", 0) < 0.7
        ]
        
        if low_entities:
            recommendations.append({
                "category": "Entity Recognition",
                "title": "Improve Entity Recognition",
                "description": f"Add more diverse entity examples for: {', '.join(low_entities)}",
                "priority": "High" if len(low_entities) > 2 else "Medium"
            })
    
    # General recommendations
    if intent_f1 < 0.8 and entity_f1 < 0.8:
        recommendations.append({
            "category": "Model Training",
            "title": "Review Overall Model Architecture",
            "description": "Both intent and entity performance are below ideal levels. Consider reviewing the model architecture or training approach.",
            "priority": "Medium"
        })
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations):
            with st.expander(f"{rec['priority']} Priority: {rec['title']}", expanded=True):
                st.markdown(f"""
                **Category:** {rec['category']}
                
                **Description:** {rec['description']}
                
                **Priority:** {rec['priority']}
                """)
    else:
        st.success("No specific recommendations generated. The model is performing well!")
    
    # General best practices
    st.subheader("General Best Practices")
    
    st.markdown("""
    📌 **Regular Benchmarking**: Continue running benchmarks regularly, especially after model updates
    
    📌 **Data Quality**: Review and clean training data, ensuring examples are correctly labeled
    
    📌 **Test Coverage**: Ensure test data covers all intents and entities with sufficient examples
    
    📌 **Balanced Training**: Maintain a balanced distribution of examples across intents
    
    📌 **User Feedback**: Incorporate real user interactions to improve the model
    """)
    
    # Final button to return to overview
    st.button("← Back to Overview", on_click=set_analysis_step, args=(1,))


def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Create a gauge chart for displaying a metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": get_color_for_value(value)},
            "steps": [
                {"range": [0, 0.6], "color": "lightgray"},
                {"range": [0.6, 0.75], "color": "gray"},
                {"range": [0.75, 0.9], "color": "lightgreen"},
                {"range": [0.9, 1], "color": "green"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 0.8
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig


def get_color_for_value(value: float) -> str:
    """Return a color based on the value."""
    if value >= 0.9:
        return "green"
    elif value >= 0.75:
        return "lightgreen"
    elif value >= 0.6:
        return "orange"
    else:
        return "red"


def get_performance_rating(value: float) -> Dict:
    """Get a performance rating based on the value."""
    if value >= 0.9:
        return {
            "label": "Excellent",
            "emoji": "🟢",
            "description": "The model is performing exceptionally well in this area, with very high accuracy. Minimal improvements needed."
        }
    elif value >= 0.8:
        return {
            "label": "Good",
            "emoji": "🟡",
            "description": "The model is performing well, with good accuracy. Some minor improvements could still be beneficial."
        }
    elif value >= 0.7:
        return {
            "label": "Fair",
            "emoji": "🟠",
            "description": "The model's performance is acceptable but has room for improvement. Consider adding more training examples."
        }
    else:
        return {
            "label": "Needs Improvement",
            "emoji": "🔴",
            "description": "The model is underperforming in this area. Significant improvements are recommended."
        } 